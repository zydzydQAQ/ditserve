"""One resident, lazy model worker speaking versioned JSON lines on stdout."""

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

from . import strict_json
from .green_context import GreenPool


BACKENDS = {
    "mock": ("mock", "MockBackend"),
    "wan": ("wan.backend", "WanBackend"),
    "wan_pipeline": ("wan.pipeline", "WanPipelineBackend"),
    "wan_profile": ("wan.profile", "WanProfileBackend"),
    "sana": ("session.backend", "SessionBackend"),
    "flux": ("flux", "FluxBackend"),
    "sd35": ("sd35", "SD35Backend"),
    "results": None,
    "evaluation": None,
    "environment": None,
}
SERVICE_OPERATIONS = {"results": ("finalize", "verify", "inspect", "compact"),
                      "evaluation": ("evaluate",), "environment": ("inspect",)}


def _model_capabilities(name, config, backend):
    if name not in ("wan", "wan_pipeline", "sana", "flux", "sd35"):
        return {}
    from ..models.xdit import engine_info

    options = config.get("options", {})
    atomic = name in ("wan_pipeline", "flux", "sd35") or (name == "sana" and options.get("execution") == "pipeline")
    execution = ("pipeline" if atomic else options.get("execution", "session" if name == "sana" else "step"))
    quanta = ["atomic"] if atomic else ["step"] + ([] if execution == "step" else ["block"])
    capabilities = dict(
        engine=engine_info(), engine_status="configured", execution=execution,
        model_wrapper={"wan": "xFuserWanTransformer3DWrapper", "wan_pipeline": "xFuserWanTransformer3DWrapper",
                       "sana": "xFuserSanaTransformer2DWrapper", "flux": "xFuserFluxTransformer2DWrapper",
                       "sd35": "xFuserSD3Transformer2DWrapper"}[name],
        quanta=quanta, atomic_pipeline=atomic,
        continuous_batch=name == "sana" and not atomic,
        batch="fixed" if name != "sana" or atomic else "ragged_continuous",
        migration=[] if name != "wan" else ["state"] + ([] if execution == "step" else ["block"]),
        parallel=dict(sp_degrees=[1, 2, 4, 8] if name == "wan" else [1], cfg=name == "wan"),
    )
    runtime = getattr(backend, "runtime", None)
    engine = getattr(backend, "xdit", None) or getattr(runtime, "xdit", None)
    if engine is not None:
        capabilities.update(engine=engine.info(), engine_status="loaded")
    return capabilities


def _operation(request):
    """Validate the wire envelope before looking up any backend or state."""
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    if type(request.get("version")) is not int or request["version"] != 1:
        raise ValueError("unsupported protocol version")
    if type(request.get("id")) is not int or request["id"] < 0:
        raise ValueError("id must be a nonnegative integer")
    if not isinstance(request.get("op"), str):
        raise ValueError("op must be a string")
    return request["op"]


def mps_control(request):
    if request.get("action") != "quit":
        raise ValueError("mps_control supports action=quit only")
    server_pid = request.get("server_pid")
    if type(server_pid) is not int or server_pid <= 0:
        raise ValueError("server_pid must be a positive integer")
    pipe = Path(request.get("pipe_directory", ""))
    if not pipe.is_absolute() or not pipe.is_dir():
        raise ValueError("pipe_directory must be an existing absolute directory")
    environment = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(pipe))
    if request.get("log_directory"):
        log = Path(request["log_directory"])
        if not log.is_absolute() or not log.is_dir():
            raise ValueError("log_directory must be an existing absolute directory")
        environment["CUDA_MPS_LOG_DIRECTORY"] = str(log)

    def command(value):
        return subprocess.run(["nvidia-cuda-mps-control"], input=value + "\n", capture_output=True,
                              text=True, timeout=10, check=True, env=environment)

    listing = command("get_server_list")
    servers = {int(line.strip()) for line in listing.stdout.splitlines() if line.strip().isdigit()}
    if server_pid not in servers:
        raise RuntimeError("declared MPS server is not owned by this control pipe")
    command("quit")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(server_pid, 0)
            stat = Path(f"/proc/{server_pid}/stat").read_text(encoding="utf-8")
            if stat[stat.rfind(")") + 2:].split()[0] == "Z":
                break
        except (ProcessLookupError, FileNotFoundError):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("MPS server did not exit after quit")
    return dict(action="quit", server_pid=server_pid, shutdown=True, verified=True)


class Worker:
    def __init__(self, backend, config):
        if backend not in BACKENDS:
            raise ValueError("unknown backend: " + backend)
        self.name = backend
        self.config = dict(config, backend=backend)
        self.backend = None
        self.closed = False
        self._occupancy_probe = None   # resident NVML owner, opened on the first nvml observation
        self._green = None

    def green_status(self):
        config = self.config.get("options", {}).get("green")
        if config is None:
            return dict(enabled=False, initialized=False)
        if self._green is not None:
            return self._green.report()
        return dict(enabled=True, initialized=False, role=config.get("role"),
                    default=config.get("default", "exclusive"))

    def _green_pool(self):
        config = self.config.get("options", {}).get("green")
        if config is None:
            raise ValueError("green_partition requires options.green in the worker config")
        if os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
            raise ValueError("MPS active-thread partitioning and Green Contexts are mutually exclusive")
        if self._green is None:
            self._green = GreenPool(config.get("role"), config.get("partitions"),
                                    config.get("default", "exclusive"))
        self._green.initialize()
        return self._green

    def dispatch(self, request):
        op = _operation(request)
        if self.closed:
            raise ValueError("worker is shut down")
        if op == "mps_control":
            return mps_control(request)
        if op == "occupancy":
            if request.get("occupancy_backend", "nvidia_smi") == "nvml":
                if self._occupancy_probe is None:
                    from .occupancy_nvml import NvmlOccupancy
                    self._occupancy_probe = NvmlOccupancy()
                return occupancy(request, mock=self.name == "mock", nvml_probe=self._occupancy_probe)
            return occupancy(request, mock=self.name == "mock")
        if op == "memory":
            return process_memory()
        if op == "shutdown":
            self.close()
            return {"shutdown": True}
        if op == "evaluate":
            from ..evaluation.reference import evaluate
            self._evaluation_device(request)
            return evaluate(request)
        definition = BACKENDS[self.name]
        cls = getattr(importlib.import_module("ditserve.models." + definition[0]), definition[1]) if definition else None
        operations = cls.operations if cls else SERVICE_OPERATIONS[self.name]
        if op == "hello":
            capabilities = {"operations": list(operations), "persistent_states": cls is not None}
            if self.name == "sana":
                options = self.config.get("options", {})
                capabilities.update(execution=options.get("execution", "session"),
                                    continuous_batch=options.get("execution") != "pipeline",
                                    hardware_profile=options.get("hardware_profile", "v100_fp32"))
            capabilities.update(_model_capabilities(self.name, self.config, self.backend))
            return {"backend": self.name, "operations": list(dict.fromkeys(["hello", "occupancy", "memory", "mps_control", *operations, "evaluate", "shutdown"])),
                    "loaded": self.backend is not None, "measured": self.name not in ("mock", "results", "environment"),
                    "capabilities": capabilities,
                    "qualification": "fixture_only" if self.name == "mock" else "unverified",
                    "pid": os.getpid()}
        if op not in operations:
            raise ValueError("unsupported operation for %s: %s" % (self.name, op))
        if cls is None:
            started = time.perf_counter()
            if self.name == "environment":
                from .environment import inspect_environment
                result = inspect_environment(request)
            else:
                from ..results import dispatch
                result = dispatch(request, self.config)
            return dict(result, load_ms=0., worker_operation_ms=(time.perf_counter() - started) * 1000)
        return self._model_call(cls, op, request)

    def _model_call(self, cls, op, request):
        """Own lazy model loading, idempotent release and operation timing."""
        if op == "release":
            if not isinstance(request.get("state_id"), str):
                raise ValueError("release requires state_id")
            if self.backend is None or request["state_id"] not in self.backend.states:
                return dict(released=request["state_id"], already_released=True)
        load_ms = 0.
        if self.backend is None:
            started = time.perf_counter()
            self.backend = cls(self.config)
            load_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        result = getattr(self.backend, op)(request)
        elapsed = (time.perf_counter() - started) * 1000
        if not isinstance(result, dict):
            raise TypeError("backend operation must return an object")
        return dict(result, load_ms=load_ms, worker_operation_ms=elapsed)

    def _evaluation_device(self, request):
        """A CPU fixture must never start an unaccounted GPU evaluator."""
        requested = request.get("device", "cpu")
        if not isinstance(requested, str) or not (requested == "cpu" or requested in ("cuda", "cuda:0")):
            raise ValueError("evaluation device must be cpu or the bound cuda:0")
        needs_gpu = requested != "cpu" or request.get("evaluator") == "vbench"
        if not needs_gpu:
            return
        if self.name in ("mock", "results", "environment"):
            raise ValueError(self.name + " permits CPU evaluation only")
        device, gpu = self.config.get("device", "cpu"), self.config.get("gpu_uuid")
        if (device not in ("cuda", "cuda:0") or not isinstance(gpu, str) or not gpu.startswith("GPU-")
                or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu):
            raise ValueError("GPU evaluation requires cuda:0 config and its GPU UUID bound in CUDA_VISIBLE_DEVICES")

    def close(self):
        try:
            if self.backend is not None:
                self.backend.close()
        finally:
            try:
                if self._occupancy_probe is not None:
                    self._occupancy_probe.close()
            finally:
                self.closed = True


def process_memory():
    """This process's CUDA allocator and the device's driver-free memory; never initializes CUDA."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return dict(cuda_initialized=False)
    free, total = torch.cuda.mem_get_info()
    device = torch.cuda.current_device()
    return dict(cuda_initialized=True, allocated_bytes=int(torch.cuda.memory_allocated()),
                reserved_bytes=int(torch.cuda.memory_reserved()), peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                driver_free_bytes=int(free), device_total_bytes=int(total),
                sm_count=int(torch.cuda.get_device_properties(device).multi_processor_count),
                mps_active_thread_percent=os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))


def occupancy(request, *, mock=False, nvml_probe=None):
    """Foreign compute processes on the requested GPUs: nvidia-smi (default) or direct NVML.

    Both backends fail closed to status "unknown" and name themselves in the result.
    """
    backend = request.get("occupancy_backend", "nvidia_smi")
    if backend == "nvml":
        if nvml_probe is not None:
            return nvml_probe.observe(request)
        from .occupancy_nvml import NvmlOccupancy
        probe = NvmlOccupancy()
        try:
            return probe.observe(request)
        finally:
            probe.close()
    if backend != "nvidia_smi":
        raise ValueError("occupancy_backend must be nvidia_smi or nvml")
    requested = request.get("gpu_uuids", [])
    empty = dict(status="unknown", owned=[], foreign=[], devices=[], cuda_initialized=False,
                 occupancy_backend="nvidia_smi")
    if not requested:
        return dict(empty, status="exclusive_observed" if mock else "unknown")
    try:
        if (not isinstance(requested, list) or any(not isinstance(value, str) or not value.startswith("GPU-")
                                                 for value in requested) or len(set(requested)) != len(requested)):
            raise ValueError("explicit unique GPU UUIDs are required")
        available = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10, check=True)
        devices = {row.split(",", 1)[0].strip(): row.split(",", 1)[1].strip()
                   for row in available.stdout.splitlines() if row.strip()}
        if any(value not in devices for value in requested):
            raise ValueError("requested full GPU UUID is not present")
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        allowed = set(request.get("allowed_pids", []))
        groups = set(request.get("allowed_pgids", []))
        owned, foreign = [], []
        for line in result.stdout.splitlines():
            pid, gpu, memory = (item.strip() for item in line.split(","))
            if gpu not in requested:
                continue
            pid = int(pid)
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                # NVML may retain an exiting owner or expose a PID outside
                # this namespace. Its GPU allocation is not proof of idle.
                pgid = None
            row = dict(pid=pid, pgid=pgid, gpu_uuid=gpu, memory_mib=memory)
            (owned if pid in allowed or pgid in groups else foreign).append(row)
        return dict(status="busy" if foreign else "exclusive_observed", owned=owned, foreign=foreign,
                    devices=[dict(uuid=value, name=devices[value]) for value in requested], cuda_initialized=False,
                    occupancy_backend="nvidia_smi")
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
        return dict(empty, reason=str(error))


def serve(worker, incoming, outgoing):
    """Keep malformed requests local; every complete input line gets one reply."""
    try:
        for line in incoming:
            request = None
            try:
                request = strict_json(line)
                with contextlib.redirect_stdout(sys.stderr):
                    result = worker.dispatch(request)
                response = {"version": 1, "id": request["id"], "ok": True, "result": result}
                # Validate before writing, so a bad backend cannot break framing.
                encoded = json.dumps(response, allow_nan=False, ensure_ascii=False)
            except Exception as error:
                identity = request.get("id") if isinstance(request, dict) else None
                response = {"version": 1, "id": identity if type(identity) is int else None,
                            "ok": False, "error": {"type": type(error).__name__, "message": str(error)}}
                encoded = json.dumps(response, allow_nan=False, ensure_ascii=False)
            outgoing.write(encoded + "\n")
            outgoing.flush()
            if worker.closed:
                break
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            if not worker.closed:
                worker.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    config = strict_json(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        parser.error("worker config must be an object")
    # Native libraries may write directly to fd 1, bypassing redirect_stdout.
    # Keep a private protocol descriptor and route all ordinary stdout to stderr.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1) as protocol:
        sys.stdout.flush()
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        serve(Worker(args.backend, config), sys.stdin, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
