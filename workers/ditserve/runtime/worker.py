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


BACKENDS = {
    "mock": ("mock", "MockBackend"),
    "wan": ("wan.backend", "WanBackend"),
    "wan_pipeline": ("wan.pipeline", "WanPipelineBackend"),
    "wan_profile": ("wan.profile", "WanProfileBackend"),
    "sana": ("session.backend", "SessionBackend"),
    "mistral": ("session.backend", "SessionBackend"),
    "flux": ("flux", "FluxBackend"),
    "longlive": ("longlive", "LongLiveBackend"),
    "results": None,
    "evaluation": None,
    "environment": None,
}
SERVICE_OPERATIONS = {"results": ("finalize", "verify", "inspect", "compact"),
                      "evaluation": ("evaluate",), "environment": ("inspect",)}


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


class Worker:
    def __init__(self, backend, config):
        if backend not in BACKENDS:
            raise ValueError("unknown backend: " + backend)
        self.name = backend
        self.config = dict(config, backend=backend)
        self.backend = None
        self.closed = False

    def dispatch(self, request):
        op = _operation(request)
        if self.closed:
            raise ValueError("worker is shut down")
        if op == "occupancy":
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
            return {"backend": self.name, "operations": list(dict.fromkeys(["hello", "occupancy", "memory", *operations, "evaluate", "shutdown"])),
                    "loaded": self.backend is not None, "measured": self.name not in ("mock", "results", "environment"),
                    "capabilities": {"operations": list(operations), "persistent_states": cls is not None},
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
            self.closed = True


def process_memory():
    """This process's CUDA allocator and the device's driver-free memory; never initializes CUDA."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return dict(cuda_initialized=False)
    free, total = torch.cuda.mem_get_info()
    return dict(cuda_initialized=True, allocated_bytes=int(torch.cuda.memory_allocated()),
                reserved_bytes=int(torch.cuda.memory_reserved()), peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                driver_free_bytes=int(free), device_total_bytes=int(total))


def occupancy(request, *, mock=False):
    requested = request.get("gpu_uuids", [])
    empty = dict(status="unknown", owned=[], foreign=[], devices=[], cuda_initialized=False)
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
                    devices=[dict(uuid=value, name=devices[value]) for value in requested], cuda_initialized=False)
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
