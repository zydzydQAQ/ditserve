"""Separately authorized online-only cache factorial; archived run01 is read-only."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace

from ..research.common import atomic_json, digest, file_digest, read_json
from ..research.isolation import gpu_occupancy
from ..directions.runner import stop_owned_group
from .budget import BudgetGuard, BudgetStop
from .protocol import ROOT, CONFIG, OUTPUT as SOURCE_ROOT, LEDGER as OLD_LEDGER, PARENT, load_config, freeze

OUTPUT = ROOT / "results/optimization-cache-online/run01"
LEDGER = ROOT / "results/optimization-cache-online/gpu-budget.json"
PHASE = "cache_k4"
DEFAULTS = dict(sana_backend="eager", image_batch_size=1, wan_backend="eager",
                vae_backend="eager", encode_device="cpu", encode_on_scheduler=False)


def estimate(config):
    from .online import load_frozen_traces
    traces = load_frozen_traces(config["old_online"])
    windows = sum(max(r["arrival_ms"] for r in t["requests"]) / 1000. for t in traces) * 8
    service_windows = sum(max(r["arrival_ms"] + (r["service_ms"] if r["kind"] == "video" else 1000.)
        for r in t["requests"]) / 1000. for t in traces)
    online = service_windows * 8 * 1.08 + 60.
    return dict(traces=16, requests=384, arms=["off_step", "off_k4", "paired_step", "paired_k4"],
        minimum_arrival_window_seconds=windows, online_admission_seconds=online,
        warmup_allowance_seconds=600., setup_allowance_seconds=120., cleanup_allowance_seconds=60.,
        required_before_loading_seconds=online + 780.,
        trace_bindings=[{k: t[k] for k in ("trace_sha256", "seed", "load", "pattern")} for t in traces],
        scope="Runtime estimate only; a hard GPU-hour deadline applies only to a finite authorization")


def authorization(path):
    path = Path(path).resolve(strict=True)
    value = read_json(path)
    seconds = value.get("authorized_gpu_seconds")
    unlimited = value.get("budget_unlimited", False)
    valid_budget = ((unlimited is True and "authorized_gpu_seconds" in value and seconds is None)
        or (unlimited is False and type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0))
    if (value.get("approved_by_user") is not True or not valid_budget
            or not isinstance(value.get("user_instruction"), str) or not value["user_instruction"].strip()
            or value.get("scope") != "cache_online_factorial_only"):
        raise ValueError("explicit user-authorized finite or unlimited supplement budget is required")
    return value, dict(path=str(path), sha256=file_digest(path))


def inspect_study(config, source_root=SOURCE_ROOT, supervisor=None, *, observe=True):
    from .cache_replay import inspect_cache_replay
    from .worker import selected
    source_root = Path(source_root).resolve()
    supervisor = Path(supervisor or source_root / PHASE / "attempt00/supervisor.json").resolve()
    inherited = selected(source_root)
    expected = {k: DEFAULTS[k] for k in ("sana_backend", "image_batch_size", "wan_backend")}
    if inherited != expected:
        raise ValueError("archived selected baseline changed; fixed eager/serial/CPU factorial requires review")
    binding = inspect_cache_replay(supervisor, config, source_root)
    result = dict(status="prepared", gpu_work_started=False, replay=binding, estimate=estimate(config),
        defaults=DEFAULTS, inherited_config_sha256=digest(config), source_root=str(source_root),
        old_ledgers=[dict(path=str(p.resolve()), sha256=file_digest(p)) for p in (OLD_LEDGER, PARENT)],
        budget_scope="Independent explicit authorization; no historical balance transfer or historical writes")
    if observe:
        from .runner import inspect as inspect_gpu
        result["preflight"] = inspect_gpu(config)
        result["preflight"].pop("additional_budget_gpu_hours", None)
        result["status"] = "ready_for_authorization" if result["preflight"]["status"] == "ready" else "waiting_resource"
    return result


def _new_paths(output, ledger, source_root):
    output, ledger, source_root = map(lambda p: Path(p).resolve(), (output, ledger, source_root))
    old = (SOURCE_ROOT.resolve(), OLD_LEDGER.resolve().parent, PARENT.resolve().parent, source_root)
    if any(output == p or p in output.parents or ledger == p or p in ledger.parents for p in old):
        raise ValueError("online supplement must be outside every historical run/ledger tree")
    if ledger.parent not in output.parents or ledger.name != "gpu-budget.json":
        raise ValueError("supplement ledger must be named gpu-budget.json above the new run")
    return output, ledger, source_root


def execute(config, output, ledger, source_root, supervisor, authorization_path, *, wait_idle=0., retry_from=None):
    from .supplement_budget import SupplementAllocation
    output, ledger, source_root = _new_paths(output, ledger, source_root)
    auth, auth_binding = authorization(authorization_path)
    prepared = inspect_study(config, source_root, supervisor, observe=False)
    required = prepared["estimate"]["required_before_loading_seconds"]
    if auth["authorized_gpu_seconds"] is not None and auth["authorized_gpu_seconds"] < required:
        raise BudgetStop("authorized supplement cannot admit the complete matrix, setup, warmup and cleanup")
    root = output / PHASE
    terminal = root / "terminal.json"
    if terminal.exists() and retry_from is None:
        return read_json(terminal)
    retry = None
    if retry_from is not None:
        retry_path = Path(retry_from).resolve(strict=True)
        previous = read_json(retry_path)
        if (root not in retry_path.parents or previous.get("status") not in ("failed", "incomplete")
                or previous.get("config_sha256") != digest(config) or not previous.get("charge")):
            raise ValueError("retry must identify this supplement's settled same-configuration failure")
        retry = dict(path=str(retry_path), sha256=file_digest(retry_path))
    root.mkdir(parents=True, exist_ok=True)
    study = dict(schema_version="ditserve.optimizations.cache_online_study.v1", inherited_config_sha256=digest(config),
        source_root=str(source_root), authorization=auth_binding, authorized_gpu_seconds=auth["authorized_gpu_seconds"],
        budget_unlimited=auth.get("budget_unlimited", False),
        estimate=prepared["estimate"], defaults=DEFAULTS, replay=prepared["replay"], old_ledgers=prepared["old_ledgers"],
        scope="Online-only continuation; zero new local qualification samples; fresh charged process warmup")
    study_path = output / "study.json"
    if study_path.exists() and read_json(study_path) != study:
        raise ValueError("frozen supplement study differs")
    atomic_json(study_path, study)
    deadline = time.monotonic() + max(0., wait_idle)
    from .runner import inspect as inspect_gpu
    while True:
        observation = inspect_gpu(config)
        observation.pop("additional_budget_gpu_hours", None)
        atomic_json(root / "preflight.json", observation)
        if observation["status"] == "ready":
            break
        if time.monotonic() >= deadline:
            return dict(status="blocked", gpu_work_started=False, reason="A100/model admission not ready",
                        preflight=str(root / "preflight.json"))
        time.sleep(min(5., max(0., deadline - time.monotonic())))
    attempt = "attempt%02d" % len(list(root.glob("attempt[0-9][0-9]")))
    destination = root / attempt
    process = None
    failure = None
    returncode = None
    devices = [config["gpu_uuid"]]
    with SupplementAllocation(ledger, OLD_LEDGER, PARENT,
            authorized_gpu_seconds=auth["authorized_gpu_seconds"], authorization=auth["user_instruction"],
            gpu_uuid=config["gpu_uuid"], budget_unlimited=auth.get("budget_unlimited", False)) as budget:
        if budget.remaining(PHASE) < required:
            raise BudgetStop("remaining supplement cannot admit a complete retry matrix")
        destination.mkdir()
        source = freeze(config, output, PHASE, attempt)
        atomic_json(destination / "config.json", config)
        atomic_json(destination / "study.json", study)
        identity = dict(phase=PHASE, output=str(destination), run_root=str(output), config_sha256=digest(config),
            source_sha256=source["source_sha256"], gpu_uuids=devices, retry_from=retry,
            budget_unlimited=auth.get("budget_unlimited", False),
            cache_replay_from=prepared["replay"], authorization_source=auth_binding)
        budget.begin(PHASE, gpu_count=1, identity=identity, minimum_seconds=required)
        hard_deadline = (None if auth.get("budget_unlimited", False)
                         else time.monotonic() + budget.remaining(PHASE) - 60.)
        command = [config["python"], "-u", "-m", "ditserve.optimizations.cache_followup", "worker",
            "--config", str(destination / "config.json"), "--output", str(destination), "--run-root", str(output),
            "--source-root", str(source_root), "--source-supervisor", prepared["replay"]["path"],
            "--ledger", str(ledger), "--session-id", budget.data["active"]["session_id"]]
        if hard_deadline is not None:
            command.extend(["--deadline", str(hard_deadline)])
        env = dict(os.environ, PYTHONPATH=str(Path(source["execution_root"]) / "python"),
            CUDA_VISIBLE_DEVICES=",".join(devices), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", TORCH_HOME=config["torch_home"],
            TOKENIZERS_PARALLELISM="false")
        try:
            with (destination / "worker.log").open("x") as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                budget.child_started(process.pid)
                with (destination / "isolation.jsonl").open("x") as events:
                    while True:
                        observation = gpu_occupancy(devices, allowed_pgid=process.pid)
                        events.write(json.dumps(observation) + "\n")
                        events.flush()
                        if observation["status"] != "exclusive_observed":
                            raise RuntimeError("A100 isolation lost: " + observation["status"])
                        if hard_deadline is not None and time.monotonic() > hard_deadline + 5.:
                            raise BudgetStop("worker exceeded cooperative supplement deadline")
                        try:
                            returncode = process.wait(timeout=2.)
                            break
                        except subprocess.TimeoutExpired:
                            pass
        except BaseException as error:
            failure = dict(error_type=type(error).__name__, reason=str(error))
        finally:
            gone = stop_owned_group(process)
            final = gpu_occupancy(devices, allowed_pgid=process.pid if process else None)
            if not gone or final.get("owned") or final["status"] == "unknown":
                raise RuntimeError("cleanup unconfirmed; charged supplement intentionally left active")
            if final["status"] != "exclusive_observed":
                failure = dict(error_type="GPUIsolationError", reason="foreign process at cleanup")
            changed = [b["path"] for b in prepared["old_ledgers"] if file_digest(b["path"]) != b["sha256"]]
            changed += [name for name, sha in source["source_files"].items()
                        if file_digest(Path(source["execution_root"]) / name) != sha]
            if changed:
                failure = dict(error_type="SourceIdentityError", reason="bound source or historical ledger changed", files=changed)
            charge = budget.finish("failed" if failure or returncode else "completed")
            if charge.get("status") != "completed" and failure is None:
                failure = dict(error_type="BudgetSettlementError",
                    reason=charge.get("parent_validation_error", "charged session did not settle successfully"))
    path = destination / "worker.json"
    result = read_json(path) if path.exists() else dict(status="incomplete", reason="worker result missing")
    if failure or returncode != 0:
        result.update(status="incomplete", decision="未验证", online_qualified=False,
                      supervisor_failure=failure, reason=(failure or {}).get("reason", result.get("reason", "worker failed")))
    result.update(phase=PHASE, charge=charge, returncode=returncode, output=str(destination),
        final_occupancy=final, config_sha256=digest(config), source_sha256=source["source_sha256"],
        cache_replay_from=prepared["replay"], authorization_source=auth_binding)
    atomic_json(destination / "supervisor.json", result)
    atomic_json(terminal, result)
    return result


def run_worker(args, config, active):
    from .cache_replay import load_cache_evidence
    from .worker import capacity
    from .cache_online import run_cache_online, warm_cache_factorial
    from ..gpu.worker import WanStepWorker
    from ..directions.block_executor import WanBlockExecutor
    from .sana import OptimizedSanaAdapter
    guard = BudgetGuard(float("inf") if args.deadline is None else args.deadline, active["supervisor_pid"])
    # All source validation and local qualification reuse precede model loading.
    context = SimpleNamespace(config=config, output=args.output, run_root=args.run_root,
        cache_replay_binding=active["identity"]["cache_replay_from"],
        metadata=dict(phase=PHASE, encoder_device="cpu", cache_local_evidence_reused=True))
    result = load_cache_evidence(context, args.source_supervisor)
    context.check_budget = guard
    context.opts = config["direction1"]
    context.profile = dict(selected_shape=dict(height=480, width=832, frames=81))
    context.online_defaults = dict(DEFAULTS)
    worker = sana = None
    started = time.perf_counter()
    try:
        guard.ensure_unit(estimate(config)["online_admission_seconds"] + 600.)
        worker = context.worker = WanStepWorker(device=config["device"], model_path=config["models"]["wan"]["path"], attention="flash")
        sana = context.sana = OptimizedSanaAdapter(config["models"]["sana"]["path"], device=config["device"])
        context.executor = WanBlockExecutor(worker)
        context.metadata["model_load_ms"] = (time.perf_counter() - started) * 1000.
        atomic_json(args.output / "loading.json", dict(status="completed", model_load_ms=context.metadata["model_load_ms"],
            peak_allocated_bytes=worker.torch.cuda.max_memory_allocated(worker.device),
            peak_reserved_bytes=worker.torch.cuda.max_memory_reserved(worker.device),
            scope="fresh process loading, charged; no inherited warmup or compiled backend"))
        guard.ensure_unit(estimate(config)["online_admission_seconds"] + 540.)
        warmed = capacity(context)
        atomic_json(args.output / "capacity.json", warmed)
        warm_cache_factorial(context)
        result = run_cache_online(context, result)
        # The replay record describes admission, while this result describes
        # the fresh run. Keep the specific local/online qualification fields.
        result.pop("scientific_qualified", None)
        result.pop("system_qualified", None)
        if not result.get("online_evaluation_complete"):
            result.update(status="incomplete", decision="未验证",
                reason=result.get("online_unverified_reason", "complete paired online matrix absent"))
        else:
            result.update(status="passed", decision="值得继续" if result["online_qualified"] else "当前配置无收益",
                reason="complete online factorial; paired_008+K4 " +
                    ("passed" if result["online_qualified"] else "did not pass") + " the frozen system gate")
        result["new_local_executions"] = 0
        result["metadata"] = context.metadata
        return result
    finally:
        if worker is not None:
            memory = dict(scope="allocator counters before cleanup since most recent peak reset; after successful "
                "warmup this covers paired warmup and online traces. Loading, capacity and off-K4 warmup peaks "
                "are recorded separately; this is not a whole-process peak")
            try:
                free, total = worker.torch.cuda.mem_get_info(worker.device)
                memory.update(allocated_bytes=worker.torch.cuda.memory_allocated(worker.device),
                    reserved_bytes=worker.torch.cuda.memory_reserved(worker.device),
                    peak_allocated_bytes=worker.torch.cuda.max_memory_allocated(worker.device),
                    peak_reserved_bytes=worker.torch.cuda.max_memory_reserved(worker.device),
                    free_bytes=free, total_bytes=total)
            except Exception as error:
                memory["error"] = str(error)
            atomic_json(args.output / "memory.json", memory)
            for executor in list(getattr(context, "executors", {}).values()) + [getattr(context, "executor", None), getattr(context, "cache_executor", None)]:
                for state in list(getattr(executor, "_states", {}).values()):
                    if not state.closed:
                        executor.close(state)
            for state in list(worker._states.values()):
                worker.release(state)
        if sana is not None:
            sana.close()


def worker_main(args):
    from .worker import verify_admission
    started = time.monotonic()
    args.phase, args.auxiliary_encoder, args.online_replay_from = PHASE, False, None
    try:
        config = read_json(args.config)
        active = verify_admission(args, config)
        unlimited = active["identity"].get("budget_unlimited", False)
        if unlimited is not (args.deadline is None):
            raise BudgetStop("worker deadline differs from charged budget authorization")
        binding = active["identity"].get("cache_replay_from", {})
        if binding.get("path") != str(args.source_supervisor.resolve()) or binding.get("sha256") != file_digest(args.source_supervisor):
            raise BudgetStop("cache replay differs from charged admission")
        auth = active["identity"]["authorization_source"]
        if file_digest(auth["path"]) != auth["sha256"]:
            raise BudgetStop("supplement authorization changed")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "cache-dit", "numpy")}
        atomic_json(args.output / "environment.json", dict(versions=versions, session_id=args.session_id,
            visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]))
        result = run_worker(args, config, active)
        code = 0 if result.get("status") == "passed" else 1
    except BaseException as error:
        result = dict(status="incomplete" if isinstance(error, BudgetStop) else "failed", decision="未验证",
            reason=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc(), online_qualified=False)
        code = 2 if isinstance(error, BudgetStop) else 1
    result.update(phase=PHASE, child_wall_seconds=time.monotonic() - started)
    atomic_json(args.output / "worker.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ditserve optimize-serving cache-online")
    parser.add_argument("command", choices=("inspect", "run", "report", "worker"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--source-supervisor", type=Path, default=SOURCE_ROOT / PHASE / "attempt00/supervisor.json")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--wait-idle-seconds", type=float, default=0.)
    parser.add_argument("--retry-from", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args(argv)
    if args.command == "worker":
        return worker_main(args)
    config = load_config(args.config)
    if args.command == "inspect":
        result = inspect_study(config, args.source_root, args.source_supervisor)
    elif args.command == "report":
        from .cache_followup_report import write_report
        result = write_report(args.output, args.ledger)
    else:
        if args.authorization is None:
            parser.error("run requires --authorization with explicit additional GPU allowance")
        result = execute(config, args.output, args.ledger, args.source_root, args.source_supervisor,
            args.authorization, wait_idle=args.wait_idle_seconds, retry_from=args.retry_from)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
