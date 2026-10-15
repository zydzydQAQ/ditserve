"""Serial supervisor for the explicitly authorized serving follow-on study."""
import argparse
import importlib.metadata
import json
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
from .protocol import ROOT, freeze, load_config as original_config
from .next_protocol import CONFIG, OUTPUT, LEDGER, AUTHORIZATION, PHASES, PARENTS, DEFAULTS, load_config, authorization, safe_paths


def inspect_study(config, *, observe=True):
    from .cache_followup import inspect_study as inspect_cache
    from .next_workloads import build_workloads
    inherited = inspect_cache(original_config(), observe=False)
    baseline_path = ROOT / "results/optimization-cache-online/run01/report.json"
    baseline = read_json(baseline_path)
    if not all(baseline.get(key) is True for key in ("archive_verified", "matrix_complete", "online_evaluation_complete", "online_qualified")):
        raise ValueError("current paired-K4 online baseline lacks archived qualification")
    workloads = build_workloads(config)
    value = dict(status="prepared", gpu_work_started=False, config_sha256=digest(config),
        workloads_sha256=digest(workloads), workloads=workloads,
        cache_local_binding=inherited["replay"], defaults=DEFAULTS,
        online_baseline_binding=dict(path=str(baseline_path), sha256=file_digest(baseline_path)),
        historical_ledgers=config["next_study"]["historical_ledgers"],
        scope="independent follow-on; original local quality is reused only for unchanged cache arithmetic")
    if observe:
        from .runner import inspect as inspect_gpu
        value["preflight"] = inspect_gpu(config)
        value["preflight"].pop("additional_budget_gpu_hours", None)
        value["status"] = "ready" if value["preflight"]["status"] == "ready" else "waiting_resource"
    return value


def terminal(output, phase):
    path = Path(output) / phase / "terminal.json"
    return read_json(path) if path.exists() else None


def verify_cache_binding(binding, execution_root):
    """Match the actual executable snapshot to the inherited local qualification."""
    hashes = binding["mathematical_source_files"]
    if not hashes or digest(hashes) != binding["mathematical_source_sha256"]:
        raise ValueError("inherited cache mathematical manifest is inconsistent")
    root = Path(execution_root).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if root not in path.parents or file_digest(path) != expected:
            raise ValueError("executed cache arithmetic differs from qualified source: " + name)
    for source in binding["bindings"].values():
        if file_digest(source["path"]) != source["sha256"]:
            raise ValueError("inherited cache qualification changed: " + source["path"])


def execute(config, output, ledger, phase, authorization_path, *, wait_idle=0., retry_from=None):
    from .next_budget import NextAllocation
    output, ledger = safe_paths(output, ledger)
    _, auth_binding = authorization(authorization_path)
    if phase not in PHASES:
        raise ValueError("unknown follow-on phase")
    if any(terminal(output, p) is None for p in PHASES[:PHASES.index(phase)]):
        raise ValueError("earlier stages need conclusions before proceeding")
    root = output / phase
    previous = terminal(output, phase)
    if previous is not None and retry_from is None:
        return previous
    retry = None
    if retry_from is not None:
        retry_path = Path(retry_from).resolve(strict=True)
        if root not in retry_path.parents:
            raise ValueError("retry must belong to this phase")
        retry = dict(path=str(retry_path), sha256=file_digest(retry_path))
    prepared = inspect_study(config, observe=False)
    conditional = None
    if phase == "conditional":
        from .next_conditional import select_trigger_evidence
        conditional = select_trigger_evidence(output, ledger)
    root.mkdir(parents=True, exist_ok=True)
    study = dict(schema_version="ditserve.optimizations.next_study.v1", config_sha256=digest(config),
        authorization=auth_binding, budget_unlimited=True, authorized_gpu_seconds=None,
        workloads_sha256=prepared["workloads_sha256"], cache_local_binding=prepared["cache_local_binding"],
        online_baseline_binding=prepared["online_baseline_binding"],
        historical_ledgers=prepared["historical_ledgers"], phases=list(PHASES), defaults=DEFAULTS)
    for name, value in (("study.json", study), ("workloads.json", prepared["workloads"])):
        path = output / name
        if path.exists() and read_json(path) != value:
            raise ValueError("frozen follow-on " + name + " changed")
        atomic_json(path, value)
    # Resource waits are CPU-only. Accounting begins before any model loading.
    from .runner import inspect as inspect_gpu
    wait_deadline = time.monotonic() + max(0., wait_idle)
    auxiliary = False
    while True:
        observation = inspect_gpu(config)
        observation.pop("additional_budget_gpu_hours", None)
        if conditional and conditional.get("encoder_trigger"):
            both = gpu_occupancy([config["gpu_uuid"], config["v100_uuid"]])
            visible = {row["uuid"] for row in observation.get("devices", {}).get("devices", [])}
            auxiliary = (both.get("status") == "exclusive_observed"
                         and {config["gpu_uuid"], config["v100_uuid"]} <= visible)
            observation["conditional_encoder_admission"] = dict(occupancy=both,
                admitted=auxiliary, required_v100=config["v100_uuid"],
                wait_expired=time.monotonic() >= wait_deadline)
            # Wait without holding either card. When the bounded idle window
            # expires, independent A100 work may continue; V100 stays unverified.
            if not auxiliary and time.monotonic() < wait_deadline:
                observation["status"] = "waiting_auxiliary"
        atomic_json(root / "preflight.json", observation)
        if observation["status"] == "ready":
            break
        if time.monotonic() >= wait_deadline:
            return dict(status="blocked", decision="未验证", gpu_work_started=False,
                        reason="A100 or model admission not ready", preflight=str(root / "preflight.json"))
        time.sleep(min(5., max(0., wait_deadline - time.monotonic())))
    attempt = "attempt%02d" % len(list(root.glob("attempt[0-9][0-9]")))
    destination = root / attempt
    process = None
    failure = None
    returncode = None
    devices = [config["gpu_uuid"]] + ([config["v100_uuid"]] if auxiliary else [])
    with NextAllocation(ledger, parents=PARENTS, authorization_binding=auth_binding,
                        gpu_uuid=config["gpu_uuid"], auxiliary_gpu_uuid=config.get("v100_uuid"), stages=PHASES) as budget:
        destination.mkdir()
        source = freeze(config, output, phase, attempt)
        verify_cache_binding(prepared["cache_local_binding"], source["execution_root"])
        atomic_json(destination / "config.json", config)
        atomic_json(destination / "study.json", study)
        identity = dict(phase=phase, output=str(destination), run_root=str(output),
            config_sha256=digest(config), source_sha256=source["source_sha256"], gpu_uuids=devices,
            auxiliary_encoder=auxiliary,
            retry_from=retry, budget_unlimited=True, authorization_source=auth_binding,
            online_baseline_binding=prepared["online_baseline_binding"],
            cache_local_binding=prepared["cache_local_binding"], workloads_sha256=prepared["workloads_sha256"])
        if conditional is not None:
            admission = dict(trigger_evidence=conditional, auxiliary_encoder=auxiliary,
                devices=devices, resource_evidence=observation,
                accounting_scope="all admitted cards charged for complete phase lifetime")
            atomic_json(destination / "conditional-admission.json", admission)
            identity["conditional_admission"] = dict(path=str(destination / "conditional-admission.json"),
                sha256=file_digest(destination / "conditional-admission.json"))
        budget.begin(phase, gpu_count=len(devices), identity=identity, minimum_seconds=120.)
        command = [config["python"], "-u", "-m", "ditserve.optimizations.next_study", "worker",
            "--config", str(destination / "config.json"), "--output", str(destination),
            "--run-root", str(output), "--phase", phase, "--ledger", str(ledger),
            "--session-id", budget.data["active"]["session_id"]]
        if auxiliary:
            command.append("--auxiliary-encoder")
        env = dict(os.environ, PYTHONPATH=str(Path(source["execution_root"]) / "python"),
            CUDA_VISIBLE_DEVICES=",".join(devices), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", TORCH_HOME=config["torch_home"],
            TOKENIZERS_PARALLELISM="false", TORCHINDUCTOR_CACHE_DIR=str(destination / "compiler-cache"),
            TRITON_CACHE_DIR=str(destination / "triton-cache"))
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
                            raise RuntimeError("admitted GPU isolation lost: " + observation["status"])
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
            if (not gone or final.get("owned") != [] or not isinstance(final.get("foreign"), list)
                    or final.get("status") in (None, "unknown")):
                raise RuntimeError("cleanup unconfirmed; session remains active and charged")
            if final["status"] != "exclusive_observed":
                failure = dict(error_type="GPUIsolationError", reason="foreign process at cleanup")
            try:
                changed = [b["path"] for b in prepared["historical_ledgers"] if file_digest(b["path"]) != b["sha256"]]
                changed += [b["path"] for b in (prepared["online_baseline_binding"],)
                            if file_digest(b["path"]) != b["sha256"]]
                changed += [name for name, sha in source["source_files"].items()
                            if file_digest(Path(source["execution_root"]) / name) != sha]
                verify_cache_binding(prepared["cache_local_binding"], source["execution_root"])
                if identity.get("conditional_admission"):
                    from .next_followups import read_admission
                    read_admission(identity["conditional_admission"])
                if changed:
                    failure = dict(error_type="SourceIdentityError", reason="bound source or history changed", files=changed)
            except (OSError, ValueError, KeyError, TypeError) as error:
                # Once ownership cleanup is confirmed, missing evidence is a
                # charged failure; it must not bypass settlement.
                failure = dict(error_type="SourceIdentityError", reason=type(error).__name__ + ": " + str(error))
            charge = budget.finish("failed" if failure or returncode else "completed")
            if charge.get("status") != "completed" and failure is None and returncode == 0:
                failure = dict(error_type="BudgetSettlementError", reason=charge.get("parent_validation_error", "settlement failed"))
    path = destination / "worker.json"
    result = read_json(path) if path.exists() else dict(status="incomplete", reason="worker result missing")
    if failure or returncode != 0:
        result.update(status="incomplete", decision="未验证", online_qualified=False,
                      supervisor_failure=failure, reason=(failure or {}).get("reason", result.get("reason", "worker failed")))
    result.update(phase=phase, charge=charge, returncode=returncode, output=str(destination),
        final_occupancy=final, config_sha256=digest(config), source_sha256=source["source_sha256"],
        authorization_source=auth_binding)
    atomic_json(destination / "supervisor.json", result)
    atomic_json(root / "terminal.json", result)
    return result


def _cleanup(context):
    worker, sana = getattr(context, "worker", None), getattr(context, "sana", None)
    if worker is not None:
        for executor in list(getattr(context, "executors", {}).values()) + [getattr(context, "executor", None)]:
            for state in list(getattr(executor, "_states", {}).values()):
                if not state.closed:
                    executor.close(state)
        for state in list(worker._states.values()):
            worker.release(state)
        try:
            atomic_json(context.output / "memory.json", dict(worker._memory(),
                peak_allocated_bytes=worker.torch.cuda.max_memory_allocated(worker.device),
                scope="allocator peak since last request reset; per-operation peaks in evidence"))
        except Exception as error:
            atomic_json(context.output / "memory.json", dict(error=str(error)))
    if sana is not None:
        sana.close()


def run_worker(args, config, active):
    from ..gpu.worker import WanStepWorker
    from ..directions.block_executor import WanBlockExecutor
    from .next_sana import NextSanaAdapter
    from .worker import capacity
    from .next_workloads import run_diagnosis, run_cache_load
    from .cache_online import warm_cache_factorial
    guard = BudgetGuard(float("inf"), active["supervisor_pid"])
    workloads_path = args.run_root / "workloads.json"
    workloads = read_json(workloads_path)
    if digest(workloads) != active["identity"]["workloads_sha256"]:
        raise ValueError("frozen workloads changed before loading")
    binding = active["identity"]["online_baseline_binding"]
    if file_digest(binding["path"]) != binding["sha256"]:
        raise ValueError("qualified online baseline changed before loading")
    verify_cache_binding(active["identity"]["cache_local_binding"], ROOT)
    context = SimpleNamespace(config=config, output=args.output, run_root=args.run_root, check_budget=guard,
        opts=config["direction1"], profile=dict(selected_shape=dict(height=480, width=832, frames=81)),
        ledger=args.ledger, admitted_devices=active["identity"]["gpu_uuids"],
        metadata=dict(phase=args.phase, encoder_device="cuda:1" if args.auxiliary_encoder else "cpu"),
        online_defaults=dict(DEFAULTS))
    if args.phase == "conditional":
        from .next_followups import read_admission, run_followups
        context.conditional_admission = read_admission(active["identity"]["conditional_admission"])
        triggers = context.conditional_admission["trigger_evidence"]
        context.trigger_evidence = triggers
        if not (triggers.get("vae_trigger") or (triggers.get("encoder_trigger") and args.auxiliary_encoder)):
            return run_followups(context)
    started = time.perf_counter()
    try:
        guard()
        context.worker = WanStepWorker(device=config["device"], model_path=config["models"]["wan"]["path"], attention="flash")
        if args.phase == "sana_numerics":
            from .sana import OptimizedSanaAdapter
            context.sana = OptimizedSanaAdapter(config["models"]["sana"]["path"], device=config["device"])
        else:
            context.sana = NextSanaAdapter(config["models"]["sana"]["path"], device=config["device"])
        context.executor = WanBlockExecutor(context.worker)
        context.executors = {"eager": context.executor}
        context.metadata["model_load_ms"] = (time.perf_counter() - started) * 1000.
        atomic_json(args.output / "loading.json", dict(status="completed", **context.metadata))
        if args.phase == "sana_numerics":
            from .next_numerics import run_numerics
            result = run_numerics(context)
        elif args.phase in ("sana_vae", "sana_graph"):
            from .next_phases import run_candidate
            result = run_candidate(context, "vae_compile" if args.phase == "sana_vae" else "transformer_graph")
        else:
            atomic_json(args.output / "capacity.json", capacity(context))
            for i in range(2):
                context.sana.generate(context.opts["prompts"][i], seed=101+i, check_budget=guard)
            context.metadata["sana_warmed"] = True
            if args.phase == "diagnosis":
                from .next_phases import profile_sana
                profile_sana(context)
                result = run_diagnosis(context, workloads)
            elif args.phase == "conditional":
                from .next_followups import run_followups
                warm_cache_factorial(context)
                result = run_followups(context)
            else:
                warm_cache_factorial(context)
                key = {"cache_load_05": "0.5", "cache_load_07": "0.7", "cache_load_09": "0.9"}[args.phase]
                result = run_cache_load(context, workloads["highload"][key], args.phase)
        result["metadata"] = context.metadata
        return result
    finally:
        _cleanup(context)


def worker_main(args):
    from .worker import verify_admission
    started = time.monotonic()
    args.online_replay_from = None
    try:
        config = read_json(args.config)
        active = verify_admission(args, config)
        binding = active["identity"]["authorization_source"]
        if file_digest(binding["path"]) != binding["sha256"] or active["identity"].get("budget_unlimited") is not True:
            raise BudgetStop("authorization changed")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "cache-dit", "numpy")}
        atomic_json(args.output / "environment.json", dict(versions=versions, session_id=args.session_id,
            visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            compiler_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            triton_cache=os.environ.get("TRITON_CACHE_DIR"),
            compiler_cache_scope="new private attempt directories; shared driver cache is not purged"))
        result = run_worker(args, config, active)
        # Detailed replay/local rows live once in their phase evidence file.
        # Supervisor/ledger summaries must not duplicate large event streams.
        if "rows" in result:
            result["recorded_rows"] = len(result.pop("rows"))
        result["evidence_path"] = str(args.output / (
            "sana-candidate.json" if args.phase in ("sana_vae", "sana_graph") else
            "sana-numerics.json" if args.phase == "sana_numerics" else args.phase + ".json"))
        code = 1 if result.get("status") in ("failed", "incomplete") else 0
    except BaseException as error:
        result = dict(status="incomplete" if isinstance(error, BudgetStop) else "failed", decision="未验证",
            reason=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc(), online_qualified=False)
        code = 2 if isinstance(error, BudgetStop) else 1
    result.update(phase=args.phase, child_wall_seconds=time.monotonic() - started)
    atomic_json(args.output / "worker.json", result)
    print(json.dumps({k: result.get(k) for k in ("phase", "status", "decision", "reason")}, ensure_ascii=False), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ditserve optimize-serving next-study")
    parser.add_argument("command", choices=("inspect", "run", "report", "worker"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--wait-idle-seconds", type=float, default=0.)
    parser.add_argument("--retry-from", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--auxiliary-encoder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "worker":
        return worker_main(args)
    config = load_config(args.config)
    if args.command == "inspect":
        result = inspect_study(config)
        result.pop("workloads", None)
    elif args.command == "report":
        from .next_report import write_report
        result = write_report(args.output, args.ledger)
    else:
        if args.retry_from and args.phase is None:
            parser.error("retry requires an explicit phase")
        result = {}
        for phase in ((args.phase,) if args.phase else PHASES):
            print(json.dumps(dict(phase=phase, status="starting")), flush=True)
            row = execute(config, args.output, args.ledger, phase, args.authorization,
                wait_idle=args.wait_idle_seconds, retry_from=args.retry_from)
            result[phase] = {k: row.get(k) for k in ("status", "decision", "reason", "output")}
            print(json.dumps(result[phase], ensure_ascii=False), flush=True)
            if row.get("status") == "blocked":
                break
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
