"""Serial, isolated and fully charged optimization supervisor."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..research.common import atomic_json, read_json, file_digest, digest
from ..research.isolation import gpu_occupancy
from ..directions.runner import inspect as inspect_models, stop_owned_group
from .budget import Allocation, BudgetStop
from .protocol import ROOT, CONFIG, OUTPUT, LEDGER, PARENT, PHASES, TOTAL, load_config, freeze, qualified


def terminal(output, phase):
    path = Path(output) / phase / "terminal.json"
    return read_json(path) if path.exists() else None


def inspect(config):
    result = inspect_models(config, 1)
    result["historical_ledger_sha256"] = file_digest(PARENT)
    result["additional_budget_gpu_hours"] = TOTAL / 3600
    result["gpu_work_started"] = False
    return result


def write_report(output, ledger):
    from .reporting import write_report as render
    return render(output, ledger)


def _trigger(output, phase):
    if phase not in ("text_encoder", "vae"):
        return None
    evidence = []
    # Re-evaluate against the newest measured default. A newly qualified
    # candidate can remove an earlier bottleneck; older triggers must not win.
    for name in reversed(PHASES[:PHASES.index(phase)]):
        record = terminal(output, name)
        if (not record or not record.get("online_path") or record.get("status") in ("failed", "incomplete", "blocked")
                or record.get("returncode") != 0 or not record.get("charge") or record.get("supervisor_failure")
                or record.get("final_occupancy", {}).get("status") != "exclusive_observed"):
            continue
        path = Path(record["online_path"])
        if not path.exists():
            continue
        # Rejected online measurements cannot supply either positive or
        # negative bottleneck evidence. A valid unqualified candidate may
        # still supply its baseline, so this is separate from qualified().
        invalid_online = False
        for attempt in {path.parent, Path(record.get("output") or path.parent)}:
            audit = attempt / "online-measurement-audit.json"
            if audit.exists() or audit.is_symlink():
                try:
                    note = read_json(audit)
                    invalid_online = (not isinstance(note, dict) or
                                      note.get("invalidates_online_qualification") is True)
                except (OSError, ValueError, TypeError):
                    invalid_online = True
                if invalid_online:
                    break
        if invalid_online:
            continue
        online = read_json(path)
        if online.get("status") != "completed":
            continue
        arm = "candidate" if qualified(record) else "baseline"
        for row in online.get("rows", []):
            if (row.get("arm") == arm and row.get("status") == "completed" and row.get("complete")
                    and all(r.get("status") == "completed" for r in row.get("requests", []))):
                evidence.append(row.get("attribution", {}))
        if evidence:
            break
    if not evidence:
        return dict(status="stopped", decision="未验证", local_qualified=False, online_qualified=False,
                    reason="no complete instrumented online baseline to evaluate conditional trigger", gpu_work_started=False)
    key = "encoding_trigger" if phase == "text_encoder" else "vae_trigger"
    if not any(value.get(key) is True for value in evidence):
        return dict(status="stopped", decision="简单策略已足够", local_qualified=False, online_qualified=False,
                    reason="predeclared bottleneck trigger absent in measured baseline", trigger_evidence=evidence,
                    gpu_work_started=False, scope="conditional optimization not run; not proof of universal no benefit")
    return None


def execute(config, output, ledger, phase, *, wait_idle=0., retry_from=None, auxiliary_encoder=False,
            online_replay_from=None):
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    replay = None
    if online_replay_from is not None:
        if phase != "wan_compile" or retry_from is not None or auxiliary_encoder:
            raise ValueError("online replay is an explicit Wan measurement repair, separate from ordinary retries")
        from .replay import inspect_local_replay
        replay = inspect_local_replay(online_replay_from, config, output)
    explicit_encoder_fallback = auxiliary_encoder
    index = PHASES.index(phase)
    if any(terminal(output, previous) is None for previous in PHASES[:index]):
        raise ValueError("earlier optimization stages must have terminal conclusions")
    previous = terminal(output, phase)
    if auxiliary_encoder and (phase != "text_encoder" or not previous or not previous.get("needs_v100")):
        raise ValueError("V100 fallback requires recorded A100 encoder capacity failure")
    inherited_encoder = terminal(output, "text_encoder") if phase in ("vae", "cache_k4") else None
    if qualified(inherited_encoder) and inherited_encoder.get("encoder_device") == "cuda:1":
        auxiliary_encoder = True
    devices = [config["gpu_uuid"]] + ([config["v100_uuid"]] if auxiliary_encoder else [])
    gpu_count = len(devices)
    retry = None
    if retry_from is not None:
        retry_path = Path(retry_from).resolve()
        failure = read_json(retry_path)
        if (failure.get("status") not in ("failed", "incomplete") or not failure.get("charge")
                or failure.get("phase") != phase or failure.get("config_sha256") != digest(config)):
            raise ValueError("retry requires charged same-configuration phase failure")
        retry = dict(path=str(retry_path), sha256=file_digest(retry_path))
    elif previous and not explicit_encoder_fallback and replay is None:
        return previous
    root = output / phase
    root.mkdir(parents=True, exist_ok=True)
    conditional = None if retry else _trigger(output, phase)
    if conditional is not None:
        conditional.update(phase=phase, config_sha256=digest(config))
        atomic_json(root / "terminal.json", conditional)
        return conditional
    wait_deadline = time.monotonic() + wait_idle
    while True:
        observation = inspect(config)
        if auxiliary_encoder:
            extra = gpu_occupancy(devices)
            observation["both_devices_occupancy"] = extra
            observed_devices = {row["uuid"] for row in observation.get("devices", {}).get("devices", [])}
            if extra["status"] != "exclusive_observed" or not set(devices) <= observed_devices:
                observation["status"] = "blocked"
        atomic_json(root / "preflight.json", observation)
        if observation["status"] == "ready":
            break
        if time.monotonic() >= wait_deadline:
            occupancy_key = ("both_devices_occupancy" if "both_devices_occupancy" in observation
                             else "occupancy")
            occupancy = observation.get(occupancy_key, {})
            discovery = observation.get("devices", {})
            visible = {row["uuid"]: row.get("name", row["uuid"])
                       for row in discovery.get("devices", [])}
            labels = {config["gpu_uuid"]: "A100", config.get("v100_uuid"): "V100"}
            reasons, requested = [], []
            for uuid in devices:
                name = visible.get(uuid, labels.get(uuid, uuid))
                foreign = [row["pid"] for row in occupancy.get("foreign", []) if row.get("uuid") == uuid]
                requested.append(dict(uuid=uuid, name=name, visible=uuid in visible, foreign_pids=foreign))
                if foreign:
                    reasons.append("%s (%s) busy: foreign PID(s) %s" % (name, uuid, ", ".join(map(str, foreign))))
                if discovery.get("status") == "available" and uuid not in visible:
                    reasons.append("%s (%s) unavailable: device not visible" % (name, uuid))
            if occupancy.get("status") not in ("exclusive_observed", "contended"):
                reasons.append("required GPU occupancy unavailable: " + str(occupancy.get("reason", "unknown status")))
            if discovery.get("status") not in (None, "available"):
                reasons.append("required GPU discovery unavailable: " + str(discovery.get("reason", discovery["status"])))
            if observation.get("missing_models"):
                reasons.append("model assets incomplete: " + ", ".join(map(str, observation["missing_models"])))
            return dict(status="blocked", phase=phase,
                        reason="; ".join(reasons) or "required GPU/model preflight not ready; see admission evidence",
                        gpu_work_started=False,
                        admission_evidence=dict(occupancy_source=occupancy_key, occupancy=occupancy,
                            required_devices=requested, missing_models=observation.get("missing_models", []),
                            preflight_path=str(root / "preflight.json")))
        time.sleep(min(5., max(0., wait_deadline - time.monotonic())))
    attempt = "attempt%02d" % len(list(root.glob("attempt[0-9][0-9]")))
    destination = root / attempt
    destination.mkdir()
    process = None
    failure = None
    returncode = None
    with Allocation(ledger, PARENT) as budget:
        # A repair does not discard unused allowance of its current stage.
        # Charge the same phase while that allowance can admit a worker; the
        # dedicated reserve is eligible only for a recorded same-config retry.
        stage = ("reserve" if (retry or replay) and budget.remaining(phase) < 180. * gpu_count else phase)
        available = budget.remaining(stage)
        if available < 180. * gpu_count:
            result = dict(status="incomplete", decision="未验证", phase=phase, reason="budget cannot admit complete unit and cleanup",
                          config_sha256=digest(config), gpu_work_started=False, local_qualified=False, online_qualified=False)
            atomic_json(destination / "supervisor.json", result)
            atomic_json(root / "terminal.json", result)
            return result
        source = freeze(config, output, phase, attempt)
        atomic_json(destination / "config.json", config)
        budget.begin(stage, gpu_count=gpu_count, identity=dict(phase=phase, output=str(destination), run_root=str(output),
                     config_sha256=digest(config), source_sha256=source["source_sha256"], retry_from=retry,
                     gpu_uuids=devices, auxiliary_encoder=auxiliary_encoder, online_replay_from=replay))
        deadline = time.monotonic() + available / gpu_count - 60.
        command = [config["python"], "-u", "-m", "ditserve.optimizations.worker", "--config", str(destination / "config.json"),
                   "--output", str(destination), "--run-root", str(output), "--phase", phase, "--deadline", str(deadline),
                   "--ledger", str(ledger), "--session-id", budget.data["active"]["session_id"]]
        if auxiliary_encoder:
            command.append("--auxiliary-encoder")
        if replay is not None:
            command += ["--online-replay-from", replay["path"]]
        env = dict(os.environ, PYTHONPATH=str(Path(source["execution_root"]) / "python"),
                   CUDA_VISIBLE_DEVICES=",".join(devices), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", TORCH_HOME=config["torch_home"],
                   TORCHINDUCTOR_CACHE_DIR=str(destination / "compiler-cache"), TOKENIZERS_PARALLELISM="false")
        try:
            with (destination / "worker.log").open("x") as log:
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                budget.child_started(process.pid)
                with (destination / "isolation.jsonl").open("x") as events:
                    while True:
                        observation = gpu_occupancy(devices, allowed_pgid=process.pid)
                        events.write(json.dumps(observation) + "\n")
                        events.flush()
                        if observation["status"] != "exclusive_observed":
                            raise RuntimeError("GPU isolation lost: " + observation["status"])
                        if time.monotonic() > deadline + 5.:
                            raise BudgetStop("worker exceeded cooperative phase deadline")
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
                raise RuntimeError("cleanup unconfirmed; allocation intentionally left active")
            if final["status"] != "exclusive_observed":
                failure = dict(error_type="GPUIsolationError", reason="foreign process at cleanup")
            if file_digest(PARENT) != budget.data["parent"]["sha256"]:
                failure = dict(error_type="ParentIdentityError", reason="historical ledger changed")
            changed = [name for name, sha in source["source_files"].items()
                       if file_digest(Path(source["execution_root"]) / name) != sha]
            if changed:
                failure = dict(error_type="SourceIdentityError", reason="frozen source changed", files=changed)
            charge = budget.finish("failed" if failure or returncode else "completed")
    path = destination / "worker.json"
    result = read_json(path) if path.exists() else dict(status="incomplete", reason="worker result missing")
    if failure:
        result.update(status="incomplete", decision="未验证", supervisor_failure=failure,
                      reason=failure["reason"], local_qualified=False, online_qualified=False)
    if returncode != 0:
        result.update(status="failed" if result.get("status") == "failed" else "incomplete",
                      decision="未验证", local_qualified=False, online_qualified=False)
    result.update(phase=phase, charge=charge, config_sha256=digest(config), source_sha256=source["source_sha256"],
                  returncode=returncode, output=str(destination), final_occupancy=final)
    if replay is not None:
        result["online_replay_from"] = replay
    result.setdefault("decision", "未验证" if result.get("status") in ("failed", "incomplete") else "当前配置无收益")
    atomic_json(destination / "supervisor.json", result)
    atomic_json(root / "terminal.json", result)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "next-study":
        from .next_study import main as next_main
        return next_main(argv[1:])
    if argv and argv[0] == "cache-online":
        from .cache_followup import main as cache_main
        return cache_main(argv[1:])
    parser = argparse.ArgumentParser(prog="ditserve optimize-serving")
    parser.add_argument("command", choices=("inspect", "run", "report"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--wait-idle-seconds", type=float, default=0.)
    parser.add_argument("--retry-from", type=Path)
    parser.add_argument("--online-replay-from", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "inspect":
        result = inspect(config)
    elif args.command == "report":
        result = write_report(args.output, args.ledger)
    else:
        if args.retry_from and not args.phase:
            parser.error("retry requires explicit phase")
        if args.online_replay_from and (args.phase != "wan_compile" or args.retry_from):
            parser.error("online replay requires explicit wan_compile and cannot be combined with retry")
        result = None
        for phase in (args.phase,) if args.phase else PHASES:
            result = execute(config, args.output, args.ledger, phase, wait_idle=args.wait_idle_seconds,
                             retry_from=args.retry_from, online_replay_from=args.online_replay_from)
            if result.get("needs_v100") and not args.retry_from:
                result = execute(config, args.output, args.ledger, phase, wait_idle=args.wait_idle_seconds, auxiliary_encoder=True)
            print(json.dumps(dict(phase=phase, status=result["status"], reason=result.get("reason")), ensure_ascii=False), flush=True)
            write_report(args.output, args.ledger)
            if result["status"] == "blocked":
                break
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
