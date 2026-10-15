"""Charged scheduling child; no model imports precede verified admission."""

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

from ..research.common import atomic_json, digest, read_json
from .next_budget import AUTHORIZATION, SCHEMA, WatchdogStop, DeadlineCheck
from .next_runner import GPU_UUID, PHASES


def verify_admission(args, config, *, getpid=os.getpid, getppid=os.getppid,
                     getpgrp=os.getpgrp, monotonic=time.monotonic,
                     sleep=time.sleep, environ=None, module_file=__file__, executable=sys.executable):
    """Bind device, source archive, deadline and interpreter request to one charge."""
    environ = os.environ if environ is None else environ
    deadline = monotonic() + 5.0
    while True:
        ledger = read_json(args.ledger)
        if (ledger.get("schema_version") != SCHEMA or "limit_gpu_seconds" not in ledger or ledger["limit_gpu_seconds"] is not None
                or ledger.get("authorization") != AUTHORIZATION):
            raise WatchdogStop("scheduling allocation identity differs from admission")
        active = ledger.get("active")
        if not active or active.get("session_id") != args.session_id:
            raise WatchdogStop("worker does not belong to the active charged scheduling session")
        identity = active.get("identity", {})
        if (active.get("supervisor_pid") != getppid() or active.get("supervisor_pid", 0) <= 1
                or getpgrp() != getpid() or active.get("gpu_count") != 1
                or active.get("phase") != args.phase or identity.get("phase") != args.phase
                or identity.get("output") != str(args.output.resolve())
                or identity.get("runtime_config_sha256") != digest(config)
                or identity.get("main_config_sha256") != config.get("main_config_sha256")
                or identity.get("deadline_monotonic") != args.deadline):
            raise WatchdogStop("scheduling worker process, configuration or deadline differs from admission")
        if (config.get("device") != "cuda:0" or config.get("gpu_uuid") != GPU_UUID
                or config.get("authorization") != AUTHORIZATION
                or identity.get("authorization") != AUTHORIZATION
                or config.get("sana_profile") != "v100_fp32" or config.get("wan_attention") != "efficient"
                or identity.get("gpu_uuid") != config["gpu_uuid"]
                or environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]):
            raise WatchdogStop("scheduling worker GPU visibility differs from admission")
        if Path(executable).resolve() != Path(config["python"]).resolve():
            raise WatchdogStop("scheduling worker interpreter differs from admission")
        execution_root = Path(identity.get("execution_root", "")).resolve()
        if (Path(module_file).resolve() != execution_root / "python/ditserve/scheduling/next_worker.py"
                or args.source_manifest.resolve() != execution_root / "manifest.json"):
            raise WatchdogStop("scheduling worker is not executing its frozen source archive")
        source = read_json(args.source_manifest)
        if (source.get("execution_root") != str(execution_root)
                or source.get("source_sha256") != identity.get("source_sha256")
                or source.get("runtime_config_sha256") != digest(config)
                or source.get("phase") != args.phase):
            raise WatchdogStop("scheduling source archive differs from admission")
        if active.get("child_pgid") is not None:
            if active["child_pgid"] != getpgrp():
                raise WatchdogStop("scheduling worker process group differs from admission")
            from .next_runner import verify_archive
            verify_archive(source)
            if args.deadline <= monotonic():
                raise WatchdogStop("finite phase watchdog already expired")
            return active
        if monotonic() >= deadline:
            raise WatchdogStop("supervisor did not register scheduling worker process group")
        sleep(0.05)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.output / args.phase
    started = time.monotonic()
    try:
        config = read_json(args.config)
        admission = verify_admission(args, config)
        check_watchdog = DeadlineCheck(args.deadline, parent_pid=admission["supervisor_pid"])
        check_watchdog()
        versions = {}
        for name in ("torch", "diffusers", "transformers", "cache-dit", "numpy", "flash-attn"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        atomic_json(root / "environment.json", dict(versions=versions,
                    session_id=args.session_id, gpu_uuid=config["gpu_uuid"],
                    python=config["python"], sana_profile=config["sana_profile"],
                    wan_attention=config["wan_attention"], runtime_config_sha256=digest(config)))
        execution = importlib.import_module("ditserve.scheduling.next_execution")
        result = execution.run_phase(config, args.output, check_watchdog, args.phase)
        if not isinstance(result, dict) or result.get("status") not in ("passed", "stopped", "blocked", "incomplete", "failed"):
            raise ValueError("execution returned an invalid scheduling phase result")
        code = 0 if result["status"] in ("passed", "stopped") else 2 if result["status"] == "incomplete" else 1
    except WatchdogStop as error:
        result, code = dict(status="incomplete", eligible=False, reason=str(error), watchdog_stopped=True), 2
    except BaseException as error:
        result, code = dict(status="failed", eligible=False, reason=str(error),
                            error_type=type(error).__name__, traceback=traceback.format_exc()), 1
    result = dict(result, phase=args.phase, session_id=args.session_id,
                  child_wall_seconds=time.monotonic() - started)
    atomic_json(root / "worker.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
