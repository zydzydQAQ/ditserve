"""Charged child process. Never import a model until the supervisor admits it."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path

from ..research.common import atomic_json, digest, read_json
from .budget import BudgetStop, deadline_check


def _verify_reserve(binding, args, config):
    if (not isinstance(binding, dict) or binding.get("direction") != args.direction or
            binding.get("phase") != args.phase or not isinstance(binding.get("path"), str) or
            not Path(binding["path"]).is_absolute()):
        raise BudgetStop("reserve requires an explicit same-phase failed supervisor report")
    try:
        raw = Path(binding["path"]).read_bytes()
        failed = json.loads(raw)
    except (OSError, ValueError) as error:
        raise BudgetStop("reserve failure evidence is unavailable") from error
    if (hashlib.sha256(raw).hexdigest() != binding.get("sha256") or
            failed.get("status") not in ("failed", "incomplete") or
            not isinstance(failed.get("charge"), dict) or not failed["charge"] or
            failed.get("direction") != args.direction or failed.get("phase") != args.phase or
            failed.get("config_sha256") != digest(config)):
        raise BudgetStop("reserve failure evidence changed or does not qualify for this retry")


def verify_admission(args, config, *, getpid=os.getpid, getppid=os.getppid,
                     getpgrp=os.getpgrp, monotonic=time.monotonic,
                     sleep=time.sleep, environ=None):
    """Bind this child to the supervisor's already charged session.

    Popen may run the child before child_started() atomically saves its PGID.
    Only that missing field may wait; identity failures never admit model work.
    """
    environ = os.environ if environ is None else environ
    deadline = monotonic() + 5.
    while True:
        try:
            active = read_json(args.ledger).get("active")
        except (OSError, ValueError) as error:
            raise BudgetStop("charged allocation is unavailable") from error
        if not active or active.get("session_id") != args.session_id:
            raise BudgetStop("worker does not belong to the active charged session")
        identity = active.get("identity", {})
        if active.get("stage") == "reserve":
            _verify_reserve(identity.get("reserve_for"), args, config)
        elif active.get("stage") != "direction%d" % args.direction or identity.get("reserve_for"):
            raise BudgetStop("worker budget stage differs from admission")
        if (active.get("supervisor_pid") != getppid() or
                active.get("supervisor_pid", 0) <= 1 or getpgrp() != getpid() or
                active.get("gpu_count") != 1 or
                identity.get("direction") != args.direction or
                identity.get("phase") != args.phase or
                identity.get("output") != str(args.output.resolve()) or
                identity.get("config_sha256") != digest(config)):
            raise BudgetStop("worker process or scientific identity differs from admission")
        if (config.get("device") != "cuda:0" or not config.get("gpu_uuid") or
                environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]):
            raise BudgetStop("worker GPU visibility differs from the admitted physical GPU")
        if active.get("child_pgid") is not None:
            if active["child_pgid"] != getpgrp():
                raise BudgetStop("worker process group differs from admission")
            return active
        if monotonic() >= deadline:
            raise BudgetStop("supervisor did not register this child process group")
        sleep(.05)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        config = read_json(args.config)
        admission = verify_admission(args, config)
        deadline_guard = deadline_check(args.deadline)
        def check_budget():
            deadline_guard()
            if os.getppid() != admission["supervisor_pid"]:
                raise BudgetStop("charged supervisor exited; stop orphaned worker")
        check_budget()
        versions = {}
        for name in ("torch", "diffusers", "transformers", "cache-dit", "numpy", "flash-attn"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        atomic_json(args.output / ("environment-%s.json" % args.phase),
                    dict(versions=versions, session_id=args.session_id))
        module_name = ("ditserve.directions.direction2_costs"
                       if args.direction == 2 and args.phase == "state_cost"
                       else "ditserve.directions.direction%d" % args.direction)
        module = importlib.import_module(module_name)
        result = module.run_phase(config, args.output, check_budget, args.phase)
        code = 0 if result.get("status") in ("passed", "stopped", "blocked") else 1
    except BudgetStop as error:
        result, code = dict(status="incomplete", reason=str(error), budget_stopped=True), 2
    except BaseException as error:
        result, code = dict(status="failed", reason=str(error), error_type=type(error).__name__,
                            traceback=traceback.format_exc()), 1
    result = dict(result, direction=args.direction, phase=args.phase,
                  child_wall_seconds=time.monotonic() - started)
    atomic_json(args.output / ("worker-%s.json" % args.phase), result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
