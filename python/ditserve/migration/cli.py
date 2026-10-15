"""Ordered migration operations. Help and inspect never initialize CUDA."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from ..util import child_environment


def _groups(value):
    groups = tuple(part.strip() for part in value.split(","))
    if not groups or set(groups) - {"core", "sessions", "longlive"} or len(set(groups)) != len(groups):
        raise argparse.ArgumentTypeError("use unique comma-separated core,sessions,longlive groups")
    return groups


def smoke(config_path, groups, *, timeout=900):
    from .checks import run_owned, select_a100
    from .models import verify_models
    from .configure import validate_environment
    from ..research.preflight import discover_gpus
    config_path = Path(config_path).resolve()
    config = validate_environment(json.loads(config_path.read_text()))
    # A session also requires Sana, so verify default assets for all smoke groups.
    needed = tuple(dict.fromkeys(("core",) + tuple(groups)))
    models = verify_models(config["model_root"], needed)
    if models["status"] != "verified":
        return dict(status="failed", models=models, gpu_work_started=False)
    if "longlive" in groups:
        from ..directions.runner import model_inventory
        environment = model_inventory("longlive", config["models"]["longlive"])
        if not environment["available"]:
            return dict(status="failed", models=models, longlive_environment=environment,
                        reason="LongLive CPU environment evidence missing or invalid; rerun models download --groups longlive",
                        gpu_work_started=False)
    selected = select_a100(discover_gpus(), config["gpu_uuid"], expected=1)
    env = dict(child_environment(), CUDA_VISIBLE_DEVICES=selected[0]["uuid"],
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_HOME=config["torch_home"],
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    probes = []
    requested = (["wan", "sana"] if "core" in groups else [])
    requested += [name for name in ("sessions", "longlive") if name in groups]
    for name in requested:
        python = config["models"]["longlive"]["python"] if name == "longlive" else sys.executable
        if not Path(python).is_file():
            probes.append(dict(status="failed", probe=name, reason="interpreter not installed: " + python))
            break
        row = run_owned([python, "-m", "ditserve.migration.probes", name, "--config", str(config_path)],
                        env=env, timeout=timeout, gpu_uuids=[selected[0]["uuid"]])
        row["name"] = name
        if row["status"] == "passed" and row.get("probe", {}).get("status") != "passed":
            row.update(status="failed", reason="probe did not publish passing completion evidence")
        probes.append(row)
        if row["status"] != "passed":
            break
    return dict(status="passed" if all(row["status"] == "passed" for row in probes) else "failed",
                models=models, checks=probes, gpu_seconds=sum(row.get("gpu_seconds", 0) for row in probes),
                scope="environment_only; historical qualification is not transferred")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ditserve migrate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Read hardware, dependency and capacity inventory; no CUDA init")
    models = sub.add_parser("models", help="Download or verify pinned external model assets")
    models.add_argument("action", choices=("download", "verify"))
    models.add_argument("--root", type=Path, default=Path(os.environ.get("DITSERVE_MODEL_ROOT", "/models")))
    models.add_argument("--groups", type=_groups, default=("core",))
    models.add_argument("--longlive-python", default="/opt/longlive-venv/bin/python")
    models.add_argument("--longlive-source", default="/opt/longlive-source")
    config = sub.add_parser("configure", help="Generate immutable environment and fresh scheduling bindings")
    config.add_argument("--models-root", type=Path, default=Path("/models"))
    config.add_argument("--data-root", type=Path, default=Path("/data"))
    config.add_argument("--gpu-uuid", required=True)
    config.add_argument("--output", type=Path, default=Path("/data/configs"))
    config.add_argument("--longlive-python", default="/opt/longlive-venv/bin/python")
    config.add_argument("--longlive-source", default="/opt/longlive-source")
    check = sub.add_parser("check", help="Explicit CPU or GPU acceptance; GPU suite runs CUDA/NCCL")
    check.add_argument("--suite", choices=("cpu", "gpu"), default="cpu")
    check.add_argument("--devices", help="Comma-separated physical UUIDs or indices; default all exposed GPUs")
    check.add_argument("--expected-gpus", type=int, default=8)
    check.add_argument("--timeout", type=int, default=600, help="Per subprocess deadline in seconds")
    check.add_argument("--output", type=Path, help="New directory for results.xlsx and REPORT.md")
    sm = sub.add_parser("smoke", help="Explicit single-A100 offline model acceptance")
    sm.add_argument("--config", type=Path, required=True)
    sm.add_argument("--groups", type=_groups, default=("core",))
    sm.add_argument("--timeout", type=int, default=900)
    sm.add_argument("--output", type=Path)
    history = sub.add_parser("history-export", help="Copy stable lightweight historical evidence; never prune source")
    history.add_argument("--source-project", type=Path, required=True)
    history.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    result = None
    destination = None
    if args.command in ("check", "smoke"):
        destination = args.output or Path(os.environ.get("DITSERVE_RESULTS_ROOT", "/data/results")) / (
            "migration-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
        if any((destination / name).exists() for name in ("results.xlsx", "REPORT.md")):
            print(json.dumps(dict(status="failed", reason="acceptance output already exists; use a new directory",
                                  gpu_work_started=False, output=str(destination))))
            return 1
    try:
        if args.command in ("check", "smoke") and args.timeout <= 0:
            raise ValueError("timeout must be positive")
        if args.command == "inspect":
            from .checks import inventory
            result = inventory()
        elif args.command == "models":
            if args.action == "download":
                # Only this explicit provisioning command enables network access.
                # Set before huggingface_hub is imported (its offline flag is cached).
                os.environ["HF_HUB_OFFLINE"] = "0"
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
            from .models import download_models, verify_models
            if args.action == "download":
                result = download_models(args.root, args.groups, longlive_python=args.longlive_python,
                                         source_path=args.longlive_source)
            else:
                result = verify_models(args.root, args.groups)
        elif args.command == "configure":
            from .configure import configure
            result = configure(args.output, args.models_root, args.data_root, args.gpu_uuid,
                               longlive_python=args.longlive_python, longlive_source=args.longlive_source)
        elif args.command == "check":
            from .checks import cpu_check, gpu_check
            result = (cpu_check(args.timeout) if args.suite == "cpu" else
                      gpu_check(devices=args.devices, expected=args.expected_gpus, timeout=args.timeout))
        elif args.command == "smoke":
            result = smoke(args.config, args.groups, timeout=args.timeout)
        else:
            from .history import export_history
            result = export_history(args.source_project, args.destination)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        result = dict(status="failed", reason=str(error), operation=args.command)
    if args.command in ("check", "smoke"):
        from .reporting import write_report
        try:
            result["artifacts"] = write_report(destination, result)
        except (OSError, ValueError, ImportError) as error:
            result["report_error"] = str(error)
            result["status"] = "failed"
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 1 if result.get("status") in ("failed", "blocked", "partial") else 0


if __name__ == "__main__":
    sys.exit(main())
