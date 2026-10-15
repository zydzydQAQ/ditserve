"""Repository paths and JSON persistence helpers."""

import datetime
import json
import os
from pathlib import Path
import sys


_PACKAGE = Path(__file__).resolve().parent
_REPOSITORY = _PACKAGE.parents[1]
ROOT = _REPOSITORY if (_REPOSITORY / "pyproject.toml").is_file() else Path.cwd()
RUNS = ROOT / ".cache" / "runs"
CONFIGS = next((path for path in (
    ROOT / "configs",
    _PACKAGE.parent / "share" / "ditserve" / "configs",
    Path(sys.prefix) / "share" / "ditserve" / "configs",
) if path.is_dir()), ROOT / "configs")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_gpu_uuid(value):
    """Normalize UUIDs without accepting numeric or ambiguous selectors."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value.startswith("gpu-"):
        value = value[4:]
    parts = value.split("-")
    if [len(part) for part in parts] != [8, 4, 4, 4, 12]:
        return None
    return value if all(c in "0123456789abcdef" for part in parts for c in part) else None


def child_environment():
    """Environment for subprocesses that must import this package from source."""
    environment = os.environ.copy()
    paths = [str(_PACKAGE.parent)]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def archive_run(raw_dir, series, results_root=None, step=None):
    """Publish a runner attempt while retaining raw files for resume."""
    from .archive import finalize_run
    raw_dir = Path(raw_dir)
    return finalize_run(raw_dir, series, step or raw_dir.name,
                        results_root=results_root or ROOT / "results")
