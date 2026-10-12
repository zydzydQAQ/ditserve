"""Repository paths and JSON persistence helpers."""

import datetime
import json
from pathlib import Path


_PACKAGE = Path(__file__).resolve().parent
_REPOSITORY = _PACKAGE.parents[1]
ROOT = _REPOSITORY if (_REPOSITORY / "pyproject.toml").is_file() else Path.cwd()
RUNS = ROOT / ".cache" / "runs"


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
