"""Shared paths and JSON validation for native clients and Python workers."""

import hashlib
import json
import os
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY = PACKAGE_ROOT.parent
ROOT = Path(os.environ["DITSERVE_ROOT"]).absolute() if os.environ.get("DITSERVE_ROOT") else (
    _REPOSITORY if (_REPOSITORY / "pyproject.toml").is_file() else Path.cwd())
CONFIGS = next((path for path in (
    ROOT / "configs",
    PACKAGE_ROOT.parent / "configs",
    PACKAGE_ROOT / "share" / "ditserve" / "configs",
    Path(sys.prefix) / "share" / "ditserve" / "configs",
) if path.is_dir()), ROOT / "configs")


def strict_json(text):
    def reject_constant(value):
        raise ValueError("non-finite JSON number: " + value)
    return json.loads(text, parse_constant=reject_constant)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
