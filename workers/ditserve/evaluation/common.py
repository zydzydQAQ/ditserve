"""Small JSON provenance helpers shared by data and quality adapters."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from ..runtime import file_hash

def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)

def canonical_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def seal(payload):
    result = copy.deepcopy(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_hash(result)
    return result

def verify_seal(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("sha256"), str):
        raise ValueError("missing sealed object sha256")
    body = {key: value for key, value in payload.items() if key != "sha256"}
    if canonical_hash(body) != payload["sha256"]:
        raise ValueError("sealed object hash mismatch")
    return payload

def _reject_constant(value):
    raise ValueError("nonfinite JSON value: " + value)

def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_constant)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path

def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric")
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(name + " must be " + ("nonnegative" if allow_zero else "positive"))
    return float(value)
