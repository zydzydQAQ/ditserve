"""Append-only raw arrays on the dedicated run02 SSD, with verified link index."""
import fcntl
import hashlib
import os
from pathlib import Path

from ..util import atomic_json, read_json
from .previs_storage import INDEX_NAME, SCHEMA


def save_array(suite, stage, case_id, array, label="output", storage_root=None):
    import numpy as np
    suite, stage = Path(suite).absolute(), Path(stage).absolute()
    relative_stage = stage.relative_to(suite)
    if len(relative_stage.parts) != 1 or any("/" in v or v in ("", ".", "..") for v in (case_id, label)):
        raise ValueError("one stage and simple artifact names required")
    relative = relative_stage / "arrays" / (case_id + "-" + label + ".npy")
    logical = suite / relative
    logical.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(logical):
        raise FileExistsError("raw output is append-only: " + str(logical))
    if storage_root is None:
        physical = logical
    else:
        storage_root = Path(storage_root).absolute()
        storage_root.mkdir(parents=True, exist_ok=True)
        if (storage_root.resolve() != storage_root or storage_root == suite or suite in storage_root.parents
                or storage_root in suite.parents):
            raise ValueError("distinct real SSD storage root required")
        physical = storage_root / relative
        physical.parent.mkdir(parents=True, exist_ok=True)
        if physical.parent.resolve() != physical.parent:
            raise ValueError("artifact directory symlinks are forbidden")
    with physical.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    if storage_root is not None:
        digest = hashlib.sha256()
        with physical.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        entry = dict(logical_path=relative.as_posix(), target_path=relative.as_posix(),
                     file_sha256=digest.hexdigest(), size_bytes=physical.stat().st_size)
        # Serialize concurrent cards; publish the index before exposing the link.
        with (suite / "artifact-storage.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = suite / INDEX_NAME
            index = read_json(path) if path.exists() else dict(schema_version=SCHEMA,
                suite_root=str(suite), storage_root=str(storage_root), entries=[])
            if index["suite_root"] != str(suite) or index["storage_root"] != str(storage_root):
                raise ValueError("storage identity changed")
            if any(e["logical_path"] == entry["logical_path"] for e in index["entries"]):
                raise FileExistsError("artifact already registered")
            index["entries"].append(entry)
            atomic_json(path, index)
            logical.symlink_to(physical)
    payload = hashlib.sha256()
    flat = array.reshape(-1)
    for offset in range(0, flat.size, 2 * 1024 * 1024):
        payload.update(flat[offset:offset + 2 * 1024 * 1024].tobytes())
    return dict(path=str(logical), shape=list(array.shape), dtype=str(array.dtype),
                sha256=payload.hexdigest(), finite=bool(np.isfinite(array).all()))
