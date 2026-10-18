"""Small JSON-facing adapter contract; model objects never cross the wire."""

import hashlib
from pathlib import Path
import time

from ..runtime import file_hash


def positive_int(value, name="steps"):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


OUTPUT_MODES = ("file", "digest", "check")


def artifact(value, path=None, mode="file"):
    """Persist requested output, returning metadata rather than tensor JSON.

    ``digest`` hashes without writing; ``check`` only validates shape and
    finiteness, for serving runs whose outputs have no downstream consumer.
    """
    if mode not in OUTPUT_MODES:
        raise ValueError("output_mode must be file, digest or check")
    if mode != "file":
        path = None
    started = time.perf_counter()
    if isinstance(value, str):
        data = value.encode("utf-8")
        result = {"text": value, "output_mode": mode}
        if mode != "check":
            result["sha256"] = hashlib.sha256(data).hexdigest()
        if path:
            destination = Path(path).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(value)
            result["path"] = str(destination)
            result["bytes"] = destination.stat().st_size
        result["persistence_ms"] = (time.perf_counter() - started) * 1000
        return result
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("model output must be a finite numeric array")
    hashing = time.perf_counter()
    result = {"shape": list(array.shape), "dtype": str(array.dtype), "finite": True, "output_mode": mode}
    if mode != "check":
        result["array_sha256"] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        result["array_hash_ms"] = (time.perf_counter() - hashing) * 1000
    if path:
        destination = Path(path).resolve()
        if destination.suffix != ".npy":
            raise ValueError("array output path must end in .npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        result["path"] = str(destination)
        result["bytes"] = destination.stat().st_size
    result["persistence_ms"] = (time.perf_counter() - started) * 1000
    return result


def progress(step, steps, *, done=None, next_action=None):
    done = step >= steps if done is None else bool(done)
    return dict(step=step, steps=steps, done=done,
                next_action=next_action or ("finish" if done else "advance"))


class Backend:
    operations = ("prepare", "advance", "finish", "release")

    def __init__(self, config):
        self.config = config
        self.states = {}

    def state(self, request):
        key = request.get("state_id", request.get("snapshot_id"))
        if not isinstance(key, str) or key not in self.states:
            raise ValueError("unknown state_id")
        return self.states[key]

    def output_path(self, request, suffix=".npy"):
        key = hashlib.sha256(str(request.get("state_id", "batch")).encode()).hexdigest()[:16]
        return request.get("output_path") or str(Path(self.config.get("work_dir", ".")) /
                ("output-%s-%s%s" % (key, request["id"], suffix)))

    def persist(self, value, request, suffix=".npy"):
        return artifact(value, self.output_path(request, suffix), request.get("output_mode", "file"))

    def new_id(self, request):
        key = request.get("state_id", request.get("spec", {}).get("request_id"))
        if not isinstance(key, str) or not key:
            raise ValueError("prepare requires a nonempty state_id or spec.request_id")
        if key in self.states:
            raise ValueError("duplicate live state_id")
        return key

    def release(self, request):
        self.state(request)
        del self.states[request["state_id"]]
        return dict(released=request["state_id"])

    def close(self):
        for key in list(self.states):
            self.release({"state_id": key})
