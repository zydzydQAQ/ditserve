"""Stable correctness digests shared by current adapters and CPU contracts.

These value encodings are extracted unchanged from the historical single-device
and heterogeneous executors. Do not change their canonicalization semantics:
existing evidence and cache-state comparisons rely on their exact hashes.
"""

import dataclasses
import enum
import hashlib
from types import SimpleNamespace

from .common import digest


def state_digest(worker, state):
    """Value-based full sampler/cache/RNG digest for instrumented correctness."""
    def visit(value):
        if worker.torch.is_tensor(value):
            raw = value.detach().cpu().contiguous().reshape(-1).view(worker.torch.uint8).numpy().tobytes()
            return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(raw).hexdigest()}
        if dataclasses.is_dataclass(value):
            return visit(vars(value))
        if isinstance(value, dict):
            return {str(k): visit(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [visit(v) for v in value]
        if isinstance(value, enum.Enum):
            return str(value)
        if value is None or type(value) in (str, bool, int, float):
            return value
        if type(value).__module__ == "torch":
            return str(value)
        raise TypeError("unsupported correctness state " + type(value).__name__)
    return digest({"payload": visit(worker._payload(state)), "step_index": state.step_index,
                   "needs_anchor": state.needs_anchor, "current_step_refreshed": state.current_step_refreshed})


def correctness_state_digest(worker, state):
    """Value digest with only UniPC's set-derived bookkeeping canonicalized.

    The wire codec sorts this field because Python process hash seeds can alter
    its order. Copy the two metadata dictionaries before sorting; live scheduler
    configuration and ordered sampler histories remain untouched.
    """
    payload = dict(worker._payload(state))
    scheduler = dict(payload["scheduler"])
    config = dict(scheduler["_internal_dict"])
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    scheduler["_internal_dict"] = config
    payload["scheduler"] = scheduler
    view = SimpleNamespace(torch=worker.torch, _payload=lambda _: payload)
    return state_digest(view, state)
