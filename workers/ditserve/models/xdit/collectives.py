"""Collective backend for the Wan xDiT data path: NCCL (default) or group-free collectives.

With ``backend="gfc"`` the Ulysses all-to-all, the SP output all-gather, the CFG prediction
exchange and the doorbell flag reduction run on the vendored group-free collective runtime
(``workers/gfc``, MIT): after one symmetric-memory rendezvous on the world group, any ordered
rank subset is registered locally in microseconds and served by Triton kernels that read peer
memory directly, with no ``torch.distributed.new_group``. NCCL stays the default and stays
bit-for-bit unchanged; every GFC call either runs on GFC or raises (no silent fallback).

Adapted from GF-DiT ``vllm_omni/diffusion/distributed/collective_runtime.py``
(https://github.com/SJTU-Liquid/GF-DiT, commit 442531e, Apache-2.0): the submit helper keeps
its stream handshake and ``record_stream`` discipline; the logical-group bookkeeping is
reduced to the rank tuples DITSERVE already uses.
"""

from __future__ import annotations

import importlib
import time

BACKENDS = ("nccl", "gfc")

_BACKEND = "nccl"
_RUNTIME = None
_DESCRIPTORS = {}        # ordered rank tuple -> GFC GroupDescriptor
_GROUP_RANKS = {}        # id(torch ProcessGroup) -> ordered rank tuple
_STATS = {"register_us": [], "calls": {}}


def backend():
    return _BACKEND


def uses_gfc():
    return _BACKEND == "gfc"


def configure(name, *, device=None, max_collective_bytes=512 * 1024 * 1024, use_copy_engine=False):
    """Select the backend on this rank. ``gfc`` is collective over the initialized world group:
    every rank must call it at the same point (the Wan distributed init does)."""
    global _BACKEND, _RUNTIME
    if name not in BACKENDS:
        raise ValueError(f"collective_backend must be one of {BACKENDS}, got {name!r}")
    if name == "nccl":
        shutdown()
        return {"backend": "nccl"}
    torch = importlib.import_module("torch")
    dist = torch.distributed
    if not dist.is_initialized():
        raise RuntimeError("collective_backend='gfc' requires an initialized world process group")
    if device is None or torch.device(device).type != "cuda":
        raise RuntimeError(f"collective_backend='gfc' requires a CUDA device, got {device!r}")
    if _RUNTIME is not None:
        return info()
    gfc = importlib.import_module("gfc")
    config = gfc.SymmetricCollectiveConfig(
        max_group_size=dist.get_world_size(), max_collective_bytes=int(max_collective_bytes),
        use_copy_engine=bool(use_copy_engine), enable_debug_checks=False,
    )
    started = time.perf_counter()
    _RUNTIME = gfc.SymmetricCollectiveRuntime(config, device=torch.device(device))
    _STATS["init_ms"] = (time.perf_counter() - started) * 1000
    _STATS["config"] = {"max_collective_bytes": int(max_collective_bytes), "use_copy_engine": bool(use_copy_engine)}
    _BACKEND = "gfc"
    return info()


def info():
    calls = {name: len(values) for name, values in _STATS["calls"].items()}
    register = sorted(_STATS["register_us"])
    return {
        "backend": _BACKEND,
        "init_ms": _STATS.get("init_ms"),
        "config": _STATS.get("config"),
        "registered_groups": len(_DESCRIPTORS),
        "register_us_max": register[-1] if register else None,
        "calls": calls,
        "source": "vendored gfc (SJTU-Liquid/group-free-collectives); adapter after GF-DiT collective_runtime.py",
    }


def bind(process_group, ranks):
    """Associate a torch process group (as handed to xFuser) with its ordered ranks."""
    if process_group is not None:
        _GROUP_RANKS[id(process_group)] = tuple(int(rank) for rank in ranks)


def descriptor(ranks):
    """The GFC group for an ordered rank tuple, registered on first use (local, no collective)."""
    ranks = tuple(int(rank) for rank in ranks)
    found = _DESCRIPTORS.get(ranks)
    if found is None:
        if _RUNTIME is None:
            raise RuntimeError("GFC runtime is not configured")
        started = time.perf_counter()
        found = _RUNTIME.register_group(ranks)
        _STATS["register_us"].append((time.perf_counter() - started) * 1e6)
        _DESCRIPTORS[ranks] = found
    return found


def _ranks_of(group):
    if isinstance(group, tuple):
        return group
    ranks = _GROUP_RANKS.get(id(group))
    if ranks is None:
        raise RuntimeError("collective_backend='gfc' is active but this process group was never bound to its ranks")
    return ranks


def _count(name, nbytes):
    _STATS["calls"].setdefault(name, []).append(int(nbytes))


def _submit(tensors, fn):
    """GF-DiT's handshake: GFC stream waits for the caller's stream, buffers stay alive until the
    GFC stream is done with them, and the caller's stream waits for the result (no host sync)."""
    torch = importlib.import_module("torch")
    first = tensors[0]
    if not first.is_cuda:
        raise RuntimeError(f"collective_backend='gfc' requires CUDA tensors, got {first.device!r}")
    stream = torch.cuda.current_stream(first.device)
    start = torch.cuda.Event()
    start.record(stream)
    _RUNTIME.wait_for_external(start)
    for tensor in tensors:
        tensor.record_stream(_RUNTIME.stream)
    fn(_RUNTIME)
    stream.wait_event(_RUNTIME.record_event())


def all_to_all_flat(flat, group):
    """Equal-split all-to-all of a contiguous 1-D tensor; None when NCCL is the backend or oversized."""
    if not uses_gfc():
        return None
    ranks = _ranks_of(group)
    if len(ranks) == 1:
        return flat.clone()
    if flat.nbytes % len(ranks):
        raise ValueError("all-to-all bytes must divide by the group size")
    max_bytes = getattr(getattr(_RUNTIME, "config", None), "max_collective_bytes", None)
    if max_bytes is None:
        max_bytes = _STATS.get("config", {}).get("max_collective_bytes")
    if max_bytes is not None and int(flat.nbytes) > int(max_bytes):
        _count("all2all_oversize_fallback", flat.nbytes)
        return None
    source = flat.contiguous()
    output = importlib.import_module("torch").empty_like(source)
    target = descriptor(ranks)
    _submit((source, output), lambda runtime: runtime.all2all(source, output, target, slice_bytes=source.nbytes // len(ranks)))
    _count("all2all", source.nbytes)
    return output


def all_gather_into(output, tensor, ranks):
    """``output`` (contiguous, len(ranks) x tensor) receives every rank's tensor; False on NCCL."""
    if not uses_gfc():
        return False
    ranks = _ranks_of(ranks)
    source = tensor.contiguous()
    if len(ranks) == 1:
        output.reshape(-1).copy_(source.reshape(-1))
        return True
    if not output.is_contiguous():
        raise ValueError("GFC all-gather needs a contiguous output")
    target = descriptor(ranks)
    _submit((source, output), lambda runtime: runtime.all_gather(source, output, target))
    _count("all_gather", source.nbytes)
    return True


def all_gather_chunks(tensor, ranks):
    """List of every rank's tensor in rank order; None on NCCL."""
    if not uses_gfc():
        return None
    torch = importlib.import_module("torch")
    ranks = tuple(ranks)
    output = torch.empty((len(ranks), *tensor.shape), dtype=tensor.dtype, device=tensor.device)
    all_gather_into(output, tensor, ranks)
    return list(output.unbind(0))


def all_reduce_max(flag, ranks):
    """Element-wise max over the group (doorbell consensus), written into ``flag``; False on NCCL."""
    if not uses_gfc():
        return False
    chunks = all_gather_chunks(flag, tuple(ranks))
    flag.copy_(importlib.import_module("torch").stack(chunks).amax(dim=0))
    return True


def shutdown():
    global _BACKEND, _RUNTIME
    if _RUNTIME is not None:
        _RUNTIME.shutdown()
    _RUNTIME = None
    _BACKEND = "nccl"
    _DESCRIPTORS.clear()
    _GROUP_RANKS.clear()
    _STATS["register_us"].clear()
    _STATS["calls"].clear()
    _STATS.pop("init_ms", None)
    _STATS.pop("config", None)
