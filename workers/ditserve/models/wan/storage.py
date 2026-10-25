"""Storage-preserving copies for the explicitly admitted UniPC timestep views.

Callers first validate the complete request inventory. Arbitrary aliased feature
states remain rejected; this codec must never infer that an alias is legal.
"""

import copy
import time

from .runtime import _tree_map
from .parallel import request_engine, validate_request_engine


def snapshot_timestep_views(worker, state, mode, inventory):
    torch = worker.torch
    with torch.inference_mode():
        worker._sync()
        started = time.perf_counter()
        before = worker._memory()
        payload = worker._payload(state)
        if mode == "cold":
            payload["contexts"] = {name: copy.copy(context) for name, context in state.contexts.items()}
            for context in payload["contexts"].values():
                context.buffers = {}
        storage_copies, placements = {}, {}
        transfer_bytes = 0

        def move(tensor):
            nonlocal transfer_bytes
            if tensor.is_conj() or tensor.is_neg():
                raise ValueError("conjugate/negative storage views are unsupported")
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in storage_copies:
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),), (1,))
                host = torch.empty(storage.nbytes(), dtype=torch.uint8, device="cpu", pin_memory=tensor.device.type == "cuda")
                host.copy_(raw, non_blocking=tensor.device.type == "cuda")
                storage_copies[key] = host
                if tensor.device.type == "cuda":
                    transfer_bytes += storage.nbytes()
            host = storage_copies[key]
            converted = torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
                host.untyped_storage(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
            if tensor.device.type == "cuda":
                placements[id(converted)] = str(tensor.device)
            return converted

        host_payload = _tree_map(payload, torch.is_tensor, move)
        worker._sync()
        worker._install_payload(state, host_payload)
        state.pending_snapshot = dict(mode=mode, placements=placements, rng_state=host_payload["rng_state"],
                                      research_storage_codec=True, engine=request_engine(worker))
        state.residency = "cpu"
        state.needs_anchor = state.needs_anchor or (mode == "cold" and state.cache_enabled)
        del payload, host_payload, storage_copies
        worker._sync()
        event = dict(kind="snapshot", mode=mode, ms=(time.perf_counter() - started) * 1000,
                     transfer_bytes=transfer_bytes, bytes=transfer_bytes, state_bytes=inventory,
                     memory_before=before, memory_after=worker._memory(), residency=state.residency,
                     storage_codec="known_unipc_timestep_views")
        state.events.append(event)
        return event


def restore_timestep_views(worker, state):
    torch = worker.torch
    with torch.inference_mode():
        worker._sync()
        started = time.perf_counter()
        before = worker._memory()
        saved = state.pending_snapshot
        validate_request_engine(worker, saved.get("engine"))
        copies, transfer_bytes = {}, 0

        def move(tensor):
            nonlocal transfer_bytes
            target = saved["placements"].get(id(tensor))
            if target is None:
                return tensor
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes(), target)
            if key not in copies:
                raw = torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage, 0, (storage.nbytes(),), (1,))
                copies[key] = raw.to(target, non_blocking=True)
                transfer_bytes += storage.nbytes()
            return torch.empty(0, dtype=tensor.dtype, device=target).set_(
                copies[key].untyped_storage(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))

        payload = worker._payload(state)
        payload["rng_state"] = saved["rng_state"]
        restored = _tree_map(payload, torch.is_tensor, move)
        worker._sync()
        worker._install_payload(state, restored)
        state.pending_snapshot = None
        state.residency = "gpu"
        event = dict(kind="restore", mode=saved["mode"], ms=(time.perf_counter() - started) * 1000,
                     transfer_bytes=transfer_bytes, bytes=transfer_bytes, memory_before=before,
                     memory_after=worker._memory(), needs_anchor=state.needs_anchor,
                     storage_codec="known_unipc_timestep_views")
        state.events.append(event)
        return event
