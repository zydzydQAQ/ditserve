"""Isolated cache-off step-16 numerical diagnostics, never serving qualification.

The production source stays paused after export. Imported copies have no
production owner and may execute exactly one native 16->17 step under a private
diagnostic ID. Backend choice belongs to the separately frozen worker job.
"""

import copy
import hashlib
import re

from .common import digest
from .parallel import pack_cache_off_request, unpack_cache_off_request


SCHEMA = "ditserve.cache-off-diagnostic-snapshot.v1"
SIGNATURE_SCHEMA = "ditserve.cache-off-diagnostic-state.v1"


def _tensor_hash(torch, tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _check_off(worker, state):
    worker._validate_request_schema(state)
    if (state.cache_enabled or state.spec["cache"] or state.contexts or state.needs_anchor
            or state.current_step_refreshed or state.pending_snapshot is not None
            or state.spec["threshold"] != 0 or state.spec["max_continuous"] != 1
            or state.spec["guidance_scale"] != 5 or state.spec["steps"] != 32):
        raise ValueError("diagnostics require the unchanged cache-off CFG5/32-step request")


def state_signature(worker, state):
    """Full mathematical state including logical aliases, without process IDs.

    Event timings, residency, owner and request names are not mathematical state.
    CPU/CUDA placement is retained; ordinal and physical device identity are not.
    """
    _check_off(worker, state)
    torch = worker.torch
    payload = dict(worker._payload(state))
    # The pinned codec explicitly reconstructs this UniPC list alias and the
    # FrozenDict attribute which exposes that same configuration value. Equal
    # list contents alone must not certify the restored container relationship.
    live_scheduler = payload["scheduler"]
    live_config = live_scheduler["_internal_dict"]
    corrector_config = live_config["disable_corrector"]
    container_aliases = {
        "scheduler.disable_corrector_is_config_item": live_scheduler["disable_corrector"] is corrector_config,
        "scheduler.config_attribute_is_config_item": getattr(live_config, "disable_corrector", None) is corrector_config,
    }
    scheduler = dict(payload["scheduler"])
    config = dict(scheduler["_internal_dict"])
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    scheduler["_internal_dict"] = config
    payload["scheduler"] = scheduler
    tensor_ids, storage_ids, tensor_records = {}, {}, {}

    def visit(value, path):
        if torch.is_tensor(value):
            if id(value) in tensor_ids:
                return dict(tensor_alias=tensor_ids[id(value)])
            tensor_ids[id(value)] = path
            storage = value.untyped_storage()
            key = (str(value.device), storage._cdata)
            storage_ids.setdefault(key, path)
            record = dict(dtype=str(value.dtype), shape=list(value.shape), stride=list(value.stride()),
                storage_offset=value.storage_offset(), storage_bytes=storage.nbytes(),
                storage_alias=storage_ids[key], placement=value.device.type,
                sha256=_tensor_hash(torch, value))
            tensor_records[path] = record
            return record
        if isinstance(value, dict):
            return {str(key): visit(value[key], path + "/" + str(key)) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return dict(sequence_type=type(value).__name__, values=[visit(child, path + "/" + str(i))
                                                                   for i, child in enumerate(value)])
        if value is None or type(value) in (str, bool, int, float):
            return value
        if type(value).__module__ == "torch" and type(value).__name__ in ("device", "dtype"):
            return str(value).split(":")[0] if type(value).__name__ == "device" else str(value)
        raise TypeError("unsupported cache-off mathematical metadata: " + path)

    content = dict(payload=visit(payload, "payload"), step_index=state.step_index,
        spec={k: v for k, v in state.spec.items() if k != "request_id"},
        immutable_spec={k: v for k, v in state.immutable_spec.items() if k != "request_id"},
        cache_enabled=state.cache_enabled, needs_anchor=state.needs_anchor,
        current_step_refreshed=state.current_step_refreshed, finished=state.finished,
        container_aliases=container_aliases)
    return dict(schema_version=SIGNATURE_SCHEMA, step_index=state.step_index,
        state_sha256=digest(content), tensors=tensor_records,
        scheduler_sha256=digest(dict(values=content["payload"]["scheduler"], container_aliases=container_aliases)),
        container_aliases=container_aliases,
        sampling_metadata_sha256=digest({k: v for k, v in content.items() if k != "payload"}),
        scope="complete mathematical values, sampler metadata, aliases and request RNG; names/ownership/timing excluded")


def compare_step_tensors(actual, reference):
    """Scalar errors from the same captured step; does not mutate either set."""
    import torch
    from .cfg_adapter import _error
    required = {"cond", "uncond", "cfg", "latent_before", "latent_after"}
    if set(actual) != required or set(reference) != required:
        raise ValueError("complete diagnostic prediction and latent tensor sets required")
    for values in (actual, reference):
        if any(not torch.is_tensor(t) or t.device.type != "cpu" or not bool(torch.isfinite(t).all())
               for t in values.values()):
            raise ValueError("diagnostic comparison requires finite CPU tensors")
    return {name: _error(torch, actual[name], reference[name]) for name in sorted(required)}


class CacheOffDiagnostics:
    def __init__(self, worker, device_id, *, enabled=False):
        if enabled is not True:
            raise ValueError("cache-off diagnostics require explicit diagnostic-only mode")
        if worker.attention not in ("math", "flash", "efficient"):
            raise ValueError("diagnostics require an explicit supported native backend")
        self.worker, self.device_id = worker, device_id
        self.copies, self.used_ids, self.exports = {}, set(), set()

    def signature(self, request_id):
        return self.status(request_id)["signature"]

    def status(self, request_id):
        """Ownership evidence stays outside the comparable mathematical signature."""
        with self.worker._lock:
            state = self.worker._states[request_id]
            self.worker._check(state)
            if state.residency not in ("gpu", "diagnostic_paused"):
                raise ValueError("signature requires a complete live or diagnostically paused state")
            return dict(request_id=request_id, residency=state.residency,
                        signature=state_signature(self.worker, state),
                        device=self.device_id, backend=self.worker.attention)

    def export(self, request_id, snapshot_id):
        worker = self.worker
        if not isinstance(snapshot_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", snapshot_id):
            raise ValueError("a flat diagnostic snapshot ID is required")
        with worker._lock:
            if snapshot_id in self.exports:
                raise ValueError("diagnostic snapshot ID already used")
            state = worker._states[request_id]
            worker._check(state, "gpu")
            _check_off(worker, state)
            if state.step_index != 16 or state.finished:
                raise ValueError("diagnostic export requires exactly completed step 16")
            worker._sync()
            signature = state_signature(worker, state)
            manifest, tensors = pack_cache_off_request(worker, state)
            snapshot = dict(schema_version=SCHEMA, snapshot_id=snapshot_id, boundary_step=16,
                source_request_id=request_id, source_device=self.device_id, source_backend=worker.attention,
                state_signature=signature, wire_manifest=manifest, source_will_remain_paused=True,
                scope="independent diagnostic copies only; no production ownership or quality qualification")
            snapshot["sha256"] = digest(snapshot)
            state.residency = "diagnostic_paused"
            self.exports.add(snapshot_id)
            return dict(snapshot=snapshot, tensors=tensors, source_status="diagnostic_paused")

    def import_copy(self, snapshot, tensors, diagnostic_id):
        worker = self.worker
        if not isinstance(diagnostic_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", diagnostic_id):
            raise ValueError("a flat diagnostic copy ID is required")
        with worker._lock:
            if diagnostic_id in self.used_ids:
                raise ValueError("diagnostic copy ID already used")
            expected = {"schema_version", "snapshot_id", "boundary_step", "source_request_id", "source_device",
                        "source_backend", "state_signature", "wire_manifest", "source_will_remain_paused", "scope", "sha256"}
            if (set(snapshot) != expected or snapshot["schema_version"] != SCHEMA
                    or snapshot["sha256"] != digest({k: v for k, v in snapshot.items() if k != "sha256"})
                    or snapshot["boundary_step"] != 16 or snapshot["source_will_remain_paused"] is not True
                    or snapshot["wire_manifest"]["metadata"]["step_index"] != 16
                    or snapshot["source_request_id"] != snapshot["wire_manifest"]["metadata"]["request_id"]):
                raise ValueError("invalid step-16 diagnostic snapshot binding")
            if any(not worker.torch.is_tensor(t) or t.device.type != "cpu" for t in tensors.values()):
                raise ValueError("diagnostic import accepts CPU tensors only")
            # Independent storage is needed even for CPU test/import targets:
            # .to('cpu') alone can alias the caller's immutable snapshot tensors.
            copied = {name: tensor.detach().clone() for name, tensor in tensors.items()}
            state = unpack_cache_off_request(worker, copy.deepcopy(snapshot["wire_manifest"]), copied)
            observed = state_signature(worker, state)
            if observed != snapshot["state_signature"]:
                raise ValueError("diagnostic state differs before first target computation")
            request_id = "__diagnostic__" + diagnostic_id
            if request_id in worker._states:
                raise ValueError("diagnostic ID collides with a production request")
            state.request_id = request_id
            state.spec["request_id"] = request_id
            state.immutable_spec["request_id"] = request_id
            state.residency = "diagnostic_copy"
            self.copies[diagnostic_id] = dict(state=state, snapshot_sha256=snapshot["sha256"], status="ready",
                                              initial_signature=observed, backend=worker.attention)
            self.used_ids.add(diagnostic_id)
            return dict(diagnostic_id=diagnostic_id, state_signature=observed, byte_exact_before_first_compute=True,
                        production_owner=False, performance_eligible=False, backend=worker.attention,
                        target_device=self.device_id, snapshot_sha256=snapshot["sha256"])

    def step(self, diagnostic_id, *, return_tensors=True):
        from .cfg_adapter import CfgResearchAdapter
        worker, torch = self.worker, self.worker.torch
        if type(return_tensors) is not bool:
            raise ValueError("return_tensors must be boolean")
        with worker._lock, torch.inference_mode():
            row = self.copies[diagnostic_id]
            state = row["state"]
            if row["status"] != "ready" or state.step_index != 16 or state.residency != "diagnostic_copy":
                raise ValueError("diagnostic copy may execute exactly one 16->17 step")
            if state.request_id in worker._states:
                raise ValueError("diagnostic copy unexpectedly has a production owner")
            before = state_signature(worker, state)
            if before != row["initial_signature"] or worker.attention != row["backend"]:
                raise ValueError("diagnostic input state or backend changed after import")
            helper = CfgResearchAdapter(worker)
            rollback = helper._snapshot(state)
            original = worker._original_forward
            captures = []
            latent_before = state.latents.detach().cpu().clone()

            def capture(*args, **kwargs):
                if len(captures) >= 2:
                    raise RuntimeError("unexpected extra native forward in cache-off diagnostic step")
                result = original(*args, **kwargs)
                captures.append(result[0].detach().clone())
                return result

            try:
                # Registration exists only inside this locked native call. No
                # normal RPC can address or advance this copy before or after it.
                state.residency = "gpu"
                worker._states[state.request_id] = state
                worker._original_forward = capture
                record = worker.step(state)
                if len(captures) != 2 or state.step_index != 17 or record.get("reused_forwards") != 0:
                    raise RuntimeError("native diagnostic call did not execute exactly two off branches and one solver step")
                if not torch.equal(torch.get_rng_state(), rollback["cpu_rng"]):
                    raise RuntimeError("random operators are unsupported in numerical diagnostics")
                if rollback["cuda_rng"] is not None and not torch.equal(
                        torch.cuda.get_rng_state(worker.device), rollback["cuda_rng"]):
                    raise RuntimeError("CUDA random operators are unsupported in numerical diagnostics")
                cond, uncond = captures
                values = dict(cond=cond, uncond=uncond,
                    cfg=uncond + state.spec["guidance_scale"] * (cond - uncond),
                    latent_before=latent_before, latent_after=state.latents)
                tensors = {name: value.detach().cpu().clone() for name, value in values.items()}
                if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
                    raise RuntimeError("nonfinite diagnostic prediction or latent")
                result = dict(diagnostic_id=diagnostic_id, snapshot_sha256=row["snapshot_sha256"],
                    before=before, after=state_signature(worker, state), backend=worker.attention,
                    target_device=self.device_id,
                    tensor_summaries={name: dict(shape=list(t.shape), dtype=str(t.dtype),
                        sha256=_tensor_hash(torch, t), l2=float(torch.linalg.vector_norm(t.float()).item()),
                        max_abs=float(t.float().abs().max().item())) for name, t in tensors.items()},
                    native_step=copy.deepcopy(record), performance_eligible=False, production_owner=False,
                    scope="single isolated native step; tensors are transient numerical-diagnostic data")
                if return_tensors:
                    result["tensors"] = tensors
                row["status"] = "completed"
                return result
            except BaseException:
                helper._restore(state, rollback)
                row["status"] = "failed"
                raise
            finally:
                worker._original_forward = original
                worker._states.pop(state.request_id, None)
                state.residency = "diagnostic_copy"
                torch.set_rng_state(rollback["cpu_rng"])
                if rollback["cuda_rng"] is not None:
                    torch.cuda.set_rng_state(rollback["cuda_rng"], worker.device)

    def drop(self, diagnostic_id):
        with self.worker._lock:
            row = self.copies.pop(diagnostic_id)
            if row["state"].request_id in self.worker._states:
                raise RuntimeError("diagnostic copy unexpectedly remains registered")
            return dict(diagnostic_id=diagnostic_id, status="dropped", production_owner=False)
