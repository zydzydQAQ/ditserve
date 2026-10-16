"""Read-only state descriptions and conservative, version-bound research plans.

Certificates are evidence for a narrowly supported control-flow condition, not
GPU trajectory qualification. Required tensor contents are hashed when issuing
and validating them; that inspection can copy CUDA data and must be timed.
"""

import dataclasses
import enum
import hashlib
import importlib
import math

from ..base import digest as _hash
from .runtime import (CALIBRATORS, CACHED_CONTEXT_FIELDS, DBCACHE_CONFIG_FIELDS,
                          EXTRA_CACHE_CONFIG_FIELDS, REQUEST_STATE_FIELDS,
                          UNIPC_OPTIONAL_FIELDS, UNIPC_REQUIRED_FIELDS,
                          _require_fields, normalize_spec)


STATE_SCHEMA = "ditserve.research.state.v1"
CERTIFICATE_SCHEMA = "ditserve.research.refresh-certificate.v1"
DEPENDENCY_VERSION = "1.5.0"
ACTION_NAMES = ("keep", "hot", "exact_refresh", "cold", "reshard")


def _torch(module):
    return module if module is not None else importlib.import_module("torch")


def _tensor_descriptor(tensor):
    storage = tensor.untyped_storage()
    return dict(device=str(tensor.device), dtype=str(tensor.dtype),
                shape=list(tensor.shape), stride=list(tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                storage_pointer=int(storage.data_ptr()), storage_bytes=int(storage.nbytes()),
                logical_bytes=int(tensor.numel() * tensor.element_size()))


def _validate_state(state):
    _require_fields(state, REQUEST_STATE_FIELDS, label="research RequestState")
    spec = normalize_spec(state.spec)
    if spec != state.spec:
        raise ValueError("research state requires a normalized spec")
    if state.request_id != spec["request_id"]:
        raise ValueError("request identity differs from normalized spec")
    immutable = {k: v for k, v in spec.items() if k not in ("cache", "threshold", "max_continuous")}
    if state.immutable_spec != immutable:
        raise ValueError("immutable request identity changed")
    _require_fields(state.scheduler, UNIPC_REQUIRED_FIELDS, UNIPC_OPTIONAL_FIELDS,
                    label="research UniPC")
    if state.scheduler.solver_p is not None:
        raise ValueError("external solver state is unsupported")
    if not isinstance(state.contexts, dict):
        raise TypeError("contexts must be a dictionary")
    for context in state.contexts.values():
        _require_fields(context, CACHED_CONTEXT_FIELDS, label="research CachedContext")
        _require_fields(context.cache_config, DBCACHE_CONFIG_FIELDS, label="research DBCacheConfig")
        _require_fields(context.extra_cache_config, EXTRA_CACHE_CONFIG_FIELDS, label="research ExtraCacheConfig")


def _roots(state):
    yield "latents", "sampler", state.latents
    yield "scheduler", "sampler", vars(state.scheduler)
    yield "prompt_embeds", "conditioning", state.prompt_embeds
    yield "negative_prompt_embeds", "conditioning", state.negative_prompt_embeds
    for name, context in sorted(state.contexts.items()):
        yield "contexts." + name + ".buffers", "cache_feature", context.buffers
        yield "contexts." + name + ".metadata", "cache_metadata", {
            k: v for k, v in vars(context).items() if k != "buffers"}


def _walk(value, path, role, torch, visit, active=None):
    if torch.is_tensor(value):
        if value.layout != torch.strided:
            raise TypeError("only strided tensors are supported at " + path)
        visit(value, path, role)
        return
    if value is None or isinstance(value, (str, bool, int, float, enum.Enum)):
        return
    if type(value).__module__ == "torch" and type(value).__name__ in ("device", "dtype", "layout", "memory_format"):
        return
    active = set() if active is None else active
    if id(value) in active:
        raise TypeError("cyclic state containers are unsupported at " + path)
    container_id = id(value)
    active.add(container_id)
    try:
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise TypeError("state keys must be strings at " + path)
            children = sorted(value.items())
        elif isinstance(value, (tuple, list)):
            children = enumerate(value)
        elif dataclasses.is_dataclass(value):
            children = sorted(vars(value).items())
        else:
            raise TypeError("unsupported state object at " + path + ": " + type(value).__name__)
        for key, child in children:
            _walk(child, path + "." + str(key), role, torch, visit, active)
    finally:
        active.remove(container_id)


def state_inventory(state, *, torch_module=None):
    """Count owned storage once, retaining *all* tensor paths and role aliases.

    No live tensors escape in the report. RNG serialization bytes are separate:
    get_state() creates a new export tensor, not request-owned resident storage.
    Feature-only bytes are an ownership upper bound, not an allocator guarantee.
    """
    torch = _torch(torch_module)
    _validate_state(state)
    storages, tensors = {}, {}

    def visit(tensor, path, role):
        desc = _tensor_descriptor(tensor)
        key = (desc["device"], desc["storage_pointer"], desc["storage_bytes"])
        # Zero-sized storage has no reclaimable bytes and often a shared null pointer.
        if not desc["storage_bytes"]:
            key += (id(tensor),)
        if key not in storages:
            storages[key] = dict(storage_id="s%d" % len(storages), device=desc["device"],
                                 pointer=desc["storage_pointer"], bytes=desc["storage_bytes"],
                                 paths=[], roles=[], tensor_ids=[])
        storage = storages[key]
        tid = id(tensor)
        if tid not in tensors:
            tensors[tid] = dict(tensor_id="t%d" % len(tensors), storage_id=storage["storage_id"],
                                paths=[], roles=[], **desc)
        record = tensors[tid]
        for item in (storage, record):
            if path not in item["paths"]:
                item["paths"].append(path)
            if role not in item["roles"]:
                item["roles"].append(role)
        if record["tensor_id"] not in storage["tensor_ids"]:
            storage["tensor_ids"].append(record["tensor_id"])

    for path, role, value in _roots(state):
        _walk(value, path, role, torch, visit)
    rows = list(storages.values())
    tensor_rows = {row["tensor_id"]: row for row in tensors.values()}
    for row in rows:
        row["roles"].sort()
        row["paths"].sort()
        row["feature_only"] = row["roles"] == ["cache_feature"]
        row["distinct_view_alias"] = len(row["tensor_ids"]) > 1
        # UniPC stores scalar views of its full timestep tensor in its two-step
        # history. This known alias is common in real requests and is handled
        # by the research storage codec; arbitrary feature/sampler views remain
        # unsupported. All views must be int64 scalars of one contiguous vector.
        descriptions = [tensor_rows[key] for key in row["tensor_ids"]]
        paths = row["paths"]
        row["supported_timestep_views"] = bool(row["distinct_view_alias"]
            and "scheduler.timesteps" in paths
            and all(path == "scheduler.timesteps" or path in ("scheduler.timestep_list.0", "scheduler.timestep_list.1") for path in paths)
            and all(item["dtype"] == "torch.int64" for item in descriptions)
            and all((item["shape"] == [] and item["stride"] == []) or
                    (len(item["shape"]) == 1 and item["stride"] == [1] and item["storage_offset"] == 0)
                    for item in descriptions))
    feature = sum(row["bytes"] for row in rows if row["feature_only"])
    total = sum(row["bytes"] for row in rows)
    gpu = sum(row["bytes"] for row in rows if row["device"].startswith("cuda:"))
    rng = state.generator.get_state()
    return dict(schema_version=STATE_SCHEMA, request_id=state.request_id, step_index=state.step_index,
                storages=rows, tensors=list(tensors.values()), storage_bytes=total,
                unique_storage_bytes=total, unique_storage_count=len(rows), cuda_bytes=gpu,
                feature_only_bytes=feature, necessary_storage_bytes=total - feature,
                feature_only_cuda_bytes=sum(r["bytes"] for r in rows if r["feature_only"] and r["device"].startswith("cuda:")),
                shared_role_bytes=sum(r["bytes"] for r in rows if len(r["roles"]) > 1),
                logical_tensor_bytes=sum(r["logical_bytes"] for r in tensors.values()),
                rng_snapshot_bytes=int(rng.numel() * rng.element_size()),
                transfer_supported=not any(r["distinct_view_alias"] and not r["supported_timestep_views"] for r in rows),
                storage_scope="request-owned tensor storage; RNG export listed separately; excludes weights/workspace",
                release_scope="feature-only ownership upper bound; actual allocation reuse must be measured")


def assert_transfer_supported(inventory):
    if inventory.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported inventory schema")
    if any(row.get("distinct_view_alias") and not row.get("supported_timestep_views") for row in inventory["storages"]):
        raise ValueError("distinct tensor views share storage; the pinned worker transfer does not preserve these aliases")


def _layout(state, layout):
    result = dict(layout or dict(sp_size=1, ranks=[0], device=str(state.latents.device), version=1))
    if set(result) != {"sp_size", "ranks", "device", "version"}:
        raise ValueError("unsupported layout fields")
    if (type(result["sp_size"]) is not int or result["sp_size"] < 1 or
            type(result["version"]) is not int or result["version"] < 1 or
            not isinstance(result["device"], str) or not isinstance(result["ranks"], (list, tuple)) or
            len(result["ranks"]) != result["sp_size"] or
            any(type(r) is not int or r < 0 for r in result["ranks"]) or
            len(set(result["ranks"])) != len(result["ranks"])):
        raise ValueError("invalid placement layout")
    result["ranks"] = list(result["ranks"])
    return result


def _freeze(value, torch, *, hash_tensors=True, active=None):
    if torch.is_tensor(value):
        record = dict(object_id=id(value), **_tensor_descriptor(value))
        if hash_tensors:
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            record["content_sha256"] = hashlib.sha256(raw).hexdigest()
        return record
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite state metadata")
        return value
    if isinstance(value, enum.Enum):
        return dict(enum=type(value).__name__, value=value.value)
    if type(value).__module__ == "torch" and type(value).__name__ in ("device", "dtype", "layout", "memory_format"):
        return str(value)
    active = set() if active is None else active
    if id(value) in active:
        raise TypeError("cyclic state fingerprint is unsupported")
    container_id = id(value)
    active.add(container_id)
    try:
        if dataclasses.is_dataclass(value):
            value = vars(value)
        if isinstance(value, dict):
            if any(type(k) is not str for k in value):
                raise TypeError("fingerprint keys must be strings")
            return {k: _freeze(v, torch, hash_tensors=hash_tensors, active=active) for k, v in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [_freeze(v, torch, hash_tensors=hash_tensors, active=active) for v in value]
        raise TypeError("unsupported fingerprint object: " + type(value).__name__)
    finally:
        active.remove(container_id)


def _binding(state, torch, layout, dependency_version):
    contexts = {}
    for name, context in state.contexts.items():
        contexts[name] = {k: _freeze(v, torch, hash_tensors=(k != "buffers"))
                          for k, v in vars(context).items()}
    rng = state.generator.get_state().cpu().numpy().tobytes()
    return dict(state_object=id(state), owner_object=id(state.owner), request_id=state.request_id,
                spec=state.spec, immutable_spec=state.immutable_spec, step_index=state.step_index,
                cache_enabled=state.cache_enabled, residency=state.residency,
                needs_anchor=state.needs_anchor, finished=state.finished,
                current_step_refreshed=state.current_step_refreshed,
                pending_snapshot=_freeze(state.pending_snapshot, torch),
                scheduler=_freeze(vars(state.scheduler), torch), latents=_freeze(state.latents, torch),
                prompt_embeds=_freeze(state.prompt_embeds, torch),
                negative_prompt_embeds=_freeze(state.negative_prompt_embeds, torch),
                rng_object=id(state.generator), rng_device=str(state.generator.device),
                rng_sha256=hashlib.sha256(rng).hexdigest(), contexts=contexts,
                layout=layout, dependency_version=dependency_version)


def refresh_certificate(state, *, layout=None, dependency_version=DEPENDENCY_VERSION, torch_module=None):
    """Certify only the next natural m=1 refresh of *both* CFG branches.

    Never calls can_cache(): that function mutates counters. Certificates do not
    establish output parity and remain validation-only even when eligible.
    """
    torch = _torch(torch_module)
    _validate_state(state)
    placement = _layout(state, layout)
    reasons = []
    if dependency_version != DEPENDENCY_VERSION:
        reasons.append("requires_cache_dit_1.5.0")
    if placement["sp_size"] != 1 or placement["device"] != str(state.latents.device):
        reasons.append("only_current_single_device_layout_supported")
    if (state.residency != "gpu" or state.pending_snapshot is not None or state.finished or
            not 0 < state.step_index < state.spec["steps"] or state.needs_anchor):
        reasons.append("not_a_live_unmodified_step_boundary")
    if state.scheduler._step_index != state.step_index or "this_order" not in vars(state.scheduler):
        reasons.append("solver_boundary_identity_mismatch")
    if not state.cache_enabled or not state.spec["cache"] or state.spec["max_continuous"] != 1 or state.spec["guidance_scale"] <= 1:
        reasons.append("requires_cached_separate_cfg_m1")
    if len(state.contexts) != 1:
        reasons.append("requires_one_Wan_context")
    for name, context in state.contexts.items():
        cfg = context.cache_config
        pinned = dict(Fn_compute_blocks=8, Bn_compute_blocks=0, max_warmup_steps=8,
                      warmup_interval=1, max_cached_steps=-1, max_continuous_cached_steps=1,
                      enable_separate_cfg=True, cfg_compute_first=False, cfg_diff_compute_separate=True,
                      steps_computation_mask=None, steps_computation_policy="dynamic",
                      force_refresh_step_hint=None, force_refresh_step_policy="once",
                      max_accumulated_residual_diff_threshold=None, num_inference_steps=state.spec["steps"])
        if (str(cfg.cache_type) != "DBCache" or any(getattr(cfg, k) != v for k, v in pinned.items()) or
                cfg.residual_diff_threshold != state.spec["threshold"] or state.spec["threshold"] <= 0):
            reasons.append(name + ":unsupported_cache_recipe")
        if (any(getattr(context, k) is not None for k in CALIBRATORS) or context.calibrator_config is not None or
                vars(context.extra_cache_config) != dict(l1_hidden_states_diff_threshold=None,
                    important_condition_threshold=0.0, downsample_factor=1)):
            reasons.append(name + ":unsupported_extra_cache_state")
        if (context.executed_steps != state.step_index or context.accumulated_executed_steps != state.step_index or
                context.transformer_executed_steps != 2 * state.step_index or
                context.accumulated_transformer_executed_steps != 2 * state.step_index):
            reasons.append(name + ":counter_identity_mismatch")
        counters = (context.continuous_cached_steps, context.cfg_continuous_cached_steps)
        if (state.step_index < 8 or any(type(count) is not int or count < 1 for count in counters) or
                not context.cached_steps or not context.cfg_cached_steps or
                context.cached_steps[-1] != state.step_index - 1 or context.cfg_cached_steps[-1] != state.step_index - 1):
            reasons.append(name + ":both_branches_not_natural_refresh")
        if not context.buffers:
            reasons.append(name + ":no_features_to_reclaim")
        suffixes = ("_Fn_residual_buffer", "_Fn_residual_buffer_cfg", "_Bn_residual_buffer", "_Bn_residual_buffer_cfg",
                    "_Bn_residual_encoder_buffer", "_Bn_residual_encoder_buffer_cfg")
        if any(not isinstance(k, str) or not k.endswith(suffixes) or not torch.is_tensor(v)
               for k, v in context.buffers.items()):
            reasons.append(name + ":unsupported_feature_buffer")
    inventory = state_inventory(state, torch_module=torch)
    if not inventory["feature_only_bytes"]:
        reasons.append("no_exclusively_owned_feature_storage")
    certificate = dict(schema_version=CERTIFICATE_SCHEMA, eligible=not reasons, reasons=reasons,
                       request_id=state.request_id, step_index=state.step_index, layout=placement,
                       dependency_version=dependency_version,
                       binding_sha256=_hash(_binding(state, torch, placement, dependency_version)),
                       reclaimable_storage_ids=[r["storage_id"] for r in inventory["storages"] if r["feature_only"]],
                       reclaimable_bytes=inventory["feature_only_bytes"],
                       reclaimable_cuda_bytes=inventory["feature_only_cuda_bytes"],
                       proof_scope="pinned m1 CFG counter guard precedes old Fn/Bn reads; metadata preserved",
                       qualification="validation_only_requires_GPU_trajectory_evidence")
    certificate["sha256"] = _hash(certificate)
    return certificate


def validate_refresh_certificate(state, certificate, *, layout=None, dependency_version=DEPENDENCY_VERSION, torch_module=None):
    if not isinstance(certificate, dict):
        raise ValueError("exact_refresh requires a refresh certificate")
    current = refresh_certificate(state, layout=layout, dependency_version=dependency_version, torch_module=torch_module)
    if not current["eligible"]:
        raise ValueError("ineligible exact_refresh boundary: " + ",".join(current["reasons"]))
    if certificate != current:
        raise ValueError("stale or modified refresh certificate/state/layout")
    return current


def transition_plan(state, action, *, certificate=None, layout=None, target_layout=None,
                    dependency_version=DEPENDENCY_VERSION, torch_module=None):
    if action not in ACTION_NAMES:
        raise ValueError("unknown state action")
    current = _layout(state, layout)
    target = current if target_layout is None else _layout(state, target_layout)
    if action == "reshard" or target != current or current["sp_size"] != 1:
        raise NotImplementedError("SP resharding is not implemented or qualified in the local pilot")
    inventory = state_inventory(state, torch_module=torch_module)
    if state.residency != "gpu" or state.pending_snapshot is not None or state.finished:
        raise ValueError("transition requires a live GPU-resident step boundary")
    if action in ("hot", "cold"):
        assert_transfer_supported(inventory)
    checked = None
    if action == "exact_refresh":
        checked = validate_refresh_certificate(state, certificate, layout=current,
                                               dependency_version=dependency_version, torch_module=torch_module)
    return dict(schema_version="ditserve.research.transition.v1", action=action,
                request_id=state.request_id, step_index=state.step_index, layout=current, target_layout=target,
                trajectory_semantics=("approximate_cold_anchor" if action == "cold" else "preserve_cached_trajectory_candidate"),
                implementation_status="validation_only" if action == "exact_refresh" else "existing_worker",
                reclaimable_bytes=(checked["reclaimable_bytes"] if checked else inventory["cuda_bytes"] if action in ("hot", "cold") else 0),
                transfer_estimate_bytes=(inventory["cuda_bytes"] if action == "hot" else
                    sum(r["bytes"] for r in inventory["storages"] if r["device"].startswith("cuda:") and not r["feature_only"]) if action == "cold" else 0),
                actual_transfer_bytes=None, actual_reusable_bytes=None, estimated_cost_ms=None,
                qualification="not_GPU_qualified", certificate_sha256=checked["sha256"] if checked else None)
