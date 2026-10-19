"""Pinned SP1 wire state and two-phase ownership for heterogeneous migration.

The codec never mutates the source.  ``MigrationEndpoint`` is the RPC boundary:
pause/export -> prepare -> acknowledge -> release_source -> commit.  A prepared
request is deliberately absent from the worker registry.  Before source release,
abort restores the original request; after release, any failure poisons the unit.
No CUDA initialization occurs when importing this module.
"""

import copy
import hashlib
import importlib
import json
import re
import time
from collections import defaultdict

from ..base import digest as _hash
from .parallel import ParallelContractError, pack_cache_off_request, unpack_cache_off_request
from .state import assert_transfer_supported, state_inventory, validate_refresh_certificate
from .runtime import (CACHED_CONTEXT_FIELDS, CALIBRATORS, DBCACHE_CONFIG_FIELDS,
                          EXTRA_CACHE_CONFIG_FIELDS)


SCHEMA = "ditserve.dual-sp1-wire.v2"
ACTIONS = ("full_migrate", "cold_migrate", "certified_omit")
BUFFER_SUFFIXES = ("_Fn_residual_buffer", "_Fn_residual_buffer_cfg", "_Bn_residual_buffer",
                   "_Bn_residual_buffer_cfg", "_Bn_residual_encoder_buffer",
                   "_Bn_residual_encoder_buffer_cfg")
CONTEXT_SPECIAL = frozenset(("buffers", "cache_config", "extra_cache_config", "_init_args", "_init_kwargs"))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tensor_hash(value):
    torch = importlib.import_module("torch")
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _raw_storage(value):
    torch = importlib.import_module("torch")
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))


def _storage_hash(value):
    return hashlib.sha256(_raw_storage(value).cpu().numpy().tobytes()).hexdigest()


def _feature_layout(key, value):
    """Admit full contiguous storage or Wan's one observed Fn transpose.

    Wan 0.36 patch embeddings use flatten(2).transpose(1, 2); cache-dit 1.5
    subtracts the Fn residual without making it contiguous. Archived run05
    inventories show [1,S,1536] / [S*1536,1,S], offset zero, exact storage size.
    No other strided, sliced, overlapping, conjugate or padded view is admitted.
    Distinct tensors sharing storage are rejected separately by the inventory.
    """
    torch = importlib.import_module("torch")
    if not torch.is_tensor(value) or value.layout != torch.strided:
        raise ParallelContractError("unsupported cached feature tensor layout")
    known_fn = (key.endswith(("_Fn_residual_buffer", "_Fn_residual_buffer_cfg"))
                and value.dtype == torch.float16 and value.ndim == 3
                and value.shape[0] == 1 and value.shape[2] == 1536 and value.shape[1] > 0
                and value._base is None
                and tuple(value.stride()) == (value.shape[1] * 1536, 1, value.shape[1]))
    if (not (value.is_contiguous() or known_fn) or value.storage_offset() != 0
            or value.is_conj() or value.is_neg()
            or value.untyped_storage().nbytes() != value.numel() * value.element_size()
            or value.device.type not in ("cpu", "cuda")):
        raise ParallelContractError("unsupported cached feature storage/view at %s: shape=%s stride=%s offset=%s bytes=%s logical=%s" % (
            key, tuple(value.shape), tuple(value.stride()), value.storage_offset(),
            value.untyped_storage().nbytes(), value.numel() * value.element_size()))


def _copy_feature_storage(value, device):
    torch = importlib.import_module("torch")
    raw = _raw_storage(value).to(device=device, copy=True)
    return torch.empty(0, dtype=value.dtype, device=device).set_(
        raw.untyped_storage(), 0, tuple(value.shape), tuple(value.stride()))


def _context_name_mapping(manifest, target_names):
    source_names, target_names = set(manifest["contexts"]), set(target_names)
    if not source_names and not target_names:
        return {}
    if len(source_names) != 1 or len(target_names) != 1:
        raise ParallelContractError("context translation requires exactly one Wan context")
    source, target = next(iter(source_names)), next(iter(target_names))
    if source != target and not (re.fullmatch(r"blocks_[0-9]+", source) and re.fullmatch(r"blocks_[0-9]+", target)):
        raise ParallelContractError("only process-local Wan blocks object-ID names may be translated")
    return {source: target}


def _plain(value):
    """JSON metadata only; reject opaque objects, tensors and nonfinite values."""
    if value is None or type(value) in (str, int, float, bool):
        _json(value)
        return value
    if type(value) is list:
        return [_plain(item) for item in value]
    if type(value) in (dict, defaultdict):
        if any(type(key) is not str for key in value):
            raise ParallelContractError("context metadata keys must be strings")
        if type(value) is defaultdict and value.default_factory is not float:
            raise ParallelContractError("only residual defaultdict(float) is supported")
        return {key: _plain(item) for key, item in value.items()}
    raise ParallelContractError("unsupported context metadata type: " + type(value).__name__)


class _BaseView:
    def __init__(self, worker, source, shadow):
        self.worker, self.source, self.shadow = worker, source, shadow

    @property
    def pipe(self):   # the latent shape check reads the model's channels and VAE factors
        return getattr(self.worker, "pipe", None)

    def _check(self, state, residency):
        if state is not self.shadow:
            raise ParallelContractError("invalid base view")
        self.worker._check(self.source, residency)


def _base_pack(worker, state):
    shadow = copy.copy(state)
    shadow.spec = dict(state.spec, cache=False)
    shadow.contexts, shadow.cache_enabled = {}, False
    shadow.needs_anchor = shadow.current_step_refreshed = False
    return pack_cache_off_request(_BaseView(worker, state, shadow), shadow)


def _context_pack(context, name, tensors, identities, feature_records, *, include_features):
    if context.name != name or context._init_args not in ((), []):
        raise ParallelContractError("unsupported context name or positional initialization")
    kwargs = context._init_kwargs
    if (set(kwargs) != {"name", "cache_config"} or kwargs["name"] != name
            or kwargs["cache_config"] is not context.cache_config):
        raise ParallelContractError("context restart config must alias its live config")
    config = dict(vars(context.cache_config))
    if str(config["cache_type"]) != "DBCache":
        raise ParallelContractError("only pinned DBCache contexts are supported")
    config["cache_type"] = "DBCache"
    metadata = _plain({key: value for key, value in vars(context).items() if key not in CONTEXT_SPECIAL})
    if any(metadata[key] is not None for key in CALIBRATORS + ("calibrator_config",)):
        raise ParallelContractError("calibrators cannot cross the wire")
    defaults = [key for key in ("residual_diffs", "cfg_residual_diffs")
                if type(getattr(context, key)) is defaultdict]
    refs = {}
    for key, value in sorted(context.buffers.items()):
        torch = importlib.import_module("torch")
        if type(key) is not str or not key.endswith(BUFFER_SUFFIXES) or not torch.is_tensor(value):
            raise ParallelContractError("unknown cached feature buffer")
        _feature_layout(key, value)
        identity = id(value)
        if identity not in identities:
            reference = "cache/%d" % len(identities)
            identities[identity] = reference
            feature_records[reference] = dict(shape=list(value.shape), stride=list(value.stride()),
                                              storage_offset=0, storage_sha256=_storage_hash(value), dtype=str(value.dtype),
                                              device=value.device.type, sha256=_tensor_hash(value),
                                              bytes=value.numel() * value.element_size())
            if include_features:
                tensors[reference] = _copy_feature_storage(value, "cpu")
        refs[key] = identities[identity]
    return dict(metadata=metadata, cache_config=_plain(config),
                extra_cache_config=_plain(vars(context.extra_cache_config)),
                init_args_type=type(context._init_args).__name__, residual_defaultdicts=defaults,
                buffer_references=refs)


def pack_request(worker, state, *, action, target_device, migration_id, epoch=0, certificate=None):
    """Export CPU tensors at a locked complete step boundary, without mutation.

    ``target_device`` is the physical destination UUID, not a process-local CUDA
    ordinal. The caller must hold the worker lock throughout this operation.
    """
    if action not in ACTIONS or not target_device or not migration_id or type(epoch) is not int or epoch < 0:
        raise ParallelContractError("invalid migration action, destination, ID or epoch")
    worker._check(state, "gpu")
    if state.pending_snapshot is not None or state.finished or not 0 < state.step_index < state.spec["steps"]:
        raise ParallelContractError("migration requires an unfinished complete step boundary")
    if state.cache_enabled != state.spec["cache"] or bool(state.contexts) != state.cache_enabled:
        raise ParallelContractError("cache flag and owned contexts disagree")
    if state.cache_enabled and (len(state.contexts) != 1 or state.spec["guidance_scale"] <= 1):
        raise ParallelContractError("cached migration requires one separate-CFG Wan context")
    inventory = state_inventory(state)
    assert_transfer_supported(inventory)
    if any("cache_feature" in row["roles"] and len(row["roles"]) != 1 for row in inventory["storages"]):
        raise ParallelContractError("feature storage aliases non-feature state")
    checked = None
    if action == "certified_omit":
        checked = validate_refresh_certificate(state, certificate)
    elif certificate is not None:
        raise ParallelContractError("certificate supplied for a non-omission action")
    worker._sync()
    base, tensors = _base_pack(worker, state)
    identities, features, contexts = {}, {}, {}
    for name, context in sorted(state.contexts.items()):
        contexts[name] = _context_pack(context, name, tensors, identities, features,
                                       include_features=action == "full_migrate")
    cache_state = dict(spec=copy.deepcopy(state.spec), cache_enabled=state.cache_enabled,
                       needs_anchor=state.needs_anchor, current_step_refreshed=state.current_step_refreshed)
    source_payload = dict(base_sha256=base["state_sha256"], cache_state=cache_state,
                          contexts=contexts, features=features)
    source_digest = _hash(source_payload)
    proof = None if checked is None else dict(certificate=checked, source_state_sha256=source_digest,
                                              target_device=target_device, migration_id=migration_id, epoch=epoch)
    manifest = dict(schema=SCHEMA, action=action, target_device=target_device, migration_id=migration_id,
                    epoch=epoch, request_id=state.request_id, step_index=state.step_index,
                    base=base, cache_state=cache_state, contexts=contexts, features=features,
                    source_state_sha256=source_digest, certificate_binding=proof,
                    transfer_bytes=sum(t.numel() * t.element_size() for t in tensors.values()),
                    omitted_feature_bytes=0 if action == "full_migrate" else sum(f["bytes"] for f in features.values()))
    manifest["sha256"] = _hash(manifest)
    return manifest, tensors


def _verify_manifest(manifest, *, target_device, migration_id, epoch):
    expected = {"schema", "action", "target_device", "migration_id", "epoch", "request_id", "step_index",
                "base", "cache_state", "contexts", "features", "source_state_sha256", "certificate_binding",
                "transfer_bytes", "omitted_feature_bytes", "sha256"}
    if set(manifest) != expected or manifest["schema"] != SCHEMA or manifest["action"] not in ACTIONS:
        raise ParallelContractError("unknown dual-state manifest")
    if _hash({key: value for key, value in manifest.items() if key != "sha256"}) != manifest["sha256"]:
        raise ParallelContractError("dual-state manifest checksum mismatch")
    if (manifest["target_device"], manifest["migration_id"], manifest["epoch"]) != (target_device, migration_id, epoch):
        raise ParallelContractError("migration destination, ID or epoch mismatch")
    source_payload = dict(base_sha256=manifest["base"]["state_sha256"], cache_state=manifest["cache_state"],
                          contexts=manifest["contexts"], features=manifest["features"])
    digest = _hash(source_payload)
    if digest != manifest["source_state_sha256"]:
        raise ParallelContractError("source checkpoint checksum mismatch")
    proof = manifest["certificate_binding"]
    if manifest["action"] == "certified_omit":
        if not isinstance(proof, dict) or set(proof) != {"certificate", "source_state_sha256", "target_device", "migration_id", "epoch"}:
            raise ParallelContractError("missing source omission proof")
        if any(proof[key] != manifest[key] for key in ("source_state_sha256", "target_device", "migration_id", "epoch")):
            raise ParallelContractError("source omission proof binding mismatch")
        cert = proof["certificate"]
        if (not cert.get("eligible") or cert.get("request_id") != manifest["request_id"]
                or cert.get("step_index") != manifest["step_index"]
                or _hash({key: value for key, value in cert.items() if key != "sha256"}) != cert.get("sha256")):
            raise ParallelContractError("invalid source omission certificate")
    elif proof is not None:
        raise ParallelContractError("unexpected omission proof")


def unpack_request(worker, manifest, tensors, *, target_device, migration_id, epoch=0):
    """Validate each reconstructed tensor before returning an unregistered state."""
    _verify_manifest(manifest, target_device=target_device, migration_id=migration_id, epoch=epoch)
    base_names = set(manifest["base"]["tensor_hashes"])
    torch = importlib.import_module("torch")
    if any(not torch.is_tensor(value) or value.device.type != "cpu" for value in tensors.values()):
        raise ParallelContractError("migration wire payload must contain CPU tensors only")
    include_features = manifest["action"] == "full_migrate"
    feature_names = set(manifest["features"]) if include_features else set()
    if set(tensors) != base_names | feature_names or base_names & set(manifest["features"]):
        raise ParallelContractError("wire tensor set mismatch")
    if manifest["request_id"] in worker._states:
        raise ParallelContractError("destination already owns request")
    state = unpack_cache_off_request(worker, manifest["base"], {name: tensors[name] for name in base_names})
    if state.request_id != manifest["request_id"] or state.step_index != manifest["step_index"]:
        raise ParallelContractError("base request identity mismatch")
    cache_state = manifest["cache_state"]
    if set(cache_state) != {"spec", "cache_enabled", "needs_anchor", "current_step_refreshed"}:
        raise ParallelContractError("invalid cached request flags")
    for name, value in cache_state.items():
        setattr(state, name, copy.deepcopy(value))
    if state.cache_enabled != state.spec["cache"] or bool(manifest["contexts"]) != state.cache_enabled:
        raise ParallelContractError("transported cache flags disagree")
    features = {}
    for name in sorted(feature_names):
        value, record = tensors[name], manifest["features"][name]
        torch = importlib.import_module("torch")
        if (set(record) != {"shape", "stride", "storage_offset", "storage_sha256", "dtype", "device", "sha256", "bytes"}
                or not torch.is_tensor(value) or value.device.type != "cpu" or list(value.shape) != record["shape"]
                or list(value.stride()) != record["stride"] or value.storage_offset() != record["storage_offset"]
                or value.untyped_storage().nbytes() != record["bytes"]
                or str(value.dtype) != record["dtype"] or _tensor_hash(value) != record["sha256"]
                or _storage_hash(value) != record["storage_sha256"]
                or record["device"] not in ("cpu", "cuda")):
            raise ParallelContractError("cached tensor checksum/schema mismatch")
        keys = [key for context in manifest["contexts"].values()
                for key, reference in context["buffer_references"].items() if reference == name]
        if not keys:
            raise ParallelContractError("cached tensor has no declared feature reference")
        for key in keys:
            _feature_layout(key, value)
        restored = _copy_feature_storage(value, worker.device if record["device"] == "cuda" else "cpu")
        if (_tensor_hash(restored) != record["sha256"] or _storage_hash(restored) != record["storage_sha256"]
                or list(restored.stride()) != record["stride"]):
            raise ParallelContractError("reconstructed cached tensor checksum mismatch")
        features[name] = restored
    state.contexts = worker._contexts(state.spec) if state.cache_enabled else {}
    context_mapping = _context_name_mapping(manifest, state.contexts)
    for source_name, name in context_mapping.items():
        context = state.contexts[name]
        record = manifest["contexts"][source_name]
        if (set(record) != {"metadata", "cache_config", "extra_cache_config", "init_args_type",
                           "residual_defaultdicts", "buffer_references"}
                or set(record["metadata"]) != CACHED_CONTEXT_FIELDS - CONTEXT_SPECIAL
                or set(record["cache_config"]) != DBCACHE_CONFIG_FIELDS
                or set(record["extra_cache_config"]) != EXTRA_CACHE_CONFIG_FIELDS
                or record["init_args_type"] not in ("list", "tuple")
                or not set(record["residual_defaultdicts"]) <= {"residual_diffs", "cfg_residual_diffs"}):
            raise ParallelContractError("unknown transported cache context schema")
        if record["metadata"]["name"] != source_name:
            raise ParallelContractError("source context metadata name differs from wire binding")
        for field, value in record["metadata"].items():
            setattr(context, field, copy.deepcopy(_plain(value)))
        context.name = name
        cache_config = copy.deepcopy(record["cache_config"])
        cache_config["cache_type"] = type(context.cache_config.cache_type)(cache_config["cache_type"])
        vars(context.cache_config).clear()
        vars(context.cache_config).update(cache_config)
        vars(context.extra_cache_config).clear()
        vars(context.extra_cache_config).update(copy.deepcopy(record["extra_cache_config"]))
        context._init_args = () if record["init_args_type"] == "tuple" else []
        context._init_kwargs = dict(name=name, cache_config=context.cache_config)
        for field in record["residual_defaultdicts"]:
            setattr(context, field, defaultdict(float, getattr(context, field)))
        refs = record["buffer_references"]
        if any(not key.endswith(BUFFER_SUFFIXES) or ref not in manifest["features"] for key, ref in refs.items()):
            raise ParallelContractError("unknown transported feature reference")
        context.buffers = {key: features[reference] for key, reference in refs.items()} if include_features else {}
        worker._validate_context(context)
    if manifest["action"] == "cold_migrate" and state.cache_enabled:
        state.needs_anchor = True
    worker._validate_request_schema(state)
    worker._sync()
    return state


class MigrationEndpoint:
    """Process-local endpoint; all GPU execution uses the underlying worker lock.

    A paused source remains registered but its residency prevents any worker step
    or decode. A target candidate remains unregistered until a release receipt.
    Coordinator messages carry a manifest hash and monotonic ownership epoch.
    """
    def __init__(self, worker, device_id):
        self.worker, self.device_id = worker, device_id
        self._sources, self._targets, self._closed, self._epochs = {}, {}, set(), {}

    def _fresh(self, migration_id):
        if migration_id in self._closed or migration_id in self._sources or migration_id in self._targets:
            raise ParallelContractError("migration ID already used")

    def pause_export(self, state, *, action, target_device, migration_id, epoch=0, certificate=None):
        with self.worker._lock:
            self._fresh(migration_id)
            if self._epochs.get(state.request_id, 0) != epoch or target_device == self.device_id:
                raise ParallelContractError("stale source ownership epoch or identical destination")
            started = time.perf_counter()
            manifest, tensors = pack_request(self.worker, state, action=action, target_device=target_device,
                                             migration_id=migration_id, epoch=epoch, certificate=certificate)
            state.residency = "migration_paused"
            self._sources[migration_id] = dict(state=state, manifest=manifest, phase="paused")
            return manifest, tensors, dict(export_ms=(time.perf_counter() - started) * 1000)

    def prepare(self, manifest, tensors):
        migration_id = manifest["migration_id"]
        with self.worker._lock:
            self._fresh(migration_id)
            if any(row["state"].request_id == manifest["request_id"] for row in self._targets.values()):
                raise ParallelContractError("another prepared target owns this request ID")
            if self._epochs.get(manifest["request_id"], 0) > manifest["epoch"]:
                raise ParallelContractError("stale destination epoch")
            state = unpack_request(self.worker, manifest, tensors, target_device=self.device_id,
                                   migration_id=migration_id, epoch=manifest["epoch"])
            mapping = _context_name_mapping(manifest, state.contexts)
            self._targets[migration_id] = dict(state=state, manifest=manifest, phase="prepared", context_name_mapping=mapping)
            return self._token(manifest, "prepared", mapping)

    @staticmethod
    def _token(manifest, phase, context_name_mapping):
        return dict(migration_id=manifest["migration_id"], request_id=manifest["request_id"],
                    manifest_sha256=manifest["sha256"], target_device=manifest["target_device"],
                    epoch=manifest["epoch"] + 1, phase=phase, context_name_mapping=dict(context_name_mapping))

    def acknowledge(self, migration_id, acknowledgement):
        with self.worker._lock:
            row = self._sources[migration_id]
            mapping = acknowledgement.get("context_name_mapping", {})
            if (type(mapping) is not dict or set(mapping) != set(row["manifest"]["contexts"])
                    or mapping != _context_name_mapping(row["manifest"], mapping.values())
                    or row["phase"] != "paused" or acknowledgement != self._token(row["manifest"], "prepared", mapping)):
                raise ParallelContractError("invalid target acknowledgement")
            row["context_name_mapping"] = dict(mapping)
            row["phase"] = "acknowledged"

    def release_source(self, migration_id):
        with self.worker._lock:
            row = self._sources[migration_id]
            if row["phase"] != "acknowledged":
                raise ParallelContractError("source release requires a validated target acknowledgement")
            state = row["state"]
            if state.residency != "migration_paused":
                raise ParallelContractError("source changed while paused")
            state.residency = "gpu"
            try:
                self.worker.release(state)
            except BaseException:
                # Unknown release failures cannot authorize either copy to run.
                state.residency = "migration_failed"
                row["phase"] = "failed"
                raise
            row["phase"] = "released"
            self._epochs[state.request_id] = row["manifest"]["epoch"] + 1
            return self._token(row["manifest"], "released", row["context_name_mapping"])

    def commit(self, migration_id, release_receipt):
        with self.worker._lock:
            row = self._targets[migration_id]
            if row["phase"] != "prepared" or release_receipt != self._token(row["manifest"], "released", row["context_name_mapping"]):
                raise ParallelContractError("target commit requires source release receipt")
            state = row["state"]
            if state.request_id in self.worker._states:
                raise ParallelContractError("target request acquired another owner")
            self.worker._states[state.request_id] = state
            self._epochs[state.request_id] = release_receipt["epoch"]
            self._closed.add(migration_id)
            del self._targets[migration_id]
            return state

    def abort(self, migration_id):
        with self.worker._lock:
            source = self._sources.get(migration_id)
            if source is not None:
                if source["phase"] not in ("paused", "acknowledged"):
                    raise ParallelContractError("cannot resume a released or failed source")
                source["state"].residency = "gpu"
                del self._sources[migration_id]
            target = self._targets.pop(migration_id, None)
            if source is None and target is None:
                raise ParallelContractError("unknown or already closed migration")
            self._closed.add(migration_id)


def migrate_request(source, target, state, *, action, migration_id, epoch=0, certificate=None):
    """In-process reference coordinator; RPC callers use the same endpoint phases."""
    started = time.perf_counter()
    manifest, tensors, costs = source.pause_export(state, action=action, target_device=target.device_id,
                                                   migration_id=migration_id, epoch=epoch, certificate=certificate)
    prepared, released = False, False
    try:
        phase = time.perf_counter()
        ack = target.prepare(manifest, tensors)
        prepared = True
        costs["target_prepare_ms"] = (time.perf_counter() - phase) * 1000
        source.acknowledge(migration_id, ack)
        phase = time.perf_counter()
        receipt = source.release_source(migration_id)
        released = True
        costs["release_ms"] = (time.perf_counter() - phase) * 1000
        result = target.commit(migration_id, receipt)
    except BaseException:
        if prepared:
            target.abort(migration_id)
        if not released and source._sources[migration_id]["phase"] in ("paused", "acknowledged"):
            source.abort(migration_id)
        raise
    costs["transaction_ms"] = (time.perf_counter() - started) * 1000
    costs["transfer_bytes"] = manifest["transfer_bytes"]
    return result, manifest, costs


class _UnregisteredTargetView:
    """Expose the same-device constructors without its live source registry."""
    def __init__(self, worker):
        self._worker, self._states = worker, {}

    def __getattr__(self, name):
        return getattr(self._worker, name)


def local_roundtrip(worker, state, *, action, migration_id, certificate=None):
    """Same-device correctness control using the exact heterogeneous wire codec.

    A temporary registry view allows constructing a duplicate candidate, which
    cannot execute. The real registry owns only the source until reconstruction
    and byte checks succeed, then only the replacement after source release.
    """
    with worker._lock:
        started = time.perf_counter()
        manifest, tensors = pack_request(worker, state, action=action, target_device=str(worker.device),
                                         migration_id=migration_id, certificate=certificate)
        candidate = unpack_request(_UnregisteredTargetView(worker), manifest, tensors,
                                   target_device=str(worker.device), migration_id=migration_id)
        # Every possible allocation/validation failure above leaves source live.
        # Once release starts, a failure poisons the unit instead of guessing
        # whether partially released source storage remains usable.
        try:
            worker.release(state)
        except BaseException:
            state.residency = "migration_failed"
            raise
        worker._states[candidate.request_id] = candidate
        return candidate, dict(action=action, migration_id=migration_id,
                               source_state_sha256=manifest["source_state_sha256"],
                               manifest_sha256=manifest["sha256"], transfer_bytes=manifest["transfer_bytes"],
                               omitted_feature_bytes=manifest["omitted_feature_bytes"],
                               roundtrip_ms=(time.perf_counter() - started) * 1000,
                               byte_exact_before_first_compute=True)
