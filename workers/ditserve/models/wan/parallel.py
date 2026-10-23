"""Aligned SP rank groups, the step-boundary layout guard and the request codec.

This module does not initialize CUDA, create process groups, or import torch at
import time. The wire codec moves one unfinished request between workers at a
solver-step boundary; block.py extends it to block boundaries inside a step.
It is an exact copy, not a claim that elastic GPU execution has been validated.
"""

import dataclasses
import copy
import datetime
import hashlib
import importlib
import json
import math


WORLD_SIZES = (2, 4, 8)


def aligned_groups(world_size):
    """Aligned power-of-two rank groups; ranks follow PCIe/socket topology order."""
    if world_size not in WORLD_SIZES:
        raise ParallelContractError("SP world size must be 2, 4 or 8")
    groups, degree = [], 1
    while degree <= world_size:
        groups.extend(tuple(range(start, start + degree)) for start in range(0, world_size, degree))
        degree *= 2
    return tuple(groups)

class ParallelContractError(RuntimeError):
    """Fail closed before ranks can follow different execution paths."""


def _ranks(value):
    value = tuple(value)
    if any(type(rank) is not int for rank in value) or value not in aligned_groups(max(WORLD_SIZES)):
        raise ParallelContractError("unsupported fixed rank group: %r" % (value,))
    return value


def _json(value):
    def validate(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) in (tuple, list):
            for child in item:
                validate(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                validate(child)
            return
        raise ParallelContractError("consensus metadata must be finite plain JSON")
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclasses.dataclass(frozen=True)
class FixedLayout:
    ranks: tuple
    epoch: int = 0

    def __post_init__(self):
        object.__setattr__(self, "ranks", _ranks(self.ranks))
        if type(self.epoch) is not int or self.epoch < 0:
            raise ParallelContractError("layout epoch must be a nonnegative integer")

    @property
    def degree(self):
        return len(self.ranks)


class LayoutHandle:
    """Local step boundary guard; the controller commits on every affected rank.

    This guard cannot replace a distributed handoff protocol. A controller must
    complete old-group work and state transport before committing a new layout.
    A live step is never silently ended on failure.
    """

    def __init__(self, rank, initial_ranks=None):
        if type(rank) is not int or not 0 <= rank < max(WORLD_SIZES):
            raise ParallelContractError("rank must be in [0, %d)" % max(WORLD_SIZES))
        self.rank = rank
        self.current = FixedLayout((rank,) if initial_ranks is None else initial_ranks)
        if rank not in self.current.ranks:
            raise ParallelContractError("local rank is not in its execution group")
        self.active = None
        self.failed = False

    def commit(self, ranks, *, expected_epoch):
        if self.active is not None or self.failed:
            raise ParallelContractError("layout can change only at a healthy step boundary")
        if expected_epoch != self.current.epoch:
            raise ParallelContractError("stale layout epoch")
        candidate = FixedLayout(ranks, self.current.epoch + 1)
        if self.rank not in candidate.ranks:
            raise ParallelContractError("local rank is not in the new group")
        self.current = candidate
        return candidate

    def begin_step(self, request_id, step_index, *, cfg="paired", expected_epoch=None):
        if self.active is not None or self.failed:
            raise ParallelContractError("a step is already active or the handle failed")
        if expected_epoch is not None and expected_epoch != self.current.epoch:
            raise ParallelContractError("stale step layout epoch")
        if not isinstance(request_id, str) or not request_id:
            raise ParallelContractError("request_id must be nonempty")
        if type(step_index) is not int or step_index < 0:
            raise ParallelContractError("step_index must be nonnegative")
        if cfg not in ("paired", "cond", "uncond", "none"):
            raise ParallelContractError("unsupported CFG phase")
        self.active = dict(request_id=request_id, step_index=step_index, cfg=cfg,
                           ranks=list(self.current.ranks), epoch=self.current.epoch)
        return dict(self.active)

    def end_step(self):
        if self.active is None or self.failed:
            raise ParallelContractError("no healthy active step")
        self.active = None

    def fail(self):
        self.failed = True


class FixedProcessGroups:
    def __init__(self, rank, groups, *, backend, timeout_seconds):
        self.rank, self.groups = rank, groups
        self.backend, self.timeout_seconds = backend, timeout_seconds

    def group(self, ranks):
        ranks = _ranks(ranks)
        if self.rank not in ranks or ranks not in self.groups:
            raise ParallelContractError("rank cannot use this process group")
        return self.groups[ranks]


def build_fixed_groups(*, timeout_seconds=15, backend=None):
    """Every rank must call once, in identical order, before execution."""
    torch = importlib.import_module("torch")
    dist = torch.distributed
    if not dist.is_initialized() or dist.get_world_size() not in WORLD_SIZES:
        raise ParallelContractError("SP requires an initialized 2-, 4- or 8-rank world")
    if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 60:
        raise ParallelContractError("group timeout must be in (0, 60] seconds")
    backend = backend or dist.get_backend()
    if backend not in ("gloo", "nccl"):
        raise ParallelContractError("only Gloo contracts and NCCL execution are supported")
    groups = {}
    for ranks in aligned_groups(dist.get_world_size()):
        groups[ranks] = dist.new_group(list(ranks), backend=backend,
                                      timeout=datetime.timedelta(seconds=timeout_seconds))
    return FixedProcessGroups(dist.get_rank(), groups, backend=backend,
                              timeout_seconds=timeout_seconds)


class RequestGroupGate:
    """Consensus and scalar reductions scoped to the handle's request group."""

    def __init__(self, handle, groups):
        if groups.rank != handle.rank:
            raise ParallelContractError("group registry belongs to another rank")
        self.handle, self.groups = handle, groups

    def consensus(self, payload):
        if self.handle.active is None or self.handle.failed:
            raise ParallelContractError("group operations require a healthy active step")
        try:
            envelope = dict(value=_json(dict(step=self.handle.active, payload=payload)))
        except (TypeError, ValueError, ParallelContractError, RecursionError) as error:
            envelope = dict(error=type(error).__name__ + ": " + str(error))
        layout = self.handle.current
        received = [envelope]
        try:
            if layout.degree > 1:
                dist = importlib.import_module("torch").distributed
                received = [None] * layout.degree
                dist.all_gather_object(received, envelope, group=self.groups.group(layout.ranks))
        except Exception as error:
            self.handle.fail()
            raise ParallelContractError("request-group consensus failed: " + str(error)) from error
        if any("error" in item for item in received) or any(item != received[0] for item in received):
            self.handle.fail()
            raise ParallelContractError("request-group metadata mismatch: " + _json(received))
        return payload

WIRE_SCHEMA = "ditserve.wan-request-wire.v2"
_SCHEDULER_VECTOR_FIELDS = ("betas", "alphas", "alphas_cumprod", "alpha_t", "sigma_t", "lambda_t", "sigmas", "timesteps")
# Explicit logical tensor names. Step-boundary request state is replicated on
# every SP rank, so none of these is ever sharded.
_TENSOR_NAMES = frozenset((
    "latents", "prompt_embeds", "negative_prompt_embeds", "rng_state",
    "solver.model_outputs.0", "solver.model_outputs.1", "solver.last_sample")) | frozenset(
    "solver." + name for name in _SCHEDULER_VECTOR_FIELDS)


@dataclasses.dataclass(frozen=True)
class KnownTensorSpec:
    name: str
    global_shape: tuple
    dtype: str

    def __post_init__(self):
        if self.name not in _TENSOR_NAMES:
            raise ParallelContractError("unknown logical tensor: " + self.name)
        object.__setattr__(self, "global_shape", tuple(self.global_shape))
        if not self.global_shape or any(type(n) is not int or n < 1 for n in self.global_shape):
            raise ParallelContractError("global tensor shape must contain positive integers")
        if self.dtype not in ("torch.float16", "torch.float32", "torch.int64", "torch.uint8"):
            raise ParallelContractError("unsupported tensor dtype")


def _specs(specs):
    if not isinstance(specs, (tuple, list)) or not specs:
        raise ParallelContractError("an explicit nonempty tensor schema is required")
    if any(type(spec) is not KnownTensorSpec for spec in specs):
        raise ParallelContractError("unsupported tensor schema entry")
    result = {spec.name: spec for spec in specs}
    if len(result) != len(specs):
        raise ParallelContractError("duplicate logical tensor names")
    return result


def _validate_tensors(state, specs):
    torch = importlib.import_module("torch")
    if type(state) is not dict or set(state) != set(specs):
        raise ParallelContractError("state fields do not exactly match the declared schema")
    storages = set()
    for name, spec in specs.items():
        tensor = state[name]
        if (not torch.is_tensor(tensor) or tensor.layout != torch.strided
                or not tensor.is_contiguous() or tuple(tensor.shape) != tuple(spec.global_shape)
                or str(tensor.dtype) != spec.dtype):
            raise ParallelContractError("tensor violates declared schema: " + name)
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in storages or tensor.storage_offset() != 0 or storage.nbytes() != tensor.numel() * tensor.element_size():
            raise ParallelContractError("aliased or oversized storage requires a separate explicit codec")
        storages.add(key)


def _tensor_digest(tensor):
    torch = importlib.import_module("torch")
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _validate_values(state, torch):
    shape = (1, 16, (state.spec["frames"] - 1) // 4 + 1, state.spec["height"] // 8, state.spec["width"] // 8)
    device = state.latents.device

    def check(value, dimensions, dtype, placement, optional=False):
        if optional and value is None:
            return
        if (not torch.is_tensor(value) or tuple(value.shape) != tuple(dimensions)
                or value.dtype != dtype or value.device != placement):
            raise ParallelContractError("tensor violates pinned Wan/UniPC shape, dtype or placement")

    check(state.latents, shape, torch.float32, device)
    for name in ("prompt_embeds", "negative_prompt_embeds"):
        value = getattr(state, name)
        check(value, (1, 512, 4096), torch.float16, device,
              optional=name == "negative_prompt_embeds" and state.spec["guidance_scale"] <= 1)
    scheduler = state.scheduler
    for name in _SCHEDULER_VECTOR_FIELDS[:-2]:
        check(getattr(scheduler, name), (1000,), torch.float32, torch.device("cpu"))
    check(scheduler.sigmas, (state.spec["steps"] + 1,), torch.float32, torch.device("cpu"))
    check(scheduler.timesteps, (state.spec["steps"],), torch.int64, device)
    if scheduler.num_inference_steps != state.spec["steps"] or scheduler._begin_index is not None:
        raise ParallelContractError("unsupported UniPC schedule or begin index")
    if scheduler._step_index != (state.step_index if state.step_index else None):
        raise ParallelContractError("UniPC and request step counters disagree")
    for value in list(scheduler.model_outputs) + [scheduler.last_sample]:
        check(value, shape, torch.float32, device, optional=True)
    rng = state.generator.get_state()
    check(rng, torch.Generator(device="cpu").get_state().shape, torch.uint8, torch.device("cpu"))


def pack_request(worker, state, *, allow_completed=False):
    """Explicit pinned Wan/UniPC wire schema, including known timestep aliases.

    Tensor payloads are CPU storage copies. Views and shared storage outside
    the known timestep history are rejected. Events are transported, but
    excluded from mathematical-state agreement because timings differ.
    """
    torch = importlib.import_module("torch")
    worker._check(state, "gpu")
    upper = state.spec["steps"] + int(bool(allow_completed))
    if state.finished or not 0 <= state.step_index < upper:
        raise ParallelContractError("transfer supports only an unfinished request at a step boundary")
    if type(state.scheduler).__name__ != "UniPCMultistepScheduler" or state.scheduler.config.solver_order != 2:
        raise ParallelContractError("migration requires the pinned two-step UniPC scheduler")
    if str(state.generator.device) != "cpu":
        raise ParallelContractError("migration requires the request's CPU RNG")
    _validate_values(state, torch)
    tensors, references, devices, identities, storages = {}, {}, {}, {}, set()

    def tensor(name, value):
        if value is None:
            references[name] = None
            return
        if name not in _TENSOR_NAMES:
            raise ParallelContractError("unsupported request tensor path: " + name)
        if id(value) in identities:
            references[name] = identities[id(value)]
            return
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr())
        if (not value.is_contiguous() or value.storage_offset() != 0 or value.is_conj() or value.is_neg()
                or storage.nbytes() != value.numel() * value.element_size()
                or value.device.type not in ("cpu", "cuda") or key in storages):
            raise ParallelContractError("unsupported or shared storage at " + name)
        storages.add(key)
        identities[id(value)] = name
        references[name] = name
        devices[name] = value.device.type
        tensors[name] = value.detach().cpu().clone().contiguous()

    for name in ("latents", "prompt_embeds", "negative_prompt_embeds"):
        tensor(name, getattr(state, name))
    tensor("rng_state", state.generator.get_state())
    scheduler = vars(state.scheduler)
    for name in _SCHEDULER_VECTOR_FIELDS:
        tensor("solver." + name, scheduler[name])
    tensor("solver.last_sample", scheduler["last_sample"])
    if type(scheduler["model_outputs"]) is not list or len(scheduler["model_outputs"]) != 2:
        raise ParallelContractError("unsupported UniPC model history")
    for index, value in enumerate(scheduler["model_outputs"]):
        tensor("solver.model_outputs.%d" % index, value)
    timestep_views = []
    timesteps = scheduler["timesteps"]
    if (timesteps.dtype != torch.int64 or timesteps.ndim != 1 or len(timesteps) != state.spec["steps"]
            or type(scheduler["timestep_list"]) is not list or len(scheduler["timestep_list"]) != 2):
        raise ParallelContractError("unsupported UniPC timestep schedule")
    for position, value in enumerate(scheduler["timestep_list"]):
        expected_offset = state.step_index - 2 + position
        if value is None:
            if expected_offset >= 0:
                raise ParallelContractError("missing UniPC timestep history")
            timestep_views.append(None)
        elif (torch.is_tensor(value) and value.dtype == torch.int64 and value.ndim == 0
              and value.device == timesteps.device
              and value.untyped_storage().data_ptr() == timesteps.untyped_storage().data_ptr()
              and value.storage_offset() == expected_offset and 0 <= expected_offset < len(timesteps)):
            timestep_views.append(value.storage_offset())
        else:
            raise ParallelContractError("timestep history must contain known scalar views of timesteps")
    tensor_fields = set(_SCHEDULER_VECTOR_FIELDS) | {"model_outputs", "last_sample", "timestep_list"}
    scheduler_metadata = {name: dict(value) if name == "_internal_dict" else value
                          for name, value in scheduler.items() if name not in tensor_fields}
    # Diffusers builds this one known bookkeeping field from a set. Its order
    # is process-dependent and has no sampler meaning; ordinary lists retain order.
    internal = scheduler_metadata["_internal_dict"]
    if "_use_default_values" in internal:
        internal["_use_default_values"] = sorted(internal["_use_default_values"])
    metadata = {name: getattr(state, name) for name in (
        "request_id", "spec", "step_index", "events", "finished", "prepare_ms")}
    specs = tuple(KnownTensorSpec(name, tuple(value.shape), str(value.dtype)) for name, value in sorted(tensors.items()))
    _validate_tensors(tensors, _specs(specs))
    manifest = json.loads(_json(dict(schema=WIRE_SCHEMA, metadata=metadata,
                                    scheduler=scheduler_metadata, references=references, devices=devices,
                                    timestep_views=timestep_views, specs=[dataclasses.asdict(s) for s in specs],
                                    tensor_hashes={name: _tensor_digest(value) for name, value in tensors.items()})))
    mathematical = copy.deepcopy(manifest)
    mathematical["metadata"].pop("events")
    mathematical["metadata"].pop("prepare_ms")
    manifest["state_sha256"] = hashlib.sha256(_json(mathematical).encode()).hexdigest()
    return manifest, tensors


def unpack_request(worker, manifest, tensors):
    """Build an unregistered destination request; the caller installs it."""
    torch = importlib.import_module("torch")
    from .runtime import RequestState, UNIPC_REQUIRED_FIELDS, UNIPC_OPTIONAL_FIELDS
    from diffusers.configuration_utils import FrozenDict
    expected = {"schema", "metadata", "scheduler", "references", "devices", "timestep_views", "specs", "tensor_hashes", "state_sha256"}
    if set(manifest) != expected or manifest["schema"] != WIRE_SCHEMA:
        raise ParallelContractError("unknown request wire schema")
    mathematical = copy.deepcopy(manifest)
    mathematical.pop("state_sha256")
    mathematical["metadata"].pop("events")
    mathematical["metadata"].pop("prepare_ms")
    if hashlib.sha256(_json(mathematical).encode()).hexdigest() != manifest["state_sha256"]:
        raise ParallelContractError("request manifest checksum mismatch")
    specs = _specs(tuple(KnownTensorSpec(**spec) for spec in manifest["specs"]))
    _validate_tensors(tensors, specs)
    values = {name: value.clone() for name, value in tensors.items()}
    if set(values) != set(manifest["devices"]) or set(values) != set(manifest["tensor_hashes"]):
        raise ParallelContractError("request tensor manifest mismatch")
    for name, value in values.items():
        if _tensor_digest(value) != manifest["tensor_hashes"][name]:
            raise ParallelContractError("request tensor checksum mismatch: " + name)
        placement = manifest["devices"][name]
        if placement not in ("cpu", "cuda"):
            raise ParallelContractError("unsupported source tensor placement")
        values[name] = value.to(worker.device if placement == "cuda" else "cpu")
        if _tensor_digest(values[name]) != manifest["tensor_hashes"][name]:
            raise ParallelContractError("reconstructed tensor checksum mismatch: " + name)
    references = manifest["references"]
    expected_references = {"latents", "prompt_embeds", "negative_prompt_embeds", "rng_state", "solver.last_sample",
                           "solver.model_outputs.0", "solver.model_outputs.1"} | {"solver." + name for name in _SCHEDULER_VECTOR_FIELDS}
    if set(references) != expected_references:
        raise ParallelContractError("unknown request logical tensor references")

    def get(name):
        reference = references[name]
        if reference is None:
            return None
        if reference not in values:
            raise ParallelContractError("unresolved request tensor alias")
        return values[reference]

    scheduler_fields = copy.deepcopy(manifest["scheduler"])
    config = scheduler_fields["_internal_dict"]
    scheduler = worker._scheduler_type.from_config(config)
    scheduler_fields["_internal_dict"] = FrozenDict(config)
    if scheduler_fields["disable_corrector"] != config["disable_corrector"]:
        raise ParallelContractError("UniPC corrector configuration differs from runtime metadata")
    scheduler_fields["disable_corrector"] = scheduler_fields["_internal_dict"]["disable_corrector"]
    scheduler_fields.update({name: get("solver." + name) for name in _SCHEDULER_VECTOR_FIELDS})
    scheduler_fields["last_sample"] = get("solver.last_sample")
    scheduler_fields["model_outputs"] = [get("solver.model_outputs.%d" % index) for index in range(2)]
    timesteps = scheduler_fields["timesteps"]
    views = manifest["timestep_views"]
    if len(views) != 2 or any(index is not None and (type(index) is not int or not 0 <= index < len(timesteps)) for index in views):
        raise ParallelContractError("invalid transported timestep views")
    scheduler_fields["timestep_list"] = [None if index is None else timesteps[index] for index in views]
    if (set(scheduler_fields) - UNIPC_REQUIRED_FIELDS - UNIPC_OPTIONAL_FIELDS
            or UNIPC_REQUIRED_FIELDS - set(scheduler_fields)):
        raise ParallelContractError("unknown transported scheduler fields")
    vars(scheduler).clear()
    vars(scheduler).update(scheduler_fields)
    fields = copy.deepcopy(manifest["metadata"])
    generator = torch.Generator(device="cpu")
    generator.set_state(get("rng_state"))
    state = RequestState(**fields, scheduler=scheduler, generator=generator, latents=get("latents"),
                         prompt_embeds=get("prompt_embeds"), negative_prompt_embeds=get("negative_prompt_embeds"),
                         residency="gpu", owner=worker._owner)
    worker._validate_request_schema(state)
    _validate_values(state, torch)
    if state.request_id != state.spec["request_id"]:
        raise ParallelContractError("invalid transported request identity")
    return state
