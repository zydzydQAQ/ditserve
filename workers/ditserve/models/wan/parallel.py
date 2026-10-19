"""Bounded parallel-state contracts for the four-rank research harness.

This module does not initialize CUDA, create process groups, or import torch at
import time. State reshaping here is an exact, in-memory codec, not a distributed
transport or an assertion that elastic GPU execution has been validated.
"""

import dataclasses
import copy
import datetime
import hashlib
import importlib
import importlib.metadata
import json
import math
import time
import types


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

LATENT_CACHE_BUFFER_NAMES = frozenset(
    "blocks_%s_residual_buffer%s" % (part, suffix)
    for part in ("Fn", "Bn") for suffix in ("", "_cfg"))
ENCODER_CACHE_BUFFER_NAMES = frozenset(
    "blocks_Bn_residual_encoder_buffer" + suffix for suffix in ("", "_cfg"))
CACHE_BUFFER_NAMES = LATENT_CACHE_BUFFER_NAMES | ENCODER_CACHE_BUFFER_NAMES
_COUNTERS = (
    "executed_steps", "transformer_executed_steps", "accumulated_executed_steps",
    "accumulated_transformer_executed_steps", "accumulated_cached_steps",
    "cfg_accumulated_cached_steps", "continuous_cached_steps", "cfg_continuous_cached_steps",
    "accumulated_residual_diff", "cfg_accumulated_residual_diff")


class ParallelContractError(RuntimeError):
    """Fail closed before ranks can follow different execution paths."""


# The original four-rank research layout; larger worlds use aligned_groups(n).
FIXED_GROUPS = aligned_groups(4)


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
    """An aligned rank group and how its ranks split the work.

    cfg=1: every rank runs both CFG branches over its token shard (Ulysses SP on
    the whole group). cfg=2: the first half of the group runs the conditional
    branch and the second half the unconditional one, each half an aligned SP
    subgroup; the halves exchange their predictions once per solver step.
    """

    ranks: tuple
    epoch: int = 0
    cfg: int = 1

    def __post_init__(self):
        object.__setattr__(self, "ranks", _ranks(self.ranks))
        if type(self.epoch) is not int or self.epoch < 0:
            raise ParallelContractError("layout epoch must be a nonnegative integer")
        if self.cfg not in (1, 2):
            raise ParallelContractError("cfg must be 1 or 2")
        if self.cfg == 2 and len(self.ranks) < 2:
            raise ParallelContractError("CFG parallelism needs an aligned group of at least two ranks")

    @property
    def degree(self):
        return len(self.ranks)

    @property
    def sp_degree(self):
        """Ranks that share one token sequence."""
        return len(self.ranks) // self.cfg

    def cfg_role(self, rank):
        """0 for the conditional half, 1 for the unconditional half; None without CFG parallelism."""
        if self.cfg == 1:
            return None
        return self.ranks.index(rank) // self.sp_degree

    def sp_ranks(self, rank):
        """The aligned SP subgroup this rank shards tokens with."""
        if self.cfg == 1:
            return self.ranks
        role = self.cfg_role(rank)
        return self.ranks[role * self.sp_degree:(role + 1) * self.sp_degree]


class LayoutHandle:
    """Local step boundary guard; the controller commits on every affected rank.

    This guard cannot replace a distributed handoff protocol. A controller must
    complete old-group work and state transport before committing a new layout.
    A live step is never silently ended on failure.
    """

    def __init__(self, rank, initial_ranks=None, cfg=1):
        if type(rank) is not int or not 0 <= rank < max(WORLD_SIZES):
            raise ParallelContractError("rank must be in [0, %d)" % max(WORLD_SIZES))
        self.rank = rank
        self.current = FixedLayout((rank,) if initial_ranks is None else initial_ranks, cfg=cfg)
        if rank not in self.current.ranks:
            raise ParallelContractError("local rank is not in its execution group")
        self.active = None
        self.paired = None   # a second request advancing in lockstep with the active one
        self.failed = False

    def commit(self, ranks, *, expected_epoch, cfg=1):
        if self.active is not None or self.failed:
            raise ParallelContractError("layout can change only at a healthy step boundary")
        if expected_epoch != self.current.epoch:
            raise ParallelContractError("stale layout epoch")
        candidate = FixedLayout(ranks, self.current.epoch + 1, cfg)
        if self.rank not in candidate.ranks:
            raise ParallelContractError("local rank is not in the new group")
        self.current = candidate
        return candidate

    def adopt(self, ranks, *, epoch, cfg=1):
        """Take a controller placement at a step boundary.

        A cluster worker serves requests on different groups, so epochs are
        not local counters: the controller gives every rank of a placement the
        same epoch, and group consensus then also checks that all ranks act on
        one placement.
        """
        if self.active is not None or self.failed:
            raise ParallelContractError("layout can change only at a healthy step boundary")
        candidate = FixedLayout(ranks, epoch, cfg)
        if self.rank not in candidate.ranks:
            raise ParallelContractError("local rank is not in the new group")
        self.current = candidate
        return candidate

    def begin_step(self, request_id, step_index, *, cfg="paired", expected_epoch=None, pair=False):
        """Open a request's step; with ``pair`` a second request joins the active one in lockstep."""
        if self.failed or (self.active is not None and not (pair and self.paired is None)):
            raise ParallelContractError("a step is already active or the handle failed")
        if expected_epoch is not None and expected_epoch != self.current.epoch:
            raise ParallelContractError("stale step layout epoch")
        if not isinstance(request_id, str) or not request_id:
            raise ParallelContractError("request_id must be nonempty")
        if type(step_index) is not int or step_index < 0:
            raise ParallelContractError("step_index must be nonnegative")
        if cfg not in ("paired", "cond", "uncond", "none"):
            raise ParallelContractError("unsupported CFG phase")
        entry = dict(request_id=request_id, step_index=step_index, cfg=cfg,
                     ranks=list(self.current.ranks), epoch=self.current.epoch)
        if self.active is None:
            self.active = entry
        else:
            if request_id == self.active["request_id"]:
                raise ParallelContractError("request already has an active step")
            self.paired = entry
        return dict(entry)

    def end_step(self, request_id=None):
        """Close the active step, or the named one when two requests run in lockstep."""
        if self.active is None or self.failed:
            raise ParallelContractError("no healthy active step")
        if self.paired is not None and request_id == self.paired["request_id"]:
            self.paired = None
            return
        if request_id is not None and request_id != self.active["request_id"]:
            raise ParallelContractError("no active step for " + request_id)
        self.active, self.paired = self.paired, None

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
        raise ParallelContractError("the research harness requires an initialized 2-, 4- or 8-rank world")
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

    def pre_step(self, cache_manifest):
        """All members call even when their local cache is absent or invalid."""
        return self.consensus(dict(kind="pre_step", cache=cache_manifest))

    def mean_pair(self, numerator, denominator):
        """Preserve pinned equal-shard mean reduction, never use WORLD."""
        torch = importlib.import_module("torch")
        description = {}
        try:
            for name, tensor in (("numerator", numerator), ("denominator", denominator)):
                if not torch.is_tensor(tensor) or tensor.numel() != 1:
                    raise ParallelContractError("cache reductions require scalar tensors")
                if not tensor.is_floating_point():
                    raise ParallelContractError("cache reductions require floating point")
                if not bool(torch.isfinite(tensor).all()):
                    raise ParallelContractError("nonfinite cache scalar")
                description[name] = dict(dtype=str(tensor.dtype), device=tensor.device.type)
        except Exception as error:
            description = dict(invalid=type(error).__name__ + ": " + str(error))
        self.consensus(dict(kind="cache_scalar_schema", schema=description))
        if "invalid" in description:
            self.handle.fail()
            raise ParallelContractError(description["invalid"])
        output = (numerator.clone(), denominator.clone())
        layout = self.handle.current
        if layout.degree > 1:
            try:
                for scalar in output:
                    # Preserve the pinned NCCL AVG operation. Gloo lacks AVG;
                    # its SUM/division path is only the CPU contract harness.
                    operation = (torch.distributed.ReduceOp.AVG if self.groups.backend == "nccl"
                                 else torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(scalar, op=operation,
                                                 group=self.groups.group(layout.ranks))
                    if self.groups.backend != "nccl":
                        scalar.div_(layout.degree)
            except Exception as error:
                self.handle.fail()
                raise ParallelContractError("request-group reduction failed: " + str(error)) from error
        return output


def cache_manager_manifest(manager):
    """Inventory the explicitly supported Wan residual-cache schema.

    Errors are returned as metadata so every rank still reaches consensus.
    They are subsequently rejected uniformly by the bound gate wrapper.
    """
    try:
        torch = importlib.import_module("torch")
        context = manager.get_context()
        if context is None:
            raise ParallelContractError("cache context is absent")
        if set(context.buffers) - CACHE_BUFFER_NAMES:
            raise ParallelContractError("unsupported Wan cache buffer names")
        if manager.get_important_condition_threshold() != 0 or manager.get_downsample_factor() != 1:
            raise ParallelContractError("importance filtering and downsampling are unsupported")
        if context.extra_cache_config.l1_hidden_states_diff_threshold is not None:
            raise ParallelContractError("hidden-state L1 cache mode is outside the residual schema")
        if any(getattr(context, key, None) is not None for key in
               ("calibrator", "encoder_calibrator", "cfg_calibrator", "cfg_encoder_calibrator")):
            raise ParallelContractError("calibrator state is unsupported")
        if context.cache_config.steps_computation_mask is not None:
            raise ParallelContractError("static cache masks are outside this adapter")
        buffers = {}
        for name in sorted(CACHE_BUFFER_NAMES):
            tensor = context.buffers.get(name)
            if tensor is None:
                buffers[name] = None
                continue
            if not torch.is_tensor(tensor) or tensor.ndim != 3 or not tensor.is_contiguous():
                raise ParallelContractError("cache buffer must be contiguous [batch, sequence, hidden]")
            buffers[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                 device=tensor.device.type)
        result = dict(buffers=buffers, counters={key: getattr(context, key) for key in _COUNTERS},
                      cfg=bool(manager.is_separate_cfg_step()),
                      cache_config=vars(context.cache_config),
                      cached_steps=list(context.cached_steps), cfg_cached_steps=list(context.cfg_cached_steps),
                      residual_diffs=dict(context.residual_diffs), cfg_residual_diffs=dict(context.cfg_residual_diffs))
        # CacheType is a str enum in the pinned dependency. Normalize only it.
        result["cache_config"] = dict(result["cache_config"])
        result["cache_config"]["cache_type"] = str(result["cache_config"]["cache_type"])
        _json(result)
        return result
    except Exception as error:
        return dict(invalid=type(error).__name__ + ": " + str(error))


class SimilarityBinding:
    def __init__(self, manager, previous):
        self.manager, self.previous, self.closed = manager, previous, False

    def close(self):
        if self.closed:
            return
        for name, value in self.previous.items():
            if value is _MISSING:
                self.manager.__dict__.pop(name, None)
            else:
                setattr(self.manager, name, value)
        self.closed = True


_MISSING = object()


def bind_request_group_similarity(manager, gate):
    """Bind pinned cache-dit 1.5 logic to one manager instance only.

    The can_cache wrapper performs consensus before upstream early returns.
    Equal sharding, scalar thresholds, no calibrators, and residual caching are
    the deliberately narrow supported recipe. No distributed global is patched.
    """
    if importlib.metadata.version("cache-dit") != "1.5.0":
        raise ParallelContractError("similarity adapter requires cache-dit 1.5.0")
    cls = importlib.import_module("cache_dit.caching.cache_contexts.cache_manager").CachedContextManager
    if type(manager) is not cls:
        raise ParallelContractError("unsupported cache manager implementation")
    if "similarity" in vars(manager) or "can_cache" in vars(manager):
        raise ParallelContractError("manager methods are already overridden")
    previous = {name: vars(manager).get(name, _MISSING) for name in ("similarity", "can_cache")}
    original_can_cache = manager.can_cache

    def check_manifest():
        manifest = cache_manager_manifest(manager)
        gate.consensus(dict(kind="cache_manifest", manifest=manifest))
        if "invalid" in manifest:
            gate.handle.fail()
            raise ParallelContractError(manifest["invalid"])

    def can_cache(instance, states_tensor, parallelized=False, threshold=None, prefix="Fn"):
        check_manifest()
        try:
            description = dict(shape=list(states_tensor.shape), dtype=str(states_tensor.dtype),
                               device=states_tensor.device.type, threshold=threshold, prefix=prefix)
        except Exception as error:
            description = dict(invalid=str(error))
        gate.consensus(dict(kind="can_cache_input", input=description))
        if "invalid" in description:
            gate.handle.fail()
            raise ParallelContractError(description["invalid"])
        result = original_can_cache(states_tensor, parallelized=gate.handle.current.degree > 1,
                                    threshold=threshold, prefix=prefix)
        gate.consensus(dict(kind="cache_decision", decision=bool(result)))
        return result

    def similarity(instance, t1, t2, *, threshold, parallelized=False, prefix="Fn"):
        # Arithmetic/order follows cache-dit 1.5.0 similarity, with equal-shard
        # request-group reduction replacing its default-WORLD reduction.
        check_manifest()
        gate.consensus(dict(kind="similarity_input", previous_shape=list(t1.shape),
                            current_shape=list(t2.shape), threshold=threshold, prefix=prefix))
        if threshold <= 0.0:
            instance.add_residual_diff(-0.0)
            return False
        if threshold >= 1.0:
            instance.add_residual_diff(-1.0)
            return True
        if t1.shape != t2.shape:
            instance.add_residual_diff(-2.0)
            return False
        if (instance.enable_separate_cfg() and instance.is_separate_cfg_step()
                and not instance.cfg_diff_compute_separate()
                and instance.get_current_step_residual_diff() is not None):
            diff = instance.get_current_step_residual_diff()
        else:
            mean_diff, mean_t1 = gate.mean_pair((t1 - t2).abs().mean(), t1.abs().mean())
            diff = (mean_diff / mean_t1).item()
            if not math.isfinite(diff):
                gate.handle.fail()
                raise ParallelContractError("nonfinite reduced cache residual")
        instance.add_residual_diff(diff)
        return diff < threshold

    manager.similarity = types.MethodType(similarity, manager)
    manager.can_cache = types.MethodType(can_cache, manager)
    return SimilarityBinding(manager, previous)


# Explicit logical tensor names and token axes; no axis is inferred from shape.
_TENSOR_AXES = {name: 1 for name in LATENT_CACHE_BUFFER_NAMES}
# Pattern_2 retains the full text context on every SP rank. Its Bn encoder
# residual is [batch, text_tokens, hidden], not the sharded video sequence.
_TENSOR_AXES.update({name: None for name in ENCODER_CACHE_BUFFER_NAMES})
_TENSOR_AXES.update({name: None for name in (
    "latents", "prompt_embeds", "negative_prompt_embeds", "rng_state",
    "solver.model_outputs.0", "solver.model_outputs.1", "solver.last_sample")})
_SCHEDULER_VECTOR_FIELDS = ("betas", "alphas", "alphas_cumprod", "alpha_t", "sigma_t", "lambda_t", "sigmas", "timesteps")
_TENSOR_AXES.update({"solver." + name: None for name in _SCHEDULER_VECTOR_FIELDS})


@dataclasses.dataclass(frozen=True)
class KnownTensorSpec:
    name: str
    global_shape: tuple
    dtype: str

    def __post_init__(self):
        if self.name not in _TENSOR_AXES:
            raise ParallelContractError("unknown logical tensor: " + self.name)
        object.__setattr__(self, "global_shape", tuple(self.global_shape))
        if not self.global_shape or any(type(n) is not int or n < 1 for n in self.global_shape):
            raise ParallelContractError("global tensor shape must contain positive integers")
        if self.dtype not in ("torch.float16", "torch.float32", "torch.int64", "torch.uint8"):
            raise ParallelContractError("unsupported tensor dtype")
        if self.name in CACHE_BUFFER_NAMES and len(self.global_shape) != 3:
            raise ParallelContractError("Wan residual cache requires [batch, sequence, hidden]")

    @property
    def axis(self):
        return _TENSOR_AXES[self.name]


def _specs(specs):
    if not isinstance(specs, (tuple, list)) or not specs:
        raise ParallelContractError("an explicit nonempty tensor schema is required")
    if any(type(spec) is not KnownTensorSpec for spec in specs):
        raise ParallelContractError("unsupported tensor schema entry")
    result = {spec.name: spec for spec in specs}
    if len(result) != len(specs):
        raise ParallelContractError("duplicate logical tensor names")
    return result


def _validate_tensors(state, specs, degree=1):
    torch = importlib.import_module("torch")
    if type(state) is not dict or set(state) != set(specs):
        raise ParallelContractError("state fields do not exactly match the declared schema")
    storages = set()
    for name, spec in specs.items():
        tensor = state[name]
        shape = list(spec.global_shape)
        if spec.axis is not None:
            if shape[spec.axis] % degree:
                raise ParallelContractError("sequence length is not divisible by SP degree")
            shape[spec.axis] //= degree
        if (not torch.is_tensor(tensor) or tensor.layout != torch.strided
                or not tensor.is_contiguous() or tuple(tensor.shape) != tuple(shape)
                or str(tensor.dtype) != spec.dtype):
            raise ParallelContractError("tensor violates declared schema: " + name)
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key in storages or tensor.storage_offset() != 0 or storage.nbytes() != tensor.numel() * tensor.element_size():
            raise ParallelContractError("aliased or oversized storage requires a separate explicit codec")
        storages.add(key)


def _bitwise_equal(left, right):
    torch = importlib.import_module("torch")
    return (left.device == right.device and torch.equal(left.reshape(-1).view(torch.uint8),
                                                       right.reshape(-1).view(torch.uint8)))


def split_known_state(state, specs, ranks):
    """Split explicit feature fields; replicate latent/solver/RNG fields exactly."""
    ranks, specs = _ranks(ranks), _specs(specs)
    _validate_tensors(state, specs)
    output = {rank: {} for rank in ranks}
    for name, spec in specs.items():
        if spec.axis is not None and spec.global_shape[spec.axis] % len(ranks):
            raise ParallelContractError("sequence length is not divisible by SP degree")
        parts = state[name].chunk(len(ranks), dim=spec.axis) if spec.axis is not None else [state[name]] * len(ranks)
        for rank, part in zip(ranks, parts):
            output[rank][name] = part.clone().contiguous()
    return output


def merge_known_state(shards, specs, ranks):
    """Reference reconstruction; transport must move tensors to a common device."""
    torch = importlib.import_module("torch")
    ranks, specs = _ranks(ranks), _specs(specs)
    if type(shards) is not dict or set(shards) != set(ranks):
        raise ParallelContractError("missing or additional rank shards")
    for rank in ranks:
        _validate_tensors(shards[rank], specs, len(ranks))
    output = {}
    for name, spec in specs.items():
        parts = [shards[rank][name] for rank in ranks]
        if any(part.device != parts[0].device for part in parts):
            raise ParallelContractError("reference merge requires a common device")
        if spec.axis is None:
            if any(not _bitwise_equal(parts[0], part) for part in parts[1:]):
                raise ParallelContractError("replicated tensor differs between ranks: " + name)
            output[name] = parts[0].clone()
        else:
            output[name] = torch.cat(parts, dim=spec.axis)
    return output


def reshard_known_state(shards, specs, source_ranks, target_ranks):
    """Exact in-memory reference codec; deliberately not a network benchmark."""
    return split_known_state(merge_known_state(shards, specs, source_ranks), specs, target_ranks)


def _tensor_digest(tensor):
    torch = importlib.import_module("torch")
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def wan_latent_shape(worker, spec):
    """(1, C, F, H, W): transformer input channels and the VAE compression.

    Wan2.1 1.3B is 16 channels at 4x temporal / 8x spatial; Wan2.2 TI2V-5B 48 at 4x/16x.
    Worker doubles without a pipeline or model config default to Wan2.1 1.3B.
    """
    pipe = getattr(worker, "pipe", None)
    channels = getattr(getattr(getattr(pipe, "transformer", None), "config", None), "in_channels", 16)
    temporal = getattr(pipe, "vae_scale_factor_temporal", 4)
    spatial = getattr(pipe, "vae_scale_factor_spatial", 8)
    return (1, channels, (spec["frames"] - 1) // temporal + 1, spec["height"] // spatial, spec["width"] // spatial)


def _validate_cache_off_values(state, torch, shape):
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


def pack_cache_off_request(worker, state):
    """Explicit pinned Wan/UniPC wire schema, including known timestep aliases.

    Tensor payloads are CPU storage copies. Cache features and arbitrary views
    are rejected. Events are transported, but excluded from mathematical-state
    agreement because timings differ between ranks.
    """
    torch = importlib.import_module("torch")
    from .state import state_inventory, assert_transfer_supported
    worker._check(state, "gpu")
    if (state.cache_enabled or state.spec["cache"] or state.contexts or state.pending_snapshot is not None
            or state.finished or state.needs_anchor or state.current_step_refreshed
            or not 0 <= state.step_index < state.spec["steps"]):
        raise ParallelContractError("migration supports only an unfinished cache-off request at a clean boundary")
    if type(state.scheduler).__name__ != "UniPCMultistepScheduler" or state.scheduler.config.solver_order != 2:
        raise ParallelContractError("migration requires the pinned two-step UniPC scheduler")
    if str(state.generator.device) != "cpu":
        raise ParallelContractError("migration requires the request's CPU RNG")
    _validate_cache_off_values(state, torch, wan_latent_shape(worker, state.spec))
    assert_transfer_supported(state_inventory(state, torch_module=torch))
    tensors, references, devices, identities = {}, {}, {}, {}

    def tensor(name, value):
        if value is None:
            references[name] = None
            return
        if name not in _TENSOR_AXES or _TENSOR_AXES[name] is not None:
            raise ParallelContractError("unsupported cache-off tensor path: " + name)
        if id(value) in identities:
            references[name] = identities[id(value)]
            return
        storage = value.untyped_storage()
        if (not value.is_contiguous() or value.storage_offset() != 0 or value.is_conj() or value.is_neg()
                or storage.nbytes() != value.numel() * value.element_size()
                or value.device.type not in ("cpu", "cuda")):
            raise ParallelContractError("unsupported storage at " + name)
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
        "request_id", "spec", "step_index", "cache_enabled", "events", "needs_anchor", "finished",
        "prepare_ms", "current_step_refreshed", "immutable_spec")}
    specs = tuple(KnownTensorSpec(name, tuple(value.shape), str(value.dtype)) for name, value in sorted(tensors.items()))
    # Reuse the declared-axis codec. Cache-off state is entirely replicated;
    # schema validation must never guess a partition axis from these shapes.
    tensors = merge_known_state(split_known_state(tensors, specs, (0,)), specs, (0,))
    manifest = json.loads(_json(dict(schema="ditserve.cache-off-wire.v1", metadata=metadata,
                                    scheduler=scheduler_metadata, references=references, devices=devices,
                                    timestep_views=timestep_views, specs=[dataclasses.asdict(s) for s in specs],
                                    tensor_hashes={name: _tensor_digest(value) for name, value in tensors.items()})))
    mathematical = copy.deepcopy(manifest)
    mathematical["metadata"].pop("events")
    mathematical["metadata"].pop("prepare_ms")
    manifest["state_sha256"] = hashlib.sha256(_json(mathematical).encode()).hexdigest()
    return manifest, tensors


def unpack_cache_off_request(worker, manifest, tensors):
    """Build an unregistered destination request; the transaction installs it."""
    torch = importlib.import_module("torch")
    from .runtime import RequestState, UNIPC_REQUIRED_FIELDS, UNIPC_OPTIONAL_FIELDS
    from diffusers.configuration_utils import FrozenDict
    expected = {"schema", "metadata", "scheduler", "references", "devices", "timestep_views", "specs", "tensor_hashes", "state_sha256"}
    if set(manifest) != expected or manifest["schema"] != "ditserve.cache-off-wire.v1":
        raise ParallelContractError("unknown migration manifest schema")
    mathematical = copy.deepcopy(manifest)
    mathematical.pop("state_sha256")
    mathematical["metadata"].pop("events")
    mathematical["metadata"].pop("prepare_ms")
    if hashlib.sha256(_json(mathematical).encode()).hexdigest() != manifest["state_sha256"]:
        raise ParallelContractError("migration manifest checksum mismatch")
    specs = tuple(KnownTensorSpec(**spec) for spec in manifest["specs"])
    values = merge_known_state({0: tensors}, specs, (0,))
    if set(values) != set(manifest["devices"]) or set(values) != set(manifest["tensor_hashes"]):
        raise ParallelContractError("migration tensor manifest mismatch")
    for name, value in values.items():
        if _tensor_digest(value) != manifest["tensor_hashes"][name]:
            raise ParallelContractError("migration tensor checksum mismatch: " + name)
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
        raise ParallelContractError("unknown migration logical tensor references")

    def get(name):
        reference = references[name]
        if reference is None:
            return None
        if reference not in values:
            raise ParallelContractError("unresolved migration tensor alias")
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
                         contexts={}, residency="gpu", pending_snapshot=None, owner=worker._owner)
    worker._validate_request_schema(state)
    _validate_cache_off_values(state, torch, wan_latent_shape(worker, state.spec))
    if state.cache_enabled or state.contexts or state.spec["cache"] or state.request_id != state.spec["request_id"]:
        raise ParallelContractError("invalid transported cache-off identity")
    return state


class CacheOffGroupMigration:
    """Four-rank control, target-group state transfer, then boundary commit.

    All four ranks call migrate(), including ranks with state=None while SP2
    executes. Only all4 <-> group01 is supported. Candidate state is built and
    acknowledged everywhere before the old request is released. Any failure
    poisons the handle and the supervising experiment terminates all ranks.
    """

    def __init__(self, worker, handle, groups):
        self.worker, self.handle, self.groups = worker, handle, groups

    def _reports(self, report):
        torch = importlib.import_module("torch")
        try:
            _json(report)
            reports = [None] * 4
            torch.distributed.all_gather_object(reports, report, group=self.groups.group((0, 1, 2, 3)))
            if any(item.get("error") for item in reports):
                raise ParallelContractError("migration rank rejected: " + _json(reports))
            return reports
        except Exception as error:
            self.handle.fail()
            raise ParallelContractError("migration control failed: " + str(error)) from error

    def migrate(self, state, source_ranks, target_ranks, *, request_id, step_index, expected_epoch):
        torch = importlib.import_module("torch")
        started = time.perf_counter()
        source_ranks, target_ranks = tuple(source_ranks), tuple(target_ranks)
        costs = {}
        try:
            error = None
            try:
                if (source_ranks, target_ranks) not in (((0, 1, 2, 3), (0, 1)), ((0, 1), (0, 1, 2, 3))):
                    raise ParallelContractError("only SP4 <-> group01 migration is supported")
                if self.handle.failed or self.handle.active is not None or self.handle.current.epoch != expected_epoch:
                    raise ParallelContractError("migration requires a healthy committed step boundary")
                expected_layout = source_ranks if self.handle.rank in source_ranks else (self.handle.rank,)
                if self.handle.current.ranks != expected_layout:
                    raise ParallelContractError("source layout differs from the committed handle")
                if (state is None) != (self.handle.rank not in source_ranks):
                    raise ParallelContractError("request presence differs from source membership")
                if self.handle.rank not in source_ranks and request_id in self.worker._states:
                    raise ParallelContractError("destination already owns the request")
            except Exception as exception:
                error = str(exception)
            reports = self._reports(dict(error=error, source=list(source_ranks), target=list(target_ranks),
                                         request_id=request_id, step_index=step_index, epoch=expected_epoch))
            if any(report != reports[0] for report in reports):
                raise ParallelContractError("migration transition disagreement")
            before = self.worker._memory()
            phase = time.perf_counter()
            manifest = tensors = None
            error = None
            try:
                if state is not None:
                    if state.request_id != request_id or state.step_index != step_index:
                        raise ParallelContractError("request/solver boundary mismatch")
                    self.worker._sync()
                    manifest, tensors = pack_cache_off_request(self.worker, state)
            except Exception as exception:
                error = str(exception)
            reports = self._reports(dict(error=error, fingerprint=None if manifest is None else manifest["state_sha256"],
                                         manifest=manifest if self.handle.rank == 0 else None))
            costs["export_and_consensus_ms"] = (time.perf_counter() - phase) * 1000
            canonical = reports[0]["manifest"]
            if any(reports[rank]["fingerprint"] != canonical["state_sha256"] for rank in source_ranks):
                raise ParallelContractError("source ranks disagree on complete request state")
            specs = tuple(KnownTensorSpec(**spec) for spec in canonical["specs"])
            # Reuse the exact declared-axis reshaping codec on the canonical
            # source. In this first transport all fields are replicated.
            if self.handle.rank == 0:
                tensors = reshard_known_state({0: tensors}, specs, (0,), target_ranks)[0]
            received = {}
            wire_bytes = 0
            phase = time.perf_counter()
            for spec in specs:
                payload = None
                error = None
                try:
                    if self.handle.rank in target_ranks:
                        device = self.worker.device if self.groups.backend == "nccl" else "cpu"
                        dtype = getattr(torch, spec.dtype.removeprefix("torch."))
                        payload = (tensors[spec.name].to(device) if self.handle.rank == 0 else
                                   torch.empty(spec.global_shape, dtype=dtype, device=device))
                except Exception as exception:
                    error = str(exception)
                self._reports(dict(error=error, phase="payload_allocation", name=spec.name))
                if self.handle.rank in target_ranks:
                    torch.distributed.broadcast(payload, src=0, group=self.groups.group(target_ranks))
                    received[spec.name] = payload.cpu().clone()
                    wire_bytes += payload.numel() * payload.element_size()
                self._reports(dict(error=None, phase="payload_received", name=spec.name))
            costs["tensor_transport_ms"] = (time.perf_counter() - phase) * 1000
            phase = time.perf_counter()
            candidate = None
            error = None
            try:
                if self.handle.rank in target_ranks:
                    candidate = unpack_cache_off_request(self.worker, canonical, received)
                    self.worker._sync()
            except Exception as exception:
                error = str(exception)
            self._reports(dict(error=error, phase="candidate_ready", fingerprint=canonical["state_sha256"]))
            costs["reconstruct_and_ack_ms"] = (time.perf_counter() - phase) * 1000
            candidate_memory = self.worker._memory()
            phase = time.perf_counter()
            error = None
            try:
                if state is not None:
                    self.worker.release(state)
                if candidate is not None:
                    if request_id in self.worker._states:
                        raise ParallelContractError("destination already owns the request")
                    self.worker._states[request_id] = candidate
                placement = target_ranks if self.handle.rank in target_ranks else (self.handle.rank,)
                self.handle.commit(placement, expected_epoch=expected_epoch)
            except Exception as exception:
                error = str(exception)
            self._reports(dict(error=error, phase="committed", epoch=expected_epoch + 1))
            costs["release_commit_and_ack_ms"] = (time.perf_counter() - phase) * 1000
            # Export buffers are CPU-only. Drop NCCL staging before observing
            # final allocator state; reserved bytes are recorded separately.
            payload = tensors = received = None
            self.worker._sync()
            event = dict(kind="cache_off_group_migration", source_ranks=list(source_ranks), target_ranks=list(target_ranks),
                         completed_steps=step_index, old_epoch=expected_epoch, new_epoch=expected_epoch + 1,
                         state_sha256=canonical["state_sha256"], received_tensor_bytes=wire_bytes,
                         logical_payload_bytes=sum(int(torch.tensor(spec.global_shape).prod()) *
                            torch.empty((), dtype=getattr(torch, spec.dtype.removeprefix("torch."))).element_size() for spec in specs),
                         transport="CPU staging plus target-group broadcast", phase_cost_ms=costs,
                         ms=(time.perf_counter() - started) * 1000, memory_before=before,
                         memory_with_candidate=candidate_memory,
                         memory_after=self.worker._memory(), gpu_qualified=False, cache_features_transferred=False)
            source_cuda_bytes = sum(tensorspec.global_shape[0] * math.prod(tensorspec.global_shape[1:]) *
                torch.empty((), dtype=getattr(torch, tensorspec.dtype.removeprefix("torch."))).element_size()
                for tensorspec in specs if canonical["devices"][tensorspec.name] == "cuda")
            event.update(export_d2h_bytes=source_cuda_bytes if self.handle.rank in source_ranks else 0,
                         reconstructed_h2d_bytes=source_cuda_bytes if self.handle.rank in target_ranks else 0,
                         logical_delivery_bytes_excluding_sender=event["logical_payload_bytes"] * (len(target_ranks) - 1),
                         physical_link_bytes_measured=False)
            if candidate is not None:
                candidate.events.append(event)
            return candidate, event
        except BaseException:
            self.handle.fail()
            raise
