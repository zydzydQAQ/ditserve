"""Transactional CFG experiments for the pinned single-request Wan worker.

This adapter deliberately does not patch cache-dit classes or add RequestState
fields. It exposes the reference oracle and the separately qualified split-prefix
implementation. Both retain independent rollback tensor storage, including aliases
and strides; speculative cache-gate contexts borrow read-only feature tensors.
"""

import contextlib
import dataclasses
import time

from ..gpu.worker import _tree_map


_ABSENT = object()
_POLICIES = {
    "off": (False, 0.0, "independent", "native"),
    "independent_008": (True, 0.08, "independent", "native"),
    "independent_004": (True, 0.04, "independent", "native"),
    "paired_008": (True, 0.08, "paired", "oracle"),
}


def _clone_graph(torch, value, *, copy_tensors=True):
    """Clone metadata and, by default, raw storage while retaining aliases."""
    storages = {}

    def clone(tensor):
        if (tensor.layout != torch.strided or tensor.is_quantized or
                tensor.is_conj() or tensor.is_neg() or any(s < 0 for s in tensor.stride())):
            raise TypeError("CFG snapshot supports only ordinary strided tensors")
        if not copy_tensors:
            return tensor
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage._cdata)
        if key not in storages:
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,))
            storages[key] = raw.clone()
        return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            storages[key].untyped_storage(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()))

    result = _tree_map(value, torch.is_tensor, clone)
    # diffusers FrozenDict duplicates each value in both its mapping and its
    # attribute namespace. _tree_map intentionally handles mapping contents;
    # bind these aliases to the cloned values as well, never the old request.
    seen = set()

    def repair(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, dict):
            if (type(node).__module__ == "diffusers.configuration_utils" and
                    type(node).__name__ == "FrozenDict"):
                for name, child in node.items():
                    object.__setattr__(node, name, child)
            for child in node.values():
                repair(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                repair(child)
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for child in vars(node).values():
                repair(child)

    repair(result)
    return result


def _error(torch, actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("diagnostic reference tensor shape differs")
    a, b = actual.float(), reference.float()
    difference = a - b
    denominator = float(torch.linalg.vector_norm(b).item())
    numerator = float(torch.linalg.vector_norm(difference).item())
    return {"relative_l2": numerator / max(denominator, 1e-12),
            "max_abs": float(difference.abs().max().item()),
            "reference_l2": denominator, "epsilon": 1e-12}


class CfgResearchAdapter:
    """Wrap a WanStepWorker without changing its public state schema.

    ``reference_state`` is an independently advanced off trajectory at the next
    complete step. Diagnostics compare its latent after committing this step.
    Policy and implementation identities must be persisted by the caller; this
    class never stores policy on RequestState or qualifies performance evidence.
    """

    def __init__(self, worker):
        self.worker = worker
        self._split_prefix = None

    def _validate(self, state, policy):
        if not isinstance(policy, dict):
            raise TypeError("CFG policy must be an explicit mapping")
        identity = policy.get("id", policy.get("policy_id"))
        if identity not in _POLICIES:
            raise ValueError("unknown CFG research policy: " + str(identity))
        if "id" in policy and "policy_id" in policy and policy["id"] != policy["policy_id"]:
            raise ValueError("conflicting CFG policy identities")
        cache, threshold, mode, implementation = _POLICIES[identity]
        if identity == "paired_008" and policy.get("implementation") == "split_prefix":
            implementation = "split_prefix"
        if (policy.get("threshold") != threshold or policy.get("max_continuous", 1) != 1 or
                policy.get("cfg_mode") != mode or policy.get("implementation") != implementation or
                policy.get("cache", cache) is not cache):
            raise ValueError("unsupported CFG policy definition or implementation")
        if (state.spec["cache"] is not cache or state.spec["threshold"] != threshold or
                state.spec["max_continuous"] != 1 or state.cache_enabled is not cache):
            raise ValueError("request spec does not match the frozen CFG policy")
        if state.spec["guidance_scale"] <= 1:
            raise ValueError("CFG research requires two guidance branches")
        if state.pending_snapshot is not None:
            raise ValueError("CFG step cannot run during an outstanding state transition")
        if cache and len(state.contexts) != 1:
            raise ValueError("CFG research requires the single pinned Wan cache context")
        return identity, mode

    def _snapshot(self, state, copy_tensors=True):
        worker, torch = self.worker, self.worker.torch
        values = {key: value for key, value in vars(state).items()
                  if key not in ("scheduler", "generator", "owner")}
        graph = _clone_graph(torch, {"values": values, "scheduler": vars(state.scheduler)}, copy_tensors=copy_tensors)
        cuda_rng = None
        if str(worker.device).startswith("cuda:"):
            cuda_rng = torch.cuda.get_rng_state(worker.device).clone()
        return {"graph": graph, "copy_tensors": copy_tensors, "request_rng": state.generator.get_state().clone(),
                "cpu_rng": torch.get_rng_state().clone(), "cuda_rng": cuda_rng,
                "manager": dict(vars(worker.manager)),
                "forward": worker.pipe.transformer.forward}

    def _restore(self, state, snapshot):
        worker, torch = self.worker, self.worker.torch
        graph = _clone_graph(torch, snapshot["graph"], copy_tensors=snapshot["copy_tensors"])
        for key, value in graph["values"].items():
            setattr(state, key, value)
        vars(state.scheduler).clear()
        vars(state.scheduler).update(graph["scheduler"])
        state.generator.set_state(snapshot["request_rng"])
        torch.set_rng_state(snapshot["cpu_rng"])
        if snapshot["cuda_rng"] is not None:
            torch.cuda.set_rng_state(snapshot["cuda_rng"], worker.device)
        vars(worker.manager).clear()
        vars(worker.manager).update(snapshot["manager"])
        worker.pipe.transformer.forward = snapshot["forward"]

    @contextlib.contextmanager
    def _gate(self, events, branch, force_full):
        manager = self.worker.manager
        prior = vars(manager).get("can_cache", _ABSENT)
        native = manager.can_cache

        def wrapped(*args, **kwargs):
            # The native gate may reset a continuous-reuse counter or record a
            # residual. It must execute even when we force a full calculation.
            allowed = bool(native(*args, **kwargs))
            if any(event["branch"] == branch[0] for event in events):
                raise RuntimeError("unsupported multiple cache gates per Wan CFG branch")
            context = manager._current_context
            residuals = (context.residual_diffs if branch[0] == "cond"
                         else context.cfg_residual_diffs)
            event = {"branch": branch[0], "native_reuse": allowed,
                     "accepted_reuse": allowed and not force_full,
                     "forced_full": bool(force_full),
                     "residual_diff": residuals.get(str(context.executed_steps - 1))}
            events.append(event)
            return allowed and not force_full

        manager.can_cache = wrapped
        try:
            yield
        finally:
            if prior is _ABSENT:
                vars(manager).pop("can_cache", None)
            else:
                manager.can_cache = prior

    def _pair(self, state, *, force_full=False):
        worker = self.worker
        events, branch, durations, outputs = [], [None], {}, {}
        started = time.perf_counter()
        with worker._activate(state), self._gate(events, branch, force_full):
            timestep = state.scheduler.timesteps[state.step_index].expand(state.latents.shape[0])
            model_input = state.latents.to(worker.pipe.transformer.dtype)
            for name, embedding in (("cond", state.prompt_embeds),
                                    ("uncond", state.negative_prompt_embeds)):
                branch[0] = name
                worker._sync()
                branch_started = time.perf_counter()
                with worker.pipe.transformer.cache_context(name):
                    outputs[name] = worker.pipe.transformer(
                        hidden_states=model_input, timestep=timestep,
                        encoder_hidden_states=embedding, attention_kwargs=None,
                        return_dict=False)[0]
                worker._sync()
                durations[name] = (time.perf_counter() - branch_started) * 1000
        if state.cache_enabled and [event["branch"] for event in events] != ["cond", "uncond"]:
            raise RuntimeError("expected exactly one native gate per CFG branch")
        worker._sync()
        return outputs, {"gates": events, "branch_ms": durations,
                         "pair_ms": (time.perf_counter() - started) * 1000}

    def _guided(self, state, predictions):
        return predictions["uncond"] + state.spec["guidance_scale"] * (
            predictions["cond"] - predictions["uncond"])

    def _timed(self, operation, *args):
        self.worker._sync()
        started = time.perf_counter()
        result = operation(*args)
        self.worker._sync()
        return result, (time.perf_counter() - started) * 1000

    def step(self, state, *, policy, diagnostic=False, reference_state=None):
        worker, torch = self.worker, self.worker.torch
        with worker._lock, torch.inference_mode(), worker.p._sdpa_math_context(torch):
            worker._check(state, "gpu")
            identity, mode = self._validate(state, policy)
            if state.finished or state.step_index >= state.spec["steps"]:
                raise ValueError("request has no remaining denoising steps")
            if reference_state is not None:
                if not diagnostic or reference_state is state or reference_state.cache_enabled:
                    raise ValueError("latent reference must be an independent off diagnostic trajectory")
                if reference_state.step_index != state.step_index + 1:
                    raise ValueError("latent reference must have completed exactly the next solver step")
                fields = ("prompt", "negative_prompt", "seed", "height", "width", "frames",
                          "steps", "guidance_scale")
                if any(reference_state.spec[k] != state.spec[k] for k in fields):
                    raise ValueError("latent reference input differs")

            if mode == "independent" and not diagnostic:
                record = worker.step(state)
                record["cfg_research"] = {"policy_id": identity,
                    "policy_sha256": policy.get("sha256"), "implementation": "native",
                    "diagnostic": False, "performance_eligible": True}
                return record

            split = policy["implementation"] == "split_prefix"
            if split:
                if self._split_prefix is None:
                    from .cfg_split_prefix import CfgSplitPrefix
                    self._split_prefix = CfgSplitPrefix(worker)
                self._split_prefix.validate_state(state)

            worker._sync()
            started = time.perf_counter()
            # Retain the oracle's independent rollback snapshot for both paths.
            # Only disposable native-gate probes borrow read-only feature tensors.
            before, snapshot_ms = self._timed(self._snapshot, state)
            index, anchor = state.step_index, state.needs_anchor
            restore_ms = replay_ms = discarded_ms = diagnostic_ms = 0.0
            try:
                worker._update_policy(state)
                if split:
                    predictions, accepted = self._split_prefix.pair(state)
                    attempted = dict(gates=accepted["probe_gates"], pair_ms=accepted["gate_probe_ms"],
                                     cost_semantics="isolated native gate probes; included in accepted pair wall time")
                else:
                    predictions, attempted = self._pair(state)
                    accepted = attempted
                mixed = (len(attempted["gates"]) == 2 and
                         attempted["gates"][0]["accepted_reuse"] !=
                         attempted["gates"][1]["accepted_reuse"])
                if mode == "paired" and mixed and not split:
                    _, elapsed = self._timed(self._restore, state, before)
                    restore_ms += elapsed
                    predictions, accepted = self._pair(state, force_full=True)
                    replay_ms = accepted["pair_ms"]
                    discarded_ms = attempted["pair_ms"]

                errors = None
                if diagnostic and any(event["accepted_reuse"] for event in accepted["gates"]):
                    worker._sync()
                    diagnostic_started = time.perf_counter()
                    committed = self._snapshot(state)
                    self._restore(state, before)
                    full, _ = self._pair(state, force_full=True)
                    errors = {name: _error(torch, predictions[name], full[name])
                              for name in ("cond", "uncond")}
                    errors["cfg"] = _error(torch, self._guided(state, predictions), self._guided(state, full))
                    self._restore(state, committed)
                    worker._sync()
                    diagnostic_ms = (time.perf_counter() - diagnostic_started) * 1000

                timestep_scalar = state.scheduler.timesteps[index]
                state.latents = state.scheduler.step(self._guided(state, predictions),
                    timestep_scalar, state.latents, return_dict=False)[0]
                state.step_index += 1
                state.needs_anchor = False
                worker._sync()
                finite = bool(torch.isfinite(state.latents).all().item())
                if not finite:
                    raise RuntimeError("nonfinite latent after CFG solver step %d" % index)
                path = worker._path(state, index)
                if anchor and path["reused_forwards"]:
                    raise RuntimeError("cold restore unexpectedly reused a missing cache")
                if mode == "paired" and path["path"] == "mixed":
                    raise RuntimeError("paired oracle committed mixed CFG cache branches")
                for context in state.contexts.values():
                    if (context.executed_steps != state.step_index or
                            context.transformer_executed_steps != state.step_index * 2):
                        raise RuntimeError("CFG cache counters diverged from solver")
                latent_error = None
                if reference_state is not None:
                    worker._sync()
                    latent_started = time.perf_counter()
                    latent_error = _error(torch, state.latents, reference_state.latents)
                    worker._sync()
                    diagnostic_ms += (time.perf_counter() - latent_started) * 1000
                worker._sync()
                total_ms = (time.perf_counter() - started) * 1000
                optimistic = max(0.0, total_ms - snapshot_ms - restore_ms - discarded_ms - diagnostic_ms)
                research = {"policy_id": identity, "policy_sha256": policy.get("sha256"),
                    "implementation": policy["implementation"], "diagnostic": bool(diagnostic),
                    "performance_eligible": split and not diagnostic, "native_attempt": attempted,
                    "accepted_pair": accepted, "mixed_native_attempt": mixed,
                    "snapshot_ms": snapshot_ms, "restore_ms": restore_ms,
                    "replay_ms": replay_ms, "discarded_probe_ms": discarded_ms,
                    "diagnostic_ms": diagnostic_ms, "same_input_prediction_error": errors,
                    "off_trajectory_latent_error": latent_error,
                    "snapshot_semantics": "independent raw-storage copy preserving aliases and strides"}
                if split:
                    research["split_prefix"] = {key: accepted[key] for key in
                        ("prefix_ms", "gate_probe_ms", "commit_ms", "context_guard_ms", "cache_context_mode",
                         "fn_block_calls", "mn_block_calls", "gate_calls", "cost_semantics")}
                    research["measured_compute_ms"] = total_ms
                else:
                    research.update(optimistic_compute_ms=optimistic,
                        optimistic_cost_semantics="implementation-screening lower estimate only; not measured serving benefit")
                record = {"kind": "step", "step_index": index, "completed_steps": state.step_index,
                    "ms": total_ms, "forced_cold_anchor": anchor, "finite": finite,
                    "threshold": state.spec["threshold"], "max_continuous": state.spec["max_continuous"],
                    **path, "cfg_research": research}
                state.events.append(record)
                return record
            except BaseException:
                self._restore(state, before)
                worker._sync()
                raise
