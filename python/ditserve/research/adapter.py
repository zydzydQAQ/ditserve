"""Isolated Flash Wan research adapter; never changes the historical worker."""

import importlib.metadata
import math
import time

from ..gpu.worker import WanStepWorker, normalize_spec
from .state import (DEPENDENCY_VERSION, assert_transfer_supported, refresh_certificate,
                    state_inventory, transition_plan, validate_refresh_certificate)


class ResearchWanAdapter:
    """Worker-style API using normalized worker specs, not benchmark trace rows.

    Exact-refresh reclamation is available only for explicit validation runs.
    It remains unqualified until independent real GPU trajectory evidence is
    supplied outside this adapter. No method can toggle that qualification.
    """

    def __init__(self, device="cuda:0", model_path=None, *, worker=None, validation_only=False):
        self.worker = worker if worker is not None else WanStepWorker(device=device, model_path=model_path, attention="flash")
        if self.worker.attention != "flash":
            raise ValueError("research adapter requires Flash attention")
        self.dependency_version = importlib.metadata.version("cache-dit")
        if self.dependency_version != DEPENDENCY_VERSION:
            raise ValueError("research adapter requires cache-dit 1.5.0")
        self.validation_only = bool(validation_only)
        self._changed_approximation = set()

    def _layout(self, state):
        return dict(sp_size=1, ranks=[0], device=str(state.latents.device), version=1)

    def prepare(self, spec):
        spec = normalize_spec(spec)
        if spec["max_continuous"] != 1:
            raise ValueError("research pilot requires max_continuous=1")
        return self.worker.prepare(spec)

    def encode_prompt(self, spec):
        return self.worker.encode_prompt(spec)

    def prepare_encoded(self, spec, payload):
        spec = normalize_spec(spec)
        if spec["max_continuous"] != 1:
            raise ValueError("research pilot requires max_continuous=1")
        return self.worker.prepare_encoded(spec, payload)

    def step(self, state):
        return self.worker.step(state)

    def finish(self, state):
        return self.worker.finish(state)

    def release(self, state):
        self.worker.release(state)
        self._changed_approximation.discard(id(state))

    def inventory(self, state):
        with self.worker._lock:
            self.worker._check(state)
            return state_inventory(state, torch_module=self.worker.torch)

    def certificate(self, state):
        with self.worker._lock:
            self.worker._check(state, "gpu")
            return refresh_certificate(state, layout=self._layout(state), dependency_version=self.dependency_version,
                                       torch_module=self.worker.torch)

    def plan(self, state, action, *, certificate=None, target_layout=None):
        with self.worker._lock:
            self.worker._check(state, "gpu")
            return transition_plan(state, action, certificate=certificate, layout=self._layout(state),
                                   target_layout=target_layout, dependency_version=self.dependency_version,
                                   torch_module=self.worker.torch)

    def snapshot(self, state, mode="hot", *, certificate=None):
        with self.worker._lock:
            self.worker._check(state, "gpu")
            if mode != "exact_refresh":
                self.plan(state, mode)
                inventory = self.inventory(state)
                if mode in ("hot", "cold") and any(row.get("supported_timestep_views") for row in inventory["storages"]):
                    from .storage import snapshot_timestep_views
                    return snapshot_timestep_views(self.worker, state, mode, inventory)
                return self.worker.snapshot(state, mode)
            if not self.validation_only:
                raise ValueError("exact_refresh is validation-only and lacks GPU trajectory qualification")
            started = time.perf_counter()
            self.worker._sync()
            checked = validate_refresh_certificate(state, certificate, layout=self._layout(state),
                dependency_version=self.dependency_version, torch_module=self.worker.torch)
            before = self.worker._memory()
            # No copy of buffers may survive this block: it would retain their
            # storage. Required state, counters and needs_anchor are unchanged.
            for context in state.contexts.values():
                context.buffers.clear()
            self.worker._sync()
            after = self.worker._memory()
            event = dict(kind="snapshot", mode=mode, ms=(time.perf_counter() - started) * 1000,
                         transfer_bytes=0, expected_feature_release_bytes=checked["reclaimable_cuda_bytes"],
                         allocated_delta_bytes=before["allocated_bytes"] - after["allocated_bytes"],
                         memory_before=before, memory_after=after, residency=state.residency,
                         certificate_sha256=checked["sha256"], qualification="validation_only_not_GPU_qualified",
                         trajectory_semantics="natural_refresh_candidate_preserves_all_decision_metadata")
            state.events.append(event)
            return event

    def restore(self, state):
        with self.worker._lock:
            self.worker._check(state)
            if state.residency == "cpu":
                assert_transfer_supported(state_inventory(state, torch_module=self.worker.torch))
                if state.pending_snapshot.get("research_storage_codec"):
                    from .storage import restore_timestep_views
                    return restore_timestep_views(self.worker, state)
            return self.worker.restore(state)

    def set_approximation(self, state, *, threshold, cache_enabled=True):
        """One explicit step-16 suffix change; qualification belongs to the controller.

        This executes an authorized research action, not a quality decision. A
        full-compute suffix is still an altered *whole* trajectory and does not
        erase approximation in its prefix. m=1 is always retained.
        """
        if type(cache_enabled) is not bool or isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("invalid approximation action")
        if threshold not in (0.0, 0.04, 0.08) or cache_enabled != (threshold > 0):
            raise ValueError("pilot suffix must be full, 0.04/m1 or 0.08/m1")
        with self.worker._lock:
            self.worker._check(state, "gpu")
            if state.step_index != 16 or state.finished or id(state) in self._changed_approximation:
                raise ValueError("only one approximation change at completed step 16 is supported")
            before = dict(cache=state.spec["cache"], threshold=state.spec["threshold"], max_continuous=state.spec["max_continuous"])
            state.spec.update(cache=cache_enabled, threshold=float(threshold), max_continuous=1)
            self.worker._update_policy(state)
            self._changed_approximation.add(id(state))
            event = dict(kind="research_suffix_change", completed_steps=16, before=before,
                         after=dict(cache=cache_enabled, threshold=float(threshold), max_continuous=1),
                         quality_scope="whole_prefix_and_suffix_requires_independent_calibration", qualified=False)
            state.events.append(event)
            return event

    def close(self):
        """Release live requests; ownership of model lifetime stays with caller."""
        for state in list(self.worker._states.values()):
            self.release(state)
