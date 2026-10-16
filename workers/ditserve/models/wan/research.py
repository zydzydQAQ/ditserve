"""Isolated Flash Wan research adapter; never changes the historical worker."""

import importlib.metadata
import time

from .state import (DEPENDENCY_VERSION, refresh_certificate,
                    state_inventory, transition_plan, validate_refresh_certificate)


class ResearchWanAdapter:
    """Worker-style API using normalized worker specs, not benchmark trace rows.

    Exact-refresh reclamation is available only for explicit validation runs.
    It remains unqualified until independent real GPU trajectory evidence is
    supplied outside this adapter. No method can toggle that qualification.
    """

    def __init__(self, *, worker, validation_only=False):
        self.worker = worker
        if self.worker.attention != "flash":
            raise ValueError("research adapter requires Flash attention")
        self.dependency_version = importlib.metadata.version("cache-dit")
        if self.dependency_version != DEPENDENCY_VERSION:
            raise ValueError("research adapter requires cache-dit 1.5.0")
        self.validation_only = bool(validation_only)

    def _layout(self, state):
        return dict(sp_size=1, ranks=[0], device=str(state.latents.device), version=1)

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
