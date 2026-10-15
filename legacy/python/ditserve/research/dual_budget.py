"""Reallocate only unused GPU-seconds without resetting the original ledger.

The legacy sessions, used totals and stage totals remain intact. A separate
allocation records baselines and caps for this new SP1 protocol. All writes use
the same exclusive ledger lock as the original supervisor.
"""

import contextlib
import copy
import fcntl
import math
import os
from pathlib import Path
import time
import uuid

from .budget import BudgetStop, MAX_GPU_SECONDS, ResearchBudget, WorkerCleanupError
from .common import atomic_json, digest, finite, process_group_alive, read_json


STAGE_SHARES = {"correctness": .08, "calibration": .20, "migration": .30,
                "mechanism": .22, "capacity": .05, "reserve": .15}
POLICY = "ditserve.research.dual.allocation.v1"
REVISION_POLICY = "ditserve.research.dual.protocol-revision.v1"


class DualResearchBudget(ResearchBudget):
    """The caller must explicitly reallocate once, after all old workers exit."""

    def __init__(self, path, limit_gpu_seconds=MAX_GPU_SECONDS, clock=time.time, monotonic=time.monotonic):
        super().__init__(path, limit_gpu_seconds, clock, monotonic)
        if self.limit != MAX_GPU_SECONDS:
            raise ValueError("the shared lifetime ledger must retain its 28,800 GPU-second cap")

    def _locked_read(self):
        if self._lock is not None:
            raise RuntimeError("budget session already active")
        # Never create a fresh budget: doing so could discard earlier spending.
        if not self.path.is_file():
            raise ValueError("existing research lifetime ledger is required")
        self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._data = read_json(self.path)
            self._validate_ledger()
        except BlockingIOError as error:
            self._close()
            raise RuntimeError("research budget already in use") from error
        except BaseException:
            self._close()
            raise

    def _validate_ledger(self):
        data = self._data
        if (data.get("schema_version") != "ditserve.research.budget.v1"
                or data.get("limit_gpu_seconds") != MAX_GPU_SECONDS
                or not isinstance(data.get("sessions"), list)
                or not isinstance(data.get("stage_used_gpu_seconds"), dict)
                or "active" not in data):
            raise ValueError("invalid original lifetime ledger identity")
        used = finite(data["used_gpu_seconds"], "used GPU-seconds")
        stages = data["stage_used_gpu_seconds"]
        for value in stages.values():
            finite(value, "stage used GPU-seconds")
        sessions_total = 0.
        by_stage = dict.fromkeys(stages, 0.)
        for row in data["sessions"]:
            cost = finite(row["gpu_seconds"], "session GPU-seconds")
            sessions_total += cost
            if row.get("stage") not in by_stage:
                raise ValueError("session stage missing from original ledger totals")
            by_stage[row["stage"]] += cost
        if (not math.isclose(sessions_total, used, rel_tol=1e-10, abs_tol=1e-6)
                or any(not math.isclose(by_stage[k], v, rel_tol=1e-10, abs_tol=1e-6) for k, v in stages.items())):
            raise ValueError("lifetime session and stage totals do not reconcile")

    def _allocation(self):
        records = self._data.get("reallocations", [])
        if not records:
            raise ValueError("remaining lifetime budget has not been reallocated")
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("the original two-GPU allocation cannot be replaced")
        allocation = records[0]
        if (allocation.get("schema_version") != POLICY
                or allocation.get("sha256") != digest({k: v for k, v in allocation.items() if k != "sha256"})
                or allocation.get("shares") != STAGE_SHARES
                or allocation.get("limit_gpu_seconds") != self.limit):
            raise ValueError("invalid two-GPU remaining-budget allocation")
        limits = copy.deepcopy(allocation["original_stage_limits"])
        for stage, allowance in allocation["stage_allowances_gpu_seconds"].items():
            limits[stage] = allocation["baseline_stage_used"].get(stage, 0.) + allowance
        if self._data.get("stage_limits_gpu_seconds") != limits:
            raise ValueError("the original two-GPU allocation caps cannot change")
        return allocation

    def _current_protocol(self, allocation=None):
        """Validate the append-only binding chain against its recorded history."""
        allocation = allocation or self._allocation()
        current = allocation["protocol_sha256"]
        parent_sha = allocation["sha256"]
        seen = {current}
        records = self._data.get("protocol_revisions", [])
        if not isinstance(records, list):
            raise ValueError("invalid frozen protocol revision chain")
        prior_count = allocation["prior_session_count"]
        for row in records:
            if (not isinstance(row, dict)
                    or row.get("schema_version") != REVISION_POLICY
                    or row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"})
                    or row.get("allocation_id") != allocation["allocation_id"]
                    or row.get("allocation_sha256") != allocation["sha256"]
                    or row.get("previous_binding_sha256") != parent_sha
                    or row.get("previous_protocol_sha256") != current
                    or not isinstance(row.get("new_protocol_sha256"), str)
                    or not row["new_protocol_sha256"].strip()
                    or row["new_protocol_sha256"] in seen
                    or not isinstance(row.get("reason"), str) or not row["reason"].strip()):
                raise ValueError("invalid frozen protocol revision chain")
            count = row.get("session_count")
            if type(count) is not int or not prior_count <= count <= len(self._data["sessions"]):
                raise ValueError("invalid protocol revision spending history")
            history = self._data["sessions"][:count]
            historical_stages = dict.fromkeys(self._data["stage_used_gpu_seconds"], 0.)
            for session in history:
                historical_stages[session["stage"]] += session["gpu_seconds"]
            if (row.get("sessions_sha256") != digest(history)
                    or row.get("used_gpu_seconds") != sum(s["gpu_seconds"] for s in history)
                    or row.get("stage_used_gpu_seconds") != historical_stages
                    or row.get("stage_limits_sha256") != digest(self._data["stage_limits_gpu_seconds"])):
                raise ValueError("protocol revision cannot change historical spending or caps")
            current, parent_sha, prior_count = row["new_protocol_sha256"], row["sha256"], count
            seen.add(current)
        return current

    def bind_protocol_revision(self, previous_sha, new_sha, reason):
        """Bind a new frozen source/protocol without replenishing any allowance.

        This is an explicit compare-and-set operation, only while the ledger is
        idle. Repeating the identical latest binding is idempotent. It does not
        recover active reservations or alter allocation caps.
        """
        if any(not isinstance(value, str) or not value.strip() for value in (previous_sha, new_sha, reason)):
            raise ValueError("previous/new frozen protocol checksums and a reason are required")
        if previous_sha == new_sha:
            raise ValueError("a protocol revision requires a different new checksum")
        self._locked_read()
        try:
            if self._data["active"] is not None:
                raise WorkerCleanupError("active reservation must be reconciled before protocol revision")
            allocation = self._allocation()
            current_protocol = self._current_protocol(allocation)
            revisions = self._data.get("protocol_revisions", [])
            if (current_protocol == new_sha and revisions
                    and revisions[-1]["previous_protocol_sha256"] == previous_sha
                    and revisions[-1]["reason"] == reason):
                return copy.deepcopy(revisions[-1])
            if current_protocol != previous_sha:
                raise ValueError("previous checksum is not the latest frozen protocol")
            historical_protocols = {allocation["protocol_sha256"]}
            historical_protocols.update(row["new_protocol_sha256"] for row in revisions)
            if new_sha in historical_protocols:
                raise ValueError("a protocol revision cannot roll back to an earlier binding")
            row = dict(schema_version=REVISION_POLICY, revision_id=uuid.uuid4().hex,
                allocation_id=allocation["allocation_id"], allocation_sha256=allocation["sha256"],
                previous_binding_sha256=revisions[-1]["sha256"] if revisions else allocation["sha256"],
                previous_protocol_sha256=previous_sha, new_protocol_sha256=new_sha,
                reason=reason, bound_at=self.clock(), session_count=len(self._data["sessions"]),
                sessions_sha256=digest(self._data["sessions"]), used_gpu_seconds=self._data["used_gpu_seconds"],
                stage_used_gpu_seconds=copy.deepcopy(self._data["stage_used_gpu_seconds"]),
                stage_limits_sha256=digest(self._data["stage_limits_gpu_seconds"]))
            row["sha256"] = digest(row)
            self._data.setdefault("protocol_revisions", []).append(row)
            atomic_json(self.path, self._data)
            return copy.deepcopy(row)
        finally:
            self._close()

    def reallocate_remaining(self, *, protocol_sha256):
        """Append one immutable allocation; no recovery or worker action here."""
        if not isinstance(protocol_sha256, str) or not protocol_sha256:
            raise ValueError("a frozen protocol checksum is required")
        self._locked_read()
        try:
            if self._data["active"] is not None:
                raise WorkerCleanupError("active original reservation must be reconciled before reallocating")
            if self._data.get("reallocations"):
                allocation = self._allocation()
                if self._current_protocol(allocation) != protocol_sha256:
                    raise ValueError("remaining budget was already allocated to a different frozen protocol")
                return copy.deepcopy(allocation)
            remaining = max(0., self.limit - self._data["used_gpu_seconds"])
            if remaining <= 0:
                raise BudgetStop("no lifetime GPU-seconds remain")
            baseline = dict(self._data["stage_used_gpu_seconds"])
            allocation = dict(schema_version=POLICY, allocation_id=uuid.uuid4().hex,
                              protocol_sha256=protocol_sha256, allocated_at=self.clock(),
                              limit_gpu_seconds=self.limit, lifetime_used_at_allocation=self._data["used_gpu_seconds"],
                              prior_session_count=len(self._data["sessions"]), prior_sessions_sha256=digest(self._data["sessions"]),
                              original_stage_limits=copy.deepcopy(self._data.get("stage_limits_gpu_seconds", {})),
                              baseline_stage_used=baseline, remaining_gpu_seconds=remaining, shares=dict(STAGE_SHARES),
                              stage_allowances_gpu_seconds={k: remaining * v for k, v in STAGE_SHARES.items()})
            allocation["sha256"] = digest(allocation)
            self._data.setdefault("reallocations", []).append(allocation)
            # Preserve historical spending. Add the new limits to make the old
            # supervisor reject this now-reallocated ledger instead of borrowing
            # from this allocation through its old hard-coded stage limits.
            for stage in STAGE_SHARES:
                self._data["stage_used_gpu_seconds"].setdefault(stage, 0.)
                self._data["stage_limits_gpu_seconds"][stage] = (baseline.get(stage, 0.)
                    + allocation["stage_allowances_gpu_seconds"][stage])
            atomic_json(self.path, self._data)
            return copy.deepcopy(allocation)
        finally:
            self._close()

    def _open(self):
        self._locked_read()
        try:
            allocation = self._allocation()
            protocol_sha256 = self._current_protocol(allocation)
            active = self._data["active"]
            if active:
                if (active.get("allocation_id") != allocation["allocation_id"]
                        or active.get("protocol_sha256") != protocol_sha256):
                    raise WorkerCleanupError("old or foreign active reservation cannot be recovered by the two-GPU budget")
                if active.get("workers_expected") and process_group_alive(active.get("worker_pgid")):
                    raise WorkerCleanupError("GPU worker cleanup unconfirmed; reservation remains active")
                wall = self.clock() - active["started_at"]
                wall = wall if wall >= 0 else self.limit / active["gpu_count"]
                self._charge(wall, "interrupted_conservatively_charged")
        except BaseException:
            self._close()
            raise

    def allocation_remaining(self):
        allocation = self._allocation()
        return {stage: max(0., allowance - (self._data["stage_used_gpu_seconds"].get(stage, 0.)
                    - allocation["baseline_stage_used"].get(stage, 0.)))
                for stage, allowance in allocation["stage_allowances_gpu_seconds"].items()}

    def confirm_no_workers_started(self):
        """Only for Popen failing before returning any process handle.

        Once a process group was registered, cleanup must be proved instead.
        """
        if self._started is None or self._data["active"].get("worker_pgid") is not None:
            raise WorkerCleanupError("cannot clear an existing or unowned worker group")
        self._data["active"]["workers_expected"] = False
        atomic_json(self.path, self._data)

    def check(self, estimate_wall_seconds=0., cleanup_wall_seconds=0.):
        if self._started is None:
            raise RuntimeError("GPU work requires an active research budget session")
        estimate = finite(estimate_wall_seconds, "complete unit estimate")
        cleanup = finite(cleanup_wall_seconds, "cleanup estimate")
        active = self._data["active"]
        running = max(0., self.monotonic() - self._started) * active["gpu_count"]
        remaining = min(self.limit - self._data["used_gpu_seconds"], self.allocation_remaining()[active["stage"]]) - running
        if (estimate + cleanup) * active["gpu_count"] >= remaining:
            raise BudgetStop("insufficient remaining GPU-seconds for a complete unit and cleanup")
        return remaining / active["gpu_count"]

    @contextlib.contextmanager
    def session(self, stage, devices, estimate_wall_seconds, cleanup_wall_seconds=30., *, protocol_sha256=None):
        devices = list(devices)
        if (stage not in STAGE_SHARES or len(devices) not in (1, 2)
                or any(not isinstance(d, str) or not d.startswith("GPU-") for d in devices)
                or len(set(devices)) != len(devices)):
            raise ValueError("a dual stage and one or two unique physical GPU UUIDs are required")
        estimate = finite(estimate_wall_seconds, "complete unit estimate", positive=True)
        cleanup = finite(cleanup_wall_seconds, "cleanup reserve")
        self._open()
        status = "failed"
        self._outcome = None
        cleanup_error = False
        try:
            allocation = self._allocation()
            current_protocol = self._current_protocol(allocation)
            if protocol_sha256 is not None and protocol_sha256 != current_protocol:
                raise ValueError("session checksum is not the latest frozen protocol")
            needed = (estimate + cleanup) * len(devices)
            if needed >= min(self.limit - self._data["used_gpu_seconds"], self.allocation_remaining()[stage]):
                raise BudgetStop("complete unit does not fit the remaining stage and lifetime allocation")
            self._started = self.monotonic()
            self._data["active"] = dict(session_id=uuid.uuid4().hex, pid=os.getpid(), stage=stage,
                devices=devices, gpu_count=len(devices), started_at=self.clock(),
                allocation_id=allocation["allocation_id"], protocol_sha256=current_protocol,
                complete_unit_estimate_wall_seconds=estimate, cleanup_estimate_wall_seconds=cleanup)
            atomic_json(self.path, self._data)
            yield self
            status = self._outcome or "completed"
        except BudgetStop:
            status = "budget_stopped"
            raise
        except WorkerCleanupError:
            cleanup_error = True
            raise
        finally:
            try:
                if self._started is not None:
                    active = self._data["active"]
                    # Check even on KeyboardInterrupt/SystemExit and on a normal
                    # return: failed cleanup must never erase a live reservation.
                    unconfirmed = cleanup_error or (active.get("workers_expected")
                                                    and process_group_alive(active.get("worker_pgid")))
                    if unconfirmed:
                        active["cleanup_status"] = "unreaped_workers"
                        atomic_json(self.path, self._data)
                        if not cleanup_error:
                            raise WorkerCleanupError("worker exit unconfirmed; lifetime reservation remains active")
                    else:
                        self._charge(max(0., self.monotonic() - self._started), status)
            finally:
                self._started = None
                self._close()
