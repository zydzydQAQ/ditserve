"""Explicit, independent single-A100 allowance for the cache online comparison.

Importing this module never creates a ledger. The caller must obtain a concrete
additional authorization before entering ``SupplementAllocation``. Start before
model loading and finish only after the child group and GPU cleanup are audited;
loading, warmup, failed work, evaluation and cleanup all consume this allowance.
The two earlier ledgers are immutable evidence, never a source of unused credit.
"""

import copy
import fcntl
import math
import os
import time
import uuid
from pathlib import Path

from ..research.common import atomic_json, file_digest, finite, read_json
from .budget import BudgetGuard, BudgetStop


DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "results/optimization-cache-online/gpu-budget.json"
STAGE = "cache_k4"
SCHEMA = "ditserve.optimizations.supplement_budget.v1"
FAILURES = frozenset(("failed", "incomplete", "budget_stopped", "blocked_contention"))


def _sha256(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def _ordinary_path(value):
    path = Path(value).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("budget paths must not traverse symbolic links")
    if path.exists() and not path.is_file():
        raise ValueError("budget path must be an ordinary file")
    return path.resolve()


class SupplementAllocation:
    """Allocation-compatible lifecycle; explicit amount/device and two parents.

    Hardware admission remains the supervisor's responsibility. ``gpu_uuid`` is
    the A100 UUID it has independently verified; only that single device may be
    charged here. An unsettled session is never silently cleared or refunded.
    """

    def __init__(self, path, optimization_parent, research_parent, *,
                 authorized_gpu_seconds, authorization, gpu_uuid, budget_unlimited=False,
                 monotonic=time.monotonic, clock=time.time):
        if type(budget_unlimited) is not bool:
            raise ValueError("budget_unlimited must be an explicit boolean")
        if budget_unlimited and authorized_gpu_seconds is not None:
            raise ValueError("unlimited authorization requires authorized_gpu_seconds=None")
        self.budget_unlimited = budget_unlimited
        self.limit = None if budget_unlimited else finite(authorized_gpu_seconds, "authorized_gpu_seconds", positive=True)
        if not isinstance(authorization, str) or not authorization.strip():
            raise ValueError("an explicit additional authorization description is required")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-") or len(gpu_uuid) <= 4:
            raise ValueError("the independently admitted A100 GPU UUID is required")
        self.path = _ordinary_path(path)
        self.parent = _ordinary_path(optimization_parent)
        self.research_parent = _ordinary_path(research_parent)
        if len({self.path, self.parent, self.research_parent}) != 3:
            raise ValueError("supplement and both historical ledgers must be distinct")
        if self.path.exists() and any(os.path.samefile(self.path, parent)
                                     for parent in (self.parent, self.research_parent)):
            raise ValueError("supplement must not alias a historical ledger")
        self.authorization, self.gpu_uuid = authorization.strip(), gpu_uuid
        self.monotonic, self.clock, self.handle = monotonic, clock, None
        self.data = None

    def _parents(self):
        result = {}
        for name, path in (("optimization", self.parent), ("research", self.research_parent)):
            before = file_digest(path)
            value = read_json(path)
            if before != file_digest(path):
                raise BudgetStop("historical ledger changed while reading")
            if value.get("active") is not None:
                raise BudgetStop("historical allocation is still active")
            result[name] = dict(path=str(path), sha256=before,
                used_gpu_seconds=finite(value["used_gpu_seconds"], "historical usage"),
                limit_gpu_seconds=finite(value["limit_gpu_seconds"], "historical limit", positive=True))
        return result

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("supplement allocation already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(".lock")
        _ordinary_path(lock)
        self.handle = lock.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parents = self._parents()
            if self.path.exists():
                self.data = read_json(self.path)
                self._validate(parents)
            else:
                self.data = dict(schema_version=SCHEMA, limit_gpu_seconds=self.limit, budget_unlimited=self.budget_unlimited,
                    stage_limits_gpu_seconds={STAGE: self.limit}, used_gpu_seconds=0.,
                    stage_used_gpu_seconds={STAGE: 0.}, active=None, sessions=[],
                    authorization=self.authorization, gpu_uuid=self.gpu_uuid,
                    gpu_model_scope="single independently admitted A100", gpu_count=1,
                    parents=parents, parent=parents["optimization"],
                    research_parent=parents["research"], config_sha256=None,
                    accounting_scope="new authorization only; no historical unused allowance borrowed")
                self.save()
            if self.data["active"] is not None:
                raise BudgetStop("unsettled GPU session requires ownership and cleanup audit")
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *unused):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _require_lock(self):
        if self.handle is None:
            raise RuntimeError("supplement operation requires the allocation lock")

    def _validate(self, parents):
        data = self.data
        expected = dict(schema_version=SCHEMA, limit_gpu_seconds=self.limit,
            stage_limits_gpu_seconds={STAGE: self.limit}, authorization=self.authorization,
            gpu_uuid=self.gpu_uuid, gpu_count=1, parents=parents,
            parent=parents["optimization"], research_parent=parents["research"])
        if (type(data.get("budget_unlimited", False)) is not bool
                or data.get("budget_unlimited", False) != self.budget_unlimited
                or any(data.get(key) != value for key, value in expected.items())):
            raise ValueError("supplement authorization, device, or immutable parent differs")
        used = finite(data["used_gpu_seconds"], "supplement usage")
        if data.get("stage_used_gpu_seconds") != {STAGE: used}:
            raise ValueError("supplement stage accounting differs")
        sessions = data.get("sessions")
        if not isinstance(sessions, list):
            raise ValueError("supplement sessions must be a list")
        total, identities = 0., set()
        for row in sessions:
            if (row.get("stage") != STAGE or type(row.get("gpu_count")) is not int
                    or row["gpu_count"] != 1 or row.get("status") not in FAILURES | {"completed"}
                    or row.get("identity", {}).get("config_sha256") != data["config_sha256"]
                    or row["identity"].get("gpu_uuids") != [self.gpu_uuid]
                    or not row.get("session_id") or row["session_id"] in identities):
                raise ValueError("invalid settled supplement session")
            charge = finite(row["gpu_seconds"], "session charge")
            if charge != finite(row["elapsed_seconds"], "session elapsed"):
                raise ValueError("single-GPU charge must equal the complete elapsed lifetime")
            total += charge
            identities.add(row["session_id"])
        if not math.isclose(total, used, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("supplement total differs from settled sessions")
        if data.get("config_sha256") is not None:
            _sha256(data["config_sha256"], "fixed config_sha256")

    def save(self):
        self._require_lock()
        atomic_json(self.path, self.data)

    def remaining(self, stage):
        self._require_lock()
        if stage != STAGE:
            raise ValueError("supplement is authorized only for cache_k4")
        running = 0.
        if self.data["active"] is not None:
            running = finite(self.monotonic() - self.data["active"]["started_monotonic"],
                             "unsettled elapsed")
        # Infinity is an internal admission result, never serialized in JSON.
        return math.inf if self.budget_unlimited else max(0., self.limit - self.data["used_gpu_seconds"] - running)

    def _retry(self, identity):
        sessions = self.data["sessions"]
        retry = identity.get("retry_from")
        if not sessions:
            if retry is not None:
                raise ValueError("first supplement session cannot borrow an earlier allocation retry")
            return
        last = sessions[-1]
        if last["status"] not in FAILURES or not isinstance(retry, dict):
            raise ValueError("only an explicit charged same-config failure may be retried")
        path = _ordinary_path(retry.get("path", ""))
        if file_digest(path) != _sha256(retry.get("sha256"), "retry supervisor hash"):
            raise ValueError("retry supervisor hash differs")
        failure = read_json(path)
        if (failure.get("status") not in FAILURES or failure.get("phase") != STAGE
                or failure.get("config_sha256") != self.data["config_sha256"]
                or failure.get("charge") != last
                or failure.get("final_occupancy", {}).get("status") != "exclusive_observed"
                or failure["final_occupancy"].get("owned") != []
                or failure["final_occupancy"].get("foreign") != []):
            raise ValueError("retry requires the latest settled same-config failure and exclusive cleanup")

    def begin(self, stage, *, gpu_count, identity, minimum_seconds=120.):
        self._require_lock()
        if self.data["active"] is not None:
            raise BudgetStop("a charged supplement session is already active")
        if type(gpu_count) is not int or gpu_count != 1:
            raise ValueError("supplement admits exactly one A100")
        minimum = finite(minimum_seconds, "complete unit and cleanup minimum", positive=True)
        if self.remaining(stage) < minimum:
            raise BudgetStop("insufficient authorized supplement for complete unit and cleanup")
        if not isinstance(identity, dict) or identity.get("phase") != STAGE:
            raise ValueError("cache_k4 identity is required")
        config = _sha256(identity.get("config_sha256"), "config_sha256")
        if identity.get("gpu_uuids") != [self.gpu_uuid] or identity.get("auxiliary_encoder"):
            raise ValueError("supplement device differs or requests an auxiliary GPU")
        if self.data["config_sha256"] not in (None, config):
            raise ValueError("supplement retry configuration must not change")
        if self._parents() != self.data["parents"]:
            raise ValueError("immutable historical ledger changed before admission")
        self._retry(identity)
        self.data["config_sha256"] = config
        self.data["active"] = dict(stage=stage, gpu_count=1, identity=copy.deepcopy(identity),
            started_at=self.clock(), started_monotonic=self.monotonic(), session_id=str(uuid.uuid4()),
            supervisor_pid=os.getpid(), supervisor_pgid=os.getpgrp(), child_pgid=None)
        self.save()

    def child_started(self, pid):
        self._require_lock()
        if self.data["active"] is None or type(pid) is not int or pid <= 0:
            raise ValueError("invalid active child process group")
        if self.data["active"]["child_pgid"] is not None:
            raise ValueError("child process group already recorded")
        self.data["active"]["child_pgid"] = pid
        self.save()

    def finish(self, status):
        """Settle after cleanup; record overruns fully rather than truncate cost."""
        self._require_lock()
        active = self.data["active"]
        if active is None or status not in FAILURES | {"completed"}:
            raise ValueError("an active session and valid outcome are required")
        ended = self.monotonic()
        elapsed = ended - active["started_monotonic"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise BudgetStop("monotonic clock reversed; session remains unsettled")
        record = dict(active, ended_at=self.clock(), ended_monotonic=ended,
                      elapsed_seconds=elapsed, gpu_seconds=elapsed, status=status)
        # Parent invalidation cannot make admitted GPU time free. Persist the
        # complete charge and fail the session even when a parent changed.
        try:
            if self._parents() != self.data["parents"]:
                raise ValueError("immutable historical ledger changed")
        except (OSError, ValueError, BudgetStop) as error:
            record.update(status="failed", parent_validation_error=str(error))
        updated_total = self.data["used_gpu_seconds"] + elapsed
        if not math.isfinite(updated_total):
            raise BudgetStop("nonfinite total charge; session remains unsettled")
        self.data["used_gpu_seconds"] = updated_total
        self.data["stage_used_gpu_seconds"][STAGE] = updated_total
        record["authorized_overrun_gpu_seconds"] = (0. if self.budget_unlimited else
                                                    max(0., self.data["used_gpu_seconds"] - self.limit))
        self.data["sessions"].append(record)
        self.data["active"] = None
        self.save()
        return record


# Callers can retain the existing Allocation/BudgetGuard lifecycle vocabulary.
Allocation = SupplementAllocation
