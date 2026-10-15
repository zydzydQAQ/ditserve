"""Independent, explicitly authorized unlimited accounting for serving validation.

No ledger is created by import or construction. The supervisor enters the lock,
begins before loading, registers its child's process group, and calls ``finish``
only after confirming that group and its GPU allocations have been cleaned up.
An exception or context exit never settles or refunds an active session. This
module records ownership; it does not discover devices or kill processes.
"""

import copy
import fcntl
import math
import os
from pathlib import Path
import time
import uuid

from ..research.common import atomic_json, file_digest, finite, read_json
from .budget import BudgetGuard, BudgetStop


SCHEMA = "ditserve.optimizations.next_budget.v1"
FAILURES = frozenset(("failed", "incomplete", "budget_stopped", "blocked_contention"))
CONTINUATION_AUTHORIZATION = "就按照这个继续"
PRIOR_UNLIMITED_AUTHORIZATION = "取消8小时GPU小时硬上限，GPU无上限，开始在线对照"


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def _path(value):
    path = Path(value).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("budget evidence must not traverse symbolic links")
    if path.exists() and not path.is_file():
        raise ValueError("budget evidence must be an ordinary file")
    return path.resolve()


def _timestamp(value, name, nonnegative=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise BudgetStop(name + " must be finite" + (" and nonnegative" if nonnegative else ""))
    return float(value)


def _stable_read(path):
    before = file_digest(path)
    value = read_json(path)
    if before != file_digest(path):
        raise BudgetStop("bound evidence changed while reading")
    return value, before


class NextAllocation:
    """A100 lifetime accounting with an explicitly admitted conditional V100.

    ``stages`` fixes the allowed order. Untriggered stages may be skipped, but a
    later stage cannot be followed by an earlier one. A completed stage cannot
    be repeated; the most recent failed session of the current stage requires
    ``identity['retry_from'] = {'path': ..., 'sha256': ...}``, identifying its
    settled supervisor report with explicit empty owned/foreign cleanup lists.
    Configuration is fixed across the run; source snapshots may change between
    stages or explicitly retried attempts. Infinity is returned only in memory.
    ``auxiliary_gpu_uuid`` must come from the supervisor's verified frozen
    configuration. Only ``conditional`` may hold both devices, with explicit
    ``auxiliary_encoder=True`` and ordered UUIDs; its entire lifetime is billed
    twice. The unchanged legacy header's ``gpu_count=1`` describes the primary
    default. Each active/settled row's count and UUIDs govern actual accounting.
    Supply the same optional auxiliary UUID when reopening a ledger containing
    such a row; this class neither discovers nor admits an unverified device.
    """

    def __init__(self, path, *, parents, authorization_binding, gpu_uuid, stages,
                 auxiliary_gpu_uuid=None, monotonic=time.monotonic, clock=time.time):
        if (not isinstance(stages, tuple) or not stages or len(set(stages)) != len(stages)
                or any(not isinstance(stage, str) or not stage or not stage.replace("_", "").isalnum()
                       for stage in stages)):
            raise ValueError("stages must be a nonempty ordered tuple of distinct stage names")
        if not isinstance(parents, (list, tuple)) or not parents:
            raise ValueError("immutable historical parent ledgers are required")
        if not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-") or len(gpu_uuid) <= 4:
            raise ValueError("an independently admitted A100 UUID is required")
        if auxiliary_gpu_uuid is not None and (not isinstance(auxiliary_gpu_uuid, str)
                or not auxiliary_gpu_uuid.startswith("GPU-") or len(auxiliary_gpu_uuid) <= 4
                or auxiliary_gpu_uuid == gpu_uuid):
            raise ValueError("optional auxiliary GPU must be a distinct verified UUID")
        if not isinstance(authorization_binding, dict) or set(authorization_binding) != {"path", "sha256"}:
            raise ValueError("authorization_binding must contain path and sha256")
        auth_path = Path(authorization_binding["path"])
        if not auth_path.is_absolute():
            raise ValueError("authorization path must be absolute")
        self.path, self.parents = _path(path), tuple(_path(parent) for parent in parents)
        self.authorization_binding = dict(path=str(_path(auth_path)),
            sha256=_sha(authorization_binding["sha256"], "authorization SHA256"))
        paths = (self.path,) + self.parents + (Path(self.authorization_binding["path"]),)
        if len(set(paths)) != len(paths):
            raise ValueError("new ledger, historical parents, and authorization must be distinct")
        existing = [p for p in paths if p.exists()]
        if any(os.path.samefile(a, b) for i, a in enumerate(existing) for b in existing[i + 1:]):
            raise ValueError("budget evidence must not alias another bound file")
        self.gpu_uuid, self.stages = gpu_uuid, stages
        self.auxiliary_gpu_uuid = auxiliary_gpu_uuid
        self.monotonic, self.clock = monotonic, clock
        self.handle = self.data = self.owner_pid = None

    def _authorization(self):
        value, sha = _stable_read(Path(self.authorization_binding["path"]))
        instruction = value.get("user_instruction")
        if (sha != self.authorization_binding["sha256"] or value.get("approved_by_user") is not True
                or value.get("budget_unlimited") is not True or "authorized_gpu_seconds" not in value
                or value["authorized_gpu_seconds"] is not None or value.get("scope") != "serving_next_validation"
                or not isinstance(instruction, str) or CONTINUATION_AUTHORIZATION not in instruction
                or PRIOR_UNLIMITED_AUTHORIZATION not in instruction):
            raise ValueError("current continuation and prior unlimited authorization must be explicitly bound")
        return instruction

    def _parents(self):
        bindings = {}
        for path in self.parents:
            value, sha = _stable_read(path)
            if "active" not in value:
                raise ValueError("historical ledger lacks explicit active state")
            if value["active"] is not None:
                raise BudgetStop("historical allocation is still active")
            limit = value["limit_gpu_seconds"]
            unlimited = value.get("budget_unlimited", False)
            if type(unlimited) is not bool or (limit is None) is not unlimited:
                raise ValueError("historical finite/unlimited limit is inconsistent")
            bindings[str(path)] = dict(path=str(path), sha256=sha,
                used_gpu_seconds=finite(value["used_gpu_seconds"], "historical usage"),
                limit_gpu_seconds=None if unlimited else finite(limit, "historical limit", positive=True),
                budget_unlimited=unlimited)
        return bindings

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("allocation already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = _path(self.path.with_suffix(".lock"))
        if lock in self.parents or str(lock) == self.authorization_binding["path"]:
            raise ValueError("lock file aliases bound evidence")
        self.handle = lock.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.owner_pid = os.getpid()
            authorization, parents = self._authorization(), self._parents()
            if self.path.exists():
                self.data = read_json(self.path)
                # Never attempt recovery or clear ownership merely on reopen.
                if self.data.get("active") is not None:
                    raise BudgetStop("unsettled GPU session requires ownership and cleanup audit")
                self._validate(parents, authorization)
            else:
                self.data = dict(schema_version=SCHEMA, budget_unlimited=True, limit_gpu_seconds=None,
                    stages=list(self.stages), stage_limits_gpu_seconds={s: None for s in self.stages},
                    used_gpu_seconds=0., stage_used_gpu_seconds={s: 0. for s in self.stages},
                    config_sha256=None, active=None, sessions=[], gpu_count=1, gpu_uuid=self.gpu_uuid,
                    gpu_model_scope="single independently admitted A100", parents=parents,
                    authorization=authorization, authorization_binding=copy.deepcopy(self.authorization_binding),
                    accounting_scope="new authorization; all actual holding time charged; immutable historical ledgers")
                self.save()
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *unused):
        if self.handle is not None:
            self.handle.close()
        self.handle = self.owner_pid = None

    def _require_lock(self):
        if self.handle is None or self.owner_pid != os.getpid():
            raise RuntimeError("operation requires the owning supervisor's allocation lock")

    def _stage(self, stage):
        if stage not in self.stages:
            raise ValueError("stage is outside the authorized sequence")
        return self.stages.index(stage)

    def _identity(self, stage, identity, config):
        if not isinstance(identity, dict) or identity.get("phase") != stage:
            raise ValueError("phase identity differs from the charged stage")
        actual_config = _sha(identity.get("config_sha256"), "config_sha256")
        _sha(identity.get("source_sha256"), "source_sha256")
        if config not in (None, actual_config):
            raise ValueError("configuration must remain fixed across the validation run")
        self._gpu_count(stage, identity)
        if identity.get("budget_unlimited", True) is not True:
            raise ValueError("identity must preserve the unlimited authorization")
        for key in ("output", "run_root"):
            if not isinstance(identity.get(key), str) or not Path(identity[key]).is_absolute():
                raise ValueError(key + " must be an absolute path")
        output, run_root = Path(identity["output"]).resolve(), Path(identity["run_root"]).resolve()
        if run_root not in output.parents:
            raise ValueError("attempt output must lie within its run root")
        auth = identity.get("authorization_source", self.authorization_binding)
        if auth != self.authorization_binding:
            raise ValueError("identity authorization binding differs")
        return actual_config

    def _gpu_count(self, stage, identity):
        auxiliary = identity.get("auxiliary_encoder", False)
        if auxiliary is False and identity.get("gpu_uuids") == [self.gpu_uuid]:
            return 1
        if (stage == "conditional" and auxiliary is True and self.auxiliary_gpu_uuid is not None
                and identity.get("gpu_uuids") == [self.gpu_uuid, self.auxiliary_gpu_uuid]):
            return 2
        raise ValueError("only conditional may explicitly hold the admitted A100 and auxiliary encoder GPU")

    def _validate(self, parents, authorization):
        data = self.data
        expected = dict(schema_version=SCHEMA, budget_unlimited=True, limit_gpu_seconds=None,
            stages=list(self.stages), stage_limits_gpu_seconds={s: None for s in self.stages},
            gpu_uuid=self.gpu_uuid, gpu_count=1, parents=parents, authorization=authorization,
            authorization_binding=self.authorization_binding)
        if data.get("budget_unlimited") is not True or any(data.get(k) != v for k, v in expected.items()):
            raise ValueError("allocation authorization, stages, GPU, or historical parent differs")
        if "active" not in data or not isinstance(data.get("sessions"), list):
            raise ValueError("allocation lacks explicit session state")
        config = data.get("config_sha256")
        if config is not None:
            _sha(config, "fixed config_sha256")
        totals, session_ids, completed, last_index = {s: 0. for s in self.stages}, set(), set(), -1
        previous = None
        for row in data["sessions"]:
            stage = row.get("stage")
            index = self._stage(stage)
            self._identity(stage, row.get("identity"), config)
            count = self._gpu_count(stage, row["identity"])
            if (index < last_index or stage in completed or type(row.get("gpu_count")) is not int
                    or row["gpu_count"] != count or row.get("status") not in FAILURES | {"completed"}
                    or not isinstance(row.get("session_id"), str) or not row["session_id"]
                    or row["session_id"] in session_ids):
                raise ValueError("invalid ordered settled session identity")
            if previous is not None and stage == previous["stage"]:
                if previous["status"] not in FAILURES or not isinstance(row["identity"].get("retry_from"), dict):
                    raise ValueError("same-stage session lacks explicit failure retry")
            elif row["identity"].get("retry_from") is not None:
                raise ValueError("new stage cannot import another stage's retry")
            elapsed = finite(row["elapsed_seconds"], "settled elapsed")
            start = _timestamp(row["started_monotonic"], "settled start", nonnegative=True)
            end = _timestamp(row["ended_monotonic"], "settled end", nonnegative=True)
            _timestamp(row["started_at"], "settled wall start")
            _timestamp(row["ended_at"], "settled wall end")
            charge = finite(row["gpu_seconds"], "settled charge")
            if (elapsed * count != charge or elapsed != end - start
                    or type(row.get("supervisor_pid")) is not int or row["supervisor_pid"] <= 0
                    or type(row.get("supervisor_pgid")) is not int or row["supervisor_pgid"] <= 0
                    or (row.get("child_pgid") is not None and
                        (type(row["child_pgid"]) is not int or row["child_pgid"] <= 0))
                    or (row["status"] == "completed" and row.get("child_pgid") is None)):
                raise ValueError("settled time or process-group ownership differs")
            totals[stage] += charge
            session_ids.add(row["session_id"])
            if row["status"] == "completed":
                completed.add(stage)
            previous, last_index = row, index
        if data["sessions"] and config is None:
            raise ValueError("settled run has no fixed configuration")
        if set(data.get("stage_used_gpu_seconds", {})) != set(self.stages):
            raise ValueError("stage accounting keys differ")
        for stage, total in totals.items():
            stored = finite(data["stage_used_gpu_seconds"][stage], "stage total")
            if not math.isfinite(total) or not math.isclose(stored, total, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("stage usage differs from settled sessions")
        used = finite(data["used_gpu_seconds"], "run usage")
        if not math.isclose(used, sum(totals.values()), rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("run usage differs from settled sessions")

    def save(self):
        self._require_lock()
        atomic_json(self.path, self.data)

    def remaining(self, stage):
        self._require_lock()
        self._stage(stage)
        if self.data["active"] is not None:
            now = _timestamp(self.monotonic(), "active monotonic time", nonnegative=True)
            finite(now - self.data["active"]["started_monotonic"], "active elapsed")
        return math.inf

    def _retry(self, stage, identity):
        sessions, retry = self.data["sessions"], identity.get("retry_from")
        if not sessions or stage != sessions[-1]["stage"]:
            if retry is not None:
                raise ValueError("first session of a stage cannot import an earlier retry")
            return
        last = sessions[-1]
        if last["status"] not in FAILURES or not isinstance(retry, dict):
            raise ValueError("same stage requires an explicit latest settled failure retry")
        if not isinstance(retry.get("path"), str) or not Path(retry["path"]).is_absolute():
            raise ValueError("retry supervisor path must be absolute")
        path = _path(retry["path"])
        failure, sha = _stable_read(path)
        if sha != _sha(retry.get("sha256"), "retry supervisor SHA256"):
            raise ValueError("retry supervisor hash differs")
        final = failure.get("final_occupancy", {})
        if (failure.get("status") not in FAILURES or failure.get("phase") != stage
                or failure.get("config_sha256") != self.data["config_sha256"] or failure.get("charge") != last
                or final.get("status") != "exclusive_observed" or final.get("owned") != [] or final.get("foreign") != []):
            raise ValueError("retry requires the latest same-config failure with explicit exclusive cleanup")
        if identity["output"] == last["identity"]["output"]:
            raise ValueError("retry requires a new attempt output")

    def begin(self, stage, *, gpu_count, identity, minimum_seconds=120.):
        self._require_lock()
        if self.data["active"] is not None:
            raise BudgetStop("a charged validation session is already active")
        index = self._stage(stage)
        finite(minimum_seconds, "complete unit and cleanup minimum", positive=True)
        config = self._identity(stage, identity, self.data["config_sha256"])
        if type(gpu_count) is not int or gpu_count != self._gpu_count(stage, identity):
            raise ValueError("charged GPU count must match the admitted session devices")
        if self.data["sessions"] and index < self._stage(self.data["sessions"][-1]["stage"]):
            raise ValueError("validation stages may only advance")
        if self._parents() != self.data["parents"] or self._authorization() != self.data["authorization"]:
            raise ValueError("bound historical ledger or authorization changed before admission")
        self._retry(stage, identity)
        start = _timestamp(self.monotonic(), "session start", nonnegative=True)
        wall = _timestamp(self.clock(), "session wall start")
        self.data["config_sha256"] = config
        self.data["active"] = dict(stage=stage, gpu_count=gpu_count, identity=copy.deepcopy(identity),
            started_at=wall, started_monotonic=start, session_id=str(uuid.uuid4()),
            supervisor_pid=os.getpid(), supervisor_pgid=os.getpgrp(), child_pgid=None)
        self.save()

    def child_started(self, pid):
        self._require_lock()
        active = self.data["active"]
        if (active is None or type(pid) is not int or pid <= 0 or pid == active["supervisor_pid"]
                or pid == active["supervisor_pgid"] or active["child_pgid"] is not None):
            raise ValueError("invalid or already registered owned child process group")
        active["child_pgid"] = pid
        self.save()

    def finish(self, status):
        """Caller has confirmed cleanup; settle all elapsed time, even on failure."""
        self._require_lock()
        active = self.data["active"]
        if active is None or status not in FAILURES | {"completed"}:
            raise ValueError("an active session and valid settlement status are required")
        if status == "completed" and active["child_pgid"] is None:
            raise ValueError("completed work requires a registered owned process group")
        end = _timestamp(self.monotonic(), "session end", nonnegative=True)
        wall = _timestamp(self.clock(), "session wall end")
        elapsed = end - active["started_monotonic"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise BudgetStop("monotonic time reversed; session remains active")
        count = self._gpu_count(active["stage"], active["identity"])
        if type(active.get("gpu_count")) is not int or active["gpu_count"] != count:
            raise ValueError("active session count and admitted device identity differ")
        charged = elapsed * count
        if not math.isfinite(charged):
            raise BudgetStop("nonfinite GPU charge; session remains active")
        record = dict(active, ended_at=wall, ended_monotonic=end, elapsed_seconds=elapsed,
                      gpu_seconds=charged, status=status)
        for name, operation, expected in (("parent", self._parents, self.data["parents"]),
                                         ("authorization", self._authorization, self.data["authorization"])):
            try:
                if operation() != expected:
                    raise ValueError("immutable " + name + " binding changed")
            except (OSError, ValueError, KeyError, TypeError, BudgetStop) as error:
                record.update(status="failed")
                record[name + "_validation_error"] = str(error)
        total = self.data["used_gpu_seconds"] + charged
        stage_total = self.data["stage_used_gpu_seconds"][active["stage"]] + charged
        if not math.isfinite(total) or not math.isfinite(stage_total):
            raise BudgetStop("nonfinite total charge; session remains active")
        self.data["used_gpu_seconds"] = total
        self.data["stage_used_gpu_seconds"][active["stage"]] = stage_total
        self.data["sessions"].append(record)
        self.data["active"] = None
        self.save()
        return record


Allocation = NextAllocation
