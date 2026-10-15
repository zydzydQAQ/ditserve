"""Seal independent complete measurement units; never rehabilitate old runs."""

import contextlib
import json
import os
from pathlib import Path
import time

from .common import atomic_json, digest, finite, read_json
from .isolation import GPUIsolationError, gpu_occupancy


INVALIDATIONS_FILE = "parent-isolation-invalidations.jsonl"
INVALIDATION_SCHEMA = "ditserve.dual.isolation_invalidation.v1"


class UnitJournal:
    def __init__(self, root, frozen, devices, observe=gpu_occupancy):
        self.root, self.frozen, self.devices = Path(root), frozen, devices
        self.observe = observe
        self.active = None

    def append(self, record):
        if self.active is None:
            raise RuntimeError("records require an active complete unit")
        row = dict(record, unit_id=self.active["unit_id"], protocol_sha256=self.frozen["sha256"],
                   source_sha256=self.frozen["source_sha256"], recorded_at=time.time())
        row["sha256"] = digest(row)
        with (self.root / "records.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.active["records"].append(row["sha256"])
        return row

    def observation(self, phase):
        value = self.observe(self.devices, allowed_pgid=os.getpgrp())
        value.pop("sha256", None)
        value.update(phase=phase, unit_id=self.active["unit_id"], protocol_sha256=self.frozen["sha256"],
                     source_sha256=self.frozen["source_sha256"])
        value["sha256"] = digest(value)
        with (self.root / "unit-isolation.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
        self.active["observations"].append(value)
        if value["status"] != "exclusive_observed":
            raise GPUIsolationError("active paired unit lost GPU isolation")

    @contextlib.contextmanager
    def unit(self, unit_id, estimate_seconds, deadline):
        if self.active is not None or not _flat_unit_id(unit_id):
            raise ValueError("unique flat complete unit ID required")
        estimate_seconds = finite(estimate_seconds, "complete unit estimate", positive=True)
        deadline = finite(deadline, "unit monotonic deadline", positive=True)
        path = self.root / "units" / (unit_id + ".json")
        if path.exists():
            raise ValueError("sealed unit IDs cannot be reused")
        if time.monotonic() + estimate_seconds >= deadline:
            from .budget import BudgetStop
            raise BudgetStop("next full paired unit exceeds deadline")
        self.active = dict(schema_version="ditserve.dual.unit.v1", unit_id=unit_id,
                           protocol_sha256=self.frozen["sha256"], source_sha256=self.frozen["source_sha256"],
                           started_at=time.time(), records=[], observations=[], status="running")
        atomic_json(self.root / "active-unit.json", self.active)
        try:
            self.observation("start")
            yield self
            self.observation("completed")
            self.active.update(status="completed", performance_valid=True)
        except BaseException as error:
            self.active.update(status="failed", performance_valid=False,
                               error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            self.active["ended_at"] = time.time()
            self.active["sha256"] = digest(self.active)
            atomic_json(path, self.active)
            atomic_json(self.root / "active-unit.json", dict(status="idle", last_unit=unit_id))
            self.active = None


def invalidate_pending(root, reason):
    """Supervisor cleanup for SIGKILL; completed unit files remain immutable."""
    root = Path(root)
    active = read_json(root / "active-unit.json") if (root / "active-unit.json").exists() else {}
    if active.get("status") != "running":
        return
    if not _flat_unit_id(active.get("unit_id")):
        raise ValueError("invalid pending unit ID")
    path = root / "units" / (active["unit_id"] + ".json")
    if not path.exists():
        active.update(status="failed", performance_valid=False, error=reason, ended_at=time.time())
        active["sha256"] = digest(active)
        atomic_json(path, active)
    atomic_json(root / "active-unit.json", dict(status="idle", last_unit=active["unit_id"]))


def _invalidation_windows(root, frozen):
    path = Path(root) / INVALIDATIONS_FILE
    windows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        record = json.loads(line)
        if (record.get("schema_version") != INVALIDATION_SCHEMA
                or record.get("sha256") != digest({k: v for k, v in record.items() if k != "sha256"})
                or record.get("protocol_sha256") != frozen["sha256"]
                or record.get("source_sha256") != frozen["source_sha256"]):
            raise ValueError("parent isolation invalidation is not bound to this protocol/source")
        start = finite(record.get("started_at"), "parent observation start")
        end = finite(record.get("ended_at"), "parent observation end")
        if end < start or not isinstance(record.get("reason"), str) or not record["reason"]:
            raise ValueError("invalid parent isolation observation window")
        windows.append(record)
    return windows


def invalidate_observation(root, frozen, started_at, ended_at, reason):
    """Append a parent query's wall-clock interval without rewriting unit seals.

    Timestamps use time.time(), matching unit started_at/ended_at. Any unit
    overlapping this inclusive query interval loses cost eligibility, even if
    it completed before the parent received the GPU inventory or cleaned up.
    Only the ledger-owning supervisor appends this sidecar.
    """
    start = finite(started_at, "parent observation start")
    end = finite(ended_at, "parent observation end")
    if end < start or not isinstance(reason, str) or not reason:
        raise ValueError("invalid parent isolation observation window")
    _invalidation_windows(root, frozen)
    record = dict(schema_version=INVALIDATION_SCHEMA, protocol_sha256=frozen["sha256"],
                  source_sha256=frozen["source_sha256"], started_at=start, ended_at=end,
                  reason=reason, recorded_at=time.time())
    record["sha256"] = digest(record)
    path = Path(root) / INVALIDATIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _overlaps_invalidation(unit, windows):
    if not windows:
        return False
    try:
        start = finite(unit.get("started_at"), "unit start")
        end = finite(unit.get("ended_at"), "unit end")
    except ValueError:
        return True  # Missing timing cannot prove the unit predates contention.
    return end < start or any(start <= row["ended_at"] and end >= row["started_at"] for row in windows)


def _flat_unit_id(value):
    return isinstance(value, str) and value not in ("", ".", "..") and "/" not in value and "\\" not in value


def _valid_observations(unit, frozen):
    rows = unit.get("observations", [])
    return (isinstance(rows, list) and len(rows) >= 2
            and rows[0].get("phase") == "start" and rows[-1].get("phase") == "completed"
            and all(row.get("sha256") == digest({k: v for k, v in row.items() if k != "sha256"})
                    and row.get("status") == "exclusive_observed"
                    and row.get("unit_id") == unit["unit_id"]
                    and row.get("protocol_sha256") == frozen["sha256"]
                    and row.get("source_sha256") == frozen["source_sha256"] for row in rows))


def read_records(root, frozen):
    root = Path(root)
    invalidations = _invalidation_windows(root, frozen)
    path = root / "records.jsonl"
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"}):
            raise ValueError("record digest mismatch")
        if row.get("protocol_sha256") != frozen["sha256"] or row.get("source_sha256") != frozen["source_sha256"]:
            raise ValueError("record uses a different protocol/source")
        if not _flat_unit_id(row.get("unit_id")):
            raise ValueError("invalid record unit ID")
        unit_path = root / "units" / (row["unit_id"] + ".json")
        unit = read_json(unit_path) if unit_path.exists() else {}
        valid = (unit.get("sha256") == digest({k: v for k, v in unit.items() if k != "sha256"})
                 and row["sha256"] in unit.get("records", []) and unit.get("status") == "completed"
                 and unit.get("unit_id") == row["unit_id"]
                 and unit.get("protocol_sha256") == frozen["sha256"]
                 and unit.get("source_sha256") == frozen["source_sha256"]
                 and unit.get("performance_valid") is True and _valid_observations(unit, frozen)
                 and not _overlaps_invalidation(unit, invalidations))
        rows.append(dict(row, performance_valid=valid and row.get("performance_valid", True) is True))
    return rows
