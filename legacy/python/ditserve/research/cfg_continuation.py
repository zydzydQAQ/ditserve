"""Verified prototype evidence authorizes implementation work, never serving.

A new source snapshot may consume a completed oracle screen as its development
continuation decision. All development, holdout and migration qualifications
remain bound to that new snapshot's own trajectory records.
"""

import copy
from pathlib import Path

from .common import digest, read_json
from .dual_evidence import read_records


SCHEMA = "ditserve.research.cfg.prototype_continuation.v1"
FROZEN_SCHEMA = "ditserve.research.cfg.frozen.v1"
PURPOSE = "implementation_continuation_only"
_RUNTIME_FIELDS = frozenset(("estimate_wall_seconds", "timeout_wall_seconds", "cleanup_wall_seconds"))


def _sealed(value):
    return isinstance(value, dict) and value.get("sha256") == digest(
        {key: child for key, child in value.items() if key != "sha256"})


def _recipe(config):
    return {key: value for key, value in config.items() if key not in _RUNTIME_FIELDS}


def build_continuation(prototype_run, *, config, ledger):
    """Read a closed direct prototype and recompute both finite screening gates.

    No model weights or raw output arrays are read. The historical metrics and
    unit seals are sufficient for the limited decision to continue development;
    they are not a new raw-data quality evaluation.
    """
    from .dual_protocol import verify_protocol
    from .cfg_matrix import CANDIDATES, stage_a_gate

    root = Path(prototype_run).resolve()
    path = root / "protocol.json"
    preliminary = read_json(path)
    if preliminary.get("schema_version") != FROZEN_SCHEMA or "prototype_continuation" in preliminary:
        raise ValueError("continuation requires a direct CFG oracle prototype, not a chained decision")
    parent = verify_protocol(path, verify_current=False)
    if _recipe(parent["config"]) != _recipe(config):
        raise ValueError("prototype and new run differ in the fixed scientific recipe")
    if Path(parent["ledger"]).resolve() != Path(ledger).resolve():
        raise ValueError("prototype continuation cannot change the lifetime ledger")
    active = root / "active-unit.json"
    if active.exists() and read_json(active).get("status") == "running":
        raise ValueError("prototype still has an active paired unit")
    probe = read_json(root / "probe.json")
    if (not _sealed(probe) or probe.get("status") != "completed"
            or probe.get("protocol_sha256") != parent["sha256"]
            or probe.get("source_sha256") != parent["source_sha256"]
            or not isinstance(probe.get("context_sha256"), str)):
        raise ValueError("completed bound prototype screening decision required")
    for role in ("a100", "v100"):
        attempt = read_json(root / ("screen-" + role + "-attempt.json"))
        if (not _sealed(attempt) or attempt.get("status") != "completed"
                or attempt.get("returncode") != 0 or attempt.get("protocol_sha256") != parent["sha256"]):
            raise ValueError("prototype screening supervisor did not finish successfully")
    rows = read_records(root, parent)
    context = probe["context_sha256"]
    gates = {candidate: stage_a_gate(rows, parent["config"], candidate, context) for candidate in CANDIDATES}
    if (gates != probe.get("stage_a") or any(gate["status"] not in ("passed", "failed") for gate in gates.values())
            or gates["paired_008"]["eligible"] is not True):
        raise ValueError("complete sealed prototype gates with eligible paired oracle required")
    screen = [row for row in rows if row.get("type") == "cfg_screen" and row.get("phase") == "stage_a"
              and row.get("recipe_id") in CANDIDATES]
    references = []
    for row in screen:
        unit = read_json(root / "units" / (row["unit_id"] + ".json"))
        references.append(dict(record_sha256=row["sha256"], unit_id=row["unit_id"], unit_sha256=unit["sha256"]))
    references.sort(key=lambda value: (value["unit_id"], value["record_sha256"]))
    value = dict(schema_version=SCHEMA, parent_run=str(root), parent_protocol_sha256=parent["sha256"],
                 parent_source_sha256=parent["source_sha256"], parent_probe_sha256=probe["sha256"],
                 parent_context_sha256=context, screen_records=references, stage_a=gates,
                 purpose=PURPOSE, quality_qualified=False, performance_qualified=False,
                 evidence_scope="previously recorded prototype metrics; no raw-data reevaluation",
                 target_implementations=dict(independent_004="native", paired_008="split_prefix"))
    value["sha256"] = digest(value)
    return value


def verify_continuation(value, *, config, ledger):
    if (not _sealed(value) or value.get("schema_version") != SCHEMA
            or value.get("purpose") != PURPOSE
            or value.get("quality_qualified") is not False or value.get("performance_qualified") is not False
            or not isinstance(value.get("parent_run"), str) or not Path(value["parent_run"]).is_absolute()):
        raise ValueError("invalid prototype continuation proof")
    expected = build_continuation(value["parent_run"], config=config, ledger=ledger)
    if value != expected:
        raise ValueError("prototype continuation proof differs from its sealed parent evidence")
    return copy.deepcopy(expected)


def screening_decisions(frozen, rows, context_sha256):
    """Return trusted screening gates from local rows or a verified parent proof.

    A parent decision affects which candidates are implemented and remeasured.
    It does not change the context supplied to the development/holdout gates.
    """
    from .cfg_matrix import CANDIDATES, stage_a_gate

    if not _sealed(frozen) or frozen.get("schema_version") != FROZEN_SCHEMA:
        raise ValueError("screening decisions require a checksum-bound CFG frozen protocol")
    config = frozen["config"]
    value = frozen.get("prototype_continuation")
    if value is None:
        return {candidate: stage_a_gate(rows, config, candidate, context_sha256) for candidate in CANDIDATES}
    proof = verify_continuation(value, config=config, ledger=frozen["ledger"])
    # Local screen rows must not silently compete with a frozen parent decision.
    if any(row.get("type") == "cfg_screen" and row.get("phase") == "stage_a" for row in rows):
        raise ValueError("continued implementation run cannot replace its frozen prototype screening decision")
    return copy.deepcopy(proof["stage_a"])
