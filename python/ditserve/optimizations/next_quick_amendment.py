"""An explicit, source-snapshotted reduction of the unstarted load-.9 matrix.

The original config, workloads, charges and completed phases remain immutable.
Only the named next worker consumes this amendment; the full matrix remains
unqualified and the single-seed screening has its own evidence and report.
"""
from pathlib import Path
from types import SimpleNamespace

from ..research.common import atomic_json, digest, file_digest, read_json


RELATIVE_PATH = "configs/serving_next_quick_load09.json"
INSTRUCTION = "做完0.7的用1个seed快速验证"


def admit_context_amendment(context, stage_name):
    if stage_name != "cache_load_09" or not hasattr(context, "run_root") or not hasattr(context, "ledger"):
        return None
    from .protocol import ROOT
    if not (ROOT / RELATIVE_PATH).exists():
        return None
    args = SimpleNamespace(phase=stage_name, run_root=context.run_root,
        ledger=context.ledger, output=context.output)
    active = read_json(context.ledger).get("active")
    if not active or active.get("stage") != stage_name:
        raise ValueError("quick screening lacks its admitted cache-load session")
    receipt = admit_quick_amendment(args, context.config, active, ROOT)
    context.metadata["quick_amendment"] = receipt
    return receipt


def admit_quick_amendment(args, config, active, execution_root):
    if args.phase != "cache_load_09":
        return None
    root = Path(execution_root).absolute()
    path = root / RELATIVE_PATH
    if not path.exists():
        return None
    if path.resolve() != path or path.is_symlink():
        raise ValueError("quick amendment must be a regular snapshot file")
    note = read_json(path)
    expected = dict(schema_version="ditserve.optimizations.quick_amendment.v1",
        approved_by_user=True, user_instruction=INSTRUCTION, phase="cache_load_09",
        load=.9, seed=101, patterns=["ordinary", "burst"],
        orders=[["baseline", "candidate"], ["candidate", "baseline"]],
        planned_replays=8, planned_requests=192, full_qualification=False)
    if any(note.get(key) != value for key, value in expected.items()):
        raise ValueError("quick amendment scientific scope or authorization differs")
    if type(note.get("seed")) is not int or note.get("full_qualification") is not False:
        raise ValueError("quick amendment cannot grant full qualification")
    identity = active["identity"]
    run_root = Path(args.run_root).absolute()
    if (note.get("run_root") != str(run_root) or note.get("ledger") != str(Path(args.ledger).absolute())
            or note.get("base_config_sha256") != digest(config)
            or note.get("base_config_sha256") != identity.get("config_sha256")
            or note.get("workloads_sha256") != identity.get("workloads_sha256")
            or note.get("authorization_binding") != identity.get("authorization_source")):
        raise ValueError("quick amendment does not match this admitted run")
    manifest = read_json(root / "manifest.json")
    binding = dict(path=str(path), sha256=file_digest(path))
    if (manifest.get("source_files", {}).get(RELATIVE_PATH) != binding["sha256"]
            or manifest.get("source_sha256") != identity.get("source_sha256")):
        raise ValueError("quick amendment is not bound to the executable snapshot")
    auth = note["authorization_binding"]
    if file_digest(auth["path"]) != auth["sha256"]:
        raise ValueError("parent authorization changed")
    prior_path = run_root / "cache_load_07/terminal.json"
    prior = read_json(prior_path)
    charge = prior.get("charge", {})
    budget = read_json(args.ledger)
    settled = [row for row in budget.get("sessions", [])
               if row.get("session_id") == note.get("after_session_id")]
    if (prior.get("status") != "completed" or prior.get("returncode") != 0
            or prior.get("supervisor_failure") or prior.get("recorded_rows") != 24
            or prior.get("evaluation_complete") is not True
            or charge.get("status") != "completed" or charge.get("stage") != "cache_load_07"
            or charge.get("session_id") != note.get("after_session_id") or settled != [charge]
            or (budget.get("active") or {}).get("session_id") != active["session_id"]
            or prior.get("config_sha256") != identity.get("config_sha256")):
        raise ValueError("load .7 has not completed its original settled 24-replay matrix")
    final = prior.get("final_occupancy", {})
    if final.get("status") != "exclusive_observed" or final.get("owned") != [] or final.get("foreign") != []:
        raise ValueError("load .7 cleanup was not confirmed")
    receipt = dict(status="admitted", amendment_binding=binding, seed=101,
        protocol_mode="single_seed_screening", planned_replays=8, planned_requests=192,
        preceding_terminal=dict(path=str(prior_path), sha256=file_digest(prior_path)),
        session_id=active["session_id"], full_qualification=False,
        scope="explicit user amendment; preserve original 24-replay protocol as unverified at load .9")
    atomic_json(Path(args.output) / "quick-amendment-admission.json", receipt)
    return receipt
