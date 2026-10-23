"""Bound acceptance of the repaired Wan backend, without qualifying old SP jobs."""

from . import workflow_next_study as api
from .workflow_elastic_validation import normalize_config
from .workflow_next_report import current_job, evidence


POLICY = "full_rows_except_ffn1"
STAGES = ("transfer", "pause", "resize")
PREFIX = "elastic-repaired-"
SELECTION = "sp2_acceptance"


def configuration(stage):
    if stage not in STAGES:
        raise ValueError("unsupported repaired SP acceptance stage")
    return normalize_config(dict(
        stage=stage, sp_linear_policy=POLICY, model_path="/models/wan2.1-t2v-1.3b",
        devices=["cuda:0", "cuda:1"], frames=[17, 33], seeds=[3501, 3502], steps=20,
        schedule=[dict(step=5, degree=2), dict(step=10, degree=1)],
        threshold=.05, group_timeout_seconds=60, timeout_seconds=1400,
        pause_blocks=[23, 14, 23], cpu_threads=8,
        cpu_masks={"cuda:0": list(range(8)), "cuda:1": list(range(60, 68))}))


def _bound_input(job):
    config = api.read(job["input"])
    if config is None or api.digest(config) != job["configuration_sha256"]:
        raise ValueError("frozen input contents changed: " + job["id"])
    for expected in job["input_bindings"]:
        if api.binding(expected["path"]) != expected:
            raise ValueError("frozen dependency contents changed: " + job["id"])
    return config


def _bound_output(job, ev):
    bound = api.binding(ev["path"])
    if bound["sha256"] != ev["launch"].get("output_sha256"):
        raise ValueError("acceptance evidence changed while binding: " + job["id"])
    return bound


def _qualified(ev):
    return all(ev.get(key) is True for key in ("guarded", "qualified", "capacity"))


def _record(study, status, completed, *, blocked_at=None, reason=None, ev=None):
    result = dict(status=status, sp_linear_policy=POLICY, source=study["source"],
        source_digest=study["source_digest"], completed_jobs=list(completed),
        stages=[PREFIX + stage for stage in STAGES],
        unqualified_jobs=[PREFIX + stage for stage in STAGES if PREFIX + stage not in completed],
        scope="Two-seed 17/33-frame actual-backend transfer, pause and 1->2->1 correctness",
        performance_evidence=False, semantic="unverified",
        sana_interleave="unverified", online_scheduling="unverified",
        historical_sp_diagnostic_unchanged=True)
    if blocked_at is not None:
        result.update(blocked_at=blocked_at, reason=reason)
        if ev is not None:
            result["observed_gate"] = {key: ev.get(key) for key in ("guarded", "qualified", "capacity")}
            result["observed_gate"]["result_status"] = ev.get("result", {}).get("status")
        decision = dict(job=blocked_at, decision="repaired_sp_acceptance_stopped", reason=reason)
        if decision not in study["decisions"]:
            study["decisions"].append(decision)
    study["selections"][SELECTION] = result
    api.save(study)
    return result


def _register(study, stage, provenance, dependencies):
    identifier, config = PREFIX + stage, configuration(stage)
    requires = ["acceptance"] + ([PREFIX + STAGES[STAGES.index(stage) - 1]] if stage != "transfer" else [])
    metadata = dict(role="actual_backend_sp2_acceptance", stage=stage, sp_linear_policy=POLICY,
        performance_evidence=False, comparison="repaired_sp_acceptance",
        diagnostic_provenance=provenance,
        gate_scope="Actual WanBackend; conservative diagnostic alone cannot qualify this chain")
    # add_job checks frozen configurations but deliberately permits new controller
    # revisions elsewhere. This particular chain requires one backend revision.
    existing = [job for job in study["jobs"] if job.get("logical_id", job["id"]) == identifier
                and not job.get("superseded")]
    for job in existing:
        if (job["source"] != study["source"] or job["source_digest"] != study["source_digest"]
                or job.get("metadata") != metadata
                or job.get("estimated_seconds") != 600 or job.get("timeout_seconds") != 1500):
            raise ValueError("cannot change repaired acceptance backend or provenance: " + identifier)
        _bound_input(job)
    job = api.add_job(study, identifier, "sp", config,
        module="workflow_elastic_validation", devices=(0, 1), requires=requires,
        estimate=600, timeout=1500, metadata=metadata)
    expected_command = ["/opt/venv/bin/python", "-m", "ditserve.evaluation.workflow_elastic_validation",
                        "--config", api.container_path(study, job["input"]),
                        "--out", api.container_path(study, job["output"])]
    if job["command"] != expected_command:
        raise ValueError("repaired acceptance command differs from the frozen backend runner")
    expected_bindings = [api.binding(job["input"])]
    for bound in dependencies:
        if bound not in expected_bindings:
            expected_bindings.append(bound)
    if existing:
        if job["input_bindings"] != expected_bindings:
            raise ValueError("cannot change repaired acceptance dependency bindings: " + identifier)
    else:
        job["input_bindings"] = expected_bindings
        api.save(study)
    return job


def run_followup(study, run_job):
    """Use the existing exclusive launcher, family cap and cumulative budget."""
    completed = []
    acceptance = current_job(study, "acceptance")
    _bound_input(acceptance)
    accepted = evidence(study, acceptance)
    if not _qualified(accepted):
        return _record(study, "blocked", completed, blocked_at=PREFIX + "transfer",
                       reason="Existing acceptance lacks guarded numerical or capacity qualification", ev=accepted)

    control = current_job(study, "sp-conservative-control")
    control_config = _bound_input(control)
    controlled = evidence(study, control)
    if (control_config.get("variants") != [POLICY]
            or control_config.get("seeds") != [3501, 3502]
            or control_config.get("frames") != [17, 33]
            or control_config.get("steps") != 20
            or control_config.get("prefixes") != [1, 4, 20]
            or control_config.get("threshold") != .05):
        raise ValueError("conservative provenance does not cover the frozen acceptance recipe")
    if not _qualified(controlled):
        return _record(study, "blocked", completed, blocked_at=PREFIX + "transfer",
                       reason="Conservative diagnostic provenance is not fully guarded and qualified", ev=controlled)
    control_output = _bound_output(control, controlled)
    provenance = dict(job=control["id"], source=control["source"],
        source_digest=control["source_digest"], configuration_sha256=control["configuration_sha256"],
        output_sha256=control_output["sha256"], role="provenance_only_not_backend_acceptance")
    dependencies = [api.binding(acceptance["input"]), _bound_output(acceptance, accepted),
                    api.binding(control["input"]), control_output]

    for stage in STAGES:
        job = _register(study, stage, provenance, dependencies)
        run_job(study, job)
        _bound_input(job)
        ev = evidence(study, job)
        if not _qualified(ev):
            return _record(study, "blocked", completed, blocked_at=job["id"],
                           reason="Actual backend stage lacks guarded numerical or capacity qualification", ev=ev)
        if (ev["result"].get("schema") != "ditserve.workflow.elastic_validation.v1"
                or ev["result"].get("configuration") != configuration(stage)):
            raise ValueError("actual backend result differs from its frozen configuration: " + job["id"])
        dependencies += [api.binding(job["input"]), _bound_output(job, ev)]
        completed.append(job["id"])
        _record(study, "running", completed)
    return _record(study, "passed", completed)
