"""Independent, source-bound SP repair, adaptive scheduling and VAE experiments."""

from pathlib import Path

from . import workflow_next_study as api
from .workflow_next_report import current_job, evidence


SELECTION = "runtime_optimization"
SP_JOB = "sp2-scratch-performance"
VAE_JOB = "overlap-shared-performance"


def _control(study, identifier, *, capacity=True):
    job = current_job(study, identifier)
    config = api.read(job["input"])
    if api.digest(config) != job["configuration_sha256"]:
        raise ValueError("control configuration changed: " + identifier)
    for expected in job["input_bindings"]:
        if api.binding(expected["path"]) != expected:
            raise ValueError("control dependency changed: " + identifier)
    ev = evidence(study, job)
    if not ev["guarded"] or not ev["qualified"] or (capacity and not ev["capacity"]):
        raise ValueError("qualified control required: " + identifier)
    bound = api.binding(ev["path"])
    if bound["sha256"] != ev["launch"].get("output_sha256"):
        raise ValueError("control output changed: " + identifier)
    bindings = [api.binding(job["input"]), bound,
                api.binding(Path(study["root"]) / "launches" / identifier / "launch.json")]
    return job, config, ev, bindings


def _register(study, identifier, family, config, module, devices, requires,
              bindings, metadata, estimate, timeout):
    prior = [row for row in study["jobs"] if row.get("logical_id", row["id"]) == identifier
             and not row.get("superseded")]
    for row in prior:
        if (row["source"] != study["source"] or row["source_digest"] != study["source_digest"]
                or row.get("metadata") != metadata or row["estimated_seconds"] != estimate
                or row["timeout_seconds"] != timeout):
            raise ValueError("cannot change runtime experiment source or metadata: " + identifier)
    job = api.add_job(study, identifier, family, config, module=module, devices=devices,
                      requires=requires, estimate=estimate, timeout=timeout, metadata=metadata)
    expected_command = ["/opt/venv/bin/python", "-m", "ditserve.evaluation." + module,
                        "--config", api.container_path(study, job["input"]),
                        "--out", api.container_path(study, job["output"])]
    expected = [api.binding(job["input"])]
    for bound in bindings:
        if bound not in expected:
            expected.append(bound)
    if job["command"] != expected_command:
        raise ValueError("runtime experiment command changed: " + identifier)
    if prior and job["input_bindings"] != expected:
        raise ValueError("runtime experiment dependencies changed: " + identifier)
    job["input_bindings"] = expected
    api.save(study)
    return job


def _sp_config():
    from .workflow_sp2_performance import normalize_config
    return normalize_config(dict(model_path="/models/wan2.1-t2v-1.3b",
        devices=["cuda:0", "cuda:1"], frames=[17, 33], seeds=[3501, 3502], steps=20, timeout_seconds=1400,
        cpu_threads=8, cpu_masks={"cuda:0": list(range(8)), "cuda:1": list(range(60, 68))}))


def sp_job(study):
    bindings = []
    requires = ["elastic-repaired-transfer", "elastic-repaired-pause", "elastic-repaired-resize"]
    for identifier in requires:
        job, config, ev, bound = _control(study, identifier)
        if (config.get("sp_linear_policy") != "full_rows_except_ffn1"
                or config.get("frames") != [17, 33] or config.get("seeds") != [3501, 3502]
                or config.get("steps") != 20 or config.get("threshold") != .05
                or ev["result"].get("schema") != "ditserve.workflow.elastic_validation.v1"):
            raise ValueError("SP control does not cover the unchanged acceptance recipe")
        bindings.extend(bound)
    return _register(study, SP_JOB, "sp", _sp_config(), "workflow_sp2_performance", (0, 1),
        requires, bindings, dict(role="sp2_scratch_cost_ablation", comparison="sp2_scratch",
            reference_policy="full_rows_except_ffn1", candidate_policy="full_rows_except_ffn1_scratch",
            correctness_reference="ordinary SP1; original thresholds unchanged",
            scope="prepared fixed-SP2 20-step DiT plus GPU VAE; excludes CPU encoding, prepare, initial transfer and final D2H/hash"),
        estimate=1200, timeout=1500)


def _vae_config(old):
    from .workflow_overlap_performance import normalize_config
    keys = ("model_path", "device", "cpu_threads", "cpu_masks", "height", "width", "steps",
            "seed", "frame_pairs", "gpu_uuid", "reuse_streams", "trim_after_references",
            "release_b_before_a_decode")
    return normalize_config({key: old[key] for key in keys})


def vae_job(study):
    identifier = "overlap-isolated-warmed-lifecycle"
    # That experiment intentionally retained a capacity-failed third diagnostic
    # arm. The qualified serial/shared subset is provenance, not a waiver of the
    # new experiment's whole-launch capacity gate.
    job, config, ev, bindings = _control(study, identifier, capacity=False)
    q = ev["result"].get("qualification", {})
    if (ev["result"].get("schema") != "ditserve.workflow.overlap-isolated.v1"
            or config.get("frame_pairs") != [[17, 17]] or config.get("warmup_pairs") != 2
            or config.get("lifecycle_pairs") != 3 or config.get("steps") != 20
            or q.get("passed") is not True
            or q.get("required_arms") != ["serial", "streams-shared-decode"]
            or any(q.get("scopes", {}).get(scope, {}).get("status") != "passed"
                   for scope in ("numeric", "capacity", "steady_state"))):
        raise ValueError("VAE performance requires the qualified fixed-warmup serial/shared controls")
    for name in ("serial", "streams-shared-decode"):
        rows = [arm for arm in ev["result"].get("arms", []) if arm.get("id") == name]
        if len(rows) != 1:
            raise ValueError("missing or duplicate VAE lifecycle control")
        arm, launch = rows[0].get("result", {}), rows[0].get("launch", {})
        if (rows[0].get("frame_pair") != [17, 17] or arm.get("completed_pairs") != 5
                or arm.get("unrun_pairs") != [] or arm.get("qualification", {}).get("passed") is not True
                or launch.get("returncode") != 0 or launch.get("reaped") is not True
                or launch.get("timed_out") is not False):
            raise ValueError("VAE lifecycle control is incomplete or not reaped")
    return _register(study, VAE_JOB, "extension", _vae_config(config), "workflow_overlap_performance", (1,),
        [identifier], bindings, dict(role="isolated_shared_decode_performance", comparison="vae_shared_decode",
            provenance_required_arms=["serial", "streams-shared-decode"],
            diagnostic_current_decode_retained=True, requires_new_whole_launch_capacity=True,
            scope="17/17-frame two-request completion; kernel overlap requires separate CUDA trace"),
        estimate=900, timeout=1200)


def _run(study, run_job, job):
    run_job(study, job)
    ev = evidence(study, job)
    config = api.read(job["input"])
    if api.digest(config) != job["configuration_sha256"]:
        raise ValueError("runtime experiment configuration changed during execution")
    for bound in job["input_bindings"]:
        if api.binding(bound["path"]) != bound:
            raise ValueError("runtime experiment dependency changed during execution")
    passed = all(ev.get(key) is True for key in ("guarded", "qualified", "capacity"))
    expected_schema = {SP_JOB: "ditserve.workflow.sp2_performance.v1",
                       VAE_JOB: "ditserve.workflow.overlap-performance.v1"}[job["id"]]
    if passed and (ev["result"].get("configuration") != config
                   or ev["result"].get("schema") != expected_schema):
        raise ValueError("runtime result does not match its frozen configuration or schema")
    return dict(job=job["id"], status="passed" if passed else "unqualified_or_deferred",
                guarded=ev["guarded"], qualified=ev["qualified"], capacity=ev["capacity"],
                result_status=ev["result"].get("status"),
                result=api.binding(ev["path"]) if ev["path"].exists() else None,
                performance_scope=job["metadata"]["scope"])


def run_followup(study, run_job):
    from .workflow_adaptive_study import run_followup as adaptive_followup
    summary = dict(status="running", source=study["source"], source_digest=study["source_digest"],
                   prior_results_unchanged=True, budget_owner="existing run03 cumulative ledger", experiments=[])
    study["selections"][SELECTION] = summary
    api.save(study)
    summary["experiments"].append(_run(study, run_job, sp_job(study)))
    api.save(study)
    # Numerical or performance failure in one direction does not suppress an
    # independent experiment. Infrastructure/guard failures still raise in run_job.
    adaptive = adaptive_followup(study, run_job)
    summary["adaptive_selection"] = adaptive
    api.save(study)
    summary["experiments"].append(_run(study, run_job, vae_job(study)))
    summary["status"] = "settled"
    api.save(study)
    return summary
