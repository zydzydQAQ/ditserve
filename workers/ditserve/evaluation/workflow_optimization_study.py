"""Bound follow-up controls for SP arithmetic, contention and overlap memory."""

import copy
import math
from pathlib import Path
import statistics

from . import workflow_next_study as api
from .workflow_next_report import current_job, evidence


REPLAY_CANDIDATES = [
    "full_rows_ffn2", "full_rows_ffn2_qkv",
    "global_tile_1024", "global_tile_2048", "linear_fp32",
]


def _original_sp(study):
    job = current_job(study, "sp-diagnostic")
    config = api.read(job["input"])
    if api.digest(config) != job["configuration_sha256"]:
        raise ValueError("original SP input changed")
    return config


def replay_job(study):
    return api.sp_linear_replay_job(
        study, identifier="sp-linear-candidates", candidates=REPLAY_CANDIDATES,
        microbenchmark=dict(enabled=True, warmups=3, repeats=10),
        estimate=800, timeout=1800)


def selective_job(study):
    config = dict(_original_sp(study),
                  variants=["full_rows_ffn2", "full_rows_ffn2_qkv"],
                  extended_diagnostics=False, localize_modes=[], timeout_seconds=1700)
    return api.add_job(study, "sp-selective-controls", "sp", config,
        module="workflow_serving_sp", devices=(0, 1), requires=["acceptance"],
        estimate=800, timeout=1800, metadata=dict(
            role="selective_linear_validation", diagnostic_only=True,
            performance_evidence=False, deployment_qualification=False,
            reference="unchanged original SP1", original_threshold=.05,
            purpose="Reduce the set of padded Linear modules; validate every original trajectory"))


def conservative_job(study):
    """Drop only FFN1 padding from the previously exact full-row control."""
    config = _original_sp(study)
    if (config.get("seeds") != [3501, 3502] or config.get("frames") != [17, 33]
            or config.get("steps") != 20 or config.get("threshold", .05) != .05):
        raise ValueError("conservative control retains the original two-seed, two-frame, 20-step matrix")

    def exact(comparison):
        return (isinstance(comparison, dict) and all(comparison.get(name) is True for name in
                    ("shape_match", "finite", "reference_finite", "exact"))
                and type(comparison.get("max_abs")) in (int, float) and comparison["max_abs"] == 0)

    def exact_prefix(prefix):
        steps = prefix.get("steps", [])
        return (prefix.get("passed") is True and "first_nonzero_step" in prefix
                and prefix["first_nonzero_step"] is None
                and prefix.get("scheduler_steps") == 20 and len(prefix.get("timestep_grid", [])) == 20
                and prefix.get("initial_and_schedule_exact") is True
                and exact(prefix.get("video_comparison")) and len(steps) == prefix["prefix"]
                and [step.get("step") for step in steps] == list(range(prefix["prefix"]))
                and all(step.get("passed") is True and all(exact(step.get("comparison", {}).get(name))
                        for name in ("latent_before", "latent_after", "cond", "uncond")) for step in steps))

    source = current_job(study, "sp-linear-controls")
    ev = evidence(study, source)
    expected = {(seed, frames) for seed in config["seeds"] for frames in config["frames"]}
    ranks = ev["result"].get("ranks", [])
    if (not ev["guarded"] or len(ranks) != 2 or any(type(rank.get("rank")) is not int for rank in ranks)
            or {rank.get("rank") for rank in ranks} != {0, 1}):
        raise ValueError("conservative control requires the bound original full-row evidence")
    for rank in ranks:
        rows = [row for row in rank.get("variants", []) if row.get("mode") == "full_rows_linear"]
        if (len(rows) != len(expected) or
                {(row.get("seed"), row.get("frames")) for row in rows} != expected or
                any(row.get("status") != "qualified" or
                    [prefix.get("prefix") for prefix in row.get("prefixes", [])] != [1, 4, 20] or
                    any(not exact_prefix(prefix) for prefix in row["prefixes"]) for row in rows)):
            raise ValueError("original full-row control did not exactly qualify the full matrix")
    replay = current_job(study, "sp-linear-candidates")
    replay_ev = evidence(study, replay)
    result = replay_ev["result"]
    qualification = result.get("qualification", {})
    cases = result.get("cases", [])
    identities = {(seed, frames, branch, operator) for seed, frames in expected
                  for branch in ("cond", "uncond") for operator in ("to_q", "ffn.net.0.proj", "ffn.net.2")}
    if (not replay_ev["guarded"] or not replay_ev["qualified"]
            or result.get("schema") != "ditserve.workflow.linear-replay.v1"
            or result.get("status") != "completed"
            or any(qualification.get(name) is not True for name in ("passed", "source_integrity", "repeated_exact"))
            or result.get("reference_reproduction", {}).get("passed") is not True
            or len(cases) != 24 or {(case.get("seed"), case.get("frames"), case.get("branch"), case.get("operator"))
                                  for case in cases} != identities):
        raise ValueError("conservative control requires all 24 bound original-reference replay cases")
    for case in cases:
        arms = case.get("arms", [])
        full = [arm for arm in arms if arm.get("precision") == "default"
                and arm.get("shape_kind") == "full" and arm.get("rank") is None]
        if (case.get("qualified") is not True or case.get("reference_reproduced") is not True
                or not arms or any(arm.get("qualified") is not True or arm.get("repeated_exact") is not True
                                   for arm in arms)
                or len(full) != 1 or not exact(full[0].get("comparison_to_reference"))):
            raise ValueError("original full/default replay anchors or repeats did not qualify")
        if case["operator"] == "ffn.net.0.proj":
            halves = [arm for arm in arms if arm.get("precision") == "default" and arm.get("shape_kind") == "half"]
            if (len(halves) != 2 or any(type(arm.get("rank")) is not int for arm in halves)
                    or {arm.get("rank") for arm in halves} != {0, 1}
                    or any(not exact(arm.get("comparison_to_reference")) or
                           not exact(arm.get("comparison_to_full_same_precision")) for arm in halves)):
                raise ValueError("both native FFN1 halves must exactly match the original SP1 reference")
    source_bindings = []
    for baseline in (ev, replay_ev):
        bound = api.binding(baseline["path"])
        if bound["sha256"] != baseline["launch"]["output_sha256"]:
            raise ValueError("conservative evidence changed while binding")
        source_bindings.append(bound)
    config.update(variants=["full_rows_except_ffn1"], threshold=.05, extended_diagnostics=False,
                  localize_modes=[], timeout_seconds=1100)
    job = api.add_job(study, "sp-conservative-control", "sp", config,
        module="workflow_serving_sp", devices=(0, 1),
        requires=["acceptance", "sp-linear-candidates"], estimate=500, timeout=1200,
        metadata=dict(role="conservative_linear_validation", diagnostic_only=True,
            performance_evidence=False, deployment_qualification=False,
            reference="unchanged original SP1", original_threshold=.05,
            baseline_job=source["id"], baseline_output_sha256=ev["launch"]["output_sha256"],
            replay_job=replay["id"], replay_output_sha256=replay_ev["launch"]["output_sha256"],
            purpose="Remove only FFN1 padding; retain padding of unmeasured attention projections"))
    for source_binding in source_bindings:
        if source_binding not in job["input_bindings"]:
            if (Path(study["root"]) / "launches" / job["id"] / "launch.json").exists():
                raise ValueError("cannot change baseline provenance after conservative launch")
            job["input_bindings"].append(source_binding)
            api.save(study)
    return job


def tile_job(study, mode, replay):
    if mode not in ("global_tile_1024", "global_tile_2048"):
        raise ValueError("unsupported full-trajectory tile candidate")
    ev = evidence(study, replay)
    if not ev["guarded"] or not ev["qualified"]:
        raise ValueError("tile validation requires a complete bound candidate replay")
    config = dict(_original_sp(study), variants=[mode],
                  extended_diagnostics=False, localize_modes=[], timeout_seconds=1700)
    job = api.add_job(study, "sp-" + mode.replace("_", "-"), "sp", config,
        module="workflow_serving_sp", devices=(0, 1), requires=["acceptance", replay["logical_id"]],
        estimate=800, timeout=1800, metadata=dict(
            role="tiled_linear_validation", diagnostic_only=True,
            performance_evidence=False, deployment_qualification=False,
            reference="unchanged original SP1 with an additional candidate-SP1 bridge",
            original_threshold=.05, replay_output_sha256=ev["launch"]["output_sha256"]))
    source_binding = api.binding(ev["path"])
    if source_binding not in job["input_bindings"]:
        # New input dependencies may only be attached before the first launch.
        if (Path(study["root"]) / "launches" / job["id"] / "launch.json").exists():
            raise ValueError("cannot change replay provenance after tile launch")
        job["input_bindings"].append(source_binding)
        api.save(study)
    return job


def select_tiles(result):
    """Operator screening only; the unchanged full-request gate still follows."""
    expected = {(seed, frames, branch, op) for seed in (3501, 3502)
                for frames in (17, 33) for branch in ("cond", "uncond")
                for op in ("to_q", "ffn.net.0.proj", "ffn.net.2")}
    cases = result.get("cases", [])
    keys = {(c.get("seed"), c.get("frames"), c.get("branch"), c.get("operator")) for c in cases}
    if (result.get("status") != "completed" or
            result.get("reference_reproduction", {}).get("passed") is not True or
            not result.get("qualification", {}).get("source_integrity") or
            len(cases) != len(expected) or keys != expected):
        return []
    choices = []
    for mode in ("global_tile_1024", "global_tile_2048"):
        latencies, errors = [], []
        valid = True
        for case in cases:
            rows = [row for row in case.get("candidates", []) if row.get("candidate") == mode]
            if len(rows) != 1 or rows[0].get("qualified") is not True:
                valid = False
                break
            arms = rows[0].get("arms", [])
            if (len(arms) != 3 or {(a.get("shape_kind"), a.get("rank")) for a in arms}
                    != {("full", None), ("half", 0), ("half", 1)}):
                valid = False
                break
            half_ms = 0.
            for arm in arms:
                original = arm.get("comparison_to_reference", {})
                error = original.get("max_abs", math.inf)
                ms = arm.get("microbenchmark", {}).get("median_ms", math.inf)
                if (arm.get("qualified") is not True or arm.get("repeated_exact") is not True or
                        original.get("shape_match") is not True or original.get("finite") is not True or
                        not isinstance(error, (int, float)) or not math.isfinite(error) or error > .05 or
                        not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0):
                    valid = False
                    break
                errors.append(error)
                if arm["shape_kind"] == "half":
                    if arm.get("comparison_to_full_same_precision", {}).get("exact") is not True:
                        valid = False
                        break
                    half_ms += ms
            if not valid:
                break
            latencies.append(half_ms)
        if valid:
            choices.append(dict(mode=mode, median_two_half_operator_ms=statistics.median(latencies),
                                maximum_bridge_error=max(errors), operator_cases=len(cases)))
    return sorted(choices, key=lambda row: (row["median_two_half_operator_ms"], row["mode"]))


def run_tile_followups(study, replay, run_job):
    ev = evidence(study, replay)
    if not ev["guarded"] or not ev["qualified"]:
        return
    key = "sp_tile_followup"
    selection = dict(replay_output_sha256=ev["launch"]["output_sha256"],
                     candidates=select_tiles(ev["result"]),
                     rule="24 bound cases; original bridge <=0.05; exact halves; order by operator time",
                     scope="operator screening only, not full-request qualification")
    previous = study["selections"].get(key)
    if previous is not None and previous != selection:
        raise ValueError("cannot silently change a frozen tile screening selection")
    study["selections"][key] = selection
    api.save(study)
    # A second candidate is a bounded fallback, not a replacement for old evidence.
    for row in selection["candidates"]:
        if run_job(study, tile_job(study, row["mode"], replay)):
            break


def overlap_job(study):
    source = current_job(study, "overlap-lifecycle")
    original = api.read(source["input"])
    if api.digest(original) != source["configuration_sha256"]:
        raise ValueError("original overlap input changed")
    config = copy.deepcopy(original)
    config.update(gpu_uuid=study["gpu_mapping"][1]["uuid"], lifecycle_pairs=3,
                  arm_timeout_seconds=240,
                  arm_specs=[
                      dict(id="serial", mode="serial", decode_a_on_b_stream=False),
                      dict(id="streams-current-decode", mode="streams", decode_a_on_b_stream=False),
                      dict(id="streams-shared-decode", mode="streams", decode_a_on_b_stream=True)])
    return api.add_job(study, "overlap-isolated-lifecycle", "extension", config,
        module="workflow_overlap_isolated", devices=(1,), requires=["acceptance"],
        estimate=500, timeout=900, metadata=dict(
            role="isolated_overlap_allocator_control", performance_evidence=False,
            purpose="Separate process isolation from reuse of the VAE decode stream",
            previous_diagnostic=source["id"]))


def warmed_overlap_job(study):
    """Confirm the observed cache plateau with a predeclared warmup window."""
    source = current_job(study, "overlap-isolated-lifecycle")
    ev = evidence(study, source)
    arms = {row["id"]: row["result"] for row in ev["result"].get("arms", [])}
    if (not ev["guarded"] or len(ev["result"].get("arms", [])) != 3 or
            ev["result"].get("qualification", {}).get("scopes", {}).get("numeric", {}).get("status") != "passed" or
            set(arms) != {"serial", "streams-current-decode", "streams-shared-decode"} or
            any(row.get("completed_pairs") != 3 or
                any(row.get("qualification", {}).get("scopes", {}).get(scope, {}).get("status") != "passed"
                    for scope in ("numeric", "lifecycle"))
                for row in arms.values()) or
            arms["serial"].get("qualification", {}).get("scopes", {}).get("capacity", {}).get("status") != "passed" or
            arms["streams-shared-decode"].get("qualification", {}).get("passed") is not True):
        raise ValueError("warmed confirmation requires complete exact controls and the qualified shared-stream arm")
    launches = [row.get("launch", {}) for row in ev["result"]["arms"]]
    pids = [row.get("pid") for row in launches]
    if (any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != 3 or
            any(launch.get("reaped") is not True or launch.get("returncode") != 0 or
                launch.get("timed_out") is not False or
                row["result"].get("process", {}).get("pid") != launch["pid"]
                for row, launch in zip(ev["result"]["arms"], launches)) or
            any(type(launch.get(key)) is not int for launch in launches
                for key in ("started_unix_ns", "finished_unix_ns")) or
            any(launch["started_unix_ns"] >= launch["finished_unix_ns"] for launch in launches) or
            any(left["finished_unix_ns"] > right["started_unix_ns"]
                for left, right in zip(launches, launches[1:]))):
        raise ValueError("warmed confirmation requires three distinct sequential child processes, normally reaped")
    pairs = ev["result"].get("configuration", {}).get("frame_pairs", [])
    controls = ev["result"].get("qualification", {}).get("control_cross_arm", [])
    if (len(pairs) != 1 or any(row.get("frame_pair") != pairs[0] for row in ev["result"]["arms"]) or
            len(controls) != 1 or controls[0].get("id") != "streams-current-decode" or
            [controls[0].get("frames_a"), controls[0].get("frames_b")] != pairs[0] or
            any(controls[0].get(key) is not True for key in ("complete", "exact")) or
            any(controls[0].get(key) != 3 for key in ("observed_pairs", "expected_pairs"))):
        raise ValueError("warmed confirmation requires all three control pairs exact against serial")
    metrics = arms["serial"].get("steady_state", {}).get("metrics", {})
    if metrics.get("allocated_bytes", {}).get("maximum_growth_bytes") != 0:
        raise ValueError("live allocation growth cannot be reclassified as cache warmup")
    for name in ("reserved_bytes", "nvml_used_bytes"):
        row = metrics.get(name, {})
        values = row.get("values", [])
        if len(values) != 3 or not values[0] < values[1] == values[2]:
            raise ValueError("warmed confirmation requires the observed second/third-pair cache plateau")
    config = api.read(source["input"])
    if api.digest(config) != source["configuration_sha256"]:
        raise ValueError("original isolated lifecycle input changed")
    config.update(warmup_pairs=2, lifecycle_pairs=3, arm_timeout_seconds=400)
    job = api.add_job(study, "overlap-isolated-warmed-lifecycle", "extension", config,
        module="workflow_overlap_isolated", devices=(1,), requires=["acceptance"],
        estimate=1000, timeout=1500, metadata=dict(
            role="fixed_warmup_overlap_confirmation", performance_evidence=False,
            previous_diagnostic=source["id"], previous_output_sha256=ev["launch"]["output_sha256"],
            purpose="Two fixed warmup pairs plus three measured pairs; capacity and exactness cover all five",
            unchanged_limits="16 MiB live allocation, 64 MiB reserved and NVML growth; no cache trim"))
    source_binding = api.binding(ev["path"])
    if source_binding not in job["input_bindings"]:
        if (Path(study["root"]) / "launches" / job["id"] / "launch.json").exists():
            raise ValueError("cannot change lifecycle provenance after warmup confirmation launch")
        job["input_bindings"].append(source_binding)
        api.save(study)
    return job


def run_followup(study, run_job):
    """Serial jobs share the existing family caps, suite guard and budget."""
    from .workflow_contention_study import contention_jobs, contention_ablation_jobs, contention_summary
    root = Path(study["root"])
    replay = replay_job(study)
    if replay is not None:
        run_job(study, replay)
    run_job(study, selective_job(study))
    if replay is not None:
        run_tile_followups(study, replay, run_job)
    run_job(study, overlap_job(study))
    profile = api.read(root / "profile.json")
    calibration = api.read(root / "calibration.json")
    if profile is None or calibration is None:
        raise ValueError("contention continuation requires the existing cost calibration")
    for job in contention_jobs(study, profile, calibration, cpu=study["selections"]["cpu"]):
        run_job(study, job)
    # Retain negative trigger evidence too, including why an ablation was absent.
    study["selections"]["contention_followup"] = contention_summary(study)
    api.save(study)
    for job in contention_ablation_jobs(study, profile, cpu=study["selections"]["cpu"]):
        run_job(study, job)
