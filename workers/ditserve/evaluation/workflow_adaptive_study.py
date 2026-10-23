"""Independent, frozen three-arm confirmation of causal DiT quantum feedback."""

import copy
import math
from pathlib import Path
import statistics

from . import workflow_contention_study as contention
from . import workflow_next_report as report
from . import workflow_next_study as api


SEEDS = (3801, 3802, 3803)
SELECTION = "adaptive_quantum_confirmation"
POLICIES = (
    dict(id="block300", policy="block_elastic", quantum_ms=300),
    dict(id="block100", policy="block_elastic", quantum_ms=100),
    dict(id="adaptive", policy="block_elastic", quantum_ms=300, adaptive_quantum=True,
         adaptive_hot_quantum_ms=100, adaptive_hot_window_ms=1000),
)
RULE = ("Independent seeds 3801/3802/3803; adaptive benefit requires median image p95 reduction "
        ">=10% versus fixed300, no per-trace image SLO loss, throughput >=95%, video p95 <=105%; "
        "all exact/capacity/full-set/feedback audits must pass. Fixed100 is a separate control.")


def _bound_input(job):
    value = api.read(job["input"])
    if value is None or api.digest(value) != job["configuration_sha256"]:
        raise ValueError("frozen adaptive dependency input changed: " + job["id"])
    for bound in job["input_bindings"]:
        if api.binding(bound["path"]) != bound:
            raise ValueError("frozen adaptive dependency binding changed: " + job["id"])
    return value


def adaptive_jobs(study, profile=None, calibration=None):
    """Register only; shared run03 launcher owns GPU admission and all budgets."""
    root = Path(study["root"])
    profile = profile if profile is not None else api.read(root / "profile.json")
    calibration = calibration if calibration is not None else api.read(root / "calibration.json")
    if not profile or not calibration or calibration.get("profile") != profile:
        raise ValueError("adaptive confirmation requires the existing frozen calibration profile")
    contention._profile(profile)
    cpu = contention._controls(study["selections"].get("cpu"))
    unused, video_cells, video_bindings = contention._calibration(study, "calibration-video")
    mixed_ev, mixed_cells, mixed_bindings = contention._calibration(study, "calibration-mixed")
    bindings = video_bindings + mixed_bindings
    for name in ("profile.json", "calibration.json"):
        if (root / name).exists():
            bindings.append(api.binding(root / name))
    requires = ["acceptance", "calibration-video", "calibration-mixed"]
    for seed in contention.SEEDS:
        job = report.current_job(study, f"contention-s{seed}")
        config = _bound_input(job)
        ev = report.evidence(study, job)
        if not all(ev[key] is True for key in ("guarded", "qualified", "capacity")):
            raise ValueError("adaptive confirmation requires guarded historical contention: " + job["id"])
        phases = config.get("phases", [])
        if (len(phases) != 3 or any(phase.get("profile") != profile or len(phase.get("workload", [])) != 24
                                    for phase in phases)):
            raise ValueError("adaptive historical contention profile or workload differs")
        bindings += [api.binding(job["input"]), api.binding(ev["path"]),
                     api.binding(root / "launches" / job["id"] / "launch.json")]
        requires.append(job["id"])
    anchors = contention._anchors(profile, video_cells)
    planned = []
    for repeat, seed in enumerate(SEEDS):
        rows = contention.contention_trace(profile, seed, anchors)
        arms = copy.deepcopy(POLICIES[repeat:] + POLICIES[:repeat])
        for arm in arms:
            arm.update(decode_frames=1, chunked_decode=True, late_short_ms=2000, demote_late=True)
        config = api.cluster_config(profile, rows, arms, **cpu)
        for phase in config["phases"]:
            phase["record_dispatch_timeline"] = True
        estimate, costs = contention._estimate(profile, rows, mixed_ev, mixed_cells, 3)
        if estimate > 550:
            raise ValueError("adaptive confirmation exceeds the frozen 550-second per-job estimate ceiling")
        metadata = dict(comparison="adaptive-quantum", role="independent_confirmation", seed=seed,
            trace_sha256=api.digest(rows), independent_confirmation=True, anchors=anchors,
            estimate_basis=costs, decoding="one latent frame in all arms", short_task_protection=True,
            feedback="arrived waiting images or last admitted image arrival within 1000 ms; no future workload",
            decision_rule=RULE, diagnostic_scope="Host dispatch and actual backend block counts; not CUDA overlap")
        planned.append((f"adaptive-quantum-s{seed}", config, estimate, metadata))
    jobs = []
    for identifier, config, estimate, metadata in planned:
        existing = [job for job in study["jobs"] if job.get("logical_id", job["id"]) == identifier
                    and not job.get("superseded")]
        if any(job["source"] != study["source"] or job["source_digest"] != study["source_digest"]
               or job["metadata"] != metadata or job["estimated_seconds"] != estimate for job in existing):
            raise ValueError("cannot change the frozen adaptive implementation or decision rule")
        jobs.append(contention._bind_job(study, identifier, config, family="confirmation", requires=requires,
            estimate=estimate, metadata=metadata, bindings=bindings))
    return jobs


def feedback_audit(result):
    """Recompute every observed choice from frozen costs and past-only feedback."""
    errors, arms = [], {}
    phases = {phase["id"]: phase for phase in result.get("config", {}).get("phases", [])}
    cells = report.native_arms(result)
    if (len(result.get("config", {}).get("phases", [])) != 3 or len(result.get("phases", [])) != 3
            or set(phases) != {"block300", "block100", "adaptive"} or set(cells) != set(phases)):
        return dict(passed=False, errors=["missing or duplicate three-arm configuration/results"], arms={})
    for name, cell in cells.items():
        phase, rows = phases[name], cell.get("requests", [])
        opts = phase.get("options", {})
        expected = next(dict(arm) for arm in POLICIES if arm["id"] == name)
        expected.pop("id"); expected.pop("policy")
        expected.update(decode_frames=1, chunked_decode=True, late_short_ms=2000, demote_late=True)
        if any(opts.get(key) != value for key, value in expected.items()):
            errors.append(name + ": frozen option mismatch")
        if name != "adaptive" and opts.get("adaptive_quantum", False) is not False:
            errors.append(name + ": fixed control enabled adaptive feedback")
        if (len(rows) != 24 or len({row.get("request_id") for row in rows}) != 24
                or sum(row.get("kind") == "image" for row in rows) != 20):
            errors.append(name + ": incomplete 24-request set")
        images = [row for row in rows if row.get("kind") == "image"]
        counts, decode, executed, triggers = {}, 0, 0, {}
        for row in rows:
            row_blocks, row_decode = 0, 0
            events = row.get("dispatch_timeline")
            if not isinstance(events, list) or not events:
                errors.append(name + ": missing dispatch timeline")
                continue
            for event in events:
                try:
                    start, finish = event["start_ms"], event["finish_ms"]
                    assert all(type(value) in (int, float) and math.isfinite(value) for value in (start, finish))
                    assert 0 <= start <= finish
                    feedback = event["quantum_feedback"]
                    observed = feedback["observed_at_ms"]
                    assert type(observed) in (int, float) and math.isfinite(observed) and 0 <= observed <= start
                    last, waiting = feedback["last_image_arrival_ms"], feedback["waiting_images"]
                    possible_waiters = [image for image in images if image["arrival_ms"] <= observed <= image["start_ms"]]
                    assert type(waiting) is int and 0 <= waiting <= len(possible_waiters)
                    assert last is None or (type(last) in (int, float) and math.isfinite(last) and 0 <= last <= observed
                                            and any(image["arrival_ms"] == last for image in images))
                    assert type(feedback["oldest_image_wait_ms"]) in (int, float)
                    assert math.isfinite(feedback["oldest_image_wait_ms"])
                    assert 0 <= feedback["oldest_image_wait_ms"] <= max((observed - image["arrival_ms"] for image in possible_waiters), default=0)
                    assert (waiting != 0 or feedback["oldest_image_wait_ms"] == 0)
                    is_adaptive = name == "adaptive"
                    recent = last is not None and observed - last < 1000
                    hot = is_adaptive and (waiting > 0 or recent)
                    mode = "hot" if hot else "cool" if is_adaptive else "fixed"
                    trigger = "fixed" if not is_adaptive else "waiting_image" if waiting else "recent_image_arrival" if recent else "quiet"
                    assert feedback["adaptive"] is is_adaptive and feedback["mode"] == mode
                    assert feedback["trigger"] == trigger and feedback["hot_window_ms"] == (1000 if is_adaptive else 0)
                    quantum = event["quantum"]
                    if type(quantum) is int:
                        assert row["kind"] == "video"
                        chosen = 100 if hot or name == "block100" else 300
                        cost = phase["profile"]["classes"][row["class"]]
                        assert event["group"] in ([0], [1])
                        predicted = max(1, min(math.floor(chosen / (cost["step_compute_ms"] / cost["blocks_per_step"])), cost["blocks_per_step"]))
                        assert event["chosen_quantum_ms"] == chosen and event["block_count"] == quantum == predicted
                        assert type(event["executed_blocks"]) is int and 0 < event["executed_blocks"] <= quantum
                        counts[chosen] = counts.get(chosen, 0) + 1
                        executed += event["executed_blocks"]
                        row_blocks += event["executed_blocks"]
                        triggers[trigger] = triggers.get(trigger, 0) + 1
                    elif quantum == "finish":
                        assert row["kind"] == "video" and event["decode_frames"] == 1
                        decode += 1
                        row_decode += 1
                    else:
                        assert row["kind"] == "image" and quantum == "atomic"
                except (AssertionError, KeyError, TypeError, ValueError, ZeroDivisionError):
                    errors.append(name + ": invalid causal feedback or actual quantum record for " + str(row.get("request_id")))
            if row.get("kind") == "video":
                cost = phase.get("profile", {}).get("classes", {}).get(row.get("class"), {})
                if (row_blocks != cost.get("blocks_per_step", 0) * row.get("spec", {}).get("steps", 0)
                        or row_decode != cost.get("decode_chunks")):
                    errors.append(name + ": incomplete executed DiT/decode work for " + str(row.get("request_id")))
        if not counts or not decode:
            errors.append(name + ": no observed DiT/decode work")
        if name == "adaptive" and set(counts) != {100, 300}:
            errors.append(name + ": both hot and cool DiT modes must actually execute")
        arms[name] = dict(dit_dispatches_by_quantum_ms=counts, executed_blocks=executed,
                          one_frame_decode_dispatches=decode, dit_trigger_counts=triggers)
    return dict(passed=not errors, errors=errors, arms=arms,
                scope="Actual host decisions and backend block counts, causally bounded by arrived workload")


def summarize(study):
    jobs, gains = [], []
    safe = True
    for seed in SEEDS:
        logical = f"adaptive-quantum-s{seed}"
        try:
            job = report.current_job(study, logical)
        except ValueError:
            jobs.append(dict(job=logical, qualified=False, reason="not registered"))
            continue
        _bound_input(job)
        ev = report.evidence(study, job)
        comparisons = [report.comparison(ev["result"], left, right, capacity=ev["capacity"], guarded=ev["qualified"])
                       for left, right in (("block300", "block100"), ("block300", "adaptive"), ("block100", "adaptive"))]
        feedback = feedback_audit(ev["result"])
        frozen = api.read(job["input"])
        frozen_phases = {phase["id"]: phase for phase in frozen["phases"]}
        result_phases = {phase["id"]: phase for phase in ev["result"].get("config", {}).get("phases", [])}
        bound = set(frozen_phases) == set(result_phases) and all(
            all(frozen_phases[name].get(key) == result_phases[name].get(key) for key in
                ("options", "profile", "workload", "policy", "record_dispatch_timeline")) for name in frozen_phases)
        qualified = bound and feedback["passed"] and all(row.get("eligible") is True for row in comparisons)
        row = dict(job=job["id"], seed=seed, qualified=qualified, guarded=ev["guarded"], capacity=ev["capacity"],
                   frozen_config_matches=bound, comparisons=comparisons, feedback=feedback,
                   result_binding=api.binding(ev["path"]) if ev["path"].exists() else None)
        if qualified:
            left, right = comparisons[1]["reference_summary"], comparisons[1]["candidate_summary"]
            gain = 1 - right["image"]["latency_p95_ms"] / left["image"]["latency_p95_ms"]
            gains.append(gain)
            row["adaptive_image_p95_reduction_fraction"] = gain
            safe = safe and (right["total"]["throughput_per_second"] >= .95 * left["total"]["throughput_per_second"]
                and right["video"]["latency_p95_ms"] <= 1.05 * left["video"]["latency_p95_ms"]
                and right["image"]["slo_attainment"] >= left["image"]["slo_attainment"])
        jobs.append(row)
    qualified = len(jobs) == len(SEEDS) and all(row["qualified"] for row in jobs)
    benefit = qualified and safe and statistics.median(gains) >= .10
    return dict(schema="ditserve.workflow.adaptive-confirmation.v1", status="passed" if qualified else "unqualified",
        qualified=qualified, benefit_confirmed=benefit, jobs=jobs, decision_rule=RULE,
        median_adaptive_image_p95_reduction_fraction=statistics.median(gains) if qualified else None,
        recommendation="adaptive" if benefit else "retain_fixed300",
        scope="Three predeclared independent traces; exact outputs and observed switching do not alone prove acceleration")


def run_followup(study, run_job):
    for job in adaptive_jobs(study):
        run_job(study, job)
        study["selections"][SELECTION] = summarize(study)
        api.save(study)
    return study["selections"][SELECTION]
