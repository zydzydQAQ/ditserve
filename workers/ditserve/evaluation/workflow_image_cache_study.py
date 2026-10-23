"""Bounded, paired screening of Sana prompt reuse on two independent traces."""

import copy
from pathlib import Path
import statistics

from . import workflow_contention_study as contention
from . import workflow_next_report as report
from . import workflow_next_study as api
from .workflow_runtime_optimization_study import _control


SEEDS = (3901, 3902)
SELECTION = "image_prompt_cache_screening"
CACHE_OPTIONS = dict(prompt_cache_entries=4, prompt_cache_bytes=16 * 1024**2)
RULE = ("Two predeclared independent traces 3901/3902 with AB/BA order; repeated-prompt "
        "screening requires exact complete outputs, bounded per-worker cache provenance and capacity, "
        "median image p95 reduction >=10%, per-trace throughput >=95%, video p95 <=105%, "
        "and no image SLO loss. A positive screen does not replace independent confirmation.")


def _bound_input(job):
    value = api.read(job["input"])
    if api.digest(value) != job["configuration_sha256"]:
        raise ValueError("image cache frozen input changed: " + job["id"])
    for bound in job["input_bindings"]:
        if api.binding(bound["path"]) != bound:
            raise ValueError("image cache dependency changed: " + job["id"])
    return value


def cache_jobs(study):
    root = Path(study["root"])
    profile, calibration = api.read(root / "profile.json"), api.read(root / "calibration.json")
    if not profile or not calibration or calibration.get("profile") != profile:
        raise ValueError("image cache screening requires unchanged calibration")
    contention._profile(profile)
    controls = contention._controls(study["selections"].get("cpu"))
    _, video_cells, video_bindings = contention._calibration(study, "calibration-video")
    mixed_ev, mixed_cells, mixed_bindings = contention._calibration(study, "calibration-mixed")
    bindings = video_bindings + mixed_bindings + [api.binding(root / name) for name in
                                                 ("profile.json", "calibration.json")]
    requires = ["acceptance", "calibration-video", "calibration-mixed"]
    for seed in contention.SEEDS:
        job, config, _, bound = _control(study, f"contention-s{seed}")
        if len(config.get("phases", [])) != 3 or any(
                phase.get("profile") != profile or len(phase.get("workload", [])) != 24
                for phase in config["phases"]):
            raise ValueError("image cache historical control differs")
        bindings.extend(bound)
        requires.append(job["id"])
    anchors = contention._anchors(profile, video_cells)
    planned = []
    for repeat, seed in enumerate(SEEDS):
        rows = contention.contention_trace(profile, seed, anchors)
        arms = [dict(id=name, policy="block_elastic", quantum_ms=300, decode_frames=1,
                     chunked_decode=True, late_short_ms=2000, demote_late=True)
                for name in ("reference", "candidate")]
        if repeat % 2:
            arms.reverse()
        config = api.cluster_config(profile, rows, arms, **controls)
        for phase in config["phases"]:
            phase["record_dispatch_timeline"] = True
            for rank in range(2):
                settings = copy.deepcopy(api._worker("sana", rank, rank, controls["cores"])["config"])
                settings["options"] = dict(CACHE_OPTIONS) if phase["id"] == "candidate" else {}
                phase["worker_overrides"][f"sana{rank}"] = settings
        estimate, costs = contention._estimate(profile, rows, mixed_ev, mixed_cells, 2)
        if estimate > 370:
            raise ValueError("image cache screening exceeds 370-second pair estimate")
        metadata = dict(comparison="image-prompt-cache", role="repeated_prompt_cache_screening",
            seed=seed, trace_sha256=api.digest(rows), anchors=anchors, estimate_basis=costs,
            cache=dict(CACHE_OPTIONS), decision_rule=RULE, independent_confirmation=False,
            scope="Two independent traces with four repeated prompts; normal end-to-end host request timing, "
                  "including cache misses, output checks and queueing; warmup cache population explicitly audited")
        planned.append((f"image-prompt-cache-s{seed}", config, estimate, metadata))
    jobs = []
    for identifier, config, estimate, metadata in planned:
        prior = [row for row in study["jobs"] if row.get("logical_id", row["id"]) == identifier
                 and not row.get("superseded")]
        if any(row["source"] != study["source"] or row["source_digest"] != study["source_digest"]
               or row["metadata"] != metadata or row["estimated_seconds"] != estimate
               or row["timeout_seconds"] != 480 for row in prior):
            raise ValueError("cannot change frozen image cache screening source or rule")
        job = api.add_job(study, identifier, "extension", config, native=True, devices=(0, 1),
            requires=requires, estimate=estimate, timeout=480, metadata=metadata)
        expected_command = ["/workspace/bin/ditserve", "--root", "/data", "--package-root", "/workspace/workers",
            "--python", "/opt/venv/bin/python", "run", "--config", api.container_path(study, job["input"]),
            "--env", api.container_path(study, job["environment_input"]), "--allow-gpu", "--run-id", identifier,
            "--keep-raw", "--work-dir", api.container_path(study, job["output"]),
            "--results-root", api.container_path(study, root / "native-results"),
            "--ledger", api.container_path(study, root / "native-ledgers" / (identifier + ".json"))]
        if job["command_type"] != "native" or job["command"] != expected_command:
            raise ValueError("image cache native command changed: " + identifier)
        dependencies = {row["path"]: row for row in bindings}
        expected = [api.binding(job["input"]), api.binding(job["environment_input"])] + [
            dependencies[path] for path in sorted(dependencies)]
        if prior and job["input_bindings"] != expected:
            raise ValueError("cannot change frozen image cache dependencies")
        job["input_bindings"] = expected
        api.save(study)
        jobs.append(job)
    return jobs


def summarize(study):
    from .workflow_image_cache_audit import audit
    jobs, gains, safe = [], [], True
    for seed in SEEDS:
        logical = f"image-prompt-cache-s{seed}"
        try:
            job = report.current_job(study, logical)
        except ValueError:
            jobs.append(dict(job=logical, qualified=False, reason="not registered"))
            continue
        _bound_input(job)
        ev = report.evidence(study, job)
        paired = report.comparison(ev["result"], "reference", "candidate",
                                   capacity=ev["capacity"], guarded=ev["qualified"])
        provenance = audit(ev["result"])
        qualified = paired.get("eligible") is True and provenance["passed"]
        row = dict(job=job["id"], seed=seed, qualified=qualified, guarded=ev["guarded"],
                   capacity=ev["capacity"], comparison=paired, cache_audit=provenance,
                   result_binding=api.binding(ev["path"]) if ev["path"].exists() else None)
        if qualified:
            left, right = paired["reference_summary"], paired["candidate_summary"]
            gain = 1 - right["image"]["latency_p95_ms"] / left["image"]["latency_p95_ms"]
            gains.append(gain)
            row["image_p95_reduction_fraction"] = gain
            bounds = dict(throughput=right["total"]["throughput_per_second"] >= .95 * left["total"]["throughput_per_second"],
                video_p95=right["video"]["latency_p95_ms"] <= 1.05 * left["video"]["latency_p95_ms"],
                image_slo=right["image"]["slo_attainment"] >= left["image"]["slo_attainment"])
            row["safety_bounds"] = bounds
            safe = safe and all(bounds.values())
        jobs.append(row)
    qualified = len(jobs) == len(SEEDS) and all(row["qualified"] for row in jobs)
    benefit = qualified and safe and statistics.median(gains) >= .10
    return dict(schema="ditserve.workflow.image-cache-screening.v1",
        status="passed" if qualified else "unqualified", qualified=qualified, benefit_screened=benefit,
        jobs=jobs, decision_rule=RULE,
        median_image_p95_reduction_fraction=statistics.median(gains) if qualified else None,
        recommendation="confirm_repeated_prompt_cache" if benefit else "retain_cache_disabled",
        scope="Two-seed repeated-prompt screening; no unique-prompt or A100 acceleration claim")


def run_followup(study, run_job):
    for job in cache_jobs(study):
        run_job(study, job)
        study["selections"][SELECTION] = summarize(study)
        api.save(study)
    return study["selections"][SELECTION]
