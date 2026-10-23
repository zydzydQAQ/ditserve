"""Frozen SP1 contention screening; imports and planning never initialize CUDA."""

import copy
import math
from pathlib import Path
import random
import statistics

from . import workflow_next_report as report
from . import workflow_next_study as study_api


SEEDS = (3701, 3702, 3703)
KINDS = ("wan1.3b-480x832x17", "wan1.3b-480x832x33", "sana-1024")
POLICIES = (
    dict(id="step"),
    dict(id="block100", policy="block_elastic", quantum_ms=100),
    dict(id="block300", policy="block_elastic", quantum_ms=300),
)


def _controls(cpu):
    cpu = cpu or dict(cores=8, depth=1)
    if (cpu.get("cores"), cpu.get("depth")) != (8, 1):
        raise ValueError("contention fixes the qualified eight-core, depth-one control")
    return dict(cores=8, depth=1)


def _profile(profile):
    for name in KINDS:
        row = profile["classes"][name]
        keys = ("service_ms",) if name == "sana-1024" else (
            "encode_ms", "prepare_ms", "step_compute_ms", "finish_ms")
        if any(not isinstance(row.get(key), (int, float)) or isinstance(row[key], bool)
               or not math.isfinite(row[key]) or row[key] < 0 for key in keys):
            raise ValueError("contention needs finite, nonnegative measured costs")
        if name != "sana-1024" and row.get("steps") != 20:
            raise ValueError("contention retains the twenty-step recipe")
        if study_api.service_ms(profile, name) <= 0:
            raise ValueError("contention needs positive service costs")


def _calibration(study, logical):
    job = report.current_job(study, logical)
    ev = report.evidence(study, job)
    if not ev["qualified"] or not ev["capacity"]:
        raise ValueError("qualified calibration with sampled capacity required: " + logical)
    cells = list(report.native_arms(ev["result"]).values())
    if len(cells) != 3 or any(cell.get("status") != "completed" or not cell.get("requests")
                              or cell.get("elapsed_ms", 0) <= 0
                              or any(row.get("status") != "completed" for row in cell["requests"])
                              for cell in cells):
        raise ValueError("three complete calibration observations required: " + logical)
    launch = Path(study["root"]) / "launches" / job["id"] / "launch.json"
    bindings = [study_api.binding(ev["path"]), study_api.binding(launch)]
    return ev, cells, bindings


def _anchors(profile, cells):
    starts, finishes = [], []
    for cell in cells:
        videos = [row for row in cell["requests"] if row["kind"] == "video"]
        first = videos[:2]
        if len(first) != 2 or any(row["spec"].get("frames") != 33 or
                                  row.get("status") != "completed" for row in first):
            raise ValueError("arrival anchors require two initial completed 33-frame videos")
        for row in first:
            if not 0 <= row["start_ms"] < row["finish_ms"]:
                raise ValueError("invalid calibration video interval")
        starts.append(max(row["start_ms"] for row in first))
        finishes.append(sorted(row["finish_ms"] for row in first))
    start = statistics.median(starts)
    first_finish = statistics.median(row[0] for row in finishes)
    last_finish = statistics.median(row[1] for row in finishes)
    long = profile["classes"][KINDS[1]]
    short = profile["classes"][KINDS[0]]
    dit = 20 * long["step_compute_ms"]
    second_finish = last_finish + study_api.service_ms(profile, KINDS[0], include_encode=False)
    centers = [start + .30 * dit, start + .65 * dit,
               first_finish - .5 * long["finish_ms"], second_finish - .5 * short["finish_ms"]]
    return dict(initial_video_start_ms=start, first_video_finish_ms=first_finish,
                last_initial_video_finish_ms=last_finish, predicted_second_finish_ms=second_finish,
                centers_ms=centers,
                scope="Frozen calibration-derived host timing; actual dispatch exposure must be checked")


def contention_trace(profile, seed, anchors):
    """Four queued videos plus four image clusters, identical for every arm."""
    _profile(profile)
    rng = random.Random(seed)
    rows = []
    for index, frames in enumerate((33, 33, 17, 17)):
        name = KINDS[1] if frames == 33 else KINDS[0]
        rows.append(dict(request_id=f"contention-{seed}-v{index}", kind="video",
                         **{"class": name}, arrival_ms=0.,
                         deadline_ms=2.5 * study_api.service_ms(profile, name),
                         spec=dict(prompt=study_api.PROMPTS[index], seed=seed * 100 + index,
                                   height=480, width=832, frames=frames, steps=20,
                                   guidance_scale=5., negative_prompt="")))
    image_ms = profile["classes"]["sana-1024"]["service_ms"]
    for cluster, center in enumerate(anchors["centers_ms"]):
        spacing = (.15 if cluster < 2 else .60) * image_ms
        jitter = rng.uniform(-.25, .25) * image_ms
        for index in range(5):
            arrival = max(0., center + jitter + (index - 2) * spacing)
            ordinal = cluster * 5 + index
            rows.append(dict(request_id=f"contention-{seed}-i{ordinal}", kind="image",
                             **{"class": "sana-1024"}, arrival_ms=arrival,
                             deadline_ms=arrival + 2.5 * image_ms,
                             spec=dict(prompt=study_api.PROMPTS[ordinal % 4],
                                       seed=seed * 100 + 4 + ordinal)))
    return sorted(rows, key=lambda row: (row["arrival_ms"], row["request_id"]))


def _estimate(profile, rows, mixed_ev, mixed_cells, arms):
    wall = mixed_ev["launch"].get("wall_seconds")
    measured = sum(cell["elapsed_ms"] for cell in mixed_cells) / 1000
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < measured:
        raise ValueError("calibration launch wall time must include all three observations")
    overhead = (wall - measured) / len(mixed_cells)
    reference_images = min(sum(row["kind"] == "image" for row in cell["requests"]) for cell in mixed_cells)
    image_ms = profile["classes"]["sana-1024"]["service_ms"]
    extra = max(0, sum(row["kind"] == "image" for row in rows) - reference_images) * image_ms / 1000
    gpu_work = sum(study_api.service_ms(profile, row["class"], include_encode=False) for row in rows) / 1000
    observation = max(max(cell["elapsed_ms"] for cell in mixed_cells) / 1000,
                      max(row["arrival_ms"] for row in rows) / 1000 + 5 * image_ms / 1000)
    # Empirical identical-worker startup/warmup cost, additional image work,
    # ten percent of serial GPU work for contention, then ten percent reserve.
    estimate = int(math.ceil(arms * (overhead + observation + extra + .10 * gpu_work) * 1.10 / 10) * 10)
    return estimate, dict(reference_launch_wall_seconds=wall,
                         per_arm_overhead_seconds=overhead, per_arm_observation_seconds=observation,
                         extra_images_seconds=extra, serial_gpu_work_seconds=gpu_work,
                         contention_allowance_fraction=.10, reserve_fraction=.10, arms=arms)


def _bind_job(study, identifier, config, *, family, requires, estimate, metadata, bindings):
    existing = any(row.get("logical_id", row["id"]) == identifier and not row.get("superseded")
                   for row in study["jobs"])
    job = study_api.add_job(study, identifier, family, config, native=True, devices=(0, 1),
                            requires=requires, estimate=estimate, timeout=1000, metadata=metadata)
    base = [study_api.binding(job["input"]), study_api.binding(job["environment_input"])]
    sources = {row["path"]: row for row in bindings}
    expected = base + [sources[path] for path in sorted(sources)]
    if existing and job["input_bindings"] != expected:
        raise ValueError("cannot replace frozen contention source bindings")
    job["input_bindings"] = expected
    study_api.save(study)
    return job


def contention_jobs(study, profile, calibration, cpu=None):
    """Register only the three screening jobs; no CUDA or execution is started."""
    _profile(profile)
    cpu = _controls(cpu)
    capacities = calibration.get("capacities", calibration)
    if any(not isinstance(capacities.get(name), (int, float)) or not math.isfinite(capacities[name])
           or capacities[name] <= 0 for name in ("video", "mixed")):
        raise ValueError("positive frozen video and mixed calibration rates required")
    unused, video_cells, video_bindings = _calibration(study, "calibration-video")
    mixed_ev, mixed_cells, mixed_bindings = _calibration(study, "calibration-mixed")
    anchors = _anchors(profile, video_cells)
    planned = []
    for repeat, seed in enumerate(SEEDS):
        rows = contention_trace(profile, seed, anchors)
        arms = copy.deepcopy(POLICIES[repeat:] + POLICIES[:repeat])
        config = study_api.cluster_config(profile, rows, arms, **cpu)
        for phase in config["phases"]:
            phase["record_dispatch_timeline"] = True
        estimate, costs = _estimate(profile, rows, mixed_ev, mixed_cells, len(arms))
        if estimate > 600:
            raise ValueError("measured contention estimate exceeds the planned 600-second screening ceiling")
        planned.append((seed, config, estimate, dict(
            comparison="contention", role="screening", seed=seed, trace_sha256=study_api.digest(rows),
            pattern="two_dit_clusters_and_two_finish_probes", absolute_arrivals=True,
            anchors=anchors, calibrated_rates=dict(capacities), estimate_basis=costs,
            trigger_requirement="step baseline: >=25% images wait >=0.5 image service and overlap a video dispatch",
            diagnostic_scope="Host dispatch intervals and planner reasons; no kernel-overlap or preemption-count claim")))
    return [_bind_job(study, f"contention-s{seed}", config, family="mixed",
                      requires=["acceptance", "calibration-video", "calibration-mixed"],
                      estimate=estimate, metadata=metadata, bindings=video_bindings + mixed_bindings)
            for seed, config, estimate, metadata in planned]


def _union_ms(intervals):
    total, end = 0., None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


def trigger_summary(result):
    """Measure host wait/dispatch exposure; missing timelines are unavailable."""
    values = {}
    phases = {phase["id"]: phase for phase in result.get("config", {}).get("phases", [])}
    for name, cell in report.native_arms(result).items():
        rows = cell.get("requests", [])
        images = [row for row in rows if row.get("kind") == "image"]
        phase = phases.get(name, {})
        service = phase.get("profile", {}).get("classes", {}).get("sana-1024", {}).get("service_ms")
        options = phase.get("options", {})
        short_limit = options.get("late_short_ms", 2000)
        protect_short = (options.get("demote_late", True) is True
                         and isinstance(short_limit, (int, float))
                         and not isinstance(short_limit, bool) and math.isfinite(short_limit)
                         and isinstance(service, (int, float)) and 0 < service <= short_limit)
        waits = [max(0., row["start_ms"] - row["arrival_ms"]) for row in images if "start_ms" in row]
        summary = dict(image_requests=len(images), image_wait_p50_ms=report.percentile(waits, .5),
                       image_wait_p95_ms=report.percentile(waits, .95), timeline_available=False,
                       pressure_triggered=False, late_short_triggered=False, decode_triggered=False,
                       unavailable_reason="missing or invalid complete host dispatch timelines")
        values[name] = summary
        if (not rows or not images or not isinstance(service, (int, float))
                or isinstance(service, bool) or not math.isfinite(service) or service <= 0
                or len(waits) != len(images)):
            continue
        events = []
        valid = True
        for row in rows:
            timeline = row.get("dispatch_timeline")
            if not isinstance(timeline, list) or not timeline:
                valid = False
                break
            for event in timeline:
                start, finish, group = event.get("start_ms"), event.get("finish_ms"), event.get("group")
                if (not isinstance(start, (int, float)) or not isinstance(finish, (int, float))
                        or not math.isfinite(start) or not math.isfinite(finish) or not 0 <= start <= finish
                        or not isinstance(group, list) or not group
                        or any(type(gpu) is not int or gpu < 0 for gpu in group)
                        or "quantum" not in event or not isinstance(event.get("reason"), str)):
                    valid = False
                    break
                events.append((row, event))
        if not valid:
            continue
        reasons = {}
        for unused, event in events:
            reasons[event["reason"]] = reasons.get(event["reason"], 0) + 1
        video_events = [(row, event) for row, event in events if row["kind"] == "video"]
        waiting, exposed, decode, late, protected, overlaps = [], [], [], [], [], []
        for row in images:
            identifier = row["request_id"]
            start, arrival = row["start_ms"], row["arrival_ms"]
            target = set(row["dispatch_timeline"][0]["group"])
            spans, finish_spans = [], []
            for unused, event in video_events:
                a, b = max(arrival, event["start_ms"]), min(start, event["finish_ms"])
                if a < b and target.intersection(event["group"]):
                    spans.append((a, b))
                    if event["quantum"] == "finish":
                        finish_spans.append((a, b))
            if start - arrival >= .5 * service:
                waiting.append(identifier)
            if spans:
                exposed.append(identifier)
            if finish_spans:
                decode.append(identifier)
            if start > row["deadline_ms"]:
                late.append(identifier)
            # The planner protects short *remaining service*, not requests
            # delayed by 2000 ms. Reconstruct its applicability at the
            # observed dispatch; this does not prove an EDF ordering change.
            if protect_short and start + service > row["deadline_ms"]:
                protected.append(identifier)
            overlaps.append(dict(request_id=identifier, video_dispatch_wait_ms=_union_ms(spans),
                                 video_finish_dispatch_wait_ms=_union_ms(finish_spans)))
        minimum = max(1, math.ceil(len(images) * .25))
        summary.update(timeline_available=True, unavailable_reason=None,
                       substantial_wait_image_ids=waiting, video_dispatch_wait_image_ids=exposed,
                       video_finish_wait_image_ids=decode, late_start_image_ids=late,
                       late_short_eligible_image_ids=protected,
                       late_short_video_exposed_image_ids=sorted(set(protected).intersection(exposed)),
                       late_short_rule="demote_late and 0 < image service <= late_short_ms and dispatch start + service > deadline; at least two also waited across a same-GPU video dispatch",
                       late_short_scope="Applicability reconstructed at host dispatch time; not proof of an ordering change",
                       dispatch_reason_counts=reasons, image_wait_intersections=overlaps,
                       required_pressure_images=minimum,
                       pressure_triggered=len(set(waiting).intersection(exposed)) >= minimum,
                       late_short_triggered=len(set(protected).intersection(exposed)) >= 2,
                       decode_triggered=len(decode) >= 2)
    return dict(schema="ditserve.workflow.contention-triggers.v1", arms=values,
                scope="Observed host dispatch windows on the eventual image GPU; not GPU kernel overlap or causal preemption counts")


def contention_summary(study):
    """Read-only summary used by the driver and by conditional ablation planning."""
    rows = []
    for seed in SEEDS:
        logical = f"contention-s{seed}"
        try:
            job = report.current_job(study, logical)
        except ValueError:
            rows.append(dict(job=logical, qualified=False, reason="screening job not registered"))
            continue
        ev = report.evidence(study, job)
        comparisons = [report.comparison(ev["result"], "step", mode, capacity=ev["capacity"],
                                         guarded=ev["qualified"]) for mode in ("block100", "block300")]
        triggers = trigger_summary(ev["result"])
        qualified = (all(row["eligible"] for row in comparisons)
                     and triggers["arms"].get("step", {}).get("pressure_triggered") is True)
        rows.append(dict(job=job["id"], qualified=qualified, guarded=ev["guarded"], capacity=ev["capacity"],
                         comparisons=comparisons, triggers=triggers,
                         result_binding=study_api.binding(ev["path"]) if ev["path"].exists() else None))
    qualified = len(rows) == len(SEEDS) and all(row["qualified"] for row in rows)
    quantum = 300
    if qualified:
        advantages, safe = [], True
        for row in rows:
            by_mode = {item["candidate"]: item["candidate_summary"] for item in row["comparisons"]}
            left, right = by_mode["block300"], by_mode["block100"]
            safe = safe and (right["total"]["throughput_per_second"] >= .95 * left["total"]["throughput_per_second"]
                             and right["video"]["latency_p95_ms"] <= 1.05 * left["video"]["latency_p95_ms"]
                             and right["image"]["slo_attainment"] >= left["image"]["slo_attainment"])
            advantages.append(1 - right["image"]["latency_p95_ms"] / left["image"]["latency_p95_ms"])
        if safe and statistics.median(advantages) >= .20:
            quantum = 100
    return dict(schema="ditserve.workflow.contention-summary.v1", qualified=qualified, jobs=rows,
                selected_quantum_ms=quantum if qualified else None,
                selection_rule="100 only with median image p95 reduction >=20% versus 300 and no per-trace image SLO loss, >5% video p95 or throughput regression; otherwise 300",
                scope="Three screening traces; targeted ablations are not independent confirmation")


def contention_ablation_jobs(study, profile, cpu=None):
    """At most two paired jobs, each on an already observed triggering trace."""
    _profile(profile)
    cpu = _controls(cpu)
    summary = contention_summary(study)
    if not summary["qualified"]:
        return []
    quantum = summary["selected_quantum_ms"]
    mode = "block" + str(quantum)
    mixed_ev, mixed_cells, bindings = _calibration(study, "calibration-mixed")
    jobs = []
    for variable, trigger, change in (
            ("late-short", "late_short_triggered", dict(late_short_ms=0)),
            ("decode", "decode_triggered", dict(chunked_decode=False))):
        # Choose by observed trigger presence, never by the observed speedup.
        eligible = [row for row in summary["jobs"]
                    if row["triggers"]["arms"].get(mode, {}).get(trigger) is True]
        if not eligible:
            continue
        source = report.current_job(study, eligible[0]["job"])
        original = study_api.read(source["input"])
        if study_api.digest(original) != source["configuration_sha256"]:
            raise ValueError("frozen contention screening input changed")
        phase = next(row for row in original["phases"] if row["id"] == mode)
        if phase["profile"] != profile:
            raise ValueError("ablation must retain the frozen screening profile")
        rows = copy.deepcopy(phase["workload"])
        arms = [dict(id="reference", policy="block_elastic", quantum_ms=quantum),
                dict(id="candidate", policy="block_elastic", quantum_ms=quantum, **change)]
        if variable == "decode":
            arms.reverse()
        config = study_api.cluster_config(profile, rows, arms, **cpu)
        for row in config["phases"]:
            row["record_dispatch_timeline"] = True
        estimate, costs = _estimate(profile, rows, mixed_ev, mixed_cells, 2)
        metadata = dict(comparison="contention-ablation", variable=variable,
                        role="targeted_mechanism_ablation", source_job=source["id"],
                        trace_sha256=study_api.digest(rows), selected_quantum_ms=quantum,
                        observed_trigger=trigger, source_result=eligible[0]["result_binding"],
                        estimate_basis=costs, independent_confirmation=False,
                        selection_rule="first qualified frozen seed with the observed mechanism trigger")
        jobs.append(_bind_job(study, "contention-ablation-" + variable, config, family="confirmation",
                              requires=[row["job"] for row in summary["jobs"]], estimate=estimate,
                              metadata=metadata, bindings=bindings + [
                                  study_api.binding(source["input"]), eligible[0]["result_binding"]]))
    return jobs
