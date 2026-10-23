"""Pure, preregistered r018 online matrices; no registration, IO or CUDA."""

import copy
import math

from . import workflow_contention_study as contention
from . import workflow_encoding_optimization_study as encoding
from . import workflow_next_study as api
from . import workflow_occupancy_study as occupancy


ROLE = "remaining_online"
SCHEMA = "ditserve.workflow.remaining-online.v1"
SEEDS = dict(quantum_repeated=(4601, 4602, 4603), quantum_unique=(4701, 4702, 4703),
             scheduler=(4801, 4802, 4803), decode=(4901, 4902, 4903),
             poisson=(5001, 5002, 5003), high_load=(5101, 5102, 5103))
SCREENS = dict(quantum_repeated=4600, quantum_unique=4700)
FOUR_ARMS = ("off300", "off200", "on300", "on200")
FOUR_PAIRS = (("off300", "off200"), ("on300", "on200"), ("off300", "on300"),
              ("off200", "on200"), ("off300", "on200"))
NUMERIC_PREREQUISITES = occupancy.NUMERIC_PREREQUISITES + (
    "encoding-sana-unique-numeric", "encoding-sana-repeated-numeric")
BASE_PREREQUISITES = tuple(dict.fromkeys(NUMERIC_PREREQUISITES +
    ("acceptance", "calibration-video", "calibration-mixed") +
    tuple(occupancy.job_id(seed) for seed in occupancy.SEEDS)))
RULE = ("All frozen confirmations execute after technical/capacity gates, irrespective of screening performance. "
        "Each seed: image/video p95 <=105%, throughput >=97%, no image SLO loss. "
        "Image-tail benefit: paired p95 reduction median >=10%, at least 2/3 improving. "
        "Unique workload is a safety control, with no required service-time gain. "
        "Screening excluded from confirmation; no automatic default change.")
PROTECTED_PREFIXES = ("workers/ditserve/models/", "workers/ditserve/runtime/", "configs/",
                      "src/", "include/", "cmd/")


def job_id(family, seed):
    if family not in SEEDS or type(seed) is not int or seed not in (*SEEDS[family], SCREENS.get(family)):
        raise ValueError("unknown preregistered remaining-online family/seed")
    return f"remaining-{family.replace('_', '-')}-s{seed}"


def arm_definitions(family):
    if family in ("quantum_repeated", "quantum_unique", "poisson", "high_load"):
        return [dict(id=name, policy="block_elastic", quantum_ms=quantum, decode_frames=1,
                     chunked_decode=True, late_short_ms=2000, demote_late=True, cache=cache)
                for name, quantum, cache in zip(FOUR_ARMS, (300, 200, 300, 200), (False, False, True, True))]
    if family == "scheduler":
        return [dict(id=f"S{i}", policy="static_sp1" if i == 0 else "block_elastic", quantum_ms=300,
                     decode_frames=1 if i not in (3, 4) else None, chunked_decode=i not in (3, 4),
                     late_short_ms=0 if i in (2, 4) else 2000, demote_late=True, cache=False)
                for i in range(5)]
    if family == "decode":
        return [dict(id=name, policy="block_elastic", quantum_ms=300, decode_frames=frames,
                     chunked_decode=frames is not None, late_short_ms=2000, demote_late=True, cache=False)
                for name, frames in (("decode1", 1), ("decode2", 2), ("decodeall", None))]
    raise ValueError("unknown remaining-online family")


def pairs(family):
    if family == "scheduler":
        return (("S0", "S1"), ("S2", "S1"), ("S3", "S1"), ("S4", "S1"), ("S4", "S2"), ("S4", "S3"))
    if family == "decode":
        return (("decodeall", "decode1"), ("decodeall", "decode2"), ("decode1", "decode2"))
    return FOUR_PAIRS


def ordered_arms(family, repeat):
    if type(repeat) is not int or repeat not in (0, 1, 2):
        raise ValueError("repeat must identify one of three independent seeds")
    arms = arm_definitions(family)
    if len(arms) == 4:
        return [arms[i] for i in ((0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1))[repeat]]
    return arms[repeat:] + arms[:repeat]


def workload(profile, anchors, family, seed):
    job_id(family, seed)
    contention._profile(profile)
    if family in ("poisson", "high_load"):
        capacity = anchors.get("mixed_capacity")
        if type(capacity) not in (int, float) or not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("positive frozen calibration capacities.mixed required in anchors.mixed_capacity")
        rows = api.trajectory(profile, seed, count=40, pattern="poisson", load=.8, lambda0=capacity)
        if family == "high_load":
            for row in rows:
                slack = row["deadline_ms"] - row["arrival_ms"]
                row["arrival_ms"] /= 1.25
                row["deadline_ms"] = row["arrival_ms"] + slack
    else:
        rows = contention.contention_trace(profile, seed, anchors)
    counts = {17: 0, 33: 0}
    image_index = 0
    for row in rows:
        if row["kind"] == "video":
            frames = row["spec"]["frames"]
            row["spec"]["negative_prompt"] = "" if counts[frames] % 2 == 0 else encoding.NEGATIVE_PROMPT
            counts[frames] += 1
        else:
            row["spec"]["prompt"] = (f"A ceramic sculpture labelled specimen {image_index + 1:02d} on a wooden table."
                                     if family == "quantum_unique" else api.PROMPTS[image_index % 4])
            image_index += 1
    return rows


def native_config(profile, rows, *, family, repeat):
    """All arms use a fresh worker lifecycle and the same fixed arrival trace."""
    arms = ordered_arms(family, repeat)
    definitions = []
    for arm in arms:
        options = {key: value for key, value in arm.items() if key != "cache" and value is not None}
        definitions.append(options)
    config = api.cluster_config(profile, rows, definitions, cores=8, depth=1)
    for phase, arm in zip(config["phases"], arms):
        phase.update(controller_cpu_affinity=[59], record_dispatch_timeline=True,
                     dispatch_refresh=False, occupancy_backend="nvml",
                     artifact_dependencies=[dict(closed=False, reason="remaining-online review pending")])
        for slot in (0, 1):
            phase["worker_overrides"][f"wan{slot}"]["options"].update(encoding.WAN_CACHE)
            sana = copy.deepcopy(api._worker("sana", slot, slot, 8)["config"])
            sana["options"] = dict(encoding.SANA_CACHE, prompt_cache_sync_mode="staged") if arm["cache"] else {}
            phase["worker_overrides"][f"sana{slot}"] = sana
    return config


def validate_config(config, metadata):
    """Reconstruct traces and every control; a recomputed digest cannot hide drift."""
    try:
        family, seed, repeat = metadata["family"], metadata["seed"], metadata["repeat"]
        identifier = job_id(family, seed)
        stage = "screening" if seed == SCREENS.get(family) else "confirmation"
        expected_repeat = 0 if stage == "screening" else SEEDS[family].index(seed)
        if (metadata["role"] != ROLE or repeat != expected_repeat or metadata["stage"] != stage
                or metadata["comparisons"] != [list(pair) for pair in pairs(family)]
                or metadata["rule"] != RULE or metadata["performance_blocks_confirmation"] is not False
                or metadata["prompt_distribution"] != ("unique" if family == "quantum_unique" else "repeated")
                or metadata["monitor_cpu_affinity"] != [58]
                or type(metadata["timeout_seconds"]) is not int or metadata["timeout_seconds"] < 3600):
            raise ValueError("remaining-online metadata differs from preregistration")
        profile = config["phases"][0]["profile"]
        rows = workload(profile, metadata["anchors"], family, seed)
        if metadata["trace_sha256"] != api.digest(rows):
            raise ValueError("frozen trace differs")
        expected = native_config(profile, rows, family=family, repeat=repeat)
        observed = copy.deepcopy(config)
        experiment = observed.pop("experiment")
        if experiment not in ("run03", "run03-" + identifier):
            raise ValueError("unexpected experiment identity")
        envelope = observed.pop("budget", None)
        if envelope is not None and envelope != dict(limit_gpu_seconds=metadata["timeout_seconds"] * 2,
                watchdog_seconds=metadata["timeout_seconds"], repair_reserve_gpu_seconds=0, carry="none"):
            raise ValueError("unexpected per-job watchdog envelope")
        expected.pop("experiment")
        if api.digest(observed) != api.digest(expected):
            raise ValueError("configuration differs from the frozen remaining-online matrix")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("incomplete remaining-online configuration") from error
    return True


def templates(study, profile, anchors):
    """Return all twenty envelopes without reading or mutating study/history."""
    contention._controls(study.get("selections", {}).get("cpu"))
    jobs = []
    for family, seeds in SEEDS.items():
        cases = ([(SCREENS[family], 0, "screening")] if family in SCREENS else [])
        cases += [(seed, repeat, "confirmation") for repeat, seed in enumerate(seeds)]
        for seed, repeat, stage in cases:
            identifier = job_id(family, seed)
            if any(j.get("metadata", {}).get("seed") == seed and j.get("logical_id", j["id"]) != identifier
                   for j in study.get("jobs", [])):
                raise ValueError("remaining-online seed already used by another experiment")
            rows = workload(profile, anchors, family, seed)
            config = native_config(profile, rows, family=family, repeat=repeat)
            # Conservative serial work + last arrival + startup allowance. This
            # is a planning estimate, not a measured runtime or speedup claim.
            serial_seconds = sum(api.service_ms(profile, row["class"]) for row in rows) / 1000
            per_arm = 120 + serial_seconds + max(row["arrival_ms"] for row in rows) / 1000
            estimate = int(math.ceil(len(config["phases"]) * per_arm * 1.2 / 10) * 10)
            timeout = max(3600, int(math.ceil(estimate * 1.5 / 600) * 600))
            requires = list(BASE_PREREQUISITES)
            if stage == "confirmation" and family in SCREENS:
                requires += [job_id(name, screen) for name, screen in SCREENS.items()]
            metadata = dict(role=ROLE, family=family, seed=seed, repeat=repeat, stage=stage,
                prompt_distribution="unique" if family == "quantum_unique" else "repeated",
                anchors=copy.deepcopy(anchors), trace_sha256=api.digest(rows),
                comparisons=[list(pair) for pair in pairs(family)], rule=RULE,
                independent_confirmation=stage == "confirmation", performance_evidence=True,
                performance_blocks_confirmation=False, numeric_prerequisites=list(NUMERIC_PREREQUISITES),
                monitor_cpu_affinity=[58], timeout_seconds=timeout, default_change=False,
                protected_source_prefixes=list(PROTECTED_PREFIXES),
                workload_scope=("preregistered exploratory 1.25x Poisson arrival compression; relative SLO budgets retained"
                    if family == "high_load" else "calibration-capacity Poisson at load .8" if family == "poisson"
                    else "unchanged calibration-derived contention anchors"),
                estimate_basis=dict(kind="planning_upper_allowance", serial_seconds=serial_seconds,
                    startup_allowance_per_arm_seconds=120, last_arrival_seconds=max(r["arrival_ms"] for r in rows)/1000,
                    reserve_fraction=.2, launcher_admission="estimated_seconds * 1.2 + 90"))
            validate_config(config, metadata)
            jobs.append(dict(id=identifier, config=config, metadata=metadata, requires=requires,
                             estimated_seconds=estimate, timeout_seconds=timeout))
    # Both calibration-style screens must precede either confirmation family.
    # Their performance outcomes never remove a frozen confirmation.
    return sorted(jobs, key=lambda row: row["metadata"]["stage"] != "screening")
