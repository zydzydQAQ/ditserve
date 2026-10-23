"""CPU-only planning and explicit registration of encoding-cache experiments.

The templates retain the existing run03 ledger, family carry rules and reserve.
Native online configurations are executable by the existing study executor only
after their numerical prerequisites, source freeze and launcher admission pass.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "ditserve.workflow.encoding-optimization-plan.v1"
SANA_SEEDS = (4101, 4102, 4103)
WAN_SEEDS = (4201, 4202, 4203)
SANA_CACHE = dict(prompt_cache_entries=4, prompt_cache_bytes=16 * 1024**2)
WAN_CACHE = dict(negative_prompt_cache_entries=4, negative_prompt_cache_bytes=64 * 1024**2)
NEGATIVE_PROMPT = "blurry, low quality, distorted geometry"
MINIMUM_RESERVE_SECONDS = 900


def _number(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(name + " must be finite and nonnegative")
    return value


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _budget_snapshot(study, budget):
    limit = _number(study["budget_seconds"], "study budget")
    if _number(budget["limit_seconds"], "ledger limit") != limit:
        raise ValueError("study and existing ledger limits differ")
    charged = _number(budget["charged_seconds"], "charged seconds")
    caps = study["family_caps_seconds"]
    if not caps or list(caps)[-1] != "reserve" or "extension" not in caps:
        raise ValueError("ordered family caps must end with reserve and include extension")
    for name, cap in caps.items():
        _number(cap, "family cap " + name)
    reserve = caps["reserve"]
    if reserve < MINIMUM_RESERVE_SECONDS:
        raise ValueError("encoding plan cannot lower the existing 900-second reserve")
    jobs = {job["id"]: job for job in study["jobs"]}
    if len(jobs) != len(study["jobs"]):
        raise ValueError("historical job IDs must be unique")
    spent = dict.fromkeys(caps, 0.)
    accounted = unknown = 0.
    for attempt in budget.get("attempts", []):
        cost = _number(attempt["charged_seconds"], "attempt charge")
        accounted += cost
        family = jobs.get(attempt["job_id"], {}).get("family")
        if family in spent:
            spent[family] += cost
        else:
            unknown += cost
    if accounted > charged + 1e-6:
        raise ValueError("attempt charges exceed the cumulative ledger charge")
    # Missing historical attribution never creates a free family allowance.
    unknown += max(0., charged - accounted)
    remaining = max(0., limit - charged)
    spendable = max(0., remaining - reserve)
    available, prefix = {}, 0.
    for name, cap in caps.items():
        if name == "reserve":
            break
        prefix += cap - spent[name]
        available[name] = max(0., min(spendable, prefix - unknown))
    return dict(limit_seconds=limit, charged_seconds=charged, remaining_seconds=remaining,
                reserve_seconds=reserve, spendable_seconds=spendable,
                family_available_seconds=available, family_spent_seconds=spent,
                unattributed_charged_seconds=unknown,
                active=bool(study.get("active_job") or budget.get("active")),
                admission_rule="estimated_seconds * 1.2 + 90 per complete job",
                family_rule="ordered cumulative family caps minus historical charges; reserve excluded")


def _job(identifier, optimization, stage, estimates, config, requires=()):
    low, high = estimates
    return dict(id=identifier, logical_id=identifier, family="extension",
                devices=[1] if stage == "qualification" else [0, 1],
                optimization=optimization, stage=stage, requires=list(requires),
                execution_status="configuration_template", qualification_status="unverified",
                runner=("ditserve.evaluation.workflow_encoding_qualification"
                        if stage == "qualification" else "native_cluster"),
                estimated_seconds=high, estimate_range_seconds=[low, high],
                estimate_basis="conservative planning range including loading, warmup and cleanup; not a measurement",
                admission_required_seconds=high * 1.2 + 90,
                admission_range_seconds=[low * 1.2 + 90, high * 1.2 + 90],
                config=copy.deepcopy(config), ready_for_gpu=False)


def _templates():
    """The fixed matrix, independent of a historical admission snapshot."""
    jobs = []
    for distribution in ("unique", "repeated"):
        jobs.append(_job("encoding-sana-" + distribution + "-numeric", "sana_prompt_cache",
            "qualification", (240, 360), dict(seed=4100, prompt_distribution=distribution,
                arms=["disabled", "staged", "fused"], image_size=[1024, 1024],
                requests=8, distinct_prompts=8 if distribution == "unique" else 4,
                cache_start="same adapter per arm; explicitly clear entries after warmup for a cold cache",
                warmup="complete warmup on the same adapter, then clear entries; retain counters and warmup audit",
                cache_options=SANA_CACHE, fused_options=dict(prompt_cache_sync_mode="fused"),
                expected_cache_access=("eight compulsory misses and zero hits" if distribution == "unique" else
                                       "four compulsory misses followed by four hits on one adapter"),
                numeric="all final image hashes exact against disabled for each request",
                performance_evidence=False,
                audit=["cache key and hit/miss provenance", "backing storage byte bound",
                       "capacity", "finite outputs", "miss and hit paths both executed"])))
    for branch, estimate in (("guided", (1200, 1800)), ("cfg-bypass", (600, 900))):
        jobs.append(_job("encoding-wan-" + branch + "-numeric", "wan_negative_cache",
            "qualification", estimate, dict(seed=4200, frames=[17, 33], height=480, width=832,
                steps=20, cpu_threads=8, text_prefetch_depth=1, arms=["disabled", "enabled"],
                guidance_scale=5. if branch == "guided" else 1.,
                negative_prompts=["", NEGATIVE_PROMPT], repeats_per_case=2,
                cache_start=("empty new adapter; every negative-cache access bypassed" if branch == "cfg-bypass" else
                             "empty new adapter per arm; explicit cold/miss then repeat/hit"),
                cache_options=WAN_CACHE, negative_embedding_dtype="float32", negative_embedding_bytes=8388608,
                expected_bypass=branch == "cfg-bypass",
                numeric="ordinary uncached reference, all twenty step latents and final videos exact",
                performance_evidence=False,
                audit=["CFG <= 1 does not encode or populate negative cache",
                       "positive embeddings are never reused", "bounded independent storage",
                       "empty and nonempty negative strings remain distinct", "capacity", "state release"])))
    for optimization, seeds in (("sana_prompt_cache", SANA_SEEDS), ("wan_negative_cache", WAN_SEEDS)):
        prerequisites = [job["id"] for job in jobs
                         if job["stage"] == "qualification" and job["optimization"] == optimization]
        distributions = ("unique", "repeated") if optimization == "sana_prompt_cache" else ("mixed_negative",)
        for repeat, seed in enumerate(seeds):
            for distribution in distributions:
                jobs.append(_job(f"encoding-{optimization}-{distribution}-s{seed}", optimization,
                    "online_confirmation", (550, 900) if optimization == "sana_prompt_cache" else (400, 700),
                    dict(seed=seed, repeat=repeat, prompt_distribution=distribution,
                         requests=24, images=20, videos=4, frames=[17, 33], steps=20,
                         cpu_threads=8, text_prefetch_depth=1, quantum_ms=300, decode_frames=1,
                         late_short_ms=2000, record_dispatch_timeline=True,
                         inputs="new calibration-derived arrivals/deadlines frozen equally for every arm",
                         cache_warmup="audit warmup entries separately; unique image prompts exclude warmup prompt",
                         timing="arrival to final output including encoding, misses, checks and queueing",
                         diagnostics="no step-array saving or profiler inside normal timing",
                         independent_repetition="one trajectory per seed; requests are not independent repeats",
                         comparisons=(["disabled/staged", "disabled/fused", "staged/fused"]
                                      if optimization == "sana_prompt_cache" else ["disabled/enabled"]),
                         metrics=["image response and wait p50/p95", "video response p50/p95",
                                  "throughput", "SLO by request kind", "encode and host service ms",
                                  "hit/miss/bypass counts and retained bytes", "peak and steady memory"],
                         numeric="complete paired final outputs and video latent hashes exact",
                         decision="report each seed and median; no automatic production default change"),
                    requires=prerequisites + ["acceptance", "calibration-video", "calibration-mixed"]))
    return jobs


def build_plan(study, budget):
    """Return a deterministic reviewable plan without changing either input.

    ``study`` and ``budget`` are parsed snapshots of the *same existing* run.
    Costs are deliberately ranges, not calibrated measurements. Their upper
    ends supply conservative per-job admission estimates. This function cannot
    authorize a budget extension, register jobs, freeze files or run the GPU.
    """
    snapshot = _budget_snapshot(study, budget)
    jobs = _templates()
    for job in jobs:
        available = snapshot["family_available_seconds"][job["family"]]
        job["budget_status"] = ("deferred_active_budget" if snapshot["active"] else
                                "deferred_budget" if available < job["admission_required_seconds"] else
                                "estimate_fits_current_snapshot")
        job["available_seconds_at_snapshot"] = available
    totals = {}
    for stage in ("qualification", "online_confirmation", "complete_matrix"):
        selected = [job for job in jobs if stage == "complete_matrix" or job["stage"] == stage]
        estimate = [sum(job["estimate_range_seconds"][i] for job in selected) for i in (0, 1)]
        admission = [sum(job["admission_range_seconds"][i] for job in selected) for i in (0, 1)]
        available = snapshot["family_available_seconds"]["extension"]
        totals[stage] = dict(jobs=len(selected), estimate_range_seconds=estimate,
            admission_range_seconds=admission,
            additional_available_budget_range_seconds=[max(0., value - available) for value in admission],
            budget_status="deferred_budget" if admission[1] > available else "estimate_fits_current_snapshot")
    return dict(schema=SCHEMA, run_id=study["run_id"], root=study["root"],
                status="deferred_active_budget" if snapshot["active"] else totals["complete_matrix"]["budget_status"],
                ready_for_gpu=False, budget=snapshot, stage_costs=totals, jobs=jobs,
                source_snapshot=dict(source=study.get("source"), source_digest=study.get("source_digest")),
                input_snapshot_sha256=_digest(dict(study=study, budget=budget)),
                prerequisites=["freeze new source revision and pass CPU checks",
                               "complete new GPU numerical qualification and independent cache audits",
                               "bind unchanged model/profile/calibration and new trajectory hashes",
                               "obtain additional cumulative/family budget through the existing owner",
                               "existing exclusive launcher rechecks complete-job budget and GPU guard"],
                prohibitions=["do not edit historical jobs/results/ledger", "do not lower reserve",
                              "do not reset charges or move to a fresh run to bypass the budget",
                              "do not claim performance qualification from numerical diagnostics"])


def native_qualification_config(job, *, gpu_uuid):
    """Bind one numeric template to physical GPU 1 for the isolated runner.

    CUDA sees only the one admitted UUID as ``cuda:0``. The runner generates
    the fixed eight cases and arm options; these are not arbitrary user specs.
    Returning a configuration does not grant numerical or budget qualification.
    """
    if (job.get("stage") != "qualification" or job.get("devices") != [1]
            or job.get("runner") != "ditserve.evaluation.workflow_encoding_qualification"):
        raise ValueError("an isolated GPU-1 numerical qualification job is required")
    if not isinstance(gpu_uuid, str) or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid):
        raise ValueError("qualification requires the verified physical GPU UUID")
    config = job["config"]
    seed = config["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("qualification seed must be a nonnegative integer")
    common = dict(device="cuda:0", gpu_uuid=gpu_uuid, cpu_mask=list(range(60, 68)),
                  cpu_threads=8, seed=seed, sample_interval=.05, arm_timeout_seconds=1800)
    if job["optimization"] == "sana_prompt_cache":
        distribution = config["prompt_distribution"]
        if (distribution not in ("unique", "repeated") or config["requests"] != 8
                or config["distinct_prompts"] != (8 if distribution == "unique" else 4)
                or config["arms"] != ["disabled", "staged", "fused"]
                or config["cache_options"] != SANA_CACHE):
            raise ValueError("Sana qualification must preserve the frozen eight-case cache matrix")
        return dict(common, kind="sana", model_path="/models/sana-sprint-0.6b",
                    prompt_distribution=distribution)
    if job["optimization"] == "wan_negative_cache":
        if (config["frames"] != [17, 33] or config["height"] != 480 or config["width"] != 832
                or config["steps"] != 20 or config["repeats_per_case"] != 2
                or config["guidance_scale"] not in (1., 5.)
                or isinstance(config["guidance_scale"], bool)
                or config["negative_prompts"] != ["", NEGATIVE_PROMPT]
                or config["arms"] != ["disabled", "enabled"] or config["cache_options"] != WAN_CACHE):
            raise ValueError("Wan qualification must preserve the frozen eight-case cache matrix")
        return dict(common, kind="wan", model_path="/models/wan2.1-t2v-1.3b",
                    guidance_scale=config["guidance_scale"])
    raise ValueError("unsupported numerical qualification optimization")


def native_online_config(profile, rows, *, optimization, prompt_distribution, repeat):
    """Build paired native configs from one new calibrated 24-request trace.

    Arrivals, deadlines, seeds and mathematical recipes are retained. Prompt
    distributions below become part of the returned input and must be frozen.
    The source rows and profile are never mutated and no GPU is initialized.
    """
    from . import workflow_next_study as api
    from .workflow_contention_study import _profile

    _profile(profile)
    if type(repeat) is not int or repeat not in (0, 1, 2):
        raise ValueError("repeat must identify one of three independent seeds")
    distributions = {"sana_prompt_cache": ("unique", "repeated"),
                     "wan_negative_cache": ("mixed_negative",)}
    if optimization not in distributions or prompt_distribution not in distributions[optimization]:
        raise ValueError("unsupported optimization or prompt distribution")
    workload = copy.deepcopy(rows)
    images = [row for row in workload if row["kind"] == "image"]
    videos = [row for row in workload if row["kind"] == "video"]
    if (len(workload) != 24 or len(images) != 20 or len(videos) != 4
            or len({row["request_id"] for row in workload}) != 24
            or sorted(row["spec"].get("frames") for row in videos) != [17, 17, 33, 33]):
        raise ValueError("online comparison requires twenty images and two videos of each frame length")
    for row in workload:
        arrival = _number(row["arrival_ms"], "arrival")
        if _number(row["deadline_ms"], "deadline") <= arrival:
            raise ValueError("deadline must follow arrival")
        expected_class = (f"wan1.3b-480x832x{row['spec']['frames']}"
                          if row["kind"] == "video" else "sana-1024")
        if row["class"] != expected_class:
            raise ValueError("request class must match its measured model and shape")
    for row in videos:
        spec = row["spec"]
        if (spec.get("height"), spec.get("width"), spec.get("steps"), spec.get("guidance_scale")) != (480, 832, 20, 5.):
            raise ValueError("online comparison preserves the 480x832, twenty-step, CFG=5 recipe")
    if optimization == "sana_prompt_cache":
        for index, row in enumerate(images):
            row["spec"]["prompt"] = (f"A ceramic sculpture labelled specimen {index + 1:02d} on a wooden table."
                                     if prompt_distribution == "unique" else api.PROMPTS[index % 4])
        names = ["disabled", "staged", "fused"]
        names = names[repeat:] + names[:repeat]
    else:
        counts = {17: 0, 33: 0}
        for row in videos:
            frames = row["spec"]["frames"]
            row["spec"]["negative_prompt"] = "" if counts[frames] == 0 else NEGATIVE_PROMPT
            counts[frames] += 1
        names = ["disabled", "enabled"] if repeat % 2 == 0 else ["enabled", "disabled"]
    arms = [dict(id=name, policy="block_elastic", quantum_ms=300, decode_frames=1,
                 chunked_decode=True, late_short_ms=2000, demote_late=True) for name in names]
    config = api.cluster_config(profile, workload, arms, cores=8, depth=1)
    for phase in config["phases"]:
        phase["controller_cpu_affinity"] = [59]
        phase["record_dispatch_timeline"] = True
        for rank in (0, 1):
            sana = copy.deepcopy(api._worker("sana", rank, rank, 8)["config"])
            sana["options"] = {}
            wan = phase["worker_overrides"][f"wan{rank}"]
            if optimization == "sana_prompt_cache" and phase["id"] != "disabled":
                sana["options"] = dict(SANA_CACHE, prompt_cache_sync_mode=phase["id"])
            elif optimization == "wan_negative_cache" and phase["id"] == "enabled":
                wan["options"].update(WAN_CACHE)
            phase["worker_overrides"][f"sana{rank}"] = sana
    return config


def register(study, proposal):
    """Append the frozen four numeric and nine online jobs; never launch them.

    The proposal's admission snapshot is historical. Current admission belongs
    to the launcher, including an explicitly authorized unlimited budget. Its
    original study/ledger are checked against the pre-authorization checkpoint;
    every other proposal input must still match. All validation happens before
    adding a job. Numeric jobs run on physical GPU 1 exposed as CUDA device 0.
    """
    from . import workflow_contention_study as contention
    from . import workflow_encoding_qualification as qualification
    from . import workflow_next_study as api

    path = Path(proposal).resolve()
    if path.is_dir():
        path /= "plan.json"
    plan = api.read(path)
    if (not isinstance(plan, dict) or plan.get("schema") != SCHEMA
            or plan.get("run_id") != study["run_id"]
            or Path(plan.get("root", "")).resolve() != Path(study["root"]).resolve()):
        raise ValueError("proposal does not belong to this existing study")
    if study.get("active_job"):
        raise ValueError("cannot register while a study job is active")
    root = Path(study["root"])
    checkpoint = root / "checkpoints/before-encoding-unlimited-r012/work"
    bindings = [api.binding(path)]
    old_snapshots = {}
    for frozen in plan["input_bindings"]:
        source = Path(frozen["path"])
        if source.parent == root and source.name in ("study.json", "budget.json"):
            source = checkpoint / source.name
            actual = api.binding(source)
            if any(actual[key] != frozen[key] for key in ("bytes", "sha256")):
                raise ValueError("pre-authorization checkpoint differs from proposal: " + source.name)
            old_snapshots[source.stem] = api.read(source)
        else:
            actual = api.binding(source)
            if actual != frozen:
                raise ValueError("proposal input changed: " + str(source))
        bindings.append(actual)
    if (set(old_snapshots) != {"study", "budget"}
            or _digest(old_snapshots) != plan.get("input_snapshot_sha256")):
        raise ValueError("proposal historical study and budget snapshot binding is incomplete")
    historical = {row["id"]: row for row in study["jobs"]}
    if any(historical.get(row["id"]) != row for row in old_snapshots["study"]["jobs"]):
        raise ValueError("historical jobs were modified after proposal preparation")
    for frozen in plan["environment_bindings"]:
        if api.binding(frozen["path"]) != frozen:
            raise ValueError("proposal environment changed")
        bindings.append(frozen)
    if (api.read(path.parent / "environment.json") != api.environment(study, (0, 1))
            or api.read(path.parent / "qualification-environment.json") != api.environment(study, (1,))):
        raise ValueError("proposal GPU environment no longer matches the study")
    profile = api.read(root / "profile.json")
    _, cells, calibration_bindings = contention._calibration(study, "calibration-video")
    bindings.extend(calibration_bindings)
    anchors = contention._anchors(profile, cells)
    numeric_rows, online_rows = plan["qualification_configs"], plan["native_configs"]
    manifest = {row["id"]: row for row in numeric_rows + online_rows}
    if len(numeric_rows) != 4 or len(online_rows) != 9 or len(manifest) != 13:
        raise ValueError("proposal must contain four numeric and nine online configurations")
    prepared = []
    for template in _templates():
        numeric = template["stage"] == "qualification"
        controls = template["config"]
        identifier = (template["id"] if numeric else "%s-%s-s%d" % (
            template["optimization"].replace("_", "-"), controls["prompt_distribution"], controls["seed"]))
        entry = manifest.get(identifier)
        if entry is None or api.binding(entry["binding"]["path"]) != entry["binding"]:
            raise ValueError("missing or changed proposal configuration: " + identifier)
        config = api.read(entry["binding"]["path"])
        if api.digest(config) != entry["configuration_sha256"]:
            raise ValueError("proposal configuration digest differs: " + identifier)
        if numeric:
            expected = qualification.normalize_config(config["kind"], native_qualification_config(
                template, gpu_uuid=study["gpu_mapping"][1]["uuid"]))
        else:
            rows = contention.contention_trace(profile, controls["seed"], anchors)
            expected = native_online_config(profile, rows, optimization=template["optimization"],
                prompt_distribution=controls["prompt_distribution"], repeat=controls["repeat"])
            expected["experiment"] = "encoding-optimization-" + identifier
            expected["budget"] = dict(limit_gpu_seconds=3600, watchdog_seconds=1800,
                                      repair_reserve_gpu_seconds=0, carry="none")
            for phase in expected["phases"]:
                phase["artifact_dependencies"] = [dict(closed=False,
                    reason="encoding optimization numerical and performance review pending")]
        if config != expected:
            raise ValueError("proposal differs from fixed calibrated encoding matrix: " + identifier)
        numeric_ids = [job["id"] for job in _templates() if job["stage"] == "qualification"
                       and job["optimization"] == template["optimization"]]
        metadata = dict(role="encoding_numeric" if numeric else "encoding_online",
            optimization=template["optimization"], stage=template["stage"], seed=controls["seed"],
            prompt_distribution=controls.get("prompt_distribution", "cfg-bypass" if controls.get(
                "guidance_scale") == 1 else "guided"), proposal=str(path),
            estimate_range_seconds=template["estimate_range_seconds"],
            estimate_basis=template["estimate_basis"], performance_evidence=not numeric,
            numeric_prerequisites=[] if numeric else numeric_ids,
            historical_budget_snapshot=plan["budget"], historical_input_bindings=plan["input_bindings"])
        if not numeric:
            metadata["repeat"] = controls["repeat"]
        requires = ["acceptance"] if numeric else numeric_ids + ["acceptance", "calibration-video", "calibration-mixed"]
        prepared.append((template, identifier, config, metadata, requires, entry["binding"]))
    # Bind the original proposal and immutable calibration/checkpoint files, not
    # mutable study.json or budget.json. The job itself captures the new source.
    bindings = list({row["path"]: row for row in bindings}.values())
    jobs = []
    for template, identifier, config, metadata, requires, config_binding in prepared:
        numeric = template["stage"] == "qualification"
        existing = any(row.get("logical_id", row["id"]) == identifier and not row.get("superseded")
                       for row in study["jobs"])
        job = api.add_job(study, identifier, "extension", config, native=not numeric,
            module="workflow_encoding_qualification" if numeric else None,
            kind=config["kind"] if numeric else None, devices=template["devices"],
            requires=requires, estimate=template["estimated_seconds"],
            timeout=(len(qualification.ARMS[config["kind"]]) * config["arm_timeout_seconds"] + 180
                     if numeric else 1800),
            metadata=metadata)
        base = [api.binding(job["input"])]
        if numeric:
            prefix = ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=" + config["gpu_uuid"]]
            command = prefix + ["/opt/venv/bin/python", "-m",
                "ditserve.evaluation.workflow_encoding_qualification", "--kind", config["kind"],
                "--config", api.container_path(study, job["input"]),
                "--out", api.container_path(study, job["output"])]
            if existing and (job.get("command_type") != "isolated_gpu" or job["command"] != command
                             or job.get("gpu_mapping") != [study["gpu_mapping"][1]]):
                raise ValueError("cannot replace frozen isolated qualification command")
            job.update(command_type="isolated_gpu", command=command,
                       gpu_mapping=[copy.deepcopy(study["gpu_mapping"][1])])
        else:
            base.append(api.binding(job["environment_input"]))
        extras = {row["path"]: row for row in bindings + [config_binding]}
        frozen_bindings = base + [extras[name] for name in sorted(extras)]
        if existing and (job["input_bindings"] != frozen_bindings or job.get("metadata") != metadata):
            raise ValueError("cannot replace frozen encoding metadata or proposal bindings")
        job["input_bindings"] = frozen_bindings
        api.save(study)
        jobs.append(job)
    return jobs
