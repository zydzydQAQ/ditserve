"""Append-only SP1 dispatch-refresh screening and independent confirmation.

Planning and registration are CPU-only. The shared launcher owns admission,
source verification, GPU guards and the existing cumulative budget.
"""

import copy
import math
from pathlib import Path

from . import workflow_contention_study as contention
from . import workflow_encoding_optimization_study as encoding
from . import workflow_next_report as report
from . import workflow_next_study as api


SCHEMA = "ditserve.workflow.dispatch-refresh-plan.v1"
SCREENING_SEED = 4300
SEEDS = (4301, 4302, 4303)
ARMS = ("cache_off", "cache_on", "cache_on_refresh")
CACHE_ENABLED = dict(zip(ARMS, (False, True, True)))
REFRESH_ENABLED = dict(zip(ARMS, (False, False, True)))
NUMERIC_PREREQUISITES = ("encoding-wan-guided-numeric", "encoding-wan-cfg-bypass-numeric")
BASE_PREREQUISITES = NUMERIC_PREREQUISITES + ("acceptance", "calibration-video", "calibration-mixed")
HISTORICAL_JOBS = tuple(f"wan-negative-cache-mixed_negative-s{seed}" for seed in encoding.WAN_SEEDS)
TIMEOUT_SECONDS = 1800
ARTIFACT_DEPENDENCIES = [dict(closed=False, reason="dispatch refresh numerical and performance review pending")]


def job_id(seed):
    if type(seed) is not int or seed not in (SCREENING_SEED, *SEEDS):
        raise ValueError("dispatch refresh requires the frozen screening or confirmation seeds")
    return f"wan-dispatch-refresh-s{seed}"


def native_config(profile, rows, *, repeat=0):
    """Keep the r013 Wan matrix; vary only cache and pre-dispatch refresh.

    ``repeat`` rotates the three complete arms, not requests within an arm.
    Screening uses zero; confirmation uses zero, one and two respectively.
    """
    base = encoding.native_online_config(profile, rows, optimization="wan_negative_cache",
        prompt_distribution="mixed_negative", repeat=repeat)
    previous = {phase["id"]: phase for phase in base["phases"]}
    phases = []
    for name in ARMS[repeat:] + ARMS[:repeat]:
        phase = copy.deepcopy(previous["enabled" if CACHE_ENABLED[name] else "disabled"])
        phase.update(id=name, dispatch_refresh=REFRESH_ENABLED[name],
                     artifact_dependencies=copy.deepcopy(ARTIFACT_DEPENDENCIES))
        phases.append(phase)
    base["phases"] = phases
    return base


def validate_config(config, *, repeat):
    """Reject any unplanned arm/control change; return True or raise ValueError.

    The native registration envelope may add its experiment ID and watchdog.
    Every worker, phase, workload and explicit refresh/cache setting is exact.
    """
    try:
        phases = config["phases"]
        expected = native_config(phases[0]["profile"], phases[0]["workload"], repeat=repeat)
        observed = copy.deepcopy(config)
        experiment = observed.pop("experiment")
        if experiment != "run03" and experiment not in {"run03-" + job_id(seed)
                                                        for seed in (SCREENING_SEED, *SEEDS)}:
            raise ValueError("unexpected dispatch refresh experiment identity")
        budget = observed.pop("budget", None)
        if budget is not None and budget != dict(limit_gpu_seconds=TIMEOUT_SECONDS * 2,
                watchdog_seconds=TIMEOUT_SECONDS, repair_reserve_gpu_seconds=0, carry="none"):
            raise ValueError("unexpected dispatch refresh watchdog envelope")
        expected.pop("experiment")
        if api.digest(observed) != api.digest(expected):
            raise ValueError("configuration differs from the controlled dispatch refresh matrix")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("incomplete dispatch refresh configuration") from error
    return True


def _bound_job(study, logical):
    job = report.current_job(study, logical)
    config = api.read(job["input"])
    if api.digest(config) != job["configuration_sha256"]:
        raise ValueError("historical input changed: " + logical)
    bindings = list(job["input_bindings"])
    for bound in bindings:
        if api.binding(bound["path"]) != bound:
            raise ValueError("historical input binding changed: " + logical)
    ev = report.evidence(study, job)
    if not all(ev.get(key) is True for key in ("guarded", "qualified", "capacity")):
        raise ValueError("guarded numerical/calibration evidence required: " + logical)
    bindings += [api.binding(job["input"]), api.binding(ev["path"]),
                 api.binding(Path(study["root"]) / "launches" / job["id"] / "launch.json")]
    return job, config, ev, bindings


def snapshot(study):
    """Read and bind unchanged r013 evidence; never snapshot mutable ledgers.

    Reusing the numerical diagnostics requires byte-identical model/runtime
    Python sources. The new controller is still checked for exact final output
    and video latent hashes separately in every new three-arm job.
    """
    if study.get("active_job"):
        raise ValueError("cannot plan dispatch refresh while a study job is active")
    root = Path(study["root"])
    profile, calibration = api.read(root / "profile.json"), api.read(root / "calibration.json")
    if not profile or not calibration or calibration.get("profile") != profile:
        raise ValueError("dispatch refresh requires the unchanged calibration profile")
    contention._profile(profile)
    contention._controls(study.get("selections", {}).get("cpu"))
    bindings = [api.binding(root / name) for name in ("profile.json", "calibration.json")]
    dependencies, references, video_cells = {}, [], None
    for logical in BASE_PREREQUISITES + HISTORICAL_JOBS:
        job, config, ev, bound = _bound_job(study, logical)
        bindings.extend(bound)
        dependencies[logical] = dict(job_id=job["id"], source=job["source"],
            source_digest=job["source_digest"], configuration_sha256=job["configuration_sha256"])
        if logical == "calibration-video":
            video_cells = list(report.native_arms(ev["result"]).values())
        if logical in HISTORICAL_JOBS:
            metadata = job["metadata"]
            if (metadata.get("role") != "encoding_online" or metadata.get("optimization") != "wan_negative_cache"
                    or metadata.get("seed") not in encoding.WAN_SEEDS
                    or any(phase["profile"] != profile for phase in config["phases"])):
                raise ValueError("historical Wan reference controls differ")
            references.append((job, ev))
    if video_cells is None or len(video_cells) != 3:
        raise ValueError("three complete frozen video calibration observations required")
    origins = {job["source"] for job, _ in references}
    origins.update(dependencies[name]["source"] for name in NUMERIC_PREREQUISITES)
    if len(origins) != 1:
        raise ValueError("Wan numerical and online reference source snapshots differ")
    historical_source = origins.pop()
    revisions = {}
    for source in (historical_source, study["source"]):
        matches = [row for row in study["source_revisions"] if row["source"] == source]
        if len(matches) != 1 or api.seal(source) != matches[0]:
            raise ValueError("source snapshot changed or is absent from the manifest: " + source)
        revisions[source] = matches[0]
        binding_path = Path(source).parent / (Path(source).name + ".binding.json")
        if api.read(binding_path) != matches[0]:
            raise ValueError("source binding file differs from the study manifest")
        bindings.append(api.binding(binding_path))
    current, historical = revisions[study["source"]], revisions[historical_source]
    if current["source_digest"] != study["source_digest"]:
        raise ValueError("current source digest differs from its frozen revision")
    if any(dependencies[name]["source_digest"] != historical["source_digest"]
           for name in NUMERIC_PREREQUISITES + HISTORICAL_JOBS):
        raise ValueError("historical Wan source digest differs from its frozen revision")
    prefixes = ("workers/ditserve/models/", "workers/ditserve/runtime/", "configs/models/")
    protected = lambda revision: [row for row in revision["source_files"]
                                  if row["path"].startswith(prefixes)]
    if not protected(historical) or protected(current) != protected(historical):
        raise ValueError("model/runtime sources changed; new numerical qualification is required")
    walls = [ev["launch"].get("wall_seconds") for _, ev in references]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in walls):
        raise ValueError("positive measured historical launcher wall times required")
    # Scale complete two-arm wall time to three arms, retaining 20% reserve.
    # Launcher admission independently retains estimated_seconds * 1.2 + 90.
    estimate = int(math.ceil(max(walls) * 1.5 * 1.2 / 10) * 10)
    return dict(profile=profile, anchors=contention._anchors(profile, video_cells),
        input_bindings=sorted({row["path"]: row for row in bindings}.values(), key=lambda row: row["path"]),
        dependencies=dependencies, historical_source=historical_source,
        source=study["source"], source_digest=study["source_digest"],
        model_bindings_sha256=api.digest(study["model_bindings"]),
        estimated_seconds=estimate, estimate_basis=dict(reference_job_ids=list(HISTORICAL_JOBS),
            reference_two_arm_wall_seconds=walls, arms_scale=1.5, planning_reserve_fraction=.2,
            launcher_admission="estimated_seconds * 1.2 + 90"))


def build_plan(study):
    """Return four complete templates and immutable provenance without writes."""
    for job in study["jobs"]:
        seed = job.get("metadata", {}).get("seed")
        if seed in (SCREENING_SEED, *SEEDS) and job.get("logical_id", job["id"]) != job_id(seed):
            raise ValueError("dispatch refresh seeds must not reuse historical trajectories")
    frozen = snapshot(study)
    jobs = []
    for seed, repeat, stage in [(SCREENING_SEED, 0, "screening")] + [
            (seed, repeat, "confirmation") for repeat, seed in enumerate(SEEDS)]:
        config = native_config(frozen["profile"], contention.contention_trace(
            frozen["profile"], seed, frozen["anchors"]), repeat=repeat)
        validate_config(config, repeat=repeat)
        requires = list(BASE_PREREQUISITES)
        if stage == "confirmation":
            requires.append(job_id(SCREENING_SEED))
        metadata = dict(role="dispatch_refresh_online", optimization="dispatch_refresh",
            stage=stage, seed=seed, repeat=repeat, prompt_distribution="mixed_negative",
            independent_confirmation=stage == "confirmation", performance_evidence=True,
            numeric_prerequisites=list(NUMERIC_PREREQUISITES),
            trace_sha256=api.digest(config["phases"][0]["workload"]), anchors=frozen["anchors"],
            historical_reference_jobs=list(HISTORICAL_JOBS), historical_source=frozen["historical_source"],
            dependencies=frozen["dependencies"], model_bindings_sha256=frozen["model_bindings_sha256"],
            estimate_basis=frozen["estimate_basis"], monitor_cpu_affinity=[58],
            monitor_affinity_scope="launcher must enforce and record; metadata is not observed affinity",
            comparisons=[list(pair) for pair in ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[2]), (ARMS[0], ARMS[2]))],
            confirmation_gate="screening technical qualification and explicit performance selection before execution",
            scope="new trajectories only; screening and historical r013 observations excluded from confirmation")
        jobs.append(dict(id=job_id(seed), config=config, metadata=metadata, requires=requires,
                         estimated_seconds=frozen["estimated_seconds"], timeout_seconds=TIMEOUT_SECONDS))
    return dict(schema=SCHEMA, run_id=study["run_id"], root=study["root"], jobs=jobs,
                input_bindings=frozen["input_bindings"], source=frozen["source"],
                source_digest=frozen["source_digest"], historical_source=frozen["historical_source"],
                ready_for_gpu=False, registers_or_launches_gpu=False)


def register(study):
    """Append four new frozen jobs; never launch, replace history or edit budget."""
    plan = build_plan(study)
    if plan["source"] == plan["historical_source"]:
        raise ValueError("freeze a new dispatch refresh source revision before registration")
    # Preflight the entire matrix before add_job writes any input or appends.
    for template in plan["jobs"]:
        existing = [job for job in study["jobs"] if job.get("logical_id", job["id"]) == template["id"]
                    and not job.get("superseded")]
        if len(existing) > 1:
            raise ValueError("multiple current dispatch refresh blocks")
        if existing:
            job = existing[0]
            root = Path(study["root"])
            input_path = root / "inputs" / (template["id"] + ".json")
            env_path = root / "inputs" / (template["id"] + ".env.json")
            output_path = root / "measurements" / template["id"]
            expected_command = ["/workspace/bin/ditserve", "--root", "/data", "--package-root", "/workspace/workers",
                "--python", "/opt/venv/bin/python", "run", "--config", api.container_path(study, input_path),
                "--env", api.container_path(study, env_path), "--allow-gpu", "--run-id", template["id"],
                "--keep-raw", "--work-dir", api.container_path(study, output_path),
                "--results-root", api.container_path(study, root / "native-results"),
                "--ledger", api.container_path(study, root / "native-ledgers" / (template["id"] + ".json"))]
            if (job["source"] != plan["source"] or job["source_digest"] != plan["source_digest"]
                    or job["metadata"] != template["metadata"] or job["requires"] != template["requires"]
                    or job["estimated_seconds"] != template["estimated_seconds"]
                    or job["timeout_seconds"] != TIMEOUT_SECONDS
                    or job.get("command_type") != "native" or job["command"] != expected_command
                    or job["devices"] != [0, 1] or job["family"] != "extension"
                    or job["gpu_mapping"] != study["gpu_mapping"]
                    or (job["input"], job["environment_input"], job["output"]) !=
                       (str(input_path), str(env_path), str(output_path))):
                raise ValueError("cannot replace frozen dispatch refresh metadata or source")
            expected = copy.deepcopy(template["config"])
            expected.update(experiment="run03-" + template["id"], budget=dict(
                limit_gpu_seconds=TIMEOUT_SECONDS * 2, watchdog_seconds=TIMEOUT_SECONDS,
                repair_reserve_gpu_seconds=0, carry="none"))
            if (api.read(job["input"]) != expected or api.digest(expected) != job["configuration_sha256"]
                    or job["input_bindings"] != [api.binding(job["input"]), api.binding(job["environment_input"])]
                    + plan["input_bindings"]):
                raise ValueError("cannot replace frozen dispatch refresh inputs or bindings")
        else:
            root = Path(study["root"])
            if any(path.exists() for path in (root / "inputs" / (template["id"] + ".json"),
                    root / "inputs" / (template["id"] + ".env.json"), root / "measurements" / template["id"])):
                raise FileExistsError("dispatch refresh must not overwrite existing input/output")
    jobs = []
    for template in plan["jobs"]:
        job = api.add_job(study, template["id"], "extension", template["config"], native=True,
            devices=(0, 1), requires=template["requires"], estimate=template["estimated_seconds"],
            timeout=TIMEOUT_SECONDS, metadata=template["metadata"])
        job["input_bindings"] = [api.binding(job["input"]), api.binding(job["environment_input"])] + plan["input_bindings"]
        api.save(study)
        jobs.append(job)
    return jobs
