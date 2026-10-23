"""Frozen occupancy-probe experiment; CPU-only planning and append-only jobs."""

import copy
import math
from pathlib import Path

from . import workflow_contention_study as contention
from . import workflow_dispatch_refresh_study as previous
from . import workflow_encoding_optimization_study as encoding
from . import workflow_next_report as report
from . import workflow_next_study as api


SCHEMA = "ditserve.workflow.occupancy-plan.v1"
SCREENING_SEED = 4400
SEEDS = (4401, 4402, 4403)
ARMS = ("reference", "candidate")
BACKENDS = dict(reference="nvidia_smi", candidate="nvml")
NUMERIC_PREREQUISITES = previous.NUMERIC_PREREQUISITES
REFERENCE_JOB = "wan-dispatch-refresh-s4300"
BASE_PREREQUISITES = previous.BASE_PREREQUISITES + (REFERENCE_JOB,)
RUNTIME_ALLOWLIST = ("workers/ditserve/runtime/worker.py", "workers/ditserve/runtime/occupancy_nvml.py")
TIMEOUT_SECONDS = 1800
ARTIFACT_DEPENDENCIES = [dict(closed=False, reason="occupancy probe numerical and performance review pending")]


def job_id(seed):
    if type(seed) is not int or seed not in (SCREENING_SEED, *SEEDS):
        raise ValueError("occupancy requires the frozen screening or confirmation seeds")
    return f"occupancy-probe-s{seed}"


def native_config(profile, rows, *, repeat=0):
    """Two cache-enabled arms differing only in the occupancy query backend."""
    config = encoding.native_online_config(profile, rows, optimization="wan_negative_cache",
        prompt_distribution="mixed_negative", repeat=repeat)
    enabled = next(phase for phase in config["phases"] if phase["id"] == "enabled")
    names = ARMS if repeat % 2 == 0 else ARMS[::-1]
    config["phases"] = [dict(copy.deepcopy(enabled), id=name, dispatch_refresh=False,
        occupancy_backend=BACKENDS[name], artifact_dependencies=copy.deepcopy(ARTIFACT_DEPENDENCIES)) for name in names]
    return config


def validate_config(config, *, repeat):
    """Validate the complete controlled matrix, including explicit baseline switches."""
    try:
        phase = config["phases"][0]
        expected = native_config(phase["profile"], phase["workload"], repeat=repeat)
        observed = copy.deepcopy(config)
        experiment = observed.pop("experiment")
        if experiment != "run03" and experiment not in {"run03-" + job_id(seed) for seed in (SCREENING_SEED, *SEEDS)}:
            raise ValueError("unexpected occupancy experiment identity")
        budget = observed.pop("budget", None)
        if budget is not None and budget != dict(limit_gpu_seconds=TIMEOUT_SECONDS * 2,
                watchdog_seconds=TIMEOUT_SECONDS, repair_reserve_gpu_seconds=0, carry="none"):
            raise ValueError("unexpected occupancy watchdog envelope")
        expected.pop("experiment")
        if api.digest(observed) != api.digest(expected):
            raise ValueError("configuration differs from the controlled occupancy matrix")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("incomplete occupancy configuration") from error
    return True


def snapshot(study):
    """Bind model numerics separately from the intentionally changed guard runtime."""
    if study.get("active_job"):
        raise ValueError("cannot plan occupancy while a study job is active")
    root = Path(study["root"])
    profile, calibration = api.read(root / "profile.json"), api.read(root / "calibration.json")
    if not profile or not calibration or calibration.get("profile") != profile:
        raise ValueError("unchanged calibration profile required")
    contention._profile(profile)
    contention._controls(study.get("selections", {}).get("cpu"))
    bindings = [api.binding(root / name) for name in ("profile.json", "calibration.json")]
    dependencies, originals = {}, {}
    for logical in BASE_PREREQUISITES:
        job, config, ev, bound = previous._bound_job(study, logical)
        originals[logical] = (job, config, ev)
        bindings.extend(bound)
        dependencies[logical] = dict(job_id=job["id"], source=job["source"],
            source_digest=job["source_digest"], configuration_sha256=job["configuration_sha256"])
    reference, reference_config, reference_ev = originals[REFERENCE_JOB]
    if (reference.get("metadata", {}).get("role") != "dispatch_refresh_online"
            or reference["metadata"].get("seed") != 4300
            or any(phase["profile"] != profile for phase in reference_config["phases"])):
        raise ValueError("qualified r014 screening reference required; performance benefit is not assumed")
    previous.validate_config(reference_config, repeat=0)
    video_cells = list(report.native_arms(originals["calibration-video"][2]["result"]).values())
    if len(video_cells) != 3:
        raise ValueError("three complete video calibration observations required")
    revisions = {}
    sources = {study["source"], reference["source"], *(dependencies[name]["source"] for name in NUMERIC_PREREQUISITES)}
    for source in sources:
        matches = [row for row in study["source_revisions"] if row["source"] == source]
        if len(matches) != 1 or api.seal(source) != matches[0]:
            raise ValueError("source snapshot changed or absent from manifest")
        path = Path(source).parent / (Path(source).name + ".binding.json")
        if api.read(path) != matches[0]:
            raise ValueError("source binding differs from manifest")
        revisions[source] = matches[0]
        bindings.append(api.binding(path))
    current, old = revisions[study["source"]], revisions[reference["source"]]
    if current["source_digest"] != study["source_digest"] or any(
            revisions[dependencies[name]["source"]]["source_digest"] != dependencies[name]["source_digest"]
            for name in NUMERIC_PREREQUISITES + (REFERENCE_JOB,)):
        raise ValueError("job source digest differs from frozen revision")
    model_files = lambda revision: [row for row in revision["source_files"]
                                    if row["path"].startswith(("workers/ditserve/models/", "configs/models/"))]
    if not model_files(old) or any(model_files(row) != model_files(old) for row in revisions.values()):
        raise ValueError("model sources changed; new model numerical qualification required")
    runtime = lambda revision: {row["path"]: row for row in revision["source_files"]
                                if row["path"].startswith("workers/ditserve/runtime/")}
    before, after = runtime(old), runtime(current)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    if any(path not in RUNTIME_ALLOWLIST for path in changed) or any(path not in after for path in changed):
        raise ValueError("runtime change exceeds the occupancy guard allowlist")
    if study["source"] != reference["source"] and RUNTIME_ALLOWLIST[1] not in after:
        raise ValueError("new source is missing the NVML occupancy implementation")
    wall = reference_ev["launch"].get("wall_seconds")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("positive r014 measured launcher wall time required")
    estimate = int(math.ceil(wall * (2 / 3) * 1.2 / 10) * 10)
    return dict(profile=profile, anchors=contention._anchors(profile, video_cells),
        input_bindings=sorted({row["path"]: row for row in bindings}.values(), key=lambda row: row["path"]),
        dependencies=dependencies, historical_source=reference["source"], source=study["source"],
        source_digest=study["source_digest"], model_bindings_sha256=api.digest(study["model_bindings"]),
        model_sources_identical=True, runtime_changed_files=changed, runtime_allowlist=list(RUNTIME_ALLOWLIST),
        qualification_scope="unchanged model numerics reused; changed occupancy guard runtime separately audited in both online arms",
        estimated_seconds=estimate, estimate_basis=dict(reference_job=REFERENCE_JOB,
            reference_three_arm_wall_seconds=wall, arms_scale=2 / 3, planning_reserve_fraction=.2,
            launcher_admission="estimated_seconds * 1.2 + 90"))


def build_plan(study):
    """Four new templates; no registration, GPU work or budget mutation."""
    for job in study["jobs"]:
        seed = job.get("metadata", {}).get("seed")
        if seed in (SCREENING_SEED, *SEEDS) and job.get("logical_id", job["id"]) != job_id(seed):
            raise ValueError("occupancy seeds must not reuse historical trajectories")
    frozen = snapshot(study)
    jobs = []
    for seed, repeat, stage in [(SCREENING_SEED, 0, "screening")] + [
            (seed, repeat, "confirmation") for repeat, seed in enumerate(SEEDS)]:
        config = native_config(frozen["profile"], contention.contention_trace(frozen["profile"], seed, frozen["anchors"]), repeat=repeat)
        validate_config(config, repeat=repeat)
        requires = list(BASE_PREREQUISITES) + ([job_id(SCREENING_SEED)] if stage == "confirmation" else [])
        metadata = dict(role="occupancy_online", optimization="occupancy_probe", stage=stage,
            seed=seed, repeat=repeat, prompt_distribution="mixed_negative", independent_confirmation=stage == "confirmation",
            performance_evidence=True, numeric_prerequisites=list(NUMERIC_PREREQUISITES),
            trace_sha256=api.digest(config["phases"][0]["workload"]), anchors=frozen["anchors"],
            dependencies=frozen["dependencies"], historical_source=frozen["historical_source"],
            historical_reference_job=REFERENCE_JOB, model_bindings_sha256=frozen["model_bindings_sha256"],
            model_sources_identical=True, runtime_changed_files=frozen["runtime_changed_files"],
            runtime_allowlist=frozen["runtime_allowlist"], qualification_scope=frozen["qualification_scope"],
            estimate_basis=frozen["estimate_basis"], monitor_cpu_affinity=[58], comparisons=[["reference", "candidate"]],
            confirmation_gate="screening technical qualification and prespecified performance selection before execution",
            scope="new trajectories only; historical failed refresh screening and unused confirmation jobs remain unchanged")
        jobs.append(dict(id=job_id(seed), config=config, metadata=metadata, requires=requires,
                         estimated_seconds=frozen["estimated_seconds"], timeout_seconds=TIMEOUT_SECONDS))
    return dict(schema=SCHEMA, root=study["root"], run_id=study["run_id"], jobs=jobs,
        input_bindings=frozen["input_bindings"], source=frozen["source"], source_digest=frozen["source_digest"],
        historical_source=frozen["historical_source"], ready_for_gpu=False, registers_or_launches_gpu=False)


def _envelope(study, template):
    root, identifier = Path(study["root"]), template["id"]
    path, env, output = root / "inputs" / (identifier + ".json"), root / "inputs" / (identifier + ".env.json"), root / "measurements" / identifier
    command = ["/workspace/bin/ditserve", "--root", "/data", "--package-root", "/workspace/workers",
        "--python", "/opt/venv/bin/python", "run", "--config", api.container_path(study, path),
        "--env", api.container_path(study, env), "--allow-gpu", "--run-id", identifier, "--keep-raw",
        "--work-dir", api.container_path(study, output), "--results-root", api.container_path(study, root / "native-results"),
        "--ledger", api.container_path(study, root / "native-ledgers" / (identifier + ".json"))]
    return dict(source=study["source"], source_digest=study["source_digest"], metadata=template["metadata"],
        requires=template["requires"], estimated_seconds=template["estimated_seconds"], timeout_seconds=TIMEOUT_SECONDS,
        command_type="native", command=command, devices=[0, 1], family="extension", gpu_mapping=study["gpu_mapping"],
        input=str(path), environment_input=str(env), output=str(output))


def register(study):
    """Validate the whole matrix, then append only; shared launcher executes it."""
    plan = build_plan(study)
    if plan["source"] == plan["historical_source"]:
        raise ValueError("freeze a new occupancy source revision before registration")
    for template in plan["jobs"]:
        existing = [job for job in study["jobs"] if not job.get("superseded")
                    and job.get("logical_id", job["id"]) == template["id"]]
        if len(existing) > 1:
            raise ValueError("multiple current occupancy blocks")
        envelope = _envelope(study, template)
        if existing:
            job = existing[0]
            expected = copy.deepcopy(template["config"])
            expected.update(experiment="run03-" + template["id"], budget=dict(limit_gpu_seconds=3600,
                watchdog_seconds=TIMEOUT_SECONDS, repair_reserve_gpu_seconds=0, carry="none"))
            if (any(job.get(key) != value for key, value in envelope.items())
                    or api.read(job["input"]) != expected or api.digest(expected) != job["configuration_sha256"]
                    or job["input_bindings"] != [api.binding(job["input"]), api.binding(job["environment_input"])] + plan["input_bindings"]):
                raise ValueError("cannot replace frozen occupancy inputs, metadata, source or command")
        elif any(Path(envelope[key]).exists() for key in ("input", "environment_input", "output")):
            raise FileExistsError("occupancy must not overwrite existing input/output")
    jobs = []
    for template in plan["jobs"]:
        job = api.add_job(study, template["id"], "extension", template["config"], native=True, devices=(0, 1),
            requires=template["requires"], estimate=template["estimated_seconds"], timeout=TIMEOUT_SECONDS, metadata=template["metadata"])
        job["input_bindings"] = [api.binding(job["input"]), api.binding(job["environment_input"])] + plan["input_bindings"]
        api.save(study)
        jobs.append(job)
    return jobs
