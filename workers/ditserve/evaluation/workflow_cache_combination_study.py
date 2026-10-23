"""Append-only acceptance of staged Sana caching on the frozen r016 recipe."""

import copy
import math
from pathlib import Path

from . import workflow_contention_study as contention
from . import workflow_encoding_optimization_study as encoding
from . import workflow_next_report as report
from . import workflow_next_study as api
from . import workflow_occupancy_report as occupancy_report
from . import workflow_occupancy_study as occupancy


SCHEMA = "ditserve.workflow.cache-combination-plan.v1"
SCREENING_SEED = 4500
SEEDS = (4501, 4502, 4503)
ARMS = ("reference", "candidate")
CHECKPOINT = "checkpoints/before-cache-combination-r016"
REFERENCE_AUDIT = "maintenance/final-integrity-occupancy-r016-20261022T080822Z-0e93f5a5.json"
DIAGNOSIS = "maintenance/cache-combination-diagnosis-r016-v2.json"
HISTORICAL_JOB_COUNT = 82
HISTORICAL_ATTEMPT_COUNT = 70
REFERENCE_JOBS = tuple(occupancy.job_id(seed) for seed in occupancy.SEEDS)
SANA_NUMERIC = ("encoding-sana-unique-numeric", "encoding-sana-repeated-numeric")
NUMERIC_PREREQUISITES = occupancy.NUMERIC_PREREQUISITES + SANA_NUMERIC
SANA_REFERENCES = tuple(f"sana-prompt-cache-{distribution}-s{seed}"
                       for distribution in ("unique", "repeated") for seed in encoding.SANA_SEEDS)
BASE_PREREQUISITES = NUMERIC_PREREQUISITES + (
    "acceptance", "calibration-video", "calibration-mixed") + REFERENCE_JOBS + SANA_REFERENCES
PROTECTED_PREFIXES = ("workers/ditserve/models/", "workers/ditserve/runtime/",
                      "configs/", "src/", "include/", "cmd/")
TIMEOUT_SECONDS = 1800
ARTIFACT_DEPENDENCIES = [dict(closed=False, reason="cache combination numerical and performance review pending")]


def job_id(seed):
    if type(seed) is not int or seed not in (SCREENING_SEED, *SEEDS):
        raise ValueError("cache combination requires the frozen screening or confirmation seeds")
    return f"cache-combination-s{seed}"


def native_config(profile, rows, *, repeat=0):
    """Keep the r016 candidate recipe; only enable Sana's staged prompt cache."""
    config = occupancy.native_config(profile, rows, repeat=repeat)
    baseline = next(phase for phase in config["phases"] if phase["id"] == "candidate")
    images = [row for row in baseline["workload"] if row["kind"] == "image"]
    if [row["spec"]["prompt"] for row in images] != [api.PROMPTS[i % 4] for i in range(20)]:
        raise ValueError("cache combination retains the frozen four-prompt repeated workload")
    phases = []
    for name in ARMS if repeat % 2 == 0 else ARMS[::-1]:
        phase = copy.deepcopy(baseline)
        phase.update(id=name, artifact_dependencies=copy.deepcopy(ARTIFACT_DEPENDENCIES))
        if name == "candidate":
            for rank in (0, 1):
                phase["worker_overrides"][f"sana{rank}"]["options"] = dict(
                    encoding.SANA_CACHE, prompt_cache_sync_mode="staged")
        phases.append(phase)
    config["phases"] = phases
    return config


def validate_config(config, *, repeat):
    try:
        phase = config["phases"][0]
        expected = native_config(phase["profile"], phase["workload"], repeat=repeat)
        observed = copy.deepcopy(config)
        experiment = observed.pop("experiment")
        if experiment != "run03" and experiment not in {"run03-" + job_id(seed) for seed in (SCREENING_SEED, *SEEDS)}:
            raise ValueError("unexpected cache combination experiment identity")
        budget = observed.pop("budget", None)
        if budget is not None and api.digest(budget) != api.digest(dict(limit_gpu_seconds=TIMEOUT_SECONDS * 2,
                watchdog_seconds=TIMEOUT_SECONDS, repair_reserve_gpu_seconds=0, carry="none")):
            raise ValueError("unexpected cache combination watchdog envelope")
        expected.pop("experiment")
        if api.digest(observed) != api.digest(expected):
            raise ValueError("configuration differs from the controlled cache combination matrix")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("incomplete cache combination configuration") from error
    return True


class _Inputs:
    """Bind before reading and recheck before returning any executable plan."""

    def __init__(self):
        self.bindings = {}
        self.mutable_snapshot = []

    def add(self, path, expected=None):
        path = str(Path(path))
        actual = self.bindings.get(path)
        if actual is None:
            actual = api.binding(path)
            self.bindings[path] = actual
        if expected is not None and actual != expected:
            raise ValueError("historical input binding changed: " + path)
        return actual

    def read(self, path):
        self.add(path)
        return api.read(path)

    def finish(self):
        rows = sorted(self.bindings.values(), key=lambda row: row["path"])
        if any(api.binding(row["path"]) != row for row in rows + self.mutable_snapshot):
            raise ValueError("plan input changed during validation")
        return rows


def _history(study, inputs):
    root = Path(study["root"])
    checkpoint = root / CHECKPOINT
    manifest = inputs.read(checkpoint / "manifest.json")
    if not isinstance(manifest, list):
        raise ValueError("flat immutable checkpoint manifest required")
    snapshots = {}
    for name in ("study.json", "budget.json"):
        path = checkpoint / name
        entries = [row for row in manifest if row.get("copy") == str(path)
                   and row.get("source") == str(root / name)]
        actual = inputs.add(path)
        if len(entries) != 1 or any(entries[0].get(key) != actual[key] for key in ("bytes", "sha256")):
            raise ValueError("checkpoint snapshot differs from its manifest")
        snapshots[name] = inputs.read(path)
    old, budget_before = snapshots["study.json"], snapshots["budget.json"]
    # The mutable ledger is read for admission consistency, never frozen into
    # job input_bindings, which must survive subsequent legitimate charges.
    budget_binding = api.binding(root / "budget.json")
    inputs.mutable_snapshot.append(budget_binding)
    budget = api.read(root / "budget.json")
    if api.binding(root / "budget.json") != budget_binding:
        raise ValueError("budget changed while preparing the plan")
    if (len(old["jobs"]) != HISTORICAL_JOB_COUNT or len(budget_before["attempts"]) != HISTORICAL_ATTEMPT_COUNT
            or old.get("active_job") or budget_before.get("active") or budget.get("active")):
        raise ValueError("settled 82-job/70-attempt checkpoint and idle current ledger required")
    if api.digest(study["jobs"][:HISTORICAL_JOB_COUNT]) != api.digest(old["jobs"]):
        raise ValueError("historical job prefix changed")
    for key in ("source_revisions", "decisions"):
        if study.get(key, [])[:len(old.get(key, []))] != old.get(key, []):
            raise ValueError("historical study prefix changed: " + key)
    if api.digest(budget["attempts"][:HISTORICAL_ATTEMPT_COUNT]) != api.digest(budget_before["attempts"]):
        raise ValueError("historical attempt prefix changed")
    allowed = {job_id(seed) for seed in (SCREENING_SEED, *SEEDS)}
    added = study["jobs"][HISTORICAL_JOB_COUNT:]
    if (len({job["id"] for job in study["jobs"]}) != len(study["jobs"])
            or any(job["id"] not in allowed for job in added)
            or any(a["job_id"] not in {job["id"] for job in added}
                   for a in budget["attempts"][HISTORICAL_ATTEMPT_COUNT:])):
        raise ValueError("only new cache combination jobs may extend this checkpoint")
    for key in ("run_id", "root", "model_bindings", "gpu_mapping", "image", "model_root",
                "budget_seconds", "budget_mode", "family_caps_seconds", "assumptions"):
        if api.digest(study.get(key)) != api.digest(old.get(key)):
            raise ValueError("historical study contract changed: " + key)
    if ("limit_seconds" not in budget_before or "limit_seconds" not in budget
            or budget_before["limit_seconds"] is not None or budget["limit_seconds"] is not None
            or not budget_before.get("authorization_history")
            or budget.get("authorization_history") != budget_before["authorization_history"]):
        raise ValueError("existing explicit unlimited authorization must remain unchanged")
    expected = budget_before["charged_seconds"]
    for attempt in budget["attempts"][HISTORICAL_ATTEMPT_COUNT:]:
        value = attempt["charged_seconds"]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid appended attempt charge")
        expected += value
    if expected != budget["charged_seconds"]:
        raise ValueError("cumulative charge differs from exact historical-prefix replay")
    for job in old["jobs"]:
        for row in job.get("input_bindings", []):
            inputs.add(row["path"], row)
    selected = old.get("selections", {}).get("occupancy", {})
    if (selected.get("qualified") is not True or selected.get("promotable") is not True
            or selected.get("confirmation_seeds") != list(occupancy.SEEDS)):
        raise ValueError("checkpoint needs three qualified and promotable occupancy anchors")
    return old, dict(checkpoint=str(checkpoint), jobs=HISTORICAL_JOB_COUNT,
        attempts=HISTORICAL_ATTEMPT_COUNT, jobs_sha256=api.digest(old["jobs"]),
        attempts_sha256=api.digest(budget_before["attempts"]),
        charged_seconds=budget_before["charged_seconds"], authorization_history=budget_before["authorization_history"])


def _phase_controls(phase):
    return {key: value for key, value in phase.items() if key not in ("id", "workload", "artifact_dependencies")}


def snapshot(study):
    if study.get("active_job"):
        raise ValueError("cannot plan cache combination while a study job is active")
    inputs = _Inputs()
    root = Path(study["root"])
    old, history = _history(study, inputs)
    audit = inputs.read(root / REFERENCE_AUDIT)
    if (audit.get("schema") != "ditserve.maintenance.occupancy-final-integrity.v1"
            or audit.get("status") != "passed" or audit.get("errors") != []
            or audit.get("history", {}).get("current_jobs") != HISTORICAL_JOB_COUNT
            or audit.get("history", {}).get("current_attempts") != HISTORICAL_ATTEMPT_COUNT
            or not any(r.get("source") == old["source"] and r.get("source_digest") == old["source_digest"]
                       for r in audit.get("source_seals", []))):
        raise ValueError("passed r016 final integrity proof required")
    diagnosis = inputs.read(root / DIAGNOSIS)
    if (diagnosis.get("schema") != "ditserve.cache-combination-diagnosis.v1"
            or diagnosis.get("combination_has_gpu_evidence") is not False):
        raise ValueError("frozen pre-experiment combination diagnosis required")
    profile, calibration = inputs.read(root / "profile.json"), inputs.read(root / "calibration.json")
    if not profile or not calibration or calibration.get("profile") != profile:
        raise ValueError("unchanged calibration profile required")
    contention._profile(profile)
    contention._controls(study.get("selections", {}).get("cpu"))
    dependencies, originals = {}, {}
    for logical in BASE_PREREQUISITES:
        job = report.current_job(study, logical)
        config = inputs.read(job["input"])
        if api.digest(config) != job["configuration_sha256"]:
            raise ValueError("historical input changed: " + logical)
        for row in job["input_bindings"]:
            inputs.add(row["path"], row)
        inputs.add(Path(job["output"]) / "run.json")
        inputs.add(root / "launches" / job["id"] / "launch.json")
        ev = report.evidence(study, job)
        if not all(ev.get(key) is True for key in ("guarded", "qualified", "capacity")):
            raise ValueError("qualified numerical/mechanism/capacity evidence required: " + logical)
        originals[logical] = (job, config, ev)
        dependencies[logical] = dict(job_id=job["id"], source=job["source"],
            source_digest=job["source_digest"], configuration_sha256=job["configuration_sha256"])
    for repeat, logical in enumerate(REFERENCE_JOBS):
        job, config, ev = originals[logical]
        metadata = job.get("metadata", {})
        if (metadata.get("role") != "occupancy_online" or metadata.get("stage") != "confirmation"
                or metadata.get("seed") != occupancy.SEEDS[repeat] or metadata.get("repeat") != repeat
                or job["source"] != old["source"] or Path(job["source"]).name != "r016"):
            raise ValueError("three independent r016 occupancy confirmations required")
        occupancy.validate_config(config, repeat=repeat)
        if occupancy_report.screening_gate(ev).get("passed") is not True:
            raise ValueError("occupancy anchor is not promotable: " + logical)
        baseline = next(p for p in config["phases"] if p["id"] == "candidate")
        current = native_config(profile, baseline["workload"], repeat=repeat)
        proposed = next(p for p in current["phases"] if p["id"] == "reference")
        top = lambda value: {k: v for k, v in value.items() if k not in ("experiment", "budget", "phases")}
        if (api.digest(top(config)) != api.digest(top(current)) or baseline["profile"] != profile
                or api.digest(_phase_controls(baseline)) != api.digest(_phase_controls(proposed))):
            raise ValueError("new reference differs from the actual r016 candidate recipe")
    for logical in SANA_REFERENCES:
        job, config, _ = originals[logical]
        metadata = job.get("metadata", {})
        if (metadata.get("role") != "encoding_online" or metadata.get("optimization") != "sana_prompt_cache"
                or metadata.get("seed") not in encoding.SANA_SEEDS
                or metadata.get("prompt_distribution") not in ("unique", "repeated")
                or any(p["profile"] != profile for p in config["phases"])):
            raise ValueError("historical Sana numerical/mechanism controls differ")
    revisions = {}
    for source in {study["source"], old["source"], *(d["source"] for d in dependencies.values())}:
        matches = [r for r in study["source_revisions"] if r["source"] == source]
        seal_path = Path(source).parent / (Path(source).name + ".binding.json")
        if len(matches) != 1 or inputs.read(seal_path) != matches[0] or api.seal(source) != matches[0]:
            raise ValueError("source snapshot changed or absent from manifest")
        revisions[source] = matches[0]
    if (revisions[study["source"]]["source_digest"] != study["source_digest"]
            or any(revisions[d["source"]]["source_digest"] != d["source_digest"] for d in dependencies.values())):
        raise ValueError("source digest differs from frozen revision")
    protected = lambda revision: [r for r in revision["source_files"] if r["path"].startswith(PROTECTED_PREFIXES)]
    if not protected(revisions[old["source"]]) or protected(revisions[study["source"]]) != protected(revisions[old["source"]]):
        raise ValueError("model/worker/runtime/controller sources changed from r016; new qualification required")
    for source in {old["source"], study["source"]}:
        for row in protected(revisions[source]):
            path = Path(source) / row["path"]
            inputs.add(path, dict(row, path=str(path)))
    model_files = lambda revision: [r for r in revision["source_files"]
                                   if r["path"].startswith(("workers/ditserve/models/", "configs/models/"))]
    numeric_sources = {dependencies[name]["source"] for name in NUMERIC_PREREQUISITES + SANA_REFERENCES + REFERENCE_JOBS}
    if any(model_files(revisions[source]) != model_files(revisions[old["source"]]) for source in numeric_sources):
        raise ValueError("historical model numerical qualification no longer matches")
    video_cells = list(report.native_arms(originals["calibration-video"][2]["result"]).values())
    if len(video_cells) != 3:
        raise ValueError("three complete video calibration observations required")
    walls = [originals[name][2]["launch"].get("wall_seconds") for name in REFERENCE_JOBS]
    if any(type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in walls):
        raise ValueError("positive measured r016 launcher wall times required")
    return dict(profile=profile, anchors=contention._anchors(profile, video_cells), history=history,
        input_bindings=inputs.finish(), dependencies=dependencies, historical_source=old["source"],
        source=study["source"], source_digest=study["source_digest"],
        model_bindings_sha256=api.digest(study["model_bindings"]), protected_sources_identical=True,
        estimated_seconds=int(math.ceil(max(walls) * 1.2 / 10) * 10),
        estimate_basis=dict(reference_jobs=list(REFERENCE_JOBS), measured_wall_seconds=walls,
                            planning_reserve_fraction=.2, launcher_admission="estimated_seconds * 1.2 + 90"))


def build_plan(study):
    for job in study["jobs"]:
        seed = job.get("metadata", {}).get("seed")
        if seed in (SCREENING_SEED, *SEEDS) and job.get("logical_id", job["id"]) != job_id(seed):
            raise ValueError("cache combination seeds must not reuse historical trajectories")
    frozen = snapshot(study)
    jobs = []
    for seed, repeat, stage in [(SCREENING_SEED, 0, "screening")] + [
            (seed, repeat, "confirmation") for repeat, seed in enumerate(SEEDS)]:
        config = native_config(frozen["profile"], contention.contention_trace(frozen["profile"], seed, frozen["anchors"]), repeat=repeat)
        validate_config(config, repeat=repeat)
        metadata = dict(role="cache_combination_online", optimization="cache_combination", stage=stage,
            seed=seed, repeat=repeat, prompt_distribution="repeated", independent_confirmation=stage == "confirmation",
            performance_evidence=True, numeric_prerequisites=list(NUMERIC_PREREQUISITES),
            trace_sha256=api.digest(config["phases"][0]["workload"]), anchors=frozen["anchors"],
            dependencies=frozen["dependencies"], historical_source=frozen["historical_source"],
            historical_reference_jobs=list(REFERENCE_JOBS), history=frozen["history"],
            model_bindings_sha256=frozen["model_bindings_sha256"], protected_sources_identical=True,
            protected_source_prefixes=list(PROTECTED_PREFIXES), estimate_basis=frozen["estimate_basis"],
            monitor_cpu_affinity=[58], comparisons=[["reference", "candidate"]],
            confirmation_gate="screening technical qualification and prespecified performance selection before execution",
            scope="new repeated-prompt trajectories only; unchanged r016 NVML/Wan recipe; no automatic default change")
        requires = list(BASE_PREREQUISITES) + ([job_id(SCREENING_SEED)] if stage == "confirmation" else [])
        jobs.append(dict(id=job_id(seed), config=config, metadata=metadata, requires=requires,
                         estimated_seconds=frozen["estimated_seconds"], timeout_seconds=TIMEOUT_SECONDS))
    return dict(schema=SCHEMA, root=study["root"], run_id=study["run_id"], jobs=jobs,
        input_bindings=frozen["input_bindings"], source=frozen["source"], source_digest=frozen["source_digest"],
        historical_source=frozen["historical_source"], history=frozen["history"],
        ready_for_gpu=False, registers_or_launches_gpu=False)


def register(study):
    """Preflight all four envelopes, then append; the existing driver owns execution."""
    plan = build_plan(study)
    if plan["source"] == plan["historical_source"]:
        raise ValueError("freeze a new cache combination source revision before registration")
    for template in plan["jobs"]:
        existing = [job for job in study["jobs"] if not job.get("superseded")
                    and job.get("logical_id", job["id"]) == template["id"]]
        if len(existing) > 1:
            raise ValueError("multiple current cache combination blocks")
        envelope = occupancy._envelope(study, template)
        if existing:
            job = existing[0]
            expected = copy.deepcopy(template["config"])
            expected.update(experiment="run03-" + template["id"], budget=dict(limit_gpu_seconds=TIMEOUT_SECONDS * 2,
                watchdog_seconds=TIMEOUT_SECONDS, repair_reserve_gpu_seconds=0, carry="none"))
            if (any(job.get(key) != value for key, value in envelope.items())
                    or api.read(job["input"]) != expected or api.digest(expected) != job["configuration_sha256"]
                    or api.read(job["environment_input"]) != api.environment(study, (0, 1))
                    or job["input_bindings"] != [api.binding(job["input"]), api.binding(job["environment_input"])] + plan["input_bindings"]):
                raise ValueError("cannot replace frozen cache combination inputs, metadata, source or command")
        elif any(Path(envelope[key]).exists() for key in ("input", "environment_input", "output")):
            raise FileExistsError("cache combination must not overwrite existing input/output")
    jobs = []
    for template in plan["jobs"]:
        job = api.add_job(study, template["id"], "extension", template["config"], native=True, devices=(0, 1),
            requires=template["requires"], estimate=template["estimated_seconds"], timeout=TIMEOUT_SECONDS, metadata=template["metadata"])
        job["input_bindings"] = [api.binding(job["input"]), api.binding(job["environment_input"])] + plan["input_bindings"]
        api.save(study)
        jobs.append(job)
    return jobs
