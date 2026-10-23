"""Append-only, continuously queued validations on the existing two-GPU study."""

import copy
import importlib.util
import math
from pathlib import Path
import time
import types

from . import workflow_contention_study as contention
from . import workflow_next_study as api
from . import workflow_next_report as report
from . import workflow_remaining_online as online
from .workflow_budget import atomic_json

SCHEMA = "ditserve.workflow.remaining-plan.v1"
CHECKPOINT = "checkpoints/before-remaining-matrix-r017"
SELECTION = "remaining_validation"
PROTECTED_PREFIXES = online.PROTECTED_PREFIXES
NSYS = "/data/tools/nsight-workflow/extracted/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys"
NSYS_HOST = "/mnt/ssd0/zyd/ditserve-data/tools/nsight-workflow/extracted/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys"
UNAVAILABLE = [
    dict(item="native SP2 elastic pause/resize/interleave/online", reason="original numerical prerequisite failed; repaired chain is separately validated"),
    dict(item="16-core online comparison", reason="cross-core exact numerical gate failed; no qualification waiver"),
    dict(item="production full-row padding R0 and removing output clone", reason="new production implementations and numerical acceptance required"),
    dict(item="scratch-policy Sana interleave", reason="existing interleave acceptance supports the original repaired policy only"),
    dict(item="dual-independent-SP1 system throughput and online resize break-even strategy", reason="separate system workload and production cost policy not implemented; single-request four-arm timings do not establish system throughput"),
    dict(item="bounded shared-VAE production stream pool/admission", reason="lifecycle/performance experiments are not a production scheduler implementation"),
    dict(item="A100, 4/8 GPUs, P2P transport and Wan2.2-5B", reason="different hardware or unimplemented model/transport required"),
]


def _context(study):
    root = Path(study["root"])
    checkpoint = root / CHECKPOINT
    manifest = api.read(checkpoint / "manifest.json")
    if not isinstance(manifest, list):
        raise ValueError("remaining validations require an immutable post-bridge checkpoint")
    bindings = {}
    def bind(path, expected=None):
        actual = api.binding(path)
        if expected is not None and actual != expected:
            raise ValueError("remaining dependency changed: " + str(path))
        bindings[str(path)] = actual
        return actual
    bind(checkpoint / "manifest.json")
    for name in ("study.json", "budget.json"):
        path = checkpoint / name
        rows = [x for x in manifest if x.get("copy") == str(path)]
        if len(rows) != 1 or any(bind(path)[key] != rows[0][key] for key in ("bytes", "sha256")):
            raise ValueError("checkpoint manifest mismatch")
    before = api.read(checkpoint / "study.json")
    ledger_before = api.read(checkpoint / "budget.json")
    ledger = api.read(root / "budget.json")
    n, m = len(before["jobs"]), len(ledger_before["attempts"])
    if n != 86 or m != 77 or before.get("active_job") or ledger_before.get("active") or ledger.get("active"):
        raise ValueError("idle 86-job/77-attempt post-bridge checkpoint required")
    if study["jobs"][:n] != before["jobs"] or ledger["attempts"][:m] != ledger_before["attempts"]:
        raise ValueError("historical job or attempt prefix changed")
    for key in ("run_id", "root", "image", "model_root", "model_bindings", "gpu_mapping", "budget_seconds",
                "budget_mode", "budget_authorizations", "family_caps_seconds", "assumptions"):
        if study.get(key) != before.get(key):
            raise ValueError("historical contract changed: " + key)
    if ledger.get("limit_seconds", 1) is not None or ledger.get("authorization_history") != ledger_before.get("authorization_history"):
        raise ValueError("existing unlimited authority must remain unchanged")
    charge = ledger_before["charged_seconds"]
    added_ids = {j["id"] for j in study["jobs"][n:]}
    for attempt in ledger["attempts"][m:]:
        amount = attempt["charged_seconds"]
        if attempt["job_id"] not in added_ids or type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
            raise ValueError("invalid appended remaining charge")
        charge += amount
    if charge != ledger["charged_seconds"]:
        raise ValueError("cumulative ledger replay mismatch")
    if study["source"] == before["source"] or api.seal(study["source"]) != {k: study[k] for k in ("source", "source_files", "source_digest")}:
        raise ValueError("freeze and verify a new source revision before registering remaining work")
    old_revision = next(x for x in before["source_revisions"] if x["source"] == before["source"])
    if api.seal(old_revision["source"]) != old_revision:
        raise ValueError("prior frozen source changed")
    protected = lambda rev: [x for x in rev["source_files"] if x["path"].startswith(PROTECTED_PREFIXES)]
    if protected(study) != protected(old_revision):
        raise ValueError("model/runtime/controller/config source changed; independent numerical requalification required")
    profile, calibration = api.read(root / "profile.json"), api.read(root / "calibration.json")
    if calibration.get("profile") != profile:
        raise ValueError("frozen calibration/profile mismatch")
    contention._profile(profile)
    video = report.evidence(study, report.current_job(study, "calibration-video"))
    if not all(video[k] for k in ("guarded", "qualified", "capacity")):
        raise ValueError("video anchor calibration unqualified")
    anchors = contention._anchors(profile, list(report.native_arms(video["result"]).values()))
    anchors["mixed_capacity"] = calibration["capacities"]["mixed"]
    for name in ("profile.json", "calibration.json"):
        bind(root / name)
    prerequisites = set(online.BASE_PREREQUISITES) | {"elastic-repaired-transfer", "elastic-repaired-pause",
        "elastic-repaired-resize", "overlap-isolated-warmed-lifecycle", "cpu-screen-r0", "cpu-screen-r1"}
    for logical in sorted(prerequisites):
        job = report.current_job(study, logical)
        for path in (job["input"], root / "launches" / job["id"] / "launch.json", Path(job["output"]) / "run.json"):
            bind(path)
        if job.get("environment_input"):
            bind(job["environment_input"])
    prior_audit = root / "maintenance/final-integrity-cache-combination-r017-20261022T110030Z-c838aabb.json"
    audit = api.read(prior_audit)
    if audit.get("status") != "passed" or audit.get("errors"):
        raise ValueError("prior complete integrity audit required")
    bind(prior_audit)
    return dict(profile=profile, anchors=anchors, input_bindings=sorted(bindings.values(), key=lambda x: x["path"]),
        history=dict(checkpoint=str(checkpoint), jobs=n, attempts=m, charged_seconds=ledger_before["charged_seconds"],
                     prior_source=before["source"], prior_source_digest=before["source_digest"]),
        inheritance="Direct input/config/result/launch bindings plus the prior full integrity audit; unchanged protected computation sources. Large historical diagnostic arrays are not fresh-hashed per launch.")


def _cpu_templates(study, profile, capacity):
    rows = []
    for repeat, seed in enumerate((5301, 5302, 5303)):
        workload = api.trajectory(profile, seed, count=16, videos_only=True, pattern="bursty", load=.8, lambda0=capacity)
        definitions = [dict(id=f"depth{d}", policy="block_elastic", quantum_ms=300, decode_frames=1,
                            chunked_decode=True, late_short_ms=2000, depth=d) for d in (0, 1, 2)]
        definitions = definitions[repeat:] + definitions[:repeat]
        config = api.cluster_config(profile, workload, definitions, cores=8, depth=1)
        for phase in config["phases"]:
            phase.update(controller_cpu_affinity=[59], record_dispatch_timeline=True,
                         dispatch_refresh=False, occupancy_backend="nvml")
        rows.append(dict(id=f"remaining-cpu-depth-s{seed}", config=config, native=True, devices=[0, 1],
            requires=["acceptance", "calibration-video", "cpu-screen-r0", "cpu-screen-r1"],
            estimated_seconds=1800, timeout_seconds=4200,
            metadata=dict(role="remaining_cpu_depth", seed=seed, repeat=repeat, frozen_video_capacity=capacity, monitor_cpu_affinity=[58],
                comparisons=[["depth0", "depth1"], ["depth1", "depth2"]], source_scope="three fresh same-revision repeats; does not overwrite the old budget-exhausted block",
                rule="all outputs exact; three complete independent seeds; per-seed video p95 <=105%, throughput >=97%; descriptive paired medians; no default change")))
    return rows


def auxiliary_templates(study, profile, calibration):
    from .workflow_elastic_validation import normalize_config as elastic_config
    from .workflow_sp_comparison import normalize_config as sp_config
    from . import workflow_overlap_isolated as lifecycle
    from . import workflow_overlap_performance as performance
    from . import workflow_overlap_trace as trace
    from . import workflow_overlap_bundle as bundle
    rows = _cpu_templates(study, profile, calibration["capacities"]["video"])
    root = Path(study["root"])
    repaired = api.read(report.current_job(study, "elastic-repaired-resize")["input"])
    repaired.update(stage="interleave", seeds=[5401, 5402], timeout_seconds=3000)
    rows.append(dict(id="remaining-repaired-interleave", config=elastic_config(repaired), module="workflow_elastic_validation",
        devices=[0, 1], requires=["elastic-repaired-transfer", "elastic-repaired-pause", "elastic-repaired-resize"],
        estimated_seconds=1800, timeout_seconds=3300,
        metadata=dict(role="remaining_interleave", monitor_cpu_affinity=[58], performance_evidence=False,
                      scope="repaired policy only; actual Sana insertion and Wan continuation exactness")))
    for seed in (5201, 5202, 5203):
        config = sp_config(dict(seeds=[seed], gpu_uuids=[x["uuid"] for x in study["gpu_mapping"]],
            devices=["cuda:0", "cuda:1"], cpu_threads=8, timeout_seconds=10800, arm_timeout_seconds=2400,
            cpu_masks={"cuda:0": list(range(8)), "cuda:1": list(range(60, 68))}))
        rows.append(dict(id=f"remaining-sp-comparison-s{seed}", config=config, module="workflow_sp_comparison",
            devices=[0, 1], requires=["elastic-repaired-transfer", "elastic-repaired-pause", "elastic-repaired-resize"],
            estimated_seconds=5400, timeout_seconds=10800,
            metadata=dict(role="remaining_sp_comparison", seed=seed, monitor_cpu_affinity=[58],
                scope="native SP1/repaired SP1/fixed SP2/resize; preencoded and full CPU-inclusive timing separately; no system throughput claim")))
    cases = []
    for a, b, device in ((17, 33, 0), (33, 17, 1), (33, 33, 0), (17, 17, 1)):
        config = lifecycle.normalize_config(dict(model_path="/models/wan2.1-t2v-1.3b", device="cuda:0",
            cpu_threads=8, cpu_masks={"cuda:0": list(range(0 if device == 0 else 60, 8 if device == 0 else 68))},
            gpu_uuid=study["gpu_mapping"][device]["uuid"], height=480, width=832, steps=20, seed=5501,
            frame_pairs=[[a, b]], reuse_streams=True, trim_after_references=True, release_b_before_a_decode=True,
            repeats=1, lifecycle_pairs=3, warmup_pairs=2, arm_timeout_seconds=900))
        cases.append(dict(id=f"a{a}-b{b}", device_ordinal=device, gpu_uuid=config["gpu_uuid"], config=config))
    rows.append(dict(id="remaining-vae-lifecycle-bundle", config=bundle.normalize_config(dict(cases=cases)),
        module="workflow_overlap_bundle", devices=[0, 1], requires=["acceptance"], estimated_seconds=4200,
        timeout_seconds=6000, metadata=dict(role="remaining_vae_lifecycle_bundle", monitor_cpu_affinity=[58],
            performance_evidence=False, scope="two concurrent disjoint-GPU lifecycle diagnostics per wave; no timing comparison")))
    for case in cases:
        a, b = case["config"]["frame_pairs"][0]
        config = {key: value for key, value in case["config"].items() if key not in
                  ("arm_specs", "repeats", "memory_diagnostics", "lifecycle_pairs", "warmup_pairs")}
        if [a, b] != [17, 17]:
            config["experiment_scope"] = "long_frame_pairs_v1"
        config.update(memory_diagnostics=False, arm_timeout_seconds=900)
        config = performance.normalize_config(config)
        name = "remaining-vae-performance-" + case["id"]
        rows.append(dict(id=name, config=config, module="workflow_overlap_performance", devices=[case["device_ordinal"]],
            requires=["remaining-vae-lifecycle-bundle"], estimated_seconds=2400, timeout_seconds=4200,
            metadata=dict(role="remaining_vae_performance", lifecycle_case=case["id"], monitor_cpu_affinity=[58],
                          scope="two independent process blocks AB/BA, two warmups and three technical repeats; no kernel-overlap claim")))
        capture = copy.deepcopy(config)
        capture.update(device="cuda:0", cpu_masks={"cuda:0": config["cpu_masks"][config["device"]]})
        rows.append(dict(id="remaining-vae-trace-" + case["id"], config=trace.normalize_config(capture),
            module="workflow_overlap_trace", devices=[case["device_ordinal"]], requires=[name],
            estimated_seconds=900, timeout_seconds=1800, profiling=True,
            metadata=dict(role="remaining_vae_trace", lifecycle_case=case["id"], monitor_cpu_affinity=[58],
                          performance_evidence=False, scope="one independent captured shared-stream pair after two warmups; CUDA overlap requires parsed kernel intersections")))
    return rows


def topological_order(templates):
    by_id = {x["id"]: x for x in templates}
    if len(by_id) != len(templates):
        raise ValueError("duplicate remaining job identity")
    ordered, done = [], set()
    while len(ordered) < len(templates):
        eligible = [x for x in templates if x["id"] not in done and
                    all(dep not in by_id or dep in done for dep in x["requires"])]
        if not eligible:
            raise ValueError("remaining dependency cycle")
        item = eligible[0]
        ordered.append(item); done.add(item["id"])
    return ordered


def build_plan(study):
    context = _context(study)
    templates = online.templates(study, context["profile"], context["anchors"])
    for row in templates:
        row.update(native=True, devices=[0, 1])
    templates += auxiliary_templates(study, context["profile"], api.read(Path(study["root"]) / "calibration.json"))
    for row in templates:
        row["metadata"]["remaining_validation"] = True
        row["metadata"]["source_digest"] = study["source_digest"]
    return dict(schema=SCHEMA, source=study["source"], source_digest=study["source_digest"],
        history=context["history"], input_bindings=context["input_bindings"], inheritance=context["inheritance"],
        jobs=topological_order(templates), unavailable=copy.deepcopy(UNAVAILABLE), default_change=False,
        execution="one suite owner; dual-GPU online/SP jobs; only lifecycle diagnostics run concurrently across disjoint GPUs; one final archive")


MODULES = {"remaining_interleave": "workflow_elastic_validation",
           "remaining_sp_comparison": "workflow_sp_comparison",
           "remaining_vae_lifecycle_bundle": "workflow_overlap_bundle",
           "remaining_vae_performance": "workflow_overlap_performance",
           "remaining_vae_trace": "workflow_overlap_trace"}


def expected_command(study, identifier, config, *, native=False, role=None):
    root = Path(study["root"])
    path, output = root / "inputs" / (identifier + ".json"), root / "measurements" / identifier
    cp = lambda p: api.container_path(study, p)
    if native:
        return ["/workspace/bin/ditserve", "--root", "/data", "--package-root", "/workspace/workers",
                "--python", "/opt/venv/bin/python", "run", "--config", cp(path),
                "--env", cp(root / "inputs" / (identifier + ".env.json")), "--allow-gpu",
                "--run-id", identifier, "--keep-raw", "--work-dir", cp(output),
                "--results-root", cp(root / "native-results"),
                "--ledger", cp(root / "native-ledgers" / (identifier + ".json"))]
    command = ["/opt/venv/bin/python", "-m", "ditserve.evaluation." + MODULES[role],
               "--config", cp(path), "--out", cp(output)]
    if role == "remaining_vae_trace":
        from .workflow_overlap_trace import capture_range
        command = [NSYS, "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
            "--capture-range=nvtx", "--capture-range-end=stop", "--kill=none",
            "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0", "--nvtx-capture=" + capture_range(config),
            "--output", cp(root / "traces" / identifier)] + command
    return command


def register(study):
    plan = build_plan(study)
    allowed = {x["id"] for x in plan["jobs"]}
    prefix = plan["history"]["jobs"]
    if any(x["id"] not in allowed for x in study["jobs"][prefix:]):
        raise ValueError("unregistered suffix after remaining checkpoint")
    root = Path(study["root"])
    plan_path = root / "inputs/remaining-validation-plan.json"
    if plan_path.exists() and api.read(plan_path) != plan:
        raise ValueError("remaining plan changed after preregistration")
    if not plan_path.exists():
        atomic_json(plan_path, plan)
    common = plan["input_bindings"] + [api.binding(plan_path)]
    rows = []
    for template in plan["jobs"]:
        existed = any(j["id"] == template["id"] for j in study["jobs"])
        row = api.add_job(study, template["id"], "extension", template["config"], native=template.get("native", False),
            module=template.get("module"), devices=template["devices"], requires=template["requires"],
            estimate=template["estimated_seconds"], timeout=template["timeout_seconds"], metadata=template["metadata"])
        expected = [api.binding(row["input"])] + ([api.binding(row["environment_input"])] if row.get("environment_input") else []) + common
        fields = dict(command=expected_command(study, row["id"], template["config"],
                          native=template.get("native", False), role=template["metadata"]["role"]),
            id=template["id"], logical_id=template["id"], block=template["id"],
            input=str(root / "inputs" / (template["id"] + ".json")),
            output=str(root / "measurements" / template["id"]),
            command_type="native" if template.get("native") else "remaining_gpu",
            gpu_mapping=[study["gpu_mapping"][d] for d in template["devices"]],
            estimated_seconds=template["estimated_seconds"], timeout_seconds=template["timeout_seconds"],
            minimum_budget_seconds=120)
        if template.get("native"):
            fields["environment_input"] = str(root / "inputs" / (template["id"] + ".env.json"))
        if template.get("profiling"):
            expected += [api.binding(NSYS_HOST)]
            (root / "traces").mkdir(exist_ok=True)
            fields.update(profiling=True, counters=False)
        elif row.get("profiling") or row.get("counters"):
            raise ValueError("normal timing cannot acquire profiler/counter flags")
        if existed and (row["input_bindings"] != expected or row["metadata"] != template["metadata"] or
                        row["source"] != study["source"] or row["source_digest"] != study["source_digest"] or
                        any(row.get(key) != value for key, value in fields.items())):
            raise ValueError("remaining registration differs from frozen input/source")
        row.update(fields)
        row["input_bindings"] = expected
        api.save(study)
        rows.append(row)
    return rows


def postprocess(study):
    """CPU-only exports of this queue's qualified captures, without archiving."""
    root = Path(study["root"])
    source = Path(study["source"])
    spec = importlib.util.spec_from_file_location("remaining_postprocessor", source / "docker/postprocess-workflow-study.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args = types.SimpleNamespace(nsys=Path(NSYS_HOST), timeout=1800)
    receipts = []
    for job in study["jobs"]:
        if job.get("metadata", {}).get("role") != "remaining_vae_trace":
            continue
        ev = report.evidence(study, job)
        capture = root / "traces" / (job["id"] + ".nsys-rep")
        entry = dict(job=job["id"], status="unavailable", capture=str(capture))
        if ev["qualified"] and ev["capacity"] and capture.exists():
            output = capture.with_suffix(".parsed.json")
            try:
                before = api.binding(capture)
                # Export from this capture on every invocation. An unbound
                # SQLite left by an interrupted export cannot supply evidence.
                for prior in (capture.with_suffix(".sqlite"), output):
                    if prior.exists():
                        saved = prior.with_name(prior.name + ".prior-" + str(time.time_ns()))
                        prior.rename(saved)
                        entry.setdefault("preserved_previous_exports", []).append(api.binding(saved))
                parsed = module.parse_nsys(capture, output, args, entry,
                    capture.with_suffix(".export.stdout.log"), capture.with_suffix(".export.stderr.log"))
                if api.binding(capture) != before:
                    raise ValueError("capture changed during its CPU export")
                atomic_json(output, parsed)
                entry.update(status="parsed", capture_binding=before, parsed=api.binding(output))
            except Exception as error:
                entry.update(status="parse_failed", error=str(error))
        receipts.append(entry)
    atomic_json(root / "remaining-traces.json", dict(schema="ditserve.remaining.trace-postprocess.v1", entries=receipts))
    return receipts
