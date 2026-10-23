"""Freeze and resume run03 with cumulative accounting and explicit budget authority."""

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation import workflow_next_report as report
from ditserve.evaluation.workflow_budget import atomic_json

SOCKET = "unix:///run/ditserve-docker/docker.sock"
STOP_STATUSES = {"blocked_foreign_gpu_workload", "interrupted_foreign_gpu_workload",
                 "gpu_monitor_unavailable", "interrupted", "cleanup_failed"}


def docker(arguments, **kwargs):
    return subprocess.run(["sg", "ditserve-docker", "-c",
                           shlex.join(["docker", "--host", SOCKET, *arguments])], **kwargs)


def helper(source, filename):
    spec = importlib.util.spec_from_file_location(filename.replace("-", "_"), Path(source) / "docker" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(root, image):
    """Build the copied source on CPU, then bind every executed byte."""
    root = Path(root)
    index = 1
    while (root / "sources" / f"r{index:03}").exists():
        index += 1
    source = root / "sources" / f"r{index:03}"
    api.snapshot(REPO, source)
    build = root / "builds" / source.name
    build.mkdir(parents=True)
    commands = [
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON",
         "-DCMAKE_EXE_LINKER_FLAGS=-static", "-DOPENSSL_USE_STATIC_LIBS=TRUE"],
        ["cmake", "--build", str(build), "--parallel", "8"],
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
    ]
    with (source / "build.stdout.log").open("w") as stdout, (source / "build.stderr.log").open("w") as stderr:
        for command in commands:
            subprocess.run(command, check=True, stdout=stdout, stderr=stderr)
    (source / "bin").mkdir()
    shutil.copy2(build / "ditserve", source / "bin/ditserve")
    # The host-built executable must also run in the pinned measurement image.
    probe = docker(["run", "--rm", "--runtime=runc", "--network=none",
                    "-v", str(source) + ":/workspace:ro", image,
                    "/workspace/bin/ditserve", "--help"], capture_output=True, text=True, check=False)
    if probe.returncode:
        (source / "runtime-smoke.stderr.log").write_text(probe.stderr)
        raise RuntimeError("frozen binary failed pinned-runtime CPU smoke: " + probe.stderr.strip())
    atomic_json(source / "build.json", dict(commands=commands,
        compiler=subprocess.run(["c++", "--version"], check=True, capture_output=True, text=True).stdout,
        cmake=subprocess.run(["cmake", "--version"], check=True, capture_output=True, text=True).stdout,
        runtime_image=image, runtime_smoke_stdout=probe.stdout, cpu_only=True,
        linkage="static host toolchain; CPU tested in the immutable measurement image"))
    revision = api.seal(source)
    atomic_json(root / "sources" / (source.name + ".binding.json"), revision)
    return revision


def verify_revision(study, revision):
    current = api.seal(revision["source"])
    if current != revision:
        raise RuntimeError("frozen source/binary contents changed: " + revision["source"])
    if revision not in study["source_revisions"]:
        raise RuntimeError("controller revision is absent from the study manifest")


def remaining(study):
    budget = api.read(Path(study["root"]) / "budget.json", {})
    limit = study["budget_seconds"]
    return None if limit is None else limit - budget.get("charged_seconds", 0.)


def family_spent(study, family):
    jobs = {row["id"]: row for row in study["jobs"]}
    budget = api.read(Path(study["root"]) / "budget.json", {})
    return sum(row["charged_seconds"] for row in budget.get("attempts", [])
               if jobs.get(row["job_id"], {}).get("family") == family)


def family_available(study, family):
    if study["budget_seconds"] is None:
        return None
    names = list(study["family_caps_seconds"])
    return min(remaining(study), sum(study["family_caps_seconds"][name] - family_spent(study, name)
                                    for name in names[:names.index(family) + 1]))


def authorize_unlimited(study, authorization):
    """Apply explicit user authority without resetting charges or old results."""
    from ditserve.evaluation.workflow_budget import StudyBudget
    root = Path(study["root"])
    with (root / "driver.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = api.read(root / "study.json")
        ledger = api.read(root / "budget.json")
        if current.get("active_job") or ledger.get("active"):
            raise RuntimeError("finish/recover active work before changing budget authority")
        with StudyBudget(root, ledger["limit_seconds"]) as budget:
            if budget.limit is not None:
                budget.authorize_unlimited(authorization)
            elif not budget.record.get("authorization_history"):
                raise RuntimeError("unlimited ledger is missing its explicit authorization history")
            history = budget.record["authorization_history"]
        current.update(budget_seconds=None, budget_mode="unlimited",
                       budget_authorizations=history)
        api.save(current)
        study.clear()
        study.update(current)


def decide(study, job, decision, **details):
    row = dict(job=job["id"], decision=decision, **details)
    # Resume is idempotent; repeated views do not generate duplicate decisions.
    if row not in study["decisions"]:
        study["decisions"].append(row)
    job["decision"] = decision
    api.save(study)


def run_job(study, job, *, refresh_report=True, prerequisites_verified=False, strict_exit=False):
    root = Path(study["root"])
    ledger = api.read(root / "launches" / job["id"] / "launch.json", {})
    if ledger and ledger.get("status") != "completed":
        decide(study, job, "incomplete_attempt_requires_new_block", prior_status=ledger["status"])
        return False
    if not ledger:
        failed = ([] if prerequisites_verified else [name for name in job["requires"]
                  if not report.evidence(study, report.current_job(study, name))["qualified"]])
        if failed:
            decide(study, job, "prerequisite_not_qualified", prerequisites=failed)
            return False
        required = job["estimated_seconds"] * 1.2 + 90
        available = family_available(study, job["family"])
        if available is not None and available < required:
            decide(study, job, "complete_block_budget_deferred", estimated_seconds=required,
                   available_seconds=family_available(study, job["family"]))
            return False
        # Freeze the first admission ceiling; recomputing it on resume would
        # change the fingerprint of an already completed block.
        if available is not None:
            job["budget_ceiling_seconds"] = available
        job.pop("decision", None)
        api.save(study)
    bundle = dict(root=study["root"], source=job["source"], source_digest=job["source_digest"],
                  image=study["image"], budget_seconds=study["budget_seconds"], strict_resume=True, jobs=[job])
    path = root / "jobs" / (job["id"] + ".json")
    atomic_json(path, bundle)
    study.update(status="running", active_job=job["id"], updated_unix=time.time())
    api.save(study)
    print(json.dumps(dict(job=job["id"], family=job["family"], remaining_seconds=remaining(study))), flush=True)
    outcome = subprocess.run([sys.executable, str(Path(job["source"]) / "docker/run-workflow-suite.py"),
                              "--jobs", str(path)], check=False)
    ledger = api.read(root / "launches" / job["id"] / "launch.json", {})
    study.update(active_job=None, updated_unix=time.time())
    api.save(study)
    if refresh_report:
        report.build_report(root)
    if ledger.get("status") in STOP_STATUSES:
        raise RuntimeError("study stopped: " + ledger["status"])
    if strict_exit and outcome.returncode != 0 and ledger.get("status") == "completed":
        raise RuntimeError("suite failed while a completed launch remains; refusing stale evidence: " + job["id"])
    return outcome.returncode == 0 and report.evidence(study, job)["qualified"]


def jobs(study, *, family=None, prefix=None):
    return [row for row in study["jobs"] if not row.get("superseded")
            and (family is None or row["family"] == family)
            and (prefix is None or row["logical_id"].startswith(prefix))]


def run_many(study, rows):
    for row in rows:
        run_job(study, row)


def select_screen_depth(study):
    if not report.screening_exact(study):
        return dict(depth=1, status="numeric_gate_failed", rule="default depth1 unless both exact screening repeats favor depth2")
    values = {d: [] for d in (1, 2)}
    for job in jobs(study, prefix="cpu-screen-"):
        ev = report.evidence(study, job)
        if not ev["capacity"]:
            return dict(depth=1, status="capacity_unverified")
        arms = {a["id"]: a["result"] for a in ev["result"]["arms"]}
        for depth in values:
            values[depth].append(arms[f"c8-d{depth}"]["metrics"]["wall_ms_median"])
    depth = 2 if len(values[1]) == 2 and all(b < a for a, b in zip(values[1], values[2])) else 1
    return dict(depth=depth, status="screened", wall_ms=values,
                rule="depth2 only when both exact, capacity-qualified repeats have lower complete four-request wall time")


def select_cpu(study):
    selected = dict(cores=8, depth=1, comparisons={},
                    rule="change each control only if all three independent online pairs qualify and median total throughput improves")
    for name, setting, candidate in (("depth", "depth", 2), ("cores", "cores", 16)):
        comparisons = []
        for job in jobs(study, prefix="cpu-online-" + name):
            ev = report.evidence(study, job)
            comparisons.append(report.comparison(ev["result"], "reference", "candidate",
                                                capacity=ev["capacity"], guarded=ev["qualified"]))
        selected["comparisons"][name] = comparisons
        if len(comparisons) == 3 and all(c["eligible"] for c in comparisons):
            ratios = [c["candidate_summary"]["total"]["throughput_per_second"] /
                      c["reference_summary"]["total"]["throughput_per_second"] for c in comparisons]
            if statistics.median(ratios) > 1:
                selected[setting] = candidate
    # The core-count arm uses the selected screening depth; do not extrapolate
    # its finding to a different depth without measurements.
    if selected["cores"] == 16:
        selected["depth"] = study["selections"]["screen_depth"]["depth"]
    return selected


def archive(study):
    root = Path(study["root"])
    data, destination = root.parent.parent, root.parent.parent / "results/workflow" / root.name
    code = """import os,json,pathlib
for p in (pathlib.Path('/mnt/ssd0'),pathlib.Path('/mnt/ssd0/zyd')):
    s=p.stat()
    if s.st_uid!=0 or s.st_dev!=pathlib.Path('/').stat().st_dev:
        raise RuntimeError('refusing to chmod a host-mounted parent')
    p.chmod(0o755)
os.setgroups([GID]);os.setgid(GID);os.setuid(UID)
from ditserve.results import finalize,load,verify
work=pathlib.Path(WORK);destination=pathlib.Path(DEST)
expected=json.loads((work/'run.json').read_text())
outcome=finalize(work,destination,prune=False)
if outcome.get('status')!='completed':
    raise RuntimeError('archive finalization did not complete: status='+str(outcome.get('status'))+'; reason='+str(outcome.get('reason',outcome)))
assert load(destination)['run']==expected,'canonical workbook roundtrip mismatch'
print(json.dumps(dict(finalization=outcome,verification=verify(destination))))
""".replace("GID", str(os.getgid())).replace("UID", str(os.getuid()))
    code = code.replace("WORK", repr(str(root))).replace("DEST", repr(str(destination)))
    with (root / "archive.stdout.log").open("a") as stdout, (root / "archive.stderr.log").open("a") as stderr:
        result = docker(["run", "--rm", "--runtime=runc", "--network=none", "--cpuset-cpus=58", "--user=0:0",
                         "-e", "PYTHONPATH=/workspace/workers",
                         "-v", study["source"] + ":/workspace:ro", "-v", str(data) + ":" + str(data),
                         study["image"], "/opt/venv/bin/python", "-c", code],
                        stdout=stdout, stderr=stderr, check=False, timeout=3600)
    if result.returncode:
        raise RuntimeError("archive verification failed; see archive.stderr.log; all evidence retained")
    print(json.dumps(dict(status="archive_verified", destination=str(destination))), flush=True)


def execute(study, retry=None, only=None, diagnose_sp=False, optimize_followup=False, accept_sp2=False,
            optimize_runtime=False, optimize_image_cache=False, optimize_encoding=False, encoding_proposal=None,
            optimize_dispatch_refresh=False, optimize_occupancy=False, optimize_cache_combination=False,
            validate_remaining=False):
    if validate_remaining and (retry or only or diagnose_sp or optimize_followup or accept_sp2
                               or optimize_runtime or optimize_image_cache or optimize_encoding
                               or optimize_dispatch_refresh or optimize_occupancy or optimize_cache_combination):
        raise ValueError("remaining validation cannot be combined with another execution mode")
    if optimize_cache_combination and (retry or only or diagnose_sp or optimize_followup or accept_sp2
                                      or optimize_runtime or optimize_image_cache or optimize_encoding
                                      or optimize_dispatch_refresh or optimize_occupancy):
        raise ValueError("cache combination cannot be combined with another execution mode")
    if optimize_occupancy and (retry or only or diagnose_sp or optimize_followup or accept_sp2
                               or optimize_runtime or optimize_image_cache or optimize_encoding or optimize_dispatch_refresh):
        raise ValueError("occupancy optimization cannot be combined with another execution mode")
    if optimize_dispatch_refresh and (retry or only or diagnose_sp or optimize_followup or accept_sp2
                                      or optimize_runtime or optimize_image_cache or optimize_encoding):
        raise ValueError("dispatch refresh cannot be combined with another execution mode")
    if optimize_encoding and (retry or only or diagnose_sp or optimize_followup or accept_sp2
                              or optimize_runtime or optimize_image_cache):
        raise ValueError("encoding optimization cannot be combined with another execution mode")
    if optimize_image_cache and (retry or only or diagnose_sp or optimize_followup or accept_sp2 or optimize_runtime):
        raise ValueError("image cache screening cannot be combined with another execution mode")
    if diagnose_sp and (retry or only):
        raise ValueError("SP diagnosis cannot be combined with only/retry")
    if optimize_followup and (retry or only or diagnose_sp):
        raise ValueError("optimization follow-up cannot be combined with diagnosis/only/retry")
    if accept_sp2 and (retry or only or diagnose_sp or optimize_followup):
        raise ValueError("SP2 acceptance cannot be combined with another execution mode")
    if optimize_runtime and (retry or only or diagnose_sp or optimize_followup or accept_sp2):
        raise ValueError("runtime optimization cannot be combined with another execution mode")
    root = Path(study["root"])
    lock = (root / "driver.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        revision = next(r for r in study["source_revisions"] if r["source"] == str(REPO))
        verify_revision(study, revision)
        recovered = helper(REPO, "recover-serving-study.py").recover_study(root, caller_pid=os.getpid(), driver_lock_held=True)
        if recovered.get("status") == "blocked":
            raise RuntimeError("previous attempt cannot be recovered: " + json.dumps(recovered))
        study.update(active_job=None, status="running")
        study.pop("finished_unix", None)
        study.pop("error", None)
        study["decisions"].append(dict(decision="controller_started", pid=os.getpid(),
            start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
            source_digest=study["source_digest"],
            charged_seconds=api.read(root / "budget.json", {}).get("charged_seconds", 0.), unix=time.time()))
        api.save(study)
        if retry:
            api.retry_block(study, retry)
        if validate_remaining:
            if sorted(os.sched_getaffinity(0)) != [58]:
                raise RuntimeError("remaining validation observer must run on CPU 58")
            from ditserve.evaluation import workflow_remaining_study as remaining_study
            from ditserve.evaluation import workflow_remaining_report as remaining_report
            rows = remaining_study.register(study)
            progress = []
            for row in rows:
                gate = remaining_report.prerequisite_gate(study, row)
                if not gate["passed"]:
                    decision = dict(job=row["id"], decision="remaining_prerequisite_not_qualified", reasons=gate["reasons"])
                    if decision not in study["decisions"]:
                        study["decisions"].append(decision)
                    progress.append(dict(job_id=row["id"], technical_qualified=False, prerequisite_gate=gate))
                    study["selections"]["remaining_validation_progress"] = list(progress)
                    api.save(study)
                    continue
                # The remaining-study gate checks technical and capacity
                # evidence, including explicitly scoped historical subsets.
                # Keep original job fields intact for strict fingerprint resume.
                qualified = run_job(study, row, refresh_report=False, prerequisites_verified=True, strict_exit=True)
                launch = api.read(root / "launches" / row["id"] / "launch.json", {})
                active = api.read(root / "budget.json", {}).get("active")
                if (active is not None or launch.get("status") in STOP_STATUSES | {"budget_exhausted", "running"}
                        or (not launch and row.get("decision") not in {
                            "prerequisite_not_qualified", "incomplete_attempt_requires_new_block"})):
                    raise RuntimeError("remaining validation launch/accounting did not settle: " + row["id"])
                progress.append(dict(job_id=row["id"], technical_qualified=qualified, prerequisite_gate=gate))
                study["selections"]["remaining_validation_progress"] = list(progress)
                api.save(study)
            remaining_study.postprocess(study)
            study["selections"]["remaining_validation"] = remaining_report.summarize(study)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if optimize_cache_combination:
            if sorted(os.sched_getaffinity(0)) != [58]:
                raise RuntimeError("cache combination observer must run on CPU 58")
            from ditserve.evaluation import workflow_cache_combination_study as combination
            from ditserve.evaluation import workflow_cache_combination_report as combination_report
            rows = combination.register(study)
            screen = next(row for row in rows if row["metadata"]["stage"] == "screening")
            if run_job(study, screen):
                screening = combination_report.screening_gate(report.evidence(study, screen))
            else:
                screening = dict(passed=False, reason="screening technical qualification failed")
            study["selections"]["cache_combination_screening"] = screening
            api.save(study)
            for row in rows:
                if row is screen:
                    continue
                if not screening["passed"]:
                    decide(study, row, "cache_combination_screening_not_promising", screening=screening)
                elif not run_job(study, row):
                    raise RuntimeError("cache combination confirmation did not qualify: " + row["id"])
            study["selections"]["cache_combination"] = combination_report.summarize(study)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if optimize_occupancy:
            if sorted(os.sched_getaffinity(0)) != [58]:
                raise RuntimeError("occupancy optimization observer must run on CPU 58")
            from ditserve.evaluation import workflow_occupancy_study as refresh
            from ditserve.evaluation import workflow_occupancy_report as refresh_report
            rows = refresh.register(study)
            screen = next(row for row in rows if row["metadata"]["stage"] == "screening")
            if run_job(study, screen):
                screening = refresh_report.screening_gate(report.evidence(study, screen))
            else:
                screening = dict(passed=False, reason="screening technical qualification failed")
            study["selections"]["occupancy_screening"] = screening
            api.save(study)
            for row in rows:
                if row is screen:
                    continue
                if not screening["passed"]:
                    decide(study, row, "occupancy_screening_not_promising", screening=screening)
                elif not run_job(study, row):
                    raise RuntimeError("occupancy optimization confirmation did not qualify: " + row["id"])
            study["selections"]["occupancy"] = refresh_report.summarize(study)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if optimize_dispatch_refresh:
            if sorted(os.sched_getaffinity(0)) != [58]:
                raise RuntimeError("dispatch refresh observer must run on CPU 58")
            from ditserve.evaluation import workflow_dispatch_refresh_study as refresh
            from ditserve.evaluation import workflow_dispatch_refresh_report as refresh_report
            rows = refresh.register(study)
            screen = next(row for row in rows if row["metadata"]["stage"] == "screening")
            if run_job(study, screen):
                screening = refresh_report.screening_gate(report.evidence(study, screen))
            else:
                screening = dict(passed=False, reason="screening technical qualification failed")
            study["selections"]["dispatch_refresh_screening"] = screening
            api.save(study)
            for row in rows:
                if row is screen:
                    continue
                if not screening["passed"]:
                    decide(study, row, "dispatch_refresh_screening_not_promising", screening=screening)
                elif not run_job(study, row):
                    raise RuntimeError("dispatch refresh confirmation did not qualify: " + row["id"])
            study["selections"]["dispatch_refresh"] = refresh_report.summarize(study)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if optimize_encoding:
            from ditserve.evaluation import workflow_encoding_optimization_study as encoding
            from ditserve.evaluation import workflow_encoding_optimization_report as encoding_report
            rows = encoding.register(study, encoding_proposal)
            run_many(study, rows)
            study["selections"]["encoding_optimization"] = encoding_report.summarize(study)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if optimize_image_cache:
            from ditserve.evaluation.workflow_image_cache_study import run_followup
            run_followup(study, run_job)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if optimize_runtime:
            from ditserve.evaluation.workflow_runtime_optimization_study import run_followup
            run_followup(study, run_job)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if accept_sp2:
            from ditserve.evaluation.workflow_sp2_acceptance_study import run_followup
            run_followup(study, run_job)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if optimize_followup:
            from ditserve.evaluation.workflow_optimization_study import run_followup
            run_followup(study, run_job)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        if diagnose_sp:
            # These controls are independent of the main SP qualification.
            # A failed numeric comparison must not block the other diagnosis.
            run_many(study, api.sp_diagnosis_jobs(study))
            replay = api.sp_linear_replay_job(study)
            if replay is not None:
                run_job(study, replay)
            study.update(status="settled", active_job=None, finished_unix=time.time())
            api.save(study)
            report.build_report(root, final=True)
            archive(study)
            return
        one = lambda name: run_job(study, report.current_job(study, name))
        if only:
            one(only)
            study.update(status="checkpoint", active_job=None)
            api.save(study)
            report.build_report(root)
            return
        # Failure in an SP diagnostic never blocks independent SP1 serving work.
        acceptance = one("acceptance")
        if acceptance:
            cost_ok = one("cost")
            profile = api.profile_from_cost(study) if cost_ok else None
            capacities = {}
            if profile:
                atomic_json(root / "profile.json", profile)
                api.measured_jobs(study, profile)
                for label in ("video", "mixed"):
                    if one("calibration-" + label):
                        capacities[label] = api.calibration_capacity(report.evidence(study,
                            report.current_job(study, "calibration-" + label))["result"])
                atomic_json(root / "calibration.json", dict(profile=profile, capacities=capacities))
            run_many(study, jobs(study, prefix="cpu-screen-"))
            study["selections"].setdefault("screen_depth", select_screen_depth(study))
            api.save(study)
            if profile and set(capacities) == {"video", "mixed"}:
                api.native_followups(study, profile, capacities)
                for job in sorted(jobs(study, prefix="cpu-online-"),
                                  key=lambda j: (j["metadata"]["repeat"], j["metadata"]["comparison"])):
                    cross = job["metadata"]["comparison"] == "cpu-cores"
                    if report.screening_exact(study, cross_cores=cross):
                        run_job(study, job)
                    else:
                        decide(study, job, "cross_core_numeric_gate_failed" if cross else "cross_depth_numeric_gate_failed")
                study["selections"].setdefault("cpu", select_cpu(study))
                api.save(study)
                api.quantum_screen_job(study, profile, capacities)
                one("quantum-screen")
                study["selections"].setdefault("quantum", report.select_quantum(study))
                api.save(study)
                quantum = study["selections"]["quantum"]["quantum_ms"]
                api.mixed_jobs(study, profile, capacities["mixed"], quantum)
                run_many(study, jobs(study, prefix="mixed-"))
                api.ablation_jobs(study, profile, capacities["mixed"], quantum)
                run_many(study, jobs(study, prefix="ablation-"))
            one("sp-diagnostic")
            run_many(study, [j for j in jobs(study, prefix="elastic-")
                            if not j["logical_id"].startswith("elastic-online")])
            if profile and "mixed" in capacities:
                api.elastic_jobs(study, profile, capacities["mixed"])
                one("elastic-online")
                if report.evidence(study, report.current_job(study, "elastic-online"))["qualified"]:
                    api.elastic_repeats(study, profile, capacities["mixed"])
                    run_many(study, jobs(study, prefix="elastic-online-s"))
                api.mixed_jobs(study, profile, capacities["mixed"],
                               study["selections"].get("quantum", {}).get("quantum_ms", 300), confirmation=True)
                for job in jobs(study, family="confirmation"):
                    # A complete three-repeat main comparison precedes new seeds.
                    pattern = job["metadata"]["pattern"]
                    mains = [j for j in jobs(study, prefix="mixed-") if j["metadata"]["pattern"] == pattern]
                    eligible = len(mains) == 3 and all(report.comparison(
                        report.evidence(study, j)["result"], "step", "block",
                        capacity=report.evidence(study, j)["capacity"],
                        guarded=report.evidence(study, j)["qualified"])["eligible"] for j in mains)
                    if eligible:
                        run_job(study, job)
                    else:
                        decide(study, job, "main_comparison_incomplete_or_unqualified")
                api.extension_jobs(study)
                one("overlap-lifecycle")
                lifecycle = report.evidence(study, report.current_job(study, "overlap-lifecycle"))
                if lifecycle["qualified"] and lifecycle["capacity"]:
                    one("overlap-timing")
                    timing = report.evidence(study, report.current_job(study, "overlap-timing"))
                    if timing["qualified"] and timing["capacity"]:
                        one("overlap-trace")
                        if list((root / "traces").glob("*.nsys-rep")):
                            subprocess.run(["taskset", "-c", "58", sys.executable,
                                            str(REPO / "docker/postprocess-workflow-study.py"),
                                            "--root", str(root), "--kind", "nsys"], check=True)
                else:
                    one("capacity-81")
        else:
            for job in jobs(study):
                if job["logical_id"] != "acceptance":
                    decide(study, job, "acceptance_failed")
        study.update(status="settled", active_job=None, finished_unix=time.time())
        api.save(study)
        report.build_report(root, final=True)
        archive(study)
    except BaseException as error:
        study.update(status="interrupted", active_job=None, error=str(error), updated_unix=time.time())
        api.save(study)
        if not validate_remaining:
            report.build_report(root)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="run03")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume-preparation", action="store_true",
                        help="Keep failed CPU build evidence and prepare a new revision before any study or budget exists")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--freeze-revision", action="store_true",
                        help="Freeze new code without resetting budget; only retries/new tasks use it")
    parser.add_argument("--unlimited-budget", action="store_true",
                        help="Apply explicit user authorization for unlimited future cumulative task time")
    parser.add_argument("--budget-authorization", help="Exact user instruction authorizing the budget change")
    parser.add_argument("--retry-block")
    parser.add_argument("--only", help="Run one named logical block, then leave a resumable checkpoint")
    parser.add_argument("--diagnose-sp", action="store_true",
                        help="With --run, execute only independent SP numerical controls using the existing study budget")
    parser.add_argument("--optimize-followup", action="store_true",
                        help="With --run, execute the bounded SP/contention/overlap follow-up")
    parser.add_argument("--accept-sp2", action="store_true",
                        help="With --run, validate repaired SP1 transfer, fixed SP2 pause, and SP1-SP2-SP1 resize")
    parser.add_argument("--optimize-runtime", action="store_true",
                        help="With --run, measure bounded SP2 scratch repair, adaptive quanta and shared VAE decode")
    parser.add_argument("--optimize-image-cache", action="store_true",
                        help="With --run, screen bounded Sana prompt reuse on two paired traces within the extension budget")
    parser.add_argument("--optimize-encoding", action="store_true",
                        help="With --run, qualify and compare Sana synchronization and Wan negative caches")
    parser.add_argument("--encoding-proposal", type=Path, help="Reviewed encoding plan.json or its directory")
    parser.add_argument("--optimize-dispatch-refresh", action="store_true",
                        help="With --run, screen and confirm fresh completion/arrival admission before dispatch")
    parser.add_argument("--optimize-occupancy", action="store_true",
                        help="With --run, screen and confirm direct NVML occupancy checks")
    parser.add_argument("--optimize-cache-combination", action="store_true",
                        help="With --run, screen and confirm Sana cache on the Wan-cache/NVML baseline")
    parser.add_argument("--validate-remaining", action="store_true",
                        help="With --run, drain the authorized validation queue and archive once at the end")
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if args.validate_remaining and not args.run:
        parser.error("--validate-remaining requires --run")
    if args.validate_remaining and (args.retry_block or args.only or args.diagnose_sp or args.optimize_followup
            or args.accept_sp2 or args.optimize_runtime or args.optimize_image_cache or args.optimize_encoding
            or args.optimize_dispatch_refresh or args.optimize_occupancy or args.optimize_cache_combination):
        parser.error("--validate-remaining cannot be combined with another execution mode")
    if args.optimize_cache_combination and not args.run:
        parser.error("--optimize-cache-combination requires --run")
    if args.optimize_cache_combination and (args.diagnose_sp or args.optimize_followup or args.accept_sp2
            or args.optimize_runtime or args.optimize_image_cache or args.optimize_encoding
            or args.optimize_dispatch_refresh or args.optimize_occupancy or args.only or args.retry_block):
        parser.error("--optimize-cache-combination cannot be combined with another execution mode")
    if args.optimize_occupancy and not args.run:
        parser.error("--optimize-occupancy requires --run")
    if args.optimize_occupancy and (args.diagnose_sp or args.optimize_followup or args.accept_sp2
            or args.optimize_runtime or args.optimize_image_cache or args.optimize_encoding
            or args.optimize_dispatch_refresh or args.only or args.retry_block):
        parser.error("--optimize-occupancy cannot be combined with another execution mode")
    if args.optimize_dispatch_refresh and not args.run:
        parser.error("--optimize-dispatch-refresh requires --run")
    if args.optimize_dispatch_refresh and (args.diagnose_sp or args.optimize_followup or args.accept_sp2
            or args.optimize_runtime or args.optimize_image_cache or args.optimize_encoding or args.only or args.retry_block):
        parser.error("--optimize-dispatch-refresh cannot be combined with another execution mode")
    if args.unlimited_budget != bool(args.budget_authorization):
        parser.error("--unlimited-budget and --budget-authorization must be supplied together")
    if args.optimize_encoding and (not args.run or not args.encoding_proposal):
        parser.error("--optimize-encoding requires --run and --encoding-proposal")
    if args.optimize_encoding and (args.diagnose_sp or args.optimize_followup or args.accept_sp2
                                   or args.optimize_runtime or args.optimize_image_cache or args.only or args.retry_block):
        parser.error("--optimize-encoding cannot be combined with another execution mode")
    if args.optimize_image_cache and not args.run:
        parser.error("--optimize-image-cache requires --run")
    if args.optimize_image_cache and (args.diagnose_sp or args.optimize_followup or args.accept_sp2
                                     or args.optimize_runtime or args.only or args.retry_block):
        parser.error("--optimize-image-cache cannot be combined with another execution mode")
    if args.diagnose_sp and not args.run:
        parser.error("--diagnose-sp requires --run")
    if args.diagnose_sp and (args.only or args.retry_block):
        parser.error("--diagnose-sp cannot be combined with --only or --retry-block")
    if args.optimize_followup and not args.run:
        parser.error("--optimize-followup requires --run")
    if args.optimize_followup and (args.diagnose_sp or args.only or args.retry_block):
        parser.error("--optimize-followup cannot be combined with --diagnose-sp, --only or --retry-block")
    if args.accept_sp2 and not args.run:
        parser.error("--accept-sp2 requires --run")
    if args.accept_sp2 and (args.diagnose_sp or args.optimize_followup or args.only or args.retry_block):
        parser.error("--accept-sp2 cannot be combined with --diagnose-sp, --optimize-followup, --only or --retry-block")
    if args.optimize_runtime and not args.run:
        parser.error("--optimize-runtime requires --run")
    if args.optimize_runtime and (args.diagnose_sp or args.optimize_followup or args.accept_sp2
                                 or args.only or args.retry_block):
        parser.error("--optimize-runtime cannot be combined with another execution mode")
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("unsafe run ID")
    root = (args.root or api.DEFAULT_ROOT / args.run_id).resolve()
    if args.prepare:
        if root.exists():
            if (not args.resume_preparation or
                    any(path.name not in ("sources", "builds") for path in root.iterdir())):
                raise FileExistsError("new preparation requires a new directory: " + str(root))
        inspected = docker(["image", "inspect", "ditserve:rewrite-runtime", "--format", "{{.Id}}"],
                           capture_output=True, text=True, check=True)
        image = inspected.stdout.strip()
        if not image.startswith("sha256:") or len(image) != 71:
            raise RuntimeError("expected immutable image ID")
        output = subprocess.run(["/mnt/ssd0/zyd/ditserve-data/tools/nvidia-smi",
                                 "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout
        inventory = []
        for line in output.strip().splitlines():
            ordinal, name, uuid = (part.strip() for part in line.split(","))
            inventory.append(dict(ordinal=int(ordinal), name=name, uuid=uuid))
        inventory.sort(key=lambda r: r["ordinal"])
        if [row["ordinal"] for row in inventory] != [0, 1]:
            raise RuntimeError("expected physical GPUs 0 and 1")
        revision = freeze(root, image)
        study = api.prepare(REPO, root, image, inventory, revision, "/mnt/ssd0/zyd/model")
        api.initial_jobs(study)
        print(json.dumps(dict(status="prepared", root=str(root), source_digest=study["source_digest"])), flush=True)
    study = api.read(root / "study.json")
    if study is None:
        raise FileNotFoundError(root / "study.json")
    if args.unlimited_budget:
        authorize_unlimited(study, args.budget_authorization)
    if args.freeze_revision:
        lock = (root / "driver.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if study.get("active_job") or api.read(root / "budget.json", {}).get("active"):
                raise RuntimeError("finish/recover the active attempt before freezing a revision")
            revision = freeze(root, study["image"])
            study["source_revisions"].append(revision)
            study.update(revision)
            api.save(study)
        finally:
            lock.close()
    if args.run or args.preview or args.archive:
        entry = Path(study["source"]) / "docker/run-next-study.py"
        if entry.resolve() != Path(__file__).resolve():
            forwarded = []
            skip = False
            for argument in arguments:
                if skip:
                    skip = False
                    continue
                if argument == "--budget-authorization":
                    skip = True
                    continue
                if argument.startswith("--budget-authorization=") or argument in (
                        "--prepare", "--resume-preparation", "--freeze-revision", "--unlimited-budget"):
                    continue
                forwarded.append(argument)
            os.execv(sys.executable, [sys.executable, str(entry), *forwarded])
        if args.run:
            execute(study, retry=args.retry_block, only=args.only, diagnose_sp=args.diagnose_sp,
                    optimize_followup=args.optimize_followup, accept_sp2=args.accept_sp2,
                    optimize_runtime=args.optimize_runtime, optimize_image_cache=args.optimize_image_cache,
                    optimize_encoding=args.optimize_encoding, encoding_proposal=args.encoding_proposal,
                    optimize_dispatch_refresh=args.optimize_dispatch_refresh, optimize_occupancy=args.optimize_occupancy,
                    optimize_cache_combination=args.optimize_cache_combination, validate_remaining=args.validate_remaining)
        elif args.archive:
            if study.get("active_job") or api.read(root / "budget.json", {}).get("active"):
                raise RuntimeError("cannot finalize an active study")
            report.build_report(root, final=True)
            archive(study)
        else:
            report.build_report(root)


if __name__ == "__main__":
    main()
