#!/usr/bin/env python3
"""Single V100 phase-conflict protocol. --dry-run is CPU-only and read-only."""

from ..util import ROOT, RUNS, CONFIGS, read_json, atomic_json, child_environment, archive_run

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from . import run_v100 as common

STRATEGIES = ("immediate", "deferred", "blocking", "overlap")
SCHEMA = "ditserve.phase-conflict-suite.v1"


@dataclasses.dataclass
class Budget:
    limit_seconds: float
    used_seconds: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.limit_seconds, self.used_seconds)):
            raise ValueError("budget must be finite")
        if self.limit_seconds <= 0 or self.used_seconds < 0:
            raise ValueError("invalid budget")

    @property
    def remaining_seconds(self):
        return max(0.0, self.limit_seconds - self.used_seconds)

    def can_start(self, estimate_seconds):
        return math.isfinite(estimate_seconds) and estimate_seconds > 0 and estimate_seconds <= self.remaining_seconds

    def charge(self, elapsed_seconds):
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("invalid elapsed time")
        self.used_seconds += elapsed_seconds


def next_screen_shape(completed):
    """A 33-frame OOM branches to 25; never probes 49 after that failure."""
    for frames in (17, 33):
        if frames not in completed:
            return frames
        if completed[frames] != "completed":
            return 25 if frames == 33 and completed[frames] == "oom" and 25 not in completed else None
    return 49 if 49 not in completed else None


def classify_profile(profile, returncode=0):
    if not isinstance(profile, dict):
        return "failed", "missing_profile"
    failure = profile.get("failure") or {}
    category = failure.get("category", "worker_failed")
    if "oom" in category or "out of memory" in str(failure.get("message", "")).lower():
        return "oom", category
    if returncode or profile.get("status") != "completed":
        return "failed", category
    if profile.get("kind") != "real_gpu_phase_conflict" or profile.get("measured") is not True:
        return "failed", "missing_measured_evidence"
    remaining = (profile.get("timings") or {}).get("remaining_ms")
    if not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or not math.isfinite(remaining) or remaining <= 0:
        return "failed", "invalid_completion_timing"
    mode = profile.get("config", {}).get("mode")
    outputs = ("output_a", "output_b") if mode == "case" else ("output_b",)
    if mode not in ("prepare", "case"):
        return "failed", "missing_job_identity"
    if any(not isinstance(profile.get(k), dict) or profile[k].get("finite") is not True or
           not profile[k].get("sha256") for k in outputs):
        return "failed", "missing_finite_output"
    return "completed", None


def validate_identity(profile, job):
    actual = profile.get("config", {})
    return [k for k in job if k != "output_dir" and actual.get(k) != job.get(k)]


def validate_environment(profile, uuid):
    env = profile.get("environment") or {}
    target = common.canonical_gpu_uuid(uuid)
    return bool(env.get("ready") and env.get("device") == "cuda:0" and
                env.get("cuda_device_order") == "PCI_BUS_ID" and target and
                common.canonical_gpu_uuid(env.get("cuda_visible_devices")) == target and
                common.canonical_gpu_uuid((env.get("gpu") or {}).get("uuid")) == target)


def elapsed_from_records(records):
    total = 0.0
    for record in records:
        value = record.get("elapsed_seconds")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("missing or invalid elapsed time; cannot safely resume budget")
        total += value
    return total


def gpu_available(expected_uuid):
    gpu = common.resolve_gpu(expected_uuid)
    samples = []
    for attempt in range(4):
        result = subprocess.run(["nvidia-smi", "--id=" + expected_uuid,
                                 "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=10)
        free, utilization = [float(x.strip()) for x in result.stdout.strip().split(",")]
        samples.append(dict(memory_free_mib=free, utilization_gpu_percent=utilization))
        if math.isfinite(free) and math.isfinite(utilization) and free >= 31000 and 0 <= utilization <= 5:
            gpu.update(samples[-1], availability_samples=samples)
            return gpu
        # NVML utilization covers a recent interval and can remain nonzero just
        # after our previous child exited, even though its memory is all freed.
        if attempt < 3:
            time.sleep(2)
    raise RuntimeError("目标 V100 持续被占用，暂不启动实验: free=%s MiB, utilization=%s%%" % (free, utilization))


class Suite:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.config = common.load_config(args.config)
        self.path = self.root / "manifest.json"
        if args.resume:
            diagnostic_path = self.root / "trace-attempts.json"
            if diagnostic_path.exists():
                try:
                    diagnostic = read_json(diagnostic_path)
                    if (not isinstance(diagnostic, dict) or
                            diagnostic.get("schema_version") != "ditserve.phase-conflict-trace.v1" or
                            not isinstance(diagnostic.get("records"), list) or
                            any(not isinstance(r, dict) or r.get("status") not in ("running", "completed", "failed", "oom")
                                for r in diagnostic["records"])):
                        raise ValueError("invalid diagnostic ledger")
                    diagnostic_used = elapsed_from_records(diagnostic["records"])
                    active = any(r.get("status") == "running" for r in diagnostic["records"])
                except (OSError, ValueError, TypeError) as exc:
                    raise ValueError("cannot safely resume: trace-attempts.json is unreadable or invalid; "
                                     "diagnostic budget must not be omitted") from exc
                if active or diagnostic_used > 0:
                    raise ValueError("cannot resume with running or charged diagnostic attempts in trace-attempts.json; "
                                     "this runner does not combine diagnostic and core budgets, so resume would omit diagnostic charges")
            self.manifest = read_json(self.path)
            if self.manifest.get("schema_version") != SCHEMA:
                raise ValueError("incompatible resume manifest")
            if self.manifest["protocol_config"] != self.config or self.manifest["model_id"] != args.model_id:
                raise ValueError("resume must use the same model and protocol")
            if any(r.get("status") == "running" for r in self.manifest["records"]):
                raise ValueError("unfinished running worker: inspect its process before resuming")
            # Budget includes every previous attempt, including failures.
            used = elapsed_from_records(self.manifest["records"])
            limit = min(args.gpu_hours * 3600, self.manifest["budget"]["limit_seconds"])
            self.budget = Budget(limit, used)
        else:
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("refusing to overwrite nonempty output; use --resume")
            self.budget = Budget(args.gpu_hours * 3600)
            self.manifest = dict(schema_version=SCHEMA, status="running", created_at_utc=common.utc_now(),
                                 protocol_config=self.config, model_id=args.model_id, records=[], decisions=[],
                                 timing_semantics="Conservative child launch-to-exit wall time charged as GPU reservation, including load, warmup, failed attempts and artifact saving.")
        self.records = self.manifest["records"]
        self.manifest["status"] = "running"
        self.manifest.pop("failure", None)
        self.save()

    def save(self):
        self.manifest["budget"] = dataclasses.asdict(self.budget)
        self.manifest["updated_at_utc"] = common.utc_now()
        atomic_json(self.path, self.manifest)

    def job(self, case_id, mode="prepare", frames=17, steps=50, seed=0, cache="none", **extra):
        job = dict(id=case_id, mode=mode, height=480, width=832, frames=frames, steps=steps,
                    seed=seed, cache=cache, model_id=self.args.model_id,
                    revision=self.config["revision"], preset="v100-wan", diagnostic=False)
        job.update(extra)
        return job

    def latest(self, case_id):
        return next((r for r in reversed(self.records) if r["id"] == case_id), None)

    def estimate(self, job):
        matches = [r["elapsed_seconds"] for r in self.records if r.get("status") == "completed" and
                   all(r["job"].get(k) == job.get(k) for k in ("mode", "frames", "steps", "height", "cache"))]
        if matches:
            return max(matches) * 1.25 + 20
        # Reserve startup/CPU encoding plus a conservative quadratic token cost.
        scale = (((job["frames"] - 1) / 4 + 1) / 5) ** 2 * (job["height"] * job["width"] / (480 * 832)) ** 2
        return 150 + 6.0 * job["steps"] * max(.12, scale)

    def group_fits(self, jobs, label):
        todo = [j for j in jobs if not self.latest(j["id"]) or self.latest(j["id"])["status"] == "skipped_budget"]
        estimate = sum(self.estimate(j) for j in todo)
        if not todo or self.budget.can_start(estimate):
            return True
        for job in todo:
            self.skip(job, "skipped_budget", "group estimate exceeds remaining budget: " + label)
        return False

    def skip(self, job, status, reason):
        self.records.append(dict(id=job["id"], job=job, status=status, reason=reason, elapsed_seconds=0.0))
        self.save()

    def run(self, job):
        old = self.latest(job["id"])
        if old and old["status"] in ("completed", "oom"):
            mismatch = validate_identity({"config": old["job"]}, job)
            if mismatch:
                raise ValueError("resume job identity mismatch: " + ",".join(mismatch))
            original = read_json(old["profile_path"])
            status, _ = classify_profile(original, old.get("returncode", 0))
            if status != old["status"] or validate_identity(original, job):
                raise ValueError("resume profile missing, failed, or changed: " + job["id"])
            if old["status"] == "completed" and not validate_environment(original, self.config["gpu_uuid"]):
                raise ValueError("resume device evidence invalid: " + job["id"])
            return old
        if old and old["status"] == "failed" and not self.args.retry_failed:
            raise RuntimeError("failed job requires a reviewed fix and --retry-failed: " + job["id"])
        estimate = self.estimate(job)
        if not self.budget.can_start(estimate):
            self.skip(job, "skipped_budget", "estimated %.1fs > remaining %.1fs" % (estimate, self.budget.remaining_seconds))
            return self.latest(job["id"])
        gpu = gpu_available(self.config["gpu_uuid"])
        attempt = sum(r["id"] == job["id"] for r in self.records)
        directory = self.root / "cases" / (job["id"] + ("-attempt%d" % attempt if attempt else ""))
        directory.mkdir(parents=True, exist_ok=False)
        job = dict(job, output_dir=str(directory))
        job_path = directory / "job.json"
        atomic_json(job_path, job)
        sources = {}
        package = Path(__file__)
        base = package.parent.parent
        for source in (package, base / "gpu" / "phase_conflict.py", base / "gpu" / "profiling.py", base / "util.py"):
            data = source.read_bytes()
            sources[str(source)] = hashlib.sha256(data).hexdigest()
            (directory / (source.stem + ".source.py")).write_bytes(data)
        profile_path = directory / "profile.json"
        record = dict(id=job["id"], job=job, status="running", profile_path=str(profile_path),
                      started_at_utc=common.utc_now(), gpu=gpu, source_sha256=sources, elapsed_seconds=0.0)
        self.records.append(record)
        self.save()
        environment = child_environment()
        environment.update(CUDA_VISIBLE_DEVICES=self.config["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        command = [self.args.python, "-m", "ditserve.experiments.run_phase_conflict", "--worker-job", str(job_path)]
        started = time.monotonic()
        process = None
        timed_out = False
        record["timeout_limit_reason"] = "budget" if self.budget.remaining_seconds <= self.args.worker_timeout else "worker"
        telemetry = common.GpuTelemetry(directory, self.config["gpu_uuid"], 2.0)
        print(json.dumps({"starting": job["id"], "estimate_seconds": estimate,
                          "budget_remaining_seconds": self.budget.remaining_seconds}), flush=True)
        try:
            with (directory / "worker.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record["pid"] = process.pid
                self.save()
                telemetry.start()
                try:
                    try:
                        process.wait(timeout=min(self.args.worker_timeout, self.budget.remaining_seconds))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                finally:
                    record["telemetry"] = telemetry.stop()
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            elapsed = time.monotonic() - started
            self.budget.charge(elapsed)
            record.update(elapsed_seconds=elapsed, finished_at_utc=common.utc_now(),
                          returncode=process.returncode if process else None)
            try:
                profile = read_json(profile_path)
            except (OSError, ValueError):
                profile = None
            status, category = classify_profile(profile, record["returncode"] or 0)
            if timed_out:
                status, category = "failed", "budget_or_worker_timeout"
            if status == "completed":
                mismatch = validate_identity(profile, job)
                if mismatch:
                    status, category = "failed", "job_identity_mismatch:" + ",".join(mismatch)
                elif not validate_environment(profile, self.config["gpu_uuid"]):
                    status, category = "failed", "actual_device_mapping_mismatch"
            record.update(status=status, failure_category=category)
            self.save()
        print(json.dumps({"finished": job["id"], "status": record["status"], "elapsed_seconds": elapsed}), flush=True)
        if record["status"] == "failed":
            raise RuntimeError("worker failed; preserved and stopping: " + job["id"] + " / " + str(category))
        return record

    def prepare(self, frames, seed=0, cache="none", steps=50, label="formal", **extra):
        job = self.job("%s-b-f%d-s%d-%s" % (label, frames, seed, cache), frames=frames, steps=steps,
                       seed=seed, cache=cache, **extra)
        return self.run(job)

    def case_job(self, b, frames, strategy, seed=0, insert=25, repeat=0, height=480, width=832, steps=50,
                 cache="none", label="same", diagnostic=False):
        profile = read_json(b["profile_path"])
        job = self.job("%s-f%d-b%d-s%d-%s-k%d-r%d-%s" %
                        (label, frames, b["job"]["frames"], seed, strategy, insert, repeat, cache),
                        mode="case", frames=frames, steps=steps, seed=seed, cache=cache,
                        strategy=strategy, insert_after=insert, b_latent_path=profile["b_latent_path"],
                        b_metadata_path=profile.get("b_metadata_path", b["profile_path"]),
                        b_frames=b["job"]["frames"], repeat=repeat)
        job.update(height=height, width=width, diagnostic=diagnostic)
        for key in ("height", "width", "steps", "seed", "cache", "attention", "text_encoder_device"):
            job["b_" + key] = profile["config"][key]
        return job

    def run_group(self, jobs, label):
        if not self.group_fits(jobs, label):
            return False
        for job in jobs:
            self.run(job)
        return True

    def execute(self):
        b = self.prepare(17, steps=12, label="smoke", height=256, width=256, diagnostic=True)
        if b["status"] != "completed":
            return
        jobs = [self.case_job(b, 17, s, height=256, width=256, steps=12, insert=6,
                              label="smoke", diagnostic=True) for s in STRATEGIES]
        if not self.run_group(jobs, "smoke correctness"):
            return
        self.check_group_outputs(jobs)
        self.manifest["smoke_validated"] = True
        self.save()
        screened = {}
        while True:
            frames = next_screen_shape(screened)
            if frames is None:
                break
            r = self.prepare(frames, steps=4, label="screen")
            screened[frames] = r["status"]
            if r["status"] not in ("completed", "oom"):
                return
        candidates = sorted((f for f, status in screened.items() if status == "completed"), reverse=True)
        selected = None
        for frames in candidates:
            b = self.prepare(frames)
            if b["status"] == "oom":
                continue
            if b["status"] != "completed":
                return
            immediate = self.case_job(b, frames, "immediate")
            r = self.run(immediate)
            if r["status"] == "oom":
                continue
            if r["status"] != "completed":
                return
            selected = frames
            break
        if selected is None:
            self.manifest["decisions"].append("No candidate completed both independent 50-step requests.")
            return
        self.manifest["selected_frames"] = selected
        self.save()
        jobs = [self.case_job(b, selected, s) for s in STRATEGIES]
        if not self.run_group(jobs, "primary four strategies"):
            return
        self.check_group_outputs(jobs, allow_oom=True)
        repeated = [self.case_job(b, selected, s, repeat=1) for s in ("overlap", "blocking")]
        if not self.run_group(repeated, "reversed key pair"):
            return
        self.check_group_outputs(jobs + repeated, allow_oom=True)
        b1 = self.prepare(selected, seed=1)
        if b1["status"] != "completed":
            return
        seed1 = [self.case_job(b1, selected, s, seed=1) for s in STRATEGIES]
        if not self.run_group(seed1, "seed 1"):
            return
        self.check_group_outputs(seed1, allow_oom=True)
        early = [self.case_job(b, selected, s, insert=10) for s in ("blocking", "overlap")]
        if not self.run_group(early, "earlier insertion"):
            return
        self.check_group_outputs(jobs + early, allow_oom=True)
        if selected > 17:
            short = self.prepare(17)
            if short["status"] != "completed":
                return
            mixed = [self.case_job(short, selected, s, label="mixed") for s in STRATEGIES]
            if not self.run_group(mixed, "mixed frame counts"):
                return
            self.check_group_outputs(mixed, allow_oom=True)
        self.manifest["core_protocol_finished"] = True
        if self.args.defer_cache:
            self.manifest["decisions"].append(
                "Optional cache check deferred by --defer-cache after core completion; "
                "run attribution diagnostics first and account for their budget before any cache follow-up.")
            self.save()
            return
        self.save()
        # Cache is a conditional attribution check, not a mandatory second grid.
        if self.has_interference():
            cb = self.prepare(selected, cache="cache-dit")
            if cb["status"] == "completed":
                cached = [self.case_job(cb, selected, s, cache="cache-dit") for s in STRATEGIES]
                if self.run_group(cached, "conditional fixed-cache check"):
                    self.check_group_outputs(cached, allow_oom=True)
        else:
            self.manifest["decisions"].append("No repeated OOM or >10% makespan penalty signal; cache grid not triggered.")

    def check_group_outputs(self, jobs, allow_oom=False):
        hashes = set()
        for job in jobs:
            r = self.latest(job["id"])
            if allow_oom and r["status"] == "oom":
                continue
            if r["status"] != "completed":
                raise RuntimeError("correctness gate did not complete: " + job["id"])
            prof = read_json(r["profile_path"])
            b_meta = read_json(job["b_metadata_path"])
            if prof["output_b"]["sha256"] != b_meta["output_b"]["sha256"]:
                r.update(status="failed", failure_category="b_output_mismatch")
                self.save()
                raise RuntimeError("B output mismatch: " + job["id"])
            hashes.add(prof["output_a"]["sha256"])
        if len(hashes) > 1:
            self.manifest["decisions"].append("A output mismatch in group: " + ",".join(j["id"] for j in jobs))
            self.save()
            raise RuntimeError("A output mismatch; stop dependent experiments")

    def has_interference(self):
        latest = {r["id"]: r for r in self.records}
        fixed = dict(mode="case", height=480, width=832, steps=50, cache="none", seed=0,
                     frames=self.manifest.get("selected_frames"), b_frames=self.manifest.get("selected_frames"),
                     insert_after=25, diagnostic=False)
        relevant = [r for r in latest.values() if all(r["job"].get(k) == v for k, v in fixed.items())]
        if any(sum(r["status"] == "oom" and r["job"].get("strategy") == strategy for r in relevant) >= 2
               for strategy in ("blocking", "overlap")):
            return True
        times = {s: [] for s in STRATEGIES}
        for r in relevant:
            if r["status"] == "completed":
                value = read_json(r["profile_path"]).get("timings", {}).get("remaining_ms")
                if isinstance(value, (int, float)):
                    times[r["job"]["strategy"]].append(value)
        return bool(len(times["overlap"]) >= 2 and times["blocking"] and statistics.median(times["overlap"]) > 1.1 * statistics.median(times["blocking"]))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=RUNS / ("phase-conflict-%d" % time.time_ns()))
    p.add_argument("--results-root", type=Path, default=ROOT / "results")
    p.add_argument("--config", type=Path, default=CONFIGS / "v100.json")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--model-id")
    p.add_argument("--gpu-hours", type=float, default=6)
    p.add_argument("--worker-timeout", type=float, default=3600)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--defer-cache", action="store_true",
                   help="Finish after the core protocol; reserve remaining budget for separate attribution diagnostics")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_job:
        from ditserve.gpu.phase_conflict import run_worker
        result = run_worker(read_json(args.worker_job))
        return 0 if result["status"] == "completed" else 1
    if not math.isfinite(args.gpu_hours) or not 0 < args.gpu_hours <= 6:
        raise ValueError("budget must be positive and at most 6 GPU hours")
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        raise ValueError("worker timeout must be positive")
    if args.dry_run:
        print(json.dumps(dict(schema_version=SCHEMA, budget_gpu_hours=args.gpu_hours, strategies=STRATEGIES,
                              smoke=dict(height=256, width=256, frames=17, steps=12, insert_after=6),
                              screening="17 -> 33 -> 49; 33 OOM -> 25 and stop", max_frames=49,
                              formal_steps=50, insert_after=[25, 10], seeds=[0, 1],
                              defer_cache=args.defer_cache,
                              timing="B-ready remaining completion, not E2E or online goodput"), indent=2))
        return 0
    args.model_id = args.model_id or (ROOT / ".cache/runs/v100/environment/model-path.txt").read_text().strip()
    def stop_suite(signum, frame):
        raise KeyboardInterrupt("suite terminated; reap active worker before saving budget")
    signal.signal(signal.SIGTERM, stop_suite)
    import fcntl
    # Serializes this experiment suite across output directories, not external users.
    (ROOT / ".cache").mkdir(parents=True, exist_ok=True)
    with (ROOT / ".cache" / "phase-conflict-gpu1.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite = Suite(args)
        try:
            suite.execute()
            suite.manifest["status"] = "completed" if suite.manifest.get("core_protocol_finished") else "partial"
        except BaseException as exc:
            suite.manifest.update(status="stopped", failure={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            suite.manifest["finished_at_utc"] = common.utc_now()
            suite.save()
            try:
                from .summarize_phase_conflict import summarize, write_report, export_media
                summary = summarize(suite.root)
                write_report(suite.root, summary)
                suite.manifest["media_export"] = export_media(suite.root, summary)
                suite.save()
                archive_run(suite.root, "phase-conflict", args.results_root)
            except Exception as exc:
                print("Summary unavailable: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
