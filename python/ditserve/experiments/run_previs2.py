"""Independent, serial 12 aggregate GPU-hour supervisor for the run02 pilot.

Import and --dry-run are CPU-only. Only a child launched here may be signalled.
Elapsed charges include process launch/model load, failure and GPU release wait.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from ..util import ROOT, CONFIGS, child_environment, utc_now

SCHEMA = "ditserve.previs2.suite.v1"
STAGES = ("correctness", "probe", "calibration", "mechanism", "online", "confirm")
PACKAGES = ("torch", "diffusers", "cache-dit", "transformers")
EXPECTED_PACKAGES = {"torch": "2.10.0+cu126", "diffusers": "0.36.0",
                     "cache-dit": "1.5.0", "transformers": "4.57.6"}
DEADLINE_HOURS = dict(correctness=.5, probe=1.5, calibration=6, mechanism=7, online=11, confirm=12)
MAX_SECONDS = 43200.0
CLEANUP_RESERVE = 60.0
MIN_WORK_SECONDS = 30.0
TOLERANCE = 1e-6


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    """Persist ledger updates across ordinary process interruption."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_hashes():
    """Conservative closure: freeze all experiment and GPU-base modules before any GPU job.

    This includes the inherited worker/quality/profiling/phase_conflict base,
    the new experiment/policy and supervisor. Even adding a module invalidates it.
    """
    directory = Path(__file__).parent
    names = {p.name: p for p in directory.glob("*.py") if p.is_file()}
    names.update({"gpu/" + p.name: p for p in (directory.parent / "gpu").glob("*.py") if p.is_file()})
    required = {"previs2_experiment.py", "previs2_policy.py", "run_previs2.py", "gpu/worker.py"}
    if not required.issubset(names):
        raise ValueError("finish all run02 source modules before freezing: " + str(sorted(required - set(names))))
    return {name: file_sha(path) for name, path in sorted(names.items())}


def source_identity(python):
    script = "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in %r}))" % (PACKAGES,)
    versions = json.loads(subprocess.check_output([str(Path(python).absolute()), "-c", script], text=True, timeout=30))
    if versions != EXPECTED_PACKAGES:
        raise ValueError("run02 requires the existing fixed runtime versions: " + repr(versions))
    hashes = source_hashes()
    identity = dict(worker_sha256=hashes["gpu/worker.py"], source_hashes=hashes,
                    packages=versions, entrypoint_sha256=file_sha(Path(__file__).resolve()))
    identity["sha256"] = fingerprint(identity)
    return identity


def validate_source_identity(identity):
    if not isinstance(identity, dict):
        return False
    hashes = identity.get("source_hashes", {})
    sha = lambda v: isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
    valid_name = lambda k: isinstance(k, str) and k.endswith(".py") and (Path(k).name == k or k == "gpu/" + Path(k).name)
    return (isinstance(hashes, dict) and bool(hashes)
            and all(valid_name(k) for k in hashes)
            and all(sha(v) for v in hashes.values())
            and identity.get("worker_sha256") == hashes.get("gpu/worker.py")
            and identity.get("packages") == EXPECTED_PACKAGES and sha(identity.get("entrypoint_sha256"))
            and identity.get("sha256") == fingerprint({k: v for k, v in identity.items() if k != "sha256"}))


def environment_matches(rows, identity):
    environments = [row for row in rows if row.get("type") == "environment"]
    return (validate_source_identity(identity) and len(environments) == 1
            and environments[0].get("status") == "completed"
            and environments[0].get("versions") == identity["packages"]
            and environments[0].get("source_hashes") == identity["source_hashes"])


def attempt_records(attempt):
    try:
        rows = [json.loads(line) for line in (Path(attempt["output"]) / "records.jsonl").read_text().splitlines() if line.strip()]
        return rows if all(isinstance(row, dict) for row in rows) else []
    except (OSError, ValueError, KeyError):
        return []


def correctness_passed(attempt, identity):
    if attempt.get("status") != "completed" or attempt.get("source_identity") != identity:
        return False
    rows = attempt_records(attempt)
    checks = [row for row in rows if row.get("type") == "correctness"]
    expected = {"official-none", "official-cache", "cold-policy-interleave-16"}
    expected.update("%s-%s-%s-%d" % (cache, mode, interaction, progress)
        for cache in ("none", "cache") for mode, interaction, progress in
        (("keep", "pause", 25), ("hot", "pause", 50), ("hot", "interleave", 75)))
    return (environment_matches(rows, identity) and len(checks) == len(expected)
            and {row.get("case_id") for row in checks} == expected
            and all(row.get("status") == "completed" and all(row.get(k) is True for k in
                    ("output_exact_match", "latent_exact_match", "path_exact_match")) for row in checks)
            and all(row.get("memory_release_verified") is True for row in checks
                    if "-hot-" in row.get("case_id", "") or row.get("case_id") == "cold-policy-interleave-16")
            and all(row.get("trajectory_verified") is True for row in checks
                    if row.get("case_id") == "cold-policy-interleave-16"))


def finite_number(value, minimum=0):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= minimum


def charged_seconds(records):
    values = []
    for record in records:
        if record.get("status") in ("running", "cleanup_unconfirmed") or record.get("charge_closed") is False:
            raise ValueError("unfinished owned process/charge; manual reconciliation required before resume")
        if record.get("status") not in {"completed", "failed", "partial", "budget_exhausted", "interrupted", "not_run"}:
            raise ValueError("unknown attempt status cannot close a budget charge")
        value = record.get("elapsed_seconds")
        if not finite_number(value):
            raise ValueError("every attempt, including failures, needs a finite nonnegative charge")
        values.append(value)
    return math.fsum(values)


def validate_ledger(manifest):
    limit = manifest.get("limit_seconds")
    used = manifest.get("used_seconds")
    charged = charged_seconds(manifest.get("records", []))
    if not finite_number(limit) or not 0 < limit <= MAX_SECONDS or not finite_number(used):
        raise ValueError("invalid budget ledger")
    if abs(charged - used) > TOLERANCE:
        raise ValueError("budget used_seconds differs from all ended attempt charges")
    if used > limit + TOLERANCE:
        raise ValueError("actual budget overshoot retained; additional GPU work is forbidden")
    return used


def protected_run01():
    root = (ROOT / "results/previs/run01").resolve()
    names = ("delivery-manifest.json", "manifest.json")
    for name in names:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError("run01 protection requires its final ordinary " + name)
    return {"suite_root": str(root), "files": {name: {"sha256": file_sha(root / name),
            "bytes": (root / name).stat().st_size} for name in names}}


def check_protection(expected):
    if protected_run01() != expected:
        raise ValueError("protected run01 delivery/manifest changed; run02 stopped without rewriting run01")


def archive_sources(root, identity, config):
    directory = root / ("source-" + identity["sha256"][:12])
    if directory.is_symlink():
        raise ValueError("source archive must be an ordinary directory")
    directory.mkdir(exist_ok=True)
    package = Path(__file__).parent
    files = {name: (package.parent / "gpu" / name[4:] if name.startswith("gpu/") else package / name)
             for name in identity["source_hashes"]}
    for name, source in files.items():
        expected = identity["source_hashes"][name]
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("source changed during archive: " + name)
        target = directory / name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != data:
                raise ValueError("existing source archive differs: " + name)
        else:
            with target.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
    descriptor = {"schema_version": "previs2-source-archive.v1", "source_identity": identity,
                  "protocol": config, "protocol_sha256": fingerprint(config)}
    target = directory / "archive.json"
    if target.exists():
        if target.is_symlink() or read_json(target) != descriptor:
            raise ValueError("source archive protocol changed")
    else:
        atomic_json(target, descriptor)
    return str(directory)


def planned_jobs(stage="all", device="both", frame_candidates=(17, 33, 49)):
    jobs = [{"stage": "correctness", "device": d} for d in ("a100", "v100")]
    jobs += [{"stage": "probe", "device": "a100", "frame_probe": f} for f in frame_candidates]
    jobs += [{"stage": s, "device": "v100" if s == "confirm" else "a100"}
             for s in ("calibration", "mechanism", "online", "confirm")]
    return [j for j in jobs if (stage == "all" or j["stage"] == stage) and (device == "both" or j["device"] == device)]


def job_key(job):
    return (job["stage"], job["device"], job.get("frame_probe"))


def completed_job(manifest, job, identity):
    return any(job_key(r) == job_key(job) and r.get("status") == "completed"
               and r.get("source_identity") == identity
               and (job["stage"] != "correctness" or correctness_passed(r, identity))
               for r in manifest["records"])


def safe_probe_available(manifest, identity):
    for attempt in manifest["records"]:
        if (attempt.get("stage") != "probe" or attempt.get("source_identity") != identity
                or attempt.get("charge_closed") is False):
            continue
        rows = attempt_records(attempt)
        if environment_matches(rows, identity) and any(row.get("type") == "shape_probe"
                and row.get("status") == "completed" and row.get("safe_to_run") is True for row in rows):
            return True
    return False


def gate_passed(root, config, identity):
    path = Path(root) / "gate.json"
    try:
        if path.is_symlink() or not path.is_file():
            return False, "missing_or_symlinked_gate"
        value = read_json(path)
        okay = (value.get("schema_version") == "previs2-gate.v1" and value.get("status") == "passed"
                and value.get("source_identity_sha256") == identity["sha256"]
                and value.get("protocol_sha256") == fingerprint(config)
                and value.get("natural_resource_gate") is True and value.get("mechanism_gate") is True
                and isinstance(value.get("reasons"), list)
                and all(isinstance(reason, str) for reason in value["reasons"]))
        return okay, "passed" if okay else "resource_or_mechanism_gate_not_passed_for_current_source"
    except (OSError, ValueError, AttributeError):
        return False, "invalid_gate"


def gpu_state(uuid):
    rows = subprocess.check_output(["nvidia-smi", "--id=" + uuid,
        "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=10).strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("GPU telemetry did not select exactly one device")
    row = [item.strip() for item in rows[0].split(",")]
    if len(row) != 5 or row[0].lower() != uuid.lower():
        raise RuntimeError("GPU identity mismatch")
    result = dict(uuid=row[0], name=row[1], memory_total_mib=float(row[2]),
                  memory_used_mib=float(row[3]), utilization_percent=float(row[4]))
    if not all(finite_number(result[key]) for key in ("memory_total_mib", "memory_used_mib", "utilization_percent")):
        raise RuntimeError("unknown GPU telemetry")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                                       text=True, timeout=10).splitlines()
    result["compute_pids"] = []
    for line in processes:
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            raise RuntimeError("unknown GPU process telemetry")
        if fields[0].lower() == uuid.lower():
            result["compute_pids"].append(int(fields[1]))
    return result


def gpu_idle(uuid):
    state = gpu_state(uuid)
    if state["compute_pids"] or state["memory_used_mib"] > 512 or state["utilization_percent"] > 10:
        raise RuntimeError("target GPU is occupied; no process was signalled: " + repr(state))
    return state


def wait_gpu_release(uuid, timeout):
    """Observe cleanup without signalling any process that starts elsewhere."""
    deadline = time.monotonic() + max(0, timeout)
    while True:
        state = gpu_state(uuid)
        if not state["compute_pids"] and state["memory_used_mib"] <= 512:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU cleanup unconfirmed; stop instead of killing unrelated jobs")
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def stop_owned(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=20)


def validate_config(config):
    if config.get("schema_version") != "ditserve.previs2.protocol.v1":
        raise ValueError("not a run02 protocol")
    if config.get("stage_deadline_hours") != DEADLINE_HOURS or config.get("frame_candidates") != [17, 33, 49]:
        raise ValueError("run02 cumulative timing and independent probe order are frozen")
    if set(config.get("devices", {})) != {"a100", "v100"} or len(set(config["devices"].values())) != 2:
        raise ValueError("two distinct explicit GPU UUIDs required")
    if any(not isinstance(uuid, str) or not uuid.startswith("GPU-") for uuid in config["devices"].values()):
        raise ValueError("explicit GPU UUIDs required")
    prompts = config.get("prompts", [])
    if len(prompts) != 8 or len({p["id"] for p in prompts}) != 8 or len({p["text"].strip() for p in prompts}) != 8:
        raise ValueError("eight unique predeclared prompts required")
    if any(sum(p.get("split") == split for p in prompts) != 4 for split in ("dev", "test")):
        raise ValueError("four development and four test prompts required")
    old = read_json(CONFIGS / "previs.json")
    if {p["text"].strip() for p in prompts}.intersection(p["text"].strip() for p in old["prompts"]):
        raise ValueError("run02 must not reuse run01 prompt text")


def execute_attempt(manifest, path, job, python, identity, cap, gpu_identity):
    root, output = Path(job["suite_root"]), Path(job["output"])
    record = {key: job[key] for key in ("stage", "device", "uuid", "output", "source_identity")}
    if "frame_probe" in job:
        record["frame_probe"] = job["frame_probe"]
    if "online_gate" in job:
        record["online_gate"] = job["online_gate"]
    record.update(status="running", started_at=utc_now(), started_unix=time.time(),
        started_monotonic_s=time.monotonic(), budget_cap_seconds=cap, allowed_seconds=job["allowed_seconds"],
        identity=gpu_identity, charge_closed=False,
        charge_scope="one_device_process_lifetime_including_load_failures_and_cleanup_wait")
    manifest["records"].append(record)
    manifest["status"] = "running"
    atomic_json(path, manifest)
    started = record["started_monotonic_s"]
    child, interrupted, cleanup_error = None, None, None
    command = [str(Path(python).absolute()), "-m", "ditserve.experiments.previs2_experiment", "--job", str(output / "job.json")]
    env = child_environment()
    env.update(CUDA_VISIBLE_DEVICES=job["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1",
        HF_HOME=str(ROOT / ".cache/huggingface"), TORCH_HOME=str(ROOT / ".cache/torch"),
        OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    try:
        with (output / "worker.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = child.pid
            atomic_json(path, manifest)
            code = child.wait(timeout=max(1, cap - CLEANUP_RESERVE - (time.monotonic() - started)))
            record["returncode"] = code
            outcome = read_json(output / "outcome.json") if (output / "outcome.json").is_file() else {}
            record["outcome"] = outcome
            record["status"] = ("completed" if code == 0 and outcome.get("status") == "completed" else
                "partial" if code in (0, 2) and outcome.get("status") in ("partial", "budget_exhausted") else "failed")
    except subprocess.TimeoutExpired:
        record["status"] = "budget_exhausted"
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=type(exc).__name__ + ": " + str(exc))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupted = exc
    finally:
        try:
            if child is not None:
                stop_owned(child)
                record["cleanup_gpu_state"] = wait_gpu_release(job["uuid"], max(0, cap - (time.monotonic() - started) - 5))
        except BaseException as exc:
            cleanup_error = exc
            record.update(status="cleanup_unconfirmed", cleanup_error=type(exc).__name__ + ": " + str(exc))
        record.update(elapsed_seconds=time.monotonic() - started, finished_at=utc_now(),
                      finished_monotonic_s=time.monotonic(), charge_closed=cleanup_error is None)
        manifest["used_seconds"] = math.fsum(r.get("elapsed_seconds", 0) for r in manifest["records"])
        manifest["status"] = "partial"
        if manifest["used_seconds"] > manifest["limit_seconds"] + TOLERANCE:
            manifest["status"] = "budget_exceeded"
            manifest["budget_overrun_seconds"] = manifest["used_seconds"] - manifest["limit_seconds"]
        atomic_json(path, manifest)
    if cleanup_error is not None:
        raise RuntimeError("owned job cleanup unconfirmed; retained open charge") from cleanup_error
    if interrupted is not None:
        raise interrupted
    rows = attempt_records(record)
    envs = [row for row in rows if row.get("type") == "environment"]
    env_ok = (environment_matches(rows, identity) and envs[0].get("uuid") == job["uuid"]
              and envs[0].get("protocol") == job["config"])
    if record["status"] == "completed" and not env_ok:
        record.update(status="failed", reason="missing_or_mismatched_source_runtime_environment")
    if record["status"] == "completed" and job["stage"] == "correctness" and not correctness_passed(record, identity):
        record.update(status="failed", reason="all_nine_correctness_cases_required")
    atomic_json(path, manifest)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS / "previs2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/previs/run02")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--gpu", choices=("both", "a100", "v100"), default="both")
    parser.add_argument("--gpu-hours", type=float, default=12)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv-previs/bin/python")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--job-seconds", type=float, help="optional stricter cap including cleanup reserve")
    args = parser.parse_args(argv)
    if not finite_number(args.gpu_hours) or not 0 < args.gpu_hours <= 12:
        parser.error("gpu-hours must be finite and in (0,12]")
    if args.job_seconds is not None and (not finite_number(args.job_seconds) or args.job_seconds <= CLEANUP_RESERVE):
        parser.error("job-seconds must exceed the 60 second parent cleanup reserve")
    config = read_json(args.config)
    validate_config(config)
    jobs = planned_jobs(args.stage, args.gpu, config["frame_candidates"])
    if not jobs:
        parser.error("selected stage has no supported job on this device")
    root = args.output.resolve()
    old = (ROOT / "results/previs/run01").resolve()
    if root == old or old in root.parents or root in old.parents:
        parser.error("run02 output must be disjoint from protected run01")
    if args.dry_run:
        print(json.dumps(dict(jobs=jobs, config=config, output=str(root), gpu_seconds_limit=args.gpu_hours * 3600,
            cleanup_reserve_per_attempt_seconds=CLEANUP_RESERVE, gpu_access=False, writes=False), indent=2, ensure_ascii=False))
        return 0
    identity = source_identity(args.python)
    protection = protected_run01()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "suite.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "manifest.json"
        if path.is_symlink():
            raise ValueError("manifest must be a local ordinary file")
        manifest = read_json(path) if path.exists() else dict(schema_version=SCHEMA, created_at=utc_now(),
            protocol=config, protocol_sha256=fingerprint(config), records=[], used_seconds=0.0,
            limit_seconds=args.gpu_hours * 3600, current_source_identity=identity, run01_protection=protection,
            budget_scope="run02_only_serial_sum_of_actual_device_process_lifetimes")
        if (manifest.get("schema_version") != SCHEMA or manifest.get("protocol") != config
                or manifest.get("protocol_sha256") != fingerprint(config)
                or manifest.get("current_source_identity") != identity):
            raise ValueError("protocol/source changed; this suite cannot mix versions or reset its budget")
        check_protection(manifest["run01_protection"])
        used = validate_ledger(manifest)
        if args.gpu_hours * 3600 + TOLERANCE < used:
            raise ValueError("requested smaller budget is already spent")
        manifest["limit_seconds"] = min(manifest["limit_seconds"], args.gpu_hours * 3600)
        manifest["source_archive"] = archive_sources(root, identity, config)
        manifest["status"] = "partial"
        atomic_json(path, manifest)
        interrupted, aborted = False, False
        previous_term = signal.getsignal(signal.SIGTERM)
        def on_term(signum, frame):
            raise KeyboardInterrupt("supervisor received SIGTERM; terminate and charge only owned child")
        signal.signal(signal.SIGTERM, on_term)
        try:
            for item in jobs:
                check_protection(manifest["run01_protection"])
                if source_identity(args.python) != identity:
                    raise ValueError("frozen source/runtime changed; no further GPU work")
                used = validate_ledger(manifest)
                if completed_job(manifest, item, identity):
                    continue
                previous = [r for r in manifest["records"] if job_key(r) == job_key(item) and r.get("status") != "not_run"]
                reason = None
                if previous and not args.retry_failed:
                    reason = "prior_incomplete_attempt_preserved_explicit_retry_required"
                if item["stage"] != "correctness" and not all(any(
                        r.get("device") == device and correctness_passed(r, identity)
                        for r in manifest["records"] if r.get("stage") == "correctness") for device in ("a100", "v100")):
                    reason = "both_current_source_nine_case_correctness_gates_required"
                if item["stage"] == "calibration" and not safe_probe_available(manifest, identity):
                    reason = "no_current_source_safe_shape_probe"
                if item["stage"] == "online":
                    okay, why = gate_passed(root, config, identity)
                    if not okay:
                        reason = why
                cap = min(manifest["limit_seconds"], DEADLINE_HOURS[item["stage"]] * 3600) - used
                cap = min(cap, args.job_seconds or float("inf"))
                if item["stage"] == "correctness" and item["device"] == "a100" and not completed_job(
                        manifest, {"stage": "correctness", "device": "v100"}, identity):
                    cap = min(cap, DEADLINE_HOURS["correctness"] * 1800)
                if cap < CLEANUP_RESERVE + MIN_WORK_SECONDS:
                    reason = "global_or_cumulative_stage_budget_exhausted"
                if reason:
                    manifest["records"].append(dict(item, status="not_run", reason=reason, elapsed_seconds=0.0,
                        charge_closed=True, source_identity=identity, finished_at=utc_now()))
                    atomic_json(path, manifest)
                    print("Not run %s: %s" % (job_key(item), reason), flush=True)
                    continue
                uuid = config["devices"][item["device"]]
                cache = ROOT / ".cache"
                cache.mkdir(exist_ok=True)
                with (cache / ("previs-" + uuid + ".lock")).open("a+") as gpu_lock:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    gpu_identity = gpu_idle(uuid)
                    suffix = "-f%d" % item["frame_probe"] if "frame_probe" in item else ""
                    output = root / ("%02d-%s-%s%s" % (len(manifest["records"]), item["stage"], item["device"], suffix))
                    output.mkdir()
                    job = dict(item, uuid=uuid, config=config, output=str(output), suite_root=str(root),
                        allowed_seconds=max(1, cap - CLEANUP_RESERVE - 5), created_at=utc_now(),
                        source_identity=identity, protocol_sha256=fingerprint(config))
                    if item["stage"] == "online":
                        job["online_gate"] = {"path": str(root / "gate.json"), "sha256": file_sha(root / "gate.json")}
                    atomic_json(output / "job.json", job)
                    print("Starting %s: allowance %.1fs; used %.1f/%.1fs" % (job_key(item), job["allowed_seconds"], used, manifest["limit_seconds"]), flush=True)
                    record = execute_attempt(manifest, path, job, args.python, identity, cap, gpu_identity)
                    check_protection(manifest["run01_protection"])
                    if (source_identity(args.python) != identity or
                            (item["stage"] == "online" and file_sha(root / "gate.json") != job["online_gate"]["sha256"])):
                        record.update(status="failed", reason="frozen_source_runtime_or_gate_changed_during_measurement")
                        atomic_json(path, manifest)
                        raise ValueError("source/runtime/gate changed during measurement; preserve attempt and stop")
                    print("Finished %s: %s, charged %.3fs" % (job_key(item), record["status"], record["elapsed_seconds"]), flush=True)
        except KeyboardInterrupt:
            interrupted = True
            aborted = True
        except Exception as exc:
            aborted = True
            manifest["supervisor_error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            manifest["used_seconds"] = math.fsum(r.get("elapsed_seconds", 0) for r in manifest["records"])
            all_jobs = planned_jobs(frame_candidates=config["frame_candidates"])
            complete = all(completed_job(manifest, item, identity) for item in all_jobs)
            valid_charge = not any(r.get("charge_closed") is False for r in manifest["records"])
            within_limit = manifest["used_seconds"] <= manifest["limit_seconds"] + TOLERANCE
            manifest["status"] = "completed" if complete and valid_charge and within_limit and not aborted else "partial"
            if not within_limit:
                manifest["status"] = "budget_exceeded"
            manifest["selection_status"] = "completed" if all(completed_job(manifest, item, identity) for item in jobs) else "partial"
            manifest["updated_at"] = utc_now()
            atomic_json(path, manifest)
        if interrupted:
            return 130
        return 0 if manifest["selection_status"] == "completed" and within_limit and valid_charge else 2


if __name__ == "__main__":
    raise SystemExit(main())
