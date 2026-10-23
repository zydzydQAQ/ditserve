"""Sequential, resumable launch ledger for the explicitly authorized GPU study."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid


class MonitoringError(RuntimeError):
    """The job's exclusive GPU ownership could not be verified."""


def _option(command, name, default=None):
    for index, value in enumerate(command):
        if value == name:
            if index + 1 == len(command):
                raise ValueError("missing command argument: " + name)
            return command[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def selected_devices(job):
    """Physical NVML devices used by the suite's unmasked CUDA launch wrapper."""
    command = job["command"]
    if job.get("command_type") == "remaining_gpu":
        from ditserve.evaluation import workflow_remaining_study as remaining
        from ditserve.evaluation.workflow_serving_study import digest
        mapping = job.get("gpu_mapping", [])
        if (not job.get("metadata", {}).get("remaining_validation") or job.get("cpu_only")
                or not mapping or any(type(row.get("ordinal")) is not int or row["ordinal"] not in (0, 1)
                    or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", str(row.get("uuid"))) for row in mapping)
                or len({r["ordinal"] for r in mapping}) != len(mapping)
                or len({r["uuid"] for r in mapping}) != len(mapping)
                or job.get("devices") != [r["ordinal"] for r in mapping]):
            raise ValueError("remaining GPU job requires an exact physical GPU mapping")
        config = json.loads(Path(job["input"]).read_text())
        role = job["metadata"]["role"]
        root = Path(job["input"]).parent.parent
        if (digest(config) != job.get("configuration_sha256") or
                command != remaining.expected_command(dict(root=str(root), run_id=root.name), job["id"], config, role=role)):
            raise ValueError("remaining command/config differs from its frozen entry point")
        if role in ("remaining_vae_performance", "remaining_vae_trace"):
            if len(mapping) != 1 or config.get("device") != "cuda:0" or config.get("gpu_uuid") != mapping[0]["uuid"]:
                raise ValueError("remaining VAE task must bind one physical UUID as cuda:0")
        elif role == "remaining_vae_lifecycle_bundle":
            from ditserve.evaluation.workflow_overlap_bundle import normalize_config
            config = normalize_config(config)
            if len(mapping) != 2 or any(c["gpu_uuid"] != mapping[c["device_ordinal"]]["uuid"] for c in config["cases"]):
                raise ValueError("remaining lifecycle UUID mapping differs")
        elif role == "remaining_sp_comparison":
            if len(mapping) != 2 or config.get("gpu_uuids") != [r["uuid"] for r in mapping]:
                raise ValueError("remaining SP UUID mapping differs")
        elif role == "remaining_interleave":
            if job["devices"] != [0, 1]:
                raise ValueError("repaired interleave requires both physical GPUs")
        else:
            raise ValueError("unregistered remaining GPU entry point")
        return job["devices"]
    if job.get("command_type") == "isolated_gpu":
        mapping = job.get("gpu_mapping", [])
        if (not isinstance(mapping, list) or len(mapping) != 1 or not isinstance(mapping[0], dict)
                or type(mapping[0].get("ordinal")) is not int or mapping[0]["ordinal"] < 0
                or not isinstance(mapping[0].get("uuid"), str)
                or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", mapping[0]["uuid"])
                or job.get("devices") != [mapping[0]["ordinal"]] or job.get("cpu_only")):
            raise ValueError("isolated GPU qualification requires one exact physical GPU mapping")
        from ditserve.evaluation.workflow_serving_study import digest
        config = json.loads(Path(job["input"]).read_text())
        if (digest(config) != job.get("configuration_sha256") or config.get("gpu_uuid") != mapping[0]["uuid"]
                or config.get("device") != "cuda:0" or config.get("kind") not in ("sana", "wan")):
            raise ValueError("isolated GPU configuration differs from its frozen mapping")
        prefix = ["/usr/bin/env", "CUDA_VISIBLE_DEVICES=" + mapping[0]["uuid"], "/opt/venv/bin/python",
                  "-m", "ditserve.evaluation.workflow_encoding_qualification"]
        if (command[:5] != prefix or len(command) != 11 or command[5:8] != ["--kind", config["kind"], "--config"]
                or command[9] != "--out" or not Path(command[8]).is_absolute() or not Path(command[10]).is_absolute()):
            raise ValueError("masked CUDA command must be the fixed encoding qualification entry point")
        return [mapping[0]["ordinal"]]
    if job.get("command_type") == "native":
        mapping = job.get("gpu_mapping", [])
        if (not mapping or any(type(row.get("ordinal")) is not int or row["ordinal"] < 0
                               or not str(row.get("uuid", "")).startswith("GPU-") for row in mapping)
                or len({row["ordinal"] for row in mapping}) != len(mapping)
                or len({row["uuid"] for row in mapping}) != len(mapping)):
            raise ValueError("native jobs require a unique physical GPU mapping")
        env = json.loads(Path(job["environment_input"]).read_text())
        if (env.get("gpus", {}).get("uuids") != [row["uuid"] for row in mapping]
                or env.get("gpus", {}).get("count") != len(mapping)):
            raise ValueError("native environment GPU UUIDs differ from the frozen mapping")
        if sorted(job.get("devices", [])) != sorted(row["ordinal"] for row in mapping):
            raise ValueError("native declared devices differ from the frozen mapping")
        return sorted(row["ordinal"] for row in mapping)
    if job.get("cpu_only"):
        if job.get("devices") or _option(command, "--kind") != "cpu_micro":
            raise ValueError("cpu_only is reserved for the standalone CPU microbenchmark")
        return []
    def ordinal(value):
        value = str(value).removeprefix("cuda:")
        if not value.isdigit():
            raise ValueError("expected a numeric physical GPU identifier: " + value)
        return int(value)
    main = ordinal(_option(command, "--device", "cuda:1"))
    experiment = _option(command, "--experiment")
    devices = {main}
    if experiment in ("E2", "E6"):
        group = {ordinal(value) for value in _option(command, "--devices", "0,1").split(",")}
        devices = group | ({main} if experiment == "E2" else set())
    encoder = any("workflow_encoder" in value for value in command)
    child = _option(command, "--child")
    phases = [child] if child is not None else _option(command, "--phases", "T0,T1,T2,T3,T4,T5").split(",")
    if encoder and "T5" in phases:
        devices.add(ordinal(_option(command, "--encoder-device", "cuda:0")))
    if "devices" in job:
        declared = {ordinal(value) for value in job["devices"]}
        if not devices <= declared:
            raise ValueError("job.devices omits GPUs selected by its command")
        devices = declared
    if any(value.startswith("CUDA_VISIBLE_DEVICES=") for value in command):
        raise ValueError("masked CUDA commands need an explicit supported physical-device mapping")
    return sorted(devices)


def process_identity(pid, proc_root=Path("/proc")):
    """PID plus kernel start ticks prevent ownership from surviving PID reuse."""
    root = proc_root / str(pid)
    try:
        text = (root / "stat").read_text()
        fields = text.rsplit(")", 1)[1].split()
        groups = []
        for line in (root / "cgroup").read_text().splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            groups.append((hierarchy, controllers, path))
        return dict(pid=int(pid), ppid=int(fields[1]), start_ticks=int(fields[19]),
                    state=fields[0], name=text.split("(", 1)[1].rsplit(")", 1)[0],
                    cgroups=groups)
    except (FileNotFoundError, ProcessLookupError):
        # NVML may retain a context record briefly after the process has exited.
        return None
    except (OSError, ValueError, IndexError) as error:
        raise MonitoringError("cannot inspect GPU PID %s: %s" % (pid, error)) from error


class ContainerOwnership:
    """Recognize host-PID descendants and same-container orphaned workers."""
    def __init__(self, lookup=process_identity):
        self.lookup = lookup
        self.container_id = None
        self.root = None
        self.known = set()
        self.cgroups = set()

    @staticmethod
    def identity(process):
        return process["pid"], process["start_ticks"]

    def update(self, container):
        if self.container_id is not None and container["id"] != self.container_id:
            raise MonitoringError("container identity changed during one launch")
        self.container_id = container["id"]
        if not container["running"] or container["pid"] == 0:
            return False
        process = self.lookup(container["pid"])
        if process is None:
            return False  # Entry process may exit between Docker inspect and /proc.
        identity = self.identity(process)
        if self.root is not None and identity != self.root:
            raise MonitoringError("container root PID identity changed")
        self.root = identity
        self.known.add(identity)
        # Only a container-specific cgroup can authorize a reparented worker.
        # Never bless an entire host hierarchy such as '/' or '/system.slice'.
        self.cgroups = {tuple(value) for value in process.get("cgroups", [])
                        if self.container_id[:12] in value[2]}
        return True

    def owns(self, process):
        identity = self.identity(process)
        if identity in self.known:
            return True
        if self.cgroups.intersection(tuple(value) for value in process.get("cgroups", [])):
            self.known.add(identity)
            return True
        path, seen = [], set()
        current = process
        for _ in range(128):
            if current is None or current["pid"] in seen:
                break
            current_identity = self.identity(current)
            path.append(current_identity)
            if current_identity in self.known:
                self.known.update(path)
                return True
            seen.add(current["pid"])
            if current["ppid"] <= 0:
                break
            current = self.lookup(current["ppid"])
        return False


def foreign_processes(samples, ownership=None, lookup=process_identity):
    conflicts = {}
    for sample in samples:
        for kind in ("compute", "graphics"):
            processes = sample.get(kind + "_processes")
            if processes is None:
                raise MonitoringError("NVML %s process query unavailable for GPU %s" % (kind, sample.get("device")))
            for row in processes:
                process = lookup(row["pid"])
                if process is None or process["state"] in ("Z", "X", "x"):
                    continue
                if ownership is not None and ownership.owns(process):
                    continue
                key = (sample["device"], process["pid"], process["start_ticks"])
                conflict = conflicts.setdefault(key, dict(device=sample["device"], uuid=sample.get("uuid"),
                    pid=process["pid"], start_ticks=process["start_ticks"], name=process["name"], kinds=[]))
                conflict["kinds"].append(kind)
    return list(conflicts.values())


class NvmlProbe:
    def __init__(self, repo, devices):
        sys.path.insert(0, str(repo / "workers"))
        from ditserve.evaluation.workflow_metrics import NvmlSampler
        # Main loop owns the 2-second polling. The sampler thread stays idle;
        # this reuses its tested ctypes process-list handling without a GPU context.
        self.sampler = NvmlSampler(devices=devices, interval_s=3600)
        self.devices = set(devices)

    def __enter__(self):
        self.sampler.start()
        return self

    def sample(self):
        try:
            rows = self.sampler.sample_once()
        except Exception as error:
            raise MonitoringError("NVML process inspection raised: " + str(error)) from error
        if {row["device"] for row in rows} != self.devices:
            raise MonitoringError("NVML failed to resolve every selected GPU: " + repr(self.sampler.errors))
        for row in rows:
            if row.get("compute_processes") is None or row.get("graphics_processes") is None:
                raise MonitoringError("NVML process inspection failed: " + repr(self.sampler.errors))
        return rows

    def __exit__(self, *_):
        self.sampler.stop()


def docker_command(arguments, **kwargs):
    docker = ["docker", "--host", "unix:///run/ditserve-docker/docker.sock", *arguments]
    return subprocess.run(["sg", "ditserve-docker", "-c", shlex.join(docker)], **kwargs)


def inspect_container(name):
    try:
        result = docker_command(["inspect", "--format", "{{.Id}} {{.State.Pid}} {{.State.Running}}", name],
                                capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MonitoringError("Docker inspect failed: " + str(error)) from error
    if result.returncode:
        if "no such object" in result.stderr.lower() or "no such container" in result.stderr.lower():
            return None
        raise MonitoringError("Docker inspect failed: " + result.stderr.strip())
    try:
        identifier, pid, running = result.stdout.strip().split()
        if len(identifier) != 64 or any(value not in "0123456789abcdef" for value in identifier) or running not in ("true", "false"):
            raise ValueError("invalid Docker identity")
        return dict(id=identifier, pid=int(pid), running=running == "true")
    except (ValueError, TypeError) as error:
        raise MonitoringError("unexpected Docker inspect response") from error


def stop_own_container(name, ownership, process, stdout, stderr):
    # A fresh random suffix makes the fallback name unique to this exact Popen.
    # Prefer the captured immutable Docker ID once the daemon has exposed it.
    target = ownership.container_id or name
    result = dict(target=target, target_kind="container_id" if ownership.container_id else "unique_launch_name")
    try:
        stopped = docker_command(["stop", "--time", "10", target], stdout=stdout, stderr=stderr,
                                 check=False, timeout=25)
        result["docker_returncode"] = stopped.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    result["verified_stopped"] = False
    try:
        state = inspect_container(target)
        if state is not None and state["running"]:
            killed = docker_command(["kill", target], stdout=stdout, stderr=stderr, check=False, timeout=15)
            result["kill_returncode"] = killed.returncode
            state = inspect_container(target)
        result["verified_stopped"] = state is None or not state["running"]
    except (MonitoringError, OSError, subprocess.TimeoutExpired) as error:
        result["verification_error"] = str(error)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    return result


def monitor_process(process, probe, name, record, *, inspect=inspect_container,
                    lookup=process_identity, sleep=time.sleep, monotonic=time.monotonic):
    """Poll selected GPUs; never sends signals to another user's process."""
    ownership = ContainerOwnership(lookup)
    record["gpu_monitor"]["checks"] = 0
    record["gpu_monitor"]["samples"] = []
    started = monotonic()
    while True:
        returncode = process.poll()
        if returncode is not None:
            record["returncode"] = returncode
            record["status"] = "completed" if returncode == 0 else "failed"
            return ownership
        elapsed = monotonic() - started
        budget_stop = record.get("budget_stop_at_monotonic")
        if budget_stop is not None and monotonic() >= budget_stop:
            record["status"] = ("timeout" if record.get("deadline_reason") == "task_timeout" else "budget_exhausted")
            return ownership
        if elapsed >= record["timeout_seconds"]:
            record["status"] = "timeout"
            return ownership
        try:
            container = inspect(name)
            if container is not None:
                ownership.update(container)
                record["gpu_monitor"]["container_id"] = ownership.container_id
                record["gpu_monitor"]["root_identity"] = ownership.root
            samples = probe.sample()
            record["gpu_monitor"]["checks"] += 1
            record["gpu_monitor"]["samples"].extend(samples)
            for sample in samples:
                total, used = sample.get("memory_total_bytes"), sample.get("memory_used_bytes")
                if total is not None and used is not None:
                    item = record["gpu_monitor"].setdefault("memory", {}).setdefault(str(sample["device"]),
                        dict(samples=0, total_bytes=total, peak_used_bytes=0, minimum_free_bytes=total))
                    item["samples"] += 1
                    item["peak_used_bytes"] = max(item["peak_used_bytes"], used)
                    item["minimum_free_bytes"] = min(item["minimum_free_bytes"], total - used)
            if ownership.root is None:
                # Docker run has a short create/start window. If a GPU process
                # appears in it, re-inspect before deciding it is foreign.
                live = foreign_processes(samples, lookup=lookup)
                if live:
                    container = inspect(name)
                    if container is not None:
                        ownership.update(container)
                    if ownership.root is None:
                        if process.poll() is not None:
                            continue  # Container auto-removal raced job completion.
                        record["gpu_monitor"].setdefault("unresolved_gpu_processes", []).append(
                            dict(unix_ns=time.time_ns(), processes=live))
                        raise MonitoringError("GPU work appeared before container ownership could be established")
                elif elapsed > 30:
                    if process.poll() is not None:
                        continue
                    raise MonitoringError("container did not expose a live host PID within 30 seconds")
            conflicts = foreign_processes(samples, ownership, lookup=lookup)
            if conflicts:
                record["gpu_monitor"]["conflicts"].append(dict(unix_ns=time.time_ns(), processes=conflicts))
                record["status"] = "interrupted_foreign_gpu_workload"
                return ownership
        except MonitoringError as error:
            # A missing inspect target alone is expected on clean auto-removal;
            # other inspection failures invalidate the monitored interval.
            record["status"] = "gpu_monitor_unavailable"
            record["gpu_monitor"]["errors"].append(str(error))
            return ownership
        sleep(min(2., max(0., record["timeout_seconds"] - (monotonic() - started))))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_frozen_job(specification, job, root):
    """Validate actual bytes, not just a digest string copied into a job JSON."""
    if job.get("command_type") == "isolated_gpu":
        if not specification.get("strict_resume") or root.parent.name != "workflow":
            raise ValueError("isolated GPU qualification requires a frozen DATA_ROOT/workflow study")
        # The wrapper mounts DATA_ROOT at /data. Bind both command paths to
        # the same input/output whose host bytes the strict ledger validates.
        expected = ["/data/" + str(Path(job[key]).relative_to(root.parent.parent)) for key in ("input", "output")]
        if (_option(job["command"], "--config"), _option(job["command"], "--out")) != tuple(expected):
            raise ValueError("isolated GPU command paths differ from the frozen input/output")
    if not specification.get("strict_resume"):
        return
    from ditserve.evaluation.workflow_serving_study import digest
    value = json.loads(Path(job["input"]).read_text())
    if digest(value) != job["configuration_sha256"]:
        raise ValueError("frozen input contents changed: " + job["id"])
    study = json.loads((root / "study.json").read_text())
    revisions = study.get("source_revisions")
    if revisions is not None:
        matches = [row for row in revisions if row["source_digest"] == specification["source_digest"]
                   and row["source"] == specification["source"]]
        if len(matches) != 1:
            raise ValueError("job does not identify one frozen measurement revision")
        source_files = matches[0]["source_files"]
    else:
        source_files = study["source_files"]
    if digest(source_files) != specification["source_digest"]:
        raise ValueError("frozen source manifest changed")
    source = Path(specification["source"])
    expected = {row["path"] for row in source_files}
    for row in source_files:
        path = source / row["path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("frozen source contents changed: " + str(path))
    extra = {str(path.relative_to(source)) for path in source.rglob("*")
             if path.is_file() and path.suffix in {".py", ".sh", ".json"} and "__pycache__" not in path.parts} - expected
    if extra:
        raise ValueError("unexpected executable/configuration files in frozen source")
    for row in job.get("input_bindings", []):
        path = Path(row["path"])
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError("bound experiment input changed: " + str(path))
    for row in study.get("model_bindings", []):
        path = Path(row["path"])
        stat = path.stat()
        if (stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]
                or ("sha256" in row and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"])):
            raise ValueError("bound model file changed: " + str(path))


def job_timeout(job):
    value = job.get("timeout_seconds", 1800)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("every job requires a positive finite timeout_seconds")
    return float(value)


def start_budget_attempt(budget, identifier, job, record):
    """Retain a finite watchdog deadline even with an unlimited total ledger."""
    ceiling = job.get("budget_ceiling_seconds")
    if ceiling is not None and (type(ceiling) not in (int, float) or not math.isfinite(ceiling) or ceiling <= 90):
        raise ValueError("task budget ceiling must include cleanup reserve")
    lease = budget.start(identifier, minimum_seconds=job.get("minimum_budget_seconds", 120),
                         timeout_seconds=job_timeout(job))
    deadline, wall_deadline = lease["stop_at_monotonic"], lease["stop_at_unix"]
    reason = lease["deadline_reason"]
    if ceiling is not None and lease["started_monotonic"] + ceiling - 90 < deadline:
        deadline = lease["started_monotonic"] + ceiling - 90
        wall_deadline = lease["started_unix"] + ceiling - 90
        reason = "budget_deadline"
    record.update(budget_stop_at_unix=wall_deadline, budget_stop_at_monotonic=deadline,
                  deadline_reason=reason, budget_attempt_id=lease["id"])
    return lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--only", help="Comma separated job identifiers")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args()
    specification = json.loads(args.jobs.read_text())
    root = Path(specification["root"])
    root.mkdir(parents=True, exist_ok=True)
    jobs = specification["jobs"]
    selected = set(args.only.split(",")) if args.only else None
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "workers"))
    from ditserve.evaluation.workflow_budget import StudyBudget, BudgetExhausted
    budget_limit = args.budget_seconds if args.budget_seconds is not None else specification.get("budget_seconds")
    budget = (StudyBudget(root, budget_limit).__enter__()
              if args.budget_seconds is not None or "budget_seconds" in specification else None)
    if budget is not None:
        import atexit
        atexit.register(budget.__exit__, None, None, None)
    failures = 0
    for job in jobs:
        identifier = job["id"]
        if selected is not None and identifier not in selected:
            continue
        if not identifier.replace("-", "").replace("_", "").isalnum():
            raise ValueError("unsafe job name")
        job_dir = root / "launches" / identifier
        job_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = job_dir / "launch.json"
        validate_frozen_job(specification, job, root)
        fingerprint = hashlib.sha256(json.dumps(dict(job=job,
            source_digest=specification.get("source_digest"), image=specification.get("image")),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if ledger_path.exists():
            previous = json.loads(ledger_path.read_text())
            if previous.get("status") == "completed":
                if specification.get("strict_resume") and previous.get("job_fingerprint") != fingerprint:
                    raise RuntimeError("completed job source/configuration changed: " + identifier)
                if specification.get("strict_resume"):
                    output = Path(job["output"]) / "run.json"
                    if not output.is_file() or hashlib.sha256(output.read_bytes()).hexdigest() != previous.get("output_sha256"):
                        raise RuntimeError("completed job output missing or changed: " + identifier)
                print(json.dumps({"id": identifier, "status": "already_completed"}), flush=True)
                continue
            if specification.get("strict_resume"):
                raise RuntimeError("strict study retries require a new attempt ID and output directory: " + identifier)
            if not args.retry_failed:
                raise RuntimeError("existing incomplete job requires --retry-failed: " + identifier)
            # Keep every failed attempt's evidence.
            index = 1
            while (job_dir / ("attempt-%d.json" % index)).exists():
                index += 1
            ledger_path.rename(job_dir / ("attempt-%d.json" % index))
        name = "ditserve-workflow-" + identifier.lower() + "-" + uuid.uuid4().hex[:12]
        devices = selected_devices(job)
        command = [str(repo / "docker/run-workflow-profile.sh")]
        if job.get("counters"):
            command.append("--counters")
        command.extend(job["command"])
        source = job.get("source", specification.get("source"))
        environment = dict(os.environ, DITSERVE_WORKFLOW_CONTAINER_NAME=name)
        if specification.get("strict_resume"):
            study_manifest = json.loads((root / "study.json").read_text())
            if "model_root" in study_manifest:
                if root.parent.name != "workflow":
                    raise ValueError("study root must be DATA_ROOT/workflow/RUN_ID")
                environment["DITSERVE_DATA_DIR"] = str(root.parent.parent)
                environment["DITSERVE_MODELS_DIR"] = study_manifest["model_root"]
        if specification.get("image"):
            environment["DITSERVE_IMAGE"] = specification["image"]
        if source:
            environment["DITSERVE_WORKFLOW_SOURCE_DIR"] = source
        launcher = ["sg", "ditserve-docker", "-c", shlex.join(["sg", "zyd", "-c", shlex.join(command)])]
        record = dict(id=identifier, status="running", command=command, source=source,
                      container=name, started_unix_ns=time.time_ns(),
                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      timeout_seconds=job_timeout(job),
                      gpu_count=len(devices),
                      gpu_monitor=dict(devices=devices, interval_seconds=2, conflicts=[], errors=[],
                                       admission="pending", checks=0,
                                       policy="Reject foreign compute/graphics; stop only this unique experiment container"),
                      specification_sha256=hashlib.sha256(args.jobs.read_bytes()).hexdigest(),
                      job_fingerprint=fingerprint)
        record["observer_cpu_affinity"] = sorted(os.sched_getaffinity(0))
        if ((job.get("metadata", {}).get("role") in ("dispatch_refresh_online", "occupancy_online", "cache_combination_online")
                or job.get("metadata", {}).get("remaining_validation"))
                and record["observer_cpu_affinity"] != job["metadata"].get("monitor_cpu_affinity")):
            raise ValueError("observer CPU affinity differs from frozen controls")
        if "DITSERVE_MODELS_DIR" in environment:
            record["mounts"] = dict(models=environment["DITSERVE_MODELS_DIR"],
                                   data=environment.get("DITSERVE_DATA_DIR"))
        if job.get("command_type") in {"native", "isolated_gpu", "remaining_gpu"}:
            with NvmlProbe(repo, devices) as inventory_probe:
                observed = {row["device"]: row["uuid"] for row in inventory_probe.sample()}
            if observed != {row["ordinal"]: row["uuid"] for row in job["gpu_mapping"]}:
                raise ValueError("physical GPU inventory changed since preparation")
        if budget is not None:
            try:
                start_budget_attempt(budget, identifier, job, record)
            except BudgetExhausted:
                print(json.dumps(dict(id=identifier, status="budget_deferred",
                                     remaining_seconds=budget.remaining_seconds)), flush=True)
                break
        write(ledger_path, record)
        print(json.dumps({"id": identifier, "status": "checking_gpu_admission", "utc": record["utc"]}), flush=True)
        started = time.monotonic()
        with (job_dir / "stdout.log").open("a") as stdout, (job_dir / "stderr.log").open("a") as stderr:
            process = None
            watchdog = None
            watchdog_done = job_dir / "watchdog.done"
            watchdog_event = job_dir / "watchdog.json"
            watchdog_armed = job_dir / "watchdog.armed"
            launcher_record = job_dir / "launcher-identity.json"
            ownership = ContainerOwnership()
            try:
                class EmptyProbe:
                    def __enter__(self): return self
                    def __exit__(self, *_): pass
                    def sample(self): return []
                with (NvmlProbe(repo, devices) if devices else EmptyProbe()) as probe:
                    conflicts = foreign_processes(probe.sample())
                    if conflicts:
                        record["status"] = "blocked_foreign_gpu_workload"
                        record["gpu_monitor"]["admission"] = "rejected"
                        record["gpu_monitor"]["conflicts"].append(dict(unix_ns=time.time_ns(), processes=conflicts))
                    else:
                        record["gpu_monitor"]["admission"] = "passed"
                        if budget is not None:
                            watchdog_done.unlink(missing_ok=True)
                            watchdog_armed.unlink(missing_ok=True)
                            identity = process_identity(os.getpid())
                            watchdog = subprocess.Popen([sys.executable,
                                str(repo / "docker/workflow-budget-watchdog.py"),
                                "--container", name, "--parent-pid", str(os.getpid()),
                                "--parent-start", str(identity["start_ticks"]),
                                "--deadline", str(record["budget_stop_at_unix"]),
                                "--monotonic-deadline", str(record["budget_stop_at_monotonic"]),
                                "--deadline-reason", record["deadline_reason"],
                                "--event", str(watchdog_event), "--done", str(watchdog_done),
                                "--armed", str(watchdog_armed), "--launcher-record", str(launcher_record)],
                                stdout=stdout, stderr=stderr, start_new_session=True)
                            armed_deadline = time.monotonic() + 5
                            while not watchdog_armed.exists():
                                if watchdog.poll() is not None or time.monotonic() > armed_deadline:
                                    raise MonitoringError("budget watchdog did not arm before GPU launch")
                                time.sleep(.05)
                        process = subprocess.Popen(launcher, cwd=repo, env=environment, stdout=stdout, stderr=stderr,
                                                   start_new_session=True)
                        record["launcher_pid"] = process.pid
                        launch_identity = process_identity(process.pid)
                        if launch_identity is not None:
                            write(launcher_record, dict(pid=process.pid, start_ticks=launch_identity["start_ticks"]))
                        write(ledger_path, record)
                        print(json.dumps({"id": identifier, "status": "started", "devices": devices}), flush=True)
                        ownership = monitor_process(process, probe, name, record)
            except KeyboardInterrupt:
                record["status"] = "interrupted"
            except (MonitoringError, OSError) as error:
                record["status"] = "gpu_monitor_unavailable"
                record["gpu_monitor"]["errors"].append(str(error))
            finally:
                clean = True
                if process is not None:
                    try:
                        state = inspect_container(ownership.container_id or name)
                        clean = state is None or not state["running"]
                    except MonitoringError:
                        clean = False
                    if process.poll() is None or not clean:
                        record["shutdown"] = stop_own_container(name, ownership, process, stdout, stderr)
                        record["returncode"] = process.poll()
                        clean = record["shutdown"]["verified_stopped"]
                if watchdog is not None and clean:
                    watchdog_done.touch()
                    try:
                        watchdog.wait(timeout=35)
                    except subprocess.TimeoutExpired:
                        watchdog.terminate()
                        watchdog.wait(timeout=5)
                    if watchdog_event.exists():
                        record["watchdog"] = json.loads(watchdog_event.read_text())
                        record["status"] = {"budget_deadline": "budget_exhausted", "task_timeout": "timeout"}.get(
                            record["watchdog"]["reason"], "interrupted")
                if not clean:
                    record["status"] = "cleanup_failed"
                record["wall_seconds"] = time.monotonic() - started
                record["reserved_gpu_seconds"] = record["wall_seconds"] * record["gpu_count"] if process is not None else 0.
                record["finished_unix_ns"] = time.time_ns()
                if specification.get("strict_resume") and record["status"] == "completed":
                    output = Path(job["output"]) / "run.json"
                    if output.is_file():
                        record["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                    else:
                        record["status"] = "missing_result"
                write(ledger_path, record)
                if budget is not None and clean:
                    budget.finish(record["status"], gpu_count=record["gpu_count"],
                                  reserved_gpu_seconds=record["reserved_gpu_seconds"],
                                  launch_path=str(ledger_path))
        print(json.dumps({"id": identifier, "status": record["status"],
                          "wall_seconds": record["wall_seconds"]}), flush=True)
        if record["status"] != "completed":
            failures += 1
        if record["status"] in {"interrupted", "gpu_monitor_unavailable", "blocked_foreign_gpu_workload",
                                "interrupted_foreign_gpu_workload", "budget_exhausted", "cleanup_failed"}:
            break
    if budget is not None:
        budget.__exit__(None, None, None)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
