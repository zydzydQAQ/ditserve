"""Run the approved Wan hardware pass as uid 1002 under privileged Nsight only.

Run this module normally, never with sudo. Only the fixed NVIDIA profiler is
elevated; --run-as=zyd keeps the native controller and model workers unprivileged.
--check-only reads existing control files and locks, without sudo or GPU queries.
No driver settings, capabilities, existing results, or budget allocations change.
"""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time

from ...runtime import PACKAGE_ROOT, ROOT

NSYS = Path("/opt/nvidia/nsight-systems/2023.4.4/target-linux-x64/nsys")
GPU_UUID = "GPU-3ab2fedc-ccc1-007d-f4a1-b6058b3479a3"
EXPERIMENT = "wan-profile-corrected"
RUN_ID = "run01"
EXPECTED_UID = 1002


class GpuBusy(ValueError):
    """A known foreign workload may finish; invalid/unknown probes must not wait."""


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("symlink or oversized control file: " + str(path))
    with path.open() as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def paths(root=ROOT):
    work = root / ".cache/results-work" / EXPERIMENT / RUN_ID
    return {"root": root, "work": work, "ledger": root / ".state" / EXPERIMENT / "budget.json",
            "sources": work / "sources", "output": work / "nsys", "run": work / "run.json",
            "python": root / ".venv-previs/bin/python"}


@contextmanager
def unlocked_controls(p):
    handles = []
    try:
        for path in (p["work"] / "run.lock", Path(str(p["ledger"]) + ".lock")):
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            handles.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for descriptor in handles:
            os.close(descriptor)


def _frozen_path(base, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe frozen source path")
    target = base / relative
    if target.resolve() != target.absolute() or not target.is_file():
        raise ValueError("frozen source missing or symlinked: " + str(target))
    return target


def frozen_frontend(files):
    """Select an entry point from the bound manifest, never current sources."""
    if "ditserve/cli.py" in files:
        return "python"
    if all(name in files for name in ("ditserve/__main__.py", "ditserve/runtime/client.py")):
        return "native"
    raise ValueError("required frozen CLI absent: cli.py or __main__.py + runtime/client.py")


def inspect_run(root=ROOT, *, allow_output=False):
    p = paths(root)
    result = {"ready": False, "reasons": [], "gpu_status": "not_queried", "sudo_status": "not_attempted",
              "work_dir": str(p["work"]), "ledger": str(p["ledger"])}
    if os.geteuid() != EXPECTED_UID:
        result["reasons"].append("Run as uid 1002 (zyd); do not sudo this script.")
        return result
    try:
        # A read-only snapshot gives a useful reason while the real controller owns its locks.
        current = load(p["run"])
        result["run_status"] = current.get("status")
        if current.get("status") in ("running", "waiting_resource", "unsettled"):
            raise ValueError("software phase is still active or unsettled; wait for software_trace completion")
        with unlocked_controls(p):
            run, ledger = load(p["run"]), load(p["ledger"])
            if run.get("schema") != "ditserve.run.v1" or run.get("experiment") != EXPERIMENT or run.get("run_id") != RUN_ID:
                raise ValueError("run identity differs")
            result["run_status"] = run.get("status")
            if run.get("status") in ("running", "waiting_resource", "unsettled"):
                raise ValueError("software controller is still active or unsettled")
            if Path(run.get("budget_ledger", "")).absolute() != p["ledger"].absolute():
                raise ValueError("run uses a different cumulative ledger")
            config = run["config"]
            if run["config_sha256"] != canonical_digest(config):
                raise ValueError("frozen run config hash mismatch")
            model = config["workers"]["model"]
            if set(config["workers"]) != {"model"} or model["backend"] != "wan_profile" or \
                    model["gpu_uuid"] != GPU_UUID or model["config"].get("required_uid") != EXPECTED_UID or \
                    model["python"] != str(p["python"]):
                raise ValueError("worker identity/device/interpreter/uid guard differs")
            phases = {phase["id"]: phase for phase in config["phases"]}
            if set(phases) != {"software_trace", "hardware_metrics"} or \
                    phases["hardware_metrics"].get("admission_wait_seconds") != 0:
                raise ValueError("unexpected phase configuration or hardware admission wait")
            if any(action.get("mode") not in ("warmup", "clean", "hardware")
                   for action in phases["hardware_metrics"]["actions"]):
                raise ValueError("hardware pass cannot include torch.profiler")
            attempts = run.get("phases", [])
            software = [a for a in attempts if a["id"] == "software_trace" and a.get("status") == "completed"]
            if not software:
                raise ValueError("software_trace has not completed")
            software = software[-1]
            if not software.get("qualification", {}).get("qualified", False):
                raise ValueError("software_trace completed but did not qualify")
            if any(a["id"] == "hardware_metrics" for a in attempts):
                raise ValueError("hardware pass was already attempted; review existing evidence before any retry")
            manifest = load(p["sources"] / "manifest.json")
            if manifest != run["source"]:
                raise ValueError("source manifest no longer matches run")
            binary = _frozen_path(p["sources"], "ditserve-core")
            if digest(binary) != manifest["core_sha256"] or not os.access(binary, os.X_OK):
                raise ValueError("frozen native binary hash or executable bit differs")
            for key, directory in (("files", "python"), ("native_files", "native")):
                for relative, sha256 in manifest[key].items():
                    if digest(_frozen_path(p["sources"] / directory, relative)) != sha256:
                        raise ValueError("frozen source hash mismatch: " + relative)
            frontend = frozen_frontend(manifest["files"])
            for alternatives in (("ditserve/evaluation/nsys_metrics.py",),
                                 ("ditserve/models/wan/profile.py", "ditserve/models/wan_profile.py")):
                if not any(path in manifest["files"] for path in alternatives):
                    raise ValueError("required frozen Python module absent: " + " or ".join(alternatives))
            if ledger.get("schema_version") != "ditserve.budget.v1" or ledger.get("active") is not None:
                raise ValueError("budget ledger missing, incompatible or active")
            expected_limits = {"total_gpu_seconds": 3600.0, "repair_reserve_gpu_seconds": 300.0,
                               "carry": "none", "phase_order": ["software_trace", "hardware_metrics"], "phase_gpu_seconds": {}}
            if ledger["limits"] != expected_limits:
                raise ValueError("immutable cumulative budget allocation differs")
            sessions = ledger["sessions"]
            session = next((s for s in sessions if s["session_id"] == software["session_id"]), None)
            if not session or session["status"] != "completed" or not session.get("cleanup_verified") or \
                    session["identity"]["config_sha256"] != run["config_sha256"] or \
                    session["identity"]["source_sha256"] != canonical_digest(manifest) or \
                    session["identity"]["run_id"] != RUN_ID:
                raise ValueError("settled software session does not bind this config/source/run")
            used = ledger["used_gpu_seconds"]
            if not math.isfinite(used) or used < 0 or not math.isclose(
                    used, sum(s["gpu_seconds"] for s in sessions), rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError("budget charge history differs")
            remaining = min(3600.0 - used, 3300.0 - ledger["used_regular_gpu_seconds"])
            result["remaining_regular_gpu_seconds"] = remaining
            if remaining <= 60:
                raise ValueError("cumulative ledger has insufficient time after cleanup reserve")
            if p["output"].exists() and not allow_output:
                raise ValueError("nsys output already exists; no automatic overwrite or retry")
            result.update(ready=True, config_sha256=run["config_sha256"], source_sha256=canonical_digest(manifest),
                          allocation_id=ledger["allocation_id"], frozen_binary=str(binary),
                          frozen_python_path=str(p["sources"] / "python"), frozen_frontend=frontend)
    except BlockingIOError as error:
        result["reasons"].append("BlockingIOError: an active controller holds the run or cumulative budget lock")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["reasons"].append(type(error).__name__ + ": " + str(error))
    return result


def capture_command(p, gpu_index, *, non_interactive=False):
    frontend = frozen_frontend(load(p["sources"] / "manifest.json")["files"])
    environment = {"PYTHONPATH": str(p["sources"] / "python"), "DITSERVE_CORE": str(p["sources"] / "ditserve-core"),
                   "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": GPU_UUID,
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    controller = [str(p["python"]), "-m", "ditserve"] if frontend == "python" else [
        str(p["sources"] / "ditserve-core"), "--root", str(p["root"]),
        "--package-root", str(p["sources"] / "python"), "--python", str(p["python"])]
    return ["sudo", *(["-n"] if non_interactive else []), "--", str(NSYS), "profile", "--run-as=zyd", "--inherit-environment=false",
            "--sample=none", "--cpuctxsw=none", "--trace=cuda,nvtx",
            "--gpu-metrics-device=" + str(gpu_index), "--gpu-metrics-frequency=1000", "--kill=none",
            "--force-overwrite=false", "--export=none", "--output=" + str(p["output"] / "counters"),
            "--env-var=" + ",".join(key + "=" + value for key, value in environment.items()),
            *controller, "--json", "run", "--config", str(p["output"] / "config.json"),
            "--phase", "hardware_metrics", "--resume", "--allow-gpu", "--run-id", RUN_ID,
            "--work-dir", str(p["work"]), "--results-root", str(p["root"] / "results"), "--ledger", str(p["ledger"])]


def frozen_metrics(p, sqlite_path):
    """Parse with the frozen package, including its relative helper imports."""
    environment = dict(os.environ, PYTHONPATH=str(p["sources"] / "python"),
                       PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json,sys; from ditserve.evaluation.nsys_metrics import summarize_nsys_metrics; "
            "print(json.dumps(summarize_nsys_metrics(sys.argv[1], expected_sample_hz=1000.0), allow_nan=False))")
    completed = subprocess.run([str(p["python"]), "-c", code, str(sqlite_path)],
                               env=environment, cwd=p["sources"], capture_output=True,
                               text=True, check=True, timeout=180)
    return json.loads(completed.stdout, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _trusted_profiler():
    if NSYS.resolve() != NSYS or not NSYS.is_file():
        raise ValueError("fixed profiler missing or unexpectedly symlinked")
    evidence = []
    for path in (NSYS, *NSYS.parents):
        info = path.stat()
        mode = stat.S_IMODE(info.st_mode)
        writable = os.access(path, os.W_OK)
        evidence.append({"path": str(path), "uid": info.st_uid, "gid": info.st_gid,
                         "mode": oct(mode), "current_user_can_write": writable})
        # The actual host installation is root:root 0775. Sandbox ownership
        # remapping is not used for execution; this entry runs in the user's terminal.
        if info.st_uid != 0 or mode & 0o002 or (mode & 0o020 and info.st_gid != 0) or writable:
            raise ValueError("profiler path is not a protected root-owned installation: " + str(path))
    return {"path": str(NSYS), "sha256": digest(NSYS), "path_permissions": evidence}


def _gpu_idle(*, deadline=None):
    def timeout():
        remaining = 15 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("GPU admission wait timed out")
        return min(15, remaining)
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu,pci.bus_id", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout(), check=True)
    matches = [row for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True)
               if len(row) == 6 and row[1].strip() == GPU_UUID]
    if len(matches) != 1:
        raise ValueError("A100 UUID is missing or ambiguous")
    row = [value.strip() for value in matches[0]]
    if "A100" not in row[2]:
        raise ValueError("GPU name does not match the bound A100 UUID")
    if float(row[3]) >= 512 or float(row[4]) >= 10:
        raise GpuBusy("A100 is busy; rerun after it becomes idle")
    applications = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=timeout(), check=True)
    if any(len(app) == 2 and app[1].strip() == GPU_UUID
           for app in csv.reader(io.StringIO(applications.stdout), skipinitialspace=True)):
        raise GpuBusy("A100 has an existing compute process")
    return {"index": int(row[0]), "uuid": row[1], "name": row[2], "pci_bus_id": row[5]}


def wait_for_admission(wait_seconds):
    """Wait read-only as the ordinary user; retain an already-authenticated ticket.

    No collector/model/output is created here. Only a known busy GPU is retryable.
    Sources, ledger and control locks are rechecked on every observation. sudo -n
    cannot ask for a password and its failure aborts the queue immediately.
    """
    if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)) or \
            not math.isfinite(wait_seconds) or not 0 < wait_seconds <= 3600:
        raise ValueError("wait_seconds must be positive and at most 3600")
    started = time.monotonic()
    deadline = started + wait_seconds
    baseline = None
    last_refresh = last_notice = float("-inf")
    idle = 0
    last_gpu = None
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("GPU admission wait timed out; no hardware capture started")
        check = inspect_run()
        if not check["ready"]:
            raise ValueError("queued capture prerequisites no longer hold: " + "; ".join(check["reasons"]))
        binding = tuple(check.get(key) for key in ("config_sha256", "source_sha256", "allocation_id"))
        if baseline is None:
            baseline = binding
        elif binding != baseline:
            raise ValueError("run/source/budget binding changed while waiting")
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise TimeoutError("GPU admission wait timed out; no hardware capture started")
        if now - last_refresh >= 60:
            refreshed = subprocess.run(["sudo", "-n", "-v"], stdin=subprocess.DEVNULL,
                                       capture_output=True, text=True, check=False, timeout=min(10, remaining))
            if refreshed.returncode:
                raise ValueError("sudo authentication expired or unavailable in this terminal; no hardware capture started")
            last_refresh = time.monotonic()
        try:
            gpu = _gpu_idle(deadline=deadline)
        except GpuBusy as error:
            idle, last_gpu = 0, None
            if time.monotonic() - last_notice >= 60:
                print("Waiting for A100 (%ds remaining): %s" % (max(0, deadline - time.monotonic()), error), flush=True)
                last_notice = time.monotonic()
        else:
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU admission wait timed out; no hardware capture started")
            idle = idle + 1 if last_gpu == gpu else 1
            last_gpu = gpu
            if idle >= 2:
                print("A100 idle in two observations; rechecking capture prerequisites.", flush=True)
                return {"wait_seconds": time.monotonic() - started, "gpu": gpu, "idle_observations": idle}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("GPU admission wait timed out; no hardware capture started")
        time.sleep(min(15, remaining))


def _supported(help_text, gpu):
    if any(marker in help_text.lower() for marker in ("no permission", "permission denied", "err_nvgpuctrperm")):
        return False
    unsupported_section = False
    for line in help_text.splitlines():
        lowered = line.lower()
        if "some gpus are not supported" in lowered or "unsupported gpus:" in lowered:
            unsupported_section = True
        if unsupported_section or any(marker in lowered for marker in
                ("unsupported", "not supported", "no permission", "permission denied", "err_nvgpuctrperm")):
            continue
        match = re.match(r"\s*(\d+):\s*(.*)", line)
        if match and int(match.group(1)) == gpu["index"] and "A100" in match.group(2):
            if gpu.get("pci_bus_id"):
                pci = re.search(r"PCI\[([0-9a-fA-F:.]+)\]", match.group(2))
                normalize = lambda value: tuple(int(part, 16) for part in re.split(r"[:.]", value))
                if not pci or normalize(pci.group(1)) != normalize(gpu["pci_bus_id"]):
                    continue
            return True
    return False


def _write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def capture(*, non_interactive=False):
    p = paths()
    check = inspect_run()
    if not check["ready"]:
        raise ValueError("not ready: " + "; ".join(check["reasons"]))
    profiler = _trusted_profiler()
    gpu = _gpu_idle()
    time.sleep(1)
    if _gpu_idle() != gpu:
        raise ValueError("GPU identity changed between admission observations")
    p["output"].mkdir(mode=0o700)  # Existing output is never reused or overwritten.
    summary = {"schema": "ditserve.wan_hardware_capture.v1", "status": "failed", "measured": False,
               "preflight": check, "gpu": gpu, "profiler": profiler, "capture_returncode": None, "failure": None,
               "retention": "Retain raw report, SQLite and logs until verified final archive/figure integration closes dependencies."}
    try:
        probe_command = ["sudo", *(["-n"] if non_interactive else []), "--", str(NSYS), "profile", "--gpu-metrics-device=help"]
        probe = subprocess.run(probe_command, capture_output=True, text=True, check=False, timeout=120)
        (p["output"] / "device-help.log").write_text(probe.stdout + probe.stderr)
        if probe.returncode != 0 or not _supported(probe.stdout + probe.stderr, gpu):
            raise ValueError("privileged Nsight does not report this A100 as a supported GPU metrics device; no model launched")
        fresh = inspect_run(allow_output=True)
        if not fresh["ready"] or any(fresh.get(key) != check.get(key)
                for key in ("config_sha256", "source_sha256", "allocation_id")):
            raise ValueError("run/source/budget changed during sudo authentication")
        if _trusted_profiler()["sha256"] != profiler["sha256"]:
            raise ValueError("installed Nsight binary changed during authentication")
        if _gpu_idle() != gpu:
            raise ValueError("GPU identity changed before capture")
        _write_new(p["output"] / "config.json", load(p["run"])["config"])
        command = capture_command(p, gpu["index"], non_interactive=non_interactive)
        summary["command"] = command
        with (p["output"] / "capture.log").open("x") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        summary["capture_returncode"] = completed.returncode
        report = p["output"] / "counters.nsys-rep"
        if not report.is_file() or report.is_symlink():
            raise ValueError("Nsight report missing; raw logs preserved")
        exported = p["output"] / "counters.sqlite"
        with (p["output"] / "export.log").open("x") as log:
            export = subprocess.run([str(NSYS), "export", "--type=sqlite", "--lazy=false", "--force-overwrite=false",
                                     "--output=" + str(exported), str(report)], stdout=log, stderr=subprocess.STDOUT,
                                    timeout=180, check=False)
        if export.returncode or not exported.is_file():
            raise ValueError("unprivileged SQLite export failed; raw report/logs preserved")
        summary["metrics"] = frozen_metrics(p, exported)
        final_run = load(p["run"])
        hardware = [a for a in final_run["phases"] if a["id"] == "hardware_metrics"]
        qualified = bool(hardware) and hardware[-1]["status"] == "completed" and \
                    hardware[-1].get("qualification", {}).get("qualified", False)
        summary["hardware_phase_qualified"] = qualified
        if completed.returncode or not qualified:
            raise ValueError("hardware phase did not finish/qualify; sampling evidence retained for investigation")
        summary["status"] = summary["metrics"]["status"]
        summary["measured"] = summary["metrics"]["measured"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        summary["failure"] = type(error).__name__ + ": " + str(error)
        if "metrics" in summary:
            summary["metrics"].update(status="unknown", measured=False)
            for interval in summary["metrics"]["ranges"]:
                interval["status"] = "unknown"
                for metric in interval["metrics"]:
                    metric.update(status="unknown", measured=False, time_weighted_mean_pct=None)
                    metric["reason_codes"].append("hardware_capture_or_phase_failed")
    finally:
        summary["retained_files"] = []
        for path in p["output"].iterdir():
            if path.is_file() and not path.is_symlink():
                item = {"path": str(path), "bytes": path.stat().st_size, "sha256": None}
                try:
                    item["sha256"] = digest(path)
                except OSError as error:
                    item["hash_error"] = type(error).__name__ + ": " + str(error)
                summary["retained_files"].append(item)
        _write_new(p["output"] / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if summary["measured"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="CPU-only admission wait, at most 3600 seconds; needs an existing same-terminal sudo ticket")
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait_seconds) or not 0 <= args.wait_seconds <= 3600:
        parser.error("--wait-seconds must be between 0 and 3600")
    if args.check_only:
        result = inspect_run()
        result["note"] = "Read-only CPU preflight. GPU idleness and privileged metric support are checked only during capture."
        if result["ready"]:
            result["planned_command"] = shlex.join(capture_command(paths(), "A100_INDEX_FROM_UUID"))
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result["ready"] else 1
    try:
        if args.wait_seconds:
            wait_for_admission(args.wait_seconds)
        outcome = capture(non_interactive=bool(args.wait_seconds))
        # The user-requested combined figure and workbook are refreshed by an
        # ordinary-user, CPU-only publisher after the collector has finished.
        # Its failure must not erase the captured evidence or claim success.
        environment = dict(os.environ, PYTHONPATH=str(PACKAGE_ROOT), DITSERVE_ROOT=str(ROOT))
        published = subprocess.run([str(paths()["python"]), "-m", "ditserve.evaluation.wan_profile.publish"],
                                   env=environment, check=False)
        return outcome or int(published.returncode != 0)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Hardware capture admission cancelled; no queued capture started.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
