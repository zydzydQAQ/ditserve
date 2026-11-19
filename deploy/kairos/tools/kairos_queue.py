#!/usr/bin/env python3
"""Submit an explicit, ordered Kairos job list when account GPU slots are free."""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
TERMINAL = {"Succeeded", "Failed", "Deleted", "Stopped", "Cancelled", "Canceled"}


def load_api():
    path = ROOT / ".local" / "kairos-api.py"
    spec = importlib.util.spec_from_file_location("kairos_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.login()
    return module


def storage_gate():
    command = [
        "ssh", "-o", "ControlMaster=auto", "-o", f"ControlPath=/run/user/{os.getuid()}/ditserve-ssh-%C",
        "-o", "ControlPersist=15m", "-o", "ClearAllForwardings=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15", "jjjkairos", "df", "-BG", "/data",
    ]
    human = subprocess.run(command[:-3] + ["df", "-h", "/data"], capture_output=True, text=True, timeout=30)
    if human.returncode:
        raise RuntimeError("cannot verify /data: " + human.stderr.strip())
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError("cannot verify /data: " + result.stderr.strip())
    lines = [line.split() for line in result.stdout.splitlines() if line.strip()]
    fields = next((line for line in lines if line[-1] == "/data"), None)
    if fields is None:
        raise RuntimeError("df did not report /data")
    used_gib, percent = int(fields[2].removesuffix("G")), int(fields[4].removesuffix("%"))
    print(f"storage: {used_gib}G, {percent}%", flush=True)
    if used_gib > 805 or percent >= 81:
        raise RuntimeError("shared-storage stop threshold reached")
    return {"used_gib": used_gib, "percent": percent}


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))
    return value


def env_values(spec):
    entries = spec.get("env", [])
    values = {item["name"]: item["value"] for item in entries}
    if len(values) != len(entries):
        raise ValueError("duplicate environment keys")
    return values


def gpu_need(spec):
    workers = integer(spec["workers"], "workers", 1)
    width = integer(spec["gpus_per_worker"], "gpus_per_worker", 1)
    if workers > 7 or width > 8:
        raise ValueError("GPU request exceeds documented Kairos worker/node limits")
    return workers * width


def checked_get(api, path):
    status, value = api.req("GET", path)
    if status != 200:
        raise RuntimeError("cannot query " + path + ": HTTP " + str(status))
    return value


def capacity_from_records(me, nodes, jobs, namespace, quota_cap=None):
    """Allocation counts, never utilization, determine free slots.

    Node field names come from the cached Kairos console; unknown live schemas
    fail closed. A snapshot is only admission advice: the platform owns placement.
    """
    if not isinstance(me, dict) or me.get("namespace") != namespace:
        raise ValueError("account namespace missing or differs from requested namespace")
    quota = integer(me.get("gpu_quota"), "live gpu_quota")
    if quota_cap is not None:
        quota = min(quota, integer(quota_cap, "quota cap", 1))
    if not isinstance(nodes, list) or not isinstance(jobs, list):
        raise ValueError("unknown nodes/jobs response schema")
    free, seen = [], set()
    for node in nodes:
        name = node.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("missing or duplicate node identity")
        seen.add(name)
        for field in ("ready", "cordoned", "gpu_healthy"):
            if type(node.get(field)) is not bool:
                raise ValueError("unknown node eligibility: " + field)
        allocatable = integer(node.get("gpu_allocatable"), "gpu_allocatable")
        allocated = integer(node.get("gpu_allocated"), "gpu_allocated")
        if allocated > allocatable:
            raise ValueError("allocated GPUs exceed allocatable GPUs")
        if node["ready"] and not node["cordoned"] and node["gpu_healthy"]:
            free.append({"name": name, "free_gpus": allocatable - allocated})
    active = []
    for job in jobs:
        if job.get("state") in TERMINAL:
            continue
        if not isinstance(job.get("namespace"), str):
            raise ValueError("active job lacks namespace")
        if job["namespace"] == namespace:
            if not isinstance(job.get("name"), str) or not isinstance(job.get("state"), str):
                raise ValueError("active job lacks identity/state")
            integer(job.get("total_gpus"), "active job total_gpus", 1)
            active.append(job)
    used = sum(job["total_gpus"] for job in active)
    return {"namespace": namespace, "quota": quota, "reserved_gpus": used,
            "account_free_gpus": max(0, quota - used), "nodes": free,
            "cluster_free_gpus": sum(node["free_gpus"] for node in free),
            "unplaced_account_jobs": [job["name"] for job in active if job["state"] != "Running"]}


def live_capacity(api, jobs, namespace, quota_cap=None):
    return capacity_from_records(checked_get(api, "/api/me"), checked_get(api, "/api/cluster/nodes"),
                                 jobs, namespace, quota_cap)


def capacity_fits(spec, capacity):
    needed = gpu_need(spec)
    # A newly submitted/queued account job may not yet appear in node allocation
    # counts. Wait for placement rather than double-spending the same free GPUs.
    if capacity["unplaced_account_jobs"] or needed > capacity["account_free_gpus"]:
        return False
    eligible = sum(node["free_gpus"] >= spec["gpus_per_worker"] for node in capacity["nodes"])
    return eligible >= spec["workers"]


def stage_gate(spec, receipt_path, required_stage):
    env = env_values(spec)
    formal = env.get("DITSERVE_BENCHMARK_MODE") == "formal"
    if not receipt_path:
        if formal:
            raise ValueError("formal jobs require --stage-gate evidence receipt")
        return None
    if not required_stage:
        raise ValueError("--stage-gate requires --required-stage")
    module_path = ROOT / "ditserve/benchmarks/stage_gates.py"
    loader = importlib.util.spec_from_file_location("ditserve_stage_gates_queue", module_path)
    module = importlib.util.module_from_spec(loader)
    sys.path.insert(0, str(module_path.parent))
    try:
        loader.loader.exec_module(module)
        protocol = env.get("DITSERVE_PROTOCOL_SHA256")
        if formal and not re.fullmatch(r"[0-9a-f]{64}", protocol or ""):
            raise ValueError("formal jobs require protocol SHA256")
        receipt = module.verify_receipt(receipt_path, build_commit=env.get("DITSERVE_BUILD"),
            protocol_sha256=protocol, required_stage=required_stage)
        stage = receipt["stages"][required_stage]
        if env.get("DITSERVE_STAGE_ACTION") not in stage.get("allows", []):
            raise ValueError("receipt does not permit this stage action")
        return receipt
    finally:
        sys.path.pop(0)


def build_gate(api, spec):
    env = env_values(spec)
    build = env.get("DITSERVE_BUILD", "")
    if not re.fullmatch(r"[0-9a-f]{7,40}", build):
        raise ValueError("every job needs an immutable hexadecimal DITSERVE_BUILD")
    return {"name": spec["name"], "build": build,
            "build_info": f"/shared/code/DITSERVE/builds/{build}/BUILD_INFO.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--namespace", default="user-jjjkairos")
    parser.add_argument("--quota", type=int, help="optional cap below the live account quota; never an override")
    parser.add_argument("--stage-gate", type=Path, help="receipt revalidated against original evidence before each submission")
    parser.add_argument("--required-stage", choices=["CONTRACT", "P0", "P1", "P2", "P3", "P4", "P5"])
    parser.add_argument("--backfill", action="store_true", help="allow later independent, equally gated jobs to use fragmented free slots")
    parser.add_argument("--max-active-jobs", type=int, help="limit concurrency within this explicit queue")
    args = parser.parse_args()
    if not 1 <= args.interval <= 60:
        parser.error("--interval must be in [1, 60] seconds")
    if args.quota is not None and args.quota < 1:
        parser.error("--quota must be positive")
    if args.max_active_jobs is not None and args.max_active_jobs < 1:
        parser.error("--max-active-jobs must be positive")
    specs = [json.loads(path.read_text(encoding="utf-8")) for path in args.specs]
    names = [spec["name"] for spec in specs]
    if any(len(name) > 52 or not re.fullmatch(r"[a-z]([-a-z0-9]*[a-z0-9])?", name) for name in names):
        raise ValueError("Kairos/PyTorchJob names must start with a letter and contain at most 52 DNS-1035 characters")
    if len(set(names)) != len(names):
        raise ValueError("duplicate job names in the requested queue")
    for spec in specs:
        if not spec.get("mount_shared") or (args.quota is not None and gpu_need(spec) > args.quota):
            raise ValueError("invalid shared mount or GPU request: " + spec["name"])
        if next(item["value"] for item in spec["env"] if item["name"] == "DITSERVE_RUN_ID") != spec["name"]:
            raise ValueError("run ID must equal the unique job name")
        gpu_need(spec)
        stage_gate(spec, args.stage_gate, args.required_stage)
        print(json.dumps(build_gate(None, spec)), flush=True)
    api = load_api()
    status, jobs = api.req("GET", "/api/jobs")
    if status != 200:
        raise RuntimeError("cannot list jobs")
    if not isinstance(jobs, list):
        raise ValueError("unknown jobs response schema")
    known = {job.get("name") for job in jobs if job.get("namespace") == args.namespace}
    if known.intersection(names):
        raise ValueError("job name already exists: " + ", ".join(sorted(known.intersection(names))))
    storage_gate()
    capacity = live_capacity(api, jobs, args.namespace, args.quota)
    if any(gpu_need(spec) > capacity["quota"] for spec in specs):
        raise ValueError("requested job exceeds current account quota")
    print(json.dumps({"capacity": capacity}), flush=True)
    if not args.submit:
        print("dry run only; pass --submit after reviewing the ordered specs", flush=True)
        return
    pending = list(specs)
    submitted = {}
    previous = {}
    last_storage_check = time.monotonic()
    while pending or any(job not in TERMINAL for job in submitted.values()):
        status, jobs = api.req("GET", "/api/jobs")
        if status != 200:
            raise RuntimeError("cannot refresh jobs")
        if not isinstance(jobs, list):
            raise ValueError("unknown jobs response schema")
        by_name = {job.get("name"): job for job in jobs if job.get("namespace") == args.namespace}
        missing = set(submitted) - set(by_name)
        if missing:
            raise RuntimeError("submitted jobs disappeared; investigate before retry: " + ", ".join(sorted(missing)))
        current = {name: by_name[name].get("state", "Unknown") for name in submitted}
        submitted.update(current)
        if current != previous:
            print(json.dumps(current), flush=True)
            previous = current
        failed = [name for name, state in current.items() if state in TERMINAL and state != "Succeeded"]
        if failed:
            raise RuntimeError("job failed; no further submissions: " + ", ".join(failed))
        if pending:
            capacity = live_capacity(api, jobs, args.namespace, args.quota)
            print(json.dumps({"capacity": capacity}), flush=True)
            indices = range(len(pending)) if args.backfill else range(1)
            eligible = next((index for index in indices if capacity_fits(pending[index], capacity)), None)
            queue_active = sum(state not in TERMINAL for state in submitted.values())
            if eligible is not None and (args.max_active_jobs is None or queue_active < args.max_active_jobs):
                spec = pending[eligible]
                # Revalidate evidence and current name ownership after waiting. A
                # Succeeded platform state is never a scientific stage receipt.
                if spec["name"] in by_name:
                    raise RuntimeError("queued name appeared externally: " + spec["name"])
                stage_gate(spec, args.stage_gate, args.required_stage)
                storage_gate()
                last_storage_check = time.monotonic()
                status, body = api.req("POST", "/api/jobs", spec)
                if status not in (200, 201, 202):
                    raise RuntimeError(f"submit {spec['name']} failed: HTTP {status}; inspect platform logs")
                submitted[spec["name"]] = "Created"
                print("submitted:", spec["name"], gpu_need(spec), "GPUs", flush=True)
                pending.pop(eligible)
        if pending or any(state not in TERMINAL for state in submitted.values()):
            if time.monotonic() - last_storage_check >= 60:
                storage_gate()
                last_storage_check = time.monotonic()
            time.sleep(args.interval)
    print("all submitted jobs succeeded", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ":", str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
