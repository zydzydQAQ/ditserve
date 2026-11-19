#!/usr/bin/env python3
"""Submit b1 and b2 jobs for formal v2 benchmark (a7871f0) and monitor until Running."""

import json
from pathlib import Path
import time

from kairos_queue import TERMINAL, load_api, storage_gate

ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / ".local" / "ditserve-kairos-rewrite-runtime" / "job-specs" / "20261105-formal-v2-a7871f0"
B1_SPEC = SPEC_DIR / "d20261105-formal-v2-wan22-b1-a7871f0-r1.json"
B2_SPEC = SPEC_DIR / "d20261105-formal-v2-wan22-b2-a7871f0-r1.json"
NAMESPACE = "user-jjjkairos"


def main():
    print("=== Step 1: Storage Pre-check ===")
    storage_gate()

    print("\n=== Step 2: Initialize Kairos API ===")
    api = load_api()

    # Pre-submission job check
    s, jobs = api.req("GET", "/api/jobs")
    if s != 200:
        raise RuntimeError(f"Failed to query /api/jobs: {s}")
    existing_names = {j["name"] for j in jobs}

    specs = [
        ("b1", json.loads(B1_SPEC.read_text(encoding="utf-8"))),
        ("b2", json.loads(B2_SPEC.read_text(encoding="utf-8"))),
    ]

    for label, spec in specs:
        if spec["name"] in existing_names:
            print(f"Job {spec['name']} already exists on cluster! Skipping submit.")
        else:
            print(f"\nSubmitting {label}: {spec['name']}...")
            status, resp = api.req("POST", "/api/jobs", spec)
            print(f"Submit response: status={status}, body={json.dumps(resp)}")
            if status not in (200, 201):
                raise RuntimeError(f"Failed to submit {spec['name']}: {status} {resp}")

    print("\n=== Step 3: Monitoring until both jobs reach Running ===")
    job_names = [spec["name"] for _, spec in specs]
    states = {name: "Unknown" for name in job_names}
    start_time = time.monotonic()
    last_log_time = 0

    while True:
        elapsed = int(time.monotonic() - start_time)
        all_running = True

        for name in job_names:
            s, detail = api.req("GET", f"/api/jobs/{NAMESPACE}/{name}")
            if s == 200 and isinstance(detail, dict):
                cur_state = detail.get("state", "Unknown")
                states[name] = cur_state
                pods = detail.get("pods", [])
                pod_desc = [f"{p.get('name')}({p.get('phase')}, {p.get('node')})" for p in pods]
                if cur_state != "Running":
                    all_running = False
            else:
                all_running = False
                print(f"[{elapsed}s] Error fetching job detail for {name}: {s} {detail}")

        print(f"[{elapsed}s] Job states: " + ", ".join(f"{k}={v}" for k, v in states.items()))

        if any(v in (TERMINAL - {"Succeeded"}) for v in states.values()):
            raise RuntimeError(f"A job entered terminal failure state: {states}")

        if all_running:
            print(f"\nSUCCESS: Both jobs reached Running status after {elapsed}s!")
            break

        time.sleep(5)

    print("\n=== Step 4: Capture Details and Initial Startup Logs ===")
    final_report = {}
    for name in job_names:
        s, detail = api.req("GET", f"/api/jobs/{NAMESPACE}/{name}")
        s_logs, logs = api.req("GET", f"/api/jobs/{NAMESPACE}/{name}/logs")
        final_report[name] = {
            "detail": detail,
            "logs_preview": logs[:4000] if isinstance(logs, str) else str(logs)[:4000],
        }
        print(f"\n--- Job: {name} ---")
        print("Detail:", json.dumps(detail, indent=2))
        print("Logs Preview:\n", logs[:2000] if isinstance(logs, str) else str(logs)[:2000])

    # Save summary report for teamwork handoff
    out_file = Path("/mnt/e/000_code/zyd-2026/.agents/teamwork/teamwork_worker_submit_1/submission_result.json")
    out_file.write_text(json.dumps(final_report, indent=2))
    print(f"\nSaved submission result to {out_file}")


if __name__ == "__main__":
    main()
