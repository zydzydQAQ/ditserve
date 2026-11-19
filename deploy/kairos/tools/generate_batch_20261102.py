#!/usr/bin/env python3
"""Generate the 2026-11-02 Kairos job spec batch for a given DITSERVE commit SHA."""

import argparse
from datetime import datetime
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1] / "ditserve"
OUT_DIR = None
STAGE = "p0"
DATE = None
IMAGE = "192.168.1.114:30500/jjjkairos/ditserve:xdit-dev-016e1ca-slim"


def resolve_sha(arg: str | None) -> str:
    target = arg or "HEAD"
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "--verify", f"{target}^{{commit}}"],
        text=True,
    ).strip()
    return commit[:7]


def env_list(mapping: dict[str, str]) -> list[dict[str, str]]:
    return [{"name": k, "value": str(v)} for k, v in mapping.items()]


def write_spec(name: str, command: str, gpus: int, env_map: dict[str, str]) -> Path:
    category = ("gfc" if "gfc-" in name else "quality" if "quality-" in name
                else "sweep" if "sweep-" in name else "main" if "main-" in name else "p0")
    if STAGE != "all" and category != STAGE:
        return None
    # PyTorchJob's admission webhook requires DNS-1035 (a leading letter),
    # while Kairos also limits the name to 52 characters.
    name = "d" + DATE + "-" + name.removeprefix("ditserve-")
    if len(name) > 52 or not re.fullmatch(r"[a-z]([-a-z0-9]*[a-z0-9])?", name):
        raise ValueError(f"invalid Kairos/PyTorchJob name: {name}")
    full_env = {"DITSERVE_RUN_ID": name, **env_map}
    if category == "main":
        # Calibrate batch 2 on this allocation and expose the same limit to all arms.
        full_env.setdefault("DITSERVE_IMAGE_BATCH", "2")
    spec = {
        "name": name,
        "image": IMAGE,
        "command": command,
        "workers": 1,
        "gpus_per_worker": gpus,
        "mount_shared": True,
        "env": env_list(full_env),
    }
    path = OUT_DIR / f"{name}.json"
    content = json.dumps(spec, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("existing job specification differs; use a new run identity: " + str(path))
    else:
        path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    global OUT_DIR, STAGE, DATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit", nargs="?", default="HEAD")
    parser.add_argument("--stage", choices=["p0", "main", "quality", "gfc", "sweep", "all"], default="p0")
    parser.add_argument("--gpus", type=int, choices=[2, 4, 6, 8], default=8)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    datetime.strptime(args.date, "%Y%m%d")
    sha = resolve_sha(args.commit)
    DATE, STAGE = args.date, args.stage
    gpus = args.gpus
    OUT_DIR = ROOT / "job-specs" / f"{DATE}-validation-{sha}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # Stage 0: Contract & GFC Microbench Gates
    written.append(
        write_spec(
            f"ditserve-contract-{sha}-r1",
            "bash /shared/data/DITSERVE/scripts/kairos_targeted_tests.sh",
            1,
            {
                "DITSERVE_BUILD": sha,
                "DITSERVE_TEST_ROOT": f"/shared/data/DITSERVE/tests/{sha}/tests/python",
            },
        )
    )
    written.append(
        write_spec(
            f"ditserve-gfc-microbench-{sha}-r1",
            "bash /shared/data/DITSERVE/scripts/kairos_gfc_microbench.sh",
            8,
            {
                "DITSERVE_BUILD": sha,
                "DITSERVE_WARMUP": "5",
                "DITSERVE_ITERS": "20",
                "DITSERVE_GROUP_TRIALS": "8",
            },
        )
    )

    # Stage 1: Controlled Preemption & GFC Serving Equivalence
    for tag, wan_model in [("wan21", "wan"), ("wan22", "wan2.2-ti2v-5b")]:
        written.append(
            write_spec(
                f"ditserve-preempt-{tag}-{sha}-s205-r1",
                "bash /shared/data/DITSERVE/scripts/kairos_native_serving.sh",
                gpus,
                {
                    "DITSERVE_BUILD": sha,
                    "DITSERVE_CALIBRATE": "1",
                    "DITSERVE_SEED": "205",
                    "DITSERVE_PATTERN": "bursty",
                    "DITSERVE_VIDEO_FRACTION": "0.5",
                    "DITSERVE_LOAD": "1.2",
                    "DITSERVE_COUNT": "16",
                    "DITSERVE_IMAGE_SLO_SCALE": "1.1",
                    "DITSERVE_DOORBELL": "1",
                    "DITSERVE_OUTPUT_MODE": "digest",
                    "DITSERVE_SUITE": "preempt",
                    "DITSERVE_FORMS": "cfg2",
                    "WAN_MODEL": wan_model,
                },
            )
        )

    for tag, wan_model in [("wan21", "wan"), ("wan22", "wan2.2-ti2v-5b")]:
        written.append(write_spec(f"ditserve-no-preempt-{tag}-{sha}-s205-r1",
            "bash /shared/data/DITSERVE/scripts/kairos_native_serving.sh", gpus,
            {"DITSERVE_BUILD": sha, "DITSERVE_CALIBRATE": "1", "DITSERVE_SEED": "205",
             "DITSERVE_NO_PREEMPTION": "1", "DITSERVE_SUITE": "preempt",
             "DITSERVE_FORMS": "cfg2", "DITSERVE_COUNT": "8", "WAN_MODEL": wan_model}))

    written.append(
        write_spec(
            f"ditserve-gfc-serve-wan21-{sha}-s205-r1",
            "bash /shared/data/DITSERVE/scripts/kairos_gfc_serving.sh",
            8,
            {
                "DITSERVE_BUILD": sha,
                "DITSERVE_CALIBRATE": "1",
                "DITSERVE_SEED": "205",
                "DITSERVE_PATTERN": "bursty",
                "DITSERVE_VIDEO_FRACTION": "0.5",
                "DITSERVE_LOAD": "1.2",
                "DITSERVE_COUNT": "16",
                "DITSERVE_IMAGE_SLO_SCALE": "1.1",
                "DITSERVE_DOORBELL": "1",
                "DITSERVE_POLICIES": "static-cfg2sp3,edf-cfg2",
                "WAN_MODEL": "wan",
            },
        )
    )

    # Stage 2: Main Baselines + Ablation under Tight-SLO Bursty Load (3 seeds x 2 models)
    for tag, wan_model in [("wan21", "wan"), ("wan22", "wan2.2-ti2v-5b")]:
        for seed in (211, 212, 213):
            written.append(
                write_spec(
                    f"ditserve-main-abl-{tag}-{sha}-s{seed}-bursty-r1",
                    "bash /shared/data/DITSERVE/scripts/kairos_native_serving.sh",
                    8,
                    {
                        "DITSERVE_BUILD": sha,
                        "DITSERVE_CALIBRATE": "1",
                        "DITSERVE_SEED": str(seed),
                        "DITSERVE_PATTERN": "bursty",
                        "DITSERVE_VIDEO_FRACTION": "0.5",
                        "DITSERVE_LOAD": "1.2",
                        "DITSERVE_COUNT": "40",
                        "DITSERVE_IMAGE_SLO_SCALE": "1.1",
                        "DITSERVE_DOORBELL": "1",
                        "DITSERVE_SUITE": "main+ablation",
                        "WAN_MODEL": wan_model,
                    },
                )
            )

    # Stage 3: Main Baselines under Poisson Load (3 seeds x 2 models)
    for tag, wan_model in [("wan21", "wan"), ("wan22", "wan2.2-ti2v-5b")]:
        for seed in (211, 212, 213):
            written.append(
                write_spec(
                    f"ditserve-main-{tag}-{sha}-s{seed}-poisson-r1",
                    "bash /shared/data/DITSERVE/scripts/kairos_native_serving.sh",
                    8,
                    {
                        "DITSERVE_BUILD": sha,
                        "DITSERVE_CALIBRATE": "1",
                        "DITSERVE_SEED": str(seed),
                        "DITSERVE_PATTERN": "poisson",
                        "DITSERVE_VIDEO_FRACTION": "0.2",
                        "DITSERVE_LOAD": "1.0",
                        "DITSERVE_COUNT": "40",
                        "DITSERVE_IMAGE_SLO_SCALE": "2.5",
                        "DITSERVE_DOORBELL": "1",
                        "DITSERVE_SUITE": "main",
                        "WAN_MODEL": wan_model,
                    },
                )
            )

    # Stage 4: Approx fixed-N matched quality plus controlled fold/fission mixed serving
    for tag, wan_model in [("wan21", "wan"), ("wan22", "wan2.2-ti2v-5b")]:
        written.append(
            write_spec(
                f"ditserve-quality-{tag}-{sha}-s201-r1",
                "bash /shared/data/DITSERVE/scripts/kairos_native_approx.sh",
                8,
                {
                    "DITSERVE_BUILD": sha,
                    "DITSERVE_CALIBRATE": "1",
                    "DITSERVE_SEED": "201",
                    "DITSERVE_VIDEO_FRACTION": "1.0",
                    "DITSERVE_COUNT": "32",
                    "DITSERVE_QUALITY": "32",
                    "WAN_MODEL": wan_model,
                },
            )
        )

    # Stage 5: Sensitivity & Scalability Sweeps (Rate, SLO Scale, Gamma CV, and 4/6/8 GPUs)
    for tag, wan_model in [("wan21", "wan"), ("wan22", "wan2.2-ti2v-5b")]:
        for sweep_kind in ("rate", "slo", "cv", "gpus"):
            written.append(
                write_spec(
                    f"ditserve-sweep-{sweep_kind}-{tag}-{sha}-s211-r1",
                    "bash /shared/data/DITSERVE/scripts/kairos_native_sweep.sh",
                    8,
                    {
                        "DITSERVE_BUILD": sha,
                        "DITSERVE_CALIBRATE": "1",
                        "DITSERVE_SWEEP_KIND": sweep_kind,
                        "DITSERVE_SEED": "211",
                        "DITSERVE_SUITE": "sweep",
                        "DITSERVE_DOORBELL": "1",
                        "WAN_MODEL": wan_model,
                    },
                )
            )

    written = [path for path in written if path is not None]
    print(f"Generated {len(written)} job specs for commit {sha} in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
