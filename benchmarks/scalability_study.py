#!/usr/bin/env python3
"""Calibrated Rate, SLO, CV (Gamma), and Cluster/Planner Scalability (4..64 GPUs) study."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "workers") not in sys.path:
    sys.path.insert(0, str(ROOT / "workers"))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from ditserve.evaluation.common import read_json, verify_seal
from ditserve.evaluation.workloads import build_trace, validate_trace
from native_serving import FORM_DEGREES, FORMS, POLICY_DISPLAY, SUITES, partition_image_gpus


def find_ditserve_binary(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    for candidate in (
        ROOT / ".cache" / "build" / "ditserve",
        Path("/opt/ditserve/bin/ditserve"),
    ):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("ditserve")
    if found:
        return found
    raise FileNotFoundError("cannot locate ditserve binary; pass --binary")


def make_classes(model: str, profile: dict, video_slo_scale: float, image_slo_scale: float) -> dict:
    model_key = f"{model}-preflight"
    video_cost = profile["classes"][model_key]
    gpu_service_ms = (
        video_cost.get("prepare_ms", 0.0)
        + video_cost["steps"] * video_cost["step_compute_ms"]
        + video_cost.get("finish_ms", 0.0)
    )
    service_ms = video_cost.get("encode_ms", 0.0) + gpu_service_ms
    return {
        model_key: dict(
            video_cost,
            **{
                "class": model_key,
                "kind": "video",
                "service_ms": service_ms,
                "gpu_service_ms": gpu_service_ms,
                "slo_scale": video_slo_scale,
                "spec": profile.get("source", {}).get("shape", {}),
            },
        ),
        "sd35-preflight": {
            "class": "sd35-preflight",
            "kind": "image",
            "service_ms": profile["classes"]["sd35-preflight"]["service_ms"],
            "gpu_service_ms": profile["classes"]["sd35-preflight"]["service_ms"],
            "slo_scale": image_slo_scale,
            "spec": {"height": 1024, "width": 1024, "steps": 28, "frames": 1, "batch_size": 1},
        },
    }


def run_cell(
    binary: str,
    model: str,
    profile: dict,
    pools: dict,
    *,
    policy: str,
    gpus: int,
    seed: int,
    load: float,
    pattern: str,
    cv: float,
    video_fraction: float,
    count: int,
    video_slo_scale: float,
    image_slo_scale: float,
) -> dict:
    classes = make_classes(model, profile, video_slo_scale, image_slo_scale)
    model_key = f"{model}-preflight"
    video_cost = profile["classes"][model_key]
    image_share = partition_image_gpus(
        gpus, video_fraction, video_cost, profile["classes"]["sd35-preflight"]["service_ms"]
    )
    window = {
        "count": count,
        "video_fraction": video_fraction,
        "pattern": pattern,
        "load": load,
    }
    if pattern == "bursty":
        window.update(bursts=2, burst_active_fraction=0.25)
    elif pattern == "gamma":
        window["cv"] = float(cv)
    trace = build_trace(classes, pools, [window], slots=gpus, seed=seed)
    validate_trace(trace)
    forms = [name for name in FORMS if (model != "wan21" or name != "sp8") and FORM_DEGREES[name] <= min(gpus, 8)]
    payload = {
        "profile": profile,
        "policy": policy,
        "gpus": gpus,
        "encode_slots": gpus,
        "options": {
            "forms": forms,
            "quantum_blocks": 4,
            "image_batch": 1,
            "doorbell": True,
            "partition_image_gpus": image_share,
            "measure_planner_cpu": True,
        },
        "workload": trace["workload"],
    }
    proc = subprocess.run(
        [binary, "rpc", "simulate"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    out = json.loads(proc.stdout)
    v_sum = out["by_kind"]["video"]
    i_sum = out["by_kind"]["image"]
    tot_sum = out["metrics"]
    duration_s = trace["manifest"]["total_duration_ms"] / 1000.0
    makespan_s = out["makespan_ms"] / 1000.0
    tot_att = tot_sum["deadline_attainment"]
    goodput_rps = (tot_att * count / makespan_s) if makespan_s > 0 else 0.0
    return {
        "policy": policy,
        "display": POLICY_DISPLAY.get(policy, policy),
        "gpus": gpus,
        "seed": seed,
        "load": load,
        "pattern": pattern,
        "cv": cv if pattern == "gamma" else None,
        "count": count,
        "video_fraction": video_fraction,
        "video_slo_scale": video_slo_scale,
        "image_slo_scale": image_slo_scale,
        "arrival_rate_rps": trace["manifest"]["rate_rps"],
        "trace_duration_s": duration_s,
        "makespan_s": makespan_s,
        "total_slo_attainment": tot_att,
        "video_slo_attainment": v_sum["attainment"],
        "image_slo_attainment": i_sum["attainment"],
        "video_p50_ms": v_sum["latency_p50_ms"],
        "video_p95_ms": v_sum["latency_p95_ms"],
        "image_p50_ms": i_sum["latency_p50_ms"],
        "image_p95_ms": i_sum["latency_p95_ms"],
        "goodput_rps": goodput_rps,
        "gpu_utilization": out["gpu_utilization"],
        "plans": out["plans"],
        "block_preemptions": out["block_preemptions"],
        "resizes": out["resizes"],
        "planner_cpu_ms_total": out.get("planner_cpu_ms_total", 0.0),
        "planner_cpu_ms_mean": out.get("planner_cpu_ms_mean", 0.0),
        "planner_cpu_ms_max": out.get("planner_cpu_ms_max", 0.0),
    }


def format_markdown(report: dict) -> str:
    lines = [
        f"# DitServe Sensitivity & Scalability Study ({report['model'].upper()}, Seed {report['seed']})",
        "",
        "> Calibrated C++ `cluster_plan` + discrete-event simulation on measured profile (`provenance="
        f"{report['profile_provenance']}`).",
        "",
    ]
    for section_key, title, x_col, x_label in (
        ("rate_sweep", "1. Arrival Rate / Load Sweep (8 GPUs, Gamma CV=2.0, 50% Video, Count=40)", "load", "Load (Rate rps)"),
        ("slo_sweep", "2. Latency SLO Scale Sweep (8 GPUs, Load=1.0, Gamma CV=2.0, Count=40)", "slo_tag", "SLO Scale (V / I)"),
        ("cv_sweep", "3. Arrival Burstiness / Gamma CV Sweep (8 GPUs, Load=1.0, Count=40)", "cv", "Gamma CV"),
        ("gpu_scalability", "4. Cluster & Planner Scalability (Weak Scaling: 4 -> 64 GPUs, 5 req/GPU, Load=1.0)", "gpus", "GPUs (Reqs)"),
    ):
        rows = report["sweeps"][section_key]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            f"| {x_label} | Policy | Total SLO (%) | Video SLO (%) | Image SLO (%) | Video P95 (s) | Image P95 (s) | Goodput (req/s) | Planner Mean/Max (ms) |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            if section_key == "rate_sweep":
                x_val = f"{r['load']:.1f}x ({r['arrival_rate_rps']:.2f} rps)"
            elif section_key == "slo_sweep":
                x_val = f"{r['video_slo_scale']:.2f}x / {r['image_slo_scale']:.2f}x"
            elif section_key == "cv_sweep":
                x_val = f"CV={r['cv']:.1f}"
            else:
                x_val = f"{r['gpus']} GPUs ({r['count']} reqs)"
            lines.append(
                f"| {x_val} | {r['display']} (`{r['policy']}`) | "
                f"{r['total_slo_attainment'] * 100:.1f}% | {r['video_slo_attainment'] * 100:.1f}% | "
                f"{r['image_slo_attainment'] * 100:.1f}% | {r['video_p95_ms'] / 1000:.2f} | "
                f"{r['image_p95_ms'] / 1000:.2f} | {r['goodput_rps']:.3f} | "
                f"{r['planner_cpu_ms_mean']:.3f} / {r['planner_cpu_ms_max']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["wan21", "wan22"], default="wan21")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--binary", help="path to ditserve binary")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--policies", default=",".join(SUITES["sweep"]))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    binary = find_ditserve_binary(args.binary)
    raw_profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if "phases" in raw_profile:
        profile = raw_profile["phases"][0]["profile"]
    elif "runs" in raw_profile:
        profile = raw_profile["runs"][0]["profile"]
    else:
        profile = raw_profile

    pools = {
        kind: verify_seal(read_json(args.datasets / f"{name}.json"))
        for kind, name in (("image", "diffusiondb"), ("video", "vidprom"))
    }
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    sweeps = {"rate_sweep": [], "slo_sweep": [], "cv_sweep": [], "gpu_scalability": []}

    # 1. Rate Sweep (load in 0.6 .. 1.6)
    for load in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
        for policy in policies:
            sweeps["rate_sweep"].append(
                run_cell(
                    binary, args.model, profile, pools,
                    policy=policy, gpus=8, seed=args.seed, load=load,
                    pattern="gamma", cv=2.0, video_fraction=0.5, count=40,
                    video_slo_scale=1.15, image_slo_scale=1.05,
                )
            )

    # 2. SLO Scale Sweep
    for v_slo, i_slo in ((1.00, 1.03), (1.10, 1.05), (1.15, 1.08), (1.25, 1.15), (1.50, 1.30)):
        for policy in policies:
            sweeps["slo_sweep"].append(
                run_cell(
                    binary, args.model, profile, pools,
                    policy=policy, gpus=8, seed=args.seed, load=1.0,
                    pattern="gamma", cv=2.0, video_fraction=0.5, count=40,
                    video_slo_scale=v_slo, image_slo_scale=i_slo,
                )
            )

    # 3. Burstiness / CV Sweep (cv in 0.5 .. 6.0)
    for cv in (0.5, 1.0, 2.0, 4.0, 6.0):
        for policy in policies:
            sweeps["cv_sweep"].append(
                run_cell(
                    binary, args.model, profile, pools,
                    policy=policy, gpus=8, seed=args.seed, load=1.0,
                    pattern="gamma", cv=cv, video_fraction=0.5, count=40,
                    video_slo_scale=1.15, image_slo_scale=1.05,
                )
            )

    # 4. Cluster & Planner Scalability (Weak scaling: 4, 6, 8, 16, 32, 64 GPUs, 5 requests per GPU)
    for gpus in (4, 6, 8, 16, 32, 64):
        count = 5 * gpus
        for policy in policies:
            sweeps["gpu_scalability"].append(
                run_cell(
                    binary, args.model, profile, pools,
                    policy=policy, gpus=gpus, seed=args.seed, load=1.0,
                    pattern="gamma", cv=2.0, video_fraction=0.5, count=count,
                    video_slo_scale=1.15, image_slo_scale=1.05,
                )
            )

    report = {
        "schema": "ditserve.scalability_study.v1",
        "model": args.model,
        "seed": args.seed,
        "profile_provenance": profile.get("provenance", "unknown"),
        "policies": policies,
        "sweeps": sweeps,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(format_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md) if args.output_md else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
