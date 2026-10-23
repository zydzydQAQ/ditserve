"""Freeze source and create normal/timeline/kernel jobs for the 4090 study.

Preparation does not start containers or reserve GPUs. Run each generated job
file through run-workflow-suite.py. Normal jobs precede timeline jobs; profiler
wall times are deliberately kept in a separate evidence subtree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def jobs(root, source, container_root, nsys):
    python = "/opt/venv/bin/python"
    normal, traced = [], []
    for phase in ("T0", "T1", "T2", "T3", "T4", "T5"):
        common = [python, "-m", "ditserve.evaluation.workflow_encoder",
                  "--model-path", "/models/wan2.1-t2v-1.3b", "--device", "cuda:1",
                  "--frames", "17,33", "--steps", "20", "--warmups", "1"]
        normal.append(dict(id=phase + "-encoding", command=common + ["--output", container_root + "/encoder",
            "--phases", phase, "--repeats", "5", "--pipeline-requests", "4" if phase == "T5" else "0"],
            devices=[0, 1] if phase == "T5" else [1], gpu_count=2 if phase == "T5" else 1,
            timeout_seconds=3600 if phase == "T5" else 1800))
        prefix = container_root + "/traces/" + phase
        workload = common + ["--output", container_root + "/profiling/encoder", "--child", phase,
                             "--repeats", "1", "--pipeline-requests", "4" if phase == "T5" else "0",
                             "--queue-repeats", "1"]
        traced.append(dict(id="trace-" + phase, command=[nsys, "profile", "--trace=cuda,nvtx,osrt",
            "--sample=none", "--cpuctxsw=none", "--gpu-metrics-devices=all" if phase == "T5" else "--gpu-metrics-devices=1",
            "--gpu-metrics-frequency=1000", "--output=" + prefix] + workload,
            counters=True, devices=[0, 1] if phase == "T5" else [1], gpu_count=2 if phase == "T5" else 1,
            timeout_seconds=1800))
    # CPU reference first; then six paired overlap experiments in requested order.
    overlap = []
    for experiment in ("E1", "E2", "E3", "E4", "E5", "E6"):
        for frames in (17, 33):
            identifier = experiment + "-f" + str(frames)
            common = [python, "-m", "ditserve.evaluation.workflow_overlap", "--experiment", experiment,
                      "--frames", str(frames), "--steps", "20", "--requests", "4", "--device", "cuda:1"]
            devices = [0, 1] if experiment in ("E2", "E6") else [1]
            overlap.append(dict(id=identifier, command=common + ["--repeats", "3", "--out", container_root + "/overlap/" + identifier],
                                devices=devices, gpu_count=len(devices), timeout_seconds=5400))
            traced.append(dict(id="trace-" + identifier, command=[nsys, "profile", "--trace=cuda,nvtx,osrt",
                "--sample=none", "--cpuctxsw=none", "--gpu-metrics-devices=all" if len(devices) == 2 else "--gpu-metrics-devices=1",
                "--gpu-metrics-frequency=1000", "--output=" + container_root + "/traces/" + identifier] + common +
                ["--repeats", "1", "--out", container_root + "/profiling/overlap/" + identifier],
                counters=True, devices=devices, gpu_count=len(devices), timeout_seconds=3600))
    common = dict(root=str(root), source=str(source))
    return dict(**common, jobs=normal[:1] + overlap + normal[1:]), dict(**common, jobs=traced)


def ncu_jobs(root, source, container_root, ncu="/opt/nvidia/nsight-compute/2024.1.0/ncu"):
    """Sample up to 128 kernels in one measured range, never whole-stage occupancy."""
    workers = str(Path(__file__).resolve().parents[1] / "workers")
    if workers not in sys.path:
        sys.path.insert(0, workers)
    from ditserve.evaluation.workflow_ncu import profile_options

    selections = [("T1", "text_encoder", 17, None), ("T2", "text_encoder", 17, None)]
    selections += [("T0", stage, frames, 1 if stage == "dit_step" else None)
                   for stage in ("dit_step", "vae_decode") for frames in (17, 33)]
    sampled = []
    for variant, stage, frames, step in selections:
        identifier = "ncu-" + variant + "-" + stage + ("-f" + str(frames) if variant == "T0" else "")
        purpose = "encoder_measurement" if stage == "text_encoder" else "video_measurement"
        options = profile_options(stage, variant=variant, purpose=purpose, repeat=0,
                                  step=step, launch_count=128)
        workload = ["/opt/venv/bin/python", "-m", "ditserve.evaluation.workflow_encoder",
                    "--model-path", "/models/wan2.1-t2v-1.3b", "--device", "cuda:1",
                    "--frames", str(frames), "--steps", "20", "--warmups", "1",
                    "--repeats", "1", "--pipeline-requests", "0", "--child", variant,
                    "--output", container_root + "/profiling/ncu/" + identifier + "/encoder"]
        sampled.append(dict(id=identifier,
            command=[ncu, *options, "-o", container_root + "/ncu/" + identifier, *workload],
            counters=True, devices=[1], gpu_count=1, timeout_seconds=1800))
    return dict(root=str(root), source=str(source), jobs=sampled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/ssd0/zyd/ditserve-data"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nsys", default="/data/tools/nsight-workflow/extracted/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("run-id must be a simple directory name")
    repo = Path(__file__).resolve().parents[1]
    root = args.data_root.resolve() / "workflow" / args.run_id
    if root.exists():
        parser.error("refusing to overwrite an existing study: " + str(root))
    source = root / "sources" / "frozen"
    source.mkdir(parents=True)
    for name in ("workers", "configs"):
        shutil.copytree(repo / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob("*")) if path.is_file()}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    write(source / "manifest.json", dict(git_head=head, files=hashes))
    normal, traced = jobs(root, source, "/data/workflow/" + args.run_id, args.nsys)
    sampled = ncu_jobs(root, source, "/data/workflow/" + args.run_id)
    write(root / "jobs.json", normal)
    write(root / "jobs-trace.json", traced)
    write(root / "jobs-ncu.json", sampled)
    (root / "traces").mkdir()
    (root / "ncu").mkdir()
    print(json.dumps(dict(root=str(root), normal_jobs=len(normal["jobs"]), trace_jobs=len(traced["jobs"]),
                         ncu_jobs=len(sampled["jobs"]))))


if __name__ == "__main__":
    main()
