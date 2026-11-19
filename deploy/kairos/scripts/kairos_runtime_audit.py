import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path



def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_runtime():
    import torch
    run_id = os.environ["DITSERVE_RUN_ID"]
    output = Path("/shared/output/DITSERVE/acceptance") / run_id / "runtime-facts.json"
    models = Path("/shared/data/DITSERVE/models")
    model_roots = {
        "Wan2.1-T2V-1.3B": models / "wan",
        "Wan2.2-TI2V-5B": models / "wan2.2-ti2v-5b",
        "SD3.5-Medium": models / "sd3.5-medium",
    }
    if os.environ.get('DITSERVE_MODEL_MANIFESTS'):
        manifest_mapping = json_file(os.environ['DITSERVE_MODEL_MANIFESTS'])
        for key, name in (('wan21', 'Wan2.1-T2V-1.3B'), ('wan22', 'Wan2.2-TI2V-5B'), ('sd35', 'SD3.5-Medium')):
            if manifest_mapping.get(key, {}).get('model_path'):
                selected = Path(manifest_mapping[key]['model_path'])
                if '..' in selected.parts or not selected.is_relative_to(models):
                    raise ValueError('model snapshot escapes DITSERVE model directory')
                model_roots[name] = selected
    packages = {}
    for name in ("torch", "diffusers", "transformers", "cache-dit", "accelerate", "safetensors", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "schema": "ditserve.runtime-audit.v1",
        "run_id": run_id,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "nvidia_smi": command(
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ).splitlines(),
        "topology": command("nvidia-smi", "topo", "-m"),
        "cuda": {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "models": {},
    }
    for name, root in model_roots.items():
        record["models"][name] = {
            "root": str(root),
            "model_index": json_file(root / "model_index.json"),
            "transformer_config": json_file(root / "transformer" / "config.json"),
            "vae_config": json_file(root / "vae" / "config.json"),
            "scheduler_config": json_file(root / "scheduler" / "scheduler_config.json"),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "gpus": record["nvidia_smi"],
        "packages": packages,
        "models": {name: {
            "model_class": value["model_index"].get("_class_name"),
            "transformer_class": value["transformer_config"].get("_class_name"),
            "num_layers": value["transformer_config"].get("num_layers"),
            "num_attention_heads": value["transformer_config"].get("num_attention_heads"),
        } for name, value in record["models"].items()},
    }, indent=2))


def add_memory_sample(summary, line):
    import csv
    import math
    row = next(csv.reader([line]))
    if len(row) != 6:
        raise ValueError("expected six nvidia-smi fields")
    stamp, uuid = row[0].strip(), row[1].strip()
    used, free, total, utilization = [float(value.strip()) for value in row[2:]]
    if not all(math.isfinite(value) and value >= 0 for value in (used, free, total, utilization)):
        raise ValueError("invalid GPU measurement")
    item = summary.setdefault(uuid, dict(samples=0, first_timestamp=stamp,
        memory_used_max_mib=used, memory_free_min_mib=free, memory_total_mib=total,
        utilization_mean_pct=0.0))
    item["samples"] += 1
    item["last_timestamp"] = stamp
    item["memory_used_max_mib"] = max(item["memory_used_max_mib"], used)
    item["memory_free_min_mib"] = min(item["memory_free_min_mib"], free)
    item["utilization_mean_pct"] += (utilization - item["utilization_mean_pct"]) / item["samples"]


def memory_summary(output):
    """Keep bounded aggregates, never a CSV trace; flush periodically and on stop."""
    import signal
    import time
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(schema="ditserve.memory-summary.v1", cadence_ms=100, gpus={},
        invalid_samples=0, scope="sampled NVML extrema; not allocator peaks")
    proc = subprocess.Popen(["nvidia-smi",
        "--query-gpu=timestamp,uuid,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        proc.terminate()
    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
        temporary.replace(output)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last = time.monotonic()
    try:
        for line in proc.stdout:
            try:
                add_memory_sample(summary["gpus"], line)
            except (ValueError, TypeError):
                summary["invalid_samples"] += 1
            if time.monotonic() - last >= 30:
                save()
                last = time.monotonic()
    finally:
        proc.terminate() if proc.poll() is None else None
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        summary["status"] = "completed" if stopping and summary["gpus"] else "incomplete"
        save()
    if not stopping or not summary["gpus"]:
        raise SystemExit("GPU sampler exited unexpectedly or produced no valid samples")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-summary", type=Path)
    args = parser.parse_args()
    memory_summary(args.memory_summary) if args.memory_summary else audit_runtime()
