"""Cluster cost profiles from a measured cost_profile run, plus labeled extrapolation.

    python -m ditserve.evaluation.cost_profile --run-json WORK/run.json \\
        --hardware "2x RTX 4090 24GB" [--links host_staged|collectives.json] \\
        --out configs/profiles/rtx4090-24g.json

``measured_classes`` reads the explicit-action run of
``configs/studies/cost_profile.json`` (its ``run.json`` in the work directory,
which survives result compaction) and returns per-shape SP1 costs and peak
memory. ``extrapolate`` rescales SP1 step compute to another Wan architecture
by the analytic FLOP ratio at equal utilization, and ``ulysses_comm_mb`` gives
the full-sequence all-to-all volume per solver step. Extrapolated classes are
never measurements; profiles carry their provenance.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

# Published Diffusers transformer configs; 14B is not on the local disk.
ARCHITECTURES = {
    "wan2.1-t2v-1.3b": dict(dim=1536, ffn=8960, layers=30, text_tokens=512, verified="local transformer/config.json"),
    "wan2.1-t2v-14b": dict(dim=5120, ffn=13824, layers=40, text_tokens=512, verified="published config; re-check on arrival"),
}
CFG_BRANCHES = 2
LINK_SCENARIOS = Path(__file__).resolve().parents[3] / "configs/profiles/link-scenarios.json"


def wan_tokens(height, width, frames):
    """VAE 4x temporal / 8x spatial compression, then 1x2x2 patches."""
    if height % 16 or width % 16 or (frames - 1) % 4:
        raise ValueError("Wan shape must be 16-aligned with 4k+1 frames")
    return ((frames - 1) // 4 + 1) * (height // 16) * (width // 16)


def step_flops(architecture, tokens):
    """Dense multiply-adds x2 for one solver step (both CFG branches)."""
    arch = ARCHITECTURES[architecture]
    d, ffn = arch["dim"], arch["ffn"]
    linear = 6 * d * d + 2 * d * ffn                      # self q/k/v/o, cross q/o, FFN
    attention = 2 * tokens * d + 2 * arch["text_tokens"] * d  # QK^T and AV, self + cross
    return 2 * CFG_BRANCHES * arch["layers"] * tokens * (linear + attention)


def ulysses_comm_mb(architecture, tokens, bytes_per_element=2):
    """Q, K, V and output all-to-all of the full sequence per layer and branch."""
    arch = ARCHITECTURES[architecture]
    return 4 * tokens * arch["dim"] * bytes_per_element * arch["layers"] * CFG_BRANCHES / 1e6


def _median(values):
    if not values:
        raise ValueError("no measured samples")
    return statistics.median(values)


def _memory(result):
    return {key: result[key] for key in ("peak_allocated_bytes", "peak_reserved_bytes", "device_total_bytes") if key in result}


def measured_classes(run_json, *, steps=None, blocks_per_step=60):
    """SP1 per-shape costs from a cost_profile run; n is per sample."""
    record = json.loads(Path(run_json).read_text())
    shapes, image = {}, dict(service=[], finish=[], memory=None)
    cells = [cell for phase in record["phases"] for cell in phase.get("cells", [])]
    for cell in cells:
        current = None
        for operation in cell.get("operations", []):
            action, result = operation["action"], operation["ranks"][0]
            label, sid = action.get("label"), action.get("state_id", "")
            if action["op"] == "memory":
                if label == "memory" and current is not None:
                    current["memory"] = _memory(result)
                continue
            if sid.startswith("img"):
                current = image
                if label == "image":
                    image["service"].append(result["model_ms"])
                elif label == "image_finish":
                    image["finish"].append(result.get("model_ms", 0.))
                continue
            row = current = shapes.setdefault(sid, dict(step=[], block=[], blocks4=[]))
            if action["op"] == "prepare":
                row["spec"] = action["spec"]
                # Synchronous prepare encodes on CPU first; the cluster
                # executor prefetches that part off the GPU.
                row["prepare_ms"] = result.get("gpu_prepare_ms", result.get("model_ms"))
                row["encode_ms"] = result.get("cpu_prompt_encode_ms", 0.)
            elif label == "step":
                row["step"].append(result["model_ms"])
                for event in result.get("events", []):
                    row["block"].extend(item["ms"] for item in event.get("block_cuda_ms", []))
                    row["extra_state_bytes"] = max(row.get("extra_state_bytes", 0), event.get("extra_state_bytes", 0))
            elif label == "block4":
                row["blocks4"].append(result["model_ms"])
                for event in result.get("events", []):
                    row["extra_state_bytes"] = max(row.get("extra_state_bytes", 0), event.get("extra_state_bytes", 0))
            elif label == "decode":
                row["finish_ms"] = result["model_ms"]
    classes = {}
    for sid, row in shapes.items():
        spec = row["spec"]
        name = "wan1.3b-%dx%dx%d" % (spec["height"], spec["width"], spec["frames"])
        tokens = wan_tokens(spec["height"], spec["width"], spec["frames"])
        classes[name] = dict(kind="video", steps=steps or spec["steps"], blocks_per_step=blocks_per_step,
                             prepare_ms=row.get("prepare_ms", 0.), encode_ms=row.get("encode_ms", 0.),
                             step_compute_ms=_median(row["step"]),
                             comm_mb_per_step=ulysses_comm_mb("wan2.1-t2v-1.3b", tokens),
                             collectives_per_step=4 * ARCHITECTURES["wan2.1-t2v-1.3b"]["layers"] * CFG_BRANCHES,
                             finish_ms=row["finish_ms"], decode_chunks=(spec["frames"] - 1) // 4 + 1,
                             evidence=dict(step_samples=row["step"], block4_samples=row["blocks4"],
                                           block_cuda_ms_median=_median(row["block"]) if row["block"] else None,
                                           block_extra_state_bytes_max=row.get("extra_state_bytes"),
                                           memory=row.get("memory"), shape=[spec["height"], spec["width"], spec["frames"]],
                                           tokens=tokens))
    if image["service"]:
        classes["sana-1024"] = dict(kind="image", service_ms=_median(image["service"]) + _median(image["finish"] or [0.]),
                                    evidence=dict(service_samples=image["service"], finish_samples=image["finish"],
                                                  memory=image["memory"]))
    return classes


def extrapolate(measured, source="wan2.1-t2v-1.3b", target="wan2.1-t2v-14b", name_prefix="wan14b-x"):
    """Rescale SP1 step compute by FLOPs at equal utilization; VAE/text costs are shared."""
    result = {}
    for name, row in measured.items():
        if row["kind"] != "video":
            continue
        height, width, frames = row["evidence"]["shape"]
        tokens = wan_tokens(height, width, frames)
        ratio = step_flops(target, tokens) / step_flops(source, tokens)
        result[name_prefix + "%dx%dx%d" % (height, width, frames)] = dict(
            row, step_compute_ms=row["step_compute_ms"] * ratio,
            blocks_per_step=2 * ARCHITECTURES[target]["layers"],
            comm_mb_per_step=ulysses_comm_mb(target, tokens),
            collectives_per_step=4 * ARCHITECTURES[target]["layers"] * CFG_BRANCHES,
            evidence=dict(source_class=name, flop_ratio=ratio, assumption="same utilization as the measured 1.3B step"))
    return result


def links_from_collectives(report):
    """Per-degree links from a collectives run: the slowest concurrent group at
    the largest buffer, and latency as the small-buffer time beyond transfer."""
    links = {}
    rows = [row for row in report["rows"] if row["mode"] == "concurrent"]
    for degree in sorted({row["degree"] for row in rows}):
        mine = [row for row in rows if row["degree"] == degree]
        largest, smallest = max(row["buffer_mb"] for row in mine), min(row["buffer_mb"] for row in mine)
        gbps = min(row["per_rank_gbps"] for row in mine if row["buffer_mb"] == largest)
        small = [row for row in mine if row["buffer_mb"] == smallest]
        latency = max(max(row["ms_median"] - row["per_rank_send_mb"] / 1024 / gbps * 1000 for row in small), 0.01)
        links[str(degree)] = dict(gbps=gbps, latency_ms=latency)
    return links


def profile(classes, *, links, provenance, sp_efficiency=1.0, resize_ms=0., source=None):
    return dict(provenance=provenance, classes=classes, links=links, sp_efficiency=sp_efficiency,
                resize_ms=resize_ms, source=source or {})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-json", type=Path, required=True, help="run.json of a cost_profile study work directory")
    parser.add_argument("--hardware", required=True, help="e.g. '2x RTX 4090 24GB'")
    parser.add_argument("--links", help="a scenario in configs/profiles/link-scenarios.json or a collectives.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    classes = measured_classes(args.run_json)
    links, provenance, link_source = {}, "measured", None
    if args.links:
        path = Path(args.links)
        if path.suffix == ".json" and path.is_file():
            links, link_source = links_from_collectives(json.loads(path.read_text())), str(path)
        else:
            links, provenance, link_source = json.loads(LINK_SCENARIOS.read_text())[args.links], "mixed", "scenario " + args.links
    record = json.loads(args.run_json.read_text())
    gpu_seconds = sum(session.get("gpu_seconds", 0) for session in record.get("budget", {}).get("sessions", []))
    source = dict(run=str(args.run_json), run_json_sha256=hashlib.sha256(args.run_json.read_bytes()).hexdigest(),
                  experiment=record.get("experiment"), hardware=args.hardware, links=link_source,
                  environment=record.get("config", {}).get("environment", {}).get("name"), gpu_seconds=gpu_seconds)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(profile(classes, links=links, provenance=provenance, source=source), stream, indent=2)
        stream.write("\n")
    print(json.dumps({name: dict(step_ms=row.get("step_compute_ms"), finish_ms=row.get("finish_ms"),
                                 service_ms=row.get("service_ms"), memory=row["evidence"].get("memory"))
                      for name, row in classes.items()}, indent=2))


if __name__ == "__main__":
    main()
