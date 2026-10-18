"""Cluster cost profiles from measured Wan/Sana operations, plus labeled extrapolation.

``measured_classes`` reads the explicit-action run of
``configs/studies/local-a100/e2_cost_profile.json`` (its ``run.json``) and
returns per-shape SP1 costs. ``extrapolate`` rescales SP1 step compute to
another Wan architecture by the analytic FLOP ratio at equal utilization, and
``ulysses_comm_mb`` gives the full-sequence all-to-all volume per solver step.
Extrapolated classes are never measurements; profiles carry their provenance.
"""

import json
import statistics
from pathlib import Path

# Published Diffusers transformer configs; 14B is not on the local disk.
ARCHITECTURES = {
    "wan2.1-t2v-1.3b": dict(dim=1536, ffn=8960, layers=30, text_tokens=512, verified="local transformer/config.json"),
    "wan2.1-t2v-14b": dict(dim=5120, ffn=13824, layers=40, text_tokens=512, verified="published config; re-check on arrival"),
}
CFG_BRANCHES = 2


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


def measured_classes(run_json, *, wan_worker="wan", sana_worker="sana", steps=None, blocks_per_step=60):
    """SP1 per-shape costs from the E2 explicit-action run; n is per sample."""
    record = json.loads(Path(run_json).read_text())
    shapes, image = {}, dict(service=[], finish=[])
    cells = [cell for phase in record["phases"] for cell in phase.get("cells", [])]
    for cell in cells:
                for operation in cell.get("operations", []):
                    action, result = operation["action"], operation["ranks"][0]
                    label = action.get("label")
                    sid = action.get("state_id", "")
                    if sid.startswith("img"):
                        if label == "image":
                            image["service"].append(result["model_ms"])
                        elif label == "image_finish":
                            image["finish"].append(result.get("model_ms", 0.))
                        continue
                    row = shapes.setdefault(sid, dict(step=[], block=[], blocks4=[]))
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
                    elif label == "block4":
                        row["blocks4"].append(result["model_ms"])
                    elif label == "inventory":
                        row["inventory"] = result
                    elif label == "offload":
                        row["offload_ms"], row["offload_bytes"] = result.get("ms"), result.get("transfer_bytes")
                    elif label == "restore":
                        row["restore_ms"] = result.get("ms")
                    elif label == "decode":
                        row["finish_ms"] = result["model_ms"]
    classes = {}
    for sid, row in shapes.items():
        spec = row["spec"]
        name = "wan1.3b-%dx%dx%d" % (spec["height"], spec["width"], spec["frames"])
        classes[name] = dict(kind="video", steps=steps or spec["steps"], blocks_per_step=blocks_per_step,
                             prepare_ms=row.get("prepare_ms", 0.), encode_ms=row.get("encode_ms", 0.), step_compute_ms=_median(row["step"]),
                             comm_mb_per_step=ulysses_comm_mb("wan2.1-t2v-1.3b", wan_tokens(spec["height"], spec["width"], spec["frames"])),
                             collectives_per_step=4 * ARCHITECTURES["wan2.1-t2v-1.3b"]["layers"] * CFG_BRANCHES,
                             finish_ms=row["finish_ms"],
                             evidence=dict(step_samples=row["step"], block4_samples=row["blocks4"],
                                           block_cuda_ms_median=_median(row["block"]) if row["block"] else None,
                                           offload_ms=row.get("offload_ms"), offload_bytes=row.get("offload_bytes"),
                                           restore_ms=row.get("restore_ms"), shape=[spec["height"], spec["width"], spec["frames"]],
                                           tokens=wan_tokens(spec["height"], spec["width"], spec["frames"])))
    if image["service"]:
        classes["sana-1024"] = dict(kind="image", service_ms=_median(image["service"]) + _median(image["finish"] or [0.]),
                                    evidence=dict(service_samples=image["service"], finish_samples=image["finish"]))
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


# PCIe all-to-all per-rank bandwidth scenarios (GB/s) until the target is measured.
LINK_SCENARIOS = {
    "p2p_switch": {"2": dict(gbps=20, latency_ms=0.03), "4": dict(gbps=12, latency_ms=0.04), "8": dict(gbps=6, latency_ms=0.06)},
    "optimistic": {"2": dict(gbps=22, latency_ms=0.02), "4": dict(gbps=18, latency_ms=0.03), "8": dict(gbps=12, latency_ms=0.04)},
    "host_staged": {"2": dict(gbps=10, latency_ms=0.05), "4": dict(gbps=6, latency_ms=0.06), "8": dict(gbps=3, latency_ms=0.08)},
}


def profile(classes, *, links, provenance, sp_efficiency=0.95, resize_ms=50., source=None):
    return dict(provenance=provenance, classes=classes, links=links, sp_efficiency=sp_efficiency,
                resize_ms=resize_ms, source=source or {})
