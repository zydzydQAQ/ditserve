#!/usr/bin/env python3
"""Fit a constrained native ClusterCost profile and report held-out form errors."""

import argparse
import hashlib
import json
import math
from pathlib import Path


FIT_POLICIES = {"static_sp1", "static_cfg2", "static_sp2", "static_cfg2sp2", "static_sp4"}


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        raise ValueError("median requires at least one sample")
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def form_of(policy):
    name = policy.removeprefix("static_")
    if name.startswith("cfg2"):
        rest = name[4:]
        return 2, 1 if rest == "" else int(rest.removeprefix("sp"))
    return 1, int(name.removeprefix("sp"))


def tokens_and_dim(model_or_transformer, vae_or_spec, spec=None):
    if spec is None:
        transformer = model_or_transformer["transformer_config"]
        vae = model_or_transformer["vae_config"]
        spec = vae_or_spec
    else:
        transformer = model_or_transformer
        vae = vae_or_spec
    patch_t, patch_h, patch_w = transformer["patch_size"]
    temporal = vae.get("scale_factor_temporal", 4)
    spatial = vae.get("scale_factor_spatial", 8)
    frames = (spec["frames"] - 1) // temporal + 1
    tokens = ((frames // patch_t) * (spec["height"] // (spatial * patch_h))
              * (spec["width"] // (spatial * patch_w)))
    dim = transformer["num_attention_heads"] * transformer["attention_head_dim"]
    return tokens, dim, transformer["num_layers"], transformer["out_channels"]


def default_sd35_batch_step_ms(
    service_ms: float = 1000.0,
    steps: int = 28,
    resolution: int = 512,
    *,
    prepare_ms: float | None = None,
    finish_ms: float | None = None,
) -> dict[str, float]:
    """Calibrated SD3.5-Medium MM-DiT 1-step forward scaling table on A100-PCIE-40GB (BF16 FlashAttention).

    For <=768px (e.g. 512x512, 256 patch tokens per branch), B=1..8 scales sub-linearly because B=1 under-utilizes SMs.
    For >768px (e.g. 1024x1024 / 1440x1440), B=1..8 still amortizes weight reads, kernel launch, and CFG overhead.
    """
    steps = max(1, int(steps))
    svc = float(service_ms)
    if prepare_ms is None:
        prep_ms = min(45.0, round(svc * 0.05, 3)) if svc < 500.0 else 45.0
    else:
        prep_ms = float(prepare_ms)
    if finish_ms is None:
        fin_ms = (
            min(135.0, max(10.0, round(svc * 0.05, 3)))
            if svc < 500.0
            else min(135.0, max(10.0, round(svc * 0.075, 3)))
        )
    else:
        fin_ms = float(finish_ms)
    b1_step_ms = max(0.5, (svc - prep_ms - fin_ms) / steps)
    if int(resolution) <= 768:
        ratios = {1: 1.0, 2: 1.838, 3: 2.669, 4: 3.567, 6: 5.194, 8: 6.941}
    else:
        ratios = {1: 1.0, 2: 1.940, 3: 2.834, 4: 3.788, 6: 5.636, 8: 7.514}
    return {str(k): round(b1_step_ms * mult, 3) for k, mult in ratios.items()}


def extract_measurements(record):
    cells = {(phase.get("id", "cluster"), cell.get("cell", index)): cell
             for phase in record["phases"] for index, cell in enumerate(phase["cells"])}
    videos, images = [], []
    for row in record["requests"]:
        if row.get("status") != "completed":
            continue
        cell = cells[(row.get("phase", "cluster"), row["cell"])]
        policy = cell.get("policy") or cell["config"]["policy"]
        if row["kind"] == "image":
            size = row.get("batch_size", 1)
            if row.get("model_time_scope") == "amortized_batch_cost" and "batch_size" not in row:
                raise ValueError("image batch measurement lacks actual batch_size")
            if size > 1 and len(set(row.get("batch_request_ids", []))) != size:
                raise ValueError("image batch measurement lacks complete member identity")
            img_spec = row.get("spec") if isinstance(row.get("spec"), dict) else {}
            action_ms = row.get("action_ms") if isinstance(row.get("action_ms"), dict) else {}
            prep_val = row.get("prepare_cb_ms")
            if prep_val is None:
                prep_val = row.get("prepare_model_ms")
            if prep_val is None:
                prep_val = action_ms.get("prepare")
            fin_val = row.get("decode_cb_ms")
            if fin_val is None:
                fin_val = row.get("finish_model_ms")
            if fin_val is None:
                fin_val = action_ms.get("finish")
            images.append({
                "policy": policy,
                "batch_size": size,
                "batch_identity": (row.get("phase", "cluster"), row["cell"],
                                   tuple(row.get("batch_request_ids", [row.get("request_id")]))),
                "service_ms": row["batch_model_ms"] if size > 1 else row["model_ms"],
                "steps": int(img_spec["steps"]) if "steps" in img_spec and img_spec["steps"] is not None else None,
                "resolution": int(img_spec.get("height") or img_spec.get("width") or 1024),
                # Only genuine stage durations (<25% of service_ms). Never read row["finish_ms"],
                # which in study.cpp is the wall-clock completion timestamp now_ms.
                "prepare_ms": prep_val,
                "finish_ms": fin_val,
            })
            continue
        decode_ms = math.fsum(output.get("model_ms", 0.0) for output in row.get("outputs", []))
        steps = row["spec"]["steps"]
        steady = [sample for sample in row.get("advance_samples", []) if sample.get("steady") is True]
        step_ms = median(sample["model_ms"] for sample in steady) if steady else (row["model_ms"] - decode_ms) / steps
        if not math.isfinite(step_ms) or step_ms <= 0:
            raise ValueError("calibration step time must be finite and positive")
        groups = {tuple(sample["group"]) for sample in steady if "group" in sample}
        if len(groups) > 1:
            raise ValueError("static calibration changed placement within a request")
        group = list(next(iter(groups))) if groups and all("group" in sample for sample in steady) else None
        if group is not None and (len(group) != math.prod(form_of(policy)) or len(set(group)) != len(group)):
            raise ValueError("calibration group does not match its parallel form")
        if group is not None and group != list(range(group[0], group[0] + len(group))):
            raise ValueError("calibration placement must use consecutive ranks in topology order")
        encode_meta = row.get("encode_meta") if isinstance(row.get("encode_meta"), dict) else {}
        encode_cached = bool(
            row.get("encode_pool_cached")
            or row.get("encoder_pool_hit")
            or encode_meta.get("pos_cache_hit")
            or encode_meta.get("encoder_pool_hit")
        )
        videos.append({
            "policy": policy,
            "form": form_of(policy),
            "group": group,
            "step_ms": step_ms,
            "steady_samples": len(steady),
            "timing_scope": "median_steady_advance" if steady else "includes_initial_layout_overhead",
            "decode_ms": decode_ms,
            "encode_ms": row.get("encode_model_ms", 0.0),
            "encode_cached": encode_cached,
            "spec": row["spec"],
            "transfer_export_ms": row.get("transfer_export_ms", 0.0),
            "dispatch_gap_ms": row.get("dispatch_gap_ms", 0.0),
            "timed_quanta": row.get("timed_quanta", 0),
        })
    return videos, images


def architecture(model_or_transformer, vae_or_spec, spec=None):
    if spec is None:
        transformer = model_or_transformer["transformer_config"]
        tokens, dim, layers, out_channels = tokens_and_dim(model_or_transformer, vae_or_spec)
    else:
        transformer = model_or_transformer
        tokens, dim, layers, out_channels = tokens_and_dim(model_or_transformer, vae_or_spec, spec)
    patch_volume = math.prod(transformer["patch_size"])
    return {
        "tokens": tokens,
        "dim": dim,
        "layers": layers,
        "comm_mb_per_step": 4.0 * tokens * dim * 2.0 * layers * 2.0 / 1e6,
        "collectives_per_step": layers * 8,
        "cfg_exchange_mb": tokens * patch_volume * out_channels * 2.0 / 1e6,
    }


def predict(step_compute, arch, cfg, sp, bandwidth_inf, gamma, latency):
    compute = step_compute / (cfg * sp)
    if sp == 1:
        communication = 0.0
    else:
        bandwidth = bandwidth_inf * sp / (sp + gamma)
        communication = (arch["comm_mb_per_step"] / cfg) * (sp - 1) / (sp * sp) / bandwidth
        communication += (arch["collectives_per_step"] / cfg) * latency
    exchange = 0.0
    if cfg == 2:
        bandwidth2 = bandwidth_inf * 2 / (2 + gamma)
        exchange = arch["cfg_exchange_mb"] / bandwidth2 + latency
    return compute + communication + exchange


def fit_links(videos, arch):
    sp1 = [row["step_ms"] for row in videos if row["policy"] == "static_sp1"]
    if not sp1:
        raise ValueError("calibration requires an SP1 reference")
    step_compute = median(sp1)
    training = [row for row in videos if row["policy"] in FIT_POLICIES]
    best = None
    for bandwidth_inf in [0.25 * index for index in range(1, 201)]:
        for gamma in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            for latency in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
                error = mean([
                    (predict(step_compute, arch, *row["form"], bandwidth_inf, gamma, latency) - row["step_ms"]) ** 2
                    for row in training
                ])
                if best is None or error < best[0]:
                    best = (error, bandwidth_inf, gamma, latency)
    _, bandwidth_inf, gamma, latency = best
    rows = []
    for row in videos:
        predicted = predict(step_compute, arch, *row["form"], bandwidth_inf, gamma, latency)
        relative = abs(predicted - row["step_ms"]) / row["step_ms"] * 100
        rows.append({
            "policy": row["policy"],
            "form": list(row["form"]),
            "measured_step_ms": row["step_ms"],
            "predicted_step_ms": predicted,
            "relative_error_pct": relative,
            "partition": "fit" if row["policy"] in FIT_POLICIES else "held_out",
        })
    return step_compute, (bandwidth_inf, gamma, latency), rows


def build_profile(record, runtime_facts, model_name):
    if record.get("status") != "completed" or any(row.get("status") != "completed" for row in record["requests"]):
        raise ValueError("calibration requires a completed run with no failed or incomplete requests")
    videos, images = extract_measurements(record)
    if not videos or not images:
        raise ValueError("calibration requires completed video and image requests")
    shape_keys = ("frames", "steps", "height", "width", "guidance_scale")
    if any(any(row["spec"].get(key) != videos[0]["spec"].get(key) for key in shape_keys) for row in videos):
        raise ValueError("calibration mixes incompatible generation shapes")
    model = runtime_facts["models"][model_name]
    arch = architecture(model, videos[0]["spec"])
    step_compute, (bandwidth_inf, gamma, latency), predictions = fit_links(videos, arch)
    held_out = [row["relative_error_pct"] for row in predictions if row["partition"] == "held_out"]
    fit = [row["relative_error_pct"] for row in predictions if row["partition"] == "fit"]
    spec = videos[0]["spec"]
    experiment = record["experiment"]
    model_key = experiment.removeprefix("kairos-").removesuffix("-native-calibration")
    class_name = model_key + "-preflight"
    uncached_encodes = [row["encode_ms"] for row in videos if not row.get("encode_cached")]
    profile = {
        "provenance": "mixed",
        "sp_efficiency": 1.0,
        "sp_overlap": 0.0,
        "resize_ms": mean(row["transfer_export_ms"] for row in videos),
        "quantum_overhead_ms": mean(row["dispatch_gap_ms"] / row["timed_quanta"] for row in videos if row["timed_quanta"]),
        "block_overhead_ms": 0.0,
        "classes": {
            class_name: {
                "kind": "video",
                "steps": spec["steps"],
                "blocks_per_step": 60,
                "prepare_ms": 0.0,
                "encode_ms": mean(uncached_encodes if uncached_encodes else (row["encode_ms"] for row in videos)),
                "step_compute_ms": step_compute,
                "comm_mb_per_step": arch["comm_mb_per_step"],
                "collectives_per_step": arch["collectives_per_step"],
                "cfg_exchange_mb": arch["cfg_exchange_mb"],
                "finish_ms": mean(row["decode_ms"] for row in videos),
                "decode_chunks": (spec["frames"] - 1) // 4 + 1,
            },
            "sd35-preflight": {"kind": "image"},
        },
        "links": {},
    }
    image_batches = {}
    seen_batches = set()
    for row in images:
        if row["batch_identity"] in seen_batches:
            continue
        seen_batches.add(row["batch_identity"])
        image_batches.setdefault(row["batch_size"], []).append(row["service_ms"])
    if 1 not in image_batches:
        raise ValueError("calibration requires unbatched image measurements")
    image_cost = profile["classes"]["sd35-preflight"]
    image_service_ms = float(median(image_batches[1]))
    image_cost["service_ms"] = image_service_ms
    image_provenance = {"service_ms": "measured"}
    img_steps = {row["steps"] for row in images if row.get("steps") is not None}
    if len(img_steps) == 1:
        image_steps = int(next(iter(img_steps)))
        image_provenance["steps"] = "request_spec"
    else:
        image_steps = 28
        image_provenance["steps"] = "default"
    image_cost["steps"] = image_steps
    img_resolutions = [int(row.get("resolution") or 1024) for row in images]
    image_resolution = int(median(img_resolutions)) if img_resolutions else 1024

    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0

    prep_samples = [
        float(row["prepare_ms"])
        for row in images
        if positive(row.get("prepare_ms")) and float(row["prepare_ms"]) < image_service_ms * 0.25
    ]
    fin_samples = [
        float(row["finish_ms"])
        for row in images
        if positive(row.get("finish_ms")) and float(row["finish_ms"]) < image_service_ms * 0.25
    ]
    cb_step_samples = {}
    for phase in record.get("phases", []):
        for cell in phase.get("cells", []):
            for sample in (cell.get("continuous_batch") or {}).get("samples", []):
                bsize = sample.get("batch_size")
                if (isinstance(bsize, int) and not isinstance(bsize, bool) and 1 <= bsize <= 8
                        and positive(sample.get("model_ms"))):
                    cb_step_samples.setdefault(bsize, []).append(float(sample["model_ms"]))
    if 1 in cb_step_samples:
        image_prepare_ms = (
            round(median(prep_samples), 3)
            if prep_samples
            else (min(45.0, round(image_service_ms * 0.05, 3)) if image_service_ms < 500.0 else 45.0)
        )
        image_finish_ms = (
            round(median(fin_samples), 3)
            if fin_samples
            else min(135.0, max(10.0, round(image_service_ms * 0.075, 3)))
        )
        image_cost["prepare_ms"] = image_prepare_ms
        image_cost["finish_ms"] = image_finish_ms
        image_provenance["prepare_ms"] = (
            "measured_continuous_batch_stage" if prep_samples else "calibrated_stage_split"
        )
        image_provenance["finish_ms"] = (
            "measured_continuous_batch_stage" if fin_samples else "calibrated_stage_split"
        )
        image_cost["batch_step_ms"] = {str(size): round(median(values), 3)
                                       for size, values in sorted(cb_step_samples.items())}
        image_provenance["batch_step_ms"] = "measured_continuous_batch_rounds"
    elif cb_step_samples:
        if prep_samples:
            image_cost["prepare_ms"] = round(median(prep_samples), 3)
            image_provenance["prepare_ms"] = "measured_continuous_batch_stage"
        if fin_samples:
            image_cost["finish_ms"] = round(median(fin_samples), 3)
            image_provenance["finish_ms"] = "measured_continuous_batch_stage"
        # The executor requires a batch-1 entry; never pad it with a synthetic value.
        image_provenance["batch_step_ms"] = "omitted_no_batch1_rounds"
    else:
        image_prepare_ms = (
            round(median(prep_samples), 3)
            if prep_samples
            else (min(45.0, round(image_service_ms * 0.05, 3)) if image_service_ms < 500.0 else 45.0)
        )
        target_finish_ms = (
            round(median(fin_samples), 3)
            if fin_samples
            else min(135.0, max(10.0, round(image_service_ms * 0.075, 3)))
        )
        batch_step_ms = default_sd35_batch_step_ms(
            image_service_ms,
            steps=image_steps,
            resolution=image_resolution,
            prepare_ms=image_prepare_ms,
            finish_ms=target_finish_ms,
        )
        image_finish_ms = (
            target_finish_ms
            if fin_samples
            else round(max(10.0, round(image_service_ms, 3) - image_prepare_ms - round(image_steps * batch_step_ms["1"], 3)), 3)
        )
        image_cost["prepare_ms"] = image_prepare_ms
        image_cost["finish_ms"] = image_finish_ms
        image_cost["batch_step_ms"] = batch_step_ms
        image_provenance["prepare_ms"] = (
            "measured_continuous_batch_stage" if prep_samples else "calibrated_stage_split"
        )
        image_provenance["finish_ms"] = (
            "measured_continuous_batch_stage" if fin_samples else "calibrated_stage_split"
        )
        image_provenance["batch_step_ms"] = "calibrated_sublinear_scaling_from_measured_service_ms"
    if any(size > 1 for size in image_batches):
        image_cost["batch_ms"] = {str(size): median(values) for size, values in image_batches.items()}
    placements = {}
    for row in videos:
        if row["group"] is not None:
            key = (row["policy"].removeprefix("static_"), f'{row["group"][0]}-{row["group"][-1]}')
            placements.setdefault(key, []).append(row["step_ms"])
    forms = {}
    for (form, group), values in placements.items():
        forms.setdefault(form, {"placements": {}})["placements"][group] = {"step_ms": median(values)}
    if forms:
        profile["classes"][class_name]["forms"] = forms
    for degree in (2, 3, 4, 6, 8):
        profile["links"][str(degree)] = {
            "gbps": bandwidth_inf * degree / (degree + gamma),
            "latency_ms": latency,
        }
    sessions = record.get("budget", {}).get("sessions", [])
    placements = [list(session["identity"]["gpu_uuids"]) for session in sessions]
    if placements and any(placement != placements[0] for placement in placements[1:]):
        raise ValueError("calibration changed GPU placement between sessions")
    profile["source"] = {
        "experiment": experiment,
        "run_id": record["run_id"],
        "gpu_uuids": placements[0] if placements else [],
        "run_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
        "model": model_name,
        "shape": spec,
        "architecture": arch,
        "timing_scopes": {row["policy"]: row["timing_scope"] for row in videos},
        "steady_samples": {row["policy"]: row["steady_samples"] for row in videos},
        "bandwidth_saturation_gbps": bandwidth_inf,
        "bandwidth_gamma": gamma,
        "fit_policies": sorted(FIT_POLICIES),
        "fit_mean_relative_error_pct": mean(fit),
        "held_out_mean_relative_error_pct": mean(held_out) if held_out else None,
        "held_out_max_relative_error_pct": max(held_out) if held_out else None,
        "predictions": predictions,
        "execution_cost_mode": "placement_bound_form_table_with_analytical_fallback" if forms else "analytical_fallback_only",
        "unbound_measurements": sum(row["group"] is None for row in videos),
        "encode_uncached_samples": len(uncached_encodes),
        "encode_cached_samples": sum(bool(row.get("encode_cached")) for row in videos),
        "image_batch_samples": {str(size): len(values) for size, values in image_batches.items()},
        "image_cost_provenance": image_provenance,
        "continuous_batch_round_samples": {str(size): len(values) for size, values in sorted(cb_step_samples.items())},
        "validation_scope": "held-out forms on calibration hardware and shape; not held-out request, placement or workload validation",
        "method": "single constrained saturation model fitted once; no per-form bandwidth or overlap overrides",
    }
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime-facts", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.run.read_text(encoding="utf-8"))
    facts = json.loads(args.runtime_facts.read_text(encoding="utf-8"))
    profile = build_profile(record, facts, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "source": profile["source"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
