"""GPU numerical qualification only; the external suite owns guards and budget.

Each arm loads its model in a separate process. Hashing every Wan solver step
perturbs execution, so neither these times nor warmup/load times are performance
evidence. Importing this module and validating its configuration are CPU-only.
"""

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

SCHEMA = "ditserve.workflow.encoding-qualification.v1"
ARMS = {"sana": ("disabled", "staged", "fused"), "wan": ("disabled", "enabled")}


def normalize_config(kind, config):
    if kind not in ARMS or not isinstance(config, dict):
        raise ValueError("kind must be sana or wan and config must be an object")
    value = dict(kind=kind, device="cuda:0", cpu_threads=8, seed=20261024,
                 sample_interval=.05, arm_timeout_seconds=1800)
    allowed = set(value) | {"model_path", "gpu_uuid", "cpu_mask", "prompt_distribution", "guidance_scale"}
    if set(config) - allowed:
        raise ValueError("unknown qualification configuration fields")
    value.update(config)
    if value["kind"] != kind or value["device"] != "cuda:0" or type(value["cpu_threads"]) is not int or value["cpu_threads"] != 8:
        raise ValueError("qualification requires matching kind, cuda:0 and eight physical CPU cores")
    mask = value.get("cpu_mask")
    if not isinstance(mask, list) or len(mask) != 8 or any(type(x) is not int or x < 0 for x in mask) or len(set(mask)) != 8:
        raise ValueError("cpu_mask must contain eight distinct physical CPU IDs")
    if not isinstance(value.get("gpu_uuid"), str) or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value["gpu_uuid"]):
        raise ValueError("gpu_uuid must identify the physical GPU guarded by the external suite")
    if not isinstance(value.get("model_path"), str) or not Path(value["model_path"]).is_absolute():
        raise ValueError("model_path must be an absolute local snapshot path")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**63 - 8:
        raise ValueError("seed must be a bounded nonnegative integer")
    for name in ("sample_interval", "arm_timeout_seconds"):
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(name + " must be positive and finite")
    if kind == "sana":
        if value.get("prompt_distribution") not in ("unique", "repeated") or "guidance_scale" in value:
            raise ValueError("Sana requires unique or repeated prompt_distribution")
    elif (type(value.get("guidance_scale")) not in (int, float) or value["guidance_scale"] not in (1., 5.)
          or "prompt_distribution" in value):
        raise ValueError("Wan qualification guidance_scale must be 1 or 5")
    return json.loads(json.dumps(value, allow_nan=False))


def cases(kind, config):
    """Frozen eight-request matrix; callers cannot remove difficult cases."""
    config = normalize_config(kind, config)
    rows = []
    if kind == "sana":
        for index in range(8):
            prompt_index = index if config["prompt_distribution"] == "unique" else index % 4
            rows.append(dict(request_id="image-%d" % index, spec=dict(
                prompt="A red boat on a calm lake, composition number %d." % prompt_index,
                seed=config["seed"] + index)))
    else:
        for frames in (17, 33):
            for negative in ("", "blurry, low quality, distorted geometry"):
                for repeat in range(2):
                    index = len(rows)
                    key = "video-%d" % index
                    rows.append(dict(request_id=key, spec=dict(request_id=key,
                        prompt="A red boat moves across a calm lake, view number %d." % index, negative_prompt=negative,
                        seed=config["seed"] + index, height=480, width=832, frames=frames,
                        steps=20, guidance_scale=config["guidance_scale"])))
    return rows


def _options(kind, arm):
    if arm == "disabled":
        return {}
    if kind == "sana":
        return dict(prompt_cache_entries=4, prompt_cache_bytes=16 * 1024**2, prompt_cache_sync_mode=arm)
    return dict(negative_prompt_cache_entries=4, negative_prompt_cache_bytes=64 * 1024**2)


def _write(path, value):
    from .workflow_budget import atomic_json
    atomic_json(path, value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fingerprint(value):
    if value is None:
        return None
    from ..models.base import artifact
    return _output_fingerprint(artifact(value, mode="digest"))


def _output_fingerprint(value):
    return dict(sha256=value["array_sha256"], shape=value["shape"], dtype=value["dtype"], finite=value["finite"])


def _create_backend(kind, config, record, out):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise ValueError("external GPU guard must bind CUDA_VISIBLE_DEVICES to gpu_uuid")
    from .workflow_serving_encoder import _runtime
    torch = _runtime(dict(config, cudnn_tf32=True), record, gpu=True)
    record["runtime"]["intraop_threads"] = torch.get_num_threads()
    if kind == "sana":
        from ..models.sana.backend import SanaBackend as Backend
    else:
        from ..models.wan.backend import WanBackend as Backend
    return Backend(dict(model_path=config["model_path"], device=config["device"], work_dir=str(out),
                        attention="flash", options=_options(kind, record["arm"])))


def _execute_request(kind, backend, case, config):
    from .workflow_metrics import NvmlSampler
    from .workflow_serving_overlap import memory_margin_gate
    row = dict(case, status="running", cpu_embeddings=None, step_latents=[], outputs={}, cache=None)
    request = dict(id=0, state_id=case["request_id"], spec=case["spec"], output_mode="check")
    start = time.perf_counter_ns()
    sampler = NvmlSampler(devices=(config["gpu_uuid"],), interval_s=config["sample_interval"]).start()
    try:
        if kind == "wan":
            ticket = backend.encode_start(request)["ticket"]
            payload = backend.tickets[ticket][1].result()
            row["cpu_embeddings"] = dict(prompt=_fingerprint(payload.prompt_embeds),
                                         negative=_fingerprint(payload.negative_prompt_embeds))
            prepared = backend.prepare(dict(request, encode_ticket=ticket))
            row["cache"] = prepared.get("negative_prompt_cache")
            for step in range(20):
                backend.advance(dict(request, steps=1))
                state = backend.states[request["state_id"]]
                if state.step_index != step + 1:
                    raise RuntimeError("Wan qualification skipped a solver boundary")
                row["step_latents"].append(_fingerprint(state.latents))
        else:
            backend.prepare(request)
            row["cache"] = backend.advance(request)["prompt_cache"]
        final = backend.finish(request)
        row["outputs"]["image" if kind == "sana" else "video"] = _output_fingerprint(final["output"])
        if kind == "wan":
            row["outputs"]["latent"] = _output_fingerprint(final["latent"])
        row["status"] = "completed"
    except Exception:
        row.update(status="failed", error=traceback.format_exc())
    finally:
        try:
            if request["state_id"] in backend.states:
                backend.release(request)
        except Exception:
            row.update(status="failed", release_error=traceback.format_exc())
        sampler.stop()
        end = time.perf_counter_ns()
        row.update(start_ns=start, end_ns=end, nvml_samples=list(sampler.samples), nvml_errors=list(sampler.errors))
        row["capacity"] = memory_margin_gate(sampler.samples, sampler.errors, start, end)
    return row


def run_arm(kind, config, out, arm):
    config = normalize_config(kind, config)
    if arm not in ARMS[kind]:
        raise ValueError("unknown qualification arm")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema=SCHEMA, kind=kind, arm=arm, config=config, config_sha256=_digest(config),
                  status="running", requests=[], performance_evidence=False, warmup=None)
    backend = None
    try:
        backend = _create_backend(kind, config, result, out)
        warmup = dict(request_id="warmup", spec=dict(cases(kind, config)[0]["spec"],
                      prompt="Warmup: a blue cup on a table."))
        if kind == "wan":
            warmup["spec"]["request_id"] = "warmup"
        result["warmup"] = _execute_request(kind, backend, warmup, config)
        if result["warmup"]["status"] != "completed" or not result["warmup"]["capacity"]["eligible"]:
            raise RuntimeError("warmup or sampled memory margin failed")
        if kind == "sana":
            backend._clear_cache()
            result["cache_initial"] = backend._snapshot()
        else:
            backend.runtime.clear_negative_prompt_cache()
            cache = backend.runtime._negative_prompt_cache
            result["cache_initial"] = None if cache is None else cache.snapshot()
        for case in cases(kind, config):
            row = _execute_request(kind, backend, case, config)
            result["requests"].append(row)
            _write(out / "run.json", result)
            if row["status"] != "completed" or not row["capacity"]["eligible"]:
                raise RuntimeError("qualification request failed: " + case["request_id"])
        result["status"] = "completed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                result.update(status="failed", close_error=traceback.format_exc())
        _write(out / "run.json", result)
    return result


def _check_hash(value, shape):
    return (isinstance(value, dict) and value.get("finite") is True and value.get("dtype") == "float32"
            and value.get("shape") == shape and isinstance(value.get("sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", value["sha256"]) is not None)


def _numeric(kind, row, case, config):
    from .workflow_serving_overlap import memory_margin_gate
    if row.get("status") != "completed" or row.get("request_id") != case["request_id"] or row.get("spec") != case["spec"]:
        raise ValueError("incomplete or mismatched request")
    samples, errors = row["nvml_samples"], row["nvml_errors"]
    if not samples or any(sample.get("uuid") != config["gpu_uuid"] for sample in samples):
        raise ValueError("missing or wrong physical GPU samples")
    gate = memory_margin_gate(samples, errors, row["start_ns"], row["end_ns"])
    if not gate["eligible"] or gate != row.get("capacity"):
        raise ValueError("memory margin not independently qualified")
    outputs = row["outputs"]
    if kind == "sana":
        if set(outputs) != {"image"} or not _check_hash(outputs["image"], [1, 1024, 1024, 3]):
            raise ValueError("missing finite Sana output")
        return dict(outputs=outputs)
    spec = case["spec"]
    shape = [1, 16, (spec["frames"] - 1) // 4 + 1, 60, 104]
    embeddings, steps = row["cpu_embeddings"], row["step_latents"]
    if set(embeddings) != {"prompt", "negative"} or not _check_hash(embeddings["prompt"], [1, 512, 4096]):
        raise ValueError("missing finite CPU prompt embeddings")
    if (spec["guidance_scale"] > 1 and not _check_hash(embeddings["negative"], [1, 512, 4096])) or (
            spec["guidance_scale"] <= 1 and embeddings["negative"] is not None):
        raise ValueError("invalid CFG CPU embeddings")
    if len(steps) != 20 or not all(_check_hash(value, shape) for value in steps):
        raise ValueError("all twenty finite solver latents are required")
    if (set(outputs) != {"video", "latent"} or not _check_hash(outputs["latent"], shape)
            or outputs["latent"] != steps[-1]
            or not _check_hash(outputs["video"], [1, spec["frames"], 480, 832, 3])):
        raise ValueError("missing finite final Wan outputs")
    return dict(outputs=outputs, cpu_embeddings=embeddings, step_latents=steps)


def _wan_recipe(config):
    from ..models.wan.runtime import MODEL_ID, REVISION, REQUIRED_VERSIONS
    return dict(model_id=MODEL_ID, revision=REVISION, model_path=str(Path(config["model_path"]).resolve()),
                diffusers=REQUIRED_VERSIONS["diffusers"], text_encoder="UMT5", device="cpu",
                dtype="torch.float32", max_sequence_length=512, num_videos_per_prompt=1)


def _entry_bytes(kind, receipt):
    tensors = receipt["entry_tensors"] if kind == "sana" else [receipt["entry_tensor"]]
    if not isinstance(tensors, list) or len(tensors) != (2 if kind == "sana" else 1):
        raise ValueError("missing retained tensor storage")
    total = 0
    for tensor in tensors:
        shape = tensor["shape"]
        if not isinstance(shape, list) or not shape or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("invalid retained tensor shape")
        if kind == "sana":
            numel, size = tensor["numel"], tensor["element_size"]
            if type(numel) is not int or numel != math.prod(shape) or type(size) is not int or size <= 0:
                raise ValueError("invalid Sana storage metadata")
            logical = numel * size
        else:
            if shape != [1, 512, 4096] or tensor["dtype"] != "torch.float32" or tensor["device"] != "cpu":
                raise ValueError("invalid retained Wan negative tensor")
            logical = math.prod(shape) * 4
        stored, accounted = tensor["storage_bytes"], tensor["accounted_bytes"]
        if (type(stored) is not int or stored < 0 or (kind == "wan" and stored < logical) or tensor["logical_bytes"] != logical
                or accounted != max(stored, logical)):
            raise ValueError("retained backing storage is not fully charged")
        total += accounted
    if total != receipt["entry_bytes"]:
        raise ValueError("cache entry bytes differ from retained tensor storage")
    return total


def _check_receipt(kind, arm, row, config, previous=None, reset=False):
    receipt = row["cache"]
    if kind == "wan" and arm == "disabled":
        if receipt is not None:
            raise ValueError("disabled Wan arm unexpectedly used a cache")
        return None
    spec = row["spec"]
    names = ("lookups", "hits", "misses", "insertions", "evictions", "bypasses", "errors", "clears")
    counters = dict(previous["counters"]) if previous else dict.fromkeys(names, 0)
    before = dict(entries=0, bytes=0, lru=[]) if reset or previous is None else previous["after"]
    if reset:
        counters["clears"] += 1
    enabled = arm != "disabled"
    schema = "ditserve.sana.prompt-cache.v1" if kind == "sana" else "ditserve.wan.negative-prompt-cache.v1"
    if (not isinstance(receipt, dict) or receipt.get("schema") != schema
            or receipt.get("request_id") != row["request_id"] or receipt.get("enabled") is not enabled
            or receipt.get("before") != before or receipt.get("sequence") != (previous["sequence"] + 1 if previous else 1)):
        raise ValueError("cache request/schema/sequence/state provenance mismatch")
    identity = "adapter_id" if kind == "sana" else "instance_id"
    if (not isinstance(receipt.get(identity), str) or not re.fullmatch("[0-9a-f]{32}", receipt[identity])
            or (previous and receipt[identity] != previous[identity])):
        raise ValueError("cache worker identity changed")
    text_hash = hashlib.sha256(spec["prompt" if kind == "sana" else "negative_prompt"].encode("utf-8")).hexdigest()
    if kind == "sana":
        key = text_hash
        if (receipt.get("prompt_sha256") != text_hash or receipt.get("owner_scope") != "adapter_instance"
                or receipt.get("key_scope") != "exact_prompt_utf8"):
            raise ValueError("Sana prompt identity mismatch")
    else:
        recipe = _wan_recipe(config)
        recipe_hash = _digest(recipe)
        key = hashlib.sha256((recipe_hash + ":" + text_hash).encode("ascii")).hexdigest()
        if (receipt.get("negative_prompt_sha256") != text_hash or receipt.get("recipe") != recipe
                or receipt.get("recipe_sha256") != recipe_hash or receipt.get("key") != key
                or receipt.get("owner_scope") != "worker_instance" or receipt.get("key_scope") != "exact_negative_prompt_utf8_and_recipe"):
            raise ValueError("Wan negative prompt/recipe binding mismatch")
    lru = OrderedDict((entry["key"], entry["bytes"]) for entry in before["lru"])
    expected = ("disabled" if not enabled else "bypass" if kind == "wan" and spec["guidance_scale"] <= 1
                else "hit" if key in lru else "miss")
    if receipt.get("outcome") != expected:
        raise ValueError("incorrect cache hit/miss/bypass outcome")
    if receipt.get("admitted") is not (expected == "miss"):
        raise ValueError("missing expected cache admission")
    if expected == "bypass" and receipt.get("bypass_reason") != "cfg_disabled":
        raise ValueError("CFG bypass not recorded")
    call_key = "encode_called" if kind == "sana" else "negative_encode_called"
    if receipt.get(call_key) is not (expected == "miss"):
        raise ValueError("unexpected text encoding path")
    limits = dict(entries=4, bytes=(16 if kind == "sana" else 64) * 1024**2) if enabled else dict(entries=0, bytes=0)
    evicted = []
    if expected in ("miss", "hit"):
        size = _entry_bytes(kind, receipt)
        counters["lookups"] += 1
        counters["hits" if expected == "hit" else "misses"] += 1
        if expected == "hit":
            if lru[key] != size:
                raise ValueError("hit tensor backing storage changed")
            lru.move_to_end(key)
        else:
            if size > limits["bytes"]:
                raise ValueError("expected qualified entry exceeds byte budget")
            while len(lru) >= limits["entries"] or sum(lru.values()) + size > limits["bytes"]:
                old, old_size = lru.popitem(last=False)
                evicted.append(dict(key=old, bytes=old_size))
                counters["evictions"] += 1
            lru[key] = size
            counters["insertions"] += 1
    elif expected == "bypass":
        counters["bypasses"] += 1
    after = dict(entries=len(lru), bytes=sum(lru.values()), lru=[dict(key=k, bytes=v) for k, v in lru.items()])
    if (receipt.get("limits") != limits or receipt.get("after") != after or receipt.get("evicted") != evicted
            or receipt.get("counters") != counters):
        raise ValueError("cache LRU/backing-storage/counter replay failed")
    if kind == "sana" and arm == "fused" and (receipt.get("sync_mode") != "fused" or
            (expected == "miss" and receipt.get("timing_scope") != "fused_miss_host_wall")):
        raise ValueError("fused miss timing scope is absent")
    return receipt


def qualify(kind, config, arms):
    """Fail closed on missing arrays, steps, provenance, or raw capacity data."""
    config = normalize_config(kind, config)
    failures, baseline = [], None
    if not isinstance(arms, list) or [arm.get("id") for arm in arms] != list(ARMS[kind]):
        return dict(passed=False, failures=["missing or reordered independent arms"], performance_evidence=False)
    for arm in arms:
        try:
            result = arm["result"]
            if (arm.get("returncode") != 0 or result.get("status") != "completed" or result.get("schema") != SCHEMA
                    or result.get("kind") != kind or result.get("arm") != arm["id"]
                    or result.get("config") != config or result.get("config_sha256") != _digest(config)):
                raise ValueError("arm process or frozen configuration mismatch")
            runtime, affinity = result["runtime"], result["cpu_affinity"]
            cores = affinity["physical_cores"]
            if (runtime.get("matmul_tf32") is not False or runtime.get("cudnn_tf32") is not True
                    or runtime.get("interop_threads") != 1 or runtime.get("intraop_threads") != 8
                    or affinity.get("bound_cpus") != config["cpu_mask"]
                    or len(cores) != 8 or len({(c["socket"], c["core"]) for c in cores}) != 8):
                raise ValueError("runtime/physical-core contract is not satisfied")
            initial = result.get("cache_initial")
            if not (kind == "wan" and arm["id"] == "disabled") and initial != dict(entries=0, bytes=0, lru=[]):
                raise ValueError("measured requests must start with a cold cache")
            expected = cases(kind, config)
            if len(result["requests"]) != len(expected):
                raise ValueError("incomplete request matrix")
            warmup = dict(request_id="warmup", spec=dict(expected[0]["spec"], prompt="Warmup: a blue cup on a table."))
            if kind == "wan":
                warmup["spec"]["request_id"] = "warmup"
            numerics = [_numeric(kind, result["warmup"], warmup, config)]
            previous = _check_receipt(kind, arm["id"], result["warmup"], config)
            for index, (row, case) in enumerate(zip(result["requests"], expected)):
                numerics.append(_numeric(kind, row, case, config))
                previous = _check_receipt(kind, arm["id"], row, config, previous, reset=index == 0)
            if baseline is None:
                baseline = numerics
            elif numerics != baseline:
                raise ValueError("exact per-request CPU/step/final output comparison failed")
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            failures.append(dict(arm=arm.get("id"), reason=str(error)))
    return dict(passed=not failures, failures=failures, performance_evidence=False,
                method="exact CPU embeddings, every Wan solver latent and final arrays; raw sampled NVML capacity")


def run(kind, config, out):
    config = normalize_config(kind, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    path = out / "config.json"
    _write(path, config)
    result = dict(schema=SCHEMA, kind=kind, config=config, arms=[], qualified=False,
                  status="running", performance_evidence=False, semantic_quality="unverified")
    try:
        for arm in ARMS[kind]:
            command = [sys.executable, "-m", "ditserve.evaluation.workflow_encoding_qualification",
                       "--kind", kind, "--config", str(path), "--out", str(out / arm), "--arm", arm]
            with (out / (arm + ".stdout.log")).open("x") as stdout, (out / (arm + ".stderr.log")).open("x") as stderr:
                child = subprocess.run(command, stdout=stdout, stderr=stderr, check=False,
                                       timeout=config["arm_timeout_seconds"])
            target = out / arm / "run.json"
            measured = json.loads(target.read_text()) if target.exists() else None
            result["arms"].append(dict(id=arm, command=command, returncode=child.returncode, result=measured,
                output=str(target), output_sha256=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None))
            _write(out / "run.json", result)
            if child.returncode or not measured or measured.get("status") != "completed":
                raise RuntimeError("qualification child failed: " + arm)
        result["qualification"] = qualify(kind, config, result["arms"])
        result["qualified"] = result["qualification"]["passed"]
        result["status"] = "completed" if result["qualified"] else "qualification_failed"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        result["qualification"] = qualify(kind, config, result["arms"])
    _write(out / "run.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(ARMS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(dict.fromkeys(sum(ARMS.values(), ()))), help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run_arm(args.kind, config, args.out, args.arm) if args.arm else run(args.kind, config, args.out)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
