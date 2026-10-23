"""Single-GPU, isolated FP16 Linear replay from bound SP1 reference captures.

Imports do not initialize torch or CUDA. Numerical differences are observations:
qualification means trusted inputs, valid shapes, finite values and exact repeats.
Only an exact full/default reference replay anchors a claim about the source run.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback


SCHEMA = "ditserve.workflow.linear-replay.v1"
OPERATORS = {
    "to_q": ("blocks.0.attn1.to_q", "to_q_input", "to_q"),
    "ffn.net.0.proj": ("blocks.0.ffn.net.0.proj", "feed_forward_input", "feed_forward_first_projection"),
    "ffn.net.2": ("blocks.0.ffn.net.2", "feed_forward_activation", "feed_forward_second_projection"),
}
DEFAULT_FLAGS = dict(allow_tf32=False, allow_fp16_reduced_precision_reduction=True,
                     allow_fp16_accumulation=False)
CANDIDATES = ("full_rows_ffn2", "full_rows_ffn2_qkv", "global_tile_1024",
              "global_tile_2048", "linear_fp32")


def _binding(value):
    return (isinstance(value, dict) and isinstance(value.get("path"), str)
            and Path(value["path"]).is_absolute() and ".." not in Path(value["path"]).parts
            and isinstance(value.get("sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", value["sha256"]) is not None)


def normalize_config(config):
    value = json.loads(json.dumps(config, allow_nan=False))
    for key, default in dict(device="cuda:0", cpu_threads=8, repeats=3, zero_padded=True,
                             operators=list(OPERATORS), branches=["cond", "uncond"],
                             save_arrays=False).items():
        value.setdefault(key, default)
    if value["device"] != "cuda:0":
        raise ValueError("the isolated GPU must be addressed as cuda:0")
    if not isinstance(value.get("gpu_uuid"), str) or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value["gpu_uuid"]):
        raise ValueError("gpu_uuid must bind one physical GPU UUID")
    if type(value["cpu_threads"]) is not int or value["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be positive")
    cpus = value.get("cpu_affinity")
    if (not isinstance(cpus, list) or len(cpus) != value["cpu_threads"] or
            any(type(cpu) is not int or cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus)):
        raise ValueError("cpu_affinity must contain cpu_threads distinct CPU indices")
    if type(value["repeats"]) is not int or value["repeats"] < 2:
        raise ValueError("at least two repeats are required")
    for name in ("zero_padded", "save_arrays"):
        if type(value[name]) is not bool:
            raise ValueError(name + " must be boolean")
    if "candidates" in value:
        selected = value["candidates"]
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(item, str) or item not in CANDIDATES for item in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("unknown, empty or duplicate candidates")
    if "microbenchmark" in value:
        bench = value["microbenchmark"]
        if not isinstance(bench, dict) or set(bench) - {"enabled", "warmups", "repeats"}:
            raise ValueError("unknown microbenchmark options")
        bench.setdefault("enabled", False)
        bench.setdefault("warmups", 3)
        bench.setdefault("repeats", 10)
        if (type(bench["enabled"]) is not bool or type(bench["warmups"]) is not int
                or not 1 <= bench["warmups"] <= 100 or type(bench["repeats"]) is not int
                or not 2 <= bench["repeats"] <= 1000):
            raise ValueError("microbenchmark requires warmups 1..100 and repeats 2..1000")
    for key, choices in (("operators", OPERATORS), ("branches", ("cond", "uncond"))):
        selected = value[key]
        if (not isinstance(selected, list) or not selected or
                any(not isinstance(item, str) or item not in choices for item in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("unknown, empty or duplicate " + key)
    if not _binding(value.get("model_index")):
        raise ValueError("model_index needs an absolute path and file SHA256")
    shards = value.get("model_shards")
    if (not isinstance(shards, dict) or not shards or
            any(Path(name).name != name or not name.endswith(".safetensors") or not _binding(binding)
                for name, binding in shards.items())):
        raise ValueError("model_shards must bind named safetensors files")
    cases, identities = value.get("cases"), set()
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    for case in cases:
        if (not isinstance(case, dict) or type(case.get("seed")) is not int or case["seed"] < 0
                or type(case.get("frames")) is not int or case["frames"] not in (17, 33)):
            raise ValueError("cases require a seed and frames=17 or 33")
        identity = (case["seed"], case["frames"])
        if identity in identities:
            raise ValueError("duplicate seed/frame case")
        identities.add(identity)
        captures = case.get("captures")
        if (not isinstance(captures, list) or len(captures) != 2 or
                any(not _binding(item) or type(item.get("rank")) is not int for item in captures)
                or {item["rank"] for item in captures} != {0, 1}):
            raise ValueError("each case needs bound rank0 and rank1 capture manifests")
        keys = needed_capture_keys(value)
        for capture in captures:
            files = capture.get("files")
            if not isinstance(files, dict) or any(not _binding(files.get(key)) for key in keys):
                raise ValueError("each selected capture tensor needs a file SHA256 binding")
    return value


def needed_capture_keys(config):
    return sorted({f"{branch}.block00.{tensor}" for branch in config["branches"]
                   for name in config["operators"] for tensor in OPERATORS[name][1:]})


class Bindings:
    """Hash each immutable source once, even when many tensors share a shard."""
    def __init__(self):
        self.cache = {}

    def verify(self, binding):
        path = Path(binding["path"]).resolve(strict=True)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if path in self.cache:
            record, old_identity = self.cache[path]
            if old_identity != identity:
                raise ValueError("bound source changed while reading: " + str(path))
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(block)
            record = dict(path=str(path), bytes=stat.st_size, sha256=digest.hexdigest())
            self.cache[path] = (record, identity)
        if record["sha256"] != binding["sha256"]:
            raise ValueError("source file SHA256 mismatch: " + str(path))
        return dict(record)


def array_digest(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    raw = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(raw), 8 * 1024**2):
        digest.update(raw[start:start + 8 * 1024**2])
    return dict(sha256=digest.hexdigest(), shape=list(array.shape), dtype=str(array.dtype),
                finite=bool(np.isfinite(array).all()))


def load_capture(case, binding, config, bindings):
    import numpy as np
    manifest_binding = bindings.verify(binding)
    path = Path(binding["path"])
    suffix = (f"rank{binding['rank']}", f"seed{case['seed']}", f"f{case['frames']}",
              "localization", "reference", "step00", "block00-qkv", "capture.json")
    if tuple(path.parts[-8:]) != suffix:
        raise ValueError("capture path must select the declared SP1 reference rank/seed/frame")
    manifest = json.loads(path.read_text())
    if any(manifest.get(key) != expected for key, expected in dict(
            status="completed", rank=binding["rank"], target_step=0, target_block=0,
            scheduler_steps=20, scope="selected_block_qkv",
            comparison_layout="SP2 local sequence vs matching SP1 contiguous sequence shard").items()):
        raise ValueError("capture manifest is incomplete or has incompatible provenance")
    rows = manifest.get("layers", [])
    if len({row.get("key") for row in rows}) != len(rows):
        raise ValueError("duplicate capture tensor key")
    rows = {row["key"]: row for row in rows}
    records = {}
    for key in needed_capture_keys(config):
        file_binding = binding["files"][key]
        record = bindings.verify(file_binding)
        source = Path(file_binding["path"])
        row = rows.get(key, {})
        if (source.resolve().parent != path.resolve().parent or source.name != key + ".npy"
                or Path(row.get("path", "")).resolve() != source.resolve()
                or row.get("block") != 0 or row.get("branch") != key.split(".")[0]
                or row.get("tensor") != key.split(".")[-1]):
            raise ValueError("capture tensor path/identity mismatch: " + key)
        array = np.load(source, mmap_mode="r", allow_pickle=False)
        actual = array_digest(array)
        expected_tokens = (((case["frames"] - 1) // 4 + 1) * 30 * 52) // 2
        if (array.dtype != np.float16 or array.ndim != 3 or array.shape[0] != 1
                or array.shape[1] != expected_tokens or not actual["finite"]):
            raise ValueError("reference captures require finite [1, local_tokens, C] FP16 arrays")
        if any(row.get("hash", {}).get(name) != actual[name] for name in actual):
            raise ValueError("capture array hash/shape/dtype mismatch: " + key)
        records[key] = dict(file=record, array=actual)
    return dict(rank=binding["rank"], capture=manifest_binding, tensors=records)


def assemble_pair(captures, input_key, output_key, bindings=None):
    import numpy as np
    captures = sorted(captures, key=lambda row: row["rank"])
    if [row["rank"] for row in captures] != [0, 1]:
        raise ValueError("rank order must be exactly 0,1")
    inputs, outputs = [], []
    for rank in captures:
        if bindings is not None:
            for key in (input_key, output_key):
                bindings.verify(rank["tensors"][key]["file"])
        x, y = (np.load(rank["tensors"][key]["file"]["path"], mmap_mode="r", allow_pickle=False)
                for key in (input_key, output_key))
        if x.shape[:2] != y.shape[:2]:
            raise ValueError("input/output token geometry differs")
        inputs.append(x)
        outputs.append(y)
    if inputs[0].shape != inputs[1].shape or outputs[0].shape != outputs[1].shape:
        raise ValueError("rank shard shapes differ")
    return np.concatenate(inputs, axis=1), np.concatenate(outputs, axis=1)


def plan_model(config, bindings):
    index_binding = bindings.verify(config["model_index"])
    index = json.loads(Path(config["model_index"]["path"]).read_text())
    names = [OPERATORS[op][0] + "." + suffix for op in config["operators"] for suffix in ("weight", "bias")]
    plan = {}
    for name in names:
        shard = index.get("weight_map", {}).get(name)
        if shard not in config["model_shards"]:
            raise ValueError("required model tensor has no bound shard: " + name)
        binding = config["model_shards"][shard]
        if Path(binding["path"]).name != shard:
            raise ValueError("model shard filename differs from weight map")
        plan[name] = dict(name=name, file=bindings.verify(binding))
    return dict(index=index_binding, tensors=plan)


def validate_linear_shapes(x, y, weight, bias):
    if (len(x) != 3 or len(y) != 3 or x[:2] != y[:2] or x[0] != 1 or x[1] % 2
            or len(weight) != 2 or tuple(weight) != (y[2], x[2])
            or tuple(bias) != (y[2],)):
        raise ValueError("Linear input/output/weight/bias shapes are incompatible")


def load_weights(plan, torch):
    from safetensors import safe_open
    values, records = {}, {}
    by_file = {}
    for name, item in plan["tensors"].items():
        by_file.setdefault(item["file"]["path"], []).append(name)
    for path, names in by_file.items():
        with safe_open(path, framework="pt", device="cpu") as stored:
            for name in names:
                original = stored.get_tensor(name)
                if original.dtype not in (torch.float16, torch.float32):
                    raise ValueError("unexpected original model weight dtype: " + name)
                fp16 = original.to(dtype=torch.float16).contiguous()
                original_hash, execution_hash = array_digest(original.numpy()), array_digest(fp16.numpy())
                if not original_hash["finite"] or not execution_hash["finite"]:
                    raise ValueError("nonfinite model tensor: " + name)
                values[name] = fp16
                records[name] = dict(plan["tensors"][name], original=original_hash, execution_fp16=execution_hash)
    return values, records


def arithmetic_flags(torch):
    return {name: getattr(torch.backends.cuda.matmul, name) for name in DEFAULT_FLAGS}


@contextlib.contextmanager
def precision_mode(torch, mode):
    if mode not in ("default", "no_reduced_precision"):
        raise ValueError("unknown precision mode")
    before = arithmetic_flags(torch)
    audit = dict(before=before)
    try:
        if mode == "no_reduced_precision":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
        audit["applied"] = arithmetic_flags(torch)
        yield audit
    finally:
        for name, value in before.items():
            setattr(torch.backends.cuda.matmul, name, value)
        audit["restored"] = arithmetic_flags(torch)


@contextlib.contextmanager
def isolated_environment(config):
    if "torch" in sys.modules:
        raise RuntimeError("GPU isolation must be established before importing torch")
    changes = dict(CUDA_VISIBLE_DEVICES=config["gpu_uuid"],
                   OMP_NUM_THREADS=str(config["cpu_threads"]), MKL_NUM_THREADS=str(config["cpu_threads"]),
                   OPENBLAS_NUM_THREADS=str(config["cpu_threads"]))
    previous = {name: os.environ.get(name) for name in changes}
    before = {int(task.name): os.sched_getaffinity(int(task.name)) for task in Path("/proc/self/task").iterdir()}
    original_main = os.sched_getaffinity(0)
    if not set(config["cpu_affinity"]).issubset(original_main):
        raise ValueError("requested inference CPUs are outside allowed affinity")
    try:
        os.environ.update(changes)
        for tid in before:
            try:
                os.sched_setaffinity(tid, config["cpu_affinity"])
            except ProcessLookupError:
                pass
        yield
    finally:
        for task in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(task.name), before.get(int(task.name), original_main))
            except ProcessLookupError:
                pass
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def qualify_arm(row):
    """Differences from a reference are observations, not this contract's gate."""
    return (row["output"]["finite"] and row["output"]["dtype"] == "float16"
            and row["comparison_to_reference"]["shape_match"]
            and row["comparison_to_reference"]["reference_finite"]
            and row["repeated_exact"])


def qualification_record(passed=None):
    return dict(passed=passed is True, scope="bound sources, shape, finite values and exact repeats",
                scopes=dict(
                    numeric=dict(status="unverified",
                        reason="Independent Linear operators do not qualify a complete SP request."),
                    replay_integrity=dict(status="unverified" if passed is None else
                                          "passed" if passed else "failed")))


def benchmark_operator(torch, operation, config):
    """CUDA-event operator timing, without transfers, hashing or comparisons."""
    bench = config.get("microbenchmark", {})
    if not bench.get("enabled", False):
        return None
    samples = []
    with torch.inference_mode():
        for _ in range(bench["warmups"]):
            output = operation()
        torch.cuda.synchronize()
        for _ in range(bench["repeats"]):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            output = operation()
            end.record()
            end.synchronize()
            samples.append(float(begin.elapsed_time(end)))
    if any(not 0 < item < float("inf") for item in samples):
        raise RuntimeError("invalid CUDA event measurement")
    return dict(scope="single_operator_cuda_event", warmups=bench["warmups"], repeats=bench["repeats"],
                samples_ms=samples, median_ms=statistics.median(samples), min_ms=min(samples),
                includes="operator, candidate padding/cropping/tiling and input/output dtype conversion",
                excludes="CPU copies, hashes, comparisons, checkpoint loading and initial weight promotion",
                operator_performance_evidence=True, request_performance_qualification=False)


def candidate_operation(torch, candidate, operator, value, weight, bias, *, global_start, global_rows):
    """Return an operator-only callable; all weights originate from execution FP16."""
    from .workflow_attention import selected_linear, tiled_linear
    if candidate not in CANDIDATES or operator not in OPERATORS:
        raise ValueError("unknown replay candidate/operator")
    selected = selected_linear(OPERATORS[operator][0], candidate)
    native = lambda tensor: torch.nn.functional.linear(tensor, weight, bias)
    recipe = dict(candidate=candidate, global_rows=global_rows, global_start=global_start,
                  owned_rows=value.shape[1], weight_source="execution_fp16",
                  input_dtype=str(value.dtype), weight_dtype=str(weight.dtype))
    if candidate.startswith("full_rows_"):
        applied = selected and value.shape[1] != global_rows
        def operation():
            if not applied:
                return native(value)
            full = value.new_zeros((value.shape[0], global_rows, value.shape[2]))
            full[:, global_start:global_start + value.shape[1], :] = value
            return native(full)[:, global_start:global_start + value.shape[1], :].clone(
                memory_format=torch.contiguous_format)
        recipe.update(applied=applied, selected_module=selected, row_alignment="global_sequence_origin")
    elif candidate.startswith("global_tile_"):
        tile = int(candidate.rsplit("_", 1)[1])
        operation = lambda: tiled_linear(torch, value, native, global_start=global_start, tile=tile)
        recipe.update(applied=True, tile_rows=tile, row_alignment="global_sequence_origin")
    else:
        # Promote values that have ALREADY undergone the real FP16 checkpoint cast.
        if weight.dtype != torch.float16 or bias.dtype != torch.float16 or value.dtype != torch.float16:
            raise ValueError("local FP32 must promote the actual FP16 execution values")
        promoted_weight, promoted_bias = weight.to(dtype=torch.float32), bias.to(dtype=torch.float32)
        operation = lambda: torch.nn.functional.linear(
            value.to(dtype=torch.float32), promoted_weight, promoted_bias).to(dtype=torch.float16)
        recipe.update(applied=True, compute_dtype="float32", output_dtype="float16",
                      tf32=False, weight_promotion="FP16 execution tensor -> FP32; never original checkpoint F32")
    return operation, recipe


def replay_candidates(torch, x, reference, original_full, w, b, operator, config, out, anchored):
    import numpy as np
    from .workflow_encoder import compare_arrays
    rows = []
    length = x.shape[1] // 2
    for candidate in config.get("candidates", []):
        arms, full_array = [], None
        with precision_mode(torch, "default") as flag_audit:
            if candidate == "linear_fp32":
                torch.backends.cuda.matmul.allow_tf32 = False
                flag_audit["applied"] = arithmetic_flags(torch)
            for kind, rank in (("full", None), ("half", 0), ("half", 1)):
                start, end = (0, x.shape[1]) if rank is None else (rank * length, (rank + 1) * length)
                value = x if rank is None else x[:, start:end, :].clone(memory_format=torch.contiguous_format)
                operation, recipe = candidate_operation(torch, candidate, operator, value, w, b,
                                                        global_start=start, global_rows=x.shape[1])
                first, digests = None, []
                with torch.inference_mode():
                    for _ in range(config["repeats"]):
                        output = operation()
                        array = output.detach().cpu().numpy().copy()
                        digests.append(array_digest(array))
                        if first is None:
                            first = array
                row = dict(shape_kind=kind, rank=rank, row_range=[start, end], recipe=recipe,
                    recipe_applied=recipe["applied"], input_shape=list(value.shape), input_stride=list(value.stride()),
                    output=digests[0], repeat_outputs=digests,
                    repeated_exact=all(item == digests[0] for item in digests),
                    comparison_to_reference=compare_arrays(first, reference[:, start:end, :]),
                    comparison_to_original_full=compare_arrays(first, original_full[:, start:end, :]),
                    performance_evidence=False)
                if rank is None:
                    full_array = first
                else:
                    row["comparison_to_full_same_precision"] = compare_arrays(first, full_array[:, start:end, :])
                measurement = benchmark_operator(torch, operation, config)
                if measurement is not None:
                    row["microbenchmark"] = measurement
                if config["save_arrays"]:
                    path = out / (candidate + "-" + kind + ("" if rank is None else f"-rank{rank}") + ".npy")
                    np.save(path, first, allow_pickle=False)
                    row["array_path"] = str(path)
                row["qualified"] = qualify_arm(row)
                arms.append(row)
        rows.append(dict(candidate=candidate, arms=arms, flags=flag_audit,
                         qualified=all(arm["qualified"] for arm in arms), reference_reproduced=anchored,
                         performance_evidence=False, deployment_qualification=False,
                         scope="Operator screening only; full/candidate bridge to original SP1 remains explicit."))
    return rows


def replay_operator(torch, x_array, reference, weight, bias, config, out, operator=None):
    import numpy as np
    from .workflow_encoder import compare_arrays
    validate_linear_shapes(x_array.shape, reference.shape, weight.shape, bias.shape)
    device = config["device"]
    x, w, b = (torch.from_numpy(x_array).to(device=device, dtype=torch.float16),
               weight.to(device=device, dtype=torch.float16), bias.to(device=device, dtype=torch.float16))
    length = x_array.shape[1] // 2
    arms, flags, default_arrays = [], [], {}
    for precision in ("default", "no_reduced_precision"):
        full_array = None
        with precision_mode(torch, precision) as flag_audit:
            shapes = [("full", None), ("half", 0), ("half", 1)]
            if config["zero_padded"]:
                shapes += [("padded", 0), ("padded", 1)]
            for kind, rank in shapes:
                start, end = (0, x_array.shape[1]) if rank is None else (rank*length, (rank+1)*length)
                if kind == "full":
                    argument = x
                elif kind == "half":
                    argument = x[:, start:end, :].clone(memory_format=torch.contiguous_format)
                else:
                    argument = torch.zeros_like(x)
                    argument[:, start:end, :] = x[:, start:end, :]
                first, digests = None, []
                with torch.inference_mode():
                    for _ in range(config["repeats"]):
                        output = torch.nn.functional.linear(argument, w, b)
                        if kind == "padded":
                            output = output[:, start:end, :].clone(memory_format=torch.contiguous_format)
                        array = output.detach().cpu().numpy().copy()
                        digest = array_digest(array)
                        digests.append(digest)
                        if first is None:
                            first = array
                key = kind if rank is None else f"{kind}-rank{rank}"
                row = dict(precision=precision, shape_kind=kind, rank=rank, global_rows=x_array.shape[1],
                           row_range=[start, end], input_shape=list(argument.shape),
                           input_stride=list(argument.stride()), output=digests[0],
                           repeat_outputs=digests, repeated_exact=all(item == digests[0] for item in digests),
                           comparison_to_reference=compare_arrays(first, reference[:, start:end, :]),
                           performance_evidence=False)
                if kind == "full":
                    full_array = first
                else:
                    row["comparison_to_full_same_precision"] = compare_arrays(first, full_array[:, start:end, :])
                if precision == "default":
                    default_arrays[key] = first
                else:
                    row["comparison_to_default_same_shape"] = compare_arrays(first, default_arrays[key])
                if config["save_arrays"]:
                    path = out / (precision + "-" + key + ".npy")
                    np.save(path, first, allow_pickle=False)
                    row["array_path"] = str(path)
                row["qualified"] = qualify_arm(row)
                def operation():
                    timed_argument = argument
                    if kind == "padded":
                        timed_argument = torch.zeros_like(x)
                        timed_argument[:, start:end, :] = x[:, start:end, :]
                    computed = torch.nn.functional.linear(timed_argument, w, b)
                    return (computed[:, start:end, :].clone(memory_format=torch.contiguous_format)
                            if kind == "padded" else computed)
                measurement = benchmark_operator(torch, operation, config)
                if measurement is not None:
                    row["microbenchmark"] = measurement
                arms.append(row)
                del output, argument
        flags.append(dict(mode=precision, **flag_audit))
    anchored = arms[0]["comparison_to_reference"]["exact"] and arms[0]["qualified"]
    result = dict(arms=arms, flags=flags, qualified=all(row["qualified"] for row in arms),
                reference_reproduced=anchored, performance_evidence=False,
                interpretation=("Anchored single-operator shape/precision comparison." if anchored else
                    "Full/default does not exactly reproduce saved SP1; do not attribute source-run divergence."),
                scope=("Selected self-Q/FFN operators each use saved SP1 inputs; this does not replay a chained "
                       "FFN or diffusion step and does not explain the no_reduced cross-attention first difference."))
    if config.get("candidates"):
        if operator not in OPERATORS:
            raise ValueError("candidate replay requires an explicit operator identity")
        result["candidates"] = replay_candidates(torch, x, reference, default_arrays["full"], w, b,
                                                 operator, config, out, anchored)
        result["qualified"] = result["qualified"] and all(row["qualified"] for row in result["candidates"])
    return result


def run_replay(config, out):
    from .workflow_budget import atomic_json
    config, out = normalize_config(config), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out / "run.json")
    result = dict(schema=SCHEMA, status="running", configuration=config, started_unix_ns=time.time_ns(),
                  role="diagnostic_only", performance_evidence=False, cases=[],
                  source_recipe=dict(height=480, width=832, scheduler_steps=20, step=0, block=0),
                  qualification=qualification_record(),
                  deployment_qualification=dict(passed=False))
    atomic_json(out / "run.json", result)
    try:
        with isolated_environment(config):
            # Establish affinity and CUDA visibility before loading either torch or NumPy/BLAS.
            bindings = Bindings()
            sources = []
            for case in config["cases"]:
                sources.append(dict(seed=case["seed"], frames=case["frames"],
                    captures=[load_capture(case, item, config, bindings) for item in
                              sorted(case["captures"], key=lambda row: row["rank"])]))
            model_plan = plan_model(config, bindings)
            result["sources"] = sources
            result["model_plan"] = model_plan
            atomic_json(out / "run.json", result)
            import torch
            torch.set_num_threads(config["cpu_threads"])
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("exactly one isolated CUDA device is required")
            torch.cuda.set_device(0)
            result["runtime"] = dict(torch=torch.__version__, device_name=torch.cuda.get_device_name(0),
                                    visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
                                    cpu_affinity=sorted(os.sched_getaffinity(0)))
            original_flags = arithmetic_flags(torch)
            result["original_flags"] = original_flags
            if original_flags != DEFAULT_FLAGS:
                raise RuntimeError("runtime defaults differ from bound source arithmetic recipe")
            try:
                weights, result["model_tensors"] = load_weights(model_plan, torch)
                for source in sources:
                    for branch in config["branches"]:
                        for operator in config["operators"]:
                            model_key, input_name, output_name = OPERATORS[operator]
                            x, y = assemble_pair(source["captures"],
                                f"{branch}.block00.{input_name}", f"{branch}.block00.{output_name}", bindings)
                            folder = out / f"seed{source['seed']}" / f"f{source['frames']}" / branch / operator
                            folder.mkdir(parents=True, exist_ok=True)
                            replay = replay_operator(torch, x, y, weights[model_key + ".weight"],
                                                     weights[model_key + ".bias"], config, folder, operator=operator)
                            row = dict(seed=source["seed"], frames=source["frames"], branch=branch,
                                       operator=operator, weight_name=model_key+".weight", bias_name=model_key+".bias",
                                       full_input=array_digest(x), full_reference=array_digest(y), **replay)
                            result["cases"].append(row)
                            atomic_json(folder / "result.json", row)
                            atomic_json(out / "run.json", result)
                expected = len(sources) * len(config["branches"]) * len(config["operators"])
                passed = len(result["cases"]) == expected and all(row["qualified"] for row in result["cases"])
                result["qualification"].update(qualification_record(passed), expected_operator_cases=expected,
                    source_integrity=True, repeated_exact=all(
                        arm["repeated_exact"] for row in result["cases"] for arm in
                        row["arms"] + [arm for candidate in row.get("candidates", []) for arm in candidate["arms"]]))
                result["reference_reproduction"] = dict(
                    passed=all(row["reference_reproduced"] for row in result["cases"]),
                    scope="Exact full/default anchor, separate from arithmetic contract qualification.")
                result["status"] = "completed" if passed else "qualification_failed"
            finally:
                for name, value in original_flags.items():
                    setattr(torch.backends.cuda.matmul, name, value)
                result["restored_flags"] = arithmetic_flags(torch)
    except Exception:
        result.update(status="failed", error=traceback.format_exc(), qualification=qualification_record(False))
    finally:
        result["finished_unix_ns"] = time.time_ns()
        atomic_json(out / "run.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_replay(json.loads(args.config.read_text()), args.out)
    print(json.dumps(dict(status=result["status"], manifest=str(args.out / "run.json"))))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
