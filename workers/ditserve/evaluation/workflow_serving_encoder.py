"""Run02 encoding studies; imports are CPU-only and never load a model.

Public entry points are run_quality(config, out), run_cpu_micro(config, out),
and run_prefetch(config, out). Each invocation belongs in a fresh process and
writes an independent, non-overwritable out/run.json. The caller owns GPU
admission and process timeouts. No production worker defaults are changed.

    python -m ditserve.evaluation.workflow_serving_encoder \
        --kind cpu_micro --config cpu4.json --out /data/workflow/run02/cpu4
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import statistics
import threading
import time
import traceback
import weakref

from .workflow_encoder import PROMPTS, _decode_separate, _embedding_arrays, _spec, _standalone_encoder, compare_arrays
from .workflow_embedding_replay import array_sha256, file_sha256, load_embeddings
from .workflow_metrics import NvmlSampler, PhaseRecorder, summarize_phases


SCHEMA = "ditserve.workflow.serving.encoder.v1"
KINDS = ("quality", "cpu_micro", "prefetch")
PRECISIONS = {"T0": "float32", "T1": "float32", "T2": "bfloat16"}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def prompt_case(name):
    canonical = "english" if name == "shortenglish" else name
    case = next((dict(row) for row in PROMPTS if row["id"] == canonical), None)
    if case is None:
        raise ValueError("unknown prompt case: " + str(name))
    case["id"] = name
    return case


def validate_config(kind, config):
    if kind not in KINDS or not isinstance(config, dict):
        raise ValueError("expected a supported kind and a configuration dictionary")
    value = dict(model_path="/models/wan2.1-t2v-1.3b", device="cuda:1", cpu_threads=8,
                 cpu_affinity="explicit", attention="flash", matmul_tf32=False, cudnn_tf32=True,
                 steps=20, frames=[17, 33], seed=20261019, repeats=3, warmups=1,
                 sample_interval=0.05, depth=1, requests=4, producer_join_timeout_s=120., save_videos=False)
    value.update(config)
    value["kind"] = kind
    if type(value["save_videos"]) is not bool:
        raise ValueError("save_videos must be a boolean")
    value["cpu_threads"] = _positive_integer(value["cpu_threads"], "cpu_threads")
    mask = value.get("cpu_mask", value.get("cpu_affinity_mask"))
    if not isinstance(mask, list) or len(mask) != value["cpu_threads"] or any(type(cpu) is not int or cpu < 0 for cpu in mask):
        raise ValueError("cpu_mask must be an explicit list with cpu_threads nonnegative logical CPU IDs")
    if len(set(mask)) != len(mask):
        raise ValueError("cpu_mask contains duplicate CPUs")
    value["cpu_mask"] = sorted(mask)
    for name in ("steps", "repeats", "requests"):
        _positive_integer(value[name], name)
    _positive_integer(value["warmups"], "warmups", 0)
    if type(value["depth"]) is not int or value["depth"] not in (0, 1, 2):
        raise ValueError("depth must be 0, 1 or 2")
    for name in ("sample_interval", "producer_join_timeout_s"):
        if isinstance(value[name], bool) or not isinstance(value[name], (int, float)) or not math.isfinite(value[name]) or value[name] <= 0:
            raise ValueError(name + " must be positive and finite")
    frames = value["frames"]
    if not isinstance(frames, list) or not frames or len(set(frames)) != len(frames) or any(type(f) is not int or f < 1 or (f - 1) % 4 for f in frames):
        raise ValueError("frames must be unique positive 4*k+1 integers")
    if not isinstance(value["device"], str) or not value["device"].startswith("cuda:") or not value["device"][5:].isdigit():
        raise ValueError("device must be cuda:N, also identifying CPU NUMA provenance")
    if value["attention"] not in ("flash", "math", "efficient"):
        raise ValueError("unsupported attention backend")
    if value["matmul_tf32"] is not False or type(value["cudnn_tf32"]) is not bool:
        raise ValueError("matmul_tf32 must be false and cudnn_tf32 an explicit boolean")
    defaults = ["chinese", "truncation"] if kind == "quality" else ["shortenglish", "truncation"]
    cases = value.get("prompt_cases", value.get("cases", defaults))
    if not isinstance(cases, list) or not cases or len(set(cases)) != len(cases):
        raise ValueError("prompt_cases must be a nonempty list of unique case names")
    for case in cases:
        prompt_case(case)
    value["prompt_cases"] = cases
    seeds = value.get("seeds", [20261019, 20261020])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    value["seeds"] = seeds
    _positive_integer(value["seed"], "seed", 0)
    # Enforce serialization before creating output; no hidden runtime objects.
    return json.loads(json.dumps(value, allow_nan=False))


def validate_physical_mask(mask, allowed, topology):
    """Pure validation helper: topology(cpu) returns (socket, physical core)."""
    if not set(mask).issubset(set(allowed)):
        raise ValueError("cpu_mask is outside the process's allowed CPU set")
    identities = [tuple(topology(cpu)) for cpu in mask]
    if len(set(identities)) != len(identities):
        raise ValueError("cpu_mask selects SMT siblings of the same physical core")
    return identities


def bind_physical_mask(mask):
    allowed = sorted(os.sched_getaffinity(0))
    def topology(cpu):
        directory = Path("/sys/devices/system/cpu/cpu%d/topology" % cpu)
        return (int((directory / "physical_package_id").read_text()), int((directory / "core_id").read_text()))
    identities = validate_physical_mask(mask, allowed, topology)
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), mask)
        except ProcessLookupError:
            pass
    return dict(policy="explicit physical-core mask", bound_cpus=list(mask), allowed_before=allowed,
                physical_cores=[dict(cpu=cpu, socket=identity[0], core=identity[1]) for cpu, identity in zip(mask, identities)])


def _runtime(config, record, *, gpu):
    record["cpu_affinity"] = bind_physical_mask(config["cpu_mask"])
    os.environ["OMP_NUM_THREADS"] = str(config["cpu_threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["cpu_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(config["cpu_threads"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(config["cpu_threads"])
    # Each experiment must use a fresh process; do not silently reuse thread state.
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = config["cudnn_tf32"]
    if gpu:
        torch.cuda.set_device(config["device"])
    elif torch.cuda.is_initialized():
        raise RuntimeError("CPU microbenchmark requires a fresh process without CUDA initialization")
    record["runtime"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        cuda_initialized=bool(torch.cuda.is_initialized()), interop_threads=1,
        tokenizer_parallelism=False)
    return torch


def _nvml_device(config):
    if "nvml_device" in config:
        return config["nvml_device"]
    ordinal = int(config["device"][5:])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return ordinal
    identities = [item.strip() for item in visible.split(",")]
    if ordinal >= len(identities):
        raise ValueError("device is outside CUDA_VISIBLE_DEVICES")
    selected = identities[ordinal]
    if selected.isdigit():
        return int(selected)
    if selected.startswith("GPU-"):
        return selected
    raise ValueError("unsupported NVML device mapping")


def _run(kind, config, out, body):
    config = validate_config(kind, config)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty experiment output: " + str(out))
    record = dict(schema=SCHEMA, kind=kind, status="running", configuration=config,
                  configuration_sha256=_digest(config), started_unix_ns=time.time_ns(),
                  requests=[], samples=[], phase_records=[], qualification={}, metrics={},
                  include_in_performance=kind != "quality",
                  source=dict(module=str(Path(__file__).resolve()), sha256=file_sha256(__file__)))
    manifest = out / "run.json"
    # Exclusive initial creation also protects against another fresh launcher.
    with manifest.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
    try:
        body(config, out, record, manifest)
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = ("interrupted" if isinstance(error, KeyboardInterrupt) else
                            "oom" if type(error).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError") else "failed")
        record["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        record["finished_unix_ns"] = time.time_ns()
        record["phase_summary"] = summarize_phases(record["phase_records"])
        _write(manifest, record)
    return record


def _encode(pipe, case, torch, dtype, device):
    with torch.inference_mode():
        return pipe.encode_prompt(prompt=case["prompt"], negative_prompt=case["negative_prompt"],
            do_classifier_free_guidance=True, num_videos_per_prompt=1, max_sequence_length=512,
            device=device, dtype=getattr(torch, dtype))


def _array_info(array):
    import numpy as np
    return dict(shape=list(array.shape), dtype=str(array.dtype), finite=bool(np.isfinite(array).all()),
                sha256=array_sha256(array))


def _embedding_info(pair):
    arrays = _embedding_arrays(pair)
    metadata = {name: _array_info(array) for name, array in arrays.items()}
    if set(arrays) != {"prompt", "negative"} or not all(row["finite"] and row["shape"] == [1, 512, 4096] for row in metadata.values()):
        raise RuntimeError("embedding must contain finite prompt/negative arrays with Wan's fixed shape")
    return arrays, metadata


def _cpu_micro(config, out, record, manifest):
    torch = _runtime(config, record, gpu=False)
    import numpy as np
    recorder = PhaseRecorder(device="cpu", nvtx=False)
    record["phase_records"] = recorder.records
    with recorder.phase("cold_encoder_load", purpose="cold_load"):
        pipe = _standalone_encoder(config["model_path"], "float32", "cpu")
    hashes = {}
    for name in config["prompt_cases"]:
        case = prompt_case(name)
        for repeat in range(-config["warmups"], config["repeats"]):
            purpose = "warmup" if repeat < 0 else "cpu_micro_measurement"
            with recorder.phase("text_encode", purpose=purpose, case=name, repeat=repeat) as phase:
                pair = _encode(pipe, case, torch, "float32", "cpu")
            arrays, metadata = _embedding_info(pair)
            digest = {key: item["sha256"] for key, item in metadata.items()}
            if repeat >= 0:
                previous = hashes.setdefault(name, digest)
                sample = dict(case=name, repeat=repeat, purpose=purpose, encode_ms=phase["wall_ms"],
                    process_cpu_ms=phase["process_cpu_ms"], process_average_cores=phase["process_average_cores"],
                    embedding=metadata, repeated_exact=digest == previous, padded_tokens=512)
                if repeat == 0:
                    path = out / "embeddings" / (name + ".npz")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, **arrays)
                    sample["artifact"] = dict(path=str(path), sha256=file_sha256(path))
                record["samples"].append(sample)
                _write(manifest, record)
            del pair, arrays
    record["metrics"] = {name: dict(samples=len(rows), encode_ms_median=statistics.median(row["encode_ms"] for row in rows),
        process_average_cores=sum(row["process_cpu_ms"] for row in rows) / sum(row["encode_ms"] for row in rows))
        for name in config["prompt_cases"] if (rows := [row for row in record["samples"] if row["case"] == name])}
    record["qualification"] = dict(finite_and_shape=True,
        repeated_exact=all(row["repeated_exact"] for row in record["samples"]),
        cuda_initialized=bool(torch.cuda.is_initialized()), numerical_cross_cpu_count="not_assumed")
    if record["qualification"]["cuda_initialized"]:
        raise RuntimeError("CPU microbenchmark unexpectedly initialized CUDA")


def run_cpu_micro(config, out):
    """Pure CPU UMT5; one explicit physical-core mask per fresh process."""
    return _run("cpu_micro", config, out, _cpu_micro)


def bounded_prefetch(items, encode, consume, depth, *, events=None, dispose=None, on_complete=None, join_timeout_s=120.):
    """Single encoder producer; depth bounds *in-flight plus ready* payloads.

    The current consumer is excluded from the depth. No model/global CUDA state
    is touched here, allowing deterministic CPU-only lifecycle regression tests.
    Optional arrival_offset_ms on each item is honored before encoding begins.
    """
    if type(depth) is not int or depth not in (0, 1, 2):
        raise ValueError("depth must be 0, 1 or 2")
    items = list(items)
    events = [] if events is None else events
    dispose = dispose or (lambda value: None)
    started = time.perf_counter_ns()
    for item in items:
        offset = item.get("arrival_offset_ms", 0)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0:
            raise ValueError("arrival_offset_ms must be finite and nonnegative")
    lock, stop, done = threading.Lock(), threading.Event(), threading.Event()
    ready, slots = queue.Queue(maxsize=max(1, depth)), threading.BoundedSemaphore(max(1, depth))
    errors, results, encoder_threads = [], [], set()
    outstanding = maximum = 0

    def event(name, index, **extra):
        with lock:
            events.append(dict(name=name, index=index, timestamp_ns=time.perf_counter_ns(),
                               thread_id=threading.get_ident(), outstanding=outstanding, **extra))

    def arrive(item):
        target = started + int(item.get("arrival_offset_ms", 0) * 1e6)
        while not stop.is_set():
            delay = (target - time.perf_counter_ns()) / 1e9
            if delay <= 0:
                return True
            stop.wait(min(delay, .05))
        return False

    def encoded(index, item):
        if not arrive(item):
            return None
        before = time.perf_counter_ns()
        encoder_threads.add(threading.get_ident())
        event("encode_start", index)
        payload = encode(item)
        after = time.perf_counter_ns()
        event("encode_end", index)
        return (index, item, payload, before, after)

    def producer():
        nonlocal outstanding, maximum
        packet = None
        try:
            for index, item in enumerate(items):
                while not stop.is_set() and not slots.acquire(timeout=.05):
                    pass
                if stop.is_set():
                    break
                with lock:
                    outstanding += 1
                    maximum = max(maximum, outstanding)
                event("prefetch_slot", index)
                packet = encoded(index, item)
                if packet is None:
                    break
                while not stop.is_set():
                    try:
                        ready.put(packet, timeout=.05)
                        event("ready", index)
                        packet = None
                        break
                    except queue.Full:
                        pass
                if stop.is_set():
                    break
        except BaseException as error:
            errors.append(error)
            event("encode_failed", -1, error_type=type(error).__name__)
        finally:
            if packet is not None:
                dispose(packet[2])
            done.set()

    thread = threading.Thread(target=producer, name="serving-cpu-encoder", daemon=True) if depth else None
    if thread:
        thread.start()
    try:
        for index, item in enumerate(items):
            waiting = time.perf_counter_ns()
            if depth:
                while True:
                    if errors:
                        raise errors[0]
                    try:
                        packet = ready.get(timeout=.05)
                        break
                    except queue.Empty:
                        if done.is_set():
                            if errors:
                                raise errors[0]
                            raise RuntimeError("encoder finished before all requests were produced")
                with lock:
                    outstanding -= 1
                slots.release()
            else:
                packet = encoded(index, item)
            actual_index, actual_item, payload, encode_start, encode_end = packet
            if actual_index != index:
                dispose(payload)
                raise RuntimeError("prefetch changed FIFO request order")
            consumed = time.perf_counter_ns()
            event("consume_start", index)
            try:
                output = consume(actual_item, payload)
            finally:
                dispose(payload)
            finished = time.perf_counter_ns()
            event("consume_end", index)
            results.append(dict(index=index, encode_start_ns=encode_start, encode_end_ns=encode_end,
                consume_start_ns=consumed, complete_ns=finished, encode_ms=(encode_end-encode_start)/1e6,
                wait_ready_ms=(consumed-waiting)/1e6,
                response_ms=(finished-started)/1e6-item.get("arrival_offset_ms", 0), result=output))
            if on_complete is not None:
                on_complete(results[-1])
            packet = payload = None
    finally:
        stop.set()
        if thread:
            thread.join(timeout=join_timeout_s)
        while True:
            try:
                dispose(ready.get_nowait()[2])
            except queue.Empty:
                break
        if thread and thread.is_alive():
            raise RuntimeError("encoder thread did not stop; terminate this isolated experiment process")
    if errors:
        raise errors[0]
    finished = time.perf_counter_ns()
    return dict(requests=results, events=events, started_ns=started, finished_ns=finished,
                wall_ms=(finished-started)/1e6, max_prefetch_outstanding=maximum,
                encoder_thread_ids=sorted(encoder_threads), depth=depth)


def _make_spec(case_name, frames, seed, request_id, steps):
    result = _spec(prompt_case(case_name), frames, steps, request_id)
    result["seed"] = seed
    return result


def _video(worker, spec, payload, recorder, metadata):
    state = None
    try:
        if payload is None:
            with recorder.phase("text_encode", **metadata):
                payload = worker.encode_prompt(spec)
        with recorder.phase("wan_prepare", **metadata):
            state = worker.prepare_encoded(spec, payload)
        for step in range(spec["steps"]):
            with recorder.phase("dit_step", step=step, **metadata):
                worker.step(state)
        video = _decode_separate(worker, state, recorder, metadata)
        worker.torch.cuda.synchronize(worker.device)
        latent = state.latents.detach().float().cpu().numpy()
        arrays = dict(latent=latent, video=video)
        expected = dict(latent=[1, 16, (spec["frames"]-1)//4+1, 60, 104], video=[1, spec["frames"], 480, 832, 3])
        info = {name: _array_info(array) for name, array in arrays.items()}
        if not all(row["finite"] and row["shape"] == expected[name] for name, row in info.items()):
            raise RuntimeError("invalid full Wan output shape or nonfinite output")
        return arrays, info
    finally:
        if state is not None:
            worker.release(state)


def _prefetch(config, out, record, manifest):
    torch = _runtime(config, record, gpu=True)
    from ..models.wan.runtime import WanStepWorker
    cpu_recorder, gpu_recorder = PhaseRecorder("cpu"), PhaseRecorder(config["device"])
    sampler = NvmlSampler(devices=(_nvml_device(config),), interval_s=config["sample_interval"])
    record["queues"], record["events"] = [], []
    sampler.start()
    try:
        with gpu_recorder.phase("cold_model_load", purpose="cold_load"):
            worker = WanStepWorker(config["device"], config["model_path"], config["attention"])
        for frames in config["frames"]:
            for repeat in range(config["warmups"]):
                spec = _make_spec(config["prompt_cases"][0], frames, config["seed"], "warm-f%d-r%d" % (frames, repeat), config["steps"])
                arrays, _ = _video(worker, spec, None, gpu_recorder, dict(purpose="warmup", frames=frames, repeat=repeat))
                del arrays
        reference_hashes = {}
        for repeat in range(config["repeats"]):
            items = []
            for index in range(config["requests"]):
                frames = config["frames"][index % len(config["frames"])]
                case = config["prompt_cases"][index % len(config["prompt_cases"])]
                spec = _make_spec(case, frames, config["seed"]+index, "p3-r%d-i%d" % (repeat, index), config["steps"])
                items.append(dict(spec=spec, case=case, repeat=repeat, index=index, arrival_offset_ms=0.))
            def encode(item):
                with cpu_recorder.phase("text_encode", purpose="prefetch_measurement", repeat=repeat,
                                         depth=config["depth"], request_id=item["spec"]["request_id"]):
                    return worker.encode_prompt(item["spec"])
            def consume(item, payload):
                spec = item["spec"]
                arrays, info = _video(worker, spec, payload, gpu_recorder,
                    dict(purpose="prefetch_measurement", repeat=repeat, depth=config["depth"],
                         request_id=spec["request_id"], frames=spec["frames"]))
                del arrays
                hashes = {name: value["sha256"] for name, value in info.items()}
                repeated_exact = reference_hashes.setdefault(item["index"], hashes) == hashes
                if not repeated_exact:
                    raise RuntimeError("prefetch repetition changed video or latent hashes")
                return dict(request_id=spec["request_id"], spec=spec, spec_sha256=_digest(spec),
                            case=item["case"], frames=spec["frames"], repeat=repeat,
                            outputs=info, repeated_exact=repeated_exact)
            events = []
            record["events"].append(dict(repeat=repeat, events=events))
            def save_completed(row):
                record["requests"].append(dict({key: value for key, value in row.items() if key != "result"}, **row["result"]))
                _write(manifest, record)
            run = bounded_prefetch(items, encode, consume, config["depth"], events=events,
                                   on_complete=save_completed, join_timeout_s=config["producer_join_timeout_s"])
            record["queues"].append(dict(repeat=repeat, depth=config["depth"], requests=len(items),
                wall_ms=run["wall_ms"], videos_per_second=len(items)*1000/run["wall_ms"],
                max_prefetch_outstanding=run["max_prefetch_outstanding"], encoder_thread_ids=run["encoder_thread_ids"]))
            _write(manifest, record)
        record["qualification"] = dict(finite_and_shape=True, repeated_exact=True,
            cross_depth_exact="requires comparison by request_id/spec_sha256 across runs",
            one_encoder_thread=all(len(row["encoder_thread_ids"]) == 1 for row in record["queues"]))
        record["metrics"] = dict(wall_ms_median=statistics.median(row["wall_ms"] for row in record["queues"]),
            throughput_median=statistics.median(row["videos_per_second"] for row in record["queues"]),
            timing_scope="all requests queued at start; CPU encoding, GPU inference, hashes and final producer shutdown included")
    finally:
        sampler.stop()
        record["samples"], record["sampling_errors"] = sampler.samples, sampler.errors
        record["phase_records"] = sorted(cpu_recorder.records + gpu_recorder.records, key=lambda row: row["start_ns"])
        record["sampling_summary"] = sampler.summary()
        record["runtime"]["cuda_initialized"] = bool(torch.cuda.is_initialized())


def run_prefetch(config, out):
    """One fixed core count and depth; masks and specs are retained for pairing."""
    return _run("prefetch", config, out, _prefetch)


def quality_recipes(config):
    for case in config["prompt_cases"]:
        for seed in config["seeds"]:
            for frames in config["frames"]:
                identifier = "%s-s%d-f%d" % (case, seed, frames)
                spec = _make_spec(case, frames, seed, "quality-" + identifier, config["steps"])
                yield dict(id=identifier, case=case, seed=seed, frames=frames, spec=spec, spec_sha256=_digest(spec))


def _save_arrays(directory, arrays, info, *, save_videos=False):
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    output = {}
    for name, array in arrays.items():
        if name == "video" and not save_videos:
            output[name] = dict(info[name], retained=False)
            continue
        path = directory / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        output[name] = dict(info[name], path=str(path), file_sha256=file_sha256(path))
        if name == "video":
            output[name]["retained"] = True
    return output


def _quality_embeddings(config, out, record, variant, torch, recorder):
    import numpy as np
    device = "cpu" if variant == "T0" else config["device"]
    dtype = PRECISIONS[variant]
    allocated_before = int(torch.cuda.memory_allocated(config["device"])) if device != "cpu" else None
    pipe = None
    completed = False
    model_tensor_refs = []
    try:
        with recorder.phase("cold_encoder_load", purpose="quality", variant=variant):
            pipe = _standalone_encoder(config["model_path"], dtype, device)
        if device != "cpu":
            # Weak references distinguish live weights from persistent CUDA
            # library workspaces, which need not return to an initial zero.
            model_tensor_refs = [weakref.ref(tensor) for tensor in pipe.text_encoder.parameters()]
            model_tensor_refs.extend(weakref.ref(tensor) for tensor in pipe.text_encoder.buffers())
        for name in config["prompt_cases"]:
            with recorder.phase("text_encode", purpose="quality", variant=variant, case=name):
                pair = _encode(pipe, prompt_case(name), torch, dtype, device)
            arrays, info = _embedding_info(pair)
            directory = out / "embeddings" / variant
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (name + ".npz")
            np.savez(path, **arrays)
            # Reuse BF16 representability and finite-FP16 checks from run01.
            validated = load_embeddings(path, dtype)
            comparison = {"status": "reference"}
            if variant != "T0":
                expected = load_embeddings(out / "embeddings/T0" / (name + ".npz"), "float32")
                comparison = dict(status="compared",
                    encoder_output={key: compare_arrays(value, expected[key]) for key, value in validated.items()},
                    dit_fp16_input={key: compare_arrays(value.astype(np.float16), expected[key].astype(np.float16)) for key, value in validated.items()})
                del expected
            record["embeddings"].append(dict(variant=variant, case=name, source_dtype=dtype, storage_dtype="float32",
                artifact=dict(path=str(path), file_sha256=file_sha256(path)), arrays=info, comparison=comparison,
                bf16_storage_exact=True if variant == "T2" else None, finite_fp16_input=True))
            del pair, arrays, validated
        completed = True
    finally:
        del pipe
        gc.collect()
        if device != "cpu":
            torch.cuda.synchronize(config["device"])
            torch.cuda.empty_cache()
            allocated_after = int(torch.cuda.memory_allocated(config["device"]))
            live_tensors = sum(reference() is not None for reference in model_tensor_refs)
            record["model_lifecycle"].append(dict(event="encoder_released_before_wan", variant=variant,
                allocated_before=allocated_before, allocated_bytes=allocated_after,
                tracked_model_tensors=len(model_tensor_refs), live_model_tensors=live_tensors,
                weights_released=live_tensors == 0 if model_tensor_refs else None, timestamp_ns=time.perf_counter_ns()))
            if completed and live_tensors:
                raise RuntimeError("GPU encoder model tensors remain live after release; refusing to load Wan")


def _quality(config, out, record, manifest):
    torch = _runtime(config, record, gpu=True)
    from ..models.wan.runtime import WanStepWorker, EncodedPrompt
    recorder = PhaseRecorder(config["device"])
    record["phase_records"] = recorder.records
    record["embeddings"], record["model_lifecycle"] = [], []
    record["quality_scope"] = "numerical screening only; semantic/perceptual quality remains unverified"
    # GPU FP32 UMT5 must be gone before loading Wan; no simultaneous residency.
    for variant in PRECISIONS:
        _quality_embeddings(config, out, record, variant, torch, recorder)
        _write(manifest, record)
    with recorder.phase("cold_model_load", purpose="quality"):
        worker = WanStepWorker(config["device"], config["model_path"], config["attention"])
    for recipe in quality_recipes(config):
        spec = recipe["spec"]
        metadata = dict(purpose="quality", case=recipe["case"], seed=recipe["seed"], frames=recipe["frames"])
        # Independent direct CPU encoding is the baseline, not the stored input.
        reference, info = _video(worker, spec, None, recorder, dict(metadata, variant="cpu_direct"))
        baseline = dict(recipe, variant="cpu_direct", purpose="quality_baseline", include_in_performance=False,
            status="reference", artifacts=_save_arrays(out / "outputs/cpu_direct" / recipe["id"], reference, info,
                                                       save_videos=config["save_videos"]))
        record["requests"].append(baseline)
        _write(manifest, record)
        for variant, dtype in PRECISIONS.items():
            arrays = load_embeddings(out / "embeddings" / variant / (recipe["case"] + ".npz"), dtype)
            pair = {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}
            payload = EncodedPrompt(spec=spec, prompt_embeds=pair["prompt"], negative_prompt_embeds=pair["negative"],
                cpu_prompt_encode_ms=0., owner=worker._owner)
            actual, info = _video(worker, spec, payload, recorder, dict(metadata, variant=variant))
            comparison = {name: compare_arrays(value, reference[name]) for name, value in actual.items()}
            row = dict(recipe, variant=variant, purpose="cpu_bridge" if variant == "T0" else "quality_replay",
                include_in_performance=False, status="compared", comparison=comparison,
                artifacts=_save_arrays(out / "outputs" / variant / recipe["id"], actual, info,
                                       save_videos=config["save_videos"]))
            if variant == "T0":
                row["baseline_bridge_exact"] = all(item.get("exact") is True for item in comparison.values())
                if not row["baseline_bridge_exact"]:
                    row["status"] = "baseline_bridge_failed"
            record["requests"].append(row)
            _write(manifest, record)
            del arrays, pair, payload, actual
            if variant == "T0" and not row["baseline_bridge_exact"]:
                raise RuntimeError("CPU bridge failed for " + recipe["id"] + "; subsequent precision attribution refused")
        del reference
    recipes = len(list(quality_recipes(config)))
    record["qualification"] = dict(finite_and_shape=True, cpu_bridge_exact=True,
        requested_recipes=recipes, completed_recipes=recipes, video_inferences=len(record["requests"]),
        precision_equivalence="not_assumed", semantic_quality="unverified",
        numerical_threshold="differences reported; no automatic perceptual quality threshold")


def run_quality(config, out):
    """Default: 2 cases x 2 seeds x 2 frame counts x 4 paths = 32 videos."""
    return _run("quality", config, out, _quality)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="CPU-only config audit; no output/model/runtime creation")
    parser.add_argument("--save-videos", action="store_true", help="Retain raw video arrays for manual quality review")
    args = parser.parse_args(argv)
    raw_config = json.loads(args.config.read_text())
    if args.save_videos:
        raw_config["save_videos"] = True
    config = validate_config(args.kind, raw_config)
    if args.validate_only:
        print(json.dumps(dict(kind=args.kind, status="validated", configuration=config), ensure_ascii=False))
        return 0
    if args.out is None:
        parser.error("--out is required unless --validate-only is used")
    function = dict(quality=run_quality, cpu_micro=run_cpu_micro, prefetch=run_prefetch)[args.kind]
    result = function(config, args.out)
    print(json.dumps(dict(status=result["status"], output=str(args.out / "run.json")), ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
