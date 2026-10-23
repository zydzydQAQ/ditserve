"""Replay saved T1/T2 embeddings through Wan for numerical validation only.

The GPU text encoder is absent from this process. CPU FP32 arrays saved by the
encoder study are consumed directly; no prompt is encoded again. Timing from
this replay is deliberately excluded from normal inference performance.

    python -m ditserve.evaluation.workflow_embedding_replay \
      --encoder-root /data/workflow/run01/encoder-v2 \
      --out /data/workflow/run01/embedding-replay --variants T1,T2 \
      --model-path /models/wan2.1-t2v-1.3b --frames 17,33 --steps 20 --device cuda:1

``--validate-only`` audits source metadata, hashes and CPU array dtypes without
importing torch, creating a worker or initializing CUDA. A normal run first
replays the saved T0 embeddings and requires an exact match to the normal T0
video/latent, establishing the replay bridge before comparing T1/T2.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from .workflow_encoder import (PROMPTS, _spec, _decode_separate, compare_arrays,
                               _array_evidence, _video_reference_metadata, _verify_video_reference)


SCHEMA = "ditserve.workflow.embedding_replay.v1"
VARIANTS = {"T0": "float32", "T1": "float32", "T2": "bfloat16"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    view = memoryview(array).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[offset:offset + 8 * 1024 * 1024])
    return digest.hexdigest()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_metadata(record, variant, *, frames, steps):
    """Reject ambiguous recipe/precision provenance before loading a model."""
    if record.get("schema") != "ditserve.workflow.encoder.v1" or record.get("variant") != variant:
        raise ValueError("unexpected encoder result schema/variant for " + variant)
    if record.get("status") != "completed":
        raise ValueError("encoder result is not completed: " + variant)
    if record.get("configuration", {}).get("encoder_dtype") != VARIANTS[variant]:
        raise ValueError("encoder dtype provenance disagrees with variant " + variant)
    if record.get("matmul_tf32") is not False or type(record.get("cudnn_tf32")) is not bool:
        raise ValueError("missing or incompatible explicit TF32 policy for " + variant)
    english = [row for row in record.get("numeric", []) if row.get("case") == "english"]
    if len(english) != 1 or english[0].get("finite") is not True:
        raise ValueError("missing unique finite English embedding source for " + variant)
    if variant == "T0":
        if record.get("steps") != steps or not set(frames).issubset(record.get("frames", [])):
            raise ValueError("T0 baseline frame/step recipe does not match replay")
        if not record.get("configuration", {}).get("video"):
            raise ValueError("T0 source did not include real videos")
        for count in frames:
            rows = [row for row in record.get("requests", []) if row.get("kind") == "video"
                    and row.get("frames") == count and row.get("repeat") == 0]
            if len(rows) != 1 or rows[0].get("finite") is not True or set(rows[0].get("sha256", {})) != {"latent", "video"}:
                raise ValueError("missing unique normal T0 repeat-zero output for frames=" + str(count))
    return english[0]


def load_embeddings(path, source_dtype):
    """Read the study's FP32 storage without introducing extra rounding.

    BF16 values are saved as exact FP32 expansions by _embedding_arrays. Keeping
    those values in FP32 until Wan's mandatory FP16 conversion is equivalent to
    BF16 -> FP16, and satisfies the normal worker's CPU FP32 payload contract.
    """
    import numpy as np
    if source_dtype not in ("float32", "bfloat16"):
        raise ValueError("unsupported source embedding dtype: " + source_dtype)
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"prompt", "negative"}:
            raise ValueError("CFG=5 replay requires exactly prompt and negative embeddings")
        arrays = {name: np.array(stored[name], copy=True, order="C") for name in stored.files}
    for name, array in arrays.items():
        if array.dtype != np.dtype("float32"):
            raise ValueError(name + " embedding storage must be FP32, got " + str(array.dtype))
        if array.shape != (1, 512, 4096):
            raise ValueError("unexpected Wan1.3B embedding shape: " + str(array.shape))
        if not np.isfinite(array).all():
            raise ValueError("nonfinite saved " + name + " embedding")
        if source_dtype == "bfloat16" and np.any(array.view(np.uint32) & np.uint32(0xffff)):
            raise ValueError("BF16 source has FP32 values not exactly representable in BF16")
        if not np.isfinite(array.astype(np.float16)).all():
            raise ValueError("saved embedding overflows the DiT FP16 input dtype")
    return arrays


def _array_metadata(path, array):
    result = _array_evidence(array)
    if path is None:
        return dict(result, retained=False)
    return dict(result, path=str(path), file_sha256=file_sha256(path), retained=True)


def _save_replay_array(output, name, array, *, save_videos=False):
    """Keep all comparison evidence; raw video retention is explicitly opt-in."""
    import numpy as np
    if name == "video" and not save_videos:
        return _array_metadata(None, array)
    path = output / (name + ".npy")
    np.save(path, array, allow_pickle=False)
    return _array_metadata(path, array)


def _replay_video_reference(video, reference, variant):
    """The existing T0 replay supplies missing raw bytes only after hash proof."""
    if not isinstance(reference, dict):
        return reference
    if variant != "T0":
        raise ValueError("T0 video reference must pass the exact bridge before candidate comparison")
    _verify_video_reference(video, reference)
    return video.copy()


def audit_sources(args):
    """CPU-only audit; unavailable requested variants remain explicit results."""
    import numpy as np
    root = args.encoder_root.resolve()
    metadata, arrays, sources, unavailable = {}, {}, {}, []
    for variant in ("T0", *args.variants):
        result_path = root / variant / "result.json"
        embedding_path = root / variant / "numeric/english.npz"
        if variant != "T0" and not result_path.is_file():
            unavailable.append(dict(variant=variant, status="unavailable", reason="result_missing", path=str(result_path)))
            continue
        record = json.loads(result_path.read_text())
        if variant != "T0" and record.get("status") != "completed":
            unavailable.append(dict(variant=variant, status="unavailable", reason="encoder_" + str(record.get("status")),
                                    source_result=dict(path=str(result_path), file_sha256=file_sha256(result_path))))
            continue
        validate_metadata(record, variant, frames=args.frames, steps=args.steps)
        if record["cudnn_tf32"] != metadata.get("T0", record)["cudnn_tf32"]:
            raise ValueError("variant TF32 policy differs from normal T0 source")
        metadata[variant] = record
        arrays[variant] = load_embeddings(embedding_path, VARIANTS[variant])
        sources[variant] = dict(
            result=dict(path=str(result_path), file_sha256=file_sha256(result_path)),
            embedding=dict(path=str(embedding_path), file_sha256=file_sha256(embedding_path),
                           storage_dtype="float32", source_encoder_dtype=VARIANTS[variant],
                           arrays={name: dict(shape=list(value.shape), array_sha256=array_sha256(value))
                                   for name, value in arrays[variant].items()}))
    references, baseline_sources = {}, {}
    for frames in args.frames:
        row = next(row for row in metadata["T0"]["requests"] if row.get("kind") == "video"
                   and row.get("frames") == frames and row.get("repeat") == 0)
        references[frames], baseline_sources[str(frames)] = {}, {}
        for name in ("latent", "video"):
            path = root / "T0/outputs" / str(frames) / (name + ".npy")
            if name == "video" and not path.is_file():
                source = dict(_video_reference_metadata(row), retained=False,
                              verification="requires_exact_T0_replay_before_numerical_comparison")
                references[frames][name] = dict(source)
                baseline_sources[str(frames)][name] = source
                continue
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.dtype("float32") or not np.isfinite(array).all():
                raise ValueError("normal T0 output must be finite FP32: " + str(path))
            expected_shape = ((1, 16, (frames-1)//4+1, 60, 104) if name == "latent"
                              else (1, frames, 480, 832, 3))
            if array.shape != expected_shape or (name == "video" and row.get("shape") != list(expected_shape)):
                raise ValueError("normal T0 artifact shape disagrees with fixed Wan recipe: " + str(path))
            source = _array_metadata(path, array)
            if source["array_sha256"] != row["sha256"][name]:
                raise ValueError("normal T0 artifact differs from its result.json SHA256: " + str(path))
            references[frames][name] = array
            baseline_sources[str(frames)][name] = source
    case_json = json.dumps(PROMPTS[0], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    provenance = dict(variants=sources, normal_t0_outputs=baseline_sources,
        unavailable=unavailable, prompt_case=dict(PROMPTS[0], sha256=hashlib.sha256(case_json.encode()).hexdigest()),
        recipe_helper="workflow_encoder._spec(PROMPTS[0], frames, steps, request_id)",
        recipe_source_sha256=file_sha256(Path(__file__).with_name("workflow_encoder.py")))
    return metadata, arrays, references, provenance


class NumericalRanges:
    """Trace labels only; no performance samples are emitted by this replay."""
    @contextlib.contextmanager
    def phase(self, name, **metadata):
        import torch
        label = "ditserve.workflow/embedding_replay/" + name + "|" + json.dumps(
            dict(metadata, purpose="numerical_replay", include_in_performance=False), sort_keys=True)
        torch.cuda.nvtx.range_push(label)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def run_replay(args, metadata, arrays, references, record, manifest):
    if args.cpu_affinity == "auto":
        from .workflow_overlap import bind_cpu_to_gpu
        record["cpu_affinity"] = bind_cpu_to_gpu(args.device, args.cpu_threads)
    else:
        record["cpu_affinity"] = dict(policy="none", bound_cpus=sorted(os.sched_getaffinity(0)))
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    import numpy as np
    import torch
    from ..models.wan.runtime import WanStepWorker, EncodedPrompt
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = metadata["T0"]["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = metadata["T0"]["cudnn_tf32"]
    torch.cuda.set_device(args.device)
    worker = WanStepWorker(args.device, str(args.model_path), args.attention)
    # Guard against accidentally converting this numerical replay into another
    # encoding benchmark. All inputs must come from the audited NPZ files.
    def no_encoding(*_, **__):
        raise RuntimeError("embedding replay must never invoke prompt encoding")
    worker.encode_prompt = no_encoding
    worker.pipe.encode_prompt = no_encoding
    recorder = NumericalRanges()
    record["runtime"] = dict(torch=torch.__version__, cuda=torch.version.cuda,
        matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        gpu_name=torch.cuda.get_device_name(args.device), encoder_execution_calls=0,
        model_path=str(args.model_path.resolve()), attention=args.attention)
    for frames in args.frames:
        for variant in ("T0", *(value for value in args.variants if value in arrays)):
            spec = _spec(PROMPTS[0], frames, args.steps, "embedding-replay-%s-f%d" % (variant, frames))
            pair = {name: torch.from_numpy(value.copy()) for name, value in arrays[variant].items()}
            payload = EncodedPrompt(spec=spec, prompt_embeds=pair["prompt"], negative_prompt_embeds=pair["negative"],
                                    cpu_prompt_encode_ms=0., owner=worker._owner)
            context = dict(variant=variant, frames=frames)
            state = None
            try:
                with recorder.phase("wan_prepare", **context):
                    state = worker.prepare_encoded(spec, payload)
                for step in range(args.steps):
                    with recorder.phase("dit_step", step=step, **context):
                        worker.step(state)
                video = _decode_separate(worker, state, recorder, context)
                torch.cuda.synchronize(args.device)
                latent = state.latents.detach().float().cpu().numpy()
                output = args.out / variant / str(frames)
                output.mkdir(parents=True, exist_ok=True)
                result = dict(variant=variant, frames=frames, steps=args.steps, spec=spec,
                    purpose="baseline_bridge_check" if variant == "T0" else "numerical_replay",
                    include_in_performance=False, status="compared", artifacts={}, comparison={})
                for name, value in (("latent", latent), ("video", video)):
                    if not np.isfinite(value).all():
                        raise RuntimeError("nonfinite replay " + name)
                    result["artifacts"][name] = _save_replay_array(output, name, value,
                                                                 save_videos=args.save_videos)
                    if name == "video":
                        references[frames][name] = _replay_video_reference(value, references[frames][name], variant)
                    result["comparison"][name] = compare_arrays(value, references[frames][name])
                result["embedding_comparison"] = {
                    "encoder_output": {name: compare_arrays(value, arrays["T0"][name]) for name, value in arrays[variant].items()},
                    "dit_fp16_input": {name: compare_arrays(value.astype(np.float16), arrays["T0"][name].astype(np.float16))
                                       for name, value in arrays[variant].items()}}
                if variant == "T0":
                    result["baseline_bridge_exact"] = all(item["exact"] for item in result["comparison"].values())
                    if not result["baseline_bridge_exact"]:
                        result["status"] = "baseline_bridge_failed"
                record["results"].append(result)
                write_json(output / "comparison.json", result)
                write_json(manifest, record)
                if variant == "T0" and not result["baseline_bridge_exact"]:
                    raise RuntimeError("saved T0 embedding replay differs from normal T0 output; stop numerical attribution")
            finally:
                if state is not None:
                    worker.release(state)
            print(json.dumps(dict(variant=variant, frames=frames, status=result["status"],
                                  comparison=str(output / "comparison.json"))), flush=True)
        # The reconstructed T0 clip is needed only for this frame-count group.
        references[frames].pop("video", None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", default="T1,T2")
    parser.add_argument("--frames", default="17,33")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model-path", type=Path, default=Path("/models/wan2.1-t2v-1.3b"))
    parser.add_argument("--attention", choices=("flash", "math", "efficient"), default="flash")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cpu-affinity", choices=("auto", "none"), default="auto")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--save-videos", action="store_true", help="Retain raw video arrays for manual quality review")
    args = parser.parse_args(argv)
    args.variants = [value.strip() for value in args.variants.split(",")]
    if not args.variants or len(set(args.variants)) != len(args.variants) or any(value not in ("T1", "T2") for value in args.variants):
        parser.error("--variants must select unique T1 and/or T2 entries")
    try:
        args.frames = [int(value) for value in args.frames.split(",")]
    except ValueError:
        parser.error("--frames must contain comma-separated integers")
    if not args.frames or len(set(args.frames)) != len(args.frames) or any(value < 1 or (value-1)%4 for value in args.frames):
        parser.error("--frames must contain unique positive 4*k+1 frame counts")
    if args.steps < 1 or args.cpu_threads < 1 or not args.device.startswith("cuda:") or not args.device[5:].isdigit():
        parser.error("steps/threads must be positive and device must be cuda:N")
    args.out = args.out.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "run.json"
    if manifest.exists():
        raise FileExistsError("refusing to overwrite replay evidence: " + str(manifest))
    record = dict(schema=SCHEMA, status="running", purpose="numerical_replay", include_in_performance=False,
        started_unix_ns=time.time_ns(), results=[], variants=args.variants, frames=args.frames, steps=args.steps,
        precision_path="CPU FP32 saved values -> ordinary Wan FP16 input; no text encoding executed",
        equivalence_claim="Numerical differences only; no automatic visual-quality equivalence assertion",
        validate_only=args.validate_only, save_videos=args.save_videos)
    write_json(manifest, record)
    try:
        metadata, arrays, references, sources = audit_sources(args)
        record["sources"] = sources
        if args.validate_only:
            record["status"] = "validated"
        elif not any(variant in arrays for variant in args.variants):
            record["status"] = "unavailable"
        else:
            run_replay(args, metadata, arrays, references, record, manifest)
            record["status"] = "completed_with_unavailable_variants" if sources["unavailable"] else "completed"
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        record["finished_unix_ns"] = time.time_ns()
        write_json(manifest, record)
    print(json.dumps(dict(status=record["status"], manifest=str(manifest))), flush=True)
    return 0 if record["status"] in ("validated", "completed", "completed_with_unavailable_variants") else 1


if __name__ == "__main__":
    raise SystemExit(main())
