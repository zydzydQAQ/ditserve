"""Conditional FP32 V100 encoding against the qualified eager paired-K4 path."""
from pathlib import Path
import time
import types

from ..research.common import atomic_json, file_digest


def paired_generate(context, evidence, request, encode):
    """Only the encoder changes; both arms execute the current cache policy."""
    from .cache_online import cache_spec
    from .cache_executor import PairedCacheExecutor
    from .stage_phases import _record_full_warmup
    worker = context.worker
    if worker._states:
        raise ValueError("local encoder generation requires isolated request ownership")
    executor = PairedCacheExecutor(worker)
    state = continuation = None
    request = cache_spec(request)
    started = time.perf_counter()
    try:
        context.check_budget()
        payload = encode(request)
        encode_ms = payload.cpu_prompt_encode_ms
        state = worker.prepare_encoded(request, payload)
        continuation = executor.attach(state)
        while state.step_index < request["steps"]:
            context.check_budget()
            executor.advance(continuation, blocks=4, check_budget=context.check_budget)
            evidence.data["active_unit"]["completed_steps"] = state.step_index
        array = worker.finish(state)
        worker._sync()
        completed = state.step_index
        result = dict(encode_ms=encode_ms, generation_ms=(time.perf_counter()-started)*1000,
            frames=int(array.shape[1]), completed_steps=completed, memory=worker._memory(),
            completed=True, cache="paired_008", policy="block-4", wan_backend="eager")
    finally:
        if continuation is not None and not continuation.closed:
            executor.close(continuation)
        if state is not None:
            worker.release(state)
        # prepare_encoded can fail after registering a partially prepared
        # private state. No pre-existing state was admitted above.
        for orphan in list(worker._states.values()):
            worker.release(orphan)
    _record_full_warmup(context, request, array, completed, "next_v100_paired_generation")
    return array, result


def run_encoder(context):
    """Preserve the established local protocol, replacing only its Wan executor.

    The caller must admit and bill both cards for the entire phase. Placement
    occurs outside trace clocks; prompts still arrive before their encoding.
    This result never selects a combined VAE/encoder default.
    """
    from . import stage_phases
    from .next_conditional import online_callback
    if context.metadata.get("encoder_device") != "cuda:1":
        raise ValueError("conditional encoder requires the separately admitted V100")
    devices = getattr(context, "admitted_devices", [])
    if devices != [context.config["gpu_uuid"], context.config["v100_uuid"]]:
        raise ValueError("conditional encoder devices differ from charged admission")
    if context.worker._states:
        raise ValueError("encoder pilot requires no live request state")
    context.sana.set_candidate("eager")
    original_callback = getattr(context, "run_online", None)

    def callback(ctx, result, options):
        online_callback(ctx, result, options, filename="encoder-online.json",
                        candidate_label="v100_fp32_encoder")

    context.run_online = callback
    # An isolated globals mapping avoids changing the qualified historical
    # module or touching another request's manager/RNG state.
    dependencies = dict(stage_phases.run_text_encoder.__globals__, generate=paired_generate)
    run = types.FunctionType(stage_phases.run_text_encoder.__code__, dependencies,
                             stage_phases.run_text_encoder.__name__)
    path = Path(context.output) / "conditional-text-encoder.json"
    try:
        result = run(context)
        native = Path(context.output) / "text_encoder.json"
        summary = {key: value for key, value in result.items() if key not in ("rows", "pairs")}
        summary.update(local_evidence=dict(path=str(native), sha256=file_digest(native)),
            recorded_rows=len(result.get("rows", [])), recorded_pairs=len(result.get("pairs", [])),
            candidate="v100_fp32_encoder", baseline="CPU FP32 encoder + eager paired_008 K4 + eager Sana",
            scope="independent encoder comparison; no implicit composition with other candidates")
        summary["decision"] = ("值得继续" if result.get("online_qualified") else
            "当前配置无收益" if result.get("status") == "passed" or
            result.get("reason") == "encoder_quality_failed" else "未验证")
        atomic_json(path, summary)
        return summary
    finally:
        if original_callback is None:
            del context.run_online
        else:
            context.run_online = original_callback
