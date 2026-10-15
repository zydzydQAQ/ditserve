"""Stage-specific FP32 text encoding and functional Wan VAE compilation pilots.

GPU admission, triggering and accounting belong to the root supervisor. Imports
and CPU fixtures do not initialize CUDA. Local gains do not qualify online use.
"""

from contextlib import nullcontext
import hashlib
from pathlib import Path
import statistics
import time

from ..directions.budget import BudgetStop
from ..directions.direction1 import numerical_error, quantile
from ..util import atomic_json, utc_now


class Evidence:
    def __init__(self, context, phase):
        self.context, self.path = context, Path(context.output) / (phase + ".json")
        if self.path.exists():
            raise FileExistsError(str(self.path))
        self.data = dict(schema_version="ditserve.optimizations.stages.v1", phase=phase,
                         status="running", local_qualified=False, online_qualified=False,
                         system_qualified=False, rows=[], active_unit=None,
                         started_at=utc_now(), metadata=context.metadata,
                         quality_epsilon=.05, minimum_local_gain=.05)
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def begin(self, seconds, **unit):
        self.data["active_unit"] = dict(unit, started=False)
        self.save()
        self.context.check_budget.ensure_unit(float(seconds))
        self.context.check_budget()
        self.data["active_unit"]["started"] = True
        self.save()

    def append(self, row):
        self.data["rows"].append(row)
        self.data["active_unit"] = None
        self.save()

    def finish(self, status="passed", **values):
        self.data.update(status=status, finished_at=utc_now(), **values)
        self.save()
        return self.data

    def fail(self, error, **fields):
        local_before_failure = bool(self.data.get("local_qualified"))
        return self.finish(status="stopped" if isinstance(error, BudgetStop) else "failed",
                           local_qualified=local_before_failure, online_qualified=False,
                           local_gate_passed_before_failure=local_before_failure,
                           error=f"{type(error).__name__}: {error}",
                           budget_stopped=isinstance(error, BudgetStop), **fields)


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p95=quantile(values, .95))


def inputs(context):
    if len(context.opts["prompts"]) != 2:
        raise ValueError("stage pilots require the two frozen Wan prompts")
    return [dict(input_id=f"p{p}-s{seed}", prompt=prompt, seed=seed)
            for p, prompt in enumerate(context.opts["prompts"]) for seed in (101, 102)]


def spec(context, item, suffix):
    from ..directions.direction1 import _spec
    return _spec(context.opts, context.profile["selected_shape"],
                 f"stage-{item['input_id']}-{suffix}", item["prompt"], item["seed"])


def quality(context, reference, candidate):
    from ..gpu.quality import evaluate_quality, quality_pass
    evaluate = getattr(context, "quality_evaluator", evaluate_quality)
    result = evaluate(reference, candidate, lpips_device="cpu")
    return dict(metrics=result, passed=quality_pass(result, .05),
                numerical=numerical_error(candidate, reference),
                scope="reference-relative full 81-frame LPIPS, not semantic quality")


def retain(context, name, array):
    import numpy as np
    root = Path(context.config.get("artifact_root", Path(context.output) / "artifacts"))
    phase = context.metadata.get("phase")
    if phase not in ("text_encoder", "vae"):
        phase = "text_encoder" if name.startswith("encoder-") else "vae"
    root = root / Path(context.run_root).name / phase / Path(context.output).name
    root.mkdir(parents=True, exist_ok=True)
    path = root / (name + ".npy")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path.resolve()), sha256=digest.hexdigest(),
                shape=list(array.shape), raw_available=True)


class FP32Encoder:
    """Move one existing UMT5 model; keep request payloads CPU FP32/single-use."""
    def __init__(self, worker, device):
        if device not in ("cuda:0", "cuda:1"):
            raise ValueError("encoder GPU must be explicitly admitted cuda:0 or cuda:1")
        self.worker, self.device = worker, device
        self.current_device, self.moves = "cpu", []

    def move(self, device):
        if device not in ("cpu", self.device):
            raise ValueError("unadmitted encoder device")
        if device == self.current_device:
            return
        worker = self.worker
        started = time.perf_counter()
        # Do not change dtype or attention implementation to fit a device.
        with worker._encoder_lock:
            worker.pipe.text_encoder.to(device=device, dtype=worker.torch.float32)
            if device.startswith("cuda"):
                worker.torch.cuda.synchronize(device)
        self.current_device = device
        self.moves.append(dict(device=device, dtype="float32", ms=(time.perf_counter()-started)*1000))

    def restore_cpu(self):
        # A failed .to(cuda) can leave only some parameters moved; force .to(cpu)
        # even if the bookkeeping device still says CPU.
        with self.worker._encoder_lock:
            self.worker.pipe.text_encoder.to(device="cpu", dtype=self.worker.torch.float32)
        self.current_device = "cpu"

    def encode(self, request):
        from ..gpu.worker import EncodedPrompt, normalize_spec
        worker = self.worker
        request = normalize_spec(request)
        if self.current_device != self.device:
            raise RuntimeError("GPU encoder was not placed before the timed arm")
        with worker._encoder_lock, worker.torch.inference_mode():
            worker.torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            prompt, negative = worker.pipe.encode_prompt(
                prompt=request["prompt"], negative_prompt=request["negative_prompt"],
                do_classifier_free_guidance=request["guidance_scale"] > 1,
                num_videos_per_prompt=1, max_sequence_length=512,
                device=self.device, dtype=worker.torch.float32)
            prompt = prompt.detach().to(device="cpu", dtype=worker.torch.float32)
            negative = None if negative is None else negative.detach().to(device="cpu", dtype=worker.torch.float32)
            worker.torch.cuda.synchronize(self.device)
            worker._validate_cpu_embeddings(prompt, negative, request)
            result = EncodedPrompt(request, prompt, negative,
                                   (time.perf_counter()-started)*1000, worker._owner)
            # The old field name is retained solely for worker compatibility.
            result.encoding_device, result.encoding_dtype = self.device, "float32"
            return result


def install_encoder_defaults(context, selection):
    """Install authority-selected online defaults; the caller admits all GPUs.

    Local component probes retain their explicit CPU/preencoded scope. Placement
    changes happen only in prepare_online_arm, before that trace's clock starts.
    """
    if context.metadata.get("phase") not in ("vae", "cache_k4"):
        raise ValueError("encoder inheritance is limited to later VAE/cache phases")
    device = selection.get("encode_device", "cpu")
    if device == "cpu":
        return None
    if device == "cuda:1" and context.metadata.get("encoder_device") != "cuda:1":
        raise ValueError("inherited V100 encoder requires a separately admitted two-GPU attempt")
    encoder = FP32Encoder(context.worker, device)
    prior_prepare = getattr(context, "prepare_online_arm", None)
    prior_defaults = dict(getattr(context, "online_defaults", {}))
    defaults = dict(prior_defaults, encode_video=encoder.encode, encode_device=device,
                    encode_on_scheduler=device == context.worker.device)
    context.online_defaults = defaults
    context.metadata["inherited_encoder"] = dict(device=device, dtype="float32", payload_device="cpu",
        scope="same encoder policy in every online arm; local component probes remain CPU/preencoded",
        placement_timing="before trace origin, charged to the admitted GPU allocation")

    def prepare(options):
        if prior_prepare is not None:
            prior_prepare(options)
        encoder.move(options.get("encode_device", "cpu"))

    closed = False
    def cleanup():
        nonlocal closed
        if closed:
            return
        encoder.restore_cpu()
        if prior_prepare is None:
            if getattr(context, "prepare_online_arm", None) is prepare:
                del context.prepare_online_arm
        else:
            context.prepare_online_arm = prior_prepare
        context.online_defaults = prior_defaults
        closed = True

    context.prepare_online_arm, context.encoder_cleanup = prepare, cleanup
    return encoder


def _record_full_warmup(context, request, array, completed_steps, source):
    """Reuse only a successful complete fixed-shape generation as capacity proof."""
    if (completed_steps != 32 or request.get("steps") != 32
            or any(request.get(key) != value for key, value in (("height", 480), ("width", 832), ("frames", 81)))
            or tuple(array.shape) != (1, 81, 480, 832, 3)):
        return
    context.metadata["wan_warmed"] = True
    context.metadata["wan_warm_evidence"] = dict(source=source, phase=context.metadata.get("phase"),
        request_id=request["request_id"], completed_steps=completed_steps,
        shape=list(array.shape), scope="successful full local trajectory and decode; reuse existing co-resident capacity evidence")


def generate(context, evidence, request, encode):
    worker, state = context.worker, None
    started = time.perf_counter()
    try:
        payload = encode(request)
        encode_ms = payload.cpu_prompt_encode_ms
        state = worker.prepare_encoded(request, payload)
        while state.step_index < request["steps"]:
            context.check_budget()
            worker.step(state)
            evidence.data["active_unit"]["completed_steps"] = state.step_index
        array = worker.finish(state)
        worker._sync()
        result = dict(encode_ms=encode_ms, generation_ms=(time.perf_counter()-started)*1000,
                      frames=int(array.shape[1]), completed_steps=state.step_index,
                      memory=worker._memory(), completed=True)
    finally:
        if state is not None:
            worker.release(state)
    # Mark only after finish, synchronization, memory observation and release
    # all succeeded; an OOM or interrupted trajectory cannot create this flag.
    _record_full_warmup(context, request, array, state.step_index, "text_encoder_local_generation")
    return array, result


def _online(context, result, candidate):
    callback = getattr(context, "run_online", None)
    if callback is None:
        from .worker import _online as callback
    callback(context, result, candidate)


def run_text_encoder(context):
    evidence = Evidence(context, "text_encoder")
    worker = context.worker
    device = context.metadata.get("encoder_device", "cuda:0")
    encoder = FP32Encoder(worker, device)
    candidate_active = False
    evidence.data.update(encoder_device=device, encoder_dtype="float32",
                         payload_device="cpu", payload_dtype="float32",
                         timing_scope="request encode+denoise+decode; placement setup separately charged",
                         encode_on_scheduler=device == worker.device, needs_v100=False)
    try:
        pairs = []
        for index, item in enumerate(inputs(context)):
            # Each complete CPU/GPU quality pair is admitted together. Alternate
            # order across the four frozen inputs without changing their seeds.
            evidence.begin(360., kind="encoder_pair", input_id=item["input_id"], completed_steps=0)
            outputs, rows = {}, {}
            order = ("cpu", "gpu") if index % 2 == 0 else ("gpu", "cpu")
            for arm in order:
                context.check_budget()
                candidate_active = arm == "gpu"
                evidence.data["active_unit"].update(arm=arm, completed_steps=0)
                evidence.save()
                encoder.move("cpu" if arm == "cpu" else device)
                array, row = generate(context, evidence, spec(context, item, arm),
                                      worker.encode_prompt if arm == "cpu" else encoder.encode)
                row.update(input_id=item["input_id"], arm=arm, device="cpu" if arm == "cpu" else device,
                           artifact=retain(context, f"encoder-{item['input_id']}-{arm}", array))
                outputs[arm], rows[arm] = array, row
                evidence.data["rows"].append(row)
                evidence.save()
            check = quality(context, outputs["cpu"], outputs["gpu"])
            pair = dict(input_id=item["input_id"], order=list(order), quality=check,
                        encode_gain=1 - rows["gpu"]["encode_ms"] / rows["cpu"]["encode_ms"],
                        generation_gain=1 - rows["gpu"]["generation_ms"] / rows["cpu"]["generation_ms"])
            pairs.append(pair)
            evidence.data["pairs"] = pairs
            evidence.data["active_unit"] = None
            evidence.save()
            del outputs, array
            if not check["passed"]:
                return evidence.finish(status="stopped", reason="encoder_quality_failed", moves=encoder.moves)
        encode_gain = 1 - statistics.mean(r["encode_ms"] for r in evidence.data["rows"] if r["arm"] == "gpu") / statistics.mean(r["encode_ms"] for r in evidence.data["rows"] if r["arm"] == "cpu")
        generation_gain = 1 - statistics.mean(r["generation_ms"] for r in evidence.data["rows"] if r["arm"] == "gpu") / statistics.mean(r["generation_ms"] for r in evidence.data["rows"] if r["arm"] == "cpu")
        local = len(pairs) == 4 and all(p["quality"]["passed"] for p in pairs) and min(encode_gain, generation_gain) >= .05
        result = evidence.finish(local_qualified=local, quality_passed=True, complete_pairs=4,
            encode_gain=encode_gain, generation_gain=generation_gain, moves=encoder.moves,
            co_resident_full_generation=True,
            summaries={arm: {metric: stats([r[metric] for r in evidence.data["rows"] if r["arm"] == arm])
                             for metric in ("encode_ms", "generation_ms")} for arm in ("cpu", "gpu")},
            reason="component_gate_passed" if local else "no_local_encoder_gain")
        if local:
            previous = getattr(context, "prepare_online_arm", None)
            def prepare(options):
                nonlocal candidate_active
                if previous is not None:
                    previous(options)
                candidate_active = options.get("encode_device") == device
                encoder.move(options.get("encode_device", "cpu"))
            context.prepare_online_arm = prepare
            try:
                _online(context, result, dict(encode_video=encoder.encode, encode_device=device,
                                              encode_on_scheduler=device == worker.device))
                evidence.save()
            finally:
                if previous is None:
                    del context.prepare_online_arm
                else:
                    context.prepare_online_arm = previous
        return result
    except BaseException as error:
        needs_v100 = (isinstance(error, worker.torch.cuda.OutOfMemoryError)
                      and device == "cuda:0" and candidate_active)
        evidence.fail(error, needs_v100=needs_v100, moves=encoder.moves,
                      fallback_scope="a new independently charged two-device attempt is required" if needs_v100 else None)
        if isinstance(error, (BudgetStop, KeyboardInterrupt, SystemExit)):
            raise
        return evidence.data
    finally:
        encoder.restore_cpu()


class FunctionalDecoder:
    """Compile explicit tensor/cache input-output math, preserving official state ownership."""
    def __init__(self, worker):
        self.worker, self.decoder = worker, worker.pipe.vae.decoder
        self.original = self.decoder.forward
        self.records, self.compiled = [], None

    def pure(self, value, cache, index, first_chunk):
        # Wan's module mutates only this local container; public VAE cache
        # replacement happens after the compiled function has returned.
        local_cache = None if cache is None else list(cache)
        local_index = [index]
        output = self.original(value, feat_cache=local_cache, feat_idx=local_index, first_chunk=first_chunk)
        return output, None if local_cache is None else tuple(local_cache), local_index[0]

    def compile(self):
        if self.compiled is not None:
            return
        from torch._dynamo.backends.registry import lookup_backend
        inductor = lookup_backend("inductor")
        def backend(graph, args):
            record = dict(status="running", index=len(self.records))
            self.records.append(record)
            started = time.perf_counter()
            try:
                output = inductor(graph, args, config_patches={"triton.cudagraphs": False})
                record["status"] = "completed"
                return output
            except BaseException as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                record["ms"] = (time.perf_counter()-started)*1000
        self.compiled = self.worker.torch.compile(self.pure, backend=backend, fullgraph=True, dynamic=False)

    def forward(self, value, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_idx is None:
            feat_idx = [0]
        output, cache, index = self.compiled(value, None if feat_cache is None else tuple(feat_cache),
                                             feat_idx[0], first_chunk)
        if feat_cache is not None:
            feat_cache[:] = cache
        feat_idx[0] = index
        return output

    def select(self, compiled):
        if compiled:
            self.compile()
        self.decoder.forward = self.forward if compiled else self.original

    def close(self):
        self.decoder.forward = self.original
        self.worker.pipe.vae.clear_cache()


def final_latent(context, evidence, request):
    worker, state = context.worker, None
    try:
        state = worker.prepare(request)
        while state.step_index < request["steps"]:
            context.check_budget()
            worker.step(state)
            evidence.data["active_unit"]["completed_steps"] = state.step_index
        if state.step_index != request["steps"]:
            raise RuntimeError("final latent did not complete the requested trajectory")
        return state.latents.detach().clone()
    finally:
        if state is not None:
            worker.release(state)


def decode(context, latent, compiled=False):
    from ..gpu.phase_conflict import decode_final_latent
    worker = context.worker
    context.check_budget()
    worker._sync()
    started = time.perf_counter()
    stance = worker.torch.compiler.set_stance("fail_on_recompile") if compiled else nullcontext()
    with worker.torch.inference_mode(), worker.p._sdpa_math_context(worker.torch), stance:
        array = decode_final_latent(worker.torch, worker.pipe, latent)
    worker._sync()
    return array, dict(ms=(time.perf_counter()-started)*1000, memory=worker._memory())


def run_vae(context):
    evidence = Evidence(context, "vae")
    backend = FunctionalDecoder(context.worker)
    evidence.data.update(dtype=str(context.worker.pipe.vae.dtype), candidate="functional_decoder_inductor",
                         compile_policy=dict(fullgraph=True, dynamic=False, cudagraphs=False),
                         measurement_matrix=dict(inputs=4, orders=[["eager", "compiled"], ["compiled", "eager"]],
                                                 repeats_per_arm_per_order=1, expected_decodes=16,
                                                 full_decode_warmups_per_backend=2,
                                                 frozen_before_first_execution=True),
                         scope="same final latent; original VAE temporal loop, cache clearing and postprocessing",
                         online_atomic_finish_unchanged=True)
    latent = None
    try:
        pairs = []
        for index, item in enumerate(inputs(context)):
            evidence.begin(210., kind="prepare_final_latent", input_id=item["input_id"], completed_steps=0)
            request = spec(context, item, "vae-latent")
            latent = final_latent(context, evidence, request)
            if index == 0:
                for arm in ("eager", "compiled"):
                    evidence.begin(600. if arm == "compiled" else 60., kind="decode_warmup", arm=arm)
                    backend.select(arm == "compiled")
                    # One full 21-latent decode covers first, transition and
                    # steady temporal cache shapes; repeat verifies stable reuse.
                    for repeat in range(2):
                        array, row = decode(context, latent, compiled=arm == "compiled" and repeat > 0)
                        _record_full_warmup(context, request, array, request["steps"], "vae_local_full_decode")
                        evidence.append(dict(kind="warmup", arm=arm, repeat=repeat, service=row))
                        del array
            references, checks, costs = {}, [], {"eager": [], "compiled": []}
            for round_index, order in enumerate((("eager", "compiled"), ("compiled", "eager"))):
                evidence.begin(120., kind="decode_pair", input_id=item["input_id"], round=round_index)
                for arm in order:
                    backend.select(arm == "compiled")
                    for repeat in range(1):
                        array, service = decode(context, latent, compiled=arm == "compiled")
                        _record_full_warmup(context, request, array, request["steps"], "vae_local_full_decode")
                        row = dict(kind="measurement", input_id=item["input_id"], round=round_index,
                                   arm=arm, repeat=repeat, service=service, generation_completed=True,
                                   quality_status="pending")
                        costs[arm].append(service["ms"])
                        # A completed decode remains evidence even if LPIPS
                        # initialization, evaluation or budget checking fails.
                        row["artifact"] = retain(context, f"vae-{item['input_id']}-r{round_index}-{arm}", array)
                        evidence.data["rows"].append(row)
                        evidence.save()
                        if repeat == 0:
                            if round_index == 0 and arm == "eager":
                                references["eager"] = array.copy()
                            try:
                                context.check_budget()
                                check = quality(context, references["eager"], array)
                            except BaseException as error:
                                row.update(quality_status="interrupted" if isinstance(error, BudgetStop) else "failed",
                                           quality_error=f"{type(error).__name__}: {error}")
                                evidence.save()
                                raise
                            checks.append(check["passed"])
                            row.update(quality=check, quality_status="completed")
                        evidence.save()
                        del array
                        if not all(checks):
                            return evidence.finish(status="stopped", reason="vae_quality_failed", compilation=backend.records)
            gain = 1 - statistics.mean(costs["compiled"]) / statistics.mean(costs["eager"])
            pairs.append(dict(input_id=item["input_id"], gain=gain, quality_passed=all(checks),
                              costs={arm: stats(values) for arm, values in costs.items()}))
            evidence.data["pairs"] = pairs
            evidence.data["active_unit"] = None
            evidence.save()
            del latent, references
            latent = None
        rows = [row for row in evidence.data["rows"] if row["kind"] == "measurement"]
        means = {arm: statistics.mean(row["service"]["ms"] for row in rows if row["arm"] == arm)
                 for arm in ("eager", "compiled")}
        gain = 1 - means["compiled"] / means["eager"]
        order_gains = [1 - statistics.mean(row["service"]["ms"] for row in rows if row["arm"] == "compiled" and row["round"] == order)
                      / statistics.mean(row["service"]["ms"] for row in rows if row["arm"] == "eager" and row["round"] == order)
                      for order in (0, 1)]
        local = len(pairs) == 4 and all(pair["quality_passed"] for pair in pairs) and min(order_gains) >= .05
        result = evidence.finish(local_qualified=local, quality_passed=True, gain=gain, order_gains=order_gains,
                                 complete_inputs=4, compilation=backend.records,
                                 summaries={arm: stats([row["service"]["ms"] for row in rows if row["arm"] == arm])
                                            for arm in ("eager", "compiled")},
                                 reason="component_gate_passed" if local else "no_local_vae_gain")
        if local:
            previous = getattr(context, "prepare_online_arm", None)
            def prepare(options):
                if previous is not None:
                    previous(options)
                backend.select(options.get("vae_backend") == "compiled")
            def finish(state, emit):
                from .online import instrumented_finish
                with context.worker.torch.compiler.set_stance("fail_on_recompile"):
                    return instrumented_finish(context.worker, state, emit)
            context.prepare_online_arm = prepare
            try:
                _online(context, result, dict(vae_backend="compiled", finish_video=finish))
                evidence.save()
            finally:
                if previous is None:
                    del context.prepare_online_arm
                else:
                    context.prepare_online_arm = previous
        return result
    except BaseException as error:
        evidence.fail(error, compilation=backend.records)
        if isinstance(error, (BudgetStop, KeyboardInterrupt, SystemExit)):
            raise
        return evidence.data
    finally:
        latent = None
        backend.close()
