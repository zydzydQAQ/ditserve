"""Bounded, non-performance Sana batch/compile numerical diagnosis.

The installed transformer callable is observed from outside its compiled graph.
No model forward, precision, sampler equation, threshold, or RNG is replaced.
Device snapshots remain on device until a complete generation has returned;
only bounded CPU references and numerical summaries survive each comparison.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import traceback

from ..directions.budget import BudgetStop
from ..directions.sana import RECIPE
from ..research.common import atomic_json, file_digest


SEEDS = (101, 102)
MAX_STEPS = 2
MAX_CAPTURE_BYTES = 256 * 1024 * 1024


def _to_cpu(tensor):
    return tensor.detach().to(device="cpu", copy=True)


def _tensor_summary(tensor):
    torch = __import__("torch")
    # Singleton expanded views may count as contiguous while retaining stride
    # zero; clone()/contiguous() need not normalize it for a dtype byte view.
    # Allocate dense CPU storage explicitly, then copy in the original dtype.
    # This preserves BF16 bits and runs only after generation has completed.
    value = torch.empty(tensor.numel(), dtype=tensor.dtype, device="cpu")
    value.copy_(tensor.detach().reshape(-1))
    finite = bool(torch.isfinite(value).all())
    converted = value.to(dtype=torch.float64)
    return dict(shape=list(tensor.shape), dtype=str(value.dtype), finite=finite,
        bytes=value.numel() * value.element_size(),
        sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
        min=float(converted.min()) if finite and value.numel() else None,
        max=float(converted.max()) if finite and value.numel() else None,
        mean=float(converted.mean()) if finite and value.numel() else None)


def _error(reference, candidate, rtol, atol):
    torch = __import__("torch")
    result = dict(shape_equal=tuple(reference.shape) == tuple(candidate.shape),
        dtype_equal=reference.dtype == candidate.dtype, exact=False, passed=False,
        finite=False, rtol=rtol, atol=atol, max_abs=None, mean_abs=None, rmse=None, max_relative=None)
    if not result["shape_equal"]:
        return result
    a, b = reference.to(dtype=torch.float64), candidate.to(dtype=torch.float64)
    result["finite"] = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not result["finite"]:
        return result
    difference = (b - a).abs()
    result.update(exact=bool(torch.equal(reference, candidate)),
        passed=bool(torch.all(difference <= atol + rtol * a.abs())),
        max_abs=float(difference.max()) if difference.numel() else 0.,
        mean_abs=float(difference.mean()) if difference.numel() else 0.,
        rmse=float(difference.square().mean().sqrt()) if difference.numel() else 0.,
        max_relative=float((difference / a.abs().clamp_min(atol)).max()) if difference.numel() else 0.)
    return result


class _TransformerBoundary:
    """Outer Python call boundary; the target's forward and FX graph stay intact."""
    def __init__(self, target, capture):
        self.target, self.capture = target, capture

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __call__(self, hidden_states, **kwargs):
        index = self.capture.transformer_calls
        if index >= MAX_STEPS:
            raise RuntimeError("diagnostic transformer step bound exceeded")
        self.capture.transformer_calls += 1
        prefix = "step%d.transformer." % index
        self.capture.add(prefix + "input.hidden_states", hidden_states)
        for name in ("encoder_hidden_states", "encoder_attention_mask", "guidance", "timestep"):
            self.capture.add(prefix + "input." + name, kwargs[name])
        output = self.target(hidden_states, **kwargs)
        self.capture.add(prefix + "prediction", output[0])
        return output


class _Capture:
    def __init__(self, adapter, count):
        self.adapter, self.count = adapter, count
        self.values, self.generator_rows = [], []
        self.bytes = self.transformer_calls = self.scheduler_calls = 0
        self.generator_ids = None

    def add(self, name, tensor):
        torch = self.adapter.torch
        if not torch.is_tensor(tensor):
            raise TypeError("diagnostic boundary requires a tensor: " + name)
        size = tensor.numel() * tensor.element_size()
        if self.bytes + size > MAX_CAPTURE_BYTES:
            raise RuntimeError("bounded device diagnostic snapshot capacity exceeded")
        self.values.append((name, tensor.detach().clone(), str(tensor.device)))
        self.bytes += size

    def generators(self, stage, generators):
        values = generators if isinstance(generators, list) else [generators]
        if len(values) != self.count or len({id(g) for g in values}) != self.count:
            raise ValueError("one independent CPU generator per request is required")
        identities = tuple(id(g) for g in values)
        if self.generator_ids is None:
            self.generator_ids = identities
        if self.generator_ids != identities or any(str(g.device) != "cpu" for g in values):
            raise ValueError("initial-latent and SCM generators must be the same CPU streams")
        self.generator_rows.append(dict(stage=stage, object_ids=list(identities),
                                       devices=[str(g.device) for g in values], seeds=[g.initial_seed() for g in values]))
        # These states already live on CPU. Reading them does not consume RNG.
        self.add(stage + ".generator_state", self.adapter.torch.stack([g.get_state() for g in values]))

    @contextmanager
    def installed(self):
        pipe, scheduler = self.adapter.pipe, self.adapter.pipe.scheduler
        originals = []

        def install(owner, name, value):
            originals.append((owner, name, name in vars(owner), vars(owner).get(name)))
            # DiffusionPipeline.__setattr__ otherwise rewrites its component
            # config for the temporary observation proxy. Preserve that config.
            vars(owner)[name] = value

        prepare, step = pipe.prepare_latents, scheduler.step

        def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
            self.generators("initial.before", generator)
            result = prepare(batch_size, num_channels_latents, height, width, dtype, device, generator, latents)
            self.add("initial.latents", result)
            self.generators("initial.after", generator)
            return result

        # Keep the real explicit generator parameter: pipeline inspects this
        # signature to decide whether to forward the generators into SCM.
        def scheduler_step(model_output, timestep, sample, generator=None, return_dict=True):
            index = self.scheduler_calls
            if index >= MAX_STEPS:
                raise RuntimeError("diagnostic scheduler step bound exceeded")
            self.scheduler_calls += 1
            prefix = "step%d.scheduler." % index
            self.generators(prefix + "before", generator)
            self.add(prefix + "input.prediction", model_output)
            self.add(prefix + "input.timestep", timestep)
            self.add(prefix + "input.sample", sample)
            result = step(model_output, timestep, sample, generator=generator, return_dict=return_dict)
            self.add(prefix + "output.prev_sample", result[0] if not return_dict else result.prev_sample)
            self.add(prefix + "output.pred_x0", result[1] if not return_dict else result.pred_original_sample)
            self.generators(prefix + "after", generator)
            return result

        try:
            install(pipe, "transformer", _TransformerBoundary(pipe.transformer, self))
            install(pipe, "prepare_latents", prepare_latents)
            install(scheduler, "step", scheduler_step)
            yield self
        finally:
            for owner, name, existed, value in reversed(originals):
                if existed:
                    vars(owner)[name] = value
                else:
                    vars(owner).pop(name, None)

    def cpu_members(self):
        if self.transformer_calls != MAX_STEPS or self.scheduler_calls != MAX_STEPS:
            raise RuntimeError("incomplete two-step diagnostic capture")
        members = [dict() for _ in range(self.count)]
        for name, tensor, device in self.values:
            if tensor.ndim < 1 or tensor.shape[0] != self.count:
                raise RuntimeError("captured batch dimension differs: " + name)
            host = _to_cpu(tensor)
            for index in range(self.count):
                members[index][name] = dict(tensor=host[index:index + 1].clone(), captured_device=device)
        return members

    def clear(self):
        self.values.clear()


def _capture_once(context, inputs, backend):
    adapter = context.sana
    capture = _Capture(adapter, len(inputs))
    try:
        context.check_budget()
        adapter.set_backend(backend)
        with capture.installed():
            images, service = adapter.generate_batch([i["prompt"] for i in inputs],
                seeds=[i["seed"] for i in inputs], check_budget=context.check_budget)
        # generate_batch completed and synchronized. No snapshot transfer or
        # reduction is inserted in the model/scheduler execution path above.
        members = capture.cpu_members()
        return images, members, dict(generation_completed=True, requested_backend=backend,
            backend=service.get("backend"), compile_delta=service.get("compile_delta"),
            graph_compile_count=adapter.compilation_summary()["graph_compile_count"],
            transformer_calls=capture.transformer_calls, scheduler_calls=capture.scheduler_calls,
            device_snapshot_bytes=capture.bytes, generator_evidence=capture.generator_rows,
            same_cpu_generators_through_scm=True,
            timing_scope="instrumented numerical diagnosis; no timings or speed claims")
    finally:
        capture.clear()


def _compare_capture(reference, candidate, rtol, atol):
    components = []
    if list(reference) != list(candidate):
        return dict(complete=False, reason="captured boundary sequence differs", components=[],
                    first_exact_drift=None, first_strict_drift=None)
    for name, item in reference.items():
        observed = candidate[name]
        components.append(dict(boundary=name, numerical=_error(item["tensor"], observed["tensor"], rtol, atol),
            reference=_tensor_summary(item["tensor"]), candidate=_tensor_summary(observed["tensor"])))
    exact = next((row["boundary"] for row in components if not row["numerical"]["exact"]), None)
    strict = next((row["boundary"] for row in components if not row["numerical"]["passed"]), None)
    return dict(complete=True, components=components, first_exact_drift=exact, first_strict_drift=strict,
        all_exact=exact is None, all_strict=strict is None,
        interpretation="First observed boundary disagreement, not proof of one responsible kernel or causal mechanism")


def _quality(context, reference, candidate, epsilon, rtol, atol):
    from ..gpu.quality import evaluate_quality, quality_pass
    evaluator = getattr(context, "quality_evaluator", None) or evaluate_quality
    metrics = evaluator(reference[:, None, ...], candidate[:, None, ...], lpips_device="cpu")
    torch = context.sana.torch
    return dict(metrics=metrics, passed=quality_pass(metrics, epsilon), epsilon=epsilon,
        numerical=_error(torch.from_numpy(reference), torch.from_numpy(candidate), rtol, atol),
        reference_sha256=hashlib.sha256(reference.tobytes(order="C")).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.tobytes(order="C")).hexdigest(),
        scope="Per-image LPIPS against matching eager serial prompt/seed; no semantic quality claim")


def _source(adapter):
    result = {}
    for name, method in (("pipeline", type(adapter.pipe).__call__), ("scheduler", type(adapter.pipe.scheduler).step)):
        path = inspect.getsourcefile(method)
        if path and Path(path).is_file():
            result[name] = dict(path=str(Path(path).resolve()), sha256=file_digest(path))
    return result


def run_numerics(context):
    """Observe fixed 4 eager references, 4 two-image batch calls, 4 compiled calls.

    Requires the existing loaded ``context.sana`` OptimizedSanaAdapter, config,
    output and budget callback. Wan remains owned/resident by the caller. An
    optional quality_evaluator is used only by CPU fixtures. Nothing here loads
    another pipeline, changes defaults, searches settings, or measures speed.
    """
    adapter, config = context.sana, context.config
    if getattr(adapter, "_busy", False):
        raise RuntimeError("numerical diagnosis requires an idle Sana adapter")
    prompts = list(config["direction1"]["prompts"][:2])
    if len(prompts) != 2 or len(set(prompts)) != 2:
        raise ValueError("numerical diagnosis freezes two distinct existing prompts")
    quality = config["quality"]
    rtol, atol, epsilon = quality["rtol"], quality["atol"], quality["image_lpips"]
    if (rtol, atol, epsilon) != (1e-5, 1e-6, .01) or RECIPE["num_inference_steps"] != MAX_STEPS:
        raise ValueError("frozen numerical/LPIPS recipe differs")
    inputs = [dict(input_id="prompt%d-seed%d" % (index, seed), prompt=prompt, seed=seed)
              for index, prompt in enumerate(prompts) for seed in SEEDS]
    units = [("eager_serial", "eager", [item]) for item in inputs]
    for seed in SEEDS:
        pair = [next(i for i in inputs if i["prompt"] == p and i["seed"] == seed) for p in prompts]
        units.extend((("eager_batch2_forward", "eager", pair), ("eager_batch2_reverse", "eager", list(reversed(pair)))))
    units.extend(("compiled_single", "compiled", [item]) for item in inputs)
    path = Path(context.output) / "sana-numerics.json"
    record = dict(schema_version="ditserve.optimizations.sana_numerics.v1", status="running",
        inputs=inputs, recipe=dict(RECIPE), quality=dict(rtol=rtol, atol=atol, image_lpips=epsilon),
        rows=[], active_unit=None, required_units=12, required_generated_images=16,
        diagnostic_complete=False, local_qualified=False, online_qualified=False,
        changes_serving_default=False, performance_claims=False, source=_source(adapter),
        snapshot_policy=dict(max_device_bytes_per_call=MAX_CAPTURE_BYTES, max_steps=MAX_STEPS,
            persistent_arrays=False, transfer_scope="only after complete synchronized request return"),
        scope="Fixed settings diagnosis only; failed compilation/quality retained; no precision or threshold search")
    references, entry_backend = {}, adapter.backend

    def save():
        atomic_json(path, record)

    save()
    try:
        for unit_index, (arm, backend, members) in enumerate(units):
            context.check_budget()
            row = dict(unit_index=unit_index, arm=arm, backend=backend,
                input_ids=[i["input_id"] for i in members], status="running", comparisons=[])
            record["active_unit"] = row
            save()
            images = snapshots = None
            try:
                images, snapshots, service = _capture_once(context, members, backend)
                row["execution"] = service
                row["snapshot_summaries"] = [dict(input_id=item["input_id"],
                    boundaries={name: dict(_tensor_summary(value["tensor"]), captured_device=value["captured_device"])
                                for name, value in snap.items()}) for item, snap in zip(members, snapshots)]
                row["generation_completed"] = True
                save()
                for index, item in enumerate(members):
                    context.check_budget()
                    key, output = item["input_id"], images[index:index + 1]
                    if arm == "eager_serial":
                        references[key] = dict(output=output.copy(), capture=snapshots[index])
                    elif key not in references:
                        row["comparisons"].append(dict(input_id=key, status="unavailable", reason="eager serial reference failed"))
                    else:
                        ref = references[key]
                        comparison = dict(input_id=key, status="completed",
                            trajectory=_compare_capture(ref["capture"], snapshots[index], rtol, atol),
                            output=_quality(context, ref["output"], output, epsilon, rtol, atol))
                        row["comparisons"].append(comparison)
                        save()
                row["status"] = "completed"
            except (BudgetStop, KeyboardInterrupt, SystemExit) as error:
                row.update(status="incomplete", error=type(error).__name__ + ": " + str(error))
                raise
            except Exception as error:
                row.update(status="failed", error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
                # Failure is diagnostic evidence, not a reason to silently omit
                # another fixed arm. No alternate precision/settings are tried.
            finally:
                images = snapshots = None
                record["rows"].append(row)
                record["active_unit"] = None
                save()
        comparisons = [c for row in record["rows"] for c in row["comparisons"]]
        record.update(status="passed", diagnostic_complete=True,
            completed_units=sum(r["status"] == "completed" for r in record["rows"]),
            failed_units=sum(r["status"] == "failed" for r in record["rows"]),
            compared_images=sum(c["status"] == "completed" for c in comparisons),
            lpips_passed_images=sum(c.get("output", {}).get("passed") is True for c in comparisons),
            reason="All fixed diagnostic units attempted; passing diagnosis does not qualify a serving backend")
        return record
    except BaseException as error:
        record.update(status="incomplete" if isinstance(error, (BudgetStop, KeyboardInterrupt, SystemExit)) else "failed",
                      reason=type(error).__name__ + ": " + str(error))
        raise
    finally:
        references.clear()
        try:
            adapter.set_backend(entry_backend)
            record["backend_restored"] = adapter.backend == entry_backend
        finally:
            save()
