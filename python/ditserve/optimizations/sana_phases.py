"""Component-only Sana compile and microbatch measurements.

The caller owns GPU isolation, co-resident Wan models, lifecycle and accounting.
Local gates never grant system/online qualification. Partial evidence is saved
atomically before and after every bounded operation.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import statistics
import time

from ..directions.budget import BudgetStop
from ..directions.direction1 import numerical_error, quantile
from ..util import atomic_json, utc_now


@dataclass
class SanaPhaseContext:
    adapter: object
    output: Path
    prompts: list
    check_budget: object
    seeds: tuple = (101, 102, 201, 202)
    quality_epsilon: float = .01
    minimum_gain: float = .05
    repeats: int = 3
    capacity_check: object = None
    metadata: dict = field(default_factory=dict)
    compile_report: object = None
    quality_evaluator: object = None


def _stats(values):
    values = [float(value) for value in values]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p95=quantile(values, .95))


def _inputs(context):
    if len(context.prompts) != 2 or len(context.seeds) != 4 or context.repeats != 3:
        raise ValueError("Sana pilot freezes two prompts, four seeds and three repeats")
    if context.quality_epsilon != .01 or context.minimum_gain != .05:
        raise ValueError("Sana pilot freezes per-image LPIPS=.01 and mean speed gain=.05")
    return [dict(input_id=f"p{p}-s{seed}", prompt=prompt, seed=int(seed))
            for p, prompt in enumerate(context.prompts) for seed in context.seeds]


class _Evidence:
    def __init__(self, context, phase):
        self.context = context
        self.path = Path(context.output) / f"sana-{phase}.json"
        self.data = dict(schema_version="ditserve.optimizations.sana.v1", phase=phase,
                         status="running", eligible=False, local_qualified=False,
                         system_qualified=False, selected_backend="eager", rows=[],
                         started_at=utc_now(), active_unit=None, metadata=context.metadata,
                         scope="component pilot; no online or global-default qualification",
                         quality_epsilon=context.quality_epsilon, minimum_gain=context.minimum_gain)
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def begin(self, **unit):
        self.data["active_unit"] = unit
        self.save()
        ensure = getattr(self.context.check_budget, "ensure_unit", None)
        if ensure is not None:
            cold_compile = (unit.get("kind") == "warmup" and unit.get("backend") == "compiled"
                            and unit.get("index") == 0)
            seconds = self.context.metadata.get("compile_minimum_unit_seconds", 600.) if cold_compile else 60.
            ensure(float(seconds))
        self.context.check_budget()

    def append(self, row):
        self.data["rows"].append(row)
        self.data["active_unit"] = None
        self.save()

    def finish(self, status="passed", **values):
        self.data.update(status=status, finished_at=utc_now(), **values)
        self.save()
        return self.data

    def fail(self, error):
        return self.finish(status="stopped" if isinstance(error, BudgetStop) else "failed",
                           eligible=False, local_qualified=False, selected_backend="eager",
                           error=f"{type(error).__name__}: {error}",
                           budget_stopped=isinstance(error, BudgetStop))


def _artifact(context, name, image):
    import numpy as np
    path = Path(context.output) / "images" / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, image, allow_pickle=False)
    temporary.replace(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shape=list(image.shape), raw_available=True)


def _quality(context, reference, candidate):
    from ..gpu.quality import evaluate_quality, quality_pass
    evaluator = context.quality_evaluator or evaluate_quality
    # A single image is a one-frame clip, evaluated separately from its peers.
    quality = evaluator(reference[:, None, ...], candidate[:, None, ...], lpips_device="cpu")
    return dict(metrics=quality, numerical=numerical_error(candidate, reference),
                passed=quality_pass(quality, context.quality_epsilon),
                scope="per-image reference-relative LPIPS; not semantic quality")


def _output_hash(image):
    return hashlib.sha256(image.tobytes(order="C")).hexdigest()


def _warm(context, evidence, backend, inputs, batch_size=1):
    adapter = context.adapter
    context.check_budget()
    adapter.set_backend(backend)
    for index in range(2):
        evidence.begin(kind="warmup", backend=backend, batch_size=batch_size, index=index)
        items = ([inputs[index * 4]] if batch_size == 1
                 else [inputs[index], inputs[index + 4]])
        _, service = adapter.generate_batch([item["prompt"] for item in items],
                                           seeds=[item["seed"] for item in items],
                                           check_budget=context.check_budget)
        evidence.append(dict(kind="warmup", backend=backend, batch_size=batch_size,
                             index=index, service=service))


def _capacity(context, evidence):
    if context.capacity_check is None:
        return dict(passed=False, status="unverified", reason="co-resident capacity callback missing")
    evidence.begin(kind="wan_81_frame_capacity")
    result = context.capacity_check()
    if not isinstance(result, dict):
        raise ValueError("capacity_check must return explicit passed/status evidence")
    passed = result.get("passed") is True or result.get("status") == "passed"
    evidence.append(dict(kind="wan_81_frame_capacity", result=result, passed=passed))
    return dict(result, passed=passed)


def run_compile(context):
    """Benchmark eager/compiled batch1 and preserve full/partial component evidence."""
    evidence = _Evidence(context, "compile")
    adapter = context.adapter
    try:
        inputs = _inputs(context)
        evidence.data["inputs"] = inputs
        evidence.data["timing_scope"] = "synchronized full pipeline; includes prompt encoding and image output"
        _warm(context, evidence, "eager", inputs)
        _warm(context, evidence, "compiled", inputs)
        references, qualities, output_hashes = {}, [], {}
        for round_index, order in enumerate((("eager", "compiled"), ("compiled", "eager"))):
            for backend in order:
                adapter.set_backend(backend)
                for item in inputs:
                    for repeat in range(context.repeats):
                        evidence.begin(kind="measurement", round=round_index, backend=backend,
                                       input_id=item["input_id"], repeat=repeat)
                        with adapter.steady_context():
                            image, service = adapter.generate(item["prompt"], seed=item["seed"],
                                                              check_budget=context.check_budget)
                        if service["compile_delta"]:
                            raise RuntimeError("compilation during a measured Sana request")
                        row = dict(kind="measurement", round=round_index, backend=backend,
                                   input_id=item["input_id"], repeat=repeat, service=service,
                                   output_sha256=_output_hash(image))
                        identity = backend, item["input_id"]
                        previous = output_hashes.setdefault(identity, row["output_sha256"])
                        row["repeat_exact"] = previous == row["output_sha256"]
                        if repeat == 0:
                            if backend == "eager" and round_index == 0:
                                references[item["input_id"]] = image.copy()
                            row["artifact"] = _artifact(context, f"compile-r{round_index}-{backend}-{item['input_id']}", image)
                            row["quality"] = _quality(context, references[item["input_id"]], image)
                            qualities.append(row["quality"]["passed"])
                        evidence.append(row)
                        if not row["repeat_exact"] or ("quality" in row and not row["quality"]["passed"]):
                            adapter.set_backend("eager")
                            return evidence.finish(status="stopped", quality_passed=False,
                                quality_checked=len(qualities), selected_backend="eager",
                                compilation=adapter.compilation_summary(),
                                reason="stop_image_quality_or_repeat_consistency")
        summaries = {}
        measured = [row for row in evidence.data["rows"] if row["kind"] == "measurement"]
        for backend in ("eager", "compiled"):
            summaries[backend] = _stats([row["service"]["ms"] for row in measured if row["backend"] == backend])
        gains = []
        for round_index in range(2):
            means = {backend: statistics.mean(row["service"]["ms"] for row in measured
                     if row["round"] == round_index and row["backend"] == backend)
                     for backend in ("eager", "compiled")}
            gains.append(1 - means["compiled"] / means["eager"])
        quality_passed = (len(qualities) == 32 and all(qualities)
                          and all(row["repeat_exact"] for row in measured))
        performance_passed = all(gain >= context.minimum_gain for gain in gains)
        adapter.set_backend("compiled" if quality_passed and performance_passed else "eager")
        capacity = (_capacity(context, evidence) if quality_passed and performance_passed
                    else dict(passed=False, status="not_run", reason="component quality/speed gate failed"))
        local = quality_passed and performance_passed and capacity["passed"]
        selected = "compiled" if local else "eager"
        adapter.set_backend(selected)
        return evidence.finish(summaries=summaries, round_speed_gains=gains,
                               distinct_inputs=8, measured_requests_per_backend=48,
                               quality_passed=quality_passed, quality_checked=len(qualities),
                               performance_passed=performance_passed, capacity=capacity,
                               compilation=adapter.compilation_summary(), selected_backend=selected,
                               local_qualified=local, eligible=local,
                               reason="component_gate_passed" if local else "retain_eager")
    except BaseException as error:
        evidence.data["compilation"] = adapter.compilation_summary()
        try:
            adapter.set_backend("eager")
        except Exception:
            pass
        evidence.fail(error)
        if isinstance(error, (BudgetStop, KeyboardInterrupt, SystemExit)):
            raise
        return evidence.data


def _compile_selection(context):
    report = context.compile_report or context.metadata.get("compile_report")
    if isinstance(report, (str, Path)):
        report = json.loads(Path(report).read_text())
    if isinstance(report, dict):
        return ("compiled" if report.get("status") == "passed"
                and report.get("local_qualified") is True
                and report.get("selected_backend") == "compiled" else "eager")
    return "eager"


def _burst(context, items, batch_size, deadline_ms):
    """Simultaneous-arrival local burst; no Wan compute and no waiting window."""
    adapter = context.adapter
    completed = [dict(input_id=item["input_id"], arrival_ms=0., start_ms=None,
                      finish_ms=None, deadline_ms=deadline_ms, on_time=False, status="pending")
                 for item in items]
    services, images = [], []
    adapter._sync()
    started = time.perf_counter()
    try:
        for index in range(0, len(items), batch_size):
            context.check_budget()
            group = items[index:index + batch_size]
            dispatch_ms = (time.perf_counter() - started) * 1000
            for row in completed[index:index + len(group)]:
                row["start_ms"] = dispatch_ms
            with adapter.steady_context():
                output, service = adapter.generate_batch([item["prompt"] for item in group],
                                                         seeds=[item["seed"] for item in group],
                                                         check_budget=context.check_budget)
            if service["compile_delta"]:
                raise RuntimeError("compilation during a measured Sana burst")
            finish_ms = (time.perf_counter() - started) * 1000
            services.append(service)
            for offset, item in enumerate(group):
                completed[index + offset].update(finish_ms=finish_ms, on_time=finish_ms <= deadline_ms,
                                                 status="completed")
                images.append(output[offset:offset + 1].copy())
    except BaseException as error:
        for row in completed:
            if row["status"] == "pending":
                row["status"] = "interrupted" if isinstance(error, BudgetStop) else "failed"
        error.sana_partial_burst = dict(requests=completed, services=services, complete=False,
                                       elapsed_ms=(time.perf_counter() - started) * 1000)
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000
    return dict(requests=completed, services=services, elapsed_ms=elapsed_ms,
                timely=sum(row["on_time"] for row in completed),
                latency_ms=_stats([row["finish_ms"] for row in completed]),
                wait_ms=_stats([row["start_ms"] for row in completed]),
                goodput_rps=sum(row["on_time"] for row in completed) * 1000 / elapsed_ms), images


def run_batch(context):
    """Compare serial with zero-wait batch2, including 4/5-image deadline gates."""
    evidence = _Evidence(context, "batch")
    adapter = context.adapter
    try:
        inputs = _inputs(context)
        backend = _compile_selection(context)
        evidence.data.update(inputs=inputs, selected_backend=backend,
                             timing_scope="full pipeline; both images finish at full batch return",
                             rng_scope="independent CPU generators through initial and SCM step noise")
        references, qualities, output_hashes = {}, [], {}
        # Keep the quality anchor eager even when compiled serial is the timing
        # baseline. Otherwise two individually small deviations could compound.
        _warm(context, evidence, "eager", inputs)
        for item in inputs:
            evidence.begin(kind="eager_reference", input_id=item["input_id"])
            image, service = adapter.generate(item["prompt"], seed=item["seed"],
                                              check_budget=context.check_budget)
            references[item["input_id"]] = image.copy()
            evidence.append(dict(kind="eager_reference", input_id=item["input_id"], service=service,
                                 artifact=_artifact(context, f"batch-eager-{item['input_id']}", image)))
        for size in (1, 2):
            _warm(context, evidence, backend, inputs, size)
        # Alternate prompt identities in pairs rather than duplicating one prompt.
        paired = [inputs[i] for pair in zip(range(4), range(4, 8)) for i in pair]
        for round_index, order in enumerate(((1, 2), (2, 1))):
            for size in order:
                for offset in range(0, len(paired), size):
                    group = paired[offset:offset + size]
                    for repeat in range(context.repeats):
                        evidence.begin(kind="measurement", round=round_index, batch_size=size,
                                       input_ids=[item["input_id"] for item in group], repeat=repeat)
                        with adapter.steady_context():
                            images, service = adapter.generate_batch([item["prompt"] for item in group],
                                          seeds=[item["seed"] for item in group], check_budget=context.check_budget)
                        if service["compile_delta"]:
                            raise RuntimeError("compilation during a measured Sana batch")
                        row = dict(kind="measurement", round=round_index, batch_size=size, repeat=repeat,
                                   input_ids=[item["input_id"] for item in group], service=service)
                        row["output_sha256"] = [_output_hash(images[index:index + 1]) for index in range(len(group))]
                        row["repeat_exact"] = all(output_hashes.setdefault((size, item["input_id"]), digest) == digest
                                                  for item, digest in zip(group, row["output_sha256"]))
                        if repeat == 0:
                            checks = []
                            for index, item in enumerate(group):
                                image = images[index:index + 1]
                                check = _quality(context, references[item["input_id"]], image)
                                check["input_id"] = item["input_id"]
                                check["artifact"] = _artifact(context, f"batch-r{round_index}-b{size}-{item['input_id']}", image)
                                checks.append(check)
                                qualities.append(check["passed"])
                            row["quality"] = checks
                        evidence.append(row)
                        if not row["repeat_exact"] or any(not check["passed"] for check in row.get("quality", [])):
                            return evidence.finish(status="stopped", quality_passed=False,
                                quality_checked=len(qualities), selected_backend=backend, selected_batch_size=1,
                                compilation=adapter.compilation_summary(),
                                reason="stop_image_quality_or_repeat_consistency")
        # Deadline is inherited from the old frozen trace, never recalibrated
        # using a faster candidate's service time.
        deadline = context.metadata.get("image_deadline_ms")
        if not isinstance(deadline, (int, float)) or deadline <= 0:
            raise ValueError("batch gate requires frozen metadata.image_deadline_ms")
        burst_rows = []
        for count in (4, 5):
            for round_index, order in enumerate(((1, 2), (2, 1))):
                for size in order:
                    evidence.begin(kind="burst", count=count, round=round_index, batch_size=size)
                    row, images = _burst(context, paired[:count], size, deadline)
                    row.update(kind="burst", count=count, round=round_index, batch_size=size)
                    row["quality"] = [_quality(context, references[item["input_id"]], image)
                                      for item, image in zip(paired[:count], images)]
                    qualities.extend(check["passed"] for check in row["quality"])
                    burst_rows.append(row)
                    evidence.append(row)
                    if any(not check["passed"] for check in row["quality"]):
                        return evidence.finish(status="stopped", quality_passed=False,
                            quality_checked=len(qualities), selected_backend=backend, selected_batch_size=1,
                            compilation=adapter.compilation_summary(), reason="stop_burst_image_quality")
        comparisons = []
        for count in (4, 5):
            for round_index in range(2):
                rows = {row["batch_size"]: row for row in burst_rows
                        if row["count"] == count and row["round"] == round_index}
                comparisons.append(dict(count=count, round=round_index,
                    serial_timely=rows[1]["timely"], batch_timely=rows[2]["timely"],
                    gain=1 - rows[2]["elapsed_ms"] / rows[1]["elapsed_ms"]))
        quality_passed = (len(qualities) == 68 and all(qualities)
                          and all(row["repeat_exact"] for row in evidence.data["rows"]
                                  if row["kind"] == "measurement"))
        # A deadline-capacity improvement must appear in both execution orders;
        # neither burst size may lose timely images. Throughput alone is not a go.
        performance = (all(row["gain"] >= context.minimum_gain
                           and row["batch_timely"] >= row["serial_timely"] for row in comparisons)
                       and all(any(row["batch_timely"] > row["serial_timely"]
                                   for row in comparisons if row["round"] == index) for index in range(2)))
        capacity = (_capacity(context, evidence) if quality_passed and performance
                    else dict(passed=False, status="not_run", reason="component quality/deadline gate failed"))
        local = quality_passed and performance and capacity["passed"]
        measured = [row for row in evidence.data["rows"] if row["kind"] == "measurement"]
        summaries = {str(size): dict(invocation_ms=_stats([row["service"]["ms"] for row in measured
                                                         if row["batch_size"] == size]),
                                    cost_per_image_ms=_stats([row["service"]["cost_per_image_ms"] for row in measured
                                                             if row["batch_size"] == size]))
                     for size in (1, 2)}
        return evidence.finish(selected_backend=backend, selected_batch_size=2 if local else 1,
                               local_qualified=local, eligible=local, summaries=summaries,
                               distinct_inputs=8, quality_checked=len(qualities), quality_passed=quality_passed,
                               performance_passed=performance, burst_comparisons=comparisons,
                               capacity=capacity, compilation=adapter.compilation_summary(),
                               image_deadline_ms=deadline,
                               reason="component_gate_passed" if local else "retain_serial")
    except BaseException as error:
        evidence.data["compilation"] = adapter.compilation_summary()
        if hasattr(error, "sana_partial_burst"):
            evidence.data["active_partial"] = error.sana_partial_burst
        evidence.fail(error)
        if isinstance(error, (BudgetStop, KeyboardInterrupt, SystemExit)):
            raise
        return evidence.data
