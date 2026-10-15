"""Four-arm eager cache/preemption factorial, separate from compiled Wan."""
import time

from ..directions.block_executor import WanBlockExecutor
from ..directions.direction1 import _release, _spec
from ..research.common import atomic_json
from .online import load_frozen_traces, run_trace, paired_gate, TraceJournal
from .cache_executor import PairedCacheExecutor


ARMS = ("off_step", "off_k4", "paired_step", "paired_k4")


class StepPairedExecutor(PairedCacheExecutor):
    def native_step(self, state, *, check_budget=lambda: None):
        # Full step is the same newly validated paired arithmetic, without a
        # scheduler yield inside it; original atomic paired parity gates first.
        return self.advance(state, 60, check_budget=check_budget)


def cache_spec(spec):
    return dict(spec, cache=True, threshold=.08, max_continuous=1)


def _warm_memory(worker):
    return dict(worker._memory(),
        peak_allocated_bytes=int(worker.torch.cuda.max_memory_allocated(worker.device)),
        peak_reserved_bytes=int(worker.torch.cuda.max_memory_reserved(worker.device)))


def _warm_cache_paths(events):
    sequence = []
    for event in events:
        if event.get("kind") != "step":
            continue
        path, reused = event.get("path"), event.get("reused_forwards")
        outcome = ("full" if path == "compute" and reused == 0 else
                   "reuse" if path == "reuse" and reused == 2 else "invalid")
        sequence.append(dict(step_index=event.get("step_index"),
            completed_steps=event.get("completed_steps"), path=path,
            reused_forwards=reused, outcome=outcome))
    valid = (len(sequence) == 32 and all(row["step_index"] == index and
             row["completed_steps"] == index + 1 and row["outcome"] != "invalid"
             for index, row in enumerate(sequence)))
    counts = {name: sum(row["outcome"] == name for row in sequence)
              for name in ("full", "reuse", "invalid")}
    return dict(path_sequence=sequence, counts=counts, valid=valid,
                full_reuse_observed=valid and counts["full"] > 0 and counts["reuse"] > 0)


def warm_cache_factorial(context):
    """Warm a new process after its independently charged full off capacity run.

    These private inputs do not consume or pre-encode any online request. The
    off continuation is exercised for two steps; the paired continuation runs
    the full fixed trajectory and decoder, including actual full/reuse paths.
    """
    if not context.metadata.get("wan_warmed"):
        raise ValueError("fresh full cache-off capacity/decode must precede factorial warmup")
    path = context.output / "cache-factorial-warmup.json"
    if path.exists():
        raise FileExistsError("factorial warmup evidence already exists")
    worker = context.worker
    if worker._states:
        raise ValueError("factorial warmup requires no live request state")
    context.metadata["cache_factorial_warmed"] = False
    started = time.perf_counter()
    record = dict(schema_version="ditserve.optimizations.cache_warmup.v1", status="running", rows=[],
                  active_unit=None, fresh_capacity_precondition=True,
                  scope="charged new-process warmup before online arrival clocks; no request output or quality qualification")
    atomic_json(path, record)
    try:
        executors = {"off_k4": WanBlockExecutor(worker), "paired_k4": StepPairedExecutor(worker)}
        context.executors = getattr(context, "executors", {})
        context.executors.update(cache_factorial_off=executors["off_k4"],
                                 cache_factorial_paired=executors["paired_k4"])
        context.cache_executor = executors["paired_k4"]
        for arm, target_steps in (("off_k4", 2), ("paired_k4", 32)):
            context.check_budget()
            worker.torch.cuda.reset_peak_memory_stats(worker.device)
            unit_started = time.perf_counter()
            state = continuation = None
            row = dict(arm=arm, request_id="factorial-warm-" + arm, target_steps=target_steps,
                       seed=101, completed_steps=0, decoded=False, status="running",
                       memory_before=_warm_memory(worker))
            record["active_unit"] = row
            atomic_json(path, record)
            try:
                spec = _spec(context.opts, context.profile["selected_shape"], row["request_id"],
                             prompt=context.opts["prompts"][0], seed=101)
                if spec["steps"] != 32:
                    raise ValueError("factorial warmup requires the frozen 32-step video")
                if arm == "paired_k4":
                    spec = cache_spec(spec)
                state = worker.prepare_encoded(spec, worker.encode_prompt(spec))
                continuation = executors[arm].attach(state)
                while state.step_index < target_steps:
                    executors[arm].advance(continuation, 4, check_budget=context.check_budget)
                    row["completed_steps"] = state.step_index
                if arm == "paired_k4":
                    row["cache_paths"] = _warm_cache_paths(state.events)
                    if not row["cache_paths"]["full_reuse_observed"]:
                        raise RuntimeError("fixed paired warmup did not cover both full and reuse")
                    array = worker.finish(state)
                    row.update(decoded=True, output_shape=list(array.shape))
                    del array
                    if row["output_shape"] != [1, 81, 480, 832, 3]:
                        raise ValueError("factorial warmup decode differs from fixed output layout")
                row.update(status="completed", memory_before_release=_warm_memory(worker))
            except BaseException as error:
                row.update(status="failed", error=type(error).__name__ + ": " + str(error))
                raise
            finally:
                if state is not None:
                    row["completed_steps"] = state.step_index
                    if arm == "paired_k4":
                        row["cache_paths"] = _warm_cache_paths(state.events)
                    _release(worker, executors[arm], state, continuation)
                row.update(elapsed_ms=(time.perf_counter() - unit_started) * 1000,
                           memory_after_release=_warm_memory(worker),
                           worker_live_states_after=len(worker._states),
                           executor_live_states_after=len(executors[arm]._states))
                record["rows"].append(row)
                record["active_unit"] = None
                atomic_json(path, record)
            if worker._states or executors[arm]._states:
                raise RuntimeError("factorial warmup did not release its request state")
        record.update(status="completed", elapsed_ms=(time.perf_counter() - started) * 1000)
        atomic_json(path, record)
        context.metadata.update(cache_factorial_warmed=True, cache_factorial_warmup_path=str(path))
        return record
    except BaseException as error:
        record.update(status="failed", elapsed_ms=(time.perf_counter() - started) * 1000,
                      error=type(error).__name__ + ": " + str(error))
        atomic_json(path, record)
        raise


def run_cache_online(context, result):
    if not result.get("cache_quality_qualified") or not result.get("full_qualification_completed"):
        result.update(online_qualified=False, online_unverified_reason="complete new cache quality qualification absent")
        return result
    traces = load_frozen_traces(context.config["old_online"])
    expected = sum(max(r["arrival_ms"] + (r["service_ms"] if r["kind"] == "video" else 1000.)
                       for r in t["requests"]) / 1000. for t in traces)
    if context.check_budget.remaining() < expected * 8 * 1.08 + 60:
        result.update(online_qualified=False, online_unverified_reason="remaining allowance cannot admit complete four-arm AB/BA factorial")
        return result
    metadata = getattr(context, "metadata", {})
    if (metadata.get("cache_local_evidence_reused") is True and
            not metadata.get("cache_factorial_warmed")):
        raise ValueError("fresh process factorial warmup must complete before online measurement")
    from .worker import selected
    defaults = selected(context.run_root)
    defaults.update(getattr(context, "online_defaults", {}))
    context.sana.set_backend(defaults["sana_backend"])
    if metadata.get("cache_factorial_warmed"):
        off = context.executors["cache_factorial_off"]
        paired = context.executors["cache_factorial_paired"]
    else:
        # The original full-local-run entry is already warm from its measured
        # trajectories. Only an explicitly reused local result needs the new gate.
        off, paired = WanBlockExecutor(context.worker), StepPairedExecutor(context.worker)
        context.executors = getattr(context, "executors", {})
        context.executors.update(cache_factorial_off=off, cache_factorial_paired=paired)
    context.cache_executor = paired
    record = dict(schema_version="ditserve.optimizations.cache_factorial.v1", status="running", rows=[],
                  warmups=[], active_unit=None,
                  process_warmup_path=metadata.get("cache_factorial_warmup_path"),
                  warmup_scope="fresh process helper" if metadata.get("cache_factorial_warmed") else
                               "original entry immediately follows complete local cache trajectories",
                  wan_backend="eager", vae_backend="eager", arms=list(ARMS), image_batch_size=defaults["image_batch_size"],
                  encoder_device=defaults.get("encode_device", "cpu"),
                  traces=[{k: v for k, v in trace.items() if k != "requests"} for trace in traces],
                  scope="eager Wan and VAE fixed in all four arms to isolate cache and preemption")
    path = context.output / "cache-online.json"
    journal = TraceJournal(path, atomic_json)
    result["online_path"] = str(path)

    def save():
        journal.save_record(record)

    save()
    try:
        for trace in traces:
            for order_index, order in enumerate((ARMS, tuple(reversed(ARMS)))):
                for arm in order:
                    is_cached = arm.startswith("paired")
                    context.executor = paired if is_cached else off
                    options = dict(defaults, policy="step" if arm.endswith("step") else "block-4",
                                   wan_backend="eager", vae_backend="eager")
                    if hasattr(context, "prepare_online_arm"):
                        context.prepare_online_arm(options)
                    warm = dict(trace_sha256=trace["trace_sha256"], order_index=order_index,
                                factorial_arm=arm, status="running", services=[],
                                scope="two charged Sana warmups before this arm arrival clock")
                    record["active_unit"] = warm
                    save()
                    warm_started = time.perf_counter()
                    try:
                        for index in range(2):
                            count = defaults["image_batch_size"]
                            array, service = context.sana.generate_batch(context.opts["prompts"][:count],
                                seeds=[101 + index + n for n in range(count)], check_budget=context.check_budget)
                            del array
                            warm["services"].append(service)
                        warm["status"] = "completed"
                    except BaseException as error:
                        warm.update(status="failed", error=type(error).__name__ + ": " + str(error))
                        raise
                    finally:
                        warm["elapsed_ms"] = (time.perf_counter() - warm_started) * 1000
                        record["warmups"].append(warm)
                        record["active_unit"] = None
                        save()
                    if is_cached:
                        base_encode = options.get("encode_video", context.worker.encode_prompt)
                        options.update(encode_video=lambda spec, encode=base_encode: encode(cache_spec(spec)),
                                       prepare_video=lambda spec, payload: context.worker.prepare_encoded(cache_spec(spec), payload))
                    context.save_partial = journal.start_trace(dict(trace_sha256=trace["trace_sha256"],
                        order_index=order_index, factorial_arm=arm, arm=arm,
                        seed=trace["seed"], load=trace["load"], pattern=trace["pattern"]))
                    with context.sana.steady_context():
                        measured = run_trace(context, trace["requests"], options)
                    record["rows"].append(dict(measured, order_index=order_index, factorial_arm=arm,
                        seed=trace["seed"], load=trace["load"], pattern=trace["pattern"]))
                    save()
        comparisons = {}
        for name, baseline, candidate in (("preemption", "off_step", "off_k4"),
                ("cache", "off_step", "paired_step"), ("cache_on_k4", "off_k4", "paired_k4"),
                ("preemption_on_cache", "paired_step", "paired_k4")):
            rows = [dict(row, arm="baseline" if row["factorial_arm"] == baseline else "candidate")
                    for row in record["rows"] if row["factorial_arm"] in (baseline, candidate)]
            comparisons[name] = paired_gate(rows)
        record.update(status="completed", comparisons=comparisons)
        gate = comparisons["cache_on_k4"]
        result.update(online_qualified=gate["passed"], cache_factorial=comparisons,
                      online_evaluation_complete=bool(not gate.get("failures") and gate.get("pooled")))
        save()
        return result
    except BaseException as error:
        record.update(status="incomplete", error=type(error).__name__ + ": " + str(error))
        save()
        raise
