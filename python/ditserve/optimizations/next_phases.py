"""Local qualification before paired online Sana candidate measurements."""
import hashlib
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace

from ..research.common import atomic_json, file_digest, read_json
from ..directions.direction1 import quantile
from .budget import BudgetStop
from .next_protocol import DEFAULTS


def stats(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                p95=quantile(values, .95))


def profile_summary(rows):
    full = sum(row["service"]["ms"] for row in rows)
    stages = {}
    for row in rows:
        for stage in row["service"]["stages"]:
            if "host_ms" in stage:
                target = stages.setdefault(stage["stage"], dict(host=[], cuda=[]))
                target["host"].append(stage["host_ms"])
                target["cuda"].append(stage["cuda_elapsed_ms"])
    host_transformer = sum(stages.get("transformer", {}).get("host", []))
    return dict(full_service=stats(r["service"]["ms"] for r in rows),
        stages={key: {kind: stats(values) for kind, values in group.items()} for key, group in stages.items()},
        transformer_host_enqueue_fraction=host_transformer/full if full else None,
        graph_probe_trigger=bool(full and host_transformer/full >= .05),
        scope="host enqueue cost is a trigger for an empirical graph probe, not proof of recoverable GPU idle time; stage host/CUDA spans overlap")


def profile_sana(context):
    path = context.output / "sana-profile.json"
    record = dict(status="running", rows=[], active_unit=None, candidate="eager")
    adapter = context.sana
    previous = adapter.profile_extra
    try:
        adapter.set_candidate("eager")
        adapter.profile_extra = True
        for p, prompt in enumerate(context.opts["prompts"]):
            for seed in (101, 102):
                for repeat in range(3):
                    record["active_unit"] = dict(prompt_id=p, seed=seed, repeat=repeat)
                    atomic_json(path, record)
                    image, service = adapter.generate(prompt, seed=seed, check_budget=context.check_budget)
                    record["rows"].append(dict(record["active_unit"], service=service,
                        output_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
                    del image
        record.update(status="completed", active_unit=None, summary=profile_summary(record["rows"]))
        atomic_json(path, record)
        return record
    except BaseException as error:
        record.update(status="incomplete", reason=type(error).__name__ + ": " + str(error))
        atomic_json(path, record)
        raise
    finally:
        adapter.profile_extra = previous


def _mechanics(summary):
    return (sum(r["graph_compile_count"] for r in summary["vae_regions"]),
            (summary.get("transformer_graph") or {}).get("capture_count", 0))


def graph_trigger(run_root):
    from .reporting import _authorized
    terminal_path = Path(run_root) / "diagnosis/terminal.json"
    if not terminal_path.exists():
        return dict(triggered=False, reason="complete supervised diagnosis absent")
    diagnosis = read_json(terminal_path)
    if not _authorized(diagnosis) or diagnosis.get("status") != "completed":
        return dict(triggered=False, reason="diagnostic profile lacks completed isolated charged admission",
                    terminal=dict(path=str(terminal_path), sha256=file_digest(terminal_path)))
    profile_path = Path(diagnosis["output"]) / "sana-profile.json"
    profile = read_json(profile_path) if profile_path.exists() else {}
    return dict(triggered=bool(profile.get("status") == "completed" and
                             profile.get("summary", {}).get("graph_probe_trigger")),
        terminal=dict(path=str(terminal_path), sha256=file_digest(terminal_path)),
        profile=dict(path=str(profile_path), sha256=file_digest(profile_path) if profile_path.exists() else None),
        summary=profile.get("summary"), scope="candidate trigger only, not measured graph benefit")


def _key_image(context, image, name):
    from PIL import Image
    import numpy as np
    path = context.output / "images" / (name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(image[0] * 255).astype(np.uint8)).save(path)
    return dict(path=str(path), sha256=file_digest(path), scope="quantized preview; metrics use float output")


def _candidate_online(context, candidate, result):
    from .cache_online import warm_cache_factorial, cache_spec
    from .online import TraceJournal, load_frozen_traces, paired_gate, run_trace
    warm_cache_factorial(context)
    context.executor = context.executors["cache_factorial_paired"]
    traces = load_frozen_traces(context.config["old_online"])
    record = dict(status="running", rows=[], active_unit=None,
        candidate=candidate, wan="paired_008+K4 eager", baseline="eager Sana + qualified paired_008+K4",
        scope="same frozen requests/deadlines; full prompt encoding and output; no stage profiling wrappers")
    path = context.output / "sana-online.json"
    atomic_json(path, record)
    try:
        for trace in traces:
            for order, arms in enumerate((("baseline", "candidate"), ("candidate", "baseline"))):
                for arm in arms:
                    context.sana.set_candidate(candidate if arm == "candidate" else "eager")
                    for i in range(2):
                        context.sana.generate(context.opts["prompts"][i], seed=101+i,
                                              check_budget=context.check_budget)
                    options = dict(DEFAULTS,
                        encode_video=lambda spec: context.worker.encode_prompt(cache_spec(spec)),
                        prepare_video=lambda spec, encoded: context.worker.prepare_encoded(cache_spec(spec), encoded))
                    identity = dict(trace_sha256=trace["trace_sha256"], seed=trace["seed"], load=trace["load"],
                        pattern=trace["pattern"], order_index=order, arm=arm)
                    directory = context.output / "online-partials" / ("replay-%02d" % len(record["rows"]))
                    directory.mkdir(parents=True)
                    journal = TraceJournal(directory / "trace.json", atomic_json)
                    context.save_partial = journal.start_trace(identity)
                    record["active_unit"] = dict(identity, partial_path=str(journal.partial_path))
                    atomic_json(path, record)
                    with context.sana.steady_context():
                        row = run_trace(context, trace["requests"], options)
                    record["rows"].append(dict(row, **{k:v for k,v in identity.items() if k != "trace_sha256"}))
                    record["active_unit"] = None
                    atomic_json(path, record)
        gate = paired_gate(record["rows"])
        record.update(status="completed", comparison=gate)
        result.update(online_qualified=gate["passed"],
            online_evaluation_complete=not gate["failures"] and gate["pooled"] is not None,
            online_path=str(path), online_comparison=gate,
            decision="值得继续" if gate["passed"] else "当前配置无收益",
            reason="qualified locally and passed fixed online gate" if gate["passed"] else
                   "local speed gain did not pass fixed online system gate")
        atomic_json(path, record)
    except BaseException as error:
        record.update(status="incomplete", reason=type(error).__name__ + ": " + str(error))
        atomic_json(path, record)
        raise


def run_candidate(context, candidate):
    from .sana_phases import _quality
    from .worker import capacity
    adapter = context.sana
    path = context.output / "sana-candidate.json"
    record = dict(status="running", candidate=candidate, rows=[], active_unit=None,
        local_qualified=False, online_qualified=False, decision="未验证",
        quality_policy=context.config["quality"], scope="quality first, both local orders, capacity then online")
    atomic_json(path, record)
    if candidate == "transformer_graph":
        trigger = graph_trigger(context.run_root)
        record["trigger_evidence"] = trigger
        if not trigger["triggered"]:
            record.update(status="stopped", decision="未验证", reason="predeclared host enqueue trigger absent or profile incomplete",
                trigger_evidence=trigger)
            atomic_json(path, record)
            return record
    inputs = [dict(input_id="p%d-s%d" % (i, seed), prompt=prompt, seed=seed,
                   split="holdout" if seed in (201, 202) else "development")
              for i, prompt in enumerate(context.opts["prompts"]) for seed in context.config["seeds"]]
    quality_context = SimpleNamespace(quality_evaluator=None, quality_epsilon=.01)
    references, hashes = {}, {}
    try:
        for arm in ("eager", candidate):
            adapter.set_candidate(arm)
            for i in range(2):
                record["active_unit"] = dict(kind="warmup", arm=arm, index=i)
                atomic_json(path, record)
                started = time.perf_counter()
                _, service = adapter.generate(context.opts["prompts"][i], seed=101+i,
                                               check_budget=context.check_budget)
                service["invocation_wall_ms"] = (time.perf_counter()-started)*1000.
                record["rows"].append(dict(record["active_unit"], service=service))
        qualities = []
        for order_index, order in enumerate((("eager", candidate), (candidate, "eager"))):
            for arm in order:
                adapter.set_candidate(arm)
                for item in inputs:
                    for repeat in range(3):
                        identity = dict(kind="measurement", order_index=order_index, arm=arm,
                            input_id=item["input_id"], split=item["split"], repeat=repeat)
                        record["active_unit"] = identity
                        atomic_json(path, record)
                        before = _mechanics(adapter.candidate_summary())
                        started = time.perf_counter()
                        with adapter.steady_context():
                            image, service = adapter.generate(item["prompt"], seed=item["seed"],
                                                               check_budget=context.check_budget)
                        service["invocation_wall_ms"] = (time.perf_counter()-started)*1000.
                        if before != _mechanics(adapter.candidate_summary()):
                            raise RuntimeError("compilation or graph capture during measured request")
                        row = dict(identity, service=service, output_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                        key = arm, item["input_id"]
                        row["repeat_exact"] = hashes.setdefault(key, row["output_sha256"]) == row["output_sha256"]
                        if repeat == 0:
                            if arm == "eager" and order_index == 0:
                                references[item["input_id"]] = image.copy()
                            row["quality"] = _quality(quality_context, references[item["input_id"]], image)
                            qualities.append(row["quality"]["passed"])
                            if item["input_id"] == "p0-s101":
                                row["preview"] = _key_image(context, image, "r%d-%s" % (order_index, arm))
                        record["rows"].append(row)
                        atomic_json(path, record)
                        if not row["repeat_exact"] or ("quality" in row and not row["quality"]["passed"]):
                            record.update(status="stopped", active_unit=None, quality_passed=False,
                                quality_checked=len(qualities), decision="当前配置无收益",
                                reason="fixed image quality or repeat consistency gate failed",
                                mechanics=adapter.candidate_summary())
                            atomic_json(path, record)
                            return record
        measured = [row for row in record["rows"] if row["kind"] == "measurement"]
        gains = []
        for index in range(2):
            means = {arm: statistics.mean(row["service"]["invocation_wall_ms"] for row in measured
                     if row["order_index"] == index and row["arm"] == arm) for arm in ("eager", candidate)}
            gains.append(1-means[candidate]/means["eager"])
        quality_passed = len(qualities) == 32 and all(qualities)
        performance_passed = all(gain >= .05 for gain in gains)
        record.update(status="completed", active_unit=None, quality_passed=quality_passed,
            quality_checked=len(qualities), round_speed_gains=gains, performance_passed=performance_passed,
            timing_scope="full generate invocation including validation/metadata; synchronized pipeline ms separately recorded",
            summaries={arm: stats(row["service"]["invocation_wall_ms"] for row in measured if row["arm"] == arm)
                       for arm in ("eager", candidate)}, mechanics=adapter.candidate_summary(),
            decision="当前配置无收益", reason="local full-request gain did not reach 5% in both orders")
        if quality_passed and performance_passed:
            adapter.set_candidate(candidate)
            record["capacity"] = capacity(context)
            if record["capacity"].get("passed") is True:
                from .next_resume import run_candidate_resume
                resume = run_candidate_resume(context)
                record["recovery"] = dict(path=str(context.output / "sana-wan-resume.json"),
                    sha256=file_digest(context.output / "sana-wan-resume.json"),
                    status=resume.get("status"), passed=resume.get("resume_qualified") is True,
                    gate=resume.get("gate"), scope=resume.get("scope"))
                record["local_qualified"] = bool(resume.get("status") == "passed" and
                    resume.get("resume_qualified") is True and resume.get("eligible") is True)
                if not record["local_qualified"]:
                    record.update(decision="未验证", reason="cross-model interruption recovery regression failed")
                else:
                    record.update(decision="未验证", reason="local gates passed; fixed online evaluation pending")
            else:
                record.update(decision="未验证", reason="candidate co-residency capacity check did not pass")
            atomic_json(path, record)
            if record["local_qualified"]:
                _candidate_online(context, candidate, record)
        atomic_json(path, record)
        return record
    except BaseException as error:
        from .next_candidate_failures import classify, restore_and_verify
        failed_at = time.perf_counter()
        mechanics = adapter.candidate_summary()
        record.update(status="incomplete" if isinstance(error, BudgetStop) else "failed", decision="未验证",
            reason=type(error).__name__ + ": " + str(error), mechanics=mechanics,
            traceback=traceback.format_exc())
        rejection = (None if record.get("local_qualified") else
                     classify(error, candidate, record.get("active_unit"), mechanics))
        if rejection is not None:
            record["expected_failure"] = dict(rejection, failed_unit=record.get("active_unit"),
                failed_invocation_wall_ms=(failed_at-started)*1000.)
            # Persist the original failure and cold costs before attempting any
            # restoration. A cleanup exception remains fatal to the supervisor.
            atomic_json(path, record)
            try:
                record["expected_failure"]["cleanup"] = restore_and_verify(adapter)
            except BaseException as cleanup:
                record["expected_failure"]["cleanup"] = dict(passed=False,
                    reason=type(cleanup).__name__ + ": " + str(cleanup), traceback=traceback.format_exc())
                atomic_json(path, record)
                raise
            record.update(status="stopped", active_unit=None, local_qualified=False,
                online_qualified=False, online_evaluation_complete=False,
                reason="candidate screening rejected: " + rejection["classification"],
                failure_reason=type(error).__name__ + ": " + str(error))
            atomic_json(path, record)
            return record
        atomic_json(path, record)
        raise
    finally:
        references.clear()
        adapter.set_candidate("eager")


def conditional_conclusions(context):
    # This phase is evaluated after all three load matrices, never from old low-load p95 alone.
    rows = []
    for phase in ("cache_load_05", "cache_load_07", "cache_load_09"):
        terminal = read_json(context.run_root / phase / "terminal.json")
        path = Path(terminal["output"]) / (phase + ".json")
        if path.exists():
            data = read_json(path)
            rows.extend(data.get("rows", []))
    triggers = dict(encoding=any(r.get("attribution", {}).get("encoding_trigger") for r in rows),
                    video_vae=any(r.get("attribution", {}).get("vae_trigger") for r in rows))
    result = dict(status="stopped", decision="未验证", triggers=triggers,
        reason="conditional extensions require their specific collision/encoder/capacity evidence; no adaptive controller is selected",
        scope="K remains 4; no multi-GPU, precision, model or quality changes")
    atomic_json(context.output / "conditional.json", result)
    return result
