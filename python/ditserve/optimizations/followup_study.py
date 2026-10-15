"""Independent, serial single-seed screening; never changes an older study."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback
from types import SimpleNamespace

from ..research.common import atomic_json, digest, file_digest, read_json
from .protocol import ROOT
from .cache_executor import PairedCacheExecutor

PHASES = ("finish_copy", "sana_combo", "granularity")
WORK = ROOT / ".cache/results-work/optimization-followup/run01"
OUTPUT = ROOT / "results/optimization-followup/run01"


class FollowupPairedExecutor(PairedCacheExecutor):
    """Expose the existing advance(..., 1) path without editing old protocols."""

    def quantum(self, state, policy):
        if policy == "block-1":
            return 1
        return super().quantum(state, policy)


def quick_traces(config):
    from .next_workloads import build_workloads
    traces = [t for t in build_workloads(config)["highload"]["0.5"] if t["seed"] == 101]
    if len(traces) != 2 or {t["pattern"] for t in traces} != {"ordinary", "burst"}:
        raise ValueError("fixed seed101 ordinary/burst pair absent")
    return traces


def validate_config(config):
    from .next_protocol import load_config
    expected = load_config()
    expected["artifact_root"] = str(WORK / "raw-arrays")
    expected["followup"] = dict(phases=list(PHASES), seed=101, load=.5,
        patterns=["ordinary", "burst"], orders=[["baseline", "candidate"], ["candidate", "baseline"]],
        expected_replays=8, expected_requests=192, image_baseline="vae_compile",
        cache="paired_008", fixed_wan="eager", fixed_vae="eager", fixed_encoder="cpu",
        default_promotion=False, quality_policy="unchanged", wait_idle_seconds=None)
    if config != expected:
        raise ValueError("follow-up protocol differs from frozen single-seed screen")
    return config


def new_config():
    from .next_protocol import load_config
    config = load_config()
    config["artifact_root"] = str(WORK / "raw-arrays")
    config["followup"] = dict(phases=list(PHASES), seed=101, load=.5,
        patterns=["ordinary", "burst"], orders=[["baseline", "candidate"], ["candidate", "baseline"]],
        expected_replays=8, expected_requests=192, image_baseline="vae_compile",
        cache="paired_008", fixed_wan="eager", fixed_vae="eager", fixed_encoder="cpu",
        default_promotion=False, quality_policy="unchanged", wait_idle_seconds=None)
    return validate_config(config)


def _save(context):
    atomic_json(context.output / "phase-result.json", context.result)


def _warm_sana(context, candidate):
    context.sana.set_candidate(candidate)
    rows = []
    for index, prompt in enumerate(context.opts["prompts"]):
        started = time.perf_counter()
        image, service = context.sana.generate(prompt, seed=101 + index, check_budget=context.check_budget)
        rows.append(dict(candidate=candidate, index=index, service=service,
                         wall_ms=(time.perf_counter() - started) * 1000))
        del image
    context.result["costs"].setdefault("sana_warmups", []).extend(rows)
    _save(context)


def compare_arrays(reference, candidate):
    """Chunked strict comparison avoids full-size float64 video temporaries."""
    import numpy as np
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return dict(passed=False, exact=False, reason="shape_or_dtype_mismatch")
    finite, exact, close, max_abs, square, count = True, True, True, 0., 0., 0
    minimum, maximum = float("inf"), float("-inf")
    # Video/image leading axis slices stay bounded even for a strided NHWC view.
    a = reference.reshape((-1,) + reference.shape[-3:])
    b = candidate.reshape((-1,) + candidate.shape[-3:])
    for left, right in zip(a, b):
        chunk_finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        finite = finite and chunk_finite
        exact = exact and bool(np.array_equal(left, right))
        close = close and bool(np.allclose(left, right, rtol=1e-5, atol=1e-6))
        if not chunk_finite:
            continue
        delta = left.astype(np.float64) - right.astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
        square += float(np.square(delta).sum())
        count += delta.size
        minimum = min(minimum, float(left.min()), float(right.min()))
        maximum = max(maximum, float(left.max()), float(right.max()))
    mse = square / count if count and finite else None
    valid = finite and minimum >= 0. and maximum <= 1.
    return dict(passed=valid and close, exact=exact, finite=finite, range_valid=valid,
        minimum=minimum if finite and count else None, maximum=maximum if finite and count else None,
        shape=list(reference.shape), dtype=str(reference.dtype),
        max_abs=max_abs if finite else None, mse=mse, psnr_db=(None if not mse else -10 * np.log10(mse)),
        psnr_infinite=mse == 0., rtol=1e-5, atol=1e-6,
        lpips=0. if exact and valid else None, lpips_scope="zero inferred only for identical arrays")


def event_summary(row):
    from .online import _overlap
    from ..directions.direction1 import quantile
    events = row.get("events", [])
    intervals, summaries = {}, {}
    for event in events:
        kind = event.get("kind", "unknown")
        group = summaries.setdefault(kind, dict(count=0, durations=[], missing_duration=0))
        group["count"] += 1
        start, end = event.get("start_ms"), event.get("end_ms")
        if isinstance(start, (int,float)) and isinstance(end, (int,float)) and end >= start:
            group["durations"].append(end-start)
            intervals.setdefault(kind, []).append((start,end))
        else:
            group["missing_duration"] += 1
    for group in summaries.values():
        durations = group.pop("durations")
        group.update(n=len(durations), sum_ms=sum(durations),
            mean_ms=statistics.mean(durations) if durations else None, p95_ms=quantile(durations,.95),
            max_ms=max(durations) if durations else None)
    for request in row.get("requests", []):
        start = request.get("start_ms")
        if request["kind"] != "image" or start is None:
            continue
        request["wait_attribution"] = dict(wait_ms=start-request["arrival_ms"],
            by_event_kind_ms={kind:_overlap(spans,request["arrival_ms"],start) for kind,spans in intervals.items()},
            scope="union within each kind; nested kinds overlap and must not be summed")
    row["event_summary"] = dict(total_events=len(events), by_kind=summaries,
        scope="exact count/sum/mean/p95/max of host intervals; no GPU-kernel causal claim")
    return row


def _resume(context, item, *, quantum, copy_output, order_index):
    """Two private requests, checked after every step, with real Sana insertion."""
    from . import wan_phases as phases
    from .followup_finish import finish_copy
    from ..directions.direction1 import solver_matches
    from ..research.prompt_bank import FrozenPromptBank
    worker, torch, executor = context.worker, context.worker.torch, context.executor
    bank = FrozenPromptBank(worker)
    states, continuations, outputs = {}, {}, {}
    result = dict(input_id=item["input_id"], quantum=quantum, copy_output=copy_output,
        order_index=order_index, steps=[], interruptions=[], finish=[], passed=False)
    context.result["local"].append(result)
    try:
        for arm in ("baseline", "candidate"):
            states[arm] = bank.prepare(phases._spec(item, True))
            continuations[arm] = executor.attach(states[arm])
        left, right = states["baseline"], states["candidate"]
        for step in range(32):
            while left.step_index == step:
                executor.advance(continuations["baseline"], 4, check_budget=context.check_budget)
            injected = False
            while right.step_index == step:
                executor.advance(continuations["candidate"], quantum, check_budget=context.check_budget)
                if step in (0, 8, 16) and not injected and right.step_index == step:
                    with context.sana.steady_context():
                        _, service = context.sana.generate(item["prompt"], seed=item["seed"],
                                                           check_budget=context.check_budget)
                    result["interruptions"].append(dict(step=step, service=service))
                    injected = True
            checks = dict(latent=bool(torch.isfinite(left.latents).all() and torch.isfinite(right.latents).all()
                and torch.allclose(left.latents, right.latents, rtol=1e-5, atol=1e-6)),
                solver=solver_matches(torch, left.scheduler, right.scheduler),
                cache_state=phases._cache_equal(torch, left, right),
                rng=bool(torch.equal(left.generator.get_state(), right.generator.get_state())))
            result["steps"].append(dict(step=step+1, passed=all(checks.values()), **checks))
            _save(context)
            if not all(checks.values()):
                result["reason"] = "same-backend step state mismatch"
                result["failed_latents"] = [phases._artifact(context,
                    "%s-%s-%s" % (context.result["phase"], item["input_id"], arm),
                    state.latents.detach().float().cpu().numpy()) for arm, state in states.items()]
                return False
        from .cache_online import _warm_cache_paths
        paths = {arm: _warm_cache_paths(state.events) for arm, state in states.items()}
        result["cache_paths"] = paths
        path_pass = all(v["full_reuse_observed"] for v in paths.values()) and (
            paths["baseline"]["path_sequence"] == paths["candidate"]["path_sequence"])
        arms = ("baseline", "candidate") if order_index % 2 == 0 else ("candidate", "baseline")
        for arm in arms:
            context.check_budget()
            with finish_copy(worker, enabled=copy_output and arm == "candidate") as observed:
                started = time.perf_counter()
                outputs[arm] = worker.finish(states[arm])
                wall_ms = (time.perf_counter() - started) * 1000.
                stats = observed.validate(expected_calls=1)
            result["finish"].append(dict(arm=arm, wall_ms=wall_ms, postprocess=stats))
            _save(context)
        result["final"] = compare_arrays(outputs["baseline"], outputs["candidate"])
        result["passed"] = bool(result["final"]["passed"] and path_pass and len(result["interruptions"]) == 3
            and (not copy_output or result["final"]["exact"]))
        if not result["passed"]:
            result["failed_outputs"] = [phases._artifact(context,
                "%s-%s-%s-video" % (context.result["phase"], item["input_id"], arm), value)
                for arm, value in outputs.items()]
        return result["passed"]
    finally:
        outputs.clear()
        for continuation in continuations.values():
            if not continuation.closed:
                executor.close(continuation)
        for state in states.values():
            if state.residency != "released":
                worker.release(state)
        bank.entries.clear()
        _save(context)


def _combo_quality(context):
    from .sana_phases import _quality
    from .next_phases import _mechanics
    evaluator = SimpleNamespace(quality_evaluator=None, quality_epsilon=.01)
    references, hashes, qualities, timings = {}, {}, [], []
    inputs = [dict(input_id="p%d-s%d" % (i, seed), prompt=prompt, seed=seed,
                   split="holdout" if seed in (201, 202) else "development")
              for i, prompt in enumerate(context.opts["prompts"]) for seed in (101,102,201,202)]
    _warm_sana(context, "eager")
    for item in inputs:
        image, _ = context.sana.generate(item["prompt"], seed=item["seed"], check_budget=context.check_budget)
        references[item["input_id"]] = image.copy()
    for order_index, arms in enumerate((("baseline", "candidate"), ("candidate", "baseline"))):
        for arm in arms:
            _warm_sana(context, "vae_compile" if arm == "baseline" else "vae_graph")
            for item in inputs:
                for repeat in range(3):
                    before = _mechanics(context.sana.candidate_summary())
                    started = time.perf_counter()
                    with context.sana.steady_context():
                        image, service = context.sana.generate(item["prompt"], seed=item["seed"],
                                                               check_budget=context.check_budget)
                    elapsed = (time.perf_counter()-started)*1000
                    if before != _mechanics(context.sana.candidate_summary()):
                        raise RuntimeError("compile/capture during measurement")
                    sha = hashlib.sha256(image.tobytes()).hexdigest()
                    key = (arm, item["input_id"])
                    exact = hashes.setdefault(key, sha) == sha
                    row = dict(kind="sana_local", arm=arm, order_index=order_index, repeat=repeat,
                        input_id=item["input_id"], split=item["split"], service=service,
                        wall_ms=elapsed, output_sha256=sha, repeat_exact=exact)
                    if repeat == 0:
                        row["quality"] = _quality(evaluator, references[item["input_id"]], image)
                        qualities.append(row["quality"]["passed"])
                    context.result["local"].append(row)
                    timings.append(row)
                    del image
                    _save(context)
                    if not exact or row.get("quality", {}).get("passed", True) is not True:
                        context.result["quality"].update(passed=False, checked=len(qualities))
                        return False
    gains = []
    for order_index in range(2):
        means = {arm: statistics.mean(r["wall_ms"] for r in timings if r["arm"] == arm
            and r["order_index"] == order_index) for arm in ("baseline", "candidate")}
        gains.append(1.-means["candidate"]/means["baseline"])
    context.result["quality"].update(passed=len(qualities)==32 and all(qualities), checked=len(qualities),
        image_lpips_limit=.01, reference="eager serial per input")
    context.result["costs"]["local_gains_vs_vae_compile"] = gains
    context.result["costs"]["mechanics"] = context.sana.candidate_summary()
    _save(context)
    # A correctly combined candidate can improve burst deadlines even if its
    # standalone gain over the already optimized VAE is below five percent.
    return context.result["quality"]["passed"]


def _online(context, phase):
    from .online import TraceJournal, run_trace, paired_gate, request_identity, instrumented_finish
    from .cache_online import cache_spec
    from .followup_finish import finish_copy
    from .next_protocol import DEFAULTS
    traces = quick_traces(context.config)
    online = context.result["online"]
    online.update(status="running", expected_replays=8, expected_requests=192,
        expected_trace_ids=[t["trace_sha256"] for t in traces], replays=[])
    for trace in traces:
        for order_index, arms in enumerate((("baseline", "candidate"), ("candidate", "baseline"))):
            for arm in arms:
                if context.worker._states or context.executor._states:
                    raise RuntimeError("private request state leaked across replay boundaries")
                _warm_sana(context, "vae_graph" if phase=="sana_combo" and arm=="candidate" else "vae_compile")
                identity = dict(trace_sha256=trace["trace_sha256"], seed=trace["seed"], load=trace["load"],
                    pattern=trace["pattern"], order_index=order_index, arm=arm)
                directory = context.output / "online-partials" / ("replay-%02d" % len(online["replays"]))
                directory.mkdir(parents=True, exist_ok=False)
                journal = TraceJournal(directory / "trace.json", atomic_json)
                context.save_partial = journal.start_trace(identity)
                online["active_unit"] = dict(identity, partial_path=str(journal.partial_path))
                _save(context)
                options = dict(DEFAULTS, policy="block-1" if phase=="granularity" and arm=="candidate" else "block-4",
                    encode_video=lambda spec: context.worker.encode_prompt(cache_spec(spec)),
                    prepare_video=lambda spec, encoded: context.worker.prepare_encoded(cache_spec(spec), encoded))
                # Install the copy wrapper outside the observer so its strict
                # original-source guard sees the genuine Diffusers method.
                with finish_copy(context.worker, enabled=phase=="finish_copy" and arm=="candidate") as observed:
                    options["finish_video"] = lambda state, emit: instrumented_finish(context.worker, state, emit)
                    with context.sana.steady_context():
                        measured = run_trace(context, trace["requests"], options)
                    postprocess = observed.summary()
                row = dict(measured, **{k:v for k,v in identity.items() if k!="trace_sha256"},
                    postprocess=postprocess, partial_path=str(journal.partial_path), flush_log_path=str(journal.flush_path))
                event_summary(row)
                online["replays"].append(row)
                online["active_unit"] = None
                _save(context)
                if measured.get("trace_sha256") != trace["trace_sha256"] or request_identity(measured["requests"]) != trace["trace_sha256"]:
                    raise ValueError("frozen arrival/deadline identity changed")
                if measured.get("status") != "completed" or not measured.get("complete"):
                    raise RuntimeError("failed or incomplete requests retained in replay denominator")
                observed.validate(expected_calls=sum(r["kind"]=="video" for r in trace["requests"]))
    gate = paired_gate(online["replays"], expected_trace_ids=online["expected_trace_ids"])
    online.update(status="completed", comparison_gate=gate, quick_screen_passed=gate["passed"],
        default_qualified=False, scope="seed101 at load0.5 only; no default promotion or stable p99 claim")
    context.result.update(status="completed", verdict="值得继续" if gate["passed"] else "当前配置无收益")
    _save(context)


def _verify_admission(args, config):
    from .budget import BudgetStop
    path = args.output / "admission.json"
    admission = read_json(path)
    deadline = time.monotonic()+5
    while True:
        active = read_json(args.ledger).get("active")
        identity = (active or {}).get("identity", {})
        expected = dict(path=str(path.resolve()), sha256=file_digest(path))
        if (not active or identity.get("admission") != expected or
            active.get("supervisor_pid") != args.supervisor_pid or os.getppid()!=args.supervisor_pid or
            os.getpid()!=os.getpgrp() or identity.get("phase")!=args.phase or
            identity.get("config_sha256")!=digest(config) or identity.get("output")!=str(args.output.resolve()) or
            identity.get("run_root")!=str(args.run_root.resolve()) or active.get("gpu_count")!=1 or
            admission.get("schema_version")!="ditserve.optimizations.followup_admission.v1" or
            admission.get("phase")!=args.phase or admission.get("output")!=str(args.output.resolve()) or
            admission.get("config_sha256")!=digest(config) or
            admission.get("source_sha256")!=identity.get("source_sha256") or
            admission.get("study_binding")!=identity.get("study_binding") or
            admission.get("gpu_uuids")!=[config["gpu_uuid"]] or
            os.environ.get("CUDA_VISIBLE_DEVICES")!=config["gpu_uuid"]):
            raise BudgetStop("follow-up worker has no matching charged admission")
        if active.get("child_pgid") == os.getpgrp():
            break
        if active.get("child_pgid") is not None or time.monotonic()>=deadline:
            raise BudgetStop("child process group not admitted")
        time.sleep(.05)
    for binding in [admission["authorization_source"], admission["source_manifest"], admission["study_binding"]] + admission["parent_ledgers"]:
        if file_digest(binding["path"]) != binding["sha256"]:
            raise BudgetStop("bound admission source changed")
    source = read_json(admission["source_manifest"]["path"])
    if digest(source["source_files"]) != source.get("source_sha256") or source["source_sha256"]!=admission["source_sha256"]:
        raise BudgetStop("executed source manifest is internally inconsistent")
    for name, sha in source["source_files"].items():
        if file_digest(Path(admission["execution_root"]) / name) != sha:
            raise BudgetStop("executed source snapshot changed")
    if Path(__file__).resolve() != Path(admission["execution_root"]) / "python/ditserve/optimizations/followup_study.py":
        raise BudgetStop("worker is not executing admitted snapshot")
    from .next_study import verify_cache_binding
    study = read_json(args.run_root / "study.json")
    verify_cache_binding(study["cache_local_binding"], admission["execution_root"])
    for binding in study["baseline_bindings"]:
        if file_digest(binding["path"]) != binding["sha256"]:
            raise BudgetStop("qualified Sana baseline changed")
    if digest(quick_traces(config)) != study["trace_sha256"]:
        raise BudgetStop("frozen online request source changed")
    return active


def worker_main(args):
    from .budget import BudgetGuard
    result = dict(schema_version=1, phase=args.phase, status="running", verdict="未验证",
        quality={}, local=[], online=dict(status="unverified"), costs={}, failures=[], default_qualified=False)
    context = None
    try:
        config = read_json(args.config)
        # The config is already bound by the charged admission; do not resolve
        # workspace defaults from inside the isolated snapshot.
        _verify_admission(args, config)
        context = SimpleNamespace(config=config, output=args.output, run_root=args.run_root,
            check_budget=BudgetGuard(float("inf"), args.supervisor_pid), opts=config["direction1"],
            profile=dict(selected_shape=dict(height=480,width=832,frames=81)), metadata={},
            result=result, ledger=args.ledger, admitted_devices=[config["gpu_uuid"]], executors={})
        _save(context)
        from ..gpu.worker import WanStepWorker
        from .followup_sana import FollowupSanaAdapter
        from .worker import capacity
        started = time.perf_counter()
        context.worker = WanStepWorker(device=config["device"], model_path=config["models"]["wan"]["path"], attention="flash")
        context.sana = FollowupSanaAdapter(config["models"]["sana"]["path"], device=config["device"])
        context.executor = FollowupPairedExecutor(context.worker)
        context.executors["paired"] = context.executor
        result["costs"]["model_load_ms"] = (time.perf_counter()-started)*1000
        _warm_sana(context, "vae_compile")
        if args.phase == "sana_combo" and not _combo_quality(context):
            result.update(status="stopped", verdict="当前配置无收益", reason="combined Sana quality gate failed")
            return 0
        _warm_sana(context, "vae_graph" if args.phase=="sana_combo" else "vae_compile")
        result["costs"]["capacity"] = capacity(context)
        from . import wan_phases as phases
        inputs = phases._inputs(context) if args.phase != "sana_combo" else [phases._cache_inputs()[0]]
        for index, item in enumerate(inputs):
            if not _resume(context, item, quantum=1 if args.phase=="granularity" else 4,
                           copy_output=args.phase=="finish_copy", order_index=index):
                result.update(status="stopped", verdict="当前配置无收益", reason="same-backend recovery/copy gate failed")
                return 0
        result["quality"].update(resume_passed=True, resume_inputs=len(inputs), video_rtol=1e-5, video_atol=1e-6)
        _online(context, args.phase)
        return 0
    except BaseException as error:
        result.update(status="failed", verdict="未验证", reason=type(error).__name__+": "+str(error))
        result["failures"].append(dict(error=result["reason"], traceback=traceback.format_exc()))
        return 1
    finally:
        if context is not None:
            try:
                if getattr(context, "worker", None) is not None:
                    result["costs"]["memory"] = context.worker._memory()
                    from .next_study import _cleanup
                    _cleanup(context)
            except BaseException as error:
                result.update(status="failed", verdict="未验证")
                result["failures"].append(dict(cleanup_error=type(error).__name__+": "+str(error)))
        atomic_json(args.output / "phase-result.json", result)
        atomic_json(args.output / "worker.json", dict(phase=args.phase, status=result["status"],
            decision=result["verdict"], verdict=result["verdict"], reason=result.get("reason"),
            evidence_path=str(args.output / "phase-result.json"),
            quick_screen_passed=result["online"].get("quick_screen_passed", False),
            default_qualified=False, online_qualified=False,
            completed_replays=len(result["online"].get("replays", []))))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("worker",))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    args = parser.parse_args(argv)
    return worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
