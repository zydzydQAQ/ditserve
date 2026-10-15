"""Admitted worker; model imports and all GPU work occur inside a charged session."""
import argparse
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from ..research.common import atomic_json, read_json, digest, file_digest
from ..directions.budget import BudgetStop
from .budget import BudgetGuard
from .protocol import qualified


def verify_admission(args, config):
    deadline = time.monotonic() + 5.
    while True:
        active = read_json(args.ledger).get("active")
        identity = active.get("identity", {}) if active else {}
        devices = [config["gpu_uuid"]] + ([config["v100_uuid"]] if getattr(args, "auxiliary_encoder", False) else [])
        if (not active or active.get("session_id") != args.session_id or active.get("supervisor_pid") != os.getppid()
                or os.getpgrp() != os.getpid() or identity.get("phase") != args.phase
                or identity.get("output") != str(args.output.resolve()) or identity.get("config_sha256") != digest(config)
                or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(devices) or active.get("gpu_count") != len(devices)
                or identity.get("gpu_uuids", [config["gpu_uuid"]]) != devices):
            raise BudgetStop("worker identity does not match charged supervisor admission")
        if active.get("stage") not in (args.phase, "reserve"):
            raise BudgetStop("wrong allocation stage")
        replay_path = getattr(args, "online_replay_from", None)
        replay_binding = identity.get("online_replay_from")
        if (replay_path is None) != (replay_binding is None):
            raise BudgetStop("online replay admission binding is missing or unexpected")
        if replay_path is not None:
            try:
                valid_replay = (args.phase == "wan_compile" and isinstance(replay_binding, dict)
                    and replay_binding.get("path") == str(Path(replay_path).resolve())
                    and replay_binding.get("sha256") == file_digest(replay_path))
            except (OSError, ValueError):
                valid_replay = False
            if not valid_replay:
                raise BudgetStop("online replay source differs from charged admission")
            args.online_replay_binding = dict(replay_binding)
        if active.get("child_pgid") is not None:
            if active["child_pgid"] != os.getpgrp():
                raise BudgetStop("wrong worker process group")
            return active
        if time.monotonic() > deadline:
            raise BudgetStop("supervisor did not register child")
        time.sleep(.05)


def _terminal(root, phase):
    path = root / phase / "terminal.json"
    return read_json(path) if path.exists() else {}


def selected(root):
    """Only supervisor-qualified online candidates may become a default."""
    sana = _terminal(root, "sana_compile")
    batch = _terminal(root, "sana_batch")
    wan = _terminal(root, "wan_compile")
    result = dict(sana_backend=sana.get("selected_backend", "eager") if qualified(sana) else "eager",
                  image_batch_size=2 if qualified(batch) else 1,
                  wan_backend=wan.get("selected_backend", "eager") if qualified(wan) else "eager")
    encoder = _terminal(root, "text_encoder")
    if qualified(encoder):
        device = encoder["encoder_device"]
        if device not in ("cuda:0", "cuda:1"):
            raise ValueError("qualified encoder names an unsupported device")
        result.update(encode_device=device, encode_on_scheduler=device == "cuda:0")
    return result


def capacity(context):
    from ..directions.direction1 import _spec
    context.check_budget.ensure_unit(210.)
    context.worker.torch.cuda.reset_peak_memory_stats(context.worker.device)
    spec = _spec(context.opts, context.profile["selected_shape"], "capacity-81f")
    started = time.perf_counter()
    state = context.worker.prepare(spec)
    try:
        while state.step_index < state.spec["steps"]:
            context.check_budget()
            context.worker.step(state)
        output = context.worker.finish(state)
        context.metadata["wan_warmed"] = True
        return dict(passed=True, frames=int(output.shape[1]), elapsed_ms=(time.perf_counter() - started) * 1000,
                    peak_allocated_bytes=context.worker.torch.cuda.max_memory_allocated(context.worker.device))
    finally:
        context.worker.release(state)


def _online(context, result, candidate=None, *, baseline_only=False):
    from .online import load_frozen_traces, run_trace, paired_gate, TraceJournal
    from .wan import BlockBackend, WanCompiledExecutor
    traces = load_frozen_traces(context.config["old_online"])
    baseline = selected(context.run_root)
    baseline.update(getattr(context, "online_defaults", {}))
    baseline.update(policy="block-4")
    candidate = dict(baseline, **(candidate or {}))
    visible = lambda options: {key: (getattr(value, "__name__", "callable") if callable(value) else value)
                               for key, value in options.items()}
    record = dict(schema_version="ditserve.optimizations.online.v1", rows=[], status="running",
                  baseline=visible(baseline), candidate=visible(candidate),
                  traces=[{k: v for k, v in t.items() if k != "requests"} for t in traces])
    path = context.output / "online.json"
    journal = TraceJournal(path, atomic_json)
    result["online_path"] = str(path)
    context.executors = getattr(context, "executors", {})
    context.executors.setdefault("eager", context.executor)

    def save():
        journal.save_record(record)

    def prepare(options):
        if hasattr(context, "prepare_online_arm"):
            context.prepare_online_arm(options)
        context.sana.set_backend(options["sana_backend"])
        name = options["wan_backend"]
        if name not in context.executors:
            context.executors[name] = WanCompiledExecutor(context.worker, BlockBackend(context.worker,
                compiled=True, group_size=4 if name.endswith("4") else 1))
            from ..directions.direction1 import _spec, _release
            context.check_budget.ensure_unit(360.)
            warm_spec = _spec(context.opts, context.profile["selected_shape"], "online-warm-" + name)
            encode = options.get("encode_video", context.worker.encode_prompt)
            state = context.worker.prepare_encoded(warm_spec, encode(warm_spec))
            continuation = context.executors[name].attach(state)
            try:
                while state.step_index < 2:
                    context.executors[name].advance(continuation, 4, check_budget=context.check_budget)
            finally:
                _release(context.worker, context.executors[name], state, continuation)
            context.executors[name].backend.freeze_compilation()
        context.executor = context.executors[name]
        for index in range(2):
            batch_size = options["image_batch_size"]
            context.sana.generate_batch(context.opts["prompts"][:batch_size],
                seeds=[101 + index + n for n in range(batch_size)], check_budget=context.check_budget)

    # Admission uses complete AB/BA groups, including the frozen idle window.
    # Insufficient time produces a local result, never a partial system pass.
    expected = sum(max(r["arrival_ms"] + (r["service_ms"] if r["kind"] == "video" else 1000.)
                       for r in t["requests"]) / 1000. for t in traces)
    if not baseline_only and context.check_budget.remaining() < expected * 4 * 1.08 + 60:
        result.update(online_qualified=False, online_evaluation_complete=False,
            online_unverified_reason="remaining phase allowance cannot admit complete two-trace AB/BA comparison")
        record.update(status="not_run_budget_limited", qualification=dict(passed=False,
            reason=result["online_unverified_reason"]))
        save()
        return result
    save()
    try:
        if not context.metadata.get("wan_warmed"):
            capacity(context)
        for trace in traces:
            expected_trace = max(r["arrival_ms"] + (r["service_ms"] if r["kind"] == "video" else 1000.)
                                 for r in trace["requests"]) / 1000.
            context.check_budget.ensure_unit(expected_trace * (1 if baseline_only else 4) * 1.08 + 30)
            for order_index, order in enumerate((("baseline",),) if baseline_only else
                                                (("baseline", "candidate"), ("candidate", "baseline"))):
                for arm in order:
                    options = baseline if arm == "baseline" else candidate
                    prepare(options)
                    context.save_partial = journal.start_trace(dict(trace_sha256=trace["trace_sha256"],
                        order_index=order_index, arm=arm, seed=trace["seed"], load=trace["load"], pattern=trace["pattern"]))
                    with context.sana.steady_context():
                        measured = run_trace(context, trace["requests"], options)
                    record["rows"].append(dict(measured, seed=trace["seed"], load=trace["load"], pattern=trace["pattern"],
                        trace_sha256=trace["trace_sha256"], order_index=order_index, arm=arm))
                    save()
        record["status"] = "completed"
        if baseline_only:
            record["qualification"] = dict(passed=False, reason="instrumented baseline only", measured_baseline=True)
            result["online_qualified"] = False
            result["online_evaluation_complete"] = False
        else:
            record["qualification"] = paired_gate(record["rows"])
            result["online_qualified"] = record["qualification"].get("passed", False)
            result["online_evaluation_complete"] = bool(not record["qualification"].get("failures")
                                                        and record["qualification"].get("pooled"))
        save()
    except BaseException as error:
        record.update(status="incomplete", error=type(error).__name__ + ": " + str(error))
        save()
        result["online_qualified"] = False
        raise
    return result


def run(args, config, guard):
    from ..gpu.worker import WanStepWorker
    from ..directions.block_executor import WanBlockExecutor
    from .sana import OptimizedSanaAdapter
    from .sana_phases import SanaPhaseContext, run_compile, run_batch
    worker = sana = None
    context = None
    try:
        guard()
        worker = WanStepWorker(device=config["device"], model_path=config["models"]["wan"]["path"], attention="flash")
        sana = OptimizedSanaAdapter(config["models"]["sana"]["path"], device=config["device"])
        context = SimpleNamespace(config=config, output=args.output, run_root=args.run_root, check_budget=guard,
            worker=worker, sana=sana, executor=WanBlockExecutor(worker), opts=config["direction1"],
            profile=dict(selected_shape=dict(height=480, width=832, frames=81)), metadata={},
            replay_binding=getattr(args, "online_replay_binding", None))
        context.metadata["encoder_device"] = "cuda:1" if args.auxiliary_encoder else "cuda:0"
        context.metadata["phase"] = args.phase
        if args.phase in ("vae", "cache_k4"):
            from .stage_phases import install_encoder_defaults
            install_encoder_defaults(context, selected(args.run_root))
        context.save_partial = lambda value: atomic_json(args.output / "partial.json", value)
        from .online import load_frozen_traces
        frozen = load_frozen_traces(config["old_online"])
        deadlines = [row["deadline_ms"] - row["arrival_ms"] for row in frozen[0]["requests"] if row["kind"] == "image"]
        if max(deadlines) - min(deadlines) > 1e-6:
            raise ValueError("frozen image deadlines unexpectedly differ")
        context.metadata["image_deadline_ms"] = deadlines[0]
        previous = _terminal(args.run_root, "sana_compile")
        phase_context = SanaPhaseContext(adapter=sana, output=args.output, prompts=context.opts["prompts"], check_budget=guard,
            capacity_check=lambda: capacity(context), metadata=context.metadata,
            compile_report=previous if previous.get("online_qualified") else None)
        if args.phase == "sana_compile":
            result = run_compile(phase_context)
            if result.get("local_qualified"):
                _online(context, result, dict(sana_backend="compiled"))
            elif guard.remaining() > 1100:
                _online(context, result, baseline_only=True)
        elif args.phase == "sana_batch":
            result = run_batch(phase_context)
            if result.get("local_qualified"):
                _online(context, result, dict(image_batch_size=2, sana_backend=result["selected_backend"]))
        elif args.phase in ("wan_compile", "cache_k4"):
            from .wan_phases import run_wan, run_cache
            if args.phase == "wan_compile" and getattr(args, "online_replay_from", None) is not None:
                from .replay import load_local_evidence
                result = load_local_evidence(context, args.online_replay_from)
            else:
                result = (run_wan if args.phase == "wan_compile" else run_cache)(context)
            if args.phase == "wan_compile" and result.get("local_qualified"):
                _online(context, result, dict(wan_backend=result["selected_backend"]))
            elif args.phase == "cache_k4" and result.get("local_qualified"):
                from .cache_online import run_cache_online
                result = run_cache_online(context, result)
        else:
            from .stage_phases import run_text_encoder, run_vae
            result = (run_text_encoder if args.phase == "text_encoder" else run_vae)(context)
        result.setdefault("online_qualified", False)
        if result["online_qualified"]:
            result["decision"] = "值得继续"
        elif result.get("local_qualified"):
            if result.get("online_evaluation_complete"):
                result["decision"] = "当前配置无收益"
                result["reason"] = "local gain verified, but complete paired online comparison failed system thresholds"
            else:
                result["decision"] = "未验证"
                result.setdefault("reason", "local gain verified; complete online qualification absent")
        else:
            result.setdefault("decision", "未验证" if result.get("status") in ("failed", "incomplete") else "当前配置无收益")
        return result
    finally:
        if worker is not None:
            memory = dict(schema_version="ditserve.optimizations.memory.v1", devices=[],
                          scope="allocator counters before cleanup, since latest Sana/capacity reset; not a whole-run peak")
            try:
                for index in range(2 if args.auxiliary_encoder else 1):
                    device = "cuda:%d" % index
                    free, total = worker.torch.cuda.mem_get_info(device)
                    memory["devices"].append(dict(device=device,
                        allocated_bytes=worker.torch.cuda.memory_allocated(device),
                        reserved_bytes=worker.torch.cuda.memory_reserved(device),
                        peak_allocated_bytes=worker.torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=worker.torch.cuda.max_memory_reserved(device),
                        driver_free_bytes=free, device_total_bytes=total))
            except Exception as error:
                memory["error"] = type(error).__name__ + ": " + str(error)
            atomic_json(args.output / "memory.json", memory)
        if context is not None:
            if hasattr(context, "encoder_cleanup"):
                context.encoder_cleanup()
            executors = list(getattr(context, "executors", {}).values()) + [context.executor]
            if hasattr(context, "cache_executor"):
                executors.append(context.cache_executor)
            for executor in executors:
                for state in list(getattr(executor, "_states", {}).values()):
                    if not state.closed:
                        executor.close(state)
        if worker is not None:
            for state in list(worker._states.values()):
                worker.release(state)
        if sana is not None:
            sana.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("config", "output", "run-root", "ledger"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("phase", "session-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--auxiliary-encoder", action="store_true")
    parser.add_argument("--online-replay-from", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        config = read_json(args.config)
        active = verify_admission(args, config)
        guard = BudgetGuard(args.deadline, active["supervisor_pid"])
        versions = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers", "cache-dit", "numpy")}
        atomic_json(args.output / "environment.json", dict(versions=versions, session_id=args.session_id, visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]))
        result = run(args, config, guard)
        code = 0 if result.get("status") not in ("failed", "incomplete") else 1
    except BaseException as error:
        result = dict(status="incomplete" if isinstance(error, BudgetStop) else "failed", decision="未验证",
                      reason=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc(),
                      local_qualified=False, online_qualified=False)
        code = 2 if isinstance(error, BudgetStop) else 1
    result.update(phase=args.phase, child_wall_seconds=time.monotonic() - started)
    atomic_json(args.output / "worker.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
