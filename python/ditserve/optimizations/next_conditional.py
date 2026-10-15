"""Conditionally test the existing Wan decoder with the current paired/K4 baseline.

This module never admits hardware or grants a combined serving default. The
caller owns the charged process. Imports do not import torch or load a model.
"""

import copy
from pathlib import Path
import time
from types import FunctionType

from ..research.common import atomic_json, file_digest, read_json
from .budget import BudgetStop
from . import stage_phases as stages
from .cache_online import cache_spec, warm_cache_factorial, _warm_cache_paths
from .online import (FROZEN_TRACE_IDS, TraceJournal, attribution, load_frozen_traces,
                     paired_gate, request_identity, run_trace)
from .next_protocol import DEFAULTS


TRIGGER_PHASES = ("diagnosis", "cache_load_05", "cache_load_07", "cache_load_09")


def select_trigger_evidence(run_root, ledger):
    """Select observed triggers only from complete, settled, source-verified phases.

    A current later allocation may be active: only the *source phase* must be
    settled. An archived report without original events cannot retrigger work.
    Neither a partial arm nor an unverified worker success can admit a probe.
    """
    from .next_report import collect, _Reader
    report = collect(run_root, ledger)
    result = dict(schema_version="ditserve.optimizations.conditional_triggers.v1",
        triggered=False, vae_trigger=False, encoder_trigger=False, selected=None,
        admitted_sources=[], rejected=[], report_errors=report.get("errors", []),
        thresholds=dict(tail_finish_overlap_fraction=.20, encoding_idle_fraction=.05,
                        encoding_video_e2e_fraction=.20),
        scope="existing host-interval triggers; observed video_finish overlap is not proof that VAE kernels caused all waiting")
    if report.get("archive_only") or report.get("errors"):
        result["rejected"].append(dict(reason="original complete verified source evidence unavailable"))
        return result
    reader = _Reader()
    for phase in report.get("phases", []):
        name = phase["phase"]
        if name not in TRIGGER_PHASES:
            continue
        if not (phase.get("authoritative") is True and phase.get("status") == "completed"
                and phase.get("evaluation_complete") is True
                and phase.get("online", {}).get("matrix_complete") is True):
            result["rejected"].append(dict(phase=name, reason="phase is not a complete settled isolated authorized matrix"))
            continue
        errors = []
        binding = phase.get("evidence") or {}
        data = reader.bound(binding, errors, name)
        observed = []
        try:
            for index, row in enumerate(data.get("rows", [])):
                measured = attribution(row["requests"], row["events"], row["elapsed_ms"])
                if row.get("attribution") != measured:
                    raise ValueError("recorded attribution differs from raw interval recomputation")
                observed.append(dict(pointer="/rows/%d" % index, trace_sha256=row["trace_sha256"],
                    arm=row["arm"], order_index=row["order_index"], seed=row.get("seed"),
                    pattern=row.get("pattern"), attribution=measured,
                    trigger_arm=name != "diagnosis" or row.get("arm") == "mixed"))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
        if errors or not observed:
            result["rejected"].append(dict(phase=name, reason="trigger evidence could not be independently recomputed", errors=errors))
            continue
        entry = dict(phase=name, source=binding, terminal=phase.get("terminal_source"),
            settled_session=phase.get("charge"), session_id=phase.get("charge", {}).get("session_id"), rows=observed,
            vae_trigger=any(r["trigger_arm"] and r["attribution"]["vae_trigger"] for r in observed),
            encoder_trigger=any(r["trigger_arm"] and r["attribution"]["encoding_trigger"] for r in observed),
            scope="all paired orders retained; a trigger in any admitted mixed arm permits a probe, not a stable bottleneck claim")
        result["admitted_sources"].append(entry)
        if result["selected"] is None and (entry["vae_trigger"] or entry["encoder_trigger"]):
            result["selected"] = dict(phase=name, source=binding)
    result["vae_trigger"] = any(r["vae_trigger"] for r in result["admitted_sources"])
    result["encoder_trigger"] = any(r["encoder_trigger"] for r in result["admitted_sources"])
    result["triggered"] = result["vae_trigger"] or result["encoder_trigger"]
    result["source_bindings"] = report.get("source_bindings", [])
    immutable = {entry["path"]: dict(path=entry["path"], sha256=entry["sha256"])
                 for entry in result["source_bindings"] if Path(entry["path"]).absolute() != Path(ledger).absolute()}
    snapshot_root = Path(run_root).absolute() / "sources"
    for entry in list(immutable.values()):
        path = Path(entry["path"])
        if path.name == "manifest.json" and snapshot_root in path.parents:
            manifest = read_json(path)
            for name, sha in manifest.get("source_files", {}).items():
                file = path.parent / name
                immutable[str(file)] = dict(path=str(file), sha256=sha)
    result["immutable_bindings"] = sorted(immutable.values(), key=lambda row:row["path"])
    return result


def revalidate_trigger_sources(trigger, run_root, ledger):
    """Recheck the admitted immutable evidence while the new lease is active."""
    immutable = trigger.get("immutable_bindings", [])
    if not immutable or not trigger.get("admitted_sources"):
        raise ValueError("conditional trigger lacks admitted immutable source bindings")
    for binding in immutable:
        path = Path(binding["path"]).absolute()
        if path == Path(ledger).absolute() or path.resolve() != path or file_digest(path) != binding["sha256"]:
            raise ValueError("conditional trigger immutable source changed")
    budget = read_json(ledger)
    for source in trigger["admitted_sources"]:
        session = source["settled_session"]
        found = [row for row in budget.get("sessions", []) if row.get("session_id") == source["session_id"]]
        if found != [session] or session.get("status") != "completed":
            raise ValueError("trigger source allocation is not the identical settled session")
        for key in ("source", "terminal"):
            binding = source[key]
            if not any(row["path"] == binding["path"] and row["sha256"] == binding["sha256"] for row in immutable):
                raise ValueError("trigger terminal/native binding is not admitted")
        terminal_path = Path(source["terminal"]["path"]).absolute()
        if terminal_path != Path(run_root).absolute() / source["phase"] / "terminal.json":
            raise ValueError("trigger source leaves the admitted run")
    return copy.deepcopy(trigger)


class CountedFunctionalDecoder(stages.FunctionalDecoder):
    """Same explicit Wan decoder math, with actual compiled callable accounting."""
    def __init__(self, worker, backend_resolver=None):
        super().__init__(worker)
        self.backend_resolver = backend_resolver
        self.forward_calls = 0

    @property
    def invocations(self):
        return sum(row.get("graph_invocations", 0) for row in self.records)

    def compile(self):
        if self.compiled is not None:
            return
        if self.backend_resolver is None:
            from torch._dynamo.backends.registry import lookup_backend
            inductor = lookup_backend("inductor")
        else:
            inductor = self.backend_resolver("inductor")
        def backend(graph, args):
            record = dict(status="running", index=len(self.records), graph_invocations=0)
            self.records.append(record)
            started = time.perf_counter()
            try:
                executable = inductor(graph, args, config_patches={"triton.cudagraphs": False})
                record["status"] = "completed"
                # Do not use functools.wraps: Dynamo may follow its copied
                # _torchdynamo_orig_callable and bypass invocation accounting.
                def counted(*values, **kwargs):
                    record["graph_invocations"] += 1
                    return executable(*values, **kwargs)
                return counted
            except BaseException as error:
                record.update(status="failed", error=type(error).__name__ + ": " + str(error))
                raise
            finally:
                record["ms"] = (time.perf_counter()-started)*1000
        self.compiled = self.worker.torch.compile(self.pure, backend=backend, fullgraph=True, dynamic=False)

    def forward(self, value, feat_cache=None, feat_idx=None, first_chunk=False):
        before = self.invocations
        output = super().forward(value, feat_cache, feat_idx, first_chunk)
        if self.invocations <= before:
            raise RuntimeError("compiled Wan decoder did not execute the counted graph; no fallback accepted")
        self.forward_calls += 1
        return output

    def summary(self):
        return dict(compilation=self.records, graph_compile_count=len(self.records),
            graph_invocations=self.invocations, compiled_forward_calls=self.forward_calls,
            callable_counting=True, fullgraph=True, dynamic=False, cudagraphs=False)


def _empty(context):
    if context.worker._states or getattr(context.executor, "_states", {}):
        raise RuntimeError("conditional replay has live request/continuation state")


def _paired_latent(context, evidence, request):
    """Produce one final latent with the same paired arithmetic and K4 boundaries."""
    from ..directions.direction1 import _release
    worker, executor = context.worker, context.executors["cache_factorial_paired"]
    request = cache_spec(request)
    state = continuation = None
    try:
        state = worker.prepare_encoded(request, worker.encode_prompt(request))
        continuation = executor.attach(state)
        while state.step_index < request["steps"]:
            executor.advance(continuation, 4, check_budget=context.check_budget)
            evidence.data["active_unit"]["completed_steps"] = state.step_index
        paths = _warm_cache_paths(state.events)
        if state.step_index != 32 or request["steps"] != 32 or not paths["valid"]:
            raise RuntimeError("local paired latent lacks a valid full 32-step cache trajectory")
        evidence.data.setdefault("paired_latent_sources", []).append(dict(request_id=request["request_id"],
            seed=request["seed"], completed_steps=32, quantum=4, policy="paired_008", cache_paths=paths))
        evidence.save()
        return state.latents.detach().clone()
    finally:
        if state is not None:
            _release(worker, executor, state, continuation)


def online_callback(context, result, candidate_options, *, filename, candidate_label):
    """Generic conditional AB/BA: one component changes against paired008+K4.

    The caller installs its prepare_online_arm and decoder/encoder hooks. This
    function preserves the real frozen arrivals, CPU/GPU encoding placement,
    deadline denominators, warmup and observer implementation of run_trace.
    """
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("online evidence filename must be a simple JSON leaf")
    if result.get("local_qualified") is not True:
        raise ValueError("conditional online requires complete local qualification")
    allowed = {"vae_backend", "finish_video", "encode_video", "encode_device", "encode_on_scheduler"}
    if set(candidate_options) - allowed:
        raise ValueError("conditional candidate changes more than its admitted encoder/decoder component")
    encode_device = candidate_options.get("encode_device", "cpu")
    if encode_device == "cuda:0" and candidate_options.get("encode_on_scheduler") is not True:
        raise ValueError("A100 encoding must use the shared scheduler")
    if encode_device == "cuda:1":
        expected = [context.config.get("gpu_uuid"), context.config.get("v100_uuid")]
        if (None in expected or getattr(context, "admitted_devices", None) != expected
                or context.metadata.get("encoder_device") != "cuda:1"
                or candidate_options.get("encode_on_scheduler") is not False):
            raise ValueError("independent V100 encoding requires the explicit two-card admission")
    elif encode_device not in ("cpu", "cuda:0"):
        raise ValueError("unadmitted encoder device")
    traces = load_frozen_traces(context.config["old_online"])
    if tuple(t["trace_sha256"] for t in traces) != FROZEN_TRACE_IDS or any(len(t["requests"]) != 24 for t in traces):
        raise ValueError("conditional online requires the exact original two 24-request traces")
    context.check_budget()
    if not context.metadata.get("wan_warmed") or not context.metadata.get("cache_factorial_warmed"):
        raise ValueError("conditional online requires fresh full capacity and paired warmup")
    path = Path(context.output) / filename
    if path.exists():
        raise FileExistsError(str(path))
    record = dict(schema_version="ditserve.optimizations.conditional_online.v1", status="running", rows=[],
        candidate_label=candidate_label, planned_replays=8, planned_requests=192,
        active_unit=None, wan="eager paired_008 + K4", sana="eager", image_batch_size=1,
        baseline="CPU FP32 encoder and eager VAE", comparison=None,
        scope="one component only; original frozen clocks/deadlines; complete source-bound AB/BA; no combined default")
    result.update(online_path=str(path), online_qualified=False, system_qualified=False,
                  online_evaluation_complete=False)
    atomic_json(path, record)
    original_executor, original_partial = context.executor, getattr(context, "save_partial", None)
    try:
        context.executor = context.executors["cache_factorial_paired"]
        for trace in traces:
            for order, arms in enumerate((("baseline", "candidate"), ("candidate", "baseline"))):
                for arm in arms:
                    context.check_budget()
                    _empty(context)
                    if hasattr(context.sana, "set_candidate"):
                        context.sana.set_candidate("eager")
                    context.sana.set_backend("eager")
                    options = dict(DEFAULTS)
                    if arm == "candidate":
                        options.update(candidate_options)
                    prepare = getattr(context, "prepare_online_arm", None)
                    if prepare is not None:
                        prepare(options)
                    encode = options.get("encode_video", context.worker.encode_prompt)
                    prepare_video = context.worker.prepare_encoded
                    options.update(encode_video=lambda spec, fn=encode: fn(cache_spec(spec)),
                        prepare_video=lambda spec, payload, fn=prepare_video: fn(cache_spec(spec), payload))
                    for index in range(2):
                        context.sana.generate(context.opts["prompts"][index], seed=101+index,
                                              check_budget=context.check_budget)
                    identity = dict(trace_sha256=trace["trace_sha256"], seed=trace["seed"], load=trace["load"],
                        pattern=trace["pattern"], order_index=order, arm=arm)
                    directory = Path(context.output) / (Path(filename).stem + "-partials") / ("replay-%02d" % len(record["rows"]))
                    directory.mkdir(parents=True, exist_ok=False)
                    journal = TraceJournal(directory / "trace.json", atomic_json)
                    journal.save_record(dict(status="preparing", **identity))
                    context.save_partial = journal.start_trace(identity)
                    record.update(active_unit=dict(identity, partial_path=str(journal.partial_path)),
                                  active_trace_path=str(journal.partial_path))
                    atomic_json(path, record)
                    with context.sana.steady_context():
                        measured = run_trace(context, trace["requests"], options)
                    if measured.get("trace_sha256") != trace["trace_sha256"] or request_identity(measured.get("requests", [])) != trace["trace_sha256"]:
                        raise ValueError("conditional replay changed frozen requests")
                    record["rows"].append(dict(measured, **{key:value for key,value in identity.items() if key != "trace_sha256"}))
                    record["active_unit"] = None
                    atomic_json(path, record)
                    _empty(context)
                    if measured.get("status") != "completed" or measured.get("complete") is not True:
                        raise RuntimeError("conditional replay did not complete its full denominator")
        gate = paired_gate(record["rows"], expected_trace_ids=FROZEN_TRACE_IDS)
        complete = len(record["rows"]) == 8 and not gate["failures"] and gate["pooled"] is not None
        record.update(status="completed", comparison=gate, evaluation_complete=complete)
        result.update(online_evaluation_complete=complete, online_qualified=bool(complete and gate["passed"]),
            system_qualified=bool(complete and gate["passed"]), online_comparison=gate,
            decision="worth_continuing" if complete and gate["passed"] else "no_online_gain_in_fixed_matrix",
            reason="single conditional component fixed-matrix result; no combined default selected")
        atomic_json(path, record)
    except BaseException as error:
        record.update(status="incomplete", reason=type(error).__name__ + ": " + str(error))
        result.update(online_qualified=False, system_qualified=False, online_evaluation_complete=False,
            online_unverified_reason=record["reason"])
        atomic_json(path, record)
        raise
    finally:
        context.executor, context.save_partial = original_executor, original_partial
    return result


def run_video_vae(context):
    """Run the unchanged 16-decode local matrix and conditional fixed online pair."""
    path = Path(context.output) / "conditional-video-vae.json"
    if path.exists():
        raise FileExistsError(str(path))
    ledger = getattr(context, "ledger", Path(context.run_root).parent / "gpu-budget.json")
    admitted = getattr(context, "trigger_evidence", None)
    trigger = (revalidate_trigger_sources(admitted, context.run_root, ledger) if admitted is not None
               else select_trigger_evidence(context.run_root, ledger))
    record = dict(schema_version="ditserve.optimizations.conditional_video_vae.v1", status="running",
        decision="未验证", local_qualified=False, online_qualified=False, system_qualified=False,
        trigger_evidence=trigger, candidate="counted_functional_decoder_inductor",
        defaults=dict(DEFAULTS, wan_cache="paired_008", quantum=4),
        dependencies_closed=False, local_evidence_path=str(Path(context.output) / "vae.json"),
        scope="same cached final latent per input; original dtype, decoder temporal loop and postprocessing; no combined default")
    atomic_json(path, record)
    if not trigger["vae_trigger"]:
        record.update(status="stopped", reason="no complete authorized video_finish overlap trigger",
            decision="简单策略已足够" if trigger.get("admitted_sources") else "未验证")
        atomic_json(path, record)
        return record
    proxy = copy.copy(context)
    proxy.metadata = dict(context.metadata, phase="vae", conditional_component="video_vae")
    backend = CountedFunctionalDecoder(proxy.worker)
    try:
        proxy.check_budget()
        if hasattr(proxy.sana, "set_candidate"):
            proxy.sana.set_candidate("eager")
        proxy.sana.set_backend("eager")
        # Capacity/warmup stays charged and outside online arrival clocks. A
        # prior successful fresh-process full capacity proof can be reused.
        if not proxy.metadata.get("wan_warmed"):
            from .worker import capacity
            record["capacity"] = capacity(proxy)
            atomic_json(Path(proxy.output) / "capacity.json", record["capacity"])
        if not proxy.metadata.get("cache_factorial_warmed"):
            warm_cache_factorial(proxy)
        def callback(active_context, local, options):
            if backend.invocations <= 0 or not backend.records or any(r["status"] != "completed" for r in backend.records):
                raise RuntimeError("local decoder lacks successful actual compiled invocation evidence")
            return online_callback(active_context, local, options,
                filename="video-vae-online.json", candidate_label="compiled_video_vae")
        proxy.run_online = callback
        # Execute the original function code with private dependency bindings;
        # do not monkeypatch the shared module or change decoder math/thresholds.
        bindings = dict(stages.run_vae.__globals__, FunctionalDecoder=lambda worker: backend,
                        final_latent=_paired_latent)
        run_local = FunctionType(stages.run_vae.__code__, bindings, stages.run_vae.__name__, stages.run_vae.__defaults__)
        local = run_local(proxy)
        record.update(status=local["status"], local_qualified=local.get("local_qualified") is True,
            online_qualified=local.get("online_qualified") is True, system_qualified=local.get("system_qualified") is True,
            online_evaluation_complete=local.get("online_evaluation_complete") is True,
            reason=local.get("reason", local.get("error")), online_path=local.get("online_path"),
            online_comparison=local.get("online_comparison"),
            local_evidence=dict(path=str(Path(proxy.output) / "vae.json"), sha256=file_digest(Path(proxy.output) / "vae.json")),
            local_summary={key:local.get(key) for key in ("measurement_matrix", "complete_inputs", "quality_passed", "order_gains", "gain", "summaries", "error", "budget_stopped")},
            compiler=backend.summary(), original_local_function=dict(path=str(Path(stages.__file__).resolve()),
                sha256=file_digest(Path(stages.__file__)), function="run_vae",
                private_bindings=["FunctionalDecoder counting subclass", "final_latent paired008/K4"]))
        record["decision"] = ("值得继续" if record["online_qualified"] else "当前配置无收益"
            if local.get("reason") in ("vae_quality_failed", "no_local_vae_gain") or
               local.get("online_evaluation_complete") is True else "未验证")
        return record
    except BaseException as error:
        record.update(status="incomplete" if isinstance(error, BudgetStop) else "failed",
            reason=type(error).__name__ + ": " + str(error), online_qualified=False, system_qualified=False,
            compiler=backend.summary())
        local_path = Path(proxy.output) / "vae.json"
        if local_path.exists():
            record["local_evidence"] = dict(path=str(local_path), sha256=file_digest(local_path))
        if isinstance(error, (BudgetStop, KeyboardInterrupt, SystemExit)):
            raise
        return record
    finally:
        # run_vae also closes its backend; repeat restoration is intentionally
        # safe when setup fails before that function takes ownership.
        try:
            backend.close()
            record["decoder_cleanup_passed"] = True
        except BaseException as error:
            record.update(status="failed", decoder_cleanup_passed=False,
                cleanup_error=type(error).__name__ + ": " + str(error),
                local_qualified=False, online_qualified=False, system_qualified=False, decision="未验证")
        context.metadata.update({key:value for key,value in proxy.metadata.items() if key.startswith("wan_warm") or key.startswith("cache_factorial")})
        context.executors = getattr(proxy, "executors", getattr(context, "executors", {}))
        atomic_json(path, record)
        if not record["decoder_cleanup_passed"]:
            raise RuntimeError("conditional decoder cleanup failed; do not continue another component")
