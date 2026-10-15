"""Standard-library fixtures for conditional evidence, accounting and hooks."""

import copy
from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ditserve.research.common import atomic_json, read_json
from ditserve.optimizations import next_conditional as target
from ditserve.optimizations.online import attribution, request_identity
from ditserve.optimizations.budget import BudgetStop
import test_optimizations_next_report as report_fixtures


class Guard:
    def __call__(self):
        pass
    def ensure_unit(self, seconds):
        pass


class Array:
    shape = (1, 81, 480, 832, 3)
    def copy(self):
        return Array()


def context(directory):
    value = SimpleNamespace(output=Path(directory), run_root=Path(directory),
        config=dict(old_online="fake-old.json", gpu_uuid="GPU-A", v100_uuid="GPU-V"),
        metadata=dict(wan_warmed=True, cache_factorial_warmed=True, encoder_device="cpu"),
        opts=dict(prompts=["a", "b"], seeds=[101, 102], steps=32),
        profile=dict(selected_shape=dict(height=480, width=832, frames=81)),
        check_budget=Guard(), save_partial=None)
    value.worker = SimpleNamespace(_states={}, encode_prompt=Mock(return_value="cpu-payload"),
        prepare_encoded=Mock(return_value="prepared"),
        pipe=SimpleNamespace(vae=SimpleNamespace(dtype="float32", decoder=SimpleNamespace(forward=lambda x, **kw:x), clear_cache=Mock())),
        torch=SimpleNamespace(compiler=SimpleNamespace(set_stance=lambda stance:nullcontext())))
    value.executor = SimpleNamespace(_states={})
    value.executors = dict(cache_factorial_paired=SimpleNamespace(_states={}))
    value.sana = SimpleNamespace(set_candidate=Mock(), set_backend=Mock(),
        generate=Mock(return_value=(Array(), {})), steady_context=nullcontext)
    return value


def traces():
    result = []
    for seed in (101, 102):
        requests = [dict(request_id="r%03d" % i, kind="video" if i in (0,12) else "image",
            arrival_ms=float(i*10), deadline_ms=float(i*10+1000), service_ms=20., seed=seed+i,
            status="pending", start_ms=None, finish_ms=None) for i in range(24)]
        result.append(dict(trace_sha256=request_identity(requests), requests=requests,
            seed=seed, load=.5, pattern="ordinary" if seed == 101 else "burst"))
    return result


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = report_fixtures.NextReportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def install(self, *, failed=False, completed=True):
        fixture, rows = self.fixture, []
        for pair in fixture.workloads["diagnosis"]:
            for order in (0,1):
                for arm in ("image_only", "mixed"):
                    row = fixture.measured(pair[arm], order, arm)
                    row["events"] = []
                    if arm == "mixed" and order == 0:
                        row["events"] = [dict(kind="video_finish", start_ms=0., end_ms=900.),
                            dict(kind="scheduler_idle", reason="waiting_encoding", start_ms=0., end_ms=60.)]
                    row["attribution"] = attribution(row["requests"], row["events"], row["elapsed_ms"])
                    rows.append(row)
        return fixture.install("diagnosis", dict(status="completed" if completed else "running", rows=rows,
            evaluation_complete=completed), failed=failed)

    def test_complete_settled_trigger_retains_both_orders_and_immutable_sources(self):
        self.install()
        result = target.select_trigger_evidence(self.fixture.output, self.fixture.ledger)
        self.assertTrue(result["vae_trigger"], result)
        self.assertTrue(result["encoder_trigger"])
        source = result["admitted_sources"][0]
        self.assertEqual(len(source["rows"]), 8)
        self.assertEqual(sum(r["attribution"]["vae_trigger"] for r in source["rows"]), 2)
        self.assertEqual(source["session_id"], "diagnosis")
        paths = {r["path"] for r in result["immutable_bindings"]}
        self.assertNotIn(str(self.fixture.ledger), paths)
        self.assertIn(source["source"]["path"], paths)
        self.assertTrue(any(path.endswith("python/fixture.py") for path in paths))
        budget = read_json(self.fixture.ledger)
        budget["active"] = dict(stage="conditional", session_id="new")
        atomic_json(self.fixture.ledger, budget)
        self.assertEqual(target.revalidate_trigger_sources(result, self.fixture.output, self.fixture.ledger), result)
        atomic_json(source["source"]["path"], {"changed":True})
        with self.assertRaisesRegex(ValueError, "source changed"):
            target.revalidate_trigger_sources(result, self.fixture.output, self.fixture.ledger)

    def test_failed_supervisor_and_incomplete_matrix_cannot_trigger(self):
        self.install(failed=True)
        result = target.select_trigger_evidence(self.fixture.output, self.fixture.ledger)
        self.assertFalse(result["triggered"])
        self.assertFalse(result["admitted_sources"])

    def test_recorded_trigger_without_matching_raw_intervals_is_rejected(self):
        attempt, _ = self.install()
        path = attempt / "diagnosis.json"
        data = read_json(path)
        data["rows"][1]["attribution"]["tail_finish_overlap_fraction"] = .999
        atomic_json(path, data)
        result = target.select_trigger_evidence(self.fixture.output, self.fixture.ledger)
        self.assertFalse(result["triggered"])
        self.assertTrue(any("recomputed" in r["reason"] for r in result["rejected"]))


class DecoderTests(unittest.TestCase):
    def build(self, *, silent=False, fail=False):
        def original(value, feat_cache=None, feat_idx=None, first_chunk=False):
            if feat_cache is not None:
                feat_cache[0] = value
            feat_idx[0] += 1
            return value + 2
        decoder = SimpleNamespace(forward=original)
        calls = []
        def inductor(graph, args, **kwargs):
            calls.append(kwargs)
            if fail:
                raise RuntimeError("compiler failed")
            def executable(*args):
                return graph(*args)
            executable._torchdynamo_orig_callable = original
            return executable
        def compile(fn, *, backend, fullgraph, dynamic):
            self.assertTrue(fullgraph)
            self.assertFalse(dynamic)
            if silent:
                return fn
            implementation = None
            def run(*args):
                nonlocal implementation
                if implementation is None:
                    implementation = backend(fn, args)
                    self.assertFalse(hasattr(implementation, "_torchdynamo_orig_callable"))
                return implementation(*args)
            return run
        worker = SimpleNamespace(pipe=SimpleNamespace(vae=SimpleNamespace(decoder=decoder, clear_cache=Mock())),
            torch=SimpleNamespace(compile=compile))
        return target.CountedFunctionalDecoder(worker, backend_resolver=lambda name:inductor), decoder, original, calls

    def test_real_callable_count_preserves_mutable_cache_ownership(self):
        value, decoder, original, calls = self.build()
        value.select(True)
        cache, index = [None], [0]
        self.assertEqual(decoder.forward(3, feat_cache=cache, feat_idx=index), 5)
        self.assertEqual((cache, index), ([3], [1]))
        decoder.forward(4, feat_cache=cache, feat_idx=index)
        self.assertEqual(value.invocations, 2)
        self.assertEqual(value.forward_calls, 2)
        self.assertEqual(calls, [dict(config_patches={"triton.cudagraphs":False})])
        value.close()
        self.assertIs(decoder.forward, original)

    def test_silent_eager_and_compile_failure_are_not_fallbacks(self):
        for args, message in ((dict(silent=True), "no fallback"), (dict(fail=True), "compiler failed")):
            with self.subTest(args=args):
                value, decoder, original, _ = self.build(**args)
                value.select(True)
                with self.assertRaisesRegex(RuntimeError, message):
                    decoder.forward(3)
                value.close()
                self.assertIs(decoder.forward, original)


class PairedLatentTests(unittest.TestCase):
    def test_complete_paired_trajectory_is_cloned_and_private_state_released(self):
        for mode in ("reuse", "full_only", "invalid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                ctx = context(directory)
                state = SimpleNamespace(step_index=0, residency="gpu", events=[],
                    latents=SimpleNamespace(detach=lambda:SimpleNamespace(clone=lambda:"private clone")))
                continuation = SimpleNamespace(closed=False)
                def advance(current, quantum, check_budget):
                    self.assertEqual(quantum, 4)
                    check_budget()
                    index = state.step_index
                    reused = mode == "reuse" and index % 2 == 1
                    state.events.append(dict(kind="step", step_index=index, completed_steps=index+1,
                        path="reuse" if reused else "compute", reused_forwards=1 if mode == "invalid" else 2 if reused else 0))
                    state.step_index += 1
                executor = SimpleNamespace(attach=Mock(return_value=continuation), advance=advance,
                    close=lambda current:setattr(current, "closed", True))
                ctx.executors["cache_factorial_paired"] = executor
                ctx.worker.prepare_encoded = Mock(return_value=state)
                ctx.worker.release = lambda current:setattr(current, "residency", "released")
                evidence = SimpleNamespace(data=dict(active_unit={}), save=Mock())
                request = dict(request_id="fixed", seed=101, steps=32)
                if mode == "invalid":
                    with self.assertRaisesRegex(RuntimeError, "cache trajectory"):
                        target._paired_latent(ctx, evidence, request)
                else:
                    self.assertEqual(target._paired_latent(ctx, evidence, request), "private clone")
                    self.assertEqual(evidence.data["paired_latent_sources"][0]["completed_steps"], 32)
                    if mode == "full_only":
                        self.assertFalse(evidence.data["paired_latent_sources"][0]["cache_paths"]["full_reuse_observed"])
                self.assertTrue(continuation.closed)
                self.assertEqual(state.residency, "released")
                self.assertTrue(ctx.worker.prepare_encoded.call_args.args[0]["cache"])


class OnlineTests(unittest.TestCase):
    def run_case(self, directory, options=None, fail_at=None):
        ctx, frozen, calls = context(directory), traces(), []
        if options and options.get("encode_device") == "cuda:1":
            ctx.admitted_devices = ["GPU-A", "GPU-V"]
            ctx.metadata["encoder_device"] = "cuda:1"
        ctx.prepare_online_arm = Mock()
        old = ctx.executor
        def run(active, requests, selected):
            calls.append(selected)
            selected["encode_video"]({"cache":False})
            selected["prepare_video"]({"cache":False}, "payload")
            if len(calls) == fail_at:
                raise BudgetStop("cancelled replay")
            rows = copy.deepcopy(requests)
            candidate = selected.get("vae_backend") == "compiled" or selected.get("encode_device") == "cuda:1"
            for row in rows:
                row.update(status="completed", arrived=True, start_ms=row["arrival_ms"]+1.,
                    finish_ms=row["arrival_ms"]+(11. if candidate else 21.))
            return dict(status="completed", complete=True, requests=rows, events=[],
                trace_sha256=request_identity(requests), elapsed_ms=800. if candidate else 1000.)
        result = dict(local_qualified=True)
        with patch.object(target, "load_frozen_traces", return_value=frozen), \
             patch.object(target, "FROZEN_TRACE_IDS", tuple(r["trace_sha256"] for r in frozen)), \
             patch.object(target, "run_trace", side_effect=run):
            try:
                target.online_callback(ctx, result, options or dict(vae_backend="compiled", finish_video=lambda s,e:None),
                    filename="probe-online.json", candidate_label="fixture")
            except BudgetStop:
                if fail_at is None:
                    raise
        self.assertIs(ctx.executor, old)
        self.assertIsNone(ctx.save_partial)
        return ctx, result, calls, read_json(Path(directory) / "probe-online.json")

    def test_fixed_abba_preserves_cache_hooks_and_full_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx, result, calls, data = self.run_case(directory)
            self.assertTrue(result["online_qualified"])
            self.assertEqual(len(data["rows"]), 8)
            self.assertEqual(sum(len(r["requests"]) for r in data["rows"]), 192)
            self.assertEqual([r["arm"] for r in data["rows"]], ["baseline","candidate","candidate","baseline"]*2)
            self.assertTrue(all(call.args[0]["cache"] is True for call in ctx.worker.encode_prompt.call_args_list))
            self.assertTrue(all(call.args[0]["threshold"] == .08 for call in ctx.worker.prepare_encoded.call_args_list))
            self.assertTrue(all(r["policy"] == "block-4" and r["sana_backend"] == "eager" for r in calls))
            self.assertEqual(ctx.prepare_online_arm.call_count, 8)

    def test_explicit_v100_can_encode_independently_but_a100_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            gpu_encode = Mock(return_value="gpu-payload")
            ctx, result, calls, _ = self.run_case(directory, dict(encode_device="cuda:1", encode_on_scheduler=False,
                                                                  encode_video=gpu_encode))
            self.assertTrue(result["online_qualified"])
            self.assertEqual(gpu_encode.call_count, 4)
            self.assertEqual(ctx.worker.encode_prompt.call_count, 4)
        with tempfile.TemporaryDirectory() as directory:
            ctx = context(directory)
            for options in (dict(encode_device="cuda:0", encode_on_scheduler=False),
                            dict(encode_device="cuda:1", encode_on_scheduler=False)):
                with self.assertRaises(ValueError):
                    target.online_callback(ctx, dict(local_qualified=True), options,
                        filename="x.json", candidate_label="invalid")

    def test_cancellation_keeps_complete_rows_and_active_denominator_unqualified(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, _, data = self.run_case(directory, fail_at=3)
            self.assertFalse(result["online_qualified"])
            self.assertFalse(result["online_evaluation_complete"])
            self.assertEqual(len(data["rows"]), 2)
            self.assertIsNotNone(data["active_unit"])
            self.assertEqual(data["status"], "incomplete")


class LocalTests(unittest.TestCase):
    def run_case(self, directory, quality_failure=False, cancel=False, slow=False, cleanup_failure=False):
        ctx = context(directory)
        original_class = target.stages.FunctionalDecoder
        backend = SimpleNamespace(records=[dict(status="completed", graph_invocations=0)], invocations=0, selected=False)
        backend.select = lambda compiled:setattr(backend, "selected", compiled)
        backend.close = Mock(side_effect=RuntimeError("cleanup broke") if cleanup_failure else None)
        backend.summary = lambda:dict(graph_invocations=backend.invocations, graph_compile_count=1,
            compilation=backend.records, compiled_forward_calls=backend.invocations)
        def decode(active, latent, compiled=False):
            if backend.selected:
                backend.invocations += 21
                backend.records[0]["graph_invocations"] += 21
            return Array(), dict(ms=(110. if slow else 90.) if backend.selected else 100.)
        def quality(active, reference, image):
            if cancel:
                raise BudgetStop("quality cancellation")
            return dict(passed=not (quality_failure and backend.selected), metrics=dict(lpips=.1 if quality_failure else 0.))
        online = Mock(side_effect=lambda active, result, options, **kw:result.update(
            online_qualified=True, system_qualified=True, online_evaluation_complete=True))
        with ExitStack() as stack:
            stack.enter_context(patch.object(target, "select_trigger_evidence", return_value=dict(vae_trigger=True)))
            stack.enter_context(patch.object(target, "CountedFunctionalDecoder", return_value=backend))
            stack.enter_context(patch.object(target, "_paired_latent", return_value=object()))
            stack.enter_context(patch.object(target.stages, "decode", side_effect=decode))
            stack.enter_context(patch.object(target.stages, "quality", side_effect=quality))
            stack.enter_context(patch.object(target.stages, "retain", return_value=dict(path="fixture.npy", raw_available=True)))
            stack.enter_context(patch.object(target, "online_callback", online))
            if cleanup_failure:
                with self.assertRaisesRegex(RuntimeError, "do not continue"):
                    target.run_video_vae(ctx)
                result = read_json(Path(directory) / "conditional-video-vae.json")
            elif cancel:
                with self.assertRaises(BudgetStop):
                    target.run_video_vae(ctx)
                result = read_json(Path(directory) / "conditional-video-vae.json")
            else:
                result = target.run_video_vae(ctx)
        self.assertIs(target.stages.FunctionalDecoder, original_class)
        self.assertGreaterEqual(backend.close.call_count, 1)
        return result, online, read_json(Path(directory) / "vae.json")

    def test_full_local_matrix_reuses_original_code_without_global_patching(self):
        with tempfile.TemporaryDirectory() as directory:
            result, online, native = self.run_case(directory)
            self.assertTrue(result["local_qualified"], result)
            self.assertTrue(result["online_qualified"])
            self.assertEqual(result["decision"], "值得继续")
            self.assertEqual(len([r for r in native["rows"] if r["kind"] == "measurement"]), 16)
            self.assertTrue(result["decoder_cleanup_passed"])
            self.assertEqual(online.call_count, 1)
            self.assertGreater(result["compiler"]["graph_invocations"], 0)

    def test_quality_and_two_order_speed_failure_do_not_call_online(self):
        for options in (dict(quality_failure=True), dict(slow=True)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                result, online, native = self.run_case(directory, **options)
                self.assertFalse(result["local_qualified"])
                self.assertFalse(result["online_qualified"])
                self.assertEqual(result["decision"], "当前配置无收益")
                self.assertFalse(online.called)
                self.assertTrue(any(r.get("artifact") for r in native["rows"]))

    def test_cancellation_retains_completed_decode_artifact_and_restores_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            result, online, native = self.run_case(directory, cancel=True)
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["decoder_cleanup_passed"])
            self.assertFalse(online.called)
            row = next(r for r in native["rows"] if r["kind"] == "measurement")
            self.assertTrue(row["generation_completed"])
            self.assertEqual(row["quality_status"], "interrupted")
            self.assertTrue(row["artifact"]["raw_available"])

    def test_cleanup_failure_is_terminal_even_after_local_and_online_success(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self.run_case(directory, cleanup_failure=True)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["decoder_cleanup_passed"])
            self.assertFalse(result["online_qualified"])
            self.assertEqual(result["decision"], "未验证")


if __name__ == "__main__":
    unittest.main()
