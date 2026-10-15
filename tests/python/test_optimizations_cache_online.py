import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.optimizations.cache_online import cache_spec, StepPairedExecutor, run_cache_online, warm_cache_factorial


class CacheOnlineTests(unittest.TestCase):
    def test_cache_spec_only_changes_cache_policy(self):
        source = dict(request_id="r000", seed=101, prompt="hello", cache=False, threshold=0., max_continuous=1)
        actual = cache_spec(source)
        self.assertTrue(actual["cache"])
        self.assertEqual(actual["threshold"], .08)
        self.assertEqual({k: actual[k] for k in ("request_id", "seed", "prompt")},
                         {k: source[k] for k in ("request_id", "seed", "prompt")})
        self.assertFalse(source["cache"])

    def test_step_drains_at_most_one_actual_step(self):
        executor = object.__new__(StepPairedExecutor)
        calls = []
        executor.advance = lambda state, blocks, **kw: calls.append((state, blocks, kw))
        check = lambda: None
        executor.native_step("state", check_budget=check)
        self.assertEqual(calls, [("state", 60, {"check_budget": check})])

    def test_screen_does_not_admit_online(self):
        result = dict(local_qualified=True, scientific_qualified=True, cache_quality_qualified=False)
        measured = run_cache_online(SimpleNamespace(), result)
        self.assertFalse(measured["online_qualified"])

    def test_whole_factorial_admission_precedes_any_gpu_work(self):
        context = SimpleNamespace(config={"old_online": "ignored"}, check_budget=SimpleNamespace(remaining=lambda: 10))
        traces = [dict(requests=[dict(arrival_ms=500000, service_ms=300, kind="image")])]
        result = dict(local_qualified=True, cache_quality_qualified=True, full_qualification_completed=True)
        with patch("ditserve.optimizations.cache_online.load_frozen_traces", return_value=traces):
            measured = run_cache_online(context, result)
        self.assertFalse(measured["online_qualified"])
        self.assertIn("four-arm", measured["online_unverified_reason"])

    def test_factorial_partials_never_rewrite_completed_history(self):
        from ditserve.optimizations import cache_online
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = dict(seed=102, load=.5, pattern="burst", trace_sha256="trace",
                         requests=[dict(arrival_ms=0., service_ms=100., kind="image")])
            ctx = SimpleNamespace(config=dict(old_online="unused"), output=root, run_root=root,
                worker=SimpleNamespace(encode_prompt=lambda spec: spec, prepare_encoded=lambda spec, payload: None),
                check_budget=SimpleNamespace(remaining=lambda: 100000.), opts=dict(prompts=["a", "b"]),
                sana=SimpleNamespace(set_backend=Mock(), generate_batch=Mock(return_value=([None], dict(ms=1., finite=True))),
                                     steady_context=contextlib.nullcontext))
            seen = []
            def measure(context, requests, options):
                before = (root / "cache-online.json").read_bytes()
                for index in range(2):
                    context.save_partial(dict(status="running", complete=False, requests=[], events=[dict(index=index)]))
                    self.assertEqual((root / "cache-online.json").read_bytes(), before)
                partial = json.loads((root / "partial.json").read_text())
                self.assertEqual(partial["trace_sha256"], "trace")
                self.assertEqual(partial["arm"], partial["factorial_arm"])
                self.assertNotIn("rows", partial)
                seen.append((partial["order_index"], partial["factorial_arm"]))
                return dict(status="completed", complete=True, trace_sha256="trace")
            with patch.object(cache_online, "load_frozen_traces", return_value=[trace]), \
                 patch.object(cache_online, "WanBlockExecutor", return_value=object()), \
                 patch.object(cache_online, "StepPairedExecutor", return_value=object()), \
                 patch.object(cache_online, "run_trace", side_effect=measure), \
                 patch.object(cache_online, "paired_gate", return_value=dict(passed=True, pooled={})), \
                 patch("ditserve.optimizations.worker.selected", return_value=dict(sana_backend="eager", image_batch_size=1)):
                result = run_cache_online(ctx, dict(cache_quality_qualified=True, full_qualification_completed=True))
            self.assertTrue(result["online_qualified"])
            self.assertEqual(seen, [(0, arm) for arm in cache_online.ARMS] +
                                  [(1, arm) for arm in reversed(cache_online.ARMS)])
            document = json.loads((root / "cache-online.json").read_text())
            self.assertEqual(len(document["rows"]), 8)
            self.assertEqual(document["active_trace_path"], str(root / "partial.json"))
            self.assertTrue((root / "cache-online-flushes.json").exists())
            self.assertEqual(len(document["warmups"]), 8)
            self.assertEqual(ctx.sana.generate_batch.call_count, 16)
            self.assertTrue(all(w["status"] == "completed" and len(w["services"]) == 2
                                for w in document["warmups"]))

    def test_reused_local_result_requires_fresh_factorial_warmup(self):
        context = SimpleNamespace(config=dict(old_online="unused"),
            metadata=dict(cache_local_evidence_reused=True, wan_warmed=True),
            check_budget=SimpleNamespace(remaining=lambda: 100000.))
        traces = [dict(requests=[dict(arrival_ms=0, service_ms=1, kind="image")])]
        with patch("ditserve.optimizations.cache_online.load_frozen_traces", return_value=traces):
            with self.assertRaisesRegex(ValueError, "factorial warmup"):
                run_cache_online(context, dict(cache_quality_qualified=True, full_qualification_completed=True))

    def test_full_factorial_preserves_requests_and_cpu_encoder_hook(self):
        from ditserve.optimizations import cache_online
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = [dict(seed=seed, load=.5, pattern=pattern, trace_sha256="trace-%s" % seed,
                requests=[dict(request_id="r%03d" % i, kind="video" if i in (0, 12) else "image",
                               seed=seed, arrival_ms=i * 100., deadline_ms=i * 100. + 50., service_ms=10.)
                          for i in range(24)]) for seed, pattern in ((102, "burst"), (101, "ordinary"))]
            calls, dispatched = [], []
            def encode(spec):
                calls.append(dict(spec))
                return dict(spec)
            ctx = SimpleNamespace(config=dict(old_online="unused"), output=root, run_root=root,
                worker=SimpleNamespace(encode_prompt=Mock(side_effect=AssertionError("default hook replaced")),
                                       prepare_encoded=lambda spec, payload: (spec, payload)),
                check_budget=SimpleNamespace(remaining=lambda: 100000.), opts=dict(prompts=["private warmup", "b"]),
                online_defaults=dict(encode_video=encode, encode_device="cpu", encode_on_scheduler=False),
                sana=SimpleNamespace(set_backend=Mock(), generate_batch=Mock(return_value=([None], dict(ms=1., finite=True))),
                                     steady_context=contextlib.nullcontext))
            off, paired = object(), object()
            def measure(context, requests, options):
                # Wrapper construction/warmup must not invoke the video encoder.
                self.assertEqual(len(calls), len(dispatched))
                warm = json.loads((root / "cache-online.json").read_text())["warmups"][-1]
                self.assertEqual(len(warm["services"]), 2)
                cached = warm["factorial_arm"].startswith("paired")
                self.assertIs(context.executor, paired if cached else off)
                self.assertEqual(options["policy"], "step" if warm["factorial_arm"].endswith("step") else "block-4")
                self.assertEqual(options["encode_device"], "cpu")
                self.assertFalse(options["encode_on_scheduler"])
                spec = dict(request_id="arrived-private", prompt="arrived payload", seed=101, cache=False,
                            threshold=0., max_continuous=1)
                payload = options["encode_video"](spec)
                self.assertEqual(payload["cache"], cached)
                if cached:
                    prepared, same_payload = options["prepare_video"](spec, payload)
                    self.assertEqual(prepared, same_payload)
                dispatched.append((warm["trace_sha256"], warm["order_index"], warm["factorial_arm"], requests))
                return dict(status="completed", complete=True, trace_sha256=warm["trace_sha256"], requests=requests)
            def gate(rows):
                self.assertEqual(len(rows), 8)
                self.assertEqual({r["arm"] for r in rows}, {"baseline", "candidate"})
                self.assertEqual({r["order_index"] for r in rows}, {0, 1})
                return dict(passed=False, failures=[], pooled=dict(system_gain=False))
            with patch.object(cache_online, "load_frozen_traces", return_value=traces), \
                 patch.object(cache_online, "WanBlockExecutor", return_value=off), \
                 patch.object(cache_online, "StepPairedExecutor", return_value=paired), \
                 patch.object(cache_online, "run_trace", side_effect=measure), \
                 patch.object(cache_online, "paired_gate", side_effect=gate) as paired_gate, \
                 patch("ditserve.optimizations.worker.selected", return_value=dict(sana_backend="eager", image_batch_size=1)):
                result = run_cache_online(ctx, dict(cache_quality_qualified=True, full_qualification_completed=True))
            self.assertEqual(len(dispatched), 16)
            self.assertEqual(sum(len(row[3]) for row in dispatched), 384)
            self.assertEqual(paired_gate.call_count, 4)
            self.assertEqual(ctx.sana.generate_batch.call_count, 32)
            for trace in traces:
                group = [x for x in dispatched if x[0] == trace["trace_sha256"]]
                self.assertEqual([(x[1], x[2]) for x in group], [(0, arm) for arm in cache_online.ARMS] +
                                 [(1, arm) for arm in reversed(cache_online.ARMS)])
                self.assertTrue(all(x[3] is trace["requests"] for x in group))
            self.assertFalse(result["online_qualified"])
            self.assertTrue(result["online_evaluation_complete"])

    def test_failure_keeps_current_trace_partial_and_prior_complete_rows(self):
        from ditserve.optimizations import cache_online
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = dict(seed=102, load=.5, pattern="burst", trace_sha256="trace",
                         requests=[dict(arrival_ms=0., service_ms=100., kind="image")])
            ctx = SimpleNamespace(config=dict(old_online="unused"), output=root, run_root=root,
                worker=SimpleNamespace(encode_prompt=lambda spec: spec, prepare_encoded=lambda spec, payload: None),
                check_budget=SimpleNamespace(remaining=lambda: 100000.), opts=dict(prompts=["warm"]),
                sana=SimpleNamespace(set_backend=Mock(), generate_batch=Mock(return_value=([None], dict(ms=1.))),
                                     steady_context=contextlib.nullcontext))
            count = []
            def measure(context, requests, options):
                count.append(1)
                if len(count) == 1:
                    return dict(status="completed", complete=True, trace_sha256="trace", requests=[])
                context.save_partial(dict(status="interrupted", complete=False, trace_sha256="trace",
                    requests=[dict(request_id="failed", arrived=True, status="failed"),
                              dict(request_id="waiting", arrived=True, status="interrupted")]))
                raise RuntimeError("injected service failure")
            with patch.object(cache_online, "load_frozen_traces", return_value=[trace]), \
                 patch.object(cache_online, "WanBlockExecutor", return_value=object()), \
                 patch.object(cache_online, "StepPairedExecutor", return_value=object()), \
                 patch.object(cache_online, "run_trace", side_effect=measure), \
                 patch("ditserve.optimizations.worker.selected", return_value=dict(sana_backend="eager", image_batch_size=1)):
                with self.assertRaisesRegex(RuntimeError, "injected service failure"):
                    run_cache_online(ctx, dict(cache_quality_qualified=True, full_qualification_completed=True))
            document = json.loads((root / "cache-online.json").read_text())
            partial = json.loads((root / "partial.json").read_text())
            self.assertEqual(document["status"], "incomplete")
            self.assertEqual(len(document["rows"]), 1)
            self.assertEqual(len(partial["requests"]), 2)
            self.assertTrue(all(r["arrived"] for r in partial["requests"]))
            self.assertNotIn("comparisons", document)


class FactorialWarmupTests(unittest.TestCase):
    def context(self, root, *, fail_step=None, reuse=True):
        class Guard:
            def __call__(self):
                return None
        worker = SimpleNamespace(_states={}, device="cuda:0", encoded=[], released=[], decoded=[],
            torch=SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=Mock(),
                max_memory_allocated=lambda device: 123, max_memory_reserved=lambda device: 234)),
            _memory=lambda: dict(allocated_bytes=10, reserved_bytes=20))
        def encode(spec):
            worker.encoded.append(dict(spec))
            return spec
        def prepare(spec, payload):
            self.assertEqual(spec, payload)
            state = SimpleNamespace(spec=spec, step_index=0, residency="gpu", events=[])
            worker._states[spec["request_id"]] = state
            return state
        def release(state):
            worker.released.append(state.spec["request_id"])
            state.residency = "released"
            del worker._states[state.spec["request_id"]]
        def finish(state):
            worker.decoded.append(state.spec["request_id"])
            self.assertEqual(state.step_index, 32)
            return SimpleNamespace(shape=(1, 81, 480, 832, 3))
        worker.encode_prompt, worker.prepare_encoded, worker.release, worker.finish = encode, prepare, release, finish
        class Executor:
            def __init__(self, worker):
                self._states = {}
            def attach(self, state):
                continuation = SimpleNamespace(base=state, closed=False)
                self._states[id(state)] = continuation
                return continuation
            def advance(self, continuation, blocks, *, check_budget):
                self_test.assertEqual(blocks, 4)
                check_budget()
                state = continuation.base
                if state.spec["cache"] and state.step_index == fail_step:
                    raise RuntimeError("injected warmup failure")
                index = state.step_index
                reused = bool(state.spec["cache"] and reuse and index == 8)
                state.step_index += 1
                state.events.append(dict(kind="step", step_index=index, completed_steps=index + 1,
                    path="reuse" if reused else "compute", reused_forwards=2 if reused else 0))
            def close(self, continuation):
                continuation.closed = True
                del self._states[id(continuation.base)]
        self_test = self
        context = SimpleNamespace(output=root, worker=worker, check_budget=Guard(), metadata=dict(wan_warmed=True),
            opts=dict(prompts=["private warmup prompt"], seeds=[101], steps=32),
            profile=dict(selected_shape=dict(height=480, width=832, frames=81)))
        return context, Executor

    def test_private_warmup_covers_off_k4_paired_full_reuse_decode_and_releases(self):
        from ditserve.optimizations import cache_online
        with tempfile.TemporaryDirectory() as directory:
            context, executor = self.context(Path(directory))
            with patch.object(cache_online, "WanBlockExecutor", executor), patch.object(cache_online, "StepPairedExecutor", executor):
                report = warm_cache_factorial(context)
            self.assertEqual([r["completed_steps"] for r in report["rows"]], [2, 32])
            self.assertEqual([r["decoded"] for r in report["rows"]], [False, True])
            self.assertTrue(report["rows"][1]["cache_paths"]["full_reuse_observed"])
            self.assertTrue(context.metadata["cache_factorial_warmed"])
            self.assertEqual(context.worker._states, {})
            self.assertEqual(len(context.worker.released), 2)
            self.assertTrue(all(r["worker_live_states_after"] == r["executor_live_states_after"] == 0 for r in report["rows"]))
            self.assertEqual({s["prompt"] for s in context.worker.encoded}, {"private warmup prompt"})
            self.assertEqual({s["seed"] for s in context.worker.encoded}, {101})
            self.assertTrue(all(s["request_id"].startswith("factorial-warm-") for s in context.worker.encoded))

    def test_warmup_failure_releases_own_state_and_cannot_set_ready(self):
        from ditserve.optimizations import cache_online
        with tempfile.TemporaryDirectory() as directory:
            context, executor = self.context(Path(directory), fail_step=8)
            with patch.object(cache_online, "WanBlockExecutor", executor), patch.object(cache_online, "StepPairedExecutor", executor):
                with self.assertRaisesRegex(RuntimeError, "injected warmup failure"):
                    warm_cache_factorial(context)
            report = json.loads((Path(directory) / "cache-factorial-warmup.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(context.metadata["cache_factorial_warmed"])
            self.assertEqual(context.worker._states, {})
            self.assertTrue(all(not value._states for value in context.executors.values()))

    def test_fixed_warmup_without_reuse_fails_closed(self):
        from ditserve.optimizations import cache_online
        with tempfile.TemporaryDirectory() as directory:
            context, executor = self.context(Path(directory), reuse=False)
            with patch.object(cache_online, "WanBlockExecutor", executor), patch.object(cache_online, "StepPairedExecutor", executor):
                with self.assertRaisesRegex(RuntimeError, "both full and reuse"):
                    warm_cache_factorial(context)
            self.assertFalse(context.metadata["cache_factorial_warmed"])
            self.assertEqual(context.worker._states, {})
            self.assertEqual(context.worker.decoded, [])

    def test_capacity_precondition_is_required_without_gpu_or_encoder_work(self):
        with tempfile.TemporaryDirectory() as directory:
            context, _ = self.context(Path(directory))
            context.metadata["wan_warmed"] = False
            with self.assertRaisesRegex(ValueError, "capacity"):
                warm_cache_factorial(context)
            self.assertEqual(context.worker.encoded, [])
            context.worker.torch.cuda.reset_peak_memory_stats.assert_not_called()


if __name__ == "__main__":
    unittest.main()
