"""CPU-only observer contract tests; no models, CUDA runtime or weights loaded."""

import contextlib
import gc
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


from ditserve.experiments import profile_v100_phases as phases


class FakeDevice:
    type = "cuda"
    index = 0

    def __str__(self):
        return "cuda:0"


class FakeTensor:
    shape = (4,)
    dtype = "torch.float32"
    device = FakeDevice()
    is_cuda = True

    def numel(self):
        return 4

    def element_size(self):
        return 4


class FakeClock:
    def __init__(self, events):
        self.value = 10.0
        self.events = events

    def __call__(self):
        self.events.append("clock")
        self.value += 0.01
        return self.value


class FakeCuda:
    def __init__(self, events):
        self.events = events

    def synchronize(self, device):
        self.events.append("synchronize:" + str(device))


class FakeProgress:
    def __init__(self, events, total):
        self.events, self.total, self.n = events, total, 0

    def __enter__(self):
        self.events.append("progress_enter")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("progress_exit:" + (exc_type.__name__ if exc_type else "normal"))
        return False

    def update(self, n=1):
        self.n += n
        self.events.append("progress_update")


class FakeContext:
    def __init__(self, name="blocks", steps=3, filled=True):
        self.name = name
        self.executed_steps = steps
        self.transformer_executed_steps = 2 * steps
        self.accumulated_executed_steps = steps
        self.accumulated_transformer_executed_steps = 2 * steps
        self.buffers = {"residual": FakeTensor()} if filled else {}

    def clear_buffers(self):
        self.buffers.clear()


class FakeManager:
    persistent_context = False

    def __init__(self, events, context=None):
        self.events = events
        context = context or FakeContext()
        self._cached_context_manager = {context.name: context}
        self._current_context = None
        self.calls = []

    def get_context(self, name=None):
        return self._cached_context_manager[name] if name is not None else self._current_context

    def reset_context(self, cached_context, *args, **kwargs):
        name = cached_context if isinstance(cached_context, str) else cached_context.name
        self.events.append("original_reset")
        self.calls.append(name)
        self._cached_context_manager[name].clear_buffers()
        new = FakeContext(name, steps=0, filled=False)
        self._cached_context_manager[name] = new
        return new


class FakePipe:
    def __init__(self, events, manager=None, steps=3):
        self.events = events
        self.scheduler = SimpleNamespace(timesteps=list(range(steps)), order=1)
        self.interrupt = False
        if manager is not None:
            self._context_manager = manager
            self.transformer = SimpleNamespace(_context_manager=manager)

    def progress_bar(self, iterable=None, total=None):
        return FakeProgress(self.events, total)


def fake_torch(events):
    return SimpleNamespace(cuda=FakeCuda(events), is_tensor=lambda value: isinstance(value, FakeTensor))


class LoopObserverTests(unittest.TestCase):
    def test_complete_loop_counts_updates_and_restores_original_progress_method(self):
        events = []
        pipe = FakePipe(events)
        original = pipe.progress_bar
        observer = phases.LoopObserver(fake_torch(events), pipe, 3, clock=FakeClock(events))
        with observer:
            with pipe.progress_bar(total=3) as progress:
                progress.update()
                progress.update(2)
        observer.report()
        self.assertEqual(pipe.progress_bar, original)
        self.assertEqual(len(observer.records), 1)
        self.assertEqual(observer.records[0]["status"], "completed")
        self.assertEqual(observer.records[0]["progress_updates"], 3)
        self.assertGreater(observer.records[0]["wall_ms"], 0)
        self.assertLess(events.index("synchronize:cuda:0"), events.index("clock"))
        self.assertEqual(events[-1], "progress_exit:normal")

    def test_loop_exception_propagates_and_cannot_be_reported_as_complete(self):
        events = []
        pipe = FakePipe(events)
        original = pipe.progress_bar
        observer = phases.LoopObserver(fake_torch(events), pipe, 3, clock=FakeClock(events))
        with self.assertRaisesRegex(RuntimeError, "mock original failure"):
            with observer:
                with pipe.progress_bar(total=3) as progress:
                    progress.update()
                    raise RuntimeError("mock original failure")
        self.assertEqual(pipe.progress_bar, original)
        self.assertEqual(observer.records[0]["status"], "failed")
        self.assertIn("progress_exit:RuntimeError", events)
        with self.assertRaises(Exception):
            observer.report()

    def test_incomplete_and_multiple_loops_are_not_silently_validated(self):
        for number_of_loops, updates in ((1, 2), (2, 3)):
            with self.subTest(loops=number_of_loops, updates=updates):
                events = []
                pipe = FakePipe(events)
                observer = phases.LoopObserver(fake_torch(events), pipe, 3, clock=FakeClock(events))
                with self.assertRaises(Exception):
                    with observer:
                        for _ in range(number_of_loops):
                            with pipe.progress_bar(total=3) as progress:
                                progress.update(updates)
                    observer.report()


class ResetObserverTests(unittest.TestCase):
    def test_pipeline_context_name_metadata_is_forwarded_without_argument_collision(self):
        class RecordingManager(FakeManager):
            def reset_context(self, cached_context, *args, **kwargs):
                self.forwarded = (cached_context, args, dict(kwargs))
                return super().reset_context(cached_context, *args, **kwargs)

        events = []
        manager = RecordingManager(events)
        original = manager.reset_context
        cache_config = object()
        observer = phases.ResetObserver(fake_torch(events), FakePipe(events, manager), 3,
                                        clock=FakeClock(events))
        with observer:
            # cache-dit 1.5 passes the context positionally AND its constructor
            # metadata as name=... inside **context_kwargs on every pipeline call.
            replacement = manager.reset_context("blocks", name="blocks", cache_config=cache_config)
        report = observer.report()
        self.assertEqual(manager.forwarded[0:2], ("blocks", ()))
        self.assertEqual(manager.forwarded[2]["name"], "blocks")
        self.assertIs(manager.forwarded[2]["cache_config"], cache_config)
        self.assertIs(replacement, manager.get_context("blocks"))
        self.assertEqual(manager.calls, ["blocks"])
        self.assertEqual(manager.reset_context, original)
        self.assertEqual(report["calls"][0]["status"], "completed")
        self.assertEqual(report["calls"][0]["before"]["executed_steps"], 3)

    def test_filled_context_delegates_once_even_when_pipe_and_transformer_alias_manager(self):
        events = []
        manager = FakeManager(events)
        pipe = FakePipe(events, manager)
        original = manager.reset_context
        observer = phases.ResetObserver(fake_torch(events), pipe, 3, clock=FakeClock(events))
        with observer:
            replacement = manager.reset_context("blocks")
        report = observer.report()
        self.assertEqual(report["unique_manager_count"], 1)
        self.assertEqual(report["context_count"], 1)
        self.assertEqual(manager.reset_context, original)
        self.assertIs(replacement, manager._cached_context_manager["blocks"])
        self.assertEqual(replacement.executed_steps, 0)
        self.assertEqual(manager.calls, ["blocks"])
        self.assertEqual(len(observer.records), 1)
        record = observer.records[0]
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["before"]["executed_steps"], 3)
        self.assertEqual(record["before"]["transformer_executed_steps"], 6)
        self.assertEqual(record["before"]["buffer_inventory"]["bytes"], 16)
        self.assertFalse(record["before"]["buffer_inventory"]["is_complete_checkpoint"])
        self.assertGreaterEqual(record["synchronized_wall_ms"], record["call_wall_ms"])
        self.assertAlmostEqual(record["synchronized_wall_ms"], record["call_wall_ms"] + record["post_call_synchronize_ms"])

    def test_empty_context_is_rejected_without_invoking_reset(self):
        events = []
        manager = FakeManager(events, FakeContext(filled=False))
        pipe = FakePipe(events, manager)
        original = manager.reset_context
        with self.assertRaises(Exception):
            observer = phases.ResetObserver(fake_torch(events), pipe, 3, clock=FakeClock(events))
            with observer:
                manager.reset_context("blocks")
            observer.report()
        self.assertEqual(manager.calls, [])
        self.assertEqual(manager.reset_context, original)

    def test_partial_previous_request_cannot_be_misreported_as_full_context_reset(self):
        for steps, forwards in ((2, 4), (3, 5)):
            with self.subTest(steps=steps, forwards=forwards):
                events = []
                context = FakeContext(steps=steps)
                context.transformer_executed_steps = forwards
                manager = FakeManager(events, context)
                with self.assertRaises(Exception):
                    observer = phases.ResetObserver(fake_torch(events), FakePipe(events, manager), 3, clock=FakeClock(events))
                    with observer:
                        manager.reset_context("blocks")
                    observer.report()
                self.assertEqual(manager.calls, [])

    def test_observer_rejects_different_pipeline_and_transformer_managers(self):
        events = []
        manager = FakeManager(events)
        pipe = FakePipe(events, manager)
        pipe.transformer._context_manager = FakeManager(events)
        with self.assertRaises(Exception):
            observer = phases.ResetObserver(fake_torch(events), pipe, 3, clock=FakeClock(events))
            with observer:
                manager.reset_context("blocks")
            observer.report()
        self.assertEqual(manager.calls, [])

    def test_original_reset_exception_propagates_and_restores_method(self):
        class FailingManager(FakeManager):
            def reset_context(self, name, *args, **kwargs):
                self.calls.append(name)
                raise RuntimeError("mock reset failure")

        events = []
        manager = FailingManager(events)
        original = manager.reset_context
        observer = phases.ResetObserver(fake_torch(events), FakePipe(events, manager), 3, clock=FakeClock(events))
        with self.assertRaisesRegex(RuntimeError, "mock reset failure"):
            with observer:
                manager.reset_context("blocks")
        self.assertEqual(manager.reset_context, original)
        self.assertEqual(manager.calls, ["blocks"])
        self.assertEqual(observer.records[0]["status"], "failed")
        with self.assertRaises(Exception):
            observer.report()

    def test_every_context_must_reset_once_without_duplicate_delegation(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                events = []
                manager = FakeManager(events)
                manager._cached_context_manager["second"] = FakeContext("second")
                observer = phases.ResetObserver(fake_torch(events), FakePipe(events, manager), 3, clock=FakeClock(events))
                with self.assertRaises(Exception):
                    with observer:
                        manager.reset_context("blocks")
                        if duplicate:
                            manager.reset_context("blocks")
                    observer.report()
                self.assertEqual(manager.calls, ["blocks"])

    def test_metadata_record_does_not_keep_old_context_or_tensor_alive(self):
        events = []
        manager = FakeManager(events)
        old_context = weakref.ref(manager.get_context("blocks"))
        old_tensor = weakref.ref(manager.get_context("blocks").buffers["residual"])
        observer = phases.ResetObserver(fake_torch(events), FakePipe(events, manager), 3, clock=FakeClock(events))
        with observer:
            manager.reset_context("blocks")
        observer.report()
        gc.collect()
        self.assertIsNone(old_context())
        self.assertIsNone(old_tensor())
        self.assertEqual(observer.records[0]["before"]["buffer_inventory"]["bytes"], 16)
        # The retained evidence must also be serializable without custom tensor handling.
        json.dumps(observer.records)


class PlanTests(unittest.TestCase):
    def test_plan_does_not_import_heavy_dependencies_query_gpu_or_write_output(self):
        manifest = {"status": "running", "selected_dtype": "float16"}
        references = {arm: {"path": "/mock/" + arm + "/profile.json"} for arm in ("none", "cache-dit")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-be-created"
            stdout = io.StringIO()
            with patch.object(phases, "validate_suite", return_value=(manifest, references)) as validate, \
                    patch.object(phases.p, "preflight", side_effect=AssertionError("unexpected preflight")) as preflight, \
                    patch.object(phases.runner, "resolve_gpu", side_effect=AssertionError("unexpected GPU query")) as gpu, \
                    patch.object(phases.importlib, "import_module", side_effect=AssertionError("unexpected dynamic import")) as imports, \
                    patch.object(phases.subprocess, "run", side_effect=AssertionError("unexpected child process")) as process, \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(phases.main(["--plan", "--output", str(output)]), 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["suite_status"], "running")
            self.assertTrue(plan["execution_requires_finished_full_suite"])
            self.assertFalse(plan["new_e2e_speedup_computed"])
            self.assertFalse(output.exists())
            self.assertFalse(validate.call_args.kwargs["require_finished"])
            preflight.assert_not_called()
            gpu.assert_not_called()
            imports.assert_not_called()
            process.assert_not_called()

    def test_execution_requires_finished_suite_before_any_gpu_or_child_process_call(self):
        for manifest in ({"status": "running"}, {"status": "completed"}):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "must-not-be-created"
                with patch.object(phases, "read_json", return_value=manifest) as read, \
                        patch.object(phases.p, "preflight", side_effect=AssertionError("unexpected preflight")) as preflight, \
                        patch.object(phases.runner, "resolve_gpu", side_effect=AssertionError("unexpected GPU query")) as gpu, \
                        patch.object(phases.subprocess, "run", side_effect=AssertionError("unexpected child process")) as process, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(phases.main(["--suite-dir", "/mock/unfinished", "--output", str(output)]), 2)
                self.assertEqual(read.call_count, 1)
                self.assertFalse(output.exists())
                preflight.assert_not_called()
                gpu.assert_not_called()
                process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
