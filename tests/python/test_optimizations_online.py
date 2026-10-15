import copy
import hashlib
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from ditserve.directions.budget import BudgetStop
from ditserve.optimizations.online import (EventLog, TraceJournal, attribution, instrumented_finish,
    load_frozen_traces, paired_gate, ready_batch, request_identity, run_trace)


def request(index, kind="image", arrival=0., deadline=1000.):
    return dict(request_id="r%03d" % index, kind=kind, arrival_ms=arrival,
                deadline_ms=deadline, service_ms=10., seed=101 + index,
                status="pending", start_ms=None, finish_ms=None)


class FakeWorker:
    def __init__(self):
        self.released = []
        self.encodes = []

    def encode_prompt(self, spec):
        self.encodes.append((spec, threading.get_ident()))
        return SimpleNamespace(cpu_prompt_encode_ms=1.)

    def prepare_encoded(self, spec, payload):
        return SimpleNamespace(request_id=spec["request_id"], spec=spec, step_index=0,
                               residency="gpu")

    def release(self, state):
        state.residency = "released"
        self.released.append(state.request_id)


class FakeExecutor:
    def attach(self, state):
        return SimpleNamespace(base=state, closed=False)

    def quantum(self, continuation, policy):
        return 4

    def advance(self, continuation, blocks, check_budget):
        check_budget()
        continuation.base.step_index += 1

    def close(self, continuation):
        continuation.closed = True


class FakeSana:
    def generate(self, prompt, seed, check_budget):
        check_budget()
        return [object()], dict(finite=True)

    def generate_batch(self, prompts, seeds, check_budget):
        check_budget()
        return [object() for _ in prompts], dict(finite=True)


class FakeCudaEvent:
    def __init__(self, cuda, index):
        self.cuda, self.index, self.records, self.complete = cuda, index, 0, False

    def record(self, stream):
        self.records += 1
        self.recorded_at = time.perf_counter()
        self.complete = self.records == 1 or self.cuda.complete_work
        self.cuda.calls.append(("record", self.index, self.records, self.recorded_at, threading.get_ident()))

    def synchronize(self):
        self.cuda.calls.append(("synchronize", self.index, time.perf_counter()))
        time.sleep(.002)
        for event in self.cuda.events:
            event.complete = True

    def query(self):
        self.cuda.calls.append(("query", self.index, time.perf_counter()))
        if self.cuda.query_error:
            raise RuntimeError("fake query error")
        return self.complete

    def elapsed_time(self, event):
        if not event.complete:
            raise AssertionError("elapsed_time before completion")
        self.cuda.calls.append(("elapsed", self.index, event.index))
        return (event.recorded_at-self.recorded_at)*1000


class FakeCuda:
    def __init__(self, complete_work=True, query_error=False):
        self.events, self.calls = [], []
        self.complete_work, self.query_error = complete_work, query_error

    def current_stream(self, device):
        if device != "cuda:0":
            raise AssertionError("unadmitted device used for A100 marker")
        return SimpleNamespace(cuda_stream=17)

    def Event(self, *, enable_timing):
        if enable_timing is not True:
            raise AssertionError("timing must be enabled")
        event = FakeCudaEvent(self, len(self.events))
        self.events.append(event)
        return event


def context(check=None):
    saved = []
    return SimpleNamespace(worker=FakeWorker(), executor=FakeExecutor(), sana=FakeSana(),
                           opts=dict(prompts=["red bicycle", "blue boat"], seeds=[101], steps=2),
                           profile=dict(selected_shape=dict(height=480, width=832, frames=17)),
                           check_budget=check or (lambda: None), save_partial=saved.append, saved=saved)


def hooks(**options):
    return dict(finish_video=lambda state, emit: [object()], **options)


class StreamMarkerTests(unittest.TestCase):
    def context(self, **cuda_options):
        ctx, cuda = context(), FakeCuda(**cuda_options)
        ctx.worker.torch = SimpleNamespace(cuda=cuda)
        return ctx, cuda

    def test_pool_is_initialized_before_arrivals_and_only_origin_synchronizes(self):
        ctx, cuda = self.context()
        ctx.opts["steps"] = 5
        result = run_trace(ctx, [request(0, "video", arrival=15)], hooks())
        row, evidence = result["requests"][0], result["stream_marker_evidence"]
        self.assertEqual(len(cuda.events), 3)  # first A100, first DiT, origin
        self.assertEqual([event.records for event in cuda.events], [2, 2, 1])
        self.assertEqual(len([c for c in cuda.calls if c[0] == "synchronize"]), 1)
        self.assertEqual(evidence["origin_synchronizations"], 1)
        self.assertEqual(evidence["in_trace_added_synchronizations"], 0)
        self.assertEqual(evidence["complete_count"], 2)
        anchor = evidence["host_anchor_bracket"]
        self.assertLessEqual(anchor["record_before_trace_ms"], anchor["synchronize_after_trace_ms"])
        self.assertLessEqual(anchor["synchronize_after_trace_ms"], 0.)
        self.assertGreaterEqual(evidence["pre_trace_setup_ms"], anchor["host_bracket_width_ms"])
        for field, stage in (("first_a100_stream_marker", "video_prepare"),
                             ("first_dit_stream_marker", "video_quantum")):
            marker = row[field]
            self.assertEqual(marker["stage"], stage)
            self.assertEqual(marker["status"], "complete")
            self.assertEqual(marker["device"], "cuda:0")
            self.assertEqual(marker["stream_id"], 17)
            self.assertGreaterEqual(marker["host_record_start_ms"], row["arrival_ms"])
            self.assertGreaterEqual(marker["device_origin_elapsed_ms"], 0)
            self.assertEqual(marker["kernel_timestamp_status"], "unverified")
            self.assertIn("must not be subtracted from host arrival", marker["scope"])
        self.assertTrue(any(saved["stream_marker_evidence"]["pending_count"] == 2 for saved in ctx.saved[:-1]))
        self.assertEqual(row["first_a100_dispatch"]["stage"], "video_prepare")

    def test_a100_encode_gets_first_marker_but_v100_encoder_does_not(self):
        main = threading.get_ident()
        for device, synchronous, stage in (("cuda:0", True, "encode_service"),
                                          ("cuda:1", False, "video_prepare")):
            with self.subTest(device=device):
                ctx, cuda = self.context()
                result = run_trace(ctx, [request(0, "video", arrival=10)],
                                   hooks(encode_device=device, encode_on_scheduler=synchronous))
                row = result["requests"][0]
                self.assertEqual(row["first_a100_stream_marker"]["stage"], stage)
                self.assertEqual(row["first_dit_stream_marker"]["stage"], "video_quantum")
                self.assertTrue(all(c[4] == main for c in cuda.calls if c[0] == "record"))

    def test_failed_work_saves_pending_marker_and_does_not_wait_for_it(self):
        ctx, cuda = self.context(complete_work=False)
        def fail(spec):
            raise RuntimeError("backend failure")
        with self.assertRaisesRegex(RuntimeError, "backend failure"):
            run_trace(ctx, [request(0, "video", arrival=10),
                            request(1, arrival=10000, deadline=20000)],
                      hooks(encode_device="cuda:0", encode_on_scheduler=True, encode_video=fail))
        result = ctx.saved[-1]
        marker = result["requests"][0]["first_a100_stream_marker"]
        self.assertEqual(marker["status"], "pending")
        self.assertIsNone(marker["device_origin_elapsed_ms"])
        self.assertIsNone(result["requests"][0]["first_dit_stream_marker"])
        self.assertIsNone(result["requests"][1]["first_a100_stream_marker"])
        self.assertEqual(result["metrics"]["arrived"], 1)
        self.assertEqual(len([c for c in cuda.calls if c[0] == "synchronize"]), 1)
        self.assertFalse(any(c[0] == "elapsed" for c in cuda.calls))

    def test_query_error_does_not_replace_backend_error(self):
        ctx, cuda = self.context(query_error=True)
        def fail(batch, emit):
            raise ValueError("original backend failure")
        with self.assertRaisesRegex(ValueError, "original backend failure"):
            run_trace(ctx, [request(0)], hooks(generate_images=fail))
        result = ctx.saved[-1]
        self.assertEqual(result["requests"][0]["first_a100_stream_marker"]["status"], "error")
        self.assertIn("original backend failure", result["error"])
        self.assertEqual(result["stream_marker_evidence"]["error_count"], 1)

    def test_image_batch_shares_event_and_sana_dit_stays_unobserved(self):
        ctx, cuda = self.context()
        result = run_trace(ctx, [request(0), request(1)], hooks(image_batch_size=2))
        a, b = result["requests"]
        self.assertEqual(a["first_a100_stream_marker"], b["first_a100_stream_marker"])
        self.assertEqual(a["first_a100_stream_marker"]["request_ids"], ["r000", "r001"])
        self.assertIsNone(a["first_dit_stream_marker"])
        self.assertIsNone(b["first_dit_stream_marker"])
        self.assertEqual(result["stream_marker_evidence"]["recorded_count"], 1)
        self.assertEqual([event.records for event in cuda.events], [2, 1, 1])

    def test_no_cuda_interface_has_no_synthetic_device_timestamps(self):
        result = run_trace(context(), [request(0)], hooks())
        self.assertIsNone(result["requests"][0]["first_a100_stream_marker"])
        self.assertEqual(result["stream_marker_evidence"]["status"], "unavailable_no_worker_cuda_interface")
        self.assertIsNone(result["stream_marker_evidence"]["host_anchor_bracket"])


class PersistenceTests(unittest.TestCase):
    def test_partial_size_and_destination_do_not_depend_on_history(self):
        snapshots = []
        for history_count in (0, 7):
            written = []
            def write(path, value):
                written.append((str(path), json.dumps(value, sort_keys=True)))
            journal = TraceJournal(Path('/tmp/fake-online-test') / 'online.json', write)
            record = dict(rows=[dict(payload='old' * 1000)] * history_count)
            journal.save_record(record)
            partial = journal.start_trace(dict(trace_sha256='trace', order_index=1, arm='baseline'))
            written.clear()
            partial(dict(status='running', requests=[dict(request_id='r000')], events=[]))
            self.assertEqual(len(written), 1)
            self.assertTrue(written[0][0].endswith('/partial.json'))
            self.assertNotIn('oldold', written[0][1])
            snapshots.append(written[0][1])
        self.assertEqual(snapshots[0], snapshots[1])

    def test_flush_interval_is_nonrecursive_and_visible_to_later_requests(self):
        ctx, clock, snapshots = context(), [10.], []
        def write(value):
            snapshots.append(copy.deepcopy(value))
            clock[0] += .005
        ctx.save_partial = write
        with patch('ditserve.optimizations.online.time.perf_counter', side_effect=lambda: clock[0]):
            result = run_trace(ctx, [request(0), request(1)], hooks())
        counts = [sum(e['kind'] == 'partial_flush' for e in v['events']) for v in snapshots]
        self.assertEqual(counts, [0, 1, 2])
        flushes = [e for e in result['events'] if e['kind'] == 'partial_flush']
        self.assertEqual(len(flushes), 3)
        self.assertTrue(all(abs(e['end_ms']-e['start_ms']-5.) < 1e-6 for e in flushes))
        self.assertAlmostEqual(result['requests'][1]['start_ms'], 5.)
        self.assertAlmostEqual(result['elapsed_ms'], 10.)
        self.assertAlmostEqual(flushes[-1]['end_ms'], 15.)
        self.assertFalse(flushes[-1]['inside_trace_interval'])
        self.assertEqual(result['persistence']['partial_flush_count'], 3)
        self.assertEqual(sum(e['kind'] == 'partial_snapshot_build' for e in result['events']), 3)

    def test_failed_flush_is_carried_by_exception_snapshot_without_recursive_write(self):
        ctx, snapshots = context(), []
        def write(value):
            snapshots.append(copy.deepcopy(value))
            if len(snapshots) == 1:
                raise OSError('test write failure')
        ctx.save_partial = write
        with self.assertRaisesRegex(OSError, 'test write failure'):
            run_trace(ctx, [request(0)], hooks())
        self.assertEqual(len(snapshots), 2)
        failed = [e for e in snapshots[-1]['events'] if e['kind'] == 'partial_flush']
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['status'], 'failed')
        self.assertEqual(snapshots[-1]['status'], 'interrupted')
        self.assertFalse(snapshots[-1]['complete'])

    def test_cumulative_flush_sidecar_is_small_and_does_not_rewrite_main_recursively(self):
        written = []
        def write(path, value):
            written.append((Path(path).name, copy.deepcopy(value)))
        journal = TraceJournal(Path('/tmp/fake-online-test/online.json'), write)
        journal.save_record(dict(rows=[dict(payload='only-in-main')]))
        self.assertEqual([name for name, _ in written], ['online.json', 'online-flushes.json'])
        sidecar = written[-1][1]
        self.assertEqual(len(sidecar['flushes']), 1)
        self.assertGreaterEqual(sidecar['flushes'][0]['host_ms'], 0)
        self.assertNotIn('only-in-main', json.dumps(sidecar))

    def test_terminal_partial_own_flush_is_preserved_in_separate_small_sidecar(self):
        written = {}
        def write(path, value):
            written[Path(path).name] = copy.deepcopy(value)
        journal = TraceJournal(Path('/tmp/fake-online-test/online.json'), write)
        ctx = context()
        ctx.save_partial = journal.start_trace(dict(trace_sha256='trace', order_index=0, arm='baseline'))
        result = run_trace(ctx, [request(0)], hooks())
        partial_flushes = [e for e in written['partial.json']['events'] if e['kind'] == 'partial_flush']
        self.assertEqual(len(partial_flushes), 1)
        self.assertEqual(result['persistence']['partial_flush_count'], 2)
        final = [e for e in written['online-flushes.json']['flushes'] if e['kind'] == 'terminal_partial']
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]['report_status'], 'completed')
        self.assertEqual(final[0]['trace_identity']['trace_sha256'], 'trace')
        self.assertGreaterEqual(final[0]['host_ms'], 0)

    def test_interrupted_final_flush_has_own_sidecar_and_keeps_arrived_denominator(self):
        written, failed = {}, [False]
        def write(path, value):
            if Path(path).name == 'partial.json' and value.get('status') == 'running' and not failed[0]:
                failed[0] = True
                raise OSError('first partial failed')
            written[Path(path).name] = copy.deepcopy(value)
        journal = TraceJournal(Path('/tmp/fake-online-test/online.json'), write)
        ctx = context()
        ctx.save_partial = journal.start_trace(dict(trace_sha256='trace', order_index=0, arm='candidate'))
        with self.assertRaisesRegex(OSError, 'first partial failed'):
            run_trace(ctx, [request(0), request(1, arrival=10000, deadline=20000)], hooks())
        self.assertEqual(written['partial.json']['metrics']['arrived'], 1)
        self.assertEqual(written['partial.json']['requests'][1]['status'], 'pending')
        final = [e for e in written['online-flushes.json']['flushes'] if e['kind'] == 'terminal_partial']
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]['report_status'], 'interrupted')
        self.assertGreaterEqual(final[0]['host_ms'], 0)


class FrozenTraceTests(unittest.TestCase):
    def write_source(self, directory, rows):
        path = Path(directory) / "source.json"
        path.write_text(json.dumps(dict(status="incomplete", rows=rows)))
        return path

    def historical(self, requests, **fields):
        rows = copy.deepcopy(requests)
        for row in rows:
            row.update(status="completed", start_ms=5., finish_ms=10., arrived=True,
                       service={"irrelevant": "result"})
        return dict(seed=102, load=.5, pattern="burst", policy="block-4",
                    trace_sha256=request_identity(requests), rows=rows, **fields)

    def test_preserves_original_numbers_and_strips_results_even_in_incomplete_phase(self):
        original = [request(0, "video", arrival=1.234567890123), request(1, arrival=2.345678901234)]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_source(directory, [self.historical(original)])
            result = load_frozen_traces(path, [(102, .5, "burst")])[0]
            self.assertEqual(result["requests"], original)
            self.assertEqual(result["trace_sha256"], request_identity(original))
            self.assertEqual(result["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_rejects_disagreeing_policy_and_corrupt_saved_digest(self):
        original = [request(0)]
        base = self.historical(original)
        changed = copy.deepcopy(base)
        changed["policy"] = "step"
        changed["rows"][0]["deadline_ms"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_source(directory, [base, changed])
            with self.assertRaisesRegex(ValueError, "policies disagree"):
                load_frozen_traces(path, [(102, .5, "burst")])
            base["trace_sha256"] = "wrong"
            path = self.write_source(directory, [base])
            with self.assertRaisesRegex(ValueError, "digest"):
                load_frozen_traces(path, [(102, .5, "burst")])

    def test_rejects_missing_or_incomplete_selected_trace(self):
        row = self.historical([request(0)])
        row["rows"][0]["status"] = "failed"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_source(directory, [row])
            with self.assertRaisesRegex(ValueError, "complete historical"):
                load_frozen_traces(path, [(102, .5, "burst")])


class SchedulingTests(unittest.TestCase):
    def test_batch_requires_consecutive_ready_edf_images(self):
        rows = [request(0, deadline=10), request(1, "video", deadline=20), request(2, deadline=30)]
        ids = {r["request_id"] for r in rows}
        self.assertEqual(len(ready_batch(rows, ids, 0, 2)), 1)
        self.assertEqual(len(ready_batch(rows, ids - {"r001"}, 0, 2)), 1)
        rows[2]["arrival_ms"] = 50
        self.assertEqual(len(ready_batch(rows, ids - {"r001"}, 0, 2)), 1)

    def test_batch_does_not_cross_waiting_video_at_head_or_between_images(self):
        for video_deadline in (5, 20):
            with self.subTest(video_deadline=video_deadline):
                rows = [request(0, deadline=10), request(1, "video", deadline=video_deadline),
                        request(2, deadline=30)]
                batch = ready_batch(rows, {"r000", "r002"}, 0, 2)
                self.assertEqual([r["request_id"] for r in batch], ["r000"])
        rows = [request(0, deadline=10), request(1, "video", deadline=40), request(2, deadline=30)]
        self.assertEqual([r["request_id"] for r in ready_batch(rows, {"r000", "r002"}, 0, 2)],
                         ["r000", "r002"])

    def test_waiting_video_prevents_batching_without_blocking_single_image_work(self):
        image_served, seen_batches = threading.Event(), []
        def encode(spec):
            self.assertTrue(image_served.wait(1), "waiting encoder prevented independent image work")
            return SimpleNamespace(cpu_prompt_encode_ms=1.)
        def images(batch, emit):
            seen_batches.append([r["request_id"] for r in batch])
            image_served.set()
            return [object() for _ in batch], dict(finite=True)
        result = run_trace(context(), [request(0, deadline=10), request(1, "video", deadline=20),
                                       request(2, deadline=30)],
                           hooks(image_batch_size=2, encode_video=encode, generate_images=images))
        self.assertEqual(seen_batches, [["r000"], ["r002"]])
        self.assertTrue(result["complete"])

    def test_batch_finishes_together_with_independent_deadlines(self):
        def images(batch, emit):
            time.sleep(.01)
            return [object(), object()], dict(finite=True)
        result = run_trace(context(), [request(0, deadline=1), request(1, deadline=1000)],
                           hooks(image_batch_size=2, generate_images=images))
        a, b = result["requests"]
        self.assertEqual(a["finish_ms"], b["finish_ms"])
        self.assertEqual(a["start_ms"], b["start_ms"])
        self.assertEqual(result["image_metrics"]["on_time"], 1)
        self.assertEqual(result["image_metrics"]["arrived"], 2)

    def test_separate_arrival_thread_can_encode_while_image_service_is_running(self):
        started = threading.Event()
        main = threading.get_ident()
        ctx = context()
        def encode(spec):
            self.assertNotEqual(threading.get_ident(), main)
            started.set()
            return SimpleNamespace(cpu_prompt_encode_ms=1.)
        def images(batch, emit):
            self.assertTrue(started.wait(1), "arrival clock was blocked by image dispatch")
            return [object()], dict(finite=True)
        result = run_trace(ctx, [request(0, deadline=5), request(1, "video", arrival=20)],
                           hooks(encode_video=encode, generate_images=images))
        queued = next(e for e in result["events"] if e["kind"] == "encode_queued")
        image = next(e for e in result["events"] if e["kind"] == "image_service")
        self.assertGreaterEqual(queued["start_ms"], 20)
        self.assertLess(queued["start_ms"], image["end_ms"])
        self.assertTrue(result["complete"])

    def test_never_reads_future_prompt_payload(self):
        origin = time.perf_counter()
        class Guarded(dict):
            def __getitem__(self, key):
                if key == "prompt":
                    if (time.perf_counter() - origin) * 1000 < dict.__getitem__(self, "arrival_ms"):
                        raise AssertionError("future prompt accessed")
                return dict.__getitem__(self, key)
        rows = [Guarded(request(0), prompt="first"),
                Guarded(request(1, "video", arrival=15), prompt="later video"),
                Guarded(request(2, arrival=25), prompt="later image")]
        result = run_trace(context(), rows, hooks())
        self.assertTrue(result["complete"])
        self.assertEqual([r["status"] for r in rows], ["pending"] * 3)

    def test_scheduler_encoder_is_serial_and_reschedules_after_encode(self):
        main, sequence = threading.get_ident(), []
        def encode(spec):
            self.assertEqual(threading.get_ident(), main)
            sequence.append("encode")
            time.sleep(.02)
            return SimpleNamespace(cpu_prompt_encode_ms=20.)
        def prepare(spec, payload):
            sequence.append("prepare")
            return FakeWorker().prepare_encoded(spec, payload)
        def images(batch, emit):
            self.assertEqual(sequence, ["encode"])
            sequence.append("image")
            return [object()], dict(finite=True)
        result = run_trace(context(), [request(0, "video"), request(1, arrival=5, deadline=100)],
                           hooks(encode_on_scheduler=True, encode_video=encode,
                                 prepare_video=prepare, generate_images=images))
        self.assertEqual(sequence, ["encode", "image", "prepare"])
        self.assertTrue(result["complete"])

    def test_encoder_device_and_first_a100_dispatch_are_distinct_from_thread(self):
        main = threading.get_ident()
        for device, synchronous, first_stage in (("cpu", False, "video_prepare"),
                                                 ("cuda:0", True, "encode_service"),
                                                 ("cuda:1", False, "video_prepare")):
            with self.subTest(device=device):
                def encode(spec):
                    self.assertEqual(threading.get_ident() == main, synchronous)
                    return SimpleNamespace(cpu_prompt_encode_ms=1.)
                result = run_trace(context(), [request(0, "video", arrival=15)],
                    hooks(encode_device=device, encode_on_scheduler=synchronous, encode_video=encode))
                row = result["requests"][0]
                encode_events = [e for e in result["events"] if e["kind"].startswith("encode_")]
                self.assertEqual({e["kind"] for e in encode_events},
                                 {"encode_queued", "encode_start", "encode_service", "encode_ready"})
                for event in encode_events:
                    self.assertEqual(event["resource"], device)
                    self.assertEqual(event["compute_device"], device)
                    self.assertEqual(event["executor_actor"], "scheduler" if synchronous else "encoder_thread")
                    self.assertGreaterEqual(event["start_ms"], row["arrival_ms"])
                queued = next(e for e in encode_events if e["kind"] == "encode_queued")
                self.assertEqual(queued["queue_actor"], "scheduler" if synchronous else "arrival_thread")
                first, dit = row["first_a100_dispatch"], row["first_dit_quantum_dispatch"]
                self.assertEqual(first["stage"], first_stage)
                self.assertEqual(first["compute_device"], "cuda:0")
                self.assertGreaterEqual(first["start_ms"], row["arrival_ms"])
                self.assertGreaterEqual(first["end_ms"], first["start_ms"])
                self.assertGreaterEqual(dit["start_ms"], first["end_ms"])
                self.assertEqual(dit["stage"], "video_quantum")
                self.assertEqual(dit["status"], "completed")
                self.assertEqual(dit["kernel_timestamp_status"], "unverified")
                self.assertIn("not a CUDA kernel timestamp", dit["clock_scope"])
                self.assertEqual(sum(e.get("first_a100_dispatch", False) for e in result["events"]), 1)
                self.assertEqual(sum(e.get("first_dit_quantum_dispatch", False) for e in result["events"]), 1)
                self.assertEqual(result["dispatch_evidence"]["first_a100_dispatch_count"], 1)
                self.assertEqual(result["dispatch_evidence"]["first_dit_quantum_dispatch_count"], 1)

    def test_failed_first_dispatch_is_retained_without_inventing_kernel_or_dit_start(self):
        ctx = context()
        def fail(spec):
            raise RuntimeError("encoder failed")
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            run_trace(ctx, [request(0, "video", arrival=10)],
                hooks(encode_device="cuda:0", encode_on_scheduler=True, encode_video=fail))
        result = ctx.saved[-1]
        row = result["requests"][0]
        first = row["first_a100_dispatch"]
        self.assertEqual(first["stage"], "encode_service")
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["error"], "RuntimeError")
        self.assertGreaterEqual(first["start_ms"], row["arrival_ms"])
        self.assertGreaterEqual(first["end_ms"], first["start_ms"])
        self.assertIsNone(row["first_dit_quantum_dispatch"])
        self.assertEqual(row["kernel_timestamp_status"], "unverified")

    def test_image_batch_shares_host_dispatch_but_does_not_claim_dit_timestamp(self):
        result = run_trace(context(), [request(0), request(1)], hooks(image_batch_size=2))
        first, second = result["requests"]
        self.assertEqual(first["first_a100_dispatch"], second["first_a100_dispatch"])
        self.assertEqual(first["first_a100_dispatch"]["stage"], "image_service")
        self.assertIsNone(first["first_dit_quantum_dispatch"])
        self.assertEqual(result["dispatch_evidence"]["first_a100_dispatch_count"], 2)
        self.assertEqual(result["dispatch_evidence"]["first_dit_quantum_dispatch_count"], 0)

    def test_dispatch_observations_do_not_change_frozen_identity(self):
        original = [request(0, "video"), request(1)]
        result = run_trace(context(), original, hooks())
        self.assertEqual(result["trace_sha256"], request_identity(original))
        self.assertNotIn("first_a100_dispatch", original[0])

    def test_final_quantum_is_followed_by_edf_before_video_finish(self):
        ctx, seen = context(), []
        ctx.opts["steps"] = 1
        def advance(cont, blocks, check_budget):
            cont.base.step_index += 1
            time.sleep(.02)
        ctx.executor.advance = advance
        def images(batch, emit):
            seen.append("image")
            return [object()], dict(finite=True)
        def finish(state, emit):
            seen.append("finish")
            return [object()]
        run_trace(ctx, [request(0, "video"), request(1, arrival=5, deadline=100)],
                  dict(encode_on_scheduler=True, generate_images=images, finish_video=finish))
        self.assertEqual(seen, ["image", "finish"])

    def test_failed_batch_retains_all_arrived_and_does_not_read_future(self):
        class Future(dict):
            def __getitem__(self, key):
                if key == "prompt":
                    raise AssertionError("future prompt read")
                return super().__getitem__(key)
        ctx = context()
        def fail(batch, emit):
            raise RuntimeError("batch failure")
        requests = [request(0), request(1), Future(request(2, arrival=10000, deadline=20000), prompt="secret")]
        with self.assertRaisesRegex(RuntimeError, "batch failure"):
            run_trace(ctx, requests, hooks(image_batch_size=2, generate_images=fail))
        report = ctx.saved[-1]
        self.assertEqual(report["metrics"]["arrived"], 2)
        self.assertEqual(report["metrics"]["failed_or_incomplete"], 2)
        self.assertEqual(report["requests"][2]["status"], "pending")

    def test_budget_stop_saves_arrived_denominator_and_events(self):
        def stop():
            raise BudgetStop("test exhausted")
        ctx = context(stop)
        with self.assertRaises(BudgetStop):
            run_trace(ctx, [request(0), request(1, arrival=10000, deadline=20000)], hooks())
        report = ctx.saved[-1]
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["metrics"]["arrived"], 1)
        self.assertEqual(report["metrics"]["failed_or_incomplete"], 1)
        self.assertEqual(report["requests"][1]["status"], "pending")
        self.assertIsNone(report["requests"][1]["first_a100_dispatch"])
        self.assertIsNone(report["requests"][1]["first_dit_quantum_dispatch"])

    def test_service_stage_origin_is_converted_without_fabricating_cuda_markers(self):
        def images(batch, emit):
            origin = time.perf_counter()
            return [object()], dict(finite=True, origin_perf_counter_s=origin,
                stages=[dict(stage="text_encode", start_ms=0., end_ms=.01),
                        dict(stage="denoise_step_submitted", at_ms=.01)])
        result = run_trace(context(), [request(0)], hooks(generate_images=images))
        stages = [e for e in result["events"] if e["kind"] == "image_stage"]
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["stage"], "text_encode")


class AttributionTests(unittest.TestCase):
    def test_idle_log_is_bounded_and_snapshots_do_not_change(self):
        log = EventLog(time.perf_counter())
        log.emit("scheduler_idle", 0, 1, reason="waiting_encoding")
        snapshot = log.snapshot()
        for index in range(1, 1000):
            log.emit("scheduler_idle", index, index + 1, reason="waiting_encoding")
        self.assertEqual(len(log.events), 1)
        self.assertEqual(log.events[0]["end_ms"], 1000)
        self.assertEqual(snapshot[0]["end_ms"], 1)

    def test_encoding_trigger_and_tail_finish_union(self):
        rows = []
        for index in range(20):
            row = request(index)
            row.update(status="completed", start_ms=100 - index, finish_ms=110 - index)
            rows.append(row)
        video = request(20, "video")
        video.update(status="completed", start_ms=0, finish_ms=200)
        rows.append(video)
        events = [dict(kind="scheduler_idle", reason="waiting_encoding", start_ms=0, end_ms=10),
                  dict(kind="encode_service", request_id="r020", status="completed", start_ms=0, end_ms=50),
                  dict(kind="video_finish", start_ms=20, end_ms=40),
                  dict(kind="video_finish", start_ms=30, end_ms=50)]
        value = attribution(rows, events, 200)
        self.assertTrue(value["encoding_trigger"])
        self.assertEqual(value["encoding_video_e2e_fraction"], .25)
        self.assertEqual(value["tail_image_count"], 1)
        self.assertEqual(value["tail_finish_overlap_ms"], 30)
        self.assertTrue(value["vae_trigger"])

    def test_finish_observer_preserves_original_calls_and_restores_on_failure(self):
        calls, events = [], []
        worker = SimpleNamespace()
        worker._sync = lambda: calls.append("sync")
        worker.pipe = SimpleNamespace(vae=SimpleNamespace(decode=lambda x: calls.append("decode") or x),
                                     video_processor=SimpleNamespace(postprocess_video=lambda x: calls.append("convert") or x))
        def finish(state):
            worker._sync()
            x = worker.pipe.vae.decode(1)
            worker.pipe.video_processor.postprocess_video(x)
            worker._sync()
            raise RuntimeError("after exact tail")
        worker.finish = finish
        original = worker._sync
        with self.assertRaisesRegex(RuntimeError, "exact tail"):
            instrumented_finish(worker, SimpleNamespace(request_id="r000"),
                                lambda kind, **fields: events.append(dict(kind=kind, **fields)))
        self.assertIs(worker._sync, original)
        self.assertEqual(calls, ["sync", "decode", "convert", "sync"])
        self.assertEqual([e["kind"] for e in events], ["video_sync", "video_decode", "video_convert", "video_sync"])


class PairedGateTests(unittest.TestCase):
    def gate(self, rows):
        return paired_gate(rows, expected_trace_ids={row["trace_sha256"] for row in rows})

    def pair(self, candidate_image_ms=8., trace=0):
        initial = [request(trace * 2, "video"), request(trace * 2 + 1)]
        result = []
        for order in (0, 1):
            for arm in (("baseline", "candidate") if order == 0 else ("candidate", "baseline")):
                rows = copy.deepcopy(initial)
                for row in rows:
                    row.update(status="completed", arrived=True, start_ms=0.,
                               finish_ms=100. if row["kind"] == "video" else
                               10. if arm == "baseline" else candidate_image_ms)
                result.append(dict(trace_sha256=request_identity(initial), order_index=order, arm=arm,
                                   requests=rows, complete=True, status="completed", elapsed_ms=100.))
        return result

    def test_paired_gain_uses_all_four_runs_and_per_trace_safety(self):
        value = self.gate(self.pair() + self.pair(trace=1))
        self.assertTrue(value["passed"])
        self.assertEqual(value["sample_count"], 8)
        self.assertEqual(len(value["traces"]), 2)
        self.assertAlmostEqual(value["pooled"]["image_p95_reduction"], .2)

    def test_missing_order_or_failed_member_never_passes(self):
        rows = self.pair()
        self.assertFalse(self.gate(rows[:-1])["passed"])
        rows[-1]["requests"][0]["status"] = "failed"
        self.assertFalse(self.gate(rows)["passed"])

    def test_wrong_identity_and_duplicate_never_pass(self):
        rows = self.pair()
        rows[-1]["requests"][0]["seed"] += 1
        self.assertFalse(self.gate(rows)["passed"])
        rows = self.pair()
        self.assertFalse(self.gate(rows + [rows[0]])["passed"])

    def test_one_trace_regression_cannot_be_hidden_by_another_gain(self):
        value = self.gate(self.pair(candidate_image_ms=1.) + self.pair(candidate_image_ms=10.6, trace=1))
        self.assertFalse(value["passed"])
        self.assertFalse(value["traces"][1]["safe"])

    def test_absent_expected_trace_is_not_a_passing_subset(self):
        rows = self.pair()
        value = paired_gate(rows, expected_trace_ids={rows[0]["trace_sha256"], "missing"})
        self.assertFalse(value["passed"])
        self.assertTrue(any("complete frozen trace set" in reason for reason in value["failures"]))


if __name__ == "__main__":
    unittest.main()
