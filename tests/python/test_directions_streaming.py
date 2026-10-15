import time
import threading
import unittest
from collections import deque

from ditserve.directions.streaming import (
    Arrivals, FixedRatePlayer, PromptEvent, PromptQueue, WallClockPlayer, condition_metrics, work_accounting,
)


class StreamingTests(unittest.TestCase):
    @staticmethod
    def chunk_frames(chunk, count, version=0):
        return [{"frame_id": "%d-%d" % (chunk, index), "chunk": chunk,
                 "prompt_version": version, "pixels": object()} for index in range(count)]

    def test_chunk_limits_count_first_nine_then_twelve_and_hold_last_displayed_frame(self):
        for limit in (0, 1, 2):
            with self.subTest(limit=limit):
                player = FixedRatePlayer()
                player.reserve_chunk(limit, 0, 0.)
                player.enqueue(self.chunk_frames(0, 9), 0.)
                for chunk in range(1, limit + 1):
                    self.assertTrue(player.may_generate(limit, 12, 0.))
                    player.reserve_chunk(limit, chunk, 0.)
                    player.enqueue(self.chunk_frames(chunk, 12), 0.)
                self.assertFalse(player.may_generate(limit, 12, 0.))
                with self.assertRaisesRegex(RuntimeError, "capacity is full"):
                    player.reserve_chunk(limit, limit + 1, 0.)
                report = player.report()
                self.assertEqual(report["peak_resident_chunks"], limit + 1)
                self.assertEqual(report["peak_ahead_chunks"], limit)
                self.assertEqual(report["peak_queued_frames"], 9 if limit == 0 else 8 + 12 * limit)
                self.assertTrue(report["current_chunk_excluded_from_buffer_limit"])
                # The last first-chunk frame is displayed at 8/16, and owns
                # its slot until 9/16 even if no frame remains in that chunk.
                self.assertFalse(player.may_generate(limit, 12, 8 / 16))
                self.assertTrue(player.may_generate(limit, 12, 9 / 16))

    def test_whole_chunk_reservation_cannot_be_overbooked_or_hidden_by_flush(self):
        player = FixedRatePlayer()
        player.reserve_chunk(0, 0, 0.)
        self.assertFalse(player.may_generate(0, 12, 0.))
        with self.assertRaisesRegex(RuntimeError, "reserved playback slot"):
            player.enqueue(self.chunk_frames(1, 12), 0.)
        player.cancel_reservation(0)
        self.assertTrue(player.may_generate(0, 12, 0.))
        player.reserve_chunk(0, 0, 0.)
        player.enqueue(self.chunk_frames(0, 9), 0.)
        player.discard_older(1, .01)
        self.assertEqual(len(player.dropped), 8)
        self.assertFalse(player.may_generate(0, 12, .01))
        self.assertTrue(player.may_generate(0, 12, 1 / 16))
        self.assertEqual(player.report()["reserved_chunks"], 0)

    def test_real_wall_clock_counts_chunk_inside_sink_and_enforces_prefetch_capacity(self):
        entered, release = threading.Event(), threading.Event()
        def sink(frame):
            entered.set()
            if not release.wait(2.):
                raise AssertionError("test sink did not release")
        player = WallClockPlayer(time.monotonic(), sink=sink)
        try:
            player.reserve_chunk(2, 0)
            player.enqueue(self.chunk_frames(0, 9), 0.)
            self.assertTrue(entered.wait(1.))
            for chunk in (1, 2):
                player.reserve_chunk(2, chunk)
                player.enqueue(self.chunk_frames(chunk, 12), player._now())
            self.assertFalse(player.may_generate(2, 12))
            with self.assertRaisesRegex(RuntimeError, "capacity is full"):
                player.reserve_chunk(2, 3)
            report = player.report()
            self.assertEqual(report["peak_resident_chunks"], 3)
            self.assertEqual(report["peak_ahead_chunks"], 2)
            self.assertEqual(report["peak_queued_frames"], 32)
            player.discard_older(1, player._now())
            self.assertEqual(player.report()["resident_chunks"], 1)
            self.assertEqual(player.report()["reserved_chunks"], 0)
        finally:
            release.set()
            player.close(drain=False)

    def test_future_prompt_payload_is_not_read(self):
        class Event:
            arrival_s, version, session_id = 10., 1, "s0"
            @property
            def prompt(self):
                raise AssertionError("future prompt must not be inspected")
        arrivals = Arrivals([Event()])
        self.assertEqual(arrivals.collect(9.99), [])
        self.assertFalse(arrivals.complete)

    def test_playback_slack_includes_current_frame_until_its_display_deadline(self):
        from ditserve.directions.direction3 import choose_session
        def player(deadline):
            # Stop the clock to reproduce equal queue depths with distinct
            # real presentation deadlines, without racing a consumer thread.
            value = object.__new__(WallClockPlayer)
            value._condition = threading.Condition()
            value.queue, value.fps = deque([object()] * 6), 16.
            value.next_due, value._now = deadline, lambda: 1.
            return value
        players = {"s0": player(1.060), "s1": player(1.005)}
        self.assertAlmostEqual(players["s0"].buffered_seconds(), .435)
        self.assertAlmostEqual(players["s1"].buffered_seconds(), .380)
        selected, _ = choose_session(["s0", "s1"], ["s0", "s1"], players, 0,
                                     "minimum_playback_slack")
        self.assertEqual(selected, "s1")
        for deadline in (None, .9):
            value = player(deadline)
            self.assertAlmostEqual(value.buffered_seconds(), .375)
            value.queue.clear()
            self.assertEqual(value.buffered_seconds(), 0.)

    def test_latest_only_preserves_superseded_denominator(self):
        queue = PromptQueue("latest_only")
        for version in range(3):
            queue.arrive(PromptEvent(version, float(version), "p%d" % version), 3.)
        self.assertEqual(queue.take().version, 2)
        records = queue.finish()
        self.assertEqual(len(records), 3)
        self.assertEqual([r["status"] for r in records[:2]], ["superseded_before_encoding"] * 2)

    def test_fifo_preserves_all_updates(self):
        queue = PromptQueue("fifo")
        for version in range(3):
            queue.arrive(PromptEvent(version, 0., "p"), 0.)
        self.assertEqual([queue.take().version for _ in range(3)], [0, 1, 2])

    def test_reject_future_encoding(self):
        queue = PromptQueue("fifo")
        event = PromptEvent(1, 2., "later")
        queue.arrive(event, 2.)
        with self.assertRaises(ValueError):
            queue.applied(event, 1.9, 2.1, 2.2)

    def test_fixed_rate_stalls_do_not_slow_playback(self):
        player = FixedRatePlayer()
        frame = {"frame_id": 1, "prompt_version": 0}
        player.enqueue([frame, frame], 1.)
        player.advance(2.)
        player.enqueue([dict(frame, prompt_version=1)], 2.)
        report = player.finish()
        self.assertEqual([r["presented_s"] for r in report["presentations"]], [1., 1.0625, 2.])
        self.assertAlmostEqual(report["stall_seconds"], .875)
        self.assertEqual(report["fps"], 16.)

    def test_flush_never_retracts_presented_frames(self):
        player = FixedRatePlayer()
        player.enqueue([{"frame_id": i, "prompt_version": 0} for i in range(3)], 0.)
        player.discard_older(1, .01)
        self.assertEqual(len(player.presentations), 1)
        self.assertEqual(len(player.dropped), 2)

    def test_condition_tag_is_not_semantic_success(self):
        report = {"presentations": [{"prompt_version": 1, "released_s": 2., "presented_s": 3.}]}
        row = condition_metrics([{"version": 1, "arrival_s": 1.}], report)[0]
        self.assertEqual(row["conditioned_playout_latency_s"], 2.)
        self.assertIsNone(row["semantic_latency_s"])
        self.assertIn("adjudication", row["semantic_status"])

    def test_decoded_and_flushed_condition_retains_decode_latency(self):
        player = {"presentations": [], "dropped": [{"chunk": 0}]}
        chunks = [{"block": 0, "prompt_version": 1, "released_s": 2.,
                   "pixel_frames": 1, "denoise_context_s": .8}]
        row = condition_metrics([{"version": 1, "arrival_s": 1.}], player, chunks)[0]
        self.assertEqual(row["conditioned_decode_latency_s"], 1.)
        self.assertNotIn("first_conditioned_playout_s", row)
        accounting = work_accounting(chunks, [row], player)
        self.assertEqual(accounting["fully_flushed_chunks"], 1)
        self.assertEqual(accounting["fully_flushed_denoise_context_s"], .8)

    def test_partial_chunk_waste_is_an_explicit_estimate(self):
        player = {"presentations": [{"chunk": 0}], "dropped": [{"chunk": 0}]}
        chunks = [{"block": 0, "prompt_version": 0, "pixel_frames": 2, "denoise_context_s": 1.}]
        row = work_accounting(chunks, [], player)
        self.assertEqual(row["fully_flushed_chunks"], 0)
        self.assertEqual(row["partially_flushed_work_estimate_s"], .5)
        self.assertIn("not_independent", row["partial_chunk_estimate"])

    def test_wall_clock_thread_consumes_real_objects_and_reports_time(self):
        consumed = []
        origin = time.monotonic()
        player = WallClockPlayer(origin, sink=lambda frame: consumed.append(frame["pixels"]))
        player.enqueue([{"frame_id": i, "prompt_version": 0, "pixels": object()} for i in range(3)], 0.)
        report = player.close()
        self.assertEqual(len(consumed), 3)
        times = [r["presented_s"] for r in report["presentations"]]
        self.assertGreaterEqual(times[2] - times[0], .12)
        self.assertLessEqual(times[-1], time.monotonic() - origin)
        self.assertTrue(all("pixels" not in r for r in report["presentations"]))
        self.assertIn("actual_headless", report["scope"])

    def test_wall_clock_window_censors_unconsumed_frames(self):
        origin = time.monotonic()
        player = WallClockPlayer(origin, stop_at_s=.08)
        player.enqueue([{"frame_id": i, "prompt_version": 0, "pixels": object()} for i in range(20)], 0.)
        report = player.close()
        self.assertGreater(report["unconsumed_frames"], 0)
        self.assertTrue(all(r["presented_s"] < .08 for r in report["presentations"]))

    def test_wall_clock_counts_starvation_through_observation_end(self):
        origin = time.monotonic()
        player = WallClockPlayer(origin, stop_at_s=.13)
        player.enqueue([{"frame_id": 0, "prompt_version": 0, "pixels": object()}], 0.)
        time.sleep(.14)
        report = player.close(drain=False)
        self.assertGreater(report["stall_seconds"], .04)
        self.assertTrue(report["stalls"][-1]["ongoing_at_observation_end"])
        self.assertAlmostEqual(report["stalls"][-1]["end_s"], .13)
        self.assertFalse(report["explicit_eos"])


if __name__ == "__main__":
    unittest.main()
