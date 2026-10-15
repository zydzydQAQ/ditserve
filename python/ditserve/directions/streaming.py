"""Small, CPU-only primitives for measured streaming experiments.

FixedRatePlayer supplies deterministic simulated playout for CPU tests.
WallClockPlayer independently consumes real decoded frames and samples actual
wall-clock timestamps. Neither measures a browser/display. Prompt provenance
establishes conditioning, not semantic adherence.
"""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class PromptEvent:
    version: int
    arrival_s: float
    prompt: str
    session_id: str = "s0"


class Arrivals:
    """Reveal event payloads only when their frozen wall-clock arrival is due."""

    def __init__(self, events):
        self._events = sorted(events, key=lambda e: (e.arrival_s, e.version))
        self._cursor = 0
        identities = [(e.session_id, e.version) for e in self._events]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate prompt version within session")
        if any(not math.isfinite(e.arrival_s) or e.arrival_s < 0 for e in self._events):
            raise ValueError("invalid prompt arrival")

    def collect(self, now):
        ready = []
        while self._cursor < len(self._events):
            event = self._events[self._cursor]
            if event.arrival_s > now:
                break
            ready.append(event)
            self._cursor += 1
        return ready

    @property
    def complete(self):
        return self._cursor == len(self._events)


class PromptQueue:
    """FIFO or latest-only admission; every superseded event remains evidence."""

    def __init__(self, policy):
        if policy not in ("fifo", "latest_only", "latest_only_flush"):
            raise ValueError("unknown prompt policy")
        self.policy = policy
        self.pending = deque()
        self.records = {}

    def arrive(self, event, observed_s):
        if event.version in self.records:
            raise ValueError("prompt version was already observed")
        self.records[event.version] = {
            "version": event.version, "arrival_s": event.arrival_s,
            "observed_s": observed_s, "status": "pending", "prompt": event.prompt,
            "semantic_status": "unjudged",
        }
        if self.policy != "fifo":
            while self.pending:
                old = self.pending.popleft()
                self.records[old.version].update(status="superseded_before_encoding",
                                                 superseded_by=event.version)
        self.pending.append(event)

    def take(self):
        return self.pending.popleft() if self.pending else None

    def applied(self, event, encode_start_s, encode_end_s, applied_s):
        if encode_start_s < event.arrival_s:
            raise ValueError("future prompt was encoded")
        self.records[event.version].update(status="applied", encode_start_s=encode_start_s,
            encode_end_s=encode_end_s, condition_applied_s=applied_s,
            condition_latency_s=applied_s - event.arrival_s)

    def finish(self):
        while self.pending:
            event = self.pending.popleft()
            self.records[event.version]["status"] = "pending_at_stop"
        return [self.records[key] for key in sorted(self.records)]


class _ChunkSlots:
    """One current/startup chunk plus B prefetched chunks, including reservations.

    The caller serializes access. Chunk identity, not a nominal pixel count,
    determines occupancy; the final displayed frame retains its slot for 1/FPS.
    """

    semantics = "B prefetched chunks beyond one current/startup chunk; generation reserves one whole slot; current frame holds its slot through its display interval"

    def __init__(self):
        self.limit = None
        self.reserved = set()
        self.current = None
        self.current_until = None
        self.peak_resident = self.peak_ahead = 0

    def resident(self, queue, now):
        if self.current is not None and self.current_until is not None and now >= self.current_until:
            self.current = self.current_until = None
        chunks = {frame.get("chunk") for frame in queue}
        if self.current is not None:
            chunks.add(self.current["chunk"])
        return chunks

    def may_generate(self, queue, buffer_chunks, now):
        if type(buffer_chunks) is not int or buffer_chunks not in (0, 1, 2):
            raise ValueError("buffer_chunks must be 0, 1, or 2")
        if self.limit is not None and self.limit != buffer_chunks:
            raise ValueError("a player's buffer limit cannot change during an arm")
        return len(self.resident(queue, now)) + len(self.reserved) + 1 <= buffer_chunks + 1

    def reserve(self, queue, buffer_chunks, chunk, now):
        if chunk in self.resident(queue, now) or chunk in self.reserved:
            raise ValueError("chunk already occupies a playback slot")
        if not self.may_generate(queue, buffer_chunks, now):
            raise RuntimeError("playback chunk capacity is full")
        self.limit = buffer_chunks
        self.reserved.add(chunk)

    def accept(self, queue, frames, now):
        if self.limit is not None:
            chunks = {frame["chunk"] for frame in frames}
            if len(chunks) != 1 or not chunks <= self.reserved:
                raise RuntimeError("decoded chunk requires its own reserved playback slot")
            occupied = self.resident(queue, now) | chunks
            if len(occupied) + len(self.reserved - chunks) > self.limit + 1:
                raise RuntimeError("decoded chunk would exceed playback capacity")
            self.reserved.difference_update(chunks)

    def dispatched(self, frame):
        self.current = {"chunk": frame.get("chunk")}
        self.current_until = None  # A frame inside the sink still owns its slot.

    def presented(self, until):
        self.current_until = until

    def observe(self, queue, now):
        count = len(self.resident(queue, now))
        self.peak_resident = max(self.peak_resident, count)
        self.peak_ahead = max(self.peak_ahead, max(0, count - 1))

    def report(self, queue, now):
        count = len(self.resident(queue, now))
        return {"buffer_semantics": self.semantics, "buffer_chunks": self.limit,
                "current_chunk_excluded_from_buffer_limit": True,
                "resident_chunks": count, "ahead_chunks": max(0, count - 1),
                "reserved_chunks": len(self.reserved),
                "peak_resident_chunks": self.peak_resident, "peak_ahead_chunks": self.peak_ahead}


class FixedRatePlayer:
    """Deterministic playout of released frames, with visible stalls and drops.

    No interpolation, slow-motion, catch-up, or hidden FPS adjustment occurs.
    ``advance`` can run after a long GPU call: queued frames retain their fixed
    presentation deadlines, while unavailable frames cause a recorded stall.
    """

    def __init__(self, fps=16.0):
        if fps != 16.0:
            raise ValueError("the frozen freshness experiment plays at 16 FPS")
        self.fps = fps
        self.queue = deque()
        self.presentations = []
        self.dropped = []
        self.stalls = []
        self.next_due = None
        self.last_now = 0.0
        self.peak_queued_frames = 0
        self._slots = _ChunkSlots()

    def advance(self, now):
        if now < self.last_now:
            raise ValueError("playout clock moved backwards")
        self.last_now = now
        while self.queue and self.next_due <= now:
            frame = self.queue.popleft()
            due = max(self.next_due, frame["released_s"])
            if due > now:
                self.queue.appendleft(frame)
                break
            if due > self.next_due:
                self.stalls.append({"start_s": self.next_due, "end_s": due,
                                    "duration_s": due - self.next_due})
            self.presentations.append(dict(frame, presented_s=due))
            self.next_due = due + 1.0 / self.fps
            self._slots.dispatched(frame)
            self._slots.presented(self.next_due)
        self._slots.observe(self.queue, now)

    def enqueue(self, frames, released_s):
        self.advance(released_s)
        frames = list(frames)
        self._slots.accept(self.queue, frames, released_s)
        for frame in frames:
            self.queue.append(dict(frame, released_s=released_s))
        if self.next_due is None and self.queue:
            self.next_due = released_s
        self.peak_queued_frames = max(self.peak_queued_frames, len(self.queue))
        self._slots.observe(self.queue, released_s)
        self.advance(released_s)

    def discard_older(self, version, now):
        self.advance(now)
        retained = deque()
        while self.queue:
            frame = self.queue.popleft()
            if frame["prompt_version"] < version:
                self.dropped.append(dict(frame, dropped_s=now, reason="superseded_prompt"))
            else:
                retained.append(frame)
        self.queue = retained

    def may_generate(self, buffer_chunks, chunk_frames, now):
        self.advance(now)
        return self._slots.may_generate(self.queue, buffer_chunks, now)

    def reserve_chunk(self, buffer_chunks, chunk, now):
        self.advance(now)
        self._slots.reserve(self.queue, buffer_chunks, chunk, now)

    def cancel_reservation(self, chunk):
        self._slots.reserved.discard(chunk)

    def finish(self):
        if self.queue:
            self.advance(max(self.last_now, self.next_due) + len(self.queue) / self.fps)
        return self.report()

    def report(self):
        return {"scope": "simulated_fixed_rate_playout_over_measured_release_times",
            "fps": self.fps, "presentations": list(self.presentations),
            "dropped": list(self.dropped), "stalls": list(self.stalls),
            "stall_seconds": sum(s["duration_s"] for s in self.stalls),
            "peak_queued_frames": self.peak_queued_frames,
            **self._slots.report(self.queue, self.last_now)}


def condition_metrics(prompt_records, player_report, chunks=None):
    """Attach condition provenance latency; never infer semantic effectiveness."""
    frames = player_report["presentations"]
    result = []
    for record in prompt_records:
        row = dict(record)
        matches = [f for f in frames if f["prompt_version"] == row["version"]]
        decoded = ([chunk for chunk in chunks if chunk["prompt_version"] == row["version"]]
                   if chunks is not None else matches)
        if decoded:
            row["first_conditioned_decode_s"] = min(chunk["released_s"] for chunk in decoded)
            row["first_conditioned_output_s"] = row["first_conditioned_decode_s"]
            row["conditioned_decode_latency_s"] = row["first_conditioned_decode_s"] - row["arrival_s"]
        if matches:
            row["first_conditioned_playout_s"] = min(f["presented_s"] for f in matches)
            row["conditioned_playout_latency_s"] = row["first_conditioned_playout_s"] - row["arrival_s"]
        row["semantic_status"] = "requires_blinded_video_adjudication"
        row["semantic_latency_s"] = None
        result.append(row)
    return result


def work_accounting(chunks, updates, player_report):
    """Account measured chunk work separately from proportional frame estimates."""
    rows = []
    for chunk in chunks:
        block = chunk["block"]
        count = chunk["pixel_frames"]
        played = sum(frame.get("chunk") == block for frame in player_report["presentations"])
        flushed = sum(frame.get("chunk") == block for frame in player_report["dropped"])
        unplayed = max(0, count - played)
        cost = chunk.get("denoise_context_s", 0.)
        rows.append({"block": block, "prompt_version": chunk["prompt_version"],
            "pixel_frames": count, "played_frames": played, "flushed_frames": flushed,
            "unplayed_at_window_frames": unplayed,
            "denoise_context_wall_s": cost,
            "fully_unplayed_at_window": unplayed == count,
            "fully_flushed": flushed == count,
            "flushed_work_proportional_estimate_s": cost * flushed / count,
            "unplayed_work_proportional_estimate_s": cost * unplayed / count})
    overwritten = []
    for update in updates:
        if "condition_applied_s" not in update:
            continue
        produced = any(c["prompt_version"] == update["version"] for c in chunks)
        newer = any(u["version"] > update["version"] and "condition_applied_s" in u for u in updates)
        if not produced:
            overwritten.append({"version": update["version"], "superseded_by_applied_update": newer,
                "encode_wall_s": update["encode_end_s"] - update["encode_start_s"],
                "recache_wall_s": update.get("recache_s", 0.),
                "reason": "overwritten_without_decoded_chunk" if newer else "window_cutoff_without_decoded_chunk"})
    return {"scope": "synchronized_denoise_context_wall_seconds_not_GPU_kernel_event_time",
        "partial_chunk_estimate": "frame_fraction_times_joint_chunk_cost_not_independent_frame_compute",
        "window_censoring": "unplayed_at_cutoff_is_not_claimed_permanently_wasted",
        "chunks": rows,
        "fully_flushed_chunks": sum(row["fully_flushed"] for row in rows),
        "fully_flushed_denoise_context_s": sum(row["denoise_context_wall_s"] for row in rows if row["fully_flushed"]),
        "fully_unplayed_at_window_chunks": sum(row["fully_unplayed_at_window"] for row in rows),
        "fully_unplayed_at_window_denoise_context_s": sum(row["denoise_context_wall_s"] for row in rows if row["fully_unplayed_at_window"]),
        "partially_flushed_work_estimate_s": sum(row["flushed_work_proportional_estimate_s"] for row in rows if not row["fully_flushed"]),
        "partially_unplayed_work_estimate_s": sum(row["unplayed_work_proportional_estimate_s"] for row in rows if not row["fully_unplayed_at_window"]),
        "conditioning_without_decoded_output": overwritten}


class WallClockPlayer:
    """Independent headless client consuming decoded frames at a 16 FPS target.

    Consumption timestamps are actual monotonic-clock samples. Browser, network
    transport, compositor and monitor latency remain outside this experiment.
    """

    def __init__(self, origin, fps=16.0, sink=None, clock=time.monotonic, stop_at_s=None):
        if fps != 16.0:
            raise ValueError("the frozen freshness experiment plays at 16 FPS")
        self.origin, self.clock, self.fps = origin, clock, fps
        self.stop_at_s = stop_at_s
        self.sink = sink or (lambda frame: None)
        self.queue = deque()
        self.presentations, self.dropped, self.stalls = [], [], []
        self.next_due = None
        self.empty_since = None
        self.peak_queued_frames = 0
        self._slots = _ChunkSlots()
        self._condition = threading.Condition()
        self._stop = self._drain = False
        self._error = None
        self._thread = threading.Thread(target=self._run, name="dit-freshness-player", daemon=True)
        self._thread.start()

    def _now(self):
        return self.clock() - self.origin

    def _run(self):
        try:
            while True:
                with self._condition:
                    if self._stop or (self._drain and not self.queue):
                        return
                    now = self._now()
                    if self.stop_at_s is not None and now >= self.stop_at_s:
                        return
                    if not self.queue:
                        if self.next_due is not None and now >= self.next_due and self.empty_since is None:
                            self.empty_since = self.next_due
                        self._condition.wait(0.01)
                        continue
                    if self.next_due is not None and now < self.next_due:
                        deadline = self.next_due if self.stop_at_s is None else min(self.next_due, self.stop_at_s)
                        self._condition.wait(deadline - now)
                        continue
                    frame = self.queue.popleft()
                    self._slots.dispatched(frame)
                    due = now if self.next_due is None else self.next_due
                    if self.empty_since is not None:
                        self.stalls.append({"start_s": self.empty_since, "end_s": now,
                                            "duration_s": now - self.empty_since})
                        self.empty_since = None
                self.sink(frame)  # Consumes an actual decoded CPU frame.
                presented = self._now()
                with self._condition:
                    row = {k: v for k, v in frame.items() if k != "pixels"}
                    self.presentations.append(dict(row, presented_s=presented,
                                                   target_s=due, jitter_s=max(0.0, presented - due)))
                    # No catch-up bursts or playback-speed adaptation.
                    self.next_due = presented + 1.0 / self.fps
                    self._slots.presented(self.next_due)
                    self._slots.observe(self.queue, presented)
                    self._condition.notify_all()
        except BaseException as error:
            self._error = error

    def _check(self):
        if self._error is not None:
            raise RuntimeError("headless player failed") from self._error

    def enqueue(self, frames, released_s):
        self._check()
        with self._condition:
            if self._stop or self._drain:
                raise RuntimeError("player has closed")
            frames = list(frames)
            self._slots.accept(self.queue, frames, self._now())
            if not self.queue and self.next_due is not None and released_s > self.next_due:
                self.empty_since = self.next_due if self.empty_since is None else self.empty_since
            self.queue.extend(dict(frame, released_s=released_s) for frame in frames)
            self.peak_queued_frames = max(self.peak_queued_frames, len(self.queue))
            self._slots.observe(self.queue, self._now())
            self._condition.notify_all()

    def discard_older(self, version, now):
        with self._condition:
            retained = deque()
            while self.queue:
                frame = self.queue.popleft()
                if frame["prompt_version"] < version:
                    row = {k: v for k, v in frame.items() if k != "pixels"}
                    self.dropped.append(dict(row, dropped_s=now, reason="superseded_prompt"))
                else:
                    retained.append(frame)
            self.queue = retained

    def may_generate(self, buffer_chunks, chunk_frames, now=None):
        self._check()
        with self._condition:
            return self._slots.may_generate(self.queue, buffer_chunks, self._now())

    def reserve_chunk(self, buffer_chunks, chunk):
        self._check()
        with self._condition:
            if self._stop or self._drain:
                raise RuntimeError("player has closed")
            self._slots.reserve(self.queue, buffer_chunks, chunk, self._now())

    def cancel_reservation(self, chunk):
        with self._condition:
            self._slots.reserved.discard(chunk)
            self._condition.notify_all()

    def buffered_seconds(self):
        with self._condition:
            current_remaining = max(0., self.next_due - self._now()) if self.next_due is not None else 0.
            return current_remaining + len(self.queue) / self.fps

    def close(self, check_budget=lambda: None, drain=True):
        with self._condition:
            self._drain = drain
            self._stop = not drain
            self._condition.notify_all()
        try:
            while self._thread.is_alive():
                check_budget()
                self._thread.join(0.05)
        finally:
            self._stop = True
            with self._condition:
                self._condition.notify_all()
            self._thread.join(0.2)
        self._check()
        return self.report()

    def report(self):
        with self._condition:
            stalls = list(self.stalls)
            end = self._now() if self.stop_at_s is None else min(self._now(), self.stop_at_s)
            # With no explicit EOS during the experiment, an empty tail is
            # starvation, including a frame that never arrives before cutoff.
            start = self.empty_since
            if start is None and not self.queue and self.next_due is not None:
                start = self.next_due
            if start is not None and end > start:
                stalls.append({"start_s": start, "end_s": end, "duration_s": end - start,
                               "ongoing_at_observation_end": True})
            return {"scope": "actual_headless_wall_clock_frame_consumption_no_browser_or_network",
                "fps": self.fps, "presentations": list(self.presentations),
                "dropped": list(self.dropped), "stalls": stalls,
                "stall_seconds": sum(s["duration_s"] for s in stalls),
                "startup_wait_seconds": (self.presentations[0]["presented_s"] if self.presentations else end),
                "peak_queued_frames": self.peak_queued_frames, "unconsumed_frames": len(self.queue),
                "observation_end_s": self.stop_at_s, "explicit_eos": False,
                **self._slots.report(self.queue, end)}
