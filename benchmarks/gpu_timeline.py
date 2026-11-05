#!/usr/bin/env python3
"""Controller-level (L1) per-GPU timeline and bubble decomposition of native run records.

Every GPU's measurement window ``[0, cell.elapsed_ms]`` (the controller replay clock of one
(phase, attempt, cell, order) observation) is split into mutually exclusive categories whose
durations sum exactly to the window.  Only controller-recorded timestamps are used:

* task occupancy of a GPU: ``controller_timing[*]`` interval ``[selected_ms, observed_ms]`` on
  ``group_after`` (the controller marks the GPU busy from launch until it observes completion);
* own work inside that occupancy: ``stages`` whose ``worker`` is this GPU's video/image worker
  (controller-observed RPC wall per rank); atomic image service and video ``finish`` (VAE decode
  plus output digest) carry no stages, so their ``[async_begin_ms, all_done_ms]`` is used;
* queue state: request arrival, video text-encode readiness (``encode_ready_ms``), request
  occupancy and completion.

Categories (priority order, first match wins for a GPU at an instant):

1. ``video_compute`` | ``image_compute`` | ``decode`` | ``prepare`` | ``transfer_export`` |
   ``transfer_import`` | ``other_stage`` — an own-rank stage / atomic body is in flight.
2. ``unknown`` — the GPU is occupied but its work cannot be determined from the record
   (aborted/failed controller events, stages on unmapped workers, requests that ran without a
   controller_timing event).
3. ``held_wait_peer`` — occupied by a task, no own stage, but another rank's stage of the same task
   is running (e.g. the idle CFG partner waits for rank-0 prepare/export or another rank's import).
4. ``held_overhead`` — occupied by a task with no stage running anywhere in it: launch, RPC
   dispatch, rank skew and the controller noticing completion.
5. ``held_midstep_owner`` — not occupied, but a request suspended inside a step (``block_after>0``)
   owns this GPU's group until it resumes.
6. ``idle_with_queue`` — not occupied and at least one arrived, text-ready request is waiting or
   suspended (a scheduling bubble).  Sub-causes:
   ``turnaround_same_request`` (the previous and next task on this GPU belong to the same request:
   inter-quantum/step-boundary controller turnaround), ``dispatchable_video`` /
   ``dispatchable_image_only`` (enough GPUs are free for some waiting request, yet none was
   launched: controller latency or policy hold, e.g. image partitioning), and
   ``gang_wait_partner_<label>`` (no waiting request fits the free GPUs; the partner GPU is busy
   with ``<label>``, e.g. a single-GPU VAE decode blocking a CFG2 video gang).
7. ``idle_encode_wait`` — not occupied, nothing ready, but some arrived video is still waiting for
   its CPU text encoding to be observed ready.
8. ``idle_empty`` — not occupied and no arrived unfinished request is waiting.

These are controller observations, not hardware utilization: an ``advance`` stage is RPC wall and
includes collectives, Python and launch overhead; ``decode`` includes CPU output hashing.  Kernel
time (``block_cuda_ms``) is reported only if a record actually contains it (L2 supplement);
otherwise it is ``unknown``.
"""

import argparse
import bisect
import errno
import json
import math
import os
import time
from pathlib import Path

SCHEMA = "ditserve.gpu-timeline.v1"
WORK_LABELS = ("video_compute", "image_compute", "decode", "prepare", "transfer_export",
               "transfer_import", "other_stage")
CATEGORIES = WORK_LABELS + ("unknown", "held_wait_peer", "held_overhead", "held_midstep_owner",
                            "idle_with_queue", "idle_encode_wait", "idle_empty")
HELD = ("held_wait_peer", "held_overhead", "held_midstep_owner")
STAGE_LABEL = {"advance": "video_compute", "prepare": "prepare", "export": "transfer_export",
               "import": "transfer_import", "tiered_offload": "other_stage",
               "tiered_prefetch": "other_stage"}
PRIORITY = {label: 0 for label in WORK_LABELS}
PRIORITY.update({"unknown": 1, "held_wait_peer": 2, "held_overhead": 3})
TIE_ORDER = {label: index for index, label in enumerate(CATEGORIES)}


def read_json(path, chunk=1 << 16, retries=400, pause=0.5):
    """Read a JSON file in small chunks, retrying transient ENOMEM from drvfs mounts."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        buffer, offset, failures = bytearray(), 0, 0
        while offset < size:
            try:
                block = os.pread(fd, min(chunk, size - offset), offset)
            except OSError as error:
                if error.errno != errno.ENOMEM or failures >= retries:
                    raise
                failures += 1
                time.sleep(pause)
                continue
            if not block:
                break
            buffer += block
            offset += len(block)
    finally:
        os.close(fd)
    return json.loads(bytes(buffer))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def merge(intervals):
    result = []
    for start, finish in sorted(intervals):
        if finish <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], finish)
        else:
            result.append([start, finish])
    return result


def subtract(interval, holes):
    start, finish = interval
    pieces, cursor = [], start
    for left, right in merge(holes):
        if right <= cursor or left >= finish:
            continue
        if left > cursor:
            pieces.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < finish:
        pieces.append([cursor, finish])
    return pieces


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    return {"count": len(values), "sum": sum(values) if values else 0.0,
            "p50": percentile(values, .5), "p95": percentile(values, .95),
            "max": max(values) if values else None}


def _worker_gpus(record, phase_id, cell):
    """Map worker names to GPU indices from the phase (or cell) ``gpus`` slot table."""
    slots = None
    for phase in record.get("config", {}).get("phases", []):
        if phase.get("id") == phase_id and isinstance(phase.get("gpus"), list):
            slots = phase["gpus"]
    if isinstance(cell.get("slots"), list):
        slots = cell["slots"]
    mapping = {}
    for gpu, slot in enumerate(slots or []):
        if isinstance(slot, dict):
            for name in slot.values():
                if isinstance(name, str):
                    mapping[name] = gpu
    return mapping


def cell_rows(record, phase_id, attempt, cell):
    return [row for row in record.get("requests", [])
            if row.get("phase", "cluster") == phase_id and row.get("cell") == cell.get("cell")
            and row.get("attempt", attempt) == attempt and row.get("order", cell.get("order")) == cell.get("order")]


def _event_interval(event):
    start = event.get("selected_ms", event.get("start_ms", event.get("async_begin_ms")))
    finish = event.get("observed_ms", event.get("all_done_ms"))
    if _number(start) and _number(finish) and finish >= start:
        return start, finish
    return None


def analyze_cell(rows, gpus, window_ms, worker_gpu, cell_forms=None, top=10):
    """Return the exclusive per-GPU decomposition of one cell observation."""
    if not (_number(window_ms) and window_ms > 0) or type(gpus) is not int or gpus <= 0:
        raise ValueError("cell needs a positive window and GPU count")
    W = float(window_ms)
    labels = []          # (start, finish, gpu, label)
    occupancy = []       # (start, finish, gpu, event-ref)
    diagnostics = {"events": 0, "unplaceable_events": 0, "non_completed_events": 0,
                   "unmapped_stage_workers": [], "clipped_ms": 0.0, "same_gpu_task_overlap_ms": 0.0,
                   "requests_without_events_but_started": 0, "missing_need_requests": 0}
    unmapped = set()
    request_busy = {}
    events_by_request = {}
    owners = []          # (rid, group, start, finish)
    stage_stats = {"advance_elapsed_ms": 0.0, "advance_worker_model_ms": 0.0,
                   "advance_worker_operation_ms": 0.0, "advance_stages": 0}
    decode_walls, image_walls = [], []
    turnarounds_ctrl, turnarounds_gpu = [], []
    need = {}
    kinds = {row["request_id"]: row.get("kind") for row in rows}

    def clip(start, finish):
        left, right = max(0.0, start), min(W, finish)
        diagnostics["clipped_ms"] += max(0.0, finish - start) - max(0.0, right - left)
        return left, right

    for row in rows:
        rid = row["request_id"]
        events = row.get("controller_timing") or []
        for event in events:
            diagnostics["events"] += 1
            interval = _event_interval(event)
            group = [g for g in event.get("group_after", []) if type(g) is int and 0 <= g < gpus]
            if interval is None or not group:
                diagnostics["unplaceable_events"] += 1
                continue
            start, finish = clip(*interval)
            completed = event.get("controller_status", "completed") == "completed" and event.get("status", "completed") == "completed"
            if not completed:
                diagnostics["non_completed_events"] += 1
            members = [rid] + ([event["pair_request_id"]] if event.get("pair_request_id") else [])
            for member in members:
                request_busy.setdefault(member, []).append((interval[0], interval[1]))
            events_by_request.setdefault(rid, []).append(event)
            quantum = event.get("quantum")
            if kinds.get(rid) == "video" and quantum != "finish":
                need[rid] = max(need.get(rid, 0), len(group))
            elif kinds.get(rid) == "image":
                need[rid] = max(need.get(rid, 0), len(group))
            ref = {"request_id": rid, "transition": event.get("transition"), "quantum": quantum,
                   "event_id": event.get("event_id"), "start": interval[0], "finish": interval[1],
                   "group": group}
            for gpu in group:
                if finish > start:
                    occupancy.append((start, finish, gpu, ref))
            if finish <= start:
                continue
            body = None
            if quantum == "finish":
                body = "decode"
            elif quantum == "atomic" or quantum == "continuous_batch" or event.get("transition") == "image_service":
                body = "image_compute"
            if body:
                b0, b1 = event.get("async_begin_ms"), event.get("all_done_ms")
                if _number(b0) and _number(b1) and b1 >= b0:
                    (decode_walls if body == "decode" else image_walls).append(b1 - b0)
                    b0, b1 = clip(b0, b1)
                    for gpu in group:
                        if b1 > b0:
                            labels.append((b0, b1, gpu, body if completed else "unknown"))
            event_unknown = not completed
            stage_windows = []
            for stage in event.get("stages") or []:
                worker = stage.get("worker")
                s0, s1 = stage.get("start_ms"), stage.get("finish_ms")
                if not (_number(s0) and _number(s1) and s1 >= s0):
                    event_unknown = True
                    continue
                if worker is None:
                    continue  # aggregate spans (imports_all_ready, advances_all_done)
                if worker not in worker_gpu:
                    unmapped.add(worker)
                    event_unknown = True
                    continue
                gpu = worker_gpu[worker]
                label = STAGE_LABEL.get(stage.get("stage"), "other_stage")
                if stage.get("status", "completed") != "completed":
                    label = "unknown"
                if stage.get("stage") == "advance":
                    details = stage.get("worker_details") or {}
                    stage_stats["advance_stages"] += 1
                    stage_stats["advance_elapsed_ms"] += s1 - s0
                    stage_stats["advance_worker_model_ms"] += float(details.get("model_ms", 0.0) or 0.0)
                    stage_stats["advance_worker_operation_ms"] += float(details.get("worker_operation_ms", 0.0) or 0.0)
                c0, c1 = clip(s0, s1)
                if c1 > c0:
                    labels.append((c0, c1, gpu, label))
                    stage_windows.append((c0, c1, gpu))
            for c0, c1, gpu in stage_windows:
                for other in group:
                    if other != gpu:
                        labels.append((c0, c1, other, "held_wait_peer"))
            for gpu in group:
                labels.append((start, finish, gpu, "unknown" if event_unknown else "held_overhead"))
        ordered = sorted(events, key=lambda e: (_event_interval(e) or (math.inf, math.inf))[0])
        for previous, following in zip(ordered, ordered[1:]):
            first, second = _event_interval(previous), _event_interval(following)
            if first is None or second is None:
                continue
            if (previous.get("block_after") or 0) > 0 and previous.get("controller_status", "completed") == "completed":
                owners.append((rid, [g for g in previous.get("group_after", []) if type(g) is int],
                               first[1], second[0]))
            if previous.get("group_after") == following.get("group_after"):
                turnarounds_ctrl.append(second[0] - first[1])
                gpu_end, gpu_begin = previous.get("all_done_ms"), following.get("async_begin_ms")
                if _number(gpu_end) and _number(gpu_begin):
                    turnarounds_gpu.append(gpu_begin - gpu_end)
        if ordered:
            last = ordered[-1]
            last_interval = _event_interval(last)
            if last_interval and (last.get("block_after") or 0) > 0:
                owners.append((rid, list(last.get("group_after", [])), last_interval[1], W))
        if not events and _number(row.get("start_ms")):
            diagnostics["requests_without_events_but_started"] += 1
            end = row.get("finish_ms") if _number(row.get("finish_ms")) else W
            request_busy.setdefault(rid, []).append((row["start_ms"], end))
            for gpu in row.get("gpu_group") or []:
                if type(gpu) is int and 0 <= gpu < gpus:
                    c0, c1 = clip(row["start_ms"], end)
                    if c1 > c0:
                        labels.append((c0, c1, gpu, "unknown"))
                        occupancy.append((c0, c1, gpu, {"request_id": rid, "transition": "unrecorded",
                                                        "quantum": None, "event_id": None, "start": c0,
                                                        "finish": c1, "group": [gpu]}))
    diagnostics["unmapped_stage_workers"] = sorted(unmapped)

    # Requests that need a GPU count but never ran (or ran only their finish) inherit the cell's
    # observed video gang size, else the configured form; unknown sizes are counted.
    observed_video_need = max([n for r, n in need.items() if kinds.get(r) == "video"] or [0])
    default_video_need = observed_video_need
    if not default_video_need and cell_forms:
        sizes = [2 if str(form).startswith("cfg2") else int(str(form)[2:] or 1) if str(form).startswith("sp") else 0
                 for form in cell_forms]
        default_video_need = min([s for s in sizes if s] or [0])
    waiting, encoding = [], []
    for row in rows:
        rid = row["request_id"]
        arrival = row.get("arrival_ms")
        if not _number(arrival):
            continue
        if rid not in need:
            if row.get("kind") == "image":
                need[rid] = 1
            elif default_video_need:
                need[rid] = default_video_need
            else:
                need[rid] = 1
                diagnostics["missing_need_requests"] += 1
        ready = arrival
        if row.get("kind") == "video" and _number(row.get("encode_ready_ms")):
            ready = max(arrival, row["encode_ready_ms"])
        end = row.get("finish_ms") if row.get("status") == "completed" and _number(row.get("finish_ms")) else W
        end = min(end, W)
        if ready > arrival:
            encoding.append((max(0.0, arrival), min(ready, end), rid))
        if end > ready:
            for start, finish in subtract((max(0.0, ready), end), request_busy.get(rid, [])):
                waiting.append((start, finish, rid))

    # Same-GPU task overlap is a controller anomaly; count it but never double count time.
    per_gpu_occ = {g: sorted((o for o in occupancy if o[2] == g), key=lambda o: o[0]) for g in range(gpus)}
    for g, items in per_gpu_occ.items():
        reach = -math.inf
        for start, finish, _, _ in items:
            if start < reach:
                diagnostics["same_gpu_task_overlap_ms"] += min(reach, finish) - start
            reach = max(reach, finish)
    occ_starts = {g: [o[0] for o in items] for g, items in per_gpu_occ.items()}
    occ_by_finish = {g: sorted(items, key=lambda o: o[1]) for g, items in per_gpu_occ.items()}
    occ_finishes = {g: [o[1] for o in items] for g, items in occ_by_finish.items()}

    arrivals = [r["arrival_ms"] for r in rows if _number(r.get("arrival_ms"))]
    finishes = [r["finish_ms"] for r in rows if _number(r.get("finish_ms"))]
    active = (max(0.0, min(arrivals)), min(W, max(finishes))) if arrivals and finishes else None

    points = {0.0, W}
    for start, finish, *_ in labels:
        points.update((start, finish))
    for start, finish, *_ in occupancy:
        points.update((start, finish))
    for start, finish, _ in waiting + encoding:
        points.update((max(0.0, min(W, start)), max(0.0, min(W, finish))))
    for _, _, start, finish in owners:
        points.update((max(0.0, min(W, start)), max(0.0, min(W, finish))))
    if active:
        points.update(active)
    points = sorted(p for p in points if 0.0 <= p <= W)
    index = {p: i for i, p in enumerate(points)}
    starts = [[] for _ in points]
    ends = [[] for _ in points]

    def schedule(start, finish, item):
        start, finish = max(0.0, min(W, start)), max(0.0, min(W, finish))
        if finish > start:
            starts[index[start]].append(item)
            ends[index[finish]].append(item)

    for start, finish, gpu, label in labels:
        schedule(start, finish, ("label", gpu, label))
    for start, finish, gpu, _ in occupancy:
        schedule(start, finish, ("occ", gpu))
    for start, finish, rid in waiting:
        schedule(start, finish, ("wait", rid))
    for start, finish, rid in encoding:
        schedule(start, finish, ("enc", rid))
    for rid, group, start, finish in owners:
        schedule(start, finish, ("own", rid, tuple(group)))

    label_count = [dict() for _ in range(gpus)]
    occ_count = [0] * gpus
    wait_count, enc_count, own_count = {}, {}, {}
    totals = [{c: 0.0 for c in CATEGORIES} for _ in range(gpus)]
    active_totals = [{c: 0.0 for c in CATEGORIES} for _ in range(gpus)]
    causes = [dict() for _ in range(gpus)]
    unknown_causes = [dict() for _ in range(gpus)]
    runs = [None] * gpus
    finished_runs = []

    def bump(counter, key, delta):
        value = counter.get(key, 0) + delta
        if value:
            counter[key] = value
        else:
            counter.pop(key, None)

    def apply(item, delta):
        if item[0] == "label":
            bump(label_count[item[1]], item[2], delta)
        elif item[0] == "occ":
            occ_count[item[1]] += delta
        elif item[0] == "wait":
            bump(wait_count, item[1], delta)
        elif item[0] == "enc":
            bump(enc_count, item[1], delta)
        else:
            bump(own_count, (item[1], item[2]), delta)

    def close_run(gpu):
        if runs[gpu] is not None:
            finished_runs.append(runs[gpu])
            runs[gpu] = None

    for i in range(len(points) - 1):
        for item in ends[i]:
            apply(item, -1)
        for item in starts[i]:
            apply(item, +1)
        t0, t1 = points[i], points[i + 1]
        duration = t1 - t0
        if duration <= 0:
            continue
        in_active = max(0.0, min(t1, active[1]) - max(t0, active[0])) if active else 0.0
        busy_label = []
        for gpu in range(gpus):
            present = label_count[gpu]
            if present:
                busy_label.append(min(present, key=lambda l: (PRIORITY[l], TIE_ORDER[l])))
            elif occ_count[gpu] > 0:
                busy_label.append("unknown")
            else:
                busy_label.append(None)
        free = {g for g in range(gpus) if busy_label[g] is None}
        for gpu in range(gpus):
            category, cause = busy_label[gpu], None
            if category == "unknown":
                cause = "occupied_without_attributable_work"
                if not label_count[gpu]:
                    cause = "occupancy_without_label"
            if category is None:
                held = [key for key in own_count if gpu in key[1]]
                if held:
                    category, cause = "held_midstep_owner", held[0][0]
                elif wait_count:
                    category = "idle_with_queue"
                    fits = []
                    for rid in wait_count:
                        owned = [key for key in own_count if key[0] == rid]
                        if owned:
                            if set(owned[0][1]) <= free:
                                fits.append(rid)
                        elif need.get(rid, 1) <= len(free):
                            fits.append(rid)
                    position = bisect.bisect_right(occ_finishes[gpu], t0) - 1
                    previous = occ_by_finish[gpu][position][3] if position >= 0 else None
                    nxt_index = bisect.bisect_left(occ_starts[gpu], t1)
                    following = per_gpu_occ[gpu][nxt_index][3] if nxt_index < len(per_gpu_occ[gpu]) else None
                    if (previous and following and previous["request_id"] == following["request_id"]
                            and following["request_id"] in wait_count):
                        cause = "turnaround_same_request"
                    elif fits:
                        cause = ("dispatchable_video" if any(kinds.get(r) == "video" for r in fits)
                                 else "dispatchable_image_only")
                    else:
                        partner = [busy_label[g] for g in range(gpus) if g != gpu and busy_label[g]]
                        cause = "gang_wait_partner_" + (partner[0] if partner else "none")
                elif enc_count:
                    category = "idle_encode_wait"
                else:
                    category = "idle_empty"
            totals[gpu][category] += duration
            active_totals[gpu][category] += in_active
            if category == "idle_with_queue":
                causes[gpu][cause] = causes[gpu].get(cause, 0.0) + duration
                run = runs[gpu]
                if run is None or run["finish_ms"] != t0:
                    close_run(gpu)
                    waiting_now = sorted(wait_count)
                    runs[gpu] = run = {
                        "gpu": gpu, "start_ms": t0, "finish_ms": t1, "causes": {},
                        "waiting_requests": [{"request_id": r, "kind": kinds.get(r), "need_gpus": need.get(r)}
                                             for r in waiting_now[:8]],
                        "waiting_count": len(waiting_now),
                        "partner_state_at_start": {str(g): busy_label[g] or "idle" for g in range(gpus) if g != gpu},
                    }
                run["finish_ms"] = t1
                run["causes"][cause] = run["causes"].get(cause, 0.0) + duration
            else:
                close_run(gpu)
                if category == "unknown":
                    unknown_causes[gpu][cause] = unknown_causes[gpu].get(cause, 0.0) + duration
    for gpu in range(gpus):
        close_run(gpu)

    for run in finished_runs:
        gpu = run["gpu"]
        position = bisect.bisect_right(occ_finishes[gpu], run["start_ms"]) - 1
        previous = occ_by_finish[gpu][position][3] if position >= 0 else None
        nxt_index = bisect.bisect_left(occ_starts[gpu], run["finish_ms"])
        following = per_gpu_occ[gpu][nxt_index][3] if nxt_index < len(per_gpu_occ[gpu]) else None
        run["duration_ms"] = run["finish_ms"] - run["start_ms"]
        run["previous_task"] = None if previous is None else {k: previous[k] for k in ("request_id", "transition", "quantum", "event_id", "group")}
        run["next_task"] = None if following is None else {k: following[k] for k in ("request_id", "transition", "quantum", "event_id", "group")}
        run["clues"] = _clues(run)
    finished_runs.sort(key=lambda r: -r["duration_ms"])

    per_gpu = []
    for gpu in range(gpus):
        total = sum(totals[gpu].values())
        per_gpu.append({
            "gpu": gpu, "window_ms": W, "accounted_ms": total,
            "ms": totals[gpu], "fraction": {c: totals[gpu][c] / W for c in CATEGORIES},
            "idle_with_queue_causes_ms": causes[gpu], "unknown_causes_ms": unknown_causes[gpu],
            "active_window_ms": active_totals[gpu],
        })
    overall = {c: sum(p["ms"][c] for p in per_gpu) for c in CATEGORIES}
    gpu_ms = W * gpus
    cause_total = {}
    for p in per_gpu:
        for cause, value in p["idle_with_queue_causes_ms"].items():
            cause_total[cause] = cause_total.get(cause, 0.0) + value
    active_overall = {c: sum(p["active_window_ms"][c] for p in per_gpu) for c in CATEGORIES}
    active_gpu_ms = (active[1] - active[0]) * gpus if active else 0.0
    return {
        "window_ms": W, "gpus": gpus, "gpu_ms": gpu_ms,
        "per_gpu": per_gpu,
        "overall_ms": overall,
        "overall_fraction": {c: overall[c] / gpu_ms for c in CATEGORIES},
        "grouped_fraction": _grouped(overall, gpu_ms),
        "idle_with_queue_causes_ms": cause_total,
        "active_window": None if not active else {
            "start_ms": active[0], "finish_ms": active[1], "gpu_ms": active_gpu_ms,
            "overall_ms": active_overall,
            "grouped_fraction": _grouped(active_overall, active_gpu_ms) if active_gpu_ms else None,
            "scope": "first arrival to last request finish; excludes leading/trailing empty replay time"},
        "top_idle_with_queue": finished_runs[:top],
        "idle_with_queue_run_count": len(finished_runs),
        "secondary": {
            "advance_stage_rpc": dict(stage_stats, worker_model_over_rpc_wall=(
                stage_stats["advance_worker_model_ms"] / stage_stats["advance_elapsed_ms"]
                if stage_stats["advance_elapsed_ms"] else None),
                scope="per-rank sums; worker model_ms is a worker-timed section, not CUDA kernel time"),
            "decode_wall_ms": distribution(decode_walls),
            "image_wall_ms": distribution(image_walls),
            "same_group_turnaround_controller_ms": distribution(turnarounds_ctrl),
            "same_group_turnaround_worker_gap_ms": distribution(turnarounds_gpu),
        },
        "diagnostics": diagnostics,
        "accounting_complete": (diagnostics["unplaceable_events"] == 0 and not diagnostics["unmapped_stage_workers"]
                                and all(abs(p["accounted_ms"] - W) <= 1e-6 * max(1.0, W) for p in per_gpu)),
    }


def _grouped(values, denominator):
    work = sum(values[c] for c in WORK_LABELS)
    groups = {"work": work, "held": sum(values[c] for c in HELD), "idle_with_queue": values["idle_with_queue"],
              "idle_encode_wait": values["idle_encode_wait"], "idle_empty": values["idle_empty"],
              "unknown": values["unknown"]}
    return {k: (v / denominator if denominator else None) for k, v in groups.items()}


def _clues(run):
    causes = run["causes"]
    main = max(causes, key=causes.get)
    clues = []
    partner = set(run["partner_state_at_start"].values())
    if main.startswith("gang_wait_partner_decode") or "decode" in partner:
        clues.append("partner GPU in single-GPU video finish (VAE decode + output digest) while a CFG2 gang waits")
    if main.startswith("gang_wait_partner_image") or "image_compute" in partner:
        clues.append("partner GPU serving an image; waiting video needs both GPUs")
    if main == "turnaround_same_request":
        clues.append("step-boundary turnaround of the same request (controller observe/plan/dispatch)")
    if main.startswith("dispatchable"):
        clues.append("enough free GPUs for a waiting request but no launch (controller latency or policy hold)")
    following = run.get("next_task") or {}
    if following.get("transition") in ("initial_distribution", "regroup"):
        clues.append("next task is a placement change (prepare/export/import)")
    if following.get("transition") == "finish":
        clues.append("next task is the request's decode")
    return clues


def analyze_record(record, label=None, top=10):
    cells = []
    for phase in record.get("phases", []):
        phase_id, attempt = phase.get("id", "cluster"), phase.get("attempt", 1)
        for cell in phase.get("cells", []):
            rows = cell_rows(record, phase_id, attempt, cell)
            worker_gpu = _worker_gpus(record, phase_id, cell)
            forms = None
            for cfg in record.get("config", {}).get("phases", []):
                if cfg.get("id") == phase_id:
                    forms = (cfg.get("options") or {}).get("forms")
            identity = {"run": label or record.get("run_id"), "run_id": record.get("run_id"),
                        "phase": phase_id, "attempt": attempt, "cell": cell.get("cell"),
                        "order": cell.get("order"), "policy": cell.get("policy"), "status": cell.get("status")}
            if not _number(cell.get("elapsed_ms")) or not cell.get("gpus"):
                cells.append(dict(identity, error="missing elapsed_ms or gpus; not analyzed"))
                continue
            result = analyze_cell(rows, cell["gpus"], cell["elapsed_ms"], worker_gpu, forms, top)
            result["requests"] = {"total": len(rows), "by_status": _count(r.get("status") for r in rows),
                                  "by_kind": _count(r.get("kind") for r in rows)}
            result["doorbell"] = cell.get("doorbell")
            result["l2_block_cuda"] = l2_kernel_ratio(rows)
            cells.append(dict(identity, **result))
    return {"schema": SCHEMA, "run": label or record.get("run_id"), "run_id": record.get("run_id"),
            "experiment": record.get("experiment"), "status": record.get("status"), "cells": cells,
            "definitions": __doc__}


def _count(values):
    out = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def l2_kernel_ratio(rows):
    """Sum per-block CUDA-event time over block-quantum wall wherever ``block_cuda_ms`` exists."""
    kernel = wall = 0.0
    quanta = 0

    def walk(node):
        nonlocal kernel, wall, quanta
        if isinstance(node, dict):
            if isinstance(node.get("block_cuda_ms"), list) and _number(node.get("ms")):
                values = [b.get("ms") for b in node["block_cuda_ms"] if isinstance(b, dict) and _number(b.get("ms"))]
                if values:
                    kernel += sum(values)
                    wall += node["ms"]
                    quanta += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for row in rows:
        walk(row)
    if not quanta:
        return {"status": "unknown", "reason": "no block_cuda_ms recorded in this run (only emitted in observe mode and not persisted here)"}
    return {"status": "measured", "quanta": quanta, "kernel_ms": kernel, "wall_ms": wall,
            "kernel_over_wall": kernel / wall if wall else None,
            "scope": "CUDA events around blocks within block_quantum wall (observe mode only); L2 supplement"}


def summarize(results):
    """Compact per-cell table plus pooled per-policy descriptive totals (runs listed, never merged per cell)."""
    table, pooled = [], {}
    for result in results:
        for cell in result["cells"]:
            if "error" in cell:
                table.append({k: cell.get(k) for k in ("run", "phase", "attempt", "cell", "order", "policy", "error")})
                continue
            entry = {k: cell[k] for k in ("run", "phase", "attempt", "cell", "order", "policy", "status", "window_ms", "gpus", "gpu_ms")}
            entry["grouped_fraction"] = cell["grouped_fraction"]
            entry["overall_fraction"] = cell["overall_fraction"]
            entry["idle_with_queue_causes_fraction"] = {k: v / cell["gpu_ms"] for k, v in cell["idle_with_queue_causes_ms"].items()}
            entry["active_grouped_fraction"] = (cell["active_window"] or {}).get("grouped_fraction")
            entry["accounting_complete"] = cell["accounting_complete"]
            entry["requests"] = cell["requests"]
            table.append(entry)
            bucket = pooled.setdefault(cell["policy"], {"cells": [], "gpu_ms": 0.0, "ms": {c: 0.0 for c in CATEGORIES}, "causes_ms": {}})
            bucket["cells"].append(f"{cell['run']}/{cell['phase']}/{cell['attempt']}/{cell['cell']}/{cell['order']}")
            bucket["gpu_ms"] += cell["gpu_ms"]
            for c in CATEGORIES:
                bucket["ms"][c] += cell["overall_ms"][c]
            for k, v in cell["idle_with_queue_causes_ms"].items():
                bucket["causes_ms"][k] = bucket["causes_ms"].get(k, 0.0) + v
    for bucket in pooled.values():
        bucket["fraction"] = {c: bucket["ms"][c] / bucket["gpu_ms"] for c in CATEGORIES}
        bucket["grouped_fraction"] = _grouped(bucket["ms"], bucket["gpu_ms"])
        bucket["causes_fraction"] = {k: v / bucket["gpu_ms"] for k, v in bucket["causes_ms"].items()}
        bucket["scope"] = "GPU-ms pooled over heterogeneous runs (different models/seeds/loads); descriptive only"
    return {"schema": SCHEMA + ".summary", "cells": table, "pooled_by_policy": pooled}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, help="LABEL=PATH to a run.json (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compact = []
    for item in args.run:
        label, _, path = item.partition("=")
        if not path:
            label, path = Path(item).parent.name, item
        record = read_json(path)
        result = analyze_record(record, label, args.top)
        del record
        result["source"] = {"path": str(path), "bytes": os.path.getsize(path)}
        (args.output_dir / f"{label}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        for cell in result["cells"]:
            cell.pop("per_gpu", None)
        compact.append(result)
        print(json.dumps({"run": label, "cells": len(result["cells"])}), flush=True)
    summary = summarize(compact)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
