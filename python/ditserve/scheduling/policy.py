"""Pure CPU dispatch rules for the bounded D1 scheduling experiment.

All costs are predictions frozen before a measured replay.  A nonnegative
slack is a prediction, not a deadline guarantee.  This module never advances
a model, changes a request, or inspects arrivals that are not yet ready.
"""

import math


ARMS = {
    "edf_b1": {"dispatch_policy": "edf", "quantum_policy": 1},
    "edf_b4": {"dispatch_policy": "edf", "quantum_policy": 4},
    "guarded_b4": {"dispatch_policy": "guarded", "quantum_policy": 4},
    "guarded_adaptive": {"dispatch_policy": "guarded", "quantum_policy": "adaptive"},
    "feasible_b4": {"dispatch_policy": "feasible", "quantum_policy": 4},
    "edf_step": {"dispatch_policy": "edf", "quantum_policy": "step"},
}


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a finite number" % name)
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("%s must be finite%s" % (name, " and nonnegative" if nonnegative else ""))
    return value


def _edf(row):
    return row["deadline_ms"], row["arrival_ms"], row["request_id"]


def _slack(row, now_ms):
    return row["deadline_ms"] - now_ms - row["remaining_ms"]


def _action_ms(row, quantum):
    values = row["action_ms_by_quantum"]
    # JSON costs have string keys; accepting integer keys also keeps the
    # in-memory interface independent of JSON serialization.
    value = values.get(str(quantum), values.get(quantum))
    if value is None:
        raise ValueError("missing action cost for %s quantum %s" % (row["request_id"], quantum))
    return _number(value, "action cost", nonnegative=True)


def _threatened(rows, candidate, cost, now_ms):
    return [row for row in rows if row["request_id"] != candidate["request_id"]
            and 0 <= _slack(row, now_ms) < cost]


def _quantum(row, rows, now_ms, quantum_policy, cost_model):
    if quantum_policy != "adaptive":
        return quantum_policy, False
    # Prepare, decode and Sana generation are atomic.  Calling them quantum 1
    # never implies that they can be split or made shorter.
    candidates = (1, 4, 8) if row["action"] == "denoise" else (1,)
    deadline = _number(cost_model["image_deadline_ms"], "image_deadline_ms", nonnegative=True)
    if deadline <= 0:
        raise ValueError("image_deadline_ms must be positive")
    other_slacks = [_slack(other, now_ms) for other in rows
                    if other["request_id"] != row["request_id"] and _slack(other, now_ms) >= 0]
    limit = min([.5 * deadline] + other_slacks)
    possible = [q for q in candidates if _action_ms(row, q) <= limit]
    return (max(possible), False) if possible else (1, True)


def _result(row, quantum, reason, conflict, now_ms):
    return {"request_id": row["request_id"], "quantum": quantum, "reason": reason,
            "predicted_deadline_conflict": bool(conflict), "slack_ms": _slack(row, now_ms)}


def select_dispatch(ready, now_ms, arm_id, cost_model):
    """Select one ready request and one legal quantum without mutating input.

    ``ready_since_ms`` is the start of the *current continuous* GPU-ready
    waiting interval.  The executor resets it after each service action, so
    the waiting guard does not permanently favor an old, actively running job.

    Waiting protection is considered first, but a predicted deadline conflict
    still falls back to EDF and the shortest action.  A conflict flag records
    an encountered infeasible action bound even if the shortest
    fallback quantum subsequently avoids the predicted deadline crossing.
    """
    if arm_id not in ARMS:
        raise ValueError("unknown scheduling arm: " + str(arm_id))
    now_ms = _number(now_ms, "now_ms", nonnegative=True)
    rows = [row for row in ready if row.get("arrived") is True and row.get("gpu_ready") is True
            and row.get("status") == "pending" and row["arrival_ms"] <= now_ms]
    if not rows:
        return {"request_id": None, "quantum": None, "reason": "no_ready_request",
                "predicted_deadline_conflict": False, "slack_ms": None}
    seen = set()
    for row in rows:
        rid = row["request_id"]
        if rid in seen:
            raise ValueError("duplicate ready request: " + str(rid))
        seen.add(rid)
        for key in ("arrival_ms", "deadline_ms", "remaining_ms", "ready_since_ms", "service_ms"):
            _number(row[key], key, nonnegative=True)
        if row["ready_since_ms"] < row["arrival_ms"] or row["ready_since_ms"] > now_ms:
            raise ValueError("ready_since_ms must be between arrival and now")
        if row["service_ms"] <= 0 or row["kind"] not in ("video", "image"):
            raise ValueError("invalid request kind or independent service cost")
        if row["action"] not in ("prepare", "denoise", "decode", "image"):
            raise ValueError("unknown request action")
    arm = ARMS[arm_id]
    if arm["dispatch_policy"] == "edf":
        candidate = min(rows, key=_edf)
        quantum = arm["quantum_policy"]
        _action_ms(candidate, quantum)
        return _result(candidate, quantum, "edf", False, now_ms)

    overdue_wait = [row for row in rows
                    if now_ms - row["ready_since_ms"] >= 2.5 * row["service_ms"]]
    if overdue_wait:
        candidate = min(overdue_wait, key=lambda row: (row["ready_since_ms"] + 2.5 * row["service_ms"],) + _edf(row))
        quantum, conflict = _quantum(candidate, rows, now_ms, arm["quantum_policy"], cost_model)
        conflict = conflict or bool(_threatened(rows, candidate, _action_ms(candidate, quantum), now_ms))
        if conflict:
            candidate = min(rows, key=_edf)
            _action_ms(candidate, 1)
            return _result(candidate, 1, "deadline_conflict_edf_min_quantum", True, now_ms)
        return _result(candidate, quantum, "waiting_guard", conflict, now_ms)

    feasible = [row for row in rows if _slack(row, now_ms) >= 0]
    candidate = min(feasible or rows, key=_edf)
    reason = "feasible_edf" if feasible else "edf_all_predicted_late"
    quantum, conflict = _quantum(candidate, rows, now_ms, arm["quantum_policy"], cost_model)
    threats = _threatened(rows, candidate, _action_ms(candidate, quantum), now_ms)
    if arm["dispatch_policy"] == "guarded":
        videos = sorted((row for row in threats if row["kind"] == "video" and row.get("started")),
                        key=lambda row: (_slack(row, now_ms),) + _edf(row))
        for video in videos:
            video_quantum, video_conflict = _quantum(video, rows, now_ms, arm["quantum_policy"], cost_model)
            cost = _action_ms(video, video_quantum)
            # Both requests must remain predicted feasible.  An already late
            # candidate is never a reason to label this a successful guard.
            if (not video_conflict and _slack(candidate, now_ms) >= cost
                    and not _threatened(rows, video, cost, now_ms)):
                return _result(video, video_quantum, "video_resume_guard", False, now_ms)
    if conflict or threats:
        candidate = min(rows, key=_edf)
        _action_ms(candidate, 1)
        return _result(candidate, 1, "deadline_conflict_edf_min_quantum", True, now_ms)
    return _result(candidate, quantum, reason, False, now_ms)
