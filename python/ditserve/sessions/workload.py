"""Frozen external inputs and causal, completion-triggered conversation turns."""

import copy
import hashlib
import json
import math
import random


STAGES = ("llm_prepare", "llm_prefill", "llm_decode", "image_encode", "image_denoise", "image_decode")
DEFAULT_INPUTS = (
    ("Describe an image of a red ceramic kettle on a wooden kitchen table. Return only the image prompt.",
     "Keep the same kettle and setting, but make the kettle blue and add a white cup. Return only the updated image prompt."),
    ("Describe an image of a brass pendulum clock in a quiet study. Return only the image prompt.",
     "Keep the same clock and study, but show warm evening light and a closed book nearby. Return only the updated image prompt."),
)


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(name + " must be " + ("positive" if positive else "nonnegative"))
    return float(value)


def make_trace(config, cost_model, load, seed=301):
    """Freeze inputs, not the scheduler-dependent release of second turns.

    The load label uses two independent turn costs per external session start.
    It is neither measured GPU utilization nor an assertion about the eventual
    endogenous second-turn arrival rate. Think time adds no offered GPU work.
    """
    load = _number(load, "load", positive=True)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    count = config.get("session_count", 12)
    if type(count) is not int or count <= 0:
        raise ValueError("session_count must be a positive integer")
    costs = {stage: _number(cost_model["stage_ms"][stage], stage) for stage in STAGES}
    total = _number(sum(costs.values()), "total independent cost", positive=True)
    think = _number(config.get("think_ms", 250), "think_ms")
    multiplier = _number(config.get("deadline_multiplier", 2.5), "deadline_multiplier", positive=True)
    inputs = config.get("session_inputs", DEFAULT_INPUTS)
    if not isinstance(inputs, (list, tuple)) or not inputs or any(
            not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(
                not isinstance(text, str) or not text.strip() for text in pair) for pair in inputs):
        raise ValueError("session_inputs must contain pairs of nonempty user inputs")
    rng, sessions, start = random.Random(seed), [], 0.0
    for index in range(count):
        if index:
            start += rng.expovariate(load / (2 * total))
        sid = "s%03d" % index
        pair = inputs[index % len(inputs)]
        turns = [dict(request_id=sid + "-r%d" % (turn + 1), round=turn + 1,
                      user_input=pair[turn], think_ms=0.0 if turn == 0 else think,
                      llm_seed=seed * 100000 + index * 10 + turn * 2,
                      image_seed=seed * 100000 + index * 10 + turn * 2 + 1)
                 for turn in range(2)]
        sessions.append(dict(session_id=sid, start_ms=start, turns=turns))
    result = dict(schema_version="ditserve.sessions.trace.v1", seed=seed, load=load,
                  session_count=count, rounds_per_session=2, stage_ms=costs,
                  turn_service_ms=total, turn_deadline_window_ms=multiplier * total,
                  arrival_scope="external session starts; round 2 releases only after own round 1 image completion plus think_ms",
                  sessions=sessions)
    result["trace_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return result


def stage_deadline(row, stage, trace):
    if stage not in STAGES:
        raise ValueError("unknown session stage: " + str(stage))
    release = _number(row["release_ms"], "release_ms")
    costs = trace["stage_ms"]
    total = _number(sum(costs.values()), "total independent cost", positive=True)
    cumulative = sum(costs[name] for name in STAGES[:STAGES.index(stage) + 1])
    return release + trace["turn_deadline_window_ms"] * cumulative / total


def release_turn(session, round_index, trace, previous=None):
    """Create an eligible-time definition; never fabricate predecessor success.

    The executor still withholds this row from ready queues until release_ms.
    A failed/uncompleted predecessor leaves round two unreleased, explicitly
    counted as dependency-blocked rather than pretending it had arrived.
    """
    if type(round_index) is not int or round_index not in (0, 1):
        raise ValueError("round_index must be 0 or 1")
    turn = copy.deepcopy(session["turns"][round_index])
    if round_index:
        if previous is None or previous.get("status") != "completed":
            return None
        if previous.get("session_id") != session["session_id"] or previous.get("round") != 1:
            raise ValueError("predecessor belongs to another session or round")
        if previous.get("request_id") != session["turns"][0]["request_id"]:
            raise ValueError("predecessor request identity mismatch")
        finish = _number(previous["finish_ms"], "previous finish_ms")
        release = finish + turn["think_ms"]
    else:
        release = _number(session["start_ms"], "start_ms")
    turn.update(session_id=session["session_id"], session_start_ms=session["start_ms"],
                release_ms=release, deadline_ms=release + trace["turn_deadline_window_ms"],
                remaining_total_ms=trace["turn_service_ms"], status="pending", ready=True,
                stage="llm_prepare")
    turn["stage_deadlines_ms"] = {stage: stage_deadline(turn, stage, trace) for stage in STAGES}
    turn["stage_deadline_ms"] = turn["stage_deadlines_ms"][turn["stage"]]
    return turn
