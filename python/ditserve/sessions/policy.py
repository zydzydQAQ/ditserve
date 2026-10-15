"""Pure scheduling choices; all arms use identical model action boundaries."""

import math


ARMS = ("session_serial", "stage_edf", "session_slack")


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(field + " must be finite")
    return value


def select(ready, now, arm, active_id=None):
    """Return one request ID without reading future/unready request costs.

    ``active_id`` is the currently executing *round*, not an entire session.
    Serial keeps this round through its image completion; its think interval
    holds no reservation. Overdue work remains selectable under every arm.
    """
    if arm not in ARMS:
        raise ValueError("unknown session scheduling arm: " + str(arm))
    if _finite(now, "now") < 0:
        raise ValueError("now must be nonnegative")
    rows = [row for row in ready if row.get("ready") is True
            and row.get("status") in ("pending", "running")
            and _finite(row["release_ms"], "release_ms") <= now]
    seen = set()
    for row in rows:
        rid = row["request_id"]
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("ready request IDs must be nonempty and unique")
        seen.add(rid)
        for field in ("release_ms", "deadline_ms", "stage_deadline_ms", "remaining_total_ms"):
            if _finite(row[field], field) < 0:
                raise ValueError(field + " must be nonnegative")
        if row["deadline_ms"] < row["release_ms"]:
            raise ValueError("deadline precedes release")
    if not rows:
        return None
    if arm == "session_serial" and active_id in seen:
        return active_id
    def tie(row):
        return row["deadline_ms"], row["release_ms"], row["request_id"]
    if arm == "stage_edf":
        key = lambda row: (row["stage_deadline_ms"],) + tie(row)
    elif arm == "session_slack":
        key = lambda row: (row["deadline_ms"] - now - row["remaining_total_ms"],) + tie(row)
    else:
        key = tie
    return min(rows, key=key)["request_id"]
