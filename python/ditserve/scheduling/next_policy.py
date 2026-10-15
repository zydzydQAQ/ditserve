"""Add experimental EDF arms without changing the original experiment.

The execution adapter must provide a cost for quantum 2 before dispatch.  All
existing arms retain the original selector and their original semantics.
"""

from . import policy


def select_dispatch(ready, now_ms, arm_id, cost_model):
    if arm_id == "edf_b1_async":
        return policy.select_dispatch(ready, now_ms, "edf_b1", cost_model)
    if arm_id != "edf_b2":
        return policy.select_dispatch(ready, now_ms, arm_id, cost_model)
    rows = list(ready)
    selected = policy.select_dispatch(rows, now_ms, "edf_b1", cost_model)
    if selected["request_id"] is None:
        return selected
    row = next(row for row in rows if row["request_id"] == selected["request_id"])
    # Fail before executing any work if the new runner omitted this estimate.
    policy._action_ms(row, 2)
    return dict(selected, quantum=2)
