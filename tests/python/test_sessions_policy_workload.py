import copy
import unittest

from ditserve.sessions.policy import ARMS, select
from ditserve.sessions.workload import STAGES, make_trace, release_turn, stage_deadline


class SessionsTests(unittest.TestCase):
    def trace(self):
        return make_trace({"think_ms": 17}, {"stage_ms": {stage: i + 1 for i, stage in enumerate(STAGES)}}, .5)

    def test_external_inputs_are_frozen_but_next_turn_is_causal(self):
        trace = self.trace()
        self.assertEqual(trace, self.trace())
        self.assertEqual(12, len(trace["sessions"]))
        session = trace["sessions"][0]
        self.assertIsNone(release_turn(session, 1, trace))
        prior = release_turn(session, 0, trace)
        self.assertIsNone(release_turn(session, 1, trace, prior))
        prior.update(status="completed", finish_ms=100)
        second = release_turn(session, 1, trace, prior)
        self.assertEqual(117, second["release_ms"])
        prior["finish_ms"] = 200
        self.assertEqual(217, release_turn(session, 1, trace, prior)["release_ms"])
        self.assertEqual(trace["trace_sha256"], self.trace()["trace_sha256"])

    def test_proportional_stage_budgets_and_final_deadline(self):
        trace = self.trace()
        row = release_turn(trace["sessions"][0], 0, trace)
        self.assertEqual(2.5, stage_deadline(row, "llm_prepare", trace))
        self.assertEqual(row["deadline_ms"], stage_deadline(row, "image_decode", trace))

    def test_no_future_or_unready_visibility_and_no_mutation(self):
        trace = self.trace()
        row = release_turn(trace["sessions"][0], 0, trace)
        values = [row, {"ready": True, "status": "pending", "release_ms": 999}, {"ready": False}]
        before = copy.deepcopy(values)
        for arm in ARMS:
            self.assertEqual(row["request_id"], select(values, 0, arm))
        self.assertEqual(before, values)

    def test_serial_retains_round_and_releases_between_rounds(self):
        row = release_turn(self.trace()["sessions"][0], 0, self.trace())
        other = dict(row, request_id="other", deadline_ms=1)
        self.assertEqual(row["request_id"], select([row, other], 0, "session_serial", row["request_id"]))
        row["status"] = "completed"
        self.assertEqual("other", select([row, other], 0, "session_serial", row["request_id"]))

    def test_slack_and_stage_edf_have_distinct_predeclared_ranks(self):
        row = release_turn(self.trace()["sessions"][0], 0, self.trace())
        a = dict(row, request_id="a", deadline_ms=80, stage_deadline_ms=30, remaining_total_ms=70)
        b = dict(row, request_id="b", deadline_ms=60, stage_deadline_ms=20, remaining_total_ms=10)
        self.assertEqual("a", select([a, b], 0, "session_slack"))
        self.assertEqual("b", select([a, b], 0, "stage_edf"))

    def test_stable_tie_and_overdue_work_still_completes(self):
        row = release_turn(self.trace()["sessions"][0], 0, self.trace())
        rows = [dict(row, request_id=rid) for rid in ("b", "a")]
        for arm in ARMS:
            remaining = copy.deepcopy(rows)
            self.assertEqual("a", select(remaining, 1000, arm))
            remaining[1]["status"] = "completed"
            self.assertEqual("b", select(remaining, 1000, arm))

    def test_predecessor_validation_and_bad_costs(self):
        trace = self.trace()
        with self.assertRaises(ValueError):
            release_turn(trace["sessions"][0], 1, trace,
                         dict(status="completed", finish_ms=1, session_id="wrong", round=1))
        with self.assertRaises(ValueError):
            make_trace({}, {"stage_ms": dict.fromkeys(STAGES, float("nan"))}, .5)


if __name__ == "__main__":
    unittest.main()
