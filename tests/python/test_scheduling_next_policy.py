"""The new fixed quantum must preserve EDF and historical arm identities."""

import copy
import unittest

from ditserve.scheduling import next_policy, policy


def request(rid, deadline, *, kind="video", ready=True):
    return dict(request_id=rid, kind=kind, status="pending", arrived=True, gpu_ready=ready,
                arrival_ms=0., deadline_ms=deadline, ready_since_ms=0., service_ms=100.,
                remaining_ms=20., started=True, action="denoise" if kind == "video" else "image",
                action_ms_by_quantum={"1": 1., "2": 2., "4": 4., "8": 8., "step": 60.})


class NextSchedulingPolicyTests(unittest.TestCase):
    def test_new_arm_uses_plain_edf_even_when_earliest_request_is_predicted_late(self):
        rows = [request("later", 100.), request("first", 1.), request("unready", 0., ready=False)]
        before = copy.deepcopy(rows)
        expected = policy.select_dispatch(rows, 0., "edf_b1", {})
        actual = next_policy.select_dispatch(rows, 0., "edf_b2", {})
        self.assertEqual(actual, dict(expected, quantum=2))
        self.assertEqual(actual["request_id"], "first")
        self.assertEqual(rows, before)
        self.assertNotIn("edf_b2", policy.ARMS)

    def test_existing_arms_remain_identical_including_guarded_decisions(self):
        rows = [request("video", 100.), request("image", 50., kind="image")]
        rows[0]["remaining_ms"] = 98.
        costs = {"image_deadline_ms": 40.}
        for arm in policy.ARMS:
            with self.subTest(arm=arm):
                self.assertEqual(next_policy.select_dispatch(rows, 0., arm, costs),
                                 policy.select_dispatch(rows, 0., arm, costs))

    def test_async_arm_retains_exact_edf_one_block_decisions(self):
        rows = [request("late", 1.), request("later", 100.), request("future", 0., ready=False)]
        self.assertEqual(next_policy.select_dispatch(iter(rows), 50., "edf_b1_async", {}),
                         policy.select_dispatch(rows, 50., "edf_b1", {}))
        self.assertNotIn("edf_b1_async", policy.ARMS)

    def test_empty_ready_set_retains_no_action_and_quantum_none(self):
        actual = next_policy.select_dispatch([request("waiting", 1., ready=False)], 0., "edf_b2", {})
        self.assertIsNone(actual["request_id"])
        self.assertIsNone(actual["quantum"])
        self.assertEqual(actual["reason"], "no_ready_request")

    def test_missing_or_invalid_two_block_cost_is_rejected_before_dispatch(self):
        for cost in (None, float("nan"), -1.):
            row = request("video", 100.)
            if cost is None:
                del row["action_ms_by_quantum"]["2"]
            else:
                row["action_ms_by_quantum"]["2"] = cost
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                next_policy.select_dispatch([row], 0., "edf_b2", {})

    def test_generator_ready_view_is_supported_without_second_iteration_loss(self):
        rows = [request("later", 200.), request("first", 100.)]
        choice = next_policy.select_dispatch(iter(rows), 0., "edf_b2", {})
        self.assertEqual((choice["request_id"], choice["quantum"]), ("first", 2))

    def test_unknown_arm_retains_existing_rejection(self):
        with self.assertRaises(ValueError):
            next_policy.select_dispatch([], 0., "edf_b3", {})


if __name__ == "__main__":
    unittest.main()
