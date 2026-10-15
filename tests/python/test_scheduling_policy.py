"""CPU scheduling cases with explicit deadline conflicts and readiness."""

import copy
import unittest

from ditserve.scheduling.policy import ARMS, select_dispatch


def request(rid, *, kind="image", deadline=100., remaining=10., costs=(2., 8., 16.),
            started=False, action=None, arrival=0., ready_since=0., service=100.):
    return dict(request_id=rid, kind=kind, deadline_ms=deadline, arrival_ms=arrival,
                service_ms=service, ready_since_ms=ready_since, started=started,
                remaining_ms=remaining, action=action or ("denoise" if kind == "video" else "image"),
                action_ms_by_quantum=dict(zip(("1", "4", "8", "step"), (*costs, 60.))),
                arrived=True, gpu_ready=True, status="pending")


class SchedulingPolicyTests(unittest.TestCase):
    def choose(self, rows, arm="guarded_b4", now=0.):
        return select_dispatch(rows, now, arm, {"image_deadline_ms": 40.})

    def test_arm_identity_and_unready_requests_cannot_be_selected(self):
        self.assertEqual(set(ARMS), {"edf_b1", "edf_b4", "guarded_b4", "guarded_adaptive", "feasible_b4", "edf_step"})
        good = request("good")
        rows = [good]
        for change in ({"arrived": False}, {"gpu_ready": False}, {"status": "completed"}, {"arrival_ms": 1.}):
            row = request("excluded", deadline=1.)
            row.update(change)
            rows.append(row)
        self.assertEqual(self.choose(rows)["request_id"], "good")
        self.assertIsNone(self.choose(rows[1:])["request_id"])

    def test_baselines_do_not_skip_late_requests_or_apply_wait_protection(self):
        late = request("late", deadline=1., remaining=20.)
        okay = request("okay", deadline=100.)
        for arm, quantum in (("edf_b1", 1), ("edf_b4", 4), ("edf_step", "step")):
            choice = self.choose([okay, late], arm)
            self.assertEqual((choice["request_id"], choice["quantum"], choice["reason"]), ("late", quantum, "edf"))
        self.assertEqual(self.choose([okay, late], "feasible_b4")["request_id"], "okay")

    def test_all_predicted_late_still_drain_and_ties_are_deterministic(self):
        rows = [request("b", deadline=1., remaining=10.), request("a", deadline=1., remaining=10.)]
        choice = self.choose(rows)
        self.assertEqual((choice["request_id"], choice["reason"]), ("a", "edf_all_predicted_late"))
        rows[1]["status"] = "completed"
        self.assertEqual(self.choose(rows)["request_id"], "b")

    def test_wait_guard_overrides_feasibility_and_resets_after_service(self):
        waiting = request("waiting", deadline=5., remaining=10., service=10.)
        fresh = request("fresh", arrival=20., ready_since=20., deadline=100.)
        self.assertEqual(self.choose([fresh, waiting], now=25.)["reason"], "waiting_guard")
        waiting["ready_since_ms"] = 25.
        self.assertEqual(self.choose([fresh, waiting], now=25.)["request_id"], "fresh")

    def test_wait_guard_orders_by_trigger_time_across_service_lengths(self):
        long = request("long", kind="video", deadline=200., service=40., ready_since=0.)
        short = request("short", deadline=300., service=10., ready_since=50.)
        # The short request crossed its guard at 75 ms; the longer-waiting
        # video crossed at 100 ms.  Arrival order must not reverse protection.
        choice = self.choose([long, short], now=100.)
        self.assertEqual((choice["request_id"], choice["reason"]), ("short", "waiting_guard"))

    def test_wait_guard_conflict_returns_to_edf_and_shortest_action(self):
        waiting = request("waiting", deadline=200., remaining=20., service=10.)
        urgent = request("urgent", deadline=31., remaining=5., arrival=20., ready_since=20.)
        choice = self.choose([waiting, urgent], now=25.)
        self.assertEqual((choice["request_id"], choice["quantum"], choice["reason"]),
                         ("urgent", 1, "deadline_conflict_edf_min_quantum"))
        self.assertTrue(choice["predicted_deadline_conflict"])
        # Completion is still mandatory: when the urgent request finishes,
        # the old request remains eligible and can drain.
        urgent["status"] = "completed"
        self.assertEqual(self.choose([waiting, urgent], now=27.)["request_id"], "waiting")

    def test_video_resume_only_when_both_jobs_remain_feasible(self):
        image = request("image", deadline=50., remaining=10., costs=(10., 10., 10.))
        video = request("video", kind="video", deadline=100., remaining=95., costs=(2., 4., 8.), started=True)
        choice = self.choose([image, video])
        self.assertEqual((choice["request_id"], choice["reason"]), ("video", "video_resume_guard"))
        image["remaining_ms"] = 48.
        choice = self.choose([image, video])
        self.assertEqual((choice["request_id"], choice["quantum"]), ("image", 1))
        self.assertTrue(choice["predicted_deadline_conflict"])

    def test_video_guard_does_not_sacrifice_a_third_feasible_request(self):
        image = request("image", deadline=50., remaining=10., costs=(10., 10., 10.))
        video = request("video", kind="video", deadline=100., remaining=95., costs=(2., 4., 8.), started=True)
        third = request("third", deadline=110., remaining=107., costs=(1., 1., 1.))
        choice = self.choose([image, video, third])
        self.assertEqual(choice["reason"], "deadline_conflict_edf_min_quantum")
        self.assertEqual(choice["request_id"], "image")

    def test_feasible_ablation_has_no_video_protection(self):
        image = request("image", deadline=50., remaining=10., costs=(10., 10., 10.))
        video = request("video", kind="video", deadline=100., remaining=95., costs=(2., 4., 8.), started=True)
        self.assertEqual(self.choose([image, video], "feasible_b4")["request_id"], "image")

    def test_adaptive_largest_fitting_quantum_and_empty_feasible_set(self):
        video = request("video", kind="video", deadline=100.)
        self.assertEqual(self.choose([video], "guarded_adaptive")["quantum"], 8)
        peer = request("peer", deadline=120., remaining=110.)
        self.assertEqual(self.choose([video, peer], "guarded_adaptive")["quantum"], 4)
        peer["remaining_ms"] = 119.
        choice = self.choose([video, peer], "guarded_adaptive")
        self.assertEqual(choice["quantum"], 1)
        self.assertTrue(choice["predicted_deadline_conflict"])

    def test_atomic_actions_are_never_misrepresented_as_shorter_quanta(self):
        video = request("video", kind="video", deadline=100., action="decode", costs=(50., 50., 50.))
        choice = self.choose([video], "guarded_adaptive")
        self.assertEqual(choice["quantum"], 1)
        self.assertTrue(choice["predicted_deadline_conflict"])

    def test_selection_is_pure_and_invalid_costs_fail_closed(self):
        rows = [request("one")]
        original = copy.deepcopy(rows)
        self.choose(rows)
        self.assertEqual(rows, original)
        rows[0]["remaining_ms"] = float("nan")
        with self.assertRaises(ValueError):
            self.choose(rows)
        with self.assertRaises(ValueError):
            self.choose([], "unknown")


if __name__ == "__main__":
    unittest.main()
