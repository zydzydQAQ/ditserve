import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve.benchmark.backends import FixtureBackend, RealClock, core_policy
from ditserve.benchmark.common import read_json, seal, verify_seal
from ditserve.benchmark.runner import ArrivalSource, build_profile, run_suite, select_ready


def config():
    return dict(model=dict(id="fixture", revision="fixture-v1"),
                shapes=dict(long=dict(height=32, width=32, frames=5, steps=4, weight=.5),
                            short=dict(height=32, width=32, frames=5, steps=1, weight=.5)),
                cache=dict(enabled=True, threshold=.08, max_continuous=1), guidance_scale=5.)


def dataset():
    return seal(dict(samples=[dict(sample_id="dev1", prompt="development", split="evaluation", service_split="dev"),
                              dict(sample_id="test1", prompt="held out", split="evaluation", service_split="test")]))


def request(rid, shape, arrival, deadline):
    return dict(request_id=rid, sample_id="test1", dataset="fixture", prompt="held out",
                shape_id=shape, **{k: v for k, v in config()["shapes"][shape].items() if k != "weight"},
                seed=31, model="fixture", revision="fixture-v1", cache_enabled=True,
                cache_threshold=.08, max_continuous=1, guidance_scale=5., negative_prompt="",
                arrival_ms=arrival, deadline_ms=deadline)


def trace(profile, requests=None):
    requests = copy.deepcopy(requests or [request("long", "long", 0, 100), request("urgent", "short", 3, 20)])
    for r in requests:
        r["deadline_ms"] = r["arrival_ms"] + 2.5 * profile["shapes"][r["shape_id"]]["service_ms"]
    return seal(dict(schema_version="ditserve.benchmark.trace.v1", config=config(),
                     profile=profile, profile_sha256=profile["sha256"],
                     dataset_sha256=dataset()["sha256"], dataset=dict(sha256=dataset()["sha256"]),
                     arrival=dict(count=len(requests)), requests=requests))


class RunnerTests(unittest.TestCase):
    def make_profile(self, root):
        return build_profile(config(), dataset(), "cpu-test", root / "profile")

    def test_preemption_keeps_independent_rng_cache_and_exact_output(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            self.assertEqual(profile["measurement_kind"], "synthetic_virtual_fixture")
            self.assertEqual(profile["dev_sample_ids"], ["dev1"])
            result = run_suite(trace(profile), ["fcfs", "edf", "srtf"], "cpu-test", root / "run")
            verify_seal(read_json(root / "run" / "run.json"))
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["measured"])
            completions = {}
            for policy, run in result["policies"].items():
                self.assertEqual(run["arrival_mode"], "virtual_open_loop")
                for row in run["requests"]:
                    ref = result["references"][row["request_id"]]
                    for key in ("output_path", "latent_path"):
                        np.testing.assert_array_equal(np.load(row["artifact"][key]), np.load(ref[key]))
                    self.assertEqual(row["artifact"]["path_history"], ref["path_history"])
                    self.assertEqual(row["submit_ms"], row["arrival_ms"])
                    self.assertEqual(row["quality"]["status"], "unknown")
                    if row["request_id"] == "urgent":
                        completions[policy] = row["finish_ms"]
            self.assertLess(completions["edf"], completions["fcfs"])
            self.assertLess(completions["srtf"], completions["fcfs"])

    def test_memory_pressure_only_uses_full_hot_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            with patch("ditserve.benchmark.runner.make_backend", side_effect=lambda *a, **k: FixtureBackend(capacity_mb=2)):
                result = run_suite(trace(profile), ["edf"], "cpu-test", root / "run")
            run = result["policies"]["edf"]
            self.assertEqual(run["status"], "completed")
            transfers = [e for r in run["requests"] for e in r["events"] if e["kind"] in ("snapshot", "restore")]
            self.assertEqual([e["kind"] for e in transfers], ["snapshot", "restore"])
            self.assertTrue(all(e["mode"] == "hot" for e in transfers))

    def test_failure_preserves_arrived_and_future_denominators(self):
        class Broken(FixtureBackend):
            def step(self, state):
                if state.step_index == 1:
                    raise RuntimeError("injected solver failure")
                return super().step(state)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            # References must succeed; inject the fault only during timed replay.
            from ditserve.benchmark import runner
            original = runner._run_policy
            def broken_run(t, policy, adapter, directory, budget):
                return original(t, policy, Broken(), directory, budget)
            requests = [request("long", "long", 0, 100), request("arrived", "short", 3, 100),
                        request("future", "short", 100, 200)]
            with patch.object(runner, "_run_policy", side_effect=broken_run):
                result = run_suite(trace(profile, requests), ["fcfs", "edf"], "cpu-test", root / "run")
            rows = {r["request_id"]: r for r in result["policies"]["fcfs"]["requests"]}
            self.assertEqual(rows["long"]["status"], "arrived_aborted")
            self.assertEqual(rows["arrived"]["status"], "arrived_aborted")
            self.assertEqual(rows["future"]["status"], "not_arrived")
            self.assertEqual(result["policies"]["edf"]["status"], "not_run")
            self.assertEqual(result["status"], "partial")

    def test_keyboardinterrupt_at_step_preserves_terminal_run_evidence(self):
        class Interrupted(FixtureBackend):
            def step(self, state):
                if state.step_index == 1:
                    raise KeyboardInterrupt()
                return super().step(state)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            from ditserve.benchmark import runner
            original = runner._run_policy
            def interrupted_run(t, policy, adapter, directory, budget):
                return original(t, policy, Interrupted(), directory, budget)
            requests = [request("long", "long", 0, 100), request("arrived", "short", 3, 100),
                        request("future", "short", 100, 200)]
            with patch.object(runner, "_run_policy", side_effect=interrupted_run):
                result = run_suite(trace(profile, requests), ["fcfs", "edf"], "cpu-test", root / "run")
            persisted = verify_seal(read_json(root / "run" / "run.json"))
            self.assertEqual(persisted, result)
            first = persisted["policies"]["fcfs"]
            self.assertEqual(first["error"], "KeyboardInterrupt")
            rows = {r["request_id"]: r for r in first["requests"]}
            self.assertEqual(rows["long"]["status"], "arrived_aborted")
            self.assertEqual(rows["arrived"]["status"], "arrived_aborted")
            self.assertEqual(rows["future"]["status"], "not_arrived")
            self.assertTrue(all(r["reason"] == "KeyboardInterrupt" for r in rows.values()))
            self.assertEqual(persisted["policies"]["edf"]["status"], "not_run")

    def test_keyboardinterrupt_in_profile_keeps_completed_sample_receipt(self):
        class InterruptedProfile(FixtureBackend):
            def prepare(self, request):
                if request["request_id"].endswith("-1"):
                    raise KeyboardInterrupt()
                return super().prepare(request)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = dataset()
            ds["samples"].append(dict(sample_id="dev2", prompt="second development", split="dev"))
            ds = seal(ds)
            with patch("ditserve.benchmark.runner.make_backend", return_value=InterruptedProfile()):
                result = build_profile(config(), ds, "cpu-test", root / "profile")
            persisted = verify_seal(read_json(root / "profile" / "profile.json"))
            self.assertEqual(persisted, result)
            self.assertEqual(result["status"], "partial")
            self.assertTrue(result["interrupted"])
            self.assertEqual(result["error"], "KeyboardInterrupt")
            self.assertEqual(len(result["completed_measurements"]), 1)
            self.assertEqual(result["shapes"], {})

    def test_keyboardinterrupt_during_reference_saves_unstarted_policy_records(self):
        class InterruptedReference(FixtureBackend):
            def step(self, state):
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            with patch("ditserve.benchmark.runner.make_backend", return_value=InterruptedReference()):
                result = run_suite(trace(profile), ["fcfs"], "cpu-test", root / "run")
            persisted = verify_seal(read_json(root / "run" / "run.json"))
            self.assertEqual(persisted, result)
            self.assertEqual(result["error"], "KeyboardInterrupt")
            self.assertTrue(result["interrupted"])
            self.assertEqual(result["policies"]["fcfs"]["status"], "not_run")
            self.assertEqual(len(result["policies"]["fcfs"]["requests"]), 2)

    def test_equal_deadline_keeps_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            requests = [request("z-first", "long", 0, 100), request("a-later", "short", 3, 100)]
            self.assertEqual(select_ready("edf", requests, "z-first", {}, profile)["request_id"], "z-first")

    def test_timeout_preserves_all_arrivals_and_checks_safe_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            frozen = trace(profile)
            frozen["config"]["timeout_scale"] = .2
            frozen = seal(frozen)
            result = run_suite(frozen, ["fcfs"], "cpu-test", root / "run")
            run = result["policies"]["fcfs"]
            self.assertEqual(run["horizon_ms"], 5.)
            self.assertEqual([r["status"] for r in run["requests"]], ["timed_out", "timed_out"])
            self.assertEqual(run["requests"][0]["events"][-1]["kind"], "step")
            self.assertTrue(all(r["finish_ms"] is None for r in run["requests"]))

    def test_real_arrivals_continue_while_scheduler_is_blocked(self):
        clock = RealClock()
        arrival = ArrivalSource([request("one", "short", 0, 1000), request("two", "short", 15, 1000)], clock)
        arrival.start()
        try:
            time.sleep(.07)  # Represents one non-preemptible model stage.
            submitted = arrival.collect()
            self.assertEqual(set(submitted), {"one", "two"})
            self.assertGreaterEqual(submitted["two"], 15)
            self.assertLess(submitted["two"], 65)
        finally:
            arrival.close()

    def test_gpu_unavailable_is_unverified_and_never_claims_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            profile = seal(dict(profile, backend="wan"))
            with patch("ditserve.benchmark.runner.make_backend", side_effect=RuntimeError("CUDA unavailable")):
                result = run_suite(trace(profile), ["fcfs"], "wan", root / "run")
            self.assertEqual(result["status"], "unverified")
            self.assertEqual(result["measurement_kind"], "unverified")
            self.assertFalse(result["measured"])
            self.assertEqual(len(result["policies"]["fcfs"]["requests"]), 2)

    def test_fresh_directory_and_trace_hash_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            t = trace(profile)
            t["requests"][0]["prompt"] = "changed"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run_suite(t, ["fcfs"], "cpu-test", root / "run")
            self.assertFalse((root / "run").exists())
            (root / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                run_suite(trace(profile), ["fcfs"], "cpu-test", root / "existing")

    def test_no_fake_core_policy_alias_or_adaptive_plugin(self):
        self.assertEqual(core_policy("core:edf"), "edf")
        for name in ("fcfs", "srtf", "previs"):
            with self.assertRaises(ValueError):
                core_policy(name)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = self.make_profile(root)
            with self.assertRaisesRegex(ValueError, "calibrated adapter"):
                run_suite(trace(profile), ["previs"], "cpu-test", root / "run")


if __name__ == "__main__":
    unittest.main()
