import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.directions import direction1 as d1
from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.scheduling.legacy import (
    GPU_SOURCE_FILES, TARGET, build_legacy_plan, legacy_admission_seconds, merge_legacy, run_legacy,
)


def fixture(root):
    archive = root / "sources/direction1-online"
    config = dict(direction1=dict(steps=32, policies=list(d1.POLICIES), online_count=24,
                                 online_seeds=[101, 102, 103], online_loads=[.5, .9], patterns=["ordinary", "burst"]))
    dependencies = dict(torch="test", diffusers="test")
    atomic_json(root / "protocol.json", dict(config=config, configuration_sha256=digest(config)))
    atomic_json(root / "direction1/worker-config.json", config)
    hashes = {}
    for relative in GPU_SOURCE_FILES:
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture\n")
        hashes[relative] = file_digest(path)
    manifest = dict(config_sha256=digest(config), source_files=hashes,
                    source_sha256=digest(hashes), execution_root=str(archive))
    manifest["sha256"] = digest(manifest)
    atomic_json(archive / "manifest.json", manifest)
    atomic_json(root / "sources/direction1-correctness/manifest.json", manifest)
    atomic_json(root / "direction1/supervisor-correctness.json", dict(status="passed", eligible=True,
                returncode=0, config_sha256=digest(config), source_sha256=manifest["source_sha256"]))
    service = dict(video=100., image=2.)
    rows = []
    for seed in (101, 102, 103):
        for load in (.5, .9):
            for pattern in ("ordinary", "burst"):
                requests = d1.frozen_requests(count=24, seed=seed, load=load, pattern=pattern,
                                               video_ms=service["video"], image_ms=service["image"])
                for policy in (d1.POLICIES[:2] if (seed, load, pattern) == TARGET else d1.POLICIES):
                    finished = [dict(row, status="completed", start_ms=row["arrival_ms"],
                                     finish_ms=row["arrival_ms"]+1.) for row in requests]
                    rows.append(dict(seed=seed, load=load, pattern=pattern, policy=policy,
                                     trace_sha256=d1._digest(requests), rows=finished,
                                     elapsed_ms=max(row["finish_ms"] for row in finished)+1.))
    atomic_json(root / "direction1/direction1-online.json", dict(rows=rows, frozen_service_ms=service,
                config_sha256=d1._digest(config), execution=dict(dependencies=dependencies),
                active_trace=dict(policy="block-1", complete=False)))
    atomic_json(root / "direction1/direction1-profile.json", dict(config_sha256=d1._digest(config),
                eligible=True, selected_shape=dict(height=480, width=832, frames=81)))
    atomic_json(root / "direction1/supervisor-online.json", dict(config_sha256=digest(config),
                source_sha256=manifest["source_sha256"], budget_stopped=True,
                final_occupancy=dict(status="exclusive_observed", foreign=[], devices=["GPU-A100"])))
    atomic_json(root / "direction1/environment-online.json", dict(versions=dependencies))
    return rows


def completion(plan):
    rows = []
    for policy in plan["policies"]:
        finished = [dict(row, status="completed", start_ms=row["arrival_ms"],
                         finish_ms=row["arrival_ms"]+1.) for row in plan["target_requests"]]
        rows.append(dict(plan["target"], policy=policy, trace_sha256=plan["trace_sha256"], rows=finished,
                         elapsed_ms=max(row["finish_ms"] for row in finished)+1.,
                         source_sha256=plan["source"]["source_sha256"],
                         config_sha256=plan["source"]["config_sha256"], device_identity=dict(uuid="GPU-A100"),
                         dependencies=plan["source"]["dependencies"]))
    return rows


class SchedulingLegacyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        fixture(self.root)

    def test_exactly_eleven_old_pairs_plus_one_entire_new_pair(self):
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        plan = build_legacy_plan(self.root)
        self.assertEqual((len(plan["inherited"]), len(plan["excluded_partial_group"])), (55, 2))
        self.assertEqual(plan["excluded_active_trace"]["policy"], "block-1")
        result = merge_legacy(plan, completion(plan))
        self.assertEqual((len(result["rows"]), result["paired_groups"], result["new_replays"]), (60, 12, 5))
        self.assertTrue(all(summary["paired_traces"] == 12 for summary in result["summaries"].values()))
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_three_new_arms_cannot_be_filled_from_historical_partial_pair(self):
        plan = build_legacy_plan(self.root)
        with self.assertRaisesRegex(ValueError, "five-policy"):
            merge_legacy(plan, completion(plan)[2:])

    def test_changed_device_dependency_inputs_and_executable_rejected(self):
        plan = build_legacy_plan(self.root)
        for field, value in (("device_identity", "GPU-V100"), ("dependencies", {}),
                             ("source_sha256", "wrong"), ("config_sha256", "wrong")):
            with self.subTest(field=field):
                rows = completion(plan)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    merge_legacy(plan, rows)
        rows = completion(plan)
        rows[0]["rows"][0]["deadline_ms"] += 1.
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            merge_legacy(plan, rows)

    def test_mutated_historical_evidence_and_archive_cannot_merge(self):
        plan = build_legacy_plan(self.root)
        path = self.root / "direction1/environment-online.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "historical evidence changed"):
            merge_legacy(plan, completion(plan))
        fixture(self.root)
        plan = build_legacy_plan(self.root)
        path = self.root / "sources/direction1-online" / GPU_SOURCE_FILES[0]
        path.write_text("# edited after planning")
        with self.assertRaisesRegex(ValueError, "archived source changed"):
            merge_legacy(plan, completion(plan))
        with self.assertRaisesRegex(ValueError, "archived source changed"):
            build_legacy_plan(self.root)

    def test_cached_new_metrics_cannot_control_merged_gate(self):
        plan = build_legacy_plan(self.root)
        rows = completion(plan)
        reference = merge_legacy(plan, rows)
        for row in rows:
            row.update(metrics=dict(goodput_rps=-100.), image_metrics=dict(wait_p95_ms=1e9),
                       video_metrics=dict(on_time_fraction=0.))
        actual = merge_legacy(plan, rows)
        self.assertEqual(actual["summaries"], reference["summaries"])
        self.assertEqual(actual["decision"], reference["decision"])

    def test_incomplete_or_duplicate_old_replays_rejected(self):
        path = self.root / "direction1/direction1-online.json"
        online = read_json(path)
        online["rows"][0]["rows"][0]["status"] = "failed"
        atomic_json(path, online)
        with self.assertRaisesRegex(ValueError, "unfinished historical"):
            build_legacy_plan(self.root)
        fixture(self.root)
        online = read_json(path)
        online["rows"][1] = copy.deepcopy(online["rows"][0])
        atomic_json(path, online)
        with self.assertRaisesRegex(ValueError, "duplicate historical"):
            build_legacy_plan(self.root)

    def test_changed_plan_fails_self_digest(self):
        plan = build_legacy_plan(self.root)
        plan["target_requests"][0]["deadline_ms"] += 1.
        with self.assertRaisesRegex(ValueError, "plan mutated"):
            merge_legacy(plan, completion(plan))

    def test_legacy_admission_includes_five_arms_and_full_warmup(self):
        plan = build_legacy_plan(self.root)
        estimate = legacy_admission_seconds(plan)
        self.assertEqual(set(estimate["policy_seconds"]), set(d1.POLICIES))
        self.assertGreaterEqual(estimate["admitted_overhead_seconds"], plan["frozen_service_ms"]["video"] / 1000.)
        self.assertEqual(estimate["required_seconds"], 1.3 * (sum(estimate["policy_seconds"].values())
                         + estimate["admitted_overhead_seconds"]) + 45.)

    def test_run_legacy_bridge_only_after_whole_group_budget_admission(self):
        from ditserve.scheduling.budget import BudgetStop, DeadlineCheck
        config = dict(legacy_run=str(self.root), gpu_uuid="GPU-A100", sana_profile="a100_bf16")
        output = self.root / "new-run"
        result = {}
        check = DeadlineCheck(100., monotonic=lambda: 99.)
        with patch("ditserve.scheduling.legacy._invoke_bridge") as bridge:
            actual = run_legacy(config, output, check, result, lambda: None)
        bridge.assert_not_called()
        self.assertEqual(actual["status"], "incomplete")
        self.assertTrue(actual["budget_stopped"])
        self.assertEqual(actual["missing_policies"], list(d1.POLICIES))

    def test_v100_only_constraint_defers_legacy_without_reading_or_loading_archive(self):
        result = {}
        config = dict(device_preference=["V100"], legacy_run="does-not-exist")
        with patch("ditserve.scheduling.legacy._invoke_bridge") as bridge:
            actual = run_legacy(config, self.root / "new-run", lambda: None, result, lambda: None)
        bridge.assert_not_called()
        self.assertEqual((actual["status"], actual["reason_code"]), ("blocked", "device_constraint"))
        self.assertTrue(actual["deferred"])

    def test_bridge_result_merges_without_loading_current_models(self):
        from ditserve.scheduling.budget import DeadlineCheck
        config = dict(legacy_run=str(self.root), gpu_uuid="GPU-A100", sana_profile="a100_bf16")
        def bridge(config, plan, root, check):
            value = dict(status="complete", records=completion(plan))
            atomic_json(root / "new-replays.json", value)
            return value
        with patch("ditserve.scheduling.legacy._invoke_bridge", side_effect=bridge):
            result = run_legacy(config, self.root / "new-run", DeadlineCheck(1000., monotonic=lambda: 0.), {}, lambda: None)
        self.assertEqual((result["status"], result["new_replays"], result["paired_groups"]), ("passed", 5, 12))
        self.assertEqual(result["missing_policies"], [])
        self.assertTrue((self.root / "new-run/legacy/merged.json").is_file())

    def test_historical_correctness_source_must_match_online_gpu_code(self):
        path = self.root / "sources/direction1-correctness/manifest.json"
        manifest = read_json(path)
        manifest["source_files"][GPU_SOURCE_FILES[0]] = "unqualified-change"
        manifest["source_sha256"] = digest(manifest["source_files"])
        manifest["sha256"] = digest({key: value for key, value in manifest.items() if key != "sha256"})
        atomic_json(path, manifest)
        supervisor_path = self.root / "direction1/supervisor-correctness.json"
        supervisor = read_json(supervisor_path)
        supervisor["source_sha256"] = manifest["source_sha256"]
        atomic_json(supervisor_path, supervisor)
        with self.assertRaisesRegex(ValueError, "differs from the historical correctness"):
            build_legacy_plan(self.root)


if __name__ == "__main__":
    unittest.main()
