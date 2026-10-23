"""Small real-finalizer contracts for selective workflow retention."""

import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve import results
from ditserve.evaluation import workflow_retention as retention


class WorkflowRetentionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)
        self.root = self.base / "workflow" / "run01"
        self.root.mkdir(parents=True)
        self.destination = self.base / "results" / "workflow" / "run01"
        self.run = dict(schema="ditserve.run.v1", experiment="workflow", run_id="run01", status="completed",
                        metrics={"precise": 1.2345678901234567}, config={}, phases=[], requests=[], failures=[],
                        budget={}, retention={"policy": "retain all raw evidence; prune=False", "pending_dependencies": []}, artifacts=[])
        self.write_run()
        self.write("ANALYSIS.html", b"complete report")

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path

    def write_run(self):
        self.write("run.json", self.run)

    def final(self, **kwargs):
        return retention.finalize_workflow(self.root, self.destination, **kwargs)

    def trace(self, name="E2-f17", *, matching=True, completed=True):
        capture = self.write("traces/" + name + ".nsys-rep", b"original capture")
        binding = results._read(capture)[0]
        if not matching:
            binding["sha256"] = "0" * 64
        parsed = self.write("traces/" + name + ".parsed.json", dict(schema_version="ditserve.workflow_trace.v1",
                            status="completed" if completed else "partial", postprocess={"capture": binding}))
        sqlite = self.write("traces/" + name + ".sqlite", b"derived sqlite data")
        return sqlite, capture, parsed

    def profile(self, name="outputs/17/video.npy", payload=b"payload", *, complete=True, duplicate=True):
        document = dict(schema="ditserve.workflow.encoder.v1", variant="T0", status="completed")
        self.write("encoder-v2/T0/result.json", document)
        self.write("profiling/encoder/T0/result.json", dict(document, status="completed" if complete else "failed"))
        self.write("encoder-v2/T0/" + name, payload if duplicate else b"different normal output")
        return self.write("profiling/encoder/T0/" + name, payload)

    def e4(self, *, completed=True, finite=True):
        directory = "overlap/E4-f17/"
        self.write(directory + "run.json", dict(status="complete" if completed else "running"))
        request = dict(index=0, finite=finite, sha256="a" * 64, shape=[1, 17, 2, 2, 3], dtype="float32", complete_ns=15,
                       latent=dict(sha256="b" * 64, finite=True))
        self.write(directory + "E4-r0-sync.json", dict(start_ns=10, end_ns=20, requests=[request], output_mode="npy"))
        return self.write(directory + "E4-r0-sync-request0.npy", b"e4 redundant payload")

    def test_preserves_key_evidence_and_measurements_and_prunes_only_four_scopes(self):
        sqlite, capture, parsed = self.trace()
        profile = self.profile(payload=b"profile")
        e4 = self.e4()
        duplicate = self.write("encoder/T0/numeric/output.npz", b"exact duplicate")
        preserved = [capture, parsed, self.root / "ANALYSIS.html"]
        for name, value in (("encoder-v2/T0/numeric/output.npz", b"exact duplicate"),
                            ("encoder-v2/T1/numeric/latent.npz", b"latent"),
                            ("embedding-replay/output.npy", b"replay"),
                            ("measurements/quality/output.npy", b"quality"),
                            ("overlap/E2-f17/output.npy", b"normal"),
                            ("ncu/raw.ncu-rep", b"ncu"), ("gallery/image.png", b"png"),
                            ("sources/experiment.py", b"source"), ("logs/experiment.log", b"log"),
                            ("encoder/T0/numeric/not_duplicate.npz", b"unique"),
                            ("traces/probe.sqlite", b"unparsed")):
            preserved.append(self.write(name, value))
        result = self.final()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["deleted"], 4)
        for path in (sqlite, profile, e4, duplicate):
            self.assertFalse(path.exists(), path)
        for path in preserved:
            self.assertTrue(path.exists(), path)
        archived = results.load(self.destination)
        for key in self.run.keys() - {"retention", "artifacts"}:
            self.assertEqual(archived["run"][key], self.run[key])
        e4_artifact = next(row for row in archived["run"]["artifacts"] if row["path"].endswith("request0.npy"))
        self.assertTrue(e4_artifact["cleanup_evidence"]["mode_json"]["requests"][0]["finite"])
        self.assertEqual(retention.archived_run(self.root), archived["run"])

    def test_trace_requires_bound_capture_and_complete_parse(self):
        mismatch = self.trace("mismatch", matching=False)[0]
        partial = self.trace("partial", completed=False)[0]
        unparsed = self.write("traces/unparsed.sqlite", b"unparsed")
        value = self.final()
        self.assertEqual(value["status"], "completed", value)
        self.assertEqual(value["deleted"], 0)
        self.assertTrue(all(path.exists() for path in (mismatch, partial, unparsed)))

    def test_e4_requires_completed_parent_and_per_request_numeric_completion(self):
        path = self.e4(completed=False)
        self.assertEqual(self.final(dry_run=True)["files"], 0)
        path = self.e4(finite=False)
        self.assertEqual(self.final(dry_run=True)["files"], 0)
        path = self.e4()
        self.write("overlap/E4-f17/E4-r0-sync-request1.npy", b"missing numeric row")
        self.assertEqual(self.final(dry_run=True)["files"], 1)
        self.assertTrue(path.exists())

    def test_dry_run_and_keep_raw_leave_inventory_unchanged(self):
        path = self.profile()
        before = (self.root / "run.json").read_bytes()
        value = self.final(dry_run=True)
        self.assertEqual(value["files"], 1)
        self.assertEqual(value["bytes"], 7)
        self.assertEqual((self.root / "run.json").read_bytes(), before)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.final(keep_raw=True)["status"], "completed")
        self.assertEqual((self.root / "run.json").read_bytes(), before)
        self.assertTrue(path.exists())
        self.assertEqual(retention.archived_run(self.root), self.run)
        self.assertEqual(self.final()["status"], "completed")
        self.assertFalse(path.exists())

    def test_active_control_documents_or_open_dependencies_block_before_mutation(self):
        path = self.profile()
        before = (self.root / "run.json").read_bytes()
        for name, value in (("study.json", {"status": "running"}),
                            ("budget.json", {"active": {"job_id": "live"}}),
                            ("operator-notes.json", {"active_normal_session": 42}),
                            ("operator-notes.json", {"pending_dependencies": ["quality"]})):
            with self.subTest(name=name, value=value):
                control = self.write(name, value)
                self.assertEqual(self.final()["status"], "protected")
                self.assertEqual((self.root / "run.json").read_bytes(), before)
                self.assertTrue(path.exists())
                control.unlink()

    def test_all_producer_locks_block_before_modifying_run(self):
        self.profile()
        before = (self.root / "run.json").read_bytes()
        for name in (*retention._LOCKS, ".retention.lock"):
            with self.subTest(name=name):
                with (self.root / name).open("a+b") as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertEqual(self.final()["status"], "protected")
                self.assertEqual((self.root / "run.json").read_bytes(), before)

    def test_budget_ledger_lock_is_delegated_to_shared_finalizer(self):
        self.write("budget.json", {"active": None})
        self.write("budget.lock", b"")
        self.run["budget"]["ledger"] = "budget.json"
        self.write_run()
        self.profile()
        result = self.final()
        self.assertEqual(result["status"], "completed", result)

    def test_symlink_candidate_and_ancestor_are_rejected(self):
        outside = self.base / "outside.npy"
        outside.write_bytes(b"preserve")
        directory = self.root / "profiling"
        directory.mkdir()
        (directory / "output.npy").symlink_to(outside)
        before = (self.root / "run.json").read_bytes()
        result = self.final()
        self.assertEqual(result["status"], "deferred")
        self.assertIn("symbolic links", result["reason"])
        self.assertEqual((self.root / "run.json").read_bytes(), before)
        self.assertEqual(outside.read_bytes(), b"preserve")
        linked = self.base / "linked"
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            retention.finalize_workflow(linked, self.destination)

    def test_existing_control_binding_cannot_be_reclassified(self):
        path = self.profile(payload=b"protected output")
        binding = results._read(path)[0]
        self.run["artifacts"].append(dict(binding, path=str(path.relative_to(self.root)), kind="control", dependencies=[]))
        self.write_run()
        result = self.final()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["deleted"], 0)
        self.assertTrue(path.exists())

    def test_existing_unrelated_raw_whitelist_does_not_bypass_policy(self):
        path = self.write("measurements/output.npy", b"key output")
        self.run["artifacts"].append(dict(results._read(path)[0], path="measurements/output.npy", kind="raw", dependencies=[]))
        self.write_run()
        result = self.final()
        self.assertEqual(result["status"], "deferred", result)
        self.assertTrue(path.exists())

    def test_profile_unique_failed_and_unrecognized_outputs_are_preserved(self):
        unique = self.profile(duplicate=False)
        self.assertEqual(self.final(dry_run=True)["files"], 0)
        failed = self.profile(complete=False)
        self.assertEqual(self.final(dry_run=True)["files"], 0)
        unknown = self.write("profiling/unknown.npy", b"unique quality data")
        result = self.final()
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(all(path.exists() for path in (unique, failed, unknown)))

    def test_profile_e4_requires_matching_completed_numeric_reference(self):
        self.e4()
        normal = self.root / "overlap/E4-f17"
        for name in ("run.json", "E4-r0-sync.json"):
            self.write("profiling/overlap/E4-f17/" + name, (normal / name).read_bytes())
        profile = self.write("profiling/overlap/E4-f17/E4-r0-sync-request0.npy", b"profile video")
        plan = self.final(dry_run=True)
        self.assertEqual(plan["categories"]["profiling_array"]["files"], 1)
        mode = json.loads((normal / "E4-r0-sync.json").read_text())
        mode["requests"][0]["sha256"] = "c" * 64
        self.write("profiling/overlap/E4-f17/E4-r0-sync.json", mode)
        self.assertNotIn("profiling_array", self.final(dry_run=True)["categories"])
        self.assertTrue(profile.exists())

    def test_duplicate_change_after_comparison_blocks_inventory_write(self):
        path = self.profile()
        evidence = retention._profile_evidence(self.root, path, {})
        self.assertIsNotNone(evidence)
        path.write_bytes(b"changed after duplicate comparison")
        before = (self.root / "run.json").read_bytes()
        with patch.object(retention, "_candidates", return_value=[(path, "profiling_array", evidence)]):
            result = self.final()
        self.assertEqual(result["status"], "deferred", result)
        self.assertIn("duplicate changed", result["reason"])
        self.assertTrue(path.exists())
        self.assertEqual((self.root / "run.json").read_bytes(), before)

    def test_existing_archive_must_match_measurements(self):
        self.assertEqual(results.finalize(self.root, self.destination, prune=False)["status"], "completed")
        self.run["metrics"]["precise"] = 99
        self.write_run()
        self.profile()
        before = (self.root / "run.json").read_bytes()
        result = self.final()
        self.assertEqual(result["status"], "deferred", result)
        self.assertEqual((self.root / "run.json").read_bytes(), before)

    def test_inventory_save_then_publication_failure_can_retry(self):
        self.assertEqual(results.finalize(self.root, self.destination, prune=False)["status"], "completed")
        path = self.profile()
        with patch.object(results, "_publish", side_effect=OSError("publication interrupted")):
            failed = self.final()
        self.assertEqual(failed["status"], "deferred", failed)
        self.assertTrue(path.exists())
        retried = self.final()
        self.assertEqual(retried["status"], "completed", retried)
        self.assertFalse(path.exists())

    def test_partial_cleanup_retries_without_rescanning_and_completed_is_immutable(self):
        first = self.profile("outputs/17/video.npy", b"first")
        second = self.profile("outputs/33/video.npy", b"second")
        original = Path.unlink
        def interrupt(path, *args, **kwargs):
            if path == second:
                raise OSError("cleanup interrupted")
            return original(path, *args, **kwargs)
        with patch.object(Path, "unlink", interrupt):
            failed = self.final()
        self.assertEqual(failed["status"], "deferred", failed)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        self.assertIsNotNone(retention.archived_run(self.root))
        with patch.object(retention, "_candidates", side_effect=AssertionError("must not rescan")):
            retried = self.final()
        self.assertEqual(retried["status"], "completed", retried)
        self.assertFalse(second.exists())
        before = {path.name: path.read_bytes() for path in self.destination.iterdir()}
        self.assertEqual(self.final()["status"], "completed")
        self.assertEqual({path.name: path.read_bytes() for path in self.destination.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
