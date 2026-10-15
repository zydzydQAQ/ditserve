"""CPU-only parent-identity and explicit archive/deletion boundary tests."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ditserve.optimizations import followup_queue as queue
from ditserve.optimizations.followup_study import event_summary
from ditserve.research.common import atomic_json, file_digest, read_json


class ParentIdentityTests(unittest.TestCase):
    def test_matching_parent_is_active_and_absent_parent_is_finished(self):
        expected = dict(pid=1234, start_ticks=123456, boot_id="boot-A")
        with patch.object(queue, "process_identity", return_value=dict(expected)):
            self.assertTrue(queue.parent_active(expected))
        with patch.object(queue, "process_identity", return_value=None):
            self.assertFalse(queue.parent_active(expected))

    def test_pid_reuse_and_boot_change_fail_closed(self):
        expected = dict(pid=1234, start_ticks=123456, boot_id="boot-A")
        for actual in (dict(expected, start_ticks=123457), dict(expected, boot_id="boot-B")):
            with patch.object(queue, "process_identity", return_value=actual):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    queue.parent_active(expected)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "work/run01"
        self.ledger = self.base / "work/gpu-budget.json"
        atomic_json(self.ledger, dict(active=None, used_gpu_seconds=10.))

    def phase(self, name="finish_copy", *, failed=False):
        destination = self.root / name / "attempt000"
        replay = dict(status="completed", complete=True, events=[
            dict(kind="video_finish", start_ms=0., end_ms=10.),
            dict(kind="video_decode", start_ms=0., end_ms=7.),
            dict(kind="video_convert", start_ms=7., end_ms=10.)],
            requests=[dict(request_id="r%03d" % i, kind="image" if i < 22 else "video",
                status="completed", arrival_ms=2., start_ms=9., finish_ms=12.) for i in range(24)])
        event_summary(replay)
        result = dict(status="failed" if failed else "completed", quality=dict(passed=True),
            local=[dict(max_abs=0.)], online=dict(status="failed" if failed else "completed",
                expected_replays=8, expected_requests=192,
                replays=[copy.deepcopy(replay) for _ in range(8)]))
        result_path = destination / "phase-result.json"
        atomic_json(result_path, result)
        atomic_json(destination / "supervisor.json", dict(status="incomplete" if failed else "completed",
            returncode=1 if failed else 0, cleanup_confirmed=True,
            charge=dict(status="failed" if failed else "completed")))
        paths = [destination / "worker.log", destination / "online-partials/replay00/trace.json",
                 destination / "compiler-cache/nested/kernel.bin", destination / "triton-cache/kernel.cubin"]
        for index, path in enumerate(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("evidence-%d\n" % index).encode())
        unrelated = destination / "must-retain.txt"
        unrelated.write_text("not in cleanup allowlist")
        return destination, result_path, paths, unrelated

    def test_export_failure_prevents_all_rewrites_and_deletions(self):
        destination, result_path, files, _ = self.phase()
        before = {path: path.read_bytes() for path in files + [result_path]}
        export = Mock(side_effect=ValueError("workbook round-trip failed"))
        with self.assertRaisesRegex(ValueError, "round-trip"):
            queue.cleanup(self.root, self.ledger, export=export)
        export.assert_called_once_with()
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(read_json(self.root / "cleanup.json")["status"], "planned")

    def test_failure_phase_raw_evidence_and_outputs_are_retained(self):
        _, result_path, files, _ = self.phase(failed=True)
        before = {path: path.read_bytes() for path in files + [result_path]}
        plan = queue.cleanup(self.root, self.ledger, export=Mock())
        self.assertEqual(plan["files"], [])
        self.assertEqual(plan["rewrites"], [])
        for path, original in before.items():
            self.assertEqual(path.read_bytes(), original)

    def test_success_exports_exact_manifest_before_mutation_and_preserves_aggregates(self):
        destination, result_path, files, unrelated = self.phase()
        before = read_json(result_path)
        hashes = {str(path): (file_digest(path), path.stat().st_size) for path in files}
        calls = []
        def export():
            plan = read_json(self.root / "cleanup.json")
            calls.append(plan["status"])
            self.assertEqual({item["path"] for item in plan["files"]}, set(hashes))
            if len(calls) == 1:
                self.assertEqual(read_json(result_path), before)
                for item in plan["files"]:
                    self.assertEqual((item["sha256"], item["bytes"]), hashes[item["path"]])
                    self.assertTrue(Path(item["path"]).is_file())
            else:
                self.assertTrue(all(not path.exists() for path in files))
        plan = queue.cleanup(self.root, self.ledger, export=export)
        self.assertEqual(calls, ["planned", "completed"])
        self.assertTrue(unrelated.is_file())
        self.assertTrue(all(item["status"] == "deleted" for item in plan["files"]))
        compact = read_json(result_path)
        self.assertEqual(compact["quality"], before["quality"])
        self.assertEqual(compact["local"], before["local"])
        for original, row in zip(before["online"]["replays"], compact["online"]["replays"]):
            self.assertNotIn("events", row)
            self.assertFalse(row["raw_events_available"])
            self.assertEqual(row["event_summary"], original["event_summary"])
            self.assertEqual(row["requests"], original["requests"])
        self.assertEqual(plan["rewrites"][0]["events_removed"], 24)
        self.assertEqual(plan["rewrites"][0]["after_sha256"], file_digest(result_path))

    def test_missing_event_summary_prevents_export_or_cleanup(self):
        _, result_path, files, _ = self.phase()
        result = read_json(result_path)
        del result["online"]["replays"][0]["event_summary"]
        atomic_json(result_path, result)
        export = Mock()
        with self.assertRaisesRegex(ValueError, "aggregate"):
            queue.cleanup(self.root, self.ledger, export=export)
        export.assert_not_called()
        self.assertTrue(all(path.is_file() for path in files))

    def test_tampered_aggregate_or_wait_attribution_cannot_authorize_event_deletion(self):
        for field in ("aggregate", "wait"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                original_root = self.root
                self.root = Path(temporary) / "run01"
                try:
                    _, result_path, files, _ = self.phase()
                    result = read_json(result_path)
                    row = result["online"]["replays"][0]
                    if field == "aggregate":
                        row["event_summary"]["by_kind"]["video_finish"]["sum_ms"] = 999.
                    else:
                        row["requests"][0]["wait_attribution"]["by_event_kind_ms"]["video_finish"] = 999.
                    atomic_json(result_path, result)
                    export = Mock()
                    with self.assertRaises(ValueError):
                        queue.cleanup(self.root, self.ledger, export=export)
                    export.assert_not_called()
                    self.assertTrue(all(path.is_file() for path in files))
                finally:
                    self.root = original_root

    def test_live_lease_prevents_archive_and_any_deletion(self):
        _, result_path, files, _ = self.phase()
        atomic_json(self.ledger, dict(active=dict(pid=42)))
        export = Mock()
        with self.assertRaisesRegex(ValueError, "active GPU lease"):
            queue.cleanup(self.root, self.ledger, export=export)
        export.assert_not_called()
        self.assertTrue(all(path.exists() for path in files))

    def test_changed_planned_file_is_not_deleted(self):
        _, _, files, _ = self.phase()
        changed = files[0]
        def export():
            changed.write_bytes(b"changed during archive")
        with self.assertRaisesRegex(ValueError, "changed before deletion"):
            queue.cleanup(self.root, self.ledger, export=export)
        self.assertTrue(changed.exists())
        self.assertEqual(changed.read_bytes(), b"changed during archive")

    def test_symlink_file_and_directory_are_not_followed_for_cleanup(self):
        for mode in ("file", "directory"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                original_root = self.root
                self.root = Path(temporary) / "run01"
                try:
                    destination, _, files, _ = self.phase()
                    outside = self.base / ("external-" + mode)
                    outside.mkdir(exist_ok=True)
                    protected = outside / "protected.bin"
                    protected.write_bytes(b"must survive")
                    if mode == "file":
                        files[0].unlink()
                        files[0].symlink_to(protected)
                    else:
                        (destination / "compiler-cache/external").symlink_to(outside, target_is_directory=True)
                    with self.assertRaises(ValueError):
                        queue.cleanup(self.root, self.ledger, export=Mock())
                    self.assertEqual(protected.read_bytes(), b"must survive")
                finally:
                    self.root = original_root

    def test_cleanup_receipt_cannot_be_overwritten_by_repeat(self):
        self.phase()
        queue.cleanup(self.root, self.ledger, export=Mock())
        receipt = (self.root / "cleanup.json").read_bytes()
        with self.assertRaises(FileExistsError):
            queue.cleanup(self.root, self.ledger, export=Mock())
        self.assertEqual((self.root / "cleanup.json").read_bytes(), receipt)


if __name__ == "__main__":
    unittest.main()
