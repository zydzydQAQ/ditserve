"""Temporary real Excel/receipt fixtures; no experiment logs or GPUs touched."""

import contextlib
import copy
import json
from pathlib import Path
import sys
import subprocess
import unittest
from unittest.mock import patch

import test_optimizations_cache_followup_report as fixtures
from ditserve import compact_results
from ditserve.optimizations import cache_followup_report as reporting
from ditserve.optimizations.cache_archive_view import ArchiveIntegrityError, SCHEMA, load_archive_report
from ditserve.research.common import atomic_json, digest, file_digest, read_json


class CacheArchiveViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = cls.fixture = fixtures.CacheFollowupReportTests("test_not_started_report_is_explicit_and_creates_no_ledger")
        f.setUp()
        cls.config = dict(scope="temporary archive view fixture")
        f.config_sha = digest(cls.config)
        f.complete()
        auth_path = Path(f.study["authorization"]["path"])
        auth = read_json(auth_path)
        auth.update(budget_unlimited=True, authorized_gpu_seconds=None)
        f.study.update(budget_unlimited=True, authorized_gpu_seconds=None, authorization=f.bound(auth_path, auth))
        atomic_json(f.output / "study.json", f.study)
        f.budget.update(budget_unlimited=True, limit_gpu_seconds=None)
        f.terminal()
        f.online(partial=copy.deepcopy(f.rows[-1]))
        atomic_json(f.attempt / "config.json", cls.config)
        atomic_json(f.attempt / "study.json", f.study)
        isolation = f.attempt / "isolation.jsonl"
        isolation.write_text(json.dumps(dict(status="exclusive_observed", observed_at=1., owned=[], foreign=[])) + "\n")
        audit = f.output / "audits/final-integrity.json"
        cls.audit_binding = f.bound(audit, dict(status="passed", output=str(f.output), matrix_complete=True,
            cache_online_sha256=file_digest(f.attempt / "cache-online.json")))
        cls.frozen = reporting.write_report(f.output, f.ledger)
        archived = reporting.archive(f.output, f.ledger)
        assert archived["status"] == "verified", archived
        frozen_sha = file_digest(f.output / "archive-metadata/report-summary.json")
        cls.names = ("cache-online.json", "partial.json", "isolation.jsonl")
        summaries, deleted = {}, []
        for name in cls.names:
            path = f.attempt / name
            sha, size = file_digest(path), path.stat().st_size
            summaries[name] = dict(valid=True, source_metadata=dict(source=str(path), bytes=size, sha256=sha),
                frozen_report_sha256=frozen_sha, final_audit_sha256=cls.audit_binding["sha256"],
                evidence_status="aggregated_only", raw_events_recoverable=False,
                scope="test summary only; actual finalizer must retain required metric/count summaries")
            deleted.append(dict(path=str(path), sha256=sha, bytes=size, kind="summarized_text",
                                dependencies_closed=True, closure_reason="temporary completed fixture"))
        cls.log_archive = f.output / "log-archive"
        exported = compact_results.export(f.attempt, cls.log_archive, summarized_text=summaries)
        assert exported["status"] == "verified", exported
        compact_results.verify(cls.log_archive, check_sources=True)
        compact_results.prune(cls.log_archive, list(cls.names))

        def binding(path):
            return dict(path=str(path), sha256=file_digest(path), bytes=path.stat().st_size)

        cls.finalization = dict(schema_version=SCHEMA, status="completed", dependencies_closed=True,
            output=str(f.output), main_archive=binding(f.output / "archive.json"),
            main_manifest=binding(f.output / "excel/manifest.json"),
            log_manifest=binding(cls.log_archive / "manifest.json"),
            prune_receipt=binding(cls.log_archive / "pruned.json"), final_audit=cls.audit_binding,
            deleted_sources=deleted,
            retained_sources=[binding(f.attempt / "config.json"), binding(f.attempt / "study.json")])
        atomic_json(f.output / "log-finalization.json", cls.finalization)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    @contextlib.contextmanager
    def changed(self, path, value=None, *, remove=False):
        path = Path(path)
        original = path.read_bytes() if path.exists() else None
        try:
            if remove:
                path.unlink()
            elif isinstance(value, dict):
                atomic_json(path, value)
            else:
                path.write_bytes(value if value is not None else original + b"\n")
            yield
        finally:
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)

    def load(self):
        return load_archive_report(self.fixture.output, self.fixture.ledger)

    def test_verified_archive_only_keeps_frozen_conclusion_and_never_recomputes_gate(self):
        f = self.fixture
        with patch.object(reporting, "paired_gate", side_effect=AssertionError("must not reevaluate")):
            view = reporting.collect(f.output, f.ledger)
        self.assertEqual(view["evidence_mode"], "archive_only")
        self.assertEqual(view["online_qualified"], self.frozen["online_qualified"])
        self.assertEqual(view["decision"], self.frozen["decision"])
        self.assertEqual(view["current"]["raw_request_count"], 384)
        self.assertEqual(view["current"]["request_samples"], self.frozen["current"]["request_samples"])
        self.assertEqual(view["current"]["event_summaries"], self.frozen["current"]["event_summaries"])
        self.assertFalse(view["qualification_recomputed"])
        self.assertFalse(view["new_qualification_granted"])
        self.assertFalse(view["raw_event_re_evaluable"])
        self.assertFalse(view["archive_ready"])
        self.assertTrue(view["budget"]["budget_unlimited"])
        self.assertIsNone(view["budget"]["limit_gpu_seconds"])
        rendered = reporting.render_markdown(view)
        self.assertIn("仅查看已核验归档", rendered)
        self.assertIn("归档 gate", rendered)
        self.assertIn("明确清理回执", rendered)
        self.assertNotIn("[本轮原始在线证据]", rendered)
        self.assertNotIn("本地重算 gate", rendered)
        self.assertEqual(view["retention"]["original_log_files_deleted"], 3)
        # Test this in a fresh process: other CPU tests may already have imported
        # torch/numpy before the archive-only reader is exercised.
        subprocess.run([sys.executable, "-c",
            "import sys; from pathlib import Path; "
            "from ditserve.optimizations.cache_followup_report import collect; "
            "view=collect(Path(sys.argv[1]), Path(sys.argv[2])); "
            "assert view['evidence_mode'] == 'archive_only'; "
            "assert 'torch' not in sys.modules and 'numpy' not in sys.modules",
            str(f.output), str(f.ledger)], check=True, capture_output=True, text=True)

    def test_missing_log_without_finalization_or_receipt_fails_closed(self):
        f = self.fixture
        for path in (f.output / "log-finalization.json", self.log_archive / "pruned.json"):
            with self.subTest(path=path), self.changed(path, remove=True):
                with self.assertRaises(ArchiveIntegrityError):
                    reporting.collect(f.output, f.ledger)

    def test_manifest_receipt_audit_and_frozen_report_tampering_are_rejected(self):
        f = self.fixture
        paths = (f.output / "excel/manifest.json", self.log_archive / "manifest.json",
                 self.log_archive / "pruned.json", Path(self.audit_binding["path"]),
                 f.output / "archive-metadata/report-summary.json", f.output / "archive.json")
        for path in paths:
            with self.subTest(path=path), self.changed(path):
                with self.assertRaises(ArchiveIntegrityError):
                    self.load()

    def test_active_or_changed_settled_ledger_is_rejected(self):
        f = self.fixture
        budget = read_json(f.ledger)
        for value in (dict(budget, active=dict(session_id="new-active")),
                      dict(budget, used_gpu_seconds=budget["used_gpu_seconds"] + 1.)):
            with self.subTest(value=value), self.changed(f.ledger, value):
                with self.assertRaises(ArchiveIntegrityError):
                    self.load()

    def test_unchanged_retained_config_authorization_study_and_snapshot_are_required(self):
        f = self.fixture
        paths = (f.attempt / "config.json", f.attempt / "study.json",
                 Path(f.study["authorization"]["path"]), f.output / "study.json",
                 f.output / "sources/cache_k4-attempt00/python/fixture.py", f.attempt / "supervisor.json")
        for path in paths:
            with self.subTest(path=path), self.changed(path):
                with self.assertRaises(ArchiveIntegrityError):
                    self.load()
        with self.changed(f.attempt / "config.json", remove=True):
            with self.assertRaises(ArchiveIntegrityError):
                self.load()

    def test_unclosed_wrong_audit_scope_and_extra_cleanup_targets_are_rejected(self):
        f = self.fixture
        variants = [dict(self.finalization, dependencies_closed=False),
                    dict(self.finalization, retained_sources=[]),
                    dict(self.finalization, deleted_sources=self.finalization["deleted_sources"][:-1])]
        extra = copy.deepcopy(self.finalization)
        extra["deleted_sources"].append(dict(extra["deleted_sources"][0], path=str(f.attempt / "worker.log")))
        variants.append(extra)
        for value in variants:
            with self.subTest(value=value), self.changed(f.output / "log-finalization.json", value):
                with self.assertRaises(ArchiveIntegrityError):
                    self.load()

    def test_present_raw_source_and_second_archive_are_not_accepted(self):
        f = self.fixture
        with self.changed(f.attempt / "cache-online.json", b"{}\n"):
            with self.assertRaises(ArchiveIntegrityError):
                self.load()
        with patch.object(compact_results, "export", side_effect=AssertionError("no second archive")):
            self.assertEqual(reporting.archive(f.output, f.ledger)["status"], "protected")


if __name__ == "__main__":
    unittest.main()
