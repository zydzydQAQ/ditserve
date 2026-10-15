"""Small standard-library fixtures; no real result trees, models, or GPUs."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

from ditserve.optimizations import next_retention as retention
from ditserve.optimizations.next_report import _tables
from ditserve.research.common import atomic_json, digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def workbook(path, report, corrupt_cell=False):
    """Emit the same literal-cell subset used by next_report, without openpyxl."""
    tables = _tables(report)
    root = ET.Element(retention.NS + "workbook")
    sheets = ET.SubElement(root, retention.NS + "sheets")
    relationships = ET.Element("Relationships")
    with zipfile.ZipFile(path, "w") as archive:
        for index, (name, values) in enumerate(tables.items(), 1):
            ET.SubElement(sheets, retention.NS + "sheet", {"name": name, retention.REL + "id": "rId%d" % index})
            ET.SubElement(relationships, "Relationship", Id="rId%d" % index, Target="worksheets/sheet%d.xml" % index)
            sheet = ET.Element(retention.NS + "worksheet")
            data = ET.SubElement(sheet, retention.NS + "sheetData")
            for row_index, row in enumerate(values, 1):
                entry = ET.SubElement(data, retention.NS + "row", r=str(row_index))
                for column, value in enumerate(row, 1):
                    if value is None:
                        continue
                    label, number = "", column
                    while number:
                        number, rem = divmod(number - 1, 26)
                        label = chr(rem + 65) + label
                    cell = ET.SubElement(entry, retention.NS + "c", r=label + str(row_index), t="inlineStr")
                    item = ET.SubElement(cell, retention.NS + "is")
                    ET.SubElement(item, retention.NS + "t").text = ("CORRUPTED" if corrupt_cell and index == row_index == column == 1 else str(value))
            archive.writestr("xl/worksheets/sheet%d.xml" % index, ET.tostring(sheet))
        archive.writestr("xl/workbook.xml", ET.tostring(root))
        archive.writestr("xl/_rels/workbook.xml.rels", ET.tostring(relationships))


class NextRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch_root = patch.object(retention, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.output = self.root / "results/new/run01"
        self.ledger = self.output.parent / "gpu-budget.json"
        self.config = dict(artifact_root=str(self.root / ".cache/optimization-next"), science="unchanged",
                           gpu_uuid="GPU-a100-fixture", v100_uuid="GPU-v100-fixture")
        self.budget = dict(schema_version="ditserve.optimizations.next_budget.v1",
                           config_sha256=digest(self.config), active=None, budget_unlimited=True, limit_gpu_seconds=None,
                           parents={}, sessions=[], used_gpu_seconds=0.)
        self.sources = []
        self.attempts = {}
        for index, phase in enumerate(retention.PHASES):
            attempt = self.output / phase / "attempt00"
            self.attempts[phase] = attempt
            self.write(attempt / "config.json", self.config)
            snapshot = self.output / "sources" / (phase + "-attempt00")
            source = snapshot / "python/fixture.py"
            source.parent.mkdir(parents=True)
            source.write_text("# frozen CPU test fixture\n")
            files = {"python/fixture.py": sha(source)}
            self.write(snapshot / "manifest.json", dict(execution_root=str(snapshot), source_files=files,
                source_sha256=digest(files), config_sha256=digest(self.config)))
            charge = dict(session_id=phase, stage=phase, status="completed", gpu_seconds=10., elapsed_seconds=10., gpu_count=1,
                identity=dict(output=str(attempt), config_sha256=digest(self.config), source_sha256=digest(files),
                              gpu_uuids=[self.config["gpu_uuid"]], auxiliary_encoder=False))
            self.budget["sessions"].append(charge)
            self.budget["used_gpu_seconds"] += 10.
            terminal = dict(phase=phase, output=str(attempt), status="completed", charge=charge,
                source_sha256=digest(files), config_sha256=digest(self.config),
                final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]))
            self.write(attempt / "supervisor.json", terminal)
            self.write(self.output / phase / "terminal.json", terminal)
        self.write(self.ledger, self.budget)
        self.parent = self.root / "old/gpu-budget.json"
        self.write(self.parent, dict(active=None, used_gpu_seconds=500., budget_unlimited=True, limit_gpu_seconds=None))
        self.budget["parents"][str(self.parent)] = bound(self.parent)
        self.write(self.ledger, self.budget)
        self.raw = self.attempts["diagnosis"] / "diagnosis.json"
        self.write(self.raw, dict(status="completed", rows=[dict(events=[dict(kind="not_in_excel")])]))
        self.sources.append(bound(self.raw))
        self.partial = self.attempts["diagnosis"] / "partials/diagnosis/replay-000/partial.json"
        self.write(self.partial, dict(status="completed", requests=[], events=[{"kind": "not_in_excel"}]))
        self.isolation = self.attempts["diagnosis"] / "isolation.jsonl"
        self.isolation.write_text('{"status":"exclusive_observed","foreign":[]}\ninvalid\n{"status":"unknown","foreign":[1]}\n')
        self.compiler = self.attempts["sana_vae"] / "compiler-cache/aa/kernel.so"
        self.compiler.parent.mkdir(parents=True)
        self.compiler.write_bytes(b"tiny private compiler fixture")
        self.write(self.attempts["sana_vae"] / "environment.json",
                   dict(compiler_cache=str(self.attempts["sana_vae"] / "compiler-cache")))
        self.array = self.root / ".cache/optimization-next/run01/sana_vae/attempt00/resume.npy"
        self.array.parent.mkdir(parents=True)
        self.array.write_bytes(b"not parsed as an array; only size and hash")
        self.resume = self.attempts["sana_vae"] / "sana-wan-resume.json"
        self.write(self.resume, dict(status="failed", retention=[dict(bound(self.array), shape=[1, 81, 2, 2, 3])]))
        self.sources += [bound(self.resume), bound(self.parent), bound(self.ledger)]
        self.report = dict(schema_version="ditserve.optimizations.next_report.v1", output=str(self.output),
            budget_path=str(self.ledger), status="recorded", partial=False, errors=[],
            phases=[dict(phase=p, status="completed", authoritative=True, partial=False, decision="recorded",
                         online_qualified=False) for p in retention.PHASES],
            budget=dict(active_unsettled=False, settled_gpu_seconds=80.), source_bindings=self.sources)
        self.finish_report()

    def write(self, path, value):
        atomic_json(path, value)

    def finish_report(self, corrupt_cell=False):
        workbook(self.output / "results.xlsx", self.report, corrupt_cell=corrupt_cell)
        self.report["excel"] = dict(status="verified", path=str(self.output / "results.xlsx"),
            sha256=sha(self.output / "results.xlsx"), formulas=False, raw_events=False)
        self.write(self.output / "report.json", self.report)
        self.write(self.output / "excel-verification.json", self.report["excel"])
        self.write(self.output / "audits/final-integrity.json", dict(status="passed", output=str(self.output),
            all_phases_terminal=True, report_sha256=sha(self.output / "report.json"),
            excel_sha256=sha(self.output / "results.xlsx")))

    def approved(self, selected=None):
        plan = retention.plan(self.output, self.ledger)
        self.assertTrue(plan["ready"], plan.get("errors"))
        return [dict(row, dependencies_closed=True,
                     closure_reason="final review done; no quality, blind-review, or continuation dependency")
                for row in plan["candidates"] if selected is None or row["path"] in selected]

    def archive(self, **kwargs):
        return retention.archive_and_prune(self.output, self.ledger, explicit_dependencies_closed=True, **kwargs)

    def test_plan_whitelist_has_only_private_raw_files_and_no_mutation(self):
        original = self.ledger.read_bytes()
        result = retention.plan(self.output, self.ledger)
        self.assertTrue(result["ready"], result.get("errors"))
        self.assertEqual({r["path"] for r in result["candidates"]},
                         {str(p) for p in (self.raw, self.partial, self.isolation, self.compiler, self.array)})
        isolation = next(r for r in result["candidates"] if r["kind"] == "isolation_log")
        self.assertEqual(isolation["summary"]["malformed_lines"], 1)
        self.assertEqual(isolation["summary"]["unknown_status_lines"], 1)
        self.assertEqual(isolation["summary"]["foreign_observation_lines"], 1)
        reference = next(r for r in result["candidates"] if r["kind"] == "resume_npy")["references"][0]
        self.assertEqual(reference["pointer"], "/retention/0")
        self.assertEqual(reference["source"]["sha256"], sha(self.resume))
        self.assertEqual(self.ledger.read_bytes(), original)
        self.assertFalse((self.output / "next-archive").exists())

    def test_active_session_blocks_before_raw_scan(self):
        self.budget["active"] = dict(session_id="still-running")
        self.write(self.ledger, self.budget)
        with patch.object(retention, "_candidates", side_effect=AssertionError("raw scan forbidden")):
            result = retention.plan(self.output, self.ledger)
        self.assertFalse(result["ready"])
        self.assertIn("active", result["errors"][0])

    def test_missing_terminal_or_failed_integrity_blocks(self):
        path = self.output / "conditional/terminal.json"
        saved = path.read_bytes()
        path.unlink()
        self.assertFalse(retention.plan(self.output, self.ledger)["ready"])
        path.write_bytes(saved)
        audit = self.output / "audits/final-integrity.json"
        value = json.loads(audit.read_text())
        value["status"] = "incomplete"
        self.write(audit, value)
        self.assertFalse(retention.plan(self.output, self.ledger)["ready"])

    def test_scientifically_partial_failed_terminal_is_not_automatically_open_dependency(self):
        self.report["partial"] = True
        self.report["phases"][1].update(status="failed", partial=True, decision="fixed candidate failed quality")
        self.finish_report()
        result = retention.plan(self.output, self.ledger)
        self.assertTrue(result["ready"], result.get("errors"))
        archived = self.archive()
        self.assertEqual(archived["status"], "archived")
        self.assertTrue(self.raw.exists())

    def test_pending_dependencies_and_absent_explicit_closure_reject(self):
        with self.assertRaisesRegex(ValueError, "closure"):
            retention.archive_and_prune(self.output, self.ledger)
        self.report["pending_dependencies"] = ["blind review awaiting user"]
        self.finish_report()
        self.assertFalse(retention.plan(self.output, self.ledger)["ready"])

    def test_cell_tamper_detected_even_if_workbook_hashes_are_rebound(self):
        self.finish_report(corrupt_cell=True)
        result = retention.plan(self.output, self.ledger)
        self.assertFalse(result["ready"])
        self.assertIn("cell readback", result["errors"][0])

    def test_changed_sources_and_old_parent_block_without_mutation(self):
        self.parent.write_text("changed old source")
        result = retention.plan(self.output, self.ledger)
        self.assertFalse(result["ready"])
        self.assertIn("bound source changed", result["errors"][0])
        self.assertTrue(self.array.exists())

    def test_symlink_or_hardlink_private_file_blocks(self):
        self.compiler.unlink()
        self.compiler.symlink_to(self.parent)
        self.assertFalse(retention.plan(self.output, self.ledger)["ready"])
        self.compiler.unlink()
        os.link(str(self.parent), str(self.compiler))
        self.assertFalse(retention.plan(self.output, self.ledger)["ready"])

    def test_array_outside_private_attempt_and_unowned_cache_block(self):
        reference = json.loads(self.resume.read_text())
        reference["retention"][0]["path"] = str(self.parent)
        reference["retention"][0]["path"] += ".npy"
        self.write(self.resume, reference)
        self.sources[1] = bound(self.resume)
        self.finish_report()
        self.assertFalse(retention.plan(self.output, self.ledger)["ready"])

    def test_default_export_verifies_but_never_prunes_or_activates_archive_view(self):
        rows = self.approved()
        result = self.archive()
        self.assertEqual(result["status"], "archived")
        self.assertEqual(result["originals_deleted"], 0)
        self.assertTrue(all(Path(r["path"]).exists() for r in rows))
        self.assertFalse((self.output / "next-archive/finalization.json").exists())
        self.assertEqual(sha(self.output / "report.json"), sha(self.output / "next-archive/frozen-report.json"))

    def test_prune_requires_exact_file_hash_and_reason_and_does_not_delete_snapshot(self):
        rows = self.approved()
        for change in (dict(sha256="0"*64), dict(dependencies_closed=False), dict(closure_reason="")):
            bad = copy.deepcopy(rows)
            bad[0].update(change)
            with self.assertRaises(ValueError):
                self.archive(whitelist=bad, prune=True)
        bad = copy.deepcopy(rows)
        source = self.output / "sources/diagnosis-attempt00/python/fixture.py"
        bad[0].update(bound(source))
        with self.assertRaisesRegex(ValueError, "noncandidate"):
            self.archive(whitelist=bad, prune=True)
        self.assertTrue(all(Path(r["path"]).exists() for r in rows))

    def test_prune_exact_subset_preserves_reports_snapshots_history_and_loads_archive(self):
        parent_bytes = self.parent.read_bytes()
        ledger_bytes = self.ledger.read_bytes()
        rows = self.approved(selected={str(self.raw), str(self.array), str(self.isolation)})
        self.archive()  # A previously verified default export can be finalized.
        result = self.archive(whitelist=rows, prune=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["originals_deleted"], 3)
        self.assertFalse(self.raw.exists())
        self.assertFalse(self.array.exists())
        self.assertTrue(self.compiler.exists())
        self.assertTrue(self.partial.exists())
        self.assertTrue(self.resume.exists())
        self.assertEqual(self.parent.read_bytes(), parent_bytes)
        self.assertEqual(self.ledger.read_bytes(), ledger_bytes)
        loaded = retention.load_archive_report(self.output, self.ledger)
        self.assertTrue(loaded["archive_only"])
        self.assertFalse(loaded["raw_re_evaluable"])
        self.assertFalse(loaded["new_qualification_granted"])
        self.assertEqual(loaded["phases"], self.report["phases"])

    def test_interrupted_deletion_retains_receipt_and_refuses_raw_fallback(self):
        rows = self.approved()
        original = Path.unlink
        called = []

        def failing(path, *args, **kwargs):
            called.append(str(path))
            if len(called) == 2:
                raise OSError("fixture unlink failure")
            return original(path, *args, **kwargs)

        with patch.object(Path, "unlink", failing):
            with self.assertRaisesRegex(OSError, "fixture unlink"):
                self.archive(whitelist=rows, prune=True)
        final = json.loads((self.output / "next-archive/finalization.json").read_text())
        self.assertEqual(final["status"], "failed")
        self.assertEqual(len(final["deleted"]), 1)
        self.assertIn("pending_delete", final)
        with self.assertRaisesRegex(ValueError, "incomplete, failed"):
            retention.load_archive_report(self.output, self.ledger)

    def test_archive_detects_changed_retained_source_or_reappeared_deleted_file(self):
        rows = self.approved(selected={str(self.array)})
        self.archive(whitelist=rows, prune=True)
        self.array.write_bytes(b"reappeared")
        with self.assertRaisesRegex(ValueError, "reappeared"):
            retention.load_archive_report(self.output, self.ledger)
        self.array.unlink()
        self.ledger.write_text("changed")
        with self.assertRaisesRegex(ValueError, "bound source changed"):
            retention.load_archive_report(self.output, self.ledger)

    def test_unknown_attempt_and_unsettled_charge_block(self):
        self.write(self.output / "diagnosis/attempt01/config.json", self.config)
        result = retention.plan(self.output, self.ledger)
        self.assertFalse(result["ready"])
        self.assertTrue(self.raw.exists())

    def test_private_compiler_cache_needs_environment_binding(self):
        self.write(self.attempts["sana_vae"] / "environment.json", dict(compiler_cache="/shared/compiler-cache"))
        result = retention.plan(self.output, self.ledger)
        self.assertFalse(result["ready"])
        self.assertIn("ownership", result["errors"][0])

    def test_source_snapshot_and_fixed_configuration_remain_bound(self):
        path = self.attempts["sana_graph"] / "config.json"
        self.write(path, dict(self.config, science="changed"))
        result = retention.plan(self.output, self.ledger)
        self.assertFalse(result["ready"])
        self.assertIn("manifest differs", result["errors"][0])

    def test_shared_supervisor_lock_prevents_cleanup_start(self):
        import fcntl
        with self.ledger.with_suffix(".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.archive()
        self.assertTrue(self.array.exists())
        self.assertFalse((self.output / "next-archive").exists())

    def test_real_reporter_uses_archive_view_and_refuses_excel_overwrite(self):
        from ditserve.optimizations import next_report
        rows = self.approved(selected={str(self.raw), str(self.array)})
        self.archive(whitelist=rows, prune=True)
        before = (self.output / "results.xlsx").read_bytes()
        view = next_report.collect(self.output, self.ledger)
        self.assertTrue(view["archive_only"])
        self.assertFalse(view["qualification_recomputed"])
        with self.assertRaisesRegex(ValueError, "archive-only report is immutable"):
            next_report.write_report(self.output, self.ledger)
        self.assertEqual((self.output / "results.xlsx").read_bytes(), before)

    def test_conditional_two_device_charge_is_allowed_but_other_stage_is_not(self):
        def update(phase):
            path = self.attempts[phase] / "supervisor.json"
            value = json.loads(path.read_text())
            value["charge"].update(gpu_count=2, gpu_seconds=20.)
            value["charge"]["identity"].update(auxiliary_encoder=True,
                gpu_uuids=[self.config["gpu_uuid"], self.config["v100_uuid"]])
            self.budget["sessions"][list(retention.PHASES).index(phase)] = value["charge"]
            self.budget["used_gpu_seconds"] += 10.
            self.write(path, value)
            self.write(self.output / phase / "terminal.json", value)
            self.write(self.ledger, self.budget)
            self.sources[-1] = bound(self.ledger)
            self.finish_report()
        update("conditional")
        result = retention.plan(self.output, self.ledger)
        self.assertTrue(result["ready"], result.get("errors"))
        update("diagnosis")
        result = retention.plan(self.output, self.ledger)
        self.assertFalse(result["ready"])
        self.assertIn("count/devices", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
