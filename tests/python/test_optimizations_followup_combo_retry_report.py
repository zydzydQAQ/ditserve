"""CPU-only checks for independent retry provenance and honest terminal state."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.optimizations import followup_combo_retry_report as report


class ComboRetryReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.work = self.base / "work"
        self.work.mkdir()
        self.output = self.base / "output"
        self.ledger = self.base / "gpu-budget.json"
        self.old = self.base / "old"
        self.old.mkdir()
        self.failure = self.old / "terminal.json"
        self.save(self.failure, dict(phase="sana_combo", status="failed", returncode=-15,
            reason="old disk failure", cleanup_confirmed=True,
            rows=[dict(historical_measurement="must not pool")],
            charge=dict(session_id="old", stage="sana_combo", gpu_count=1, elapsed_seconds=12., gpu_seconds=12., status="failed")))
        self.manifest = self.old / "manifest.json"
        self.save(self.manifest, dict(source_sha256="b" * 64, source_files={"source.py": "c" * 64}))
        self.controller = self.base / "controller.py"
        self.controller.write_text("# CPU fixture only\n", encoding="utf-8")
        self.parent = self.old / "gpu-budget.json"
        self.save(self.parent, dict(active=None, used_gpu_seconds=12., sessions=[]))
        self.save(self.ledger, dict(active=None, used_gpu_seconds=0., sessions=[], budget_unlimited=True))
        self.save(self.work / "config.json", dict(seeds=[101], artifact_root=str(self.work),
            formula="=literal_only", identifier=100000000000000000001))
        self.study = dict(ledger=str(self.ledger), phases=["sana_combo"], retry_of=self.binding(self.failure),
            parent_ledgers=[self.binding(self.parent)], storage_only_change=dict(field="artifact_root", scientific_recipe_unchanged=True))
        self.save(self.work / "study.json", self.study)
        self.queue = dict(status="prepared", phases=["sana_combo"], ledger=str(self.ledger),
            original_failure=self.binding(self.failure), source_manifest=self.binding(self.manifest), controller=self.binding(self.controller))
        self.save(self.work / "queue.json", self.queue)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def binding(self, path):
        return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def install(self, worker_status="running", supervisor_status="failed", settled=True, terminal_only=False):
        directory = self.work / "sana_combo" / "attempt000"
        self.save(directory / "phase-result.json", dict(schema_version=1, phase="sana_combo", status=worker_status,
            verdict="值得继续", quality=dict(passed=True), local=[dict(service_ms=123.45)],
            online=dict(status="pending", expected_replays=8, expected_requests=192, replays=[]), failures=[]))
        charge = dict(session_id="new", stage="sana_combo", gpu_count=1, elapsed_seconds=20., gpu_seconds=20.,
            status=supervisor_status)
        control = dict(status=supervisor_status, returncode=-15 if supervisor_status == "failed" else 0,
            charge=charge, cleanup_confirmed=True, final_occupancy=dict(status="exclusive_observed", owned=[], foreign=[]))
        if not terminal_only:
            self.save(directory / "supervisor.json", control)
        self.save(self.work / "sana_combo" / "terminal.json", control)
        self.save(self.ledger, dict(active=None if settled else dict(session_id="new"),
            sessions=[charge] if settled else [], used_gpu_seconds=20. if settled else 0., budget_unlimited=True))
        self.queue["status"] = "completed" if settled else "running"
        self.save(self.work / "queue.json", self.queue)
        return directory, control

    def test_settled_failure_overrides_stale_running_and_keeps_worker_state(self):
        directory, _ = self.install()
        value = report.collect(self.work, ledger=self.ledger)
        phase = value["phases"][0]
        self.assertEqual(phase["status"], "incomplete")
        self.assertEqual(phase["recorded_verdict"], "未验证")
        self.assertEqual(phase["worker_metadata"]["worker_status"], "running")
        self.assertEqual(phase["worker_metadata"]["worker_verdict"], "值得继续")
        self.assertTrue(phase["accounting_settled"])
        self.assertFalse(phase["dependencies_closed"])
        self.assertEqual(value["status"], "incomplete")
        self.assertEqual(json.loads((directory / "phase-result.json").read_text())["status"], "running")

    def test_terminal_can_supply_settlement_when_attempt_supervisor_missing(self):
        self.install(terminal_only=True)
        value = report.collect(self.work, ledger=self.ledger)
        self.assertEqual(value["phases"][0]["status"], "incomplete")
        self.assertTrue(value["phases"][0]["accounting_settled"])
        self.assertIn(str(self.work / "sana_combo/terminal.json"), {row["path"] for row in value["sources"]})

    def test_unsettled_completed_worker_does_not_gain_success(self):
        self.install(worker_status="completed", supervisor_status="completed", settled=False)
        value = report.collect(self.work, ledger=self.ledger)
        phase = value["phases"][0]
        self.assertEqual(phase["status"], "unsettled")
        self.assertEqual(phase["recorded_verdict"], "未验证")
        self.assertFalse(phase["accounting_settled"])
        self.assertFalse(value["baseline_promotion"])
        self.assertFalse(value["dependencies_closed"])

    def test_settled_completed_gate_failure_is_a_completed_experiment(self):
        directory, _ = self.install(worker_status="completed", supervisor_status="completed")
        data = json.loads((directory / "phase-result.json").read_text())
        data["verdict"] = "当前配置无收益"
        self.save(directory / "phase-result.json", data)
        value = report.collect(self.work, ledger=self.ledger)
        self.assertEqual(value["status"], "completed")
        self.assertEqual(value["phases"][0]["recorded_verdict"], "当前配置无收益")
        self.assertFalse(value["baseline_promotion"])

    def test_normal_quality_gate_stop_keeps_its_settled_negative_conclusion(self):
        directory, control = self.install(worker_status="stopped", supervisor_status="stopped")
        data = json.loads((directory / "phase-result.json").read_text())
        data.update(verdict="当前配置无收益", reason="combined Sana quality gate failed")
        data["quality"]["passed"] = False
        self.save(directory / "phase-result.json", data)
        control["charge"]["status"] = "completed"
        for path in (directory / "supervisor.json", self.work / "sana_combo/terminal.json"):
            self.save(path, control)
        self.save(self.ledger, dict(active=None, used_gpu_seconds=20., sessions=[control["charge"]]))
        value = report.collect(self.work, ledger=self.ledger)
        phase = value["phases"][0]
        self.assertEqual(value["status"], "completed")
        self.assertEqual(phase["status"], "stopped")
        self.assertEqual(phase["recorded_verdict"], "当前配置无收益")
        self.assertTrue(phase["accounting_settled"])
        self.assertFalse(phase["dependencies_closed"])
        self.assertFalse(value["dependencies_closed"])

    def test_positive_worker_verdict_without_completed_online_stays_unverified(self):
        self.install(worker_status="completed", supervisor_status="completed")
        value = report.collect(self.work, ledger=self.ledger)
        self.assertEqual(value["phases"][0]["recorded_verdict"], "未验证")
        self.assertEqual(value["phases"][0]["worker_metadata"]["worker_verdict"], "值得继续")
        self.assertFalse(value["phases"][0]["dependencies_closed"])

    def test_inconsistent_terminal_is_flagged_without_promoting_worker(self):
        _, control = self.install(worker_status="completed", supervisor_status="completed")
        control["returncode"] = -15
        self.save(self.work / "sana_combo/terminal.json", control)
        value = report.collect(self.work, ledger=self.ledger)
        self.assertIn("outcomes disagree", " ".join(value["phases"][0]["errors"]))
        self.assertEqual(value["phases"][0]["recorded_verdict"], "未验证")
        self.assertFalse(value["phases"][0]["accounting_settled"])

    def test_write_two_files_readback_literals_and_provenance_without_old_samples(self):
        self.install()
        raw = self.old / "old-output.npy"
        raw.write_bytes(b"OLD_ARRAY_MUST_NOT_BE_READ")
        before = {str(path): path.read_bytes() for path in self.base.rglob("*") if path.is_file()}
        with patch.object(Path, "open", autospec=True, side_effect=Path.open) as opened:
            result = report.write(self.work, self.output, ledger=self.ledger)
            self.assertNotIn(raw, [args[0] for args, _ in opened.call_args_list])
        self.assertEqual({path.name for path in self.output.iterdir()}, {"results.xlsx", "REPORT.md"})
        self.assertEqual(result["excel"]["status"], "verified")
        self.assertEqual(result["historical_samples_pooled"], 0)
        self.assertEqual(result["requests"], 0)
        self.assertEqual(result["originals_deleted"], 0)
        self.assertFalse(result["errors"])
        for path, contents in before.items():
            self.assertEqual(Path(path).read_bytes(), contents)
        from openpyxl import load_workbook
        book = load_workbook(self.output / "results.xlsx", read_only=True, data_only=False)
        try:
            cells = [cell for row in book["Config"] for cell in row]
            literal = next(cell for cell in cells if cell.value == "=literal_only")
            self.assertEqual(literal.data_type, "s")
            self.assertIn("100000000000000000001", [cell.value for cell in cells])
            config_values = [cell.value for cell in cells]
            self.assertIn("old disk failure", config_values)
            self.assertIn("/storage_only_change/field", config_values)
            sources = list(book["Sources"].values)
            self.assertIn((str(self.controller), self.binding(self.controller)["sha256"], str(self.controller.stat().st_size)), sources)
            summary = list(book["Summary"].values)
            self.assertEqual(summary[1][2:4], ("incomplete", "未验证"))
            self.assertEqual(len(list(book["Requests"].values)), 1)
        finally:
            book.close()
        text = (self.output / "REPORT.md").read_text()
        self.assertIn("计划的 8 次", text)
        self.assertIn("不与本次样本合并", text)
        report.write(self.work, self.output, ledger=self.ledger)

    def test_changed_parent_binding_is_reported_and_blocks_completed_status(self):
        self.install(worker_status="completed", supervisor_status="completed")
        self.parent.write_text(self.parent.read_text() + "\n")
        value = report.collect(self.work, ledger=self.ledger)
        self.assertTrue(any("hash or size changed" in error for error in value["errors"]))
        self.assertEqual(value["status"], "incomplete")

    def test_waiter_zero_session_ledger_is_not_gpu_execution(self):
        self.queue["status"] = "waiting"
        self.save(self.work / "queue.json", self.queue)
        value = report.collect(self.work, ledger=self.ledger)
        self.assertEqual(value["status"], "waiting")
        self.assertEqual(value["phases"], [])
        self.assertEqual(value["used_gpu_seconds"], 0.)
        self.assertFalse(value["dependencies_closed"])
        self.assertFalse(value["errors"])

    def test_rejects_unrelated_existing_report_and_extra_files(self):
        self.output.mkdir()
        marker = self.output / "REPORT.md"
        marker.write_text("old report")
        with self.assertRaisesRegex(ValueError, "another experiment"):
            report.write(self.work, self.output, ledger=self.ledger)
        self.assertEqual(marker.read_text(), "old report")
        marker.unlink()
        (self.output / "keep.txt").write_text("protected")
        with self.assertRaisesRegex(ValueError, "unrelated files"):
            report.write(self.work, self.output, ledger=self.ledger)

    def test_rejects_same_report_marker_from_another_work_root(self):
        report.write(self.work, self.output, ledger=self.ledger)
        other = self.base / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "another experiment"):
            report.write(other, self.output, ledger=self.ledger)

    def test_rejects_input_output_symlinks_and_overlap(self):
        linked = self.base / "linked"
        linked.symlink_to(self.work)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            report.write(linked, self.output, ledger=self.ledger)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            report.write(self.work, self.work / "exports", ledger=self.ledger)
        self.output.mkdir()
        target = self.base / "untouched.xlsx"
        target.write_text("protected")
        (self.output / "results.xlsx").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            report.write(self.work, self.output, ledger=self.ledger)
        self.assertEqual(target.read_text(), "protected")

    def test_bound_history_symlink_is_not_followed(self):
        linked = self.base / "failure-link.json"
        linked.symlink_to(self.failure)
        self.queue["original_failure"]["path"] = str(linked)
        self.save(self.work / "queue.json", self.queue)
        value = report.collect(self.work, ledger=self.ledger)
        self.assertTrue(any("symlinks" in error for error in value["errors"]))


if __name__ == "__main__":
    unittest.main()
