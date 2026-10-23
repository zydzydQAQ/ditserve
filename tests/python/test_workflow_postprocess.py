"""Host postprocessing publishes verified artifacts and never launches profiling."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC=importlib.util.spec_from_file_location("workflow_postprocess",Path(__file__).resolve().parents[2]/"docker/postprocess-workflow-study.py")
helper=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def fake_export(command,stdout_log,stderr_log,*,timeout,csv_target=None):
    command=list(map(str,command))
    if "export" in command:
        path=Path(command[command.index("--output")+1])
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE fixture(value TEXT)")
    elif "--import" in command:
        stream=io.StringIO()
        writer=csv.writer(stream)
        writer.writerow(["ID","Process ID","Kernel Name","Device","Context","Stream","gpu__time_duration.sum"])
        writer.writerow(["","","","","","","usecond"])
        label='ditserve.workflow/dit_step|{"purpose":"video_measurement","repeat":0,"step":1,"variant":"T0"}'
        writer.writerow([0,42,label if "--print-nvtx-rename" in command else "gemm",1,1,7,10])
        csv_target.write_text(stream.getvalue())
    else:
        raise AssertionError("unexpected non-import/export command")
    return command


class WorkflowPostprocessTests(unittest.TestCase):
    def args(self,reparse=False):
        return SimpleNamespace(nsys=Path("/fake/nsys"),ncu=Path("/fake/ncu"),timeout=10,reparse=reparse)

    def test_discovery_excludes_probes_nested_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for relative in ("traces/T0.nsys-rep","ncu/T1.ncu-rep","probes/cuda.nsys-rep","traces/cuda-probe.nsys-rep","traces/nested/T2.nsys-rep"):
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"capture")
            (root/"traces/link.nsys-rep").symlink_to(root/"probes/cuda.nsys-rep")
            captures,excluded=helper.discover(root)
        self.assertEqual({(kind,path.name) for kind,path in captures},{("nsys","T0.nsys-rep"),("ncu","T1.ncu-rep")})
        self.assertEqual(len(excluded),2)

    def test_atomic_sqlite_export_and_missing_metrics_remain_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=fake_export):
            capture=Path(temporary)/"T0.nsys-rep";capture.write_bytes(b"immutable trace")
            result=helper.process_capture("nsys",capture,self.args())
            parsed=json.loads(capture.with_suffix(".parsed.json").read_text())
            self.assertEqual(result["status"],"parsed")
            self.assertTrue(capture.with_suffix(".sqlite").is_file())
            self.assertFalse(capture.with_suffix(".part.sqlite").exists())
            self.assertFalse(parsed["availability"]["gpu_metrics"])
            self.assertEqual(parsed["status"],"partial")
            self.assertTrue(any("not zero" in text for text in parsed["warnings"]))
            self.assertEqual(parsed["postprocess"]["capture"]["sha256"],result["source"]["sha256"])

    def test_corrupt_sqlite_is_not_published_or_deleted(self):
        def corrupt(command,*args,**kwargs):
            command=list(map(str,command))
            Path(command[command.index("--output")+1]).write_bytes(b"not sqlite")
            return command
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=corrupt):
            capture=Path(temporary)/"T0.nsys-rep";capture.write_bytes(b"trace")
            result=helper.process_capture("nsys",capture,self.args())
            self.assertEqual(result["status"],"failed")
            self.assertFalse(capture.with_suffix(".sqlite").exists())
            self.assertTrue(capture.with_suffix(".part.sqlite").exists())
            self.assertEqual(capture.read_bytes(),b"trace")

    def test_skip_then_reparse_archives_json_outside_discovery(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=fake_export) as export:
            capture=Path(temporary)/"T0.nsys-rep";capture.write_bytes(b"trace")
            first=helper.process_capture("nsys",capture,self.args())
            skipped=helper.process_capture("nsys",capture,self.args(),first)
            self.assertEqual(skipped["status"],"already_parsed")
            self.assertEqual(export.call_count,1)
            reparsed=helper.process_capture("nsys",capture,self.args(True),skipped)
            self.assertEqual(reparsed["status"],"parsed")
            self.assertEqual(export.call_count,1)
            self.assertEqual(len(list(Path(temporary).glob("*.json"))),1)
            self.assertTrue(reparsed["archived"][0].endswith(".bak"))

    def test_changed_capture_cannot_reset_immutable_binding_on_retry(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=fake_export):
            capture=Path(temporary)/"T0.nsys-rep";capture.write_bytes(b"original")
            original=helper.process_capture("nsys",capture,self.args())
            capture.write_bytes(b"changed")
            failed=helper.process_capture("nsys",capture,self.args(True),original)
            retried=helper.process_capture("nsys",capture,self.args(True),failed)
            self.assertEqual(failed["status"],"failed")
            self.assertEqual(retried["status"],"failed")
            self.assertEqual(retried["source"]["sha256"],original["source"]["sha256"])

    def test_ncu_import_only_and_unavailable_metrics_stay_null(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=fake_export):
            capture=Path(temporary)/"T0.ncu-rep";capture.write_bytes(b"kernel capture")
            result=helper.process_capture("ncu",capture,self.args())
            parsed=json.loads(capture.with_suffix(".parsed.json").read_text())
            self.assertEqual(result["status"],"parsed")
            self.assertTrue(all("--import" in command for command in result["commands"]))
            self.assertIsNone(parsed["summaries"][0]["metrics"]["achieved_occupancy_percent"]["duration_weighted_mean"])
            self.assertEqual(parsed["coverage"]["profiled_kernel_count"],1)

    def test_preview_never_calls_finalization_and_no_captures_is_not_zero_measurement(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as run,patch("builtins.print"):
            self.assertEqual(helper.main(["--root",temporary,"--preview"]),0)
            ledger=json.loads((Path(temporary)/"postprocess.json").read_text())
            self.assertEqual(ledger["status"],"no_captures")
            self.assertEqual(ledger["discovery"]["absent_capture_types"],["nsys","ncu"])
            command=run.call_args.args[0]
            self.assertIn("--preview",command)
            self.assertIn("ditserve.evaluation.workflow_report",command)
            self.assertNotIn("--destination",command)

    def test_missing_capture_keeps_immutable_ledger_binding(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=fake_export),patch("builtins.print"):
            root=Path(temporary)
            (root/"traces").mkdir()
            capture=root/"traces/T0.nsys-rep"
            capture.write_bytes(b"original")
            self.assertEqual(helper.main(["--root",temporary]),0)
            expected=json.loads((root/"postprocess.json").read_text())["captures"][0]["source"]["sha256"]
            capture.rename(capture.with_suffix(".saved"))
            self.assertEqual(helper.main(["--root",temporary]),0)
            missing=json.loads((root/"postprocess.json").read_text())["captures"][0]
            self.assertEqual(missing["status"],"missing")
            self.assertEqual(missing["source"]["sha256"],expected)
            capture.write_bytes(b"replacement")
            self.assertEqual(helper.main(["--root",temporary,"--reparse"]),1)
            failed=json.loads((root/"postprocess.json").read_text())["captures"][0]
            self.assertEqual(failed["source"]["sha256"],expected)

    def test_kind_reparse_preserves_other_capture_and_immutable_binding(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=fake_export),patch("builtins.print"):
            root=Path(temporary)
            for name in ("traces/T0.nsys-rep","ncu/T1.ncu-rep"):
                path=root/name;path.parent.mkdir(exist_ok=True);path.write_bytes(name.encode())
            self.assertEqual(helper.main(["--root",temporary]),0)
            before=json.loads((root/"postprocess.json").read_text())
            preserved=next(row for row in before["captures"] if row["kind"]=="nsys")
            parsed=root/"traces/T0.parsed.json";parsed_before=parsed.read_bytes()
            self.assertEqual(helper.main(["--root",temporary,"--kind","ncu","--reparse"]),0)
            after=json.loads((root/"postprocess.json").read_text())
            self.assertEqual(len(after["captures"]),2)
            self.assertEqual(next(row for row in after["captures"] if row["kind"]=="nsys"),preserved)
            self.assertEqual(parsed.read_bytes(),parsed_before)
            self.assertFalse(list((root/"traces").glob("*.bak")))
            self.assertEqual(len(list((root/"ncu").glob("*.parsed.json.previous.*.bak"))),1)
            self.assertEqual(after["discovery"]["preserved_untouched_count"],1)
            self.assertEqual(after["selected_kind"],"ncu")

    def test_kind_filter_retains_missing_unselected_evidence(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(helper,"run_logged",side_effect=fake_export),patch("builtins.print"):
            root=Path(temporary);(root/"traces").mkdir()
            capture=root/"traces/T0.nsys-rep";capture.write_bytes(b"original")
            self.assertEqual(helper.main(["--root",temporary]),0)
            expected=json.loads((root/"postprocess.json").read_text())["captures"][0]["source"]
            capture.rename(capture.with_suffix(".saved"))
            self.assertEqual(helper.main(["--root",temporary,"--kind","ncu","--reparse"]),0)
            ledger=json.loads((root/"postprocess.json").read_text())
            self.assertEqual(ledger["captures"][0]["status"],"missing")
            self.assertEqual(ledger["captures"][0]["source"],expected)


if __name__=="__main__":
    unittest.main()
