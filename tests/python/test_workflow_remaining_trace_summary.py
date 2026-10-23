"""Bind trace claims to this job's capture and exact CUDA-correlated pair."""

import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from ditserve.evaluation import workflow_remaining_report as report
from ditserve.evaluation import workflow_trace
from ditserve.evaluation import workflow_overlap_trace as capture
from test_workflow_remaining_vae import config


class TraceSummaryTests(unittest.TestCase):
    def prepare(self, *, zero_overlap=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root/"traces").mkdir()
        value = config()
        source = root/"input.json"
        source.write_text(json.dumps(value))
        self.study = dict(root=str(root))
        self.job = dict(id="remaining-vae-trace-a17-b33", input=str(source),
                        configuration_sha256=report.digest(value))
        self.capture = root/"traces"/(self.job["id"]+".nsys-rep")
        self.capture.write_bytes(b"immutable synthetic Nsight capture")
        self.sqlite = self.capture.with_suffix(".sqlite")
        connection = sqlite3.connect(self.sqlite)
        connection.executescript('''
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER,endGlobalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
        ''')
        pid, tid = 17 << 24, (17 << 24)+42
        connection.execute("INSERT INTO StringIds VALUES (1,'model_kernel')")
        label = capture.capture_range(value)
        metadata = workflow_trace.parse_label(label)["metadata"]
        connection.execute("INSERT INTO NVTX_EVENTS VALUES (1,1000,?,?,NULL)", (label,tid))
        leaves = [("vae_decode", dict(metadata,role="B"),5,20000 if zero_overlap else 10020,22000 if zero_overlap else 12010)]
        leaves += [("dit_step",dict(metadata,role="A",step=i),10+i*10,10000+i*100,10050+i*100) for i in range(20)]
        for correlation,(leaf,meta,host,start,end) in enumerate(leaves,1):
            text = "ditserve.workflow/P4/"+leaf+"|"+json.dumps(meta,sort_keys=True,separators=(",",":"))
            connection.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)", (host,host+3,text,tid))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)", (host+1,host+2,tid,correlation))
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,1)",
                               (start,end,0,pid,correlation,1 if meta["role"]=="A" else 2))
        connection.commit()
        connection.close()
        self.parsed_path = self.capture.with_suffix(".parsed.json")
        self.parsed = workflow_trace.analyze_trace(self.sqlite)
        self.parsed_path.write_text(json.dumps(self.parsed))
        self.manifest_path = root/"remaining-traces.json"
        self.entry = dict(job=self.job["id"],status="parsed",capture=str(self.capture),
                          capture_binding=report.api.binding(self.capture),sqlite=report.api.binding(self.sqlite),
                          parsed=report.api.binding(self.parsed_path))
        self.publish()

    def publish(self, *, rebind_parsed=False):
        if rebind_parsed:
            self.parsed_path.write_text(json.dumps(self.parsed))
            self.entry["parsed"] = report.api.binding(self.parsed_path)
        self.manifest_path.write_text(json.dumps(dict(schema="ditserve.remaining.trace-postprocess.v1",entries=[self.entry])))

    def summarize(self):
        return report._trace_summary(self.study,self.job)

    def test_bound_parser_role_intersection_proves_only_positive_overlap(self):
        self.prepare()
        result = self.summarize()
        self.assertTrue(result["available"], result)
        self.assertTrue(result["cuda_overlap_proven"], result)
        self.assertEqual(result["intersections"][0]["overlap_ms"],.00098)
        self.assertNotIn("performance_evidence",result)

    def test_valid_zero_overlap_is_available_without_positive_claim(self):
        self.prepare(zero_overlap=True)
        result = self.summarize()
        self.assertTrue(result["available"],result)
        self.assertFalse(result["cuda_overlap_proven"])
        self.assertEqual(result["intersections"][0]["overlap_ms"],0)

    def test_pending_missing_or_failed_postprocess_never_qualifies(self):
        self.prepare()
        for status in ("unavailable","parse_failed","running"):
            self.entry["status"] = status
            self.publish()
            self.assertFalse(self.summarize()["available"])
            self.assertFalse(self.summarize()["cuda_overlap_proven"])
        self.manifest_path.unlink()
        self.assertFalse(self.summarize()["available"])

    def test_capture_sqlite_and_parsed_bytes_remain_bound(self):
        self.prepare()
        for path in (self.capture,self.sqlite,self.parsed_path):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original+b"changed")
                result = self.summarize()
                self.assertFalse(result["available"],result)
                self.assertFalse(result["cuda_overlap_proven"])
                path.write_bytes(original)
                for key, bound_path in (("capture_binding",self.capture),("sqlite",self.sqlite),("parsed",self.parsed_path)):
                    self.entry[key]=report.api.binding(bound_path)
                self.publish()

    def test_rebound_unrelated_source_capture_or_arm_does_not_qualify(self):
        self.prepare()
        changes = [lambda x:x.update(source="/other/job.sqlite"),
                   lambda x:x["arms"][0].update(name="P4/trace/streams/frames.33/peer_frames.33/r0"),
                   lambda x:x["phases"][0]["metadata"].update(peer_frames=17),
                   lambda x:x["phases"][0].update(label="different NVTX capture"),
                   lambda x:x["arms"].append(copy.deepcopy(x["arms"][0])),
                   lambda x:x["phases"].pop(),
                   lambda x:x["phases"][1]["metadata"].update(purpose="trace_warmup"),
                   lambda x:x["phases"][1].update(cross_thread=True),
                   lambda x:x["phases"][1].update(end_ns=2000)]
        original=copy.deepcopy(self.parsed)
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                self.parsed=copy.deepcopy(original)
                change(self.parsed)
                self.publish(rebind_parsed=True)
                self.assertFalse(self.summarize()["available"])
        self.parsed=original
        self.entry["capture"]="/other/job.nsys-rep"
        self.publish(rebind_parsed=True)
        self.assertFalse(self.summarize()["available"])

    def test_unavailable_wrong_basis_and_impossible_intersections_fail_closed(self):
        self.prepare()
        changes = [lambda x:x["availability"].update(cuda_kernels=False),
                   lambda x:x["availability"].update(workflow_nvtx=False),
                   lambda x:x["counts"].update(cuda_operations_correlated_to_workflow=0),
                   lambda x:x["arms"][0]["devices"]["0"].update(dit_vae_kernel_overlap_available=False),
                   lambda x:x["arms"][0]["devices"]["0"].update(dit_vae_kernel_overlap_basis="host envelope"),
                   lambda x:x["arms"][0]["devices"]["0"].update(dit_vae_kernel_overlap_ms=1),
                   lambda x:x["arms"][0]["devices"]["0"].update(dit_vae_kernel_overlap_ms=-1),
                   lambda x:x["arms"][0]["devices"]["0"].update(dit_vae_kernel_overlap_ms=float("nan"))]
        original=copy.deepcopy(self.parsed)
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                self.parsed=copy.deepcopy(original)
                change(self.parsed)
                self.publish(rebind_parsed=True)
                result=self.summarize()
                self.assertFalse(result["available"],result)
                self.assertFalse(result["cuda_overlap_proven"])


if __name__ == "__main__":
    unittest.main()
