"""NCU kernel diagnostics must preserve units, association and sample coverage."""

import csv
import io
import json
import re
import unittest

from ditserve.evaluation.workflow_ncu import normalize_value, nvtx_filter, parse_csv, profile_options


class WorkflowNcuTests(unittest.TestCase):
    def make_csv(self, renamed=False):
        stream=io.StringIO();writer=csv.writer(stream)
        writer.writerow(["ID","Process ID","Kernel Name","Device","Context","Stream","gpu__time_duration.sum",
                         "sm__warps_active.avg.pct_of_peak_sustained_active","sm__cycles_active.avg.pct_of_peak_sustained_elapsed","dram__bytes_read.sum"])
        writer.writerow(["","","","","","","usecond","%","%","Kbyte"])
        label='ditserve.workflow/dit_step|{"purpose":"video_measurement","repeat":0,"step":1,"variant":"T0"}'
        writer.writerow(["0","42",label if renamed else "gemm",1,1,7,10,20,80,100])
        writer.writerow(["1","42",label if renamed else "attention",1,1,7,30,60,90,"n/a"])
        return stream.getvalue()

    def test_time_weighted_sampled_occupancy_and_missing_metrics(self):
        result=parse_csv(self.make_csv(),self.make_csv(True))
        row=result["summaries"][0]
        self.assertEqual(row["stage"],"dit_step")
        self.assertEqual(row["variant"],"T0")
        self.assertEqual(row["summed_profiled_kernel_duration_ns"],40000)
        self.assertEqual(row["metrics"]["achieved_occupancy_percent"]["duration_weighted_mean"],50)
        self.assertEqual(row["metrics"]["dram_read_bytes"]["sum"],100000)
        self.assertEqual(row["metrics"]["dram_read_bytes"]["effective_gb_s"],10)
        self.assertIsNone(row["metrics"]["dram_throughput_percent"]["duration_weighted_mean"])
        self.assertIsNone(result["coverage"]["full_stage_time_fraction"])
        self.assertEqual(result["kernels"][0]["kernel_name"],"gemm")
        json.dumps(result,allow_nan=False)

    def test_declared_stage_is_labelled_and_does_not_claim_nvtx_proof(self):
        result=parse_csv(self.make_csv(),declared_stage="dit_step",declared_variant="T0",expected_kernels=20)
        self.assertEqual(result["kernels"][0]["association"],"declared capture filter")
        self.assertEqual(result["coverage"]["selected_kernel_count_fraction"],.1)
        with self.assertRaises(ValueError):
            parse_csv(self.make_csv(),expected_kernels=1)

    def test_units_and_unknown_values(self):
        self.assertEqual(normalize_value("1,234.5","usecond","time"),1234500)
        self.assertEqual(normalize_value("1.25","msecond","time"),1250000)
        self.assertEqual(normalize_value("87.25","%","percent"),87.25)
        self.assertIsNone(normalize_value("n/a","%","percent"))
        with self.assertRaises(ValueError):
            normalize_value("100","cycles","time")

    def test_actual_phase_labels_match_filters_and_skip_warmups_and_step_ten(self):
        pattern=nvtx_filter("dit_step",variant="T0",purpose="video_measurement",step=1)
        regex=pattern[len("regex:"):-1]
        for step,matched in ((1,True),(10,False),(11,False)):
            label="ditserve.workflow/dit_step|"+json.dumps(dict(variant="T0",purpose="video_measurement",repeat=0,step=step),sort_keys=True,separators=(",",":"))
            self.assertEqual(bool(re.fullmatch(regex,label)),matched)
        label='ditserve.workflow/dit_step|{"purpose":"warmup","repeat":0,"step":1,"variant":"T0"}'
        self.assertIsNone(re.fullmatch(regex,label))
        options=profile_options("text_encoder",variant="T2",purpose="encoder_measurement",launch_count=64)
        self.assertIn("::1",options)
        self.assertEqual(options[-2:],["--launch-count","64"])

    def test_long_csv_and_empty_captures(self):
        stream=io.StringIO();writer=csv.writer(stream)
        writer.writerow(["ID","Process ID","Kernel Name","Device","Context","Stream","Metric Name","Metric Unit","Metric Value"])
        writer.writerow([0,42,"gemm",1,1,7,"gpu__time_duration.sum","nsecond",100])
        writer.writerow([0,42,"gemm",1,1,7,"sm__warps_active.avg.pct_of_peak_sustained_active","%",50])
        result=parse_csv(stream.getvalue())
        self.assertEqual(result["summaries"][0]["metrics"]["achieved_occupancy_percent"]["duration_weighted_mean"],50)
        self.assertEqual(parse_csv("==PROF== No kernels were profiled.")["status"],"empty")

    def nvidia_csv(self, renamed=False, broken_quotes=True, purpose="video_measurement"):
        """Minimal NCU 2024.1 export, including its verbatim malformed quoting."""
        metadata = json.dumps(dict(frames=17, purpose=purpose, repeat=0,
                                   request_id='T0,quoted"request', step=1, variant="T0"),
                              separators=(",", ":"))
        label = "ditserve.workflow/dit_step|" + metadata
        kernel = "void templated_kernel<float, int>(float *, int)"
        stack = '42  "<default domain>:' + label + ':none:none:none:none:none:none" '
        stream = io.StringIO(); writer = csv.writer(stream)
        writer.writerow(["ID", "Process ID", "Process Name", "Host Name",
                         "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
                         "Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
                         "Kernel Name", "Context", "Stream", "Block Size", "Grid Size", "Device", "CC",
                         "gpu__time_duration.sum", "sm__warps_active.avg.pct_of_peak_sustained_active",
                         "dram__bytes_read.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed"])
        writer.writerow([""] * 13 + ["usecond", "%", "Mbyte", "%"])
        writer.writerow([0, 42, "python3.10", "127.0.0.1", stack, "",
                         label + "/" + kernel if renamed else kernel, 1, 7,
                         "(256, 1, 1)", "(512, 2, 1)", 1, "8.9", "4.064000", "53.684695", "0.146688", "n/a"])
        text = stream.getvalue()
        if broken_quotes:
            text = text.replace(metadata.replace('"', '""'), metadata)
        return text

    def test_real_nvidia_unescaped_json_and_renamed_kernel_suffix(self):
        result = parse_csv(self.nvidia_csv(), self.nvidia_csv(renamed=True))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["coverage"]["profiled_kernel_count"], 1)
        kernel = result["kernels"][0]
        self.assertEqual(kernel["stage"], "dit_step")
        self.assertEqual(kernel["metadata"]["request_id"], 'T0,quoted"request')
        self.assertEqual(kernel["metadata"]["frames"], 17)
        self.assertEqual(kernel["metadata"]["variant"], "T0")
        self.assertEqual(kernel["kernel_name"], "void templated_kernel<float, int>(float *, int)")
        self.assertEqual(kernel["metrics"]["duration_ns"], 4064)
        self.assertEqual(kernel["metrics"]["dram_read_bytes"], 146688)
        self.assertAlmostEqual(kernel["metrics"]["achieved_occupancy_percent"], 53.684695)
        self.assertIsNone(kernel["metrics"]["dram_throughput_percent"])
        self.assertIn("Repaired 1", result["warnings"][0])
        self.assertIn("Repaired 2", result["warnings"][1])
        self.assertTrue(kernel["nvtx_columns"])

    def test_valid_csv_and_raw_push_pop_column_association_are_preserved(self):
        result = parse_csv(self.nvidia_csv(broken_quotes=False))
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["kernels"][0]["stage"], "dit_step")
        self.assertEqual(result["kernels"][0]["association"], "NVTX rename/export")
        warm = parse_csv(self.nvidia_csv(purpose="warmup"))
        self.assertFalse(warm["kernels"][0]["included"])
        self.assertEqual(warm["coverage"]["included_profiled_kernel_count"], 0)
        self.assertEqual(warm["summaries"], [])

    def test_rename_identity_metrics_and_kernel_suffix_must_match(self):
        raw = self.make_csv()
        renamed = self.make_csv(True).replace(",10,20,80,100", ",11,20,80,100")
        result = parse_csv(raw, renamed)
        self.assertIsNone(result["kernels"][0]["nvtx_label"])
        self.assertIn("mismatched NVTX rename", result["warnings"][0])
        foreign = self.make_csv(True).replace(",1,1,7,", ",0,1,7,")
        self.assertTrue(all(k["nvtx_label"] is None for k in parse_csv(raw, foreign)["kernels"]))
        # A mismatching renamed symbol cannot override the valid raw NVTX stack.
        renamed = self.nvidia_csv(True).replace("/void templated_kernel", "/void other_kernel")
        result = parse_csv(self.nvidia_csv(), renamed)
        self.assertTrue(any("mismatched NVTX rename" in w for w in result["warnings"]))
        self.assertEqual(result["kernels"][0]["stage"], "dit_step")

    def test_malformed_width_unknown_units_and_duplicate_identity_are_not_guessed(self):
        raw = self.make_csv()
        extra = raw + '2,42,"bad, kernel",1,1,7,10,20,80,100,extra\n'
        result = parse_csv(extra)
        self.assertEqual(result["coverage"]["profiled_kernel_count"], 2)
        self.assertTrue(any("11 fields; expected 10" in w for w in result["warnings"]))
        wrong_unit = parse_csv(raw.replace("usecond", "cycles"))
        self.assertIsNone(wrong_unit["kernels"][0]["metrics"]["duration_ns"])
        self.assertTrue(wrong_unit["kernels"][0]["metric_errors"])
        with self.assertRaisesRegex(ValueError, "duplicate kernel identities"):
            parse_csv(raw + raw.splitlines()[-1] + "\n")


if __name__ == "__main__":
    unittest.main()
