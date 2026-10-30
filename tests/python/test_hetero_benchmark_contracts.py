"""Offline contracts: execute benchmark orchestration with model/clock boundaries stubbed."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "benchmarks/run_hetero_quality_and_replay.py"
V2 = ROOT / "benchmarks/run_hetero_v2_optimizations.py"
V3 = ROOT / "benchmarks/run_hetero_v3_optimizations.py"
HELPERS = {
    "_wait_until_arrival", "_image_requests", "_completion_records",
    "_run_image_batch", "_decode_video", "_replay_summary",
}
CONSTANTS = {"REPLAY_SCOPE", "IMAGE_SCOPE", "DENOISE_SCOPE", "TAYLOR_CONFIG"}


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS for target in node.targets
        ):
            body.append(node)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class Clock:
    def __init__(self):
        self.value = 0.0
        self.waits = []

    def perf_counter(self):
        return self.value

    def sleep(self, seconds):
        if seconds <= 0:
            raise AssertionError("only a positive arrival wait is allowed")
        self.waits.append(seconds)
        self.value += seconds

    def advance(self, seconds=0.001):
        self.value += seconds


class Output:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        return Output(self.shape[1:])


class Images:
    def __init__(self, clock, wrong_count=False):
        self.clock = clock
        self.wrong_count = wrong_count
        self.calls = []
        self.pipe = SimpleNamespace(vae=SimpleNamespace(to=lambda **kwargs: None))

    def _ensure_staged(self, *args):
        pass

    def offload_encoders(self, request):
        pass

    def run_micro_batch(self, request):
        self.calls.append((dict(request), self.clock.value))
        self.clock.advance()
        count = 1 if self.wrong_count else len(request["prompts"])
        return dict(images=Output((count, request["height"], request["width"], 3)),
                    model_ms=1.0, per_image_ms=1.0 / count,
                    batch_size=len(request["prompts"]), reused_steps=0)


class Video:
    def __init__(self, clock):
        self.clock = clock
        self.states = {}
        self.execution = "step"
        self.decode_calls = {}
        self.decode_finished = []
        self.runtime = self
        self.pipe = SimpleNamespace(vae=SimpleNamespace(to=lambda **kwargs: None))

    def _spec(self, request):
        return request["spec"]

    def encode_prompt(self, spec):
        return object()

    def prepare(self, sid, spec):
        state = SimpleNamespace(spec=spec, request_id=sid, step_index=0, events=[], blocks=0)
        self.states[sid] = state
        return state

    def step(self, state):
        state.step_index += 1
        self.clock.advance()
        event = dict(kind="step", reused_forwards=0)
        state.events.append(event)
        return event

    def advance(self, request):
        state = self.states[request["state_id"]]
        if "blocks" in request:
            state.blocks += request["blocks"]
            if state.blocks >= 60:
                self.step(state)
        else:
            for _ in range(request.get("steps", 1)):
                self.step(state)

    def decode_chunks(self, state, count, **options):
        if state.step_index != state.spec["steps"]:
            raise AssertionError("decode before denoise completion")
        self.clock.advance()
        calls = self.decode_calls.get(state.request_id, 0) + 1
        self.decode_calls[state.request_id] = calls
        if calls < 2:
            return None, {"done": False}
        self.decode_finished.append(self.clock.value)
        spec = state.spec
        return Output((1, spec["frames"], spec["height"], spec["width"], 3)), {"done": True}

    def tiered_offload(self, request):
        self.clock.advance()
        return {"kind": "offload"}

    def tiered_prefetch(self, request):
        self.clock.advance()
        return {"kind": "prefetch"}

    def release(self, request):
        del self.states[request["state_id"]]


class Stepper:
    def __init__(self, *args, **kwargs):
        self.stats = {"folded_cfg_steps": 0, "saved_uncond_forwards": 0}

    def step(self, runtime, state, **kwargs):
        return runtime.step(state)


class ReportSink:
    def __truediv__(self, name):
        return self

    def write_text(self, text, **kwargs):
        self.text = text


class BenchmarkContracts(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        cuda = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None,
                               reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0)
        self.env = load_functions(V1, HELPERS, {
            "math": math, "time": self.clock, "torch": SimpleNamespace(
                cuda=cuda, bfloat16="bf16", channels_last="channels_last", contiguous_format="contiguous"),
            "print": lambda *args, **kwargs: None,
        })

    def requests(self, count=2, arrivals=None):
        return self.env["_image_requests"]([f"prompt-{i}" for i in range(count)],
                                           list(range(10, 10 + count)), arrivals)

    def test_one_image_cannot_complete_two_requests(self):
        requests = self.requests()
        with self.assertRaisesRegex(ValueError, "output count/shape"):
            self.env["_completion_records"](requests, Output((1, 512, 512, 3)), 0, 1)
        with self.assertRaises(ValueError):
            self.env["_run_image_batch"](Images(self.clock, wrong_count=True), requests, 0)
        with self.assertRaises(ValueError):
            self.env["_completion_records"](requests, None, 0, 1)

    def test_real_outputs_preserve_prompt_seed_and_identity(self):
        backend = Images(self.clock)
        requests = self.requests(arrivals=[250.0, 280.0])
        result = self.env["_run_image_batch"](backend, requests, 0, attention="flash")
        arguments, started = backend.calls[0]
        self.assertEqual(arguments["prompts"], ["prompt-0", "prompt-1"])
        self.assertEqual(arguments["seeds"], [10, 11])
        self.assertTrue(arguments["return_images"])
        self.assertGreaterEqual(started * 1000, 280)
        self.assertEqual([r["output_index"] for r in result["requests"]], [0, 1])
        self.assertEqual([r["output_count"] for r in result["requests"]], [1, 1])
        self.assertEqual([r["seed"] for r in result["requests"]], [10, 11])
        self.assertNotIn("per_image_ms", result)
        self.assertAlmostEqual(result["batch_latency_ms"], 2 * result["amortized_ms_per_image"])

    def test_arrival_wait_has_no_baseline_penalty(self):
        wait = self.env["_wait_until_arrival"]
        self.assertGreaterEqual(wait(0, 250), 250)
        self.assertEqual(self.clock.waits, [0.25])
        wait(0, 250)
        wait(0, 100)
        self.assertEqual(self.clock.waits, [0.25])
        for arrival in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                wait(0, arrival)
        with self.assertRaisesRegex(ValueError, "before arrival"):
            self.env["_completion_records"](self.requests(1, [280]), Output((1, 512, 512, 3)), 250, 300)

    def test_decode_waits_for_final_chunk_and_all_frames(self):
        wan = Video(self.clock)
        state = wan.prepare("v0", dict(steps=20, frames=17, height=480, width=832, seed=42))
        state.step_index = 20
        video = self.env["_decode_video"](wan, state)
        self.assertEqual(wan.decode_calls["v0"], 2)
        request = dict(request_id="v0", modality="video", arrival_ms=0, spec=state.spec)
        with self.assertRaisesRegex(ValueError, "all requested frames"):
            self.env["_completion_records"]([request], Output((1, 16, 480, 832, 3)), 0, 2)
        records = self.env["_completion_records"]([request], video, 0, self.clock.value * 1000)
        self.assertEqual(records[0]["finish_ms"], wan.decode_finished[-1] * 1000)

    def test_max_is_not_p99(self):
        images = self.env["_completion_records"](self.requests(3), Output((3, 512, 512, 3)), 0, 10)
        video = dict(request_id="v0", modality="video", output_count=1, latency_ms=30, finish_ms=30)
        result = self.env["_replay_summary"](images + [video], 20, 40)
        self.assertEqual(result["image_max_latency_ms"], 10)
        self.assertEqual(result["image_sample_count"], 3)
        self.assertFalse(any("p99" in key for key in result))
        self.assertEqual(result["evidence_type"], "component_replay")
        self.assertIn("not evidence of online scheduling", result["scheduler_scope"])

    def test_v1_policies_same_workload_and_decode_before_completion(self):
        wan, sd = Video(self.clock), Images(self.clock)
        env = load_functions(V1, {"run_real_mixed_trace_replay"}, self.env)
        env["_fast_prepare_wan"] = lambda backend, sid, spec, payload: backend.prepare(sid, spec)
        rows = env["run_real_mixed_trace_replay"](wan, sd, None)
        self.assertEqual(len(rows), 4)
        signatures = []
        for row in rows:
            self.assertEqual(row["completed_requests"], 4)
            self.assertEqual(row["image_sample_count"], 3)
            self.assertEqual(row["evidence_type"], "component_replay")
            signatures.append(sorted((r["request_id"], r["seed"], r["arrival_ms"],
                                      r["spec"]["prompt"], r["output_count"], r["output_mode"])
                                     for r in row["requests"]))
            for request in row["requests"]:
                self.assertGreaterEqual(request["start_ms"], request["arrival_ms"])
                self.assertGreaterEqual(request["finish_ms"], request["start_ms"])
        self.assertTrue(all(signature == signatures[0] for signature in signatures))
        self.assertEqual(len(wan.decode_finished), 4)
        self.assertFalse(wan.states)
        self.assertEqual(wan.execution, "step")
        self.assertTrue(any(len(call[0]["prompts"]) == 2 for call in sd.calls))

    def run_replay(self, path, name):
        env = load_functions(path, {name}, self.env)
        wan, sd = Video(self.clock), Images(self.clock)
        env.update(_fast_prepare_wan=lambda backend, spec, sid, payload: backend.prepare(sid, spec),
                   TaylorResidualCompensator=Stepper, AsymmetricCFGStepper=Stepper)
        return env[name](wan, sd, object())

    def test_v2_v3_fast_backends_cannot_serve_before_arrival(self):
        for path, name in ((V2, "benchmark_e2e_v2_trace_replay"), (V3, "benchmark_e2e_v3_trace_replay")):
            with self.subTest(version=path.name):
                row = self.run_replay(path, name)
                self.assertEqual(row["completed_requests"], 4)
                self.assertEqual(row["image_sample_count"], 3)
                self.assertIn("excludes prompt encoding", row["timing_scope"])
                self.assertIn("reused_steps", row)
                for request in row["requests"]:
                    self.assertGreaterEqual(request["start_ms"], request["arrival_ms"])
                    self.assertEqual(request["output_count"], 1)
                    self.assertEqual(request["seed"], request["spec"]["seed"])

    def test_missing_quality_and_cache_without_reuse(self):
        env = load_functions(V3, {"_format_quality", "benchmark_opt11_sd35_flash_channels_last_and_cache"}, self.env)
        env.update(compute_psnr=lambda *args: 40.0, compute_ssim_frame=lambda *args: 0.9,
                   np=SimpleNamespace(mean=lambda values: sum(values) / len(values)))
        rows = env["benchmark_opt11_sd35_flash_channels_last_and_cache"](Images(self.clock))["modes"]
        self.assertEqual(env["_format_quality"](None), "未评测")
        self.assertEqual(env["_format_quality"](0.9, 4), "0.9000")
        for index, status in ((0, "reference"), (3, "not_evaluated")):
            self.assertIsNone(rows[index]["psnr_db"])
            self.assertIsNone(rows[index]["ssim"])
            self.assertEqual(rows[index]["quality_status"], status)
            self.assertIsNone(rows[index]["independent_quality_pass"])
        for row in rows:
            self.assertEqual(row["reused_steps"], 0)
            self.assertIsNone(row["cache_speedup_vs_matched_exact"])
            self.assertEqual(row["batch_latency_ms"], row["amortized_ms_per_image"] * row["batch_size"])

    def test_fixed_taylor_and_evidence_labels(self):
        env = load_functions(V2, set(), self.env)
        self.assertEqual(env["TAYLOR_CONFIG"]["alpha"], 0.22)
        self.assertIs(env["TAYLOR_CONFIG"]["adaptive_lsq"], False)
        tree = ast.parse(V2.read_text(encoding="utf-8"))
        taylor = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "benchmark_opt3_taylor_cache")
        names = {n.id for n in ast.walk(taylor) if isinstance(n, ast.Name)}
        self.assertNotIn("best_taylor_row", names)
        self.assertNotIn("cand_alpha", names)
        text = V2.read_text(encoding="utf-8")
        for label in ("estimated_cross_node_comm_speedup", "communication_assumptions",
                      "codec_cost_fraction", "cross_node_bandwidth_gib_s", "network_transfer_measured=False"):
            self.assertIn(label, text)
        for path in (V1, V2, V3):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, "sleep", "only shared arrival wait may sleep")
            self.assertNotIn("Full-Stack", path.read_text(encoding="utf-8"))
        v3 = V3.read_text(encoding="utf-8")
        self.assertIn('evidence_type="simulation"', v3)
        self.assertIn("not real eight-GPU evidence", v3)

    def test_v3_report_formats_missing_quality_without_files(self):
        env = load_functions(V3, {"_format_quality", "write_markdown_report"}, self.env)
        replay = self.run_replay(V3, "benchmark_e2e_v3_trace_replay")
        missing_row = dict(mode="fixture", batch_size=4, batch_latency_ms=10, amortized_ms_per_image=2.5,
                           reused_steps=0, speedup_vs_baseline=1, cache_speedup_vs_matched_exact=None,
                           psnr_db=None, ssim=None, quality_status="not_evaluated")
        v10 = dict.fromkeys(("fp32_vae_decode_ms", "fp32_peak_hbm_mb", "bf16_vae_decode_ms", "bf16_peak_hbm_mb",
                            "bf16_stream_overlapped_with_sd35_wall_ms"), 1)
        v10.update(timing_scope="decode only", overlap_timing_scope="preencoded", bf16_vs_fp32_psnr_db=None,
                   bf16_vs_fp32_ssim=None)
        payload = dict(device_name="offline fixture", opt9_asymmetric_cfg_taylor={"modes": []},
                       opt10_bf16_3dvae=v10, opt11_sd35_flash_channels_last_cache={"modes": [missing_row]},
                       component_replay_v3=replay, opt12_8a100_study={"ablation": {"policies": {}}})
        sink = ReportSink()
        env["write_markdown_report"](sink, payload)
        self.assertIn("未评测", sink.text)
        self.assertIn("image max (n=3)", sink.text)
        self.assertIn("simulation", sink.text)
        self.assertIn("component_replay", sink.text)
        self.assertIn("Amortized", sink.text)
        self.assertNotIn("99.99", sink.text)


if __name__ == "__main__":
    unittest.main()
