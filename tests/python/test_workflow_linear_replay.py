"""CPU contracts for bound real-capture Linear replay; no CUDA or torch import."""

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.evaluation import workflow_linear_replay as replay


def bind(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(root):
    root = Path(root)
    captures = []
    for rank in (0, 1):
        directory = root / f"rank{rank}/seed3501/f17/localization/reference/step00/block00-qkv"
        directory.mkdir(parents=True)
        layers, files = [], {}
        for tensor, channels in (("to_q_input", 3), ("to_q", 2)):
            key = "cond.block00." + tensor
            path = directory / (key + ".npy")
            array = np.full((1, 3900, channels), rank + 1, dtype=np.float16)
            np.save(path, array, allow_pickle=False)
            files[key] = bind(path)
            layers.append(dict(key=key, path=str(path), hash=replay.array_digest(array),
                               block=0, branch="cond", tensor=tensor))
        path = directory / "capture.json"
        path.write_text(json.dumps(dict(status="completed", rank=rank, target_step=0, target_block=0,
            scheduler_steps=20, scope="selected_block_qkv",
            comparison_layout="SP2 local sequence vs matching SP1 contiguous sequence shard", layers=layers)))
        captures.append(dict(bind(path), rank=rank, files=files))
    shard = root / "model.safetensors"
    shard.write_bytes(b"bound fake shard; tensors provided only by the CPU safe_open double")
    index = root / "model.index.json"
    index.write_text(json.dumps(dict(weight_map={name: shard.name for name in
                           ("blocks.0.attn1.to_q.weight", "blocks.0.attn1.to_q.bias")})))
    return dict(device="cuda:0", gpu_uuid="GPU-5c1fadb6-2dcf-99be-374d-0bcbd8230979",
                cpu_affinity=list(range(60, 68)), operators=["to_q"], branches=["cond"],
                model_index=bind(index), model_shards={shard.name: bind(shard)},
                cases=[dict(seed=3501, frames=17, captures=captures)])


class SourceTests(unittest.TestCase):
    def test_import_remains_cpu_only(self):
        process = subprocess.run([sys.executable, "-c",
            "import sys; import ditserve.evaluation.workflow_linear_replay; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            check=False, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_config_rejects_ambiguous_devices_ranks_and_unbound_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            good = fixture(temp)
            self.assertEqual(replay.normalize_config(good)["repeats"], 3)
            for change in (dict(device="cuda:1"), dict(gpu_uuid="1"), dict(repeats=1),
                           dict(cpu_affinity=[60]*8), dict(operators=["other"]), dict(branches=[]),
                           dict(candidates=["global_tile_512"]), dict(candidates=[]),
                           dict(candidates=["linear_fp32", "linear_fp32"]),
                           dict(microbenchmark=dict(enabled=True, repeats=1)),
                           dict(microbenchmark=dict(enabled="true")),
                           dict(microbenchmark=dict(enabled=True, warmups=0)),
                           dict(model_index=dict(path="relative", sha256="0"*64))):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    replay.normalize_config(dict(good, **change))
            bad = copy.deepcopy(good)
            bad["cases"][0]["captures"][1]["rank"] = 0
            with self.assertRaisesRegex(ValueError, "rank0"):
                replay.normalize_config(bad)
            bad = copy.deepcopy(good)
            bad["cases"][0]["captures"][0]["files"] = {}
            with self.assertRaisesRegex(ValueError, "file SHA256"):
                replay.normalize_config(bad)

    def test_rank_concatenation_and_file_vs_array_hashes_are_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            config = replay.normalize_config(fixture(temp))
            case, bindings = config["cases"][0], replay.Bindings()
            sources = [replay.load_capture(case, item, config, bindings) for item in case["captures"]]
            key = "cond.block00.to_q_input"
            self.assertNotEqual(sources[0]["tensors"][key]["file"]["sha256"],
                                sources[0]["tensors"][key]["array"]["sha256"])
            x, y = replay.assemble_pair(list(reversed(sources)), key, "cond.block00.to_q", bindings)
            self.assertEqual(x.shape, (1, 7800, 3))
            self.assertEqual(y.shape, (1, 7800, 2))
            np.testing.assert_array_equal(x[:, :3900], 1)
            np.testing.assert_array_equal(x[:, 3900:], 2)

    def test_file_corruption_array_hash_path_and_token_shape_are_rejected(self):
        for failure in ("file", "array", "path", "shape"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                config = replay.normalize_config(fixture(temp))
                case, item = config["cases"][0], config["cases"][0]["captures"][0]
                manifest_path = Path(item["path"])
                manifest = json.loads(manifest_path.read_text())
                key = "cond.block00.to_q_input"
                tensor_path = Path(item["files"][key]["path"])
                if failure == "file":
                    tensor_path.write_bytes(tensor_path.read_bytes() + b"changed")
                elif failure == "array":
                    manifest["layers"][0]["hash"]["sha256"] = "0"*64
                elif failure == "path":
                    manifest["layers"][0]["path"] = str(tensor_path.parent / "unrelated.npy")
                else:
                    np.save(tensor_path, np.ones((1, 12, 3), dtype=np.float16), allow_pickle=False)
                    item["files"][key] = bind(tensor_path)
                if failure != "file":
                    manifest_path.write_text(json.dumps(manifest))
                    item.update(bind(manifest_path))
                with self.assertRaises(ValueError):
                    replay.load_capture(case, item, config, replay.Bindings())

    def test_shared_model_shard_is_hashed_once_and_changed_binding_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, bindings = replay.normalize_config(fixture(temp)), replay.Bindings()
            plan = replay.plan_model(config, bindings)
            self.assertEqual(len(plan["tensors"]), 2)
            self.assertEqual(len(bindings.cache), 2)  # index plus one shared shard
            broken = dict(next(iter(config["model_shards"].values())), sha256="0"*64)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                bindings.verify(broken)
            with self.assertRaises(ValueError):
                replay.validate_linear_shapes((1, 8, 3), (1, 8, 2), (3, 2), (2,))


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape, self.dtype = self.data.shape, self.data.dtype

    def to(self, *, dtype=None, device=None):
        return Tensor(self.data.astype(dtype) if dtype is not None else self.data)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data))

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, dtype=self.dtype))

    def clone(self, **kwargs):
        return Tensor(self.data.copy())

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def __setitem__(self, key, value):
        self.data[key] = value.data

    def stride(self):
        return tuple(step // self.data.itemsize for step in self.data.strides)

    def numpy(self):
        return self.data

    def detach(self):
        return self

    cpu = detach


def fake_torch():
    flags = types.SimpleNamespace(**replay.DEFAULT_FLAGS)
    def linear(x, w, b):
        output = x.data.astype(np.float32) @ w.data.astype(np.float32).T + b.data.astype(np.float32)
        # A deterministic half-row-only perturbation stands in for GEMM
        # arithmetic sensitivity, without pretending NumPy models CUDA kernels.
        if x.shape[1] == 4 and x.dtype == np.float16 and flags.allow_fp16_reduced_precision_reduction:
            output += .25
        return Tensor(output.astype(x.dtype))
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=flags)),
        float16=np.dtype("float16"), float32=np.dtype("float32"), contiguous_format="contiguous",
        from_numpy=Tensor, zeros_like=lambda x: Tensor(np.zeros_like(x.data)),
        cat=lambda values, dim: Tensor(np.concatenate([v.data for v in values], axis=dim)),
        inference_mode=contextlib.nullcontext, nn=types.SimpleNamespace(functional=types.SimpleNamespace(linear=linear)))


class ExecutionContractTests(unittest.TestCase):
    def test_candidate_recipe_selection_anchor_bridge_and_fp16_weight_promotion(self):
        torch = fake_torch()
        x = np.arange(24, dtype=np.float16).reshape(1, 8, 3) / 100
        weight = Tensor(np.array([[1., 2., 3.], [-1., .5, 2.]], dtype=np.float16))
        bias = Tensor(np.array([1., -1.], dtype=np.float16))
        reference = torch.nn.functional.linear(Tensor(x), weight, bias).data
        config = dict(device="cuda:0", repeats=3, zero_padded=True, save_arrays=False,
                      candidates=list(replay.CANDIDATES))
        with tempfile.TemporaryDirectory() as temp:
            result = replay.replay_operator(torch, x, reference, weight, bias, config, Path(temp), operator="to_q")
        rows = {row["candidate"]: row for row in result["candidates"]}
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["qualified"] and row["reference_reproduced"] for row in rows.values()))
        self.assertFalse(rows["full_rows_ffn2"]["arms"][1]["recipe_applied"])
        self.assertFalse(rows["full_rows_ffn2"]["arms"][1]["comparison_to_full_same_precision"]["exact"])
        self.assertTrue(rows["full_rows_ffn2_qkv"]["arms"][1]["comparison_to_full_same_precision"]["exact"])
        for candidate in ("global_tile_1024", "global_tile_2048", "linear_fp32"):
            arms = rows[candidate]["arms"]
            self.assertTrue(all(arm["comparison_to_reference"]["exact"] for arm in arms))
            self.assertTrue(all(arm["comparison_to_full_same_precision"]["exact"] for arm in arms[1:]))
        self.assertEqual(rows["linear_fp32"]["arms"][0]["recipe"]["weight_source"], "execution_fp16")
        self.assertEqual(replay.arithmetic_flags(torch), replay.DEFAULT_FLAGS)
        with self.assertRaisesRegex(ValueError, "actual FP16"):
            replay.candidate_operation(torch, "linear_fp32", "to_q", Tensor(x),
                weight.to(dtype=np.float32), bias, global_start=0, global_rows=8)

    def test_new_recipe_agreement_cannot_replace_original_reference_bridge(self):
        torch = fake_torch()
        x = np.ones((1, 8, 3), dtype=np.float16)
        w, b = Tensor(np.ones((2, 3), dtype=np.float16)), Tensor(np.zeros(2, dtype=np.float16))
        native = torch.nn.functional.linear
        # Candidate tiling is mutually exact but intentionally drifts from old SP1.
        def perturbed(value, weight, bias):
            output = native(value, weight, bias)
            return Tensor(output.data + np.float16(1)) if value.shape[1] == 1024 else output
        torch.nn.functional.linear = perturbed
        reference = native(Tensor(x), w, b).data
        config = dict(device="cuda:0", repeats=2, zero_padded=False, save_arrays=False,
                      candidates=["global_tile_1024"])
        with tempfile.TemporaryDirectory() as temp:
            result = replay.replay_operator(torch, x, reference, w, b, config, Path(temp), operator="to_q")
        candidate = result["candidates"][0]
        self.assertTrue(result["reference_reproduced"] and candidate["qualified"])
        self.assertEqual(candidate["arms"][0]["comparison_to_reference"]["max_abs"], 1)
        self.assertTrue(all(arm["comparison_to_full_same_precision"]["exact"] for arm in candidate["arms"][1:]))
        self.assertFalse(candidate["deployment_qualification"])

    def test_fp32_promotes_quantized_values_and_exception_restores_flags(self):
        torch, observed = fake_torch(), []
        x = Tensor(np.ones((1, 8, 3), dtype=np.float16))
        quantized = Tensor(np.full((2, 3), np.float32(1.0001), dtype=np.float32)).to(dtype=np.float16)
        bias = Tensor(np.zeros(2, dtype=np.float16))
        def broken(value, weight, bias):
            observed.append((value.dtype, weight.dtype, weight.data.copy()))
            raise RuntimeError("FP32 operator failed")
        torch.nn.functional.linear = broken
        torch.backends.cuda.matmul.allow_tf32 = True
        original_flags = replay.arithmetic_flags(torch)
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(RuntimeError, "FP32 operator failed"):
            replay.replay_candidates(torch, x, np.zeros((1, 8, 2), dtype=np.float16),
                np.zeros((1, 8, 2), dtype=np.float16), quantized, bias, "to_q",
                dict(candidates=["linear_fp32"], repeats=2, save_arrays=False), Path(temp), True)
        self.assertEqual(observed[0][0], np.float32)
        self.assertEqual(observed[0][1], np.float32)
        np.testing.assert_array_equal(observed[0][2], np.ones((2, 3), dtype=np.float32))
        self.assertEqual(replay.arithmetic_flags(torch), original_flags)

    def test_event_microbenchmark_excludes_hashing_and_executes_padding_inside_timing(self):
        torch, calls = fake_torch(), []
        class Event:
            def __init__(self, enable_timing):
                self.assert_timing = enable_timing
            def record(self):
                calls.append("event")
            def synchronize(self):
                calls.append("wait")
            def elapsed_time(self, end):
                return 1.25
        torch.cuda = types.SimpleNamespace(Event=Event, synchronize=lambda: calls.append("warmup_wait"))
        config = dict(microbenchmark=dict(enabled=True, warmups=2, repeats=3))
        with patch.object(replay, "array_digest", side_effect=AssertionError("hash inside timer")):
            result = replay.benchmark_operator(torch, lambda: calls.append("operator"), config)
        self.assertEqual(calls[:3], ["operator", "operator", "warmup_wait"])
        self.assertEqual(calls[3:], ["event", "operator", "event", "wait"] * 3)
        self.assertEqual(result["median_ms"], 1.25)
        self.assertFalse(result["request_performance_qualification"])
        self.assertIsNone(replay.benchmark_operator(torch, lambda: None, {}))

    def test_replay_integrity_never_qualifies_complete_request_numerics(self):
        for passed in (None, False, True):
            result = replay.qualification_record(passed)
            self.assertEqual(result["passed"], passed is True)
            self.assertEqual(result["scopes"]["numeric"]["status"], "unverified")
            self.assertEqual(result["scopes"]["replay_integrity"]["status"],
                             "unverified" if passed is None else "passed" if passed else "failed")

    def test_only_selected_weights_load_and_original_and_fp16_values_are_bound(self):
        torch, calls = fake_torch(), []
        names = ("blocks.0.attn1.to_q.weight", "blocks.0.attn1.to_q.bias")
        tensors = {names[0]: Tensor(np.full((2, 3), 1.0001, dtype=np.float32)),
                   names[1]: Tensor(np.array([.01, .02], dtype=np.float32))}
        @contextlib.contextmanager
        def safe_open(path, framework, device):
            self.assertEqual((framework, device), ("pt", "cpu"))
            yield types.SimpleNamespace(get_tensor=lambda name: calls.append(name) or tensors[name])
        plan = dict(tensors={name: dict(name=name, file=dict(path="/fake/shard")) for name in names})
        with patch.dict(sys.modules, {"safetensors": types.SimpleNamespace(safe_open=safe_open)}):
            values, records = replay.load_weights(plan, torch)
        self.assertEqual(calls, list(names))
        self.assertTrue(all(value.dtype == np.float16 for value in values.values()))
        self.assertEqual(records[names[0]]["original"]["dtype"], "float32")
        self.assertEqual(records[names[0]]["execution_fp16"]["dtype"], "float16")
        self.assertNotEqual(records[names[0]]["original"]["sha256"],
                            records[names[0]]["execution_fp16"]["sha256"])

    def test_precision_flags_restore_even_when_operator_fails(self):
        torch = fake_torch()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with replay.precision_mode(torch, "no_reduced_precision") as audit:
                self.assertFalse(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                self.assertFalse(torch.backends.cuda.matmul.allow_fp16_accumulation)
                raise RuntimeError("failure")
        self.assertEqual(replay.arithmetic_flags(torch), replay.DEFAULT_FLAGS)
        self.assertEqual(audit["restored"], replay.DEFAULT_FLAGS)

    def test_real_matrix_comparisons_do_not_require_half_exact_and_reference_anchor_is_separate(self):
        torch = fake_torch()
        x = np.arange(24, dtype=np.float16).reshape(1, 8, 3) / 100
        weight = Tensor(np.array([[1., 2., 3.], [-1., .5, 2.]], dtype=np.float16))
        bias = Tensor(np.array([1., -1.], dtype=np.float16))
        reference = (x.astype(np.float32) @ weight.data.astype(np.float32).T +
                     bias.data.astype(np.float32)).astype(np.float16)
        config = dict(device="cuda:0", repeats=3, zero_padded=True, save_arrays=False)
        for corrupt_anchor in (False, True):
            with self.subTest(corrupt_anchor=corrupt_anchor), tempfile.TemporaryDirectory() as temp:
                expected = reference + np.float16(1) if corrupt_anchor else reference
                result = replay.replay_operator(torch, x, expected, weight, bias, config, Path(temp))
                self.assertEqual(len(result["arms"]), 10)
                self.assertTrue(result["qualified"])
                self.assertEqual(result["reference_reproduced"], not corrupt_anchor)
                self.assertFalse(result["performance_evidence"])
                half = result["arms"][1]
                self.assertTrue(half["repeated_exact"])
                self.assertFalse(half["comparison_to_full_same_precision"]["exact"])
                self.assertTrue(result["arms"][3]["comparison_to_full_same_precision"]["exact"])
                self.assertFalse(result["arms"][6]["comparison_to_default_same_shape"]["exact"])
                if corrupt_anchor:
                    self.assertIn("do not attribute", result["interpretation"])
        self.assertEqual(replay.arithmetic_flags(torch), replay.DEFAULT_FLAGS)

    def test_environment_and_affinity_restore_after_pre_torch_exception(self):
        cpus = sorted(os.sched_getaffinity(0))
        selected = cpus[:min(2, len(cpus))]
        old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        config = dict(gpu_uuid="GPU-5c1fadb6-2dcf-99be-374d-0bcbd8230979",
                      cpu_affinity=selected, cpu_threads=len(selected))
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with replay.isolated_environment(config):
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], config["gpu_uuid"])
                self.assertEqual(os.sched_getaffinity(0), set(selected))
                raise RuntimeError("failure")
        self.assertEqual(os.sched_getaffinity(0), set(cpus))
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), old_visible)

    def test_source_failure_is_durable_and_existing_results_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = fixture(root / "inputs")
            cpus = sorted(os.sched_getaffinity(0))
            config.update(cpu_threads=1, cpu_affinity=cpus[:1])
            config["model_index"]["sha256"] = "0"*64
            result = replay.run_replay(config, root / "output")
            self.assertEqual(result["status"], "failed")
            self.assertIn("SHA256 mismatch", result["error"])
            self.assertFalse(result["qualification"]["passed"])
            self.assertFalse(result["performance_evidence"])
            self.assertTrue((root / "output/run.json").is_file())
            with self.assertRaises(FileExistsError):
                replay.run_replay(config, root / "output")


if __name__ == "__main__":
    unittest.main()
