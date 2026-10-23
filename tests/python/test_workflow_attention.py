"""Weight-free tests of diagnostic Linear padding and flag restoration."""

import types
import unittest

import numpy as np

from ditserve.evaluation import workflow_attention as workflow


class Tensor:
    def __init__(self, value):
        self.data = np.asarray(value)
        self.shape, self.ndim, self.dtype, self.device = self.data.shape, self.data.ndim, self.data.dtype, "cpu"

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, dtype=self.dtype))

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def __setitem__(self, key, value):
        self.data[key] = value.data

    def stride(self):
        return tuple(s // self.data.itemsize for s in self.data.strides)

    def clone(self, *, memory_format):
        return Tensor(self.data.copy(order="C"))

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.data))

    def storage_offset(self):
        root = self.data
        while isinstance(root.base, np.ndarray):
            root = root.base
        return (self.data.ctypes.data - root.ctypes.data) // self.data.itemsize


class Module:
    def __init__(self):
        self.prehooks, self.hooks = [], []

    def register_forward_pre_hook(self, hook, *, with_kwargs=False, prepend=False):
        item = (hook, with_kwargs)
        self.prehooks.insert(0, item) if prepend else self.prehooks.append(item)
        return types.SimpleNamespace(remove=lambda: self.prehooks.remove(item))

    def register_forward_hook(self, hook, *, prepend=False, always_call=False):
        item = (hook, always_call)
        self.hooks.insert(0, item) if prepend else self.hooks.append(item)
        return types.SimpleNamespace(remove=lambda: self.hooks.remove(item))

    def named_modules(self):
        yield "", self

    def forward(self, value):
        return value

    def __call__(self, *args, **kwargs):
        try:
            for hook, with_kwargs in self.prehooks:
                changed = hook(self, args, kwargs) if with_kwargs else hook(self, args)
                if changed is not None:
                    args, kwargs = changed
            output = self.forward(args[0] if args else kwargs["input"])
        except BaseException:
            for hook, always_call in self.hooks:
                if always_call:
                    hook(self, args, None)
            raise
        for hook, _ in self.hooks:
            changed = hook(self, args, output)
            if changed is not None:
                output = changed
        return output


class Linear(Module):
    def __init__(self):
        super().__init__()
        self.in_features, self.out_features = 3, 2
        self.weight = np.array([[1., 2., 3.], [-2., 1., .5]], dtype=np.float32)
        self.bias = np.array([7., -3.], dtype=np.float32)
        self.inputs, self.fail = [], False

    def forward(self, value):
        self.inputs.append(value)
        if self.fail:
            raise RuntimeError("GEMM failed")
        return Tensor(value.data @ self.weight.T + self.bias)


class Block(Module):
    def __init__(self):
        super().__init__()
        self.projection, self.text = Linear(), Linear()

    def named_modules(self):
        yield "", self
        yield "projection", self.projection
        yield "text", self.text


def backend(rank=0, degree=2):
    model = types.SimpleNamespace(patch_embedding=Module(), blocks=[Block()], proj_out=Linear())
    flags = types.SimpleNamespace(allow_tf32=False, allow_fp16_reduced_precision_reduction=True,
                                  allow_fp16_accumulation=False)
    torch = types.SimpleNamespace(nn=types.SimpleNamespace(Linear=Linear), contiguous_format="contiguous",
                                  cat=lambda values, dim: Tensor(np.concatenate([v.data for v in values], axis=dim)),
                                  backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=flags)))
    value = types.SimpleNamespace(
        handle=types.SimpleNamespace(rank=rank, current=types.SimpleNamespace(
            degree=degree, ranks=tuple(range(degree)), epoch=0)),
        worker=types.SimpleNamespace(torch=torch, pipe=types.SimpleNamespace(transformer=model)),
        _require_active=lambda: None)
    # Simulate a pre-existing patch split, registered before our context.
    model.patch_embedding.register_forward_hook(
        lambda module, args, out: Tensor(out.data[..., :out.shape[-1] // degree]))
    return value, model, flags


class FullRowsTests(unittest.TestCase):
    def patch(self, model):
        model.patch_embedding(Tensor(np.zeros((1, 3, 1, 1, 8), dtype=np.float32)))

    def test_rank_placement_bias_crop_and_original_parameters(self):
        full = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
        outputs = []
        for rank in (0, 1):
            worker, model, _ = backend(rank)
            linear = model.blocks[0].projection
            weights, bias = linear.weight, linear.bias
            with workflow.attention_variant(worker, "full_rows_linear") as audit:
                self.patch(model)
                value = Tensor(full[:, rank*4:(rank+1)*4])
                result = linear(value)
                outputs.append(result.data)
                padded = linear.inputs[-1].data
                self.assertEqual(padded.shape, (1, 8, 3))
                np.testing.assert_array_equal(padded[:, rank*4:(rank+1)*4], value.data)
                np.testing.assert_array_equal(padded[:, (1-rank)*4:(2-rank)*4], 0)
                self.assertEqual(result.storage_offset(), 0)
                self.assertEqual(result.stride(), (8, 2, 1))
                self.assertIs(linear.weight, weights)
                self.assertIs(linear.bias, bias)
                self.assertTrue(audit["diagnostic_only"])
                self.assertFalse(audit["deployment_qualification"])
            self.assertTrue(audit["hooks_removed"])
            self.assertEqual(audit["patch_calls"], 1)
            self.assertEqual(audit["applied_calls"], 1)
            self.assertFalse(linear.prehooks or linear.hooks)
            self.assertEqual(len(model.patch_embedding.hooks), 1)
        np.testing.assert_array_equal(np.concatenate(outputs, axis=1), full @ weights.T + bias)

    def test_proj_out_is_cropped_before_existing_gather_and_noncontiguous_input_works(self):
        worker, model, _ = backend(rank=1)
        seen = []
        model.proj_out.register_forward_hook(
            lambda module, args, out: seen.append(tuple(out.shape)))
        value = Tensor(np.arange(24, dtype=np.float32).reshape(1, 3, 8).transpose(0, 2, 1)[:, 4:])
        self.assertFalse(value.data.flags.c_contiguous)
        with workflow.full_rows_linear(worker):
            self.patch(model)
            output = model.proj_out(input=value)
        self.assertEqual(seen, [(1, 4, 2)])
        np.testing.assert_array_equal(output.data, value.data @ model.proj_out.weight.T + model.proj_out.bias)
        self.assertEqual(output.storage_offset(), 0)
        self.assertEqual(len(model.proj_out.hooks), 1)

    def test_text_and_conditioning_shapes_skip_and_sp1_is_identity(self):
        worker, model, _ = backend()
        with workflow.full_rows_linear(worker) as audit:
            self.patch(model)
            for shape in ((1, 512, 3), (1, 3)):
                value = Tensor(np.ones(shape, dtype=np.float32))
                model.blocks[0].text(value)
                self.assertIs(model.blocks[0].text.inputs[-1], value)
            self.assertEqual(audit["applied_calls"], 0)
            self.assertEqual(audit["skipped_calls"], 2)
        worker, model, _ = backend(degree=1)
        with workflow.full_rows_linear(worker) as audit:
            self.patch(model)
            value = Tensor(np.ones((1, 8, 3), dtype=np.float32))
            model.blocks[0].projection(value)
            self.assertIs(model.blocks[0].projection.inputs[-1], value)
            self.assertEqual(audit["applied_calls"], 0)
            self.assertEqual(audit["shapes"][0]["action"], "unchanged_sp1")

    def test_exception_cleans_hooks_and_missing_or_changed_geometry_is_rejected(self):
        for failure in ("GEMM", "missing", "changed"):
            worker, model, _ = backend()
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                with workflow.full_rows_linear(worker) as audit:
                    if failure != "missing":
                        self.patch(model)
                    if failure == "changed":
                        worker.handle.current.epoch += 1
                    if failure == "GEMM":
                        model.blocks[0].projection.fail = True
                    model.blocks[0].projection(Tensor(np.ones((1, 4, 3), dtype=np.float32)))
            for module in (model.blocks[0].projection, model.blocks[0].text, model.proj_out):
                self.assertFalse(module.prehooks or module.hooks)
            self.assertFalse(model.patch_embedding.prehooks)
            self.assertEqual(len(model.patch_embedding.hooks), 1)
            self.assertTrue(audit["hooks_removed"])


class PrecisionTests(unittest.TestCase):
    def test_native_precision_flags_restore_after_success_and_exception(self):
        for fail in (False, True):
            worker, _, flags = backend()
            before = workflow.arithmetic_flags(worker.worker.torch)
            try:
                with workflow.attention_variant(worker, "no_reduced_precision") as audit:
                    self.assertFalse(flags.allow_fp16_reduced_precision_reduction)
                    self.assertFalse(flags.allow_fp16_accumulation)
                    self.assertFalse(flags.allow_tf32)
                    self.assertEqual(audit["reference_flags"], before)
                    self.assertTrue(audit["diagnostic_only"])
                    if fail:
                        raise RuntimeError("candidate failed")
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(workflow.arithmetic_flags(worker.worker.torch), before)
            self.assertEqual(audit["restored_flags"], before)

    def test_missing_precision_flag_fails_before_changing_arithmetic(self):
        worker, _, flags = backend()
        del flags.allow_fp16_accumulation
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            with workflow.no_reduced_precision(worker):
                pass
        self.assertTrue(flags.allow_fp16_reduced_precision_reduction)


class CandidateTests(unittest.TestCase):
    def test_except_ffn1_preserves_all_other_full_row_modules_and_sp1(self):
        names = ("ffn.net.0.proj", "ffn.net.2", "attn1.to_q", "attn1.to_k", "attn1.to_v",
                 "attn1.to_out.0", "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0")
        for rank, degree in ((0, 2), (1, 2), (0, 1)):
            worker, model, _ = backend(rank=rank, degree=degree)
            layers = {name: Linear() for name in names}
            model.blocks[0].named_modules = lambda: iter(layers.items())
            with workflow.attention_variant(worker, "full_rows_except_ffn1") as audit:
                model.patch_embedding(Tensor(np.zeros((1, 3, 1, 1, 8), dtype=np.float32)))
                for name, layer in dict(layers, proj_out=model.proj_out).items():
                    rows = 512 if name in ("attn2.to_k", "attn2.to_v") else 8 // degree
                    value = Tensor(np.arange(rows * 3, dtype=np.float32).reshape(1, rows, 3))
                    result = layer(value)
                    should_pad = degree == 2 and name != "ffn.net.0.proj" and rows != 512
                    self.assertEqual(layer.inputs[-1].shape[1], 8 if should_pad else rows)
                    np.testing.assert_array_equal(result.data, value.data @ layer.weight.T + layer.bias)
                    if should_pad:
                        self.assertEqual(result.storage_offset(), 0)
                        np.testing.assert_array_equal(layer.inputs[-1].data[:, rank * rows:(rank + 1) * rows],
                                                      value.data)
                    else:
                        self.assertIs(layer.inputs[-1], value)
                self.assertNotIn("blocks.0.ffn.net.0.proj", audit["modules"])
                self.assertEqual(set(audit["modules"]),
                                 {"blocks.0." + name for name in names if name != "ffn.net.0.proj"} | {"proj_out"})
                self.assertFalse(audit["deployment_qualification"])
            self.assertTrue(audit["hooks_removed"])
            self.assertTrue(all(not layer.prehooks and not layer.hooks for layer in layers.values()))

    def test_selective_modules_are_exact_and_cross_attention_is_excluded(self):
        for variant, expected in (("full_rows_ffn2", {"ffn.net.2"}),
                ("full_rows_ffn2_qkv", {"ffn.net.2", "attn1.to_q", "attn1.to_k", "attn1.to_v"})):
            worker, model, _ = backend(rank=1)
            layers = {name: Linear() for name in
                      ("ffn.net.2", "ffn.net.0.proj", "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn2.to_q")}
            model.blocks[0].named_modules = lambda: iter(layers.items())
            with workflow.attention_variant(worker, variant) as audit:
                model.patch_embedding(Tensor(np.zeros((1, 3, 1, 1, 8), dtype=np.float32)))
                for name, layer in layers.items():
                    layer(Tensor(np.ones((1, 4, 3), dtype=np.float32)))
                    self.assertEqual(layer.inputs[-1].shape[1], 8 if name in expected else 4)
                self.assertEqual(set(audit["modules"]), {"blocks.0." + name for name in expected})
            self.assertTrue(all(not layer.prehooks and not layer.hooks for layer in layers.values()))

    def test_tiles_are_globally_aligned_with_partial_rank_boundary_and_tail(self):
        full = np.arange(6 * 3, dtype=np.float32).reshape(1, 6, 3)
        outputs = []
        for start, end in ((0, 6), (0, 3), (3, 6)):
            worker, model, _ = backend()
            linear = model.proj_out
            output = workflow.tiled_linear(worker.worker.torch, Tensor(full[:, start:end]), linear.forward,
                                            global_start=start, tile=4)
            np.testing.assert_array_equal(output.data, full[:, start:end] @ linear.weight.T + linear.bias)
            self.assertTrue(all(value.shape == (1, 4, 3) for value in linear.inputs))
            if start == 3:
                np.testing.assert_array_equal(linear.inputs[0].data[:, :3], 0)
                np.testing.assert_array_equal(linear.inputs[0].data[:, 3:4], full[:, 3:4])
                np.testing.assert_array_equal(linear.inputs[1].data[:, :2], full[:, 4:6])
                np.testing.assert_array_equal(linear.inputs[1].data[:, 2:], 0)
            outputs.append(output.data)
        np.testing.assert_array_equal(outputs[0], np.concatenate(outputs[1:], axis=1))

    def test_tile_context_keeps_local_hook_geometry_and_restores_forward_on_exception(self):
        for rank, degree in ((0, 1), (0, 2), (1, 2)):
            worker, model, _ = backend(rank=rank, degree=degree)
            seen = []
            linear = model.proj_out
            original = linear.forward
            linear.register_forward_hook(lambda m, args, output: seen.append(tuple(output.shape)))
            # 2050 rows place the SP2 rank boundary at 1025, one row into tile 1.
            with workflow.attention_variant(worker, "global_tile_1024") as audit:
                model.patch_embedding(Tensor(np.zeros((1, 3, 1, 1, 2050), dtype=np.float32)))
                value = Tensor(np.ones((1, 2050 // degree, 3), dtype=np.float32))
                output = linear(value)
                self.assertEqual(seen, [(1, 2050 // degree, 2)])
                self.assertEqual(output.storage_offset(), 0)
                self.assertTrue(all(item.shape[1] == 1024 for item in linear.inputs))
                row = audit["shapes"][0]
                self.assertEqual(row["row_start"], rank * (2050 // degree))
                if rank == 1:
                    self.assertEqual(row["tile_intersections"][0], [1024, 1025, 2048])
                text = Tensor(np.ones((1, 512, 3), dtype=np.float32))
                model.blocks[0].text(text)
                self.assertIs(model.blocks[0].text.inputs[-1], text)
            self.assertEqual(linear.forward, original)
            self.assertNotIn("forward", linear.__dict__)
            self.assertTrue(audit["forwards_restored"])
        worker, model, _ = backend()
        custom_forward = lambda value: (_ for _ in ()).throw(RuntimeError("forced"))
        model.proj_out.forward = custom_forward
        with self.assertRaisesRegex(RuntimeError, "forced"):
            with workflow.global_tile_linear(worker, "global_tile_2048") as audit:
                model.patch_embedding(Tensor(np.zeros((1, 3, 1, 1, 8), dtype=np.float32)))
                model.proj_out(Tensor(np.ones((1, 4, 3), dtype=np.float32)))
        self.assertIs(model.proj_out.forward, custom_forward)
        self.assertTrue(audit["hooks_removed"] and audit["forwards_restored"])


if __name__ == "__main__":
    unittest.main()
