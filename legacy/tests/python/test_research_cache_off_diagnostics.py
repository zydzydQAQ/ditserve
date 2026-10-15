"""CPU ownership/codec/native-step contracts for cache-off numerical probes."""

import contextlib
import copy
from types import SimpleNamespace
import unittest

import torch

from ditserve.research.cache_off_diagnostics import CacheOffDiagnostics, compare_step_tensors, state_signature
from ditserve.research.common import digest
from ditserve.research.dual_state import pack_request
from ditserve.research.dual_runtime import ResidentWorker
from ditserve.research.parallel import ParallelContractError
from test_research_dual_state import fixture, empty_target


class Transformer:
    dtype = torch.float16

    def __init__(self):
        self.forward = self.native_forward
        self.calls = 0
        self.fail_at = None
        self.random = False

    def native_forward(self, *, hidden_states, timestep, encoder_hidden_states, **_):
        self.calls += 1
        if self.random:
            torch.rand(1)
        if self.fail_at == self.calls:
            raise RuntimeError("deliberate forward failure")
        return (hidden_states * .1 + encoder_hidden_states.mean() * .01 + .02,)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def cache_context(self, _):
        return contextlib.nullcontext()


def with_model(worker):
    model = Transformer()
    worker.pipe = SimpleNamespace(transformer=model)
    worker._original_forward = worker._cached_forward = model.forward
    worker.p = SimpleNamespace(_sdpa_math_context=lambda _: contextlib.nullcontext())
    worker.attention = "math"
    return worker


class CacheOffDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_pair(self):
        worker, state = fixture(cached=False)
        self.source, self.state = with_model(worker), state
        self.origin = CacheOffDiagnostics(self.source, "GPU-source", enabled=True)
        self.target = with_model(empty_target())
        self.probe = CacheOffDiagnostics(self.target, "GPU-target", enabled=True)
        self.export = self.origin.export(state.request_id, "boundary-16")
        return self.export

    def import_copy(self, name="probe"):
        return self.probe.import_copy(self.export["snapshot"], self.export["tensors"], name)

    def test_explicit_mode_required_and_ordinary_worker_cannot_enable_implicitly(self):
        worker, _ = fixture(cached=False)
        with self.assertRaisesRegex(ValueError, "explicit"):
            CacheOffDiagnostics(with_model(worker), "GPU-source")
        with self.assertRaisesRegex(ValueError, "explicit research"):
            ResidentWorker("GPU-source", "math", "unused", "unused", 100., diagnostic_mode=True)

    def test_export_pauses_source_and_codec_does_not_create_two_production_owners(self):
        self.setup_pair()
        initial = self.export["snapshot"]["state_signature"]
        self.assertEqual(self.state.residency, "diagnostic_paused")
        with self.assertRaisesRegex(ValueError, "must reside"):
            self.source.step(self.state)
        with self.assertRaisesRegex(ValueError, "must reside"):
            pack_request(self.source, self.state, action="full_migrate", target_device="GPU-target", migration_id="bad")
        imported = self.import_copy()
        self.assertEqual(imported["state_signature"], initial)
        self.assertEqual(imported["target_device"], "GPU-target")
        self.assertEqual(imported["backend"], "math")
        self.assertEqual(imported["snapshot_sha256"], self.export["snapshot"]["sha256"])
        self.assertFalse(self.target._states)
        clone = self.probe.copies["probe"]["state"]
        self.assertNotEqual(clone.request_id, self.state.request_id)
        with self.assertRaisesRegex(ValueError, "not owned"):
            self.target.step(clone)
        self.assertEqual(self.origin.signature(self.state.request_id), initial)
        status = self.origin.status(self.state.request_id)
        self.assertEqual(status["residency"], "diagnostic_paused")
        self.assertEqual(status["signature"], initial)
        self.assertNotIn("residency", status["signature"])
        self.source.release(self.state)
        self.assertEqual(self.state.residency, "released")

    def test_import_preserves_unipc_aliases_rng_and_independent_storage(self):
        self.setup_pair()
        self.import_copy()
        clone = self.probe.copies["probe"]["state"]
        self.assertIs(clone.scheduler.disable_corrector, clone.scheduler.config.disable_corrector)
        self.assertEqual(state_signature(self.target, clone)["container_aliases"],
                         self.export["snapshot"]["state_signature"]["container_aliases"])
        self.assertTrue(all(state_signature(self.target, clone)["container_aliases"].values()))
        for view in clone.scheduler.timestep_list:
            self.assertEqual(view.untyped_storage().data_ptr(), clone.scheduler.timesteps.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(clone.generator.get_state(), self.state.generator.get_state()))
        self.assertNotEqual(clone.latents.untyped_storage().data_ptr(), self.export["tensors"]["latents"].untyped_storage().data_ptr())
        with self.assertRaisesRegex(ValueError, "already used"):
            self.import_copy()
        self.probe.drop("probe")
        with self.assertRaisesRegex(ValueError, "already used"):
            self.import_copy()

    def test_one_native_step_matches_continuous_complete_state_and_predictions(self):
        self.setup_pair()
        self.import_copy()
        regular, reference = fixture(cached=False)
        regular = with_model(regular)
        before = state_signature(regular, reference)
        ambient = torch.get_rng_state().clone()
        regular.step(reference)
        output = self.probe.step("probe")
        self.assertEqual(output["before"], before)
        self.assertEqual(output["after"], state_signature(regular, reference))
        self.assertEqual(output["after"]["step_index"], 17)
        self.assertEqual(output["target_device"], "GPU-target")
        self.assertEqual(output["backend"], "math")
        self.assertEqual(output["snapshot_sha256"], self.export["snapshot"]["sha256"])
        self.assertEqual(self.target.pipe.transformer.calls, 2)
        self.assertEqual(output["native_step"]["completed_steps"], 17)
        self.assertTrue(torch.equal(torch.get_rng_state(), ambient))
        self.assertFalse(self.target._states)
        self.assertFalse(output["performance_eligible"])
        self.assertEqual(self.state.step_index, 16)
        self.assertEqual(self.state.residency, "diagnostic_paused")
        source_status = self.origin.status(self.state.request_id)
        self.assertEqual(source_status["residency"], "diagnostic_paused")
        self.assertEqual(source_status["signature"], self.export["snapshot"]["state_signature"])
        tensors = output["tensors"]
        self.assertTrue(torch.equal(tensors["cfg"], tensors["uncond"] + 5 * (tensors["cond"] - tensors["uncond"])))
        self.assertTrue(all(t.device.type == "cpu" for t in tensors.values()))
        self.assertTrue(all(v["relative_l2"] == 0 and v["max_abs"] == 0
                            for v in compare_step_tensors(tensors, tensors).values()))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.probe.step("probe")

    def test_signature_ignores_request_names_timing_but_detects_sampler_rng_and_tensor_changes(self):
        worker, state = fixture(cached=False)
        before = state_signature(worker, state)
        state.request_id = state.spec["request_id"] = state.immutable_spec["request_id"] = "other-name"
        state.events.append(dict(ms=12345.))
        state.prepare_ms = 9999.
        self.assertEqual(state_signature(worker, state), before)
        state.scheduler.lower_order_nums += 1
        self.assertNotEqual(state_signature(worker, state)["state_sha256"], before["state_sha256"])
        state.scheduler.lower_order_nums -= 1
        state.generator.manual_seed(500)
        self.assertNotEqual(state_signature(worker, state)["state_sha256"], before["state_sha256"])

    def test_non_off_and_non_step16_exports_are_rejected(self):
        for cached, boundary in ((True, 16), (False, 15), (False, 17)):
            worker, state = fixture(cached=cached, boundary=boundary)
            diagnostic = CacheOffDiagnostics(with_model(worker), "GPU-source", enabled=True)
            with self.assertRaisesRegex(ValueError, "cache-off|step 16"):
                diagnostic.export(state.request_id, "invalid")
            self.assertEqual(state.residency, "gpu")

    def test_equal_corrector_contents_do_not_hide_broken_unipc_container_aliases(self):
        worker, state = fixture(cached=False)
        before = state_signature(worker, state)
        self.assertTrue(all(before["container_aliases"].values()))
        original = state.scheduler.disable_corrector
        state.scheduler.disable_corrector = list(original)
        self.assertEqual(state.scheduler.disable_corrector, original)
        broken = state_signature(worker, state)
        self.assertFalse(broken["container_aliases"]["scheduler.disable_corrector_is_config_item"])
        self.assertNotEqual(broken["state_sha256"], before["state_sha256"])
        self.assertNotEqual(broken["scheduler_sha256"], before["scheduler_sha256"])
        self.assertNotEqual(broken["sampling_metadata_sha256"], before["sampling_metadata_sha256"])
        self.assertEqual(broken["tensors"], before["tensors"])
        state.scheduler.disable_corrector = original
        self.assertEqual(state_signature(worker, state), before)
        object.__setattr__(state.scheduler._internal_dict, "disable_corrector", list(original))
        frozen_dict_broken = state_signature(worker, state)
        self.assertFalse(frozen_dict_broken["container_aliases"]["scheduler.config_attribute_is_config_item"])
        self.assertNotEqual(frozen_dict_broken["state_sha256"], before["state_sha256"])

    def test_tampered_wire_snapshot_and_imported_input_are_rejected(self):
        self.setup_pair()
        bad = copy.deepcopy(self.export["snapshot"])
        bad["source_will_remain_paused"] = False
        bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "snapshot binding"):
            self.probe.import_copy(bad, self.export["tensors"], "bad")
        changed = {k: v.clone() for k, v in self.export["tensors"].items()}
        changed["latents"].flatten()[0] += 1
        with self.assertRaisesRegex(ParallelContractError, "checksum"):
            self.probe.import_copy(self.export["snapshot"], changed, "bad-tensor")
        self.assertFalse(self.probe.copies)
        self.import_copy()
        self.probe.copies["probe"]["state"].latents.flatten()[0] += 1
        with self.assertRaisesRegex(ValueError, "changed after import"):
            self.probe.step("probe")
        self.assertFalse(self.target._states)

    def test_exception_rolls_back_sampler_rng_and_registry(self):
        self.setup_pair()
        self.import_copy()
        before = self.probe.copies["probe"]["initial_signature"]
        original = self.target._original_forward
        self.target.pipe.transformer.fail_at = 2
        ambient = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            self.probe.step("probe")
        clone = self.probe.copies["probe"]["state"]
        self.assertEqual(state_signature(self.target, clone), before)
        self.assertIs(self.target._original_forward, original)
        self.assertFalse(self.target._states)
        self.assertTrue(torch.equal(torch.get_rng_state(), ambient))
        self.assertEqual(self.probe.copies["probe"]["status"], "failed")

    def test_random_operator_rejected_with_no_global_rng_side_effect(self):
        self.setup_pair()
        self.import_copy()
        self.target.pipe.transformer.random = True
        ambient = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "random operators"):
            self.probe.step("probe")
        self.assertTrue(torch.equal(torch.get_rng_state(), ambient))
        self.assertFalse(self.target._states)

    def test_backend_drift_rejected_and_optional_tensor_return_is_explicit(self):
        self.setup_pair()
        self.import_copy()
        self.target.attention = "efficient"
        with self.assertRaisesRegex(ValueError, "backend changed"):
            self.probe.step("probe")
        self.target.attention = "math"
        output = self.probe.step("probe", return_tensors=False)
        self.assertNotIn("tensors", output)
        self.assertEqual(set(output["tensor_summaries"]), {"cond", "uncond", "cfg", "latent_before", "latent_after"})

    def test_identical_backends_preserve_distinct_physical_target_identity(self):
        self.setup_pair()
        other = CacheOffDiagnostics(with_model(empty_target()), "GPU-other", enabled=True)
        first = self.import_copy("first")
        second = other.import_copy(self.export["snapshot"], self.export["tensors"], "second")
        self.assertEqual(first["backend"], second["backend"])
        self.assertEqual(first["snapshot_sha256"], second["snapshot_sha256"])
        self.assertNotEqual(first["target_device"], second["target_device"])
        a, b = self.probe.step("first"), other.step("second")
        self.assertEqual(a["after"], b["after"])
        self.assertEqual((a["target_device"], b["target_device"]), ("GPU-target", "GPU-other"))


if __name__ == "__main__":
    unittest.main()
