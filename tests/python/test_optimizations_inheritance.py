"""CPU-only inheritance fixtures; device/process APIs are replaced by fakes."""
import contextlib
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.directions.budget import BudgetStop
from ditserve.optimizations import cache_online, online, runner, stage_phases, worker
from ditserve.optimizations.protocol import PHASES
from ditserve.research.common import atomic_json, digest


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.ledger = self.root / "gpu-budget.json"
        self.parent = self.root / "parent.json"
        atomic_json(self.parent, dict(active=None, used_gpu_seconds=0., limit_gpu_seconds=86400.))
        self.config = dict(gpu_uuid="A100-uuid", v100_uuid="V100-uuid", python="/never/run", torch_home=str(self.root))

    def tearDown(self):
        self.temp.cleanup()

    def qualified_encoder(self, device="cuda:1"):
        return dict(status="passed", returncode=0, local_qualified=True, online_qualified=True,
            charge=dict(gpu_seconds=2., gpu_count=2 if device == "cuda:1" else 1),
            final_occupancy=dict(status="exclusive_observed"), encoder_device=device)

    def publish(self, phase, value):
        atomic_json(self.output / phase / "terminal.json", value)

    def predecessors(self, phase):
        for prior in PHASES[:PHASES.index(phase)]:
            self.publish(prior, dict(status="stopped", gpu_work_started=False))
        self.publish("text_encoder", self.qualified_encoder())

    def test_incomplete_supervisor_identity_cannot_select_encoder(self):
        base = self.qualified_encoder()
        variants = [dict(base, status="failed"), dict(base, status="running"), dict(base, returncode=1),
                    dict(base, online_qualified=False), dict(base, local_qualified=False),
                    dict(base, supervisor_failure={"reason": "isolation lost"}), dict(base, charge={}),
                    dict(base, final_occupancy=dict(status="unknown"))]
        for key in ("charge", "returncode", "final_occupancy", "online_qualified", "local_qualified"):
            missing = dict(base)
            missing.pop(key)
            variants.append(missing)
        for value in variants:
            with self.subTest(value=value):
                self.publish("text_encoder", value)
                self.assertNotIn("encode_device", worker.selected(self.output))

    def test_qualified_device_controls_scheduler_vs_auxiliary_encoding(self):
        for device, synchronous in (("cuda:0", True), ("cuda:1", False)):
            self.publish("text_encoder", self.qualified_encoder(device))
            selected = worker.selected(self.output)
            self.assertEqual(selected["encode_device"], device)
            self.assertIs(selected["encode_on_scheduler"], synchronous)

    def test_v100_defaults_require_double_gpu_admission_and_restore_hooks(self):
        old_hook = Mock()
        context = SimpleNamespace(worker=SimpleNamespace(device="cuda:0"),
            metadata=dict(phase="vae", encoder_device="cuda:0"),
            online_defaults=dict(image_batch_size=2), prepare_online_arm=old_hook)
        with self.assertRaisesRegex(ValueError, "two-GPU"):
            stage_phases.install_encoder_defaults(context, dict(encode_device="cuda:1"))
        context.metadata["encoder_device"] = "cuda:1"
        fake = SimpleNamespace(encode=Mock(), move=Mock(), restore_cpu=Mock())
        with patch.object(stage_phases, "FP32Encoder", return_value=fake):
            stage_phases.install_encoder_defaults(context, dict(encode_device="cuda:1"))
        self.assertIs(context.online_defaults["encode_video"], fake.encode)
        self.assertFalse(context.online_defaults["encode_on_scheduler"])
        options = context.online_defaults
        context.prepare_online_arm(options)
        old_hook.assert_called_once_with(options)
        fake.move.assert_called_once_with("cuda:1")
        context.encoder_cleanup()
        context.encoder_cleanup()
        fake.restore_cpu.assert_called_once()
        self.assertIs(context.prepare_online_arm, old_hook)
        self.assertEqual(context.online_defaults, dict(image_batch_size=2))

    def test_worker_rejects_v100_visibility_with_single_gpu_charge(self):
        args = SimpleNamespace(ledger=self.ledger, output=self.output, phase="vae", session_id="test-session",
                               auxiliary_encoder=True)
        active = dict(session_id=args.session_id, supervisor_pid=101, stage="vae", child_pgid=202, gpu_count=1,
            identity=dict(phase="vae", output=str(self.output), config_sha256=digest(self.config),
                          gpu_uuids=[self.config["gpu_uuid"], self.config["v100_uuid"]]))
        atomic_json(self.ledger, dict(active=active))
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="A100-uuid,V100-uuid"), \
             patch.object(worker.os, "getppid", return_value=101), \
             patch.object(worker.os, "getpid", return_value=202), \
             patch.object(worker.os, "getpgrp", return_value=202):
            with self.assertRaisesRegex(BudgetStop, "charged supervisor"):
                worker.verify_admission(args, self.config)
            active["gpu_count"] = 2
            atomic_json(self.ledger, dict(active=active))
            self.assertEqual(worker.verify_admission(args, self.config)["gpu_count"], 2)

    def test_inherited_v100_supervisor_counts_both_cards_and_checks_both(self):
        self.predecessors("vae")
        process = SimpleNamespace(pid=123456, wait=Mock(return_value=0))
        admissions = []
        def popen(command, **kwargs):
            active = json.loads(self.ledger.read_text())["active"]
            admissions.append(active)
            self.assertEqual(active["gpu_count"], 2)
            self.assertEqual(active["identity"]["gpu_uuids"], ["A100-uuid", "V100-uuid"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "A100-uuid,V100-uuid")
            self.assertIn("--auxiliary-encoder", command)
            destination = Path(command[command.index("--output") + 1])
            atomic_json(destination / "worker.json", dict(status="stopped", local_qualified=False, online_qualified=False))
            return process
        observation = dict(status="exclusive_observed", owned=[], foreign=[])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "PARENT", self.parent))
            stack.enter_context(patch.object(runner, "_trigger", return_value=None))
            stack.enter_context(patch.object(runner, "inspect", return_value=dict(status="ready",
                devices=dict(devices=[dict(uuid="A100-uuid"), dict(uuid="V100-uuid")]))))
            occupancy = stack.enter_context(patch.object(runner, "gpu_occupancy", return_value=observation))
            stack.enter_context(patch.object(runner, "freeze", return_value=dict(execution_root=str(self.root),
                source_sha256="fixed", source_files={})))
            stack.enter_context(patch.object(runner.subprocess, "Popen", side_effect=popen))
            stack.enter_context(patch.object(runner, "stop_owned_group", return_value=True))
            result = runner.execute(self.config, self.output, self.ledger, "vae")
        self.assertEqual(len(admissions), 1)
        self.assertEqual(result["charge"]["gpu_count"], 2)
        self.assertAlmostEqual(result["charge"]["gpu_seconds"], result["charge"]["elapsed_seconds"] * 2)
        self.assertTrue(all(call.args[0] == ["A100-uuid", "V100-uuid"] for call in occupancy.call_args_list))

    def test_completed_v100_inheriting_phase_is_idempotent(self):
        self.predecessors("vae")
        previous = dict(status="passed", local_qualified=False, online_qualified=False, reason="already settled")
        self.publish("vae", previous)
        with patch.object(runner, "inspect") as inspect, patch.object(runner, "_trigger") as trigger:
            self.assertEqual(runner.execute(self.config, self.output, self.ledger, "vae"), previous)
        inspect.assert_not_called()
        trigger.assert_not_called()

    def test_missing_v100_blocks_before_any_child_or_charge(self):
        self.predecessors("vae")
        with patch.object(runner, "_trigger", return_value=None), \
             patch.object(runner, "inspect", return_value=dict(status="ready", devices=dict(devices=[dict(uuid="A100-uuid")]))), \
             patch.object(runner, "gpu_occupancy", return_value=dict(status="exclusive_observed")), \
             patch.object(runner.subprocess, "Popen") as popen:
            result = runner.execute(self.config, self.output, self.ledger, "vae")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(self.ledger.exists())
        popen.assert_not_called()


class OnlineInheritanceTests(unittest.TestCase):
    def test_cache_spec_wraps_inherited_encoder_without_replacing_it(self):
        traces = [dict(seed=102, load=.5, pattern="burst", trace_sha256="fixture",
                       requests=[dict(kind="video", arrival_ms=0., service_ms=10.)])]
        calls, placements = [], []
        def inherited(spec):
            calls.append(dict(spec))
            return object()
        worker_fake = SimpleNamespace(encode_prompt=Mock(side_effect=AssertionError("CPU fallback forbidden")),
                                      prepare_encoded=lambda spec, payload: dict(spec))
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(config=dict(old_online="fixture"), run_root=Path(directory), output=Path(directory),
                worker=worker_fake, opts=dict(prompts=["a", "b"]), check_budget=SimpleNamespace(remaining=lambda: 100000.),
                sana=SimpleNamespace(set_backend=Mock(), generate_batch=Mock(return_value=(None, dict(elapsed_ms=1.))),
                                     steady_context=contextlib.nullcontext),
                online_defaults=dict(encode_video=inherited, encode_device="cuda:1", encode_on_scheduler=False),
                prepare_online_arm=lambda options: placements.append(options["encode_device"]))
            options_seen = []
            def serve(context, requests, options):
                options_seen.append(dict(options))
                spec = dict(request_id="r000", cache=False, threshold=0., max_continuous=1)
                payload = options["encode_video"](spec)
                if "prepare_video" in options:
                    prepared = options["prepare_video"](spec, payload)
                    self.assertTrue(prepared["cache"])
                    self.assertEqual(prepared["threshold"], .08)
                return dict(status="completed", complete=True, trace_sha256="fixture", requests=[])
            with patch.object(cache_online, "load_frozen_traces", return_value=traces), \
                 patch.object(worker, "selected", return_value=dict(sana_backend="eager", image_batch_size=1, wan_backend="compiled")), \
                 patch.object(cache_online, "WanBlockExecutor", return_value=object()), \
                 patch.object(cache_online, "StepPairedExecutor", return_value=object()), \
                 patch.object(cache_online, "run_trace", side_effect=serve), \
                 patch.object(cache_online, "paired_gate", return_value=dict(passed=True)):
                result = cache_online.run_cache_online(context, dict(cache_quality_qualified=True, full_qualification_completed=True))
            self.assertTrue(result["online_qualified"])
            self.assertEqual(len(calls), 8)
            self.assertEqual(sum(spec["cache"] for spec in calls), 4)
            self.assertEqual(placements, ["cuda:1"] * 8)
            self.assertTrue(all(options["encode_on_scheduler"] is False for options in options_seen))
            self.assertTrue(all(options["wan_backend"] == "eager" for options in options_seen))
            worker_fake.encode_prompt.assert_not_called()

    def test_compile_warmup_uses_inherited_gpu_encoder_after_placement(self):
        # A fresh later-phase compiled executor must not call the CPU encoder
        # after its inherited text model has been moved onto the GPU.
        from ditserve.optimizations import wan
        on_gpu, encoded = [False], []
        def place(options):
            on_gpu[0] = True
        def cpu_prepare(spec):
            if on_gpu[0]:
                raise AssertionError("CPU encoder called while encoder model is on GPU")
            return state(spec)
        def state(spec):
            return SimpleNamespace(spec=spec, step_index=0, residency="gpu")
        def encode(spec):
            self.assertTrue(on_gpu[0])
            encoded.append(spec)
            return object()
        backend = SimpleNamespace(freeze_compilation=Mock())
        class Executor:
            def __init__(self, *unused):
                self.backend = backend
            def attach(self, base):
                return SimpleNamespace(base=base, closed=False)
            def advance(self, continuation, count, check_budget):
                continuation.base.step_index += 1
            def close(self, continuation):
                continuation.closed = True
        with tempfile.TemporaryDirectory() as directory:
            context = SimpleNamespace(config=dict(old_online="fixture"), output=Path(directory), run_root=Path(directory),
                worker=SimpleNamespace(prepare=cpu_prepare, encode_prompt=Mock(side_effect=AssertionError("CPU fallback")),
                    prepare_encoded=lambda spec, payload: state(spec), release=lambda state: setattr(state, "residency", "released")),
                executor=object(), opts=dict(prompts=["a", "b"], seeds=[101], steps=32),
                profile=dict(selected_shape=dict(height=480, width=832, frames=81)), metadata=dict(wan_warmed=True),
                sana=SimpleNamespace(set_backend=Mock(), generate_batch=Mock(), steady_context=contextlib.nullcontext),
                check_budget=SimpleNamespace(remaining=lambda: 100000., ensure_unit=Mock()), prepare_online_arm=place,
                online_defaults=dict(encode_video=encode, encode_device="cuda:0", encode_on_scheduler=True))
            traces = [dict(seed=102, load=.5, pattern="burst", trace_sha256="fixture",
                           requests=[dict(kind="video", arrival_ms=0., service_ms=10.)])]
            with patch.object(online, "load_frozen_traces", return_value=traces), \
                 patch.object(worker, "selected", return_value=dict(sana_backend="eager", image_batch_size=1, wan_backend="inductor-block1")), \
                 patch.object(wan, "WanCompiledExecutor", Executor), patch.object(wan, "BlockBackend", return_value=backend), \
                 patch.object(online, "run_trace", return_value=dict(status="completed", complete=True)), \
                 patch.object(online, "paired_gate", return_value=dict(passed=True)):
                worker._online(context, {})
            self.assertEqual(len(encoded), 1)
            backend.freeze_compilation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
