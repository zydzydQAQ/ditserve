"""Supervisor/selection audit with fake processes; never contacts a GPU."""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ditserve.optimizations import runner, worker
from ditserve.optimizations.protocol import PHASES
from ditserve.research.common import atomic_json


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.ledger = self.root / "budget.json"
        self.parent = self.root / "parent.json"
        atomic_json(self.parent, dict(active=None, used_gpu_seconds=10., limit_gpu_seconds=86400.))
        self.config = dict(gpu_uuid="test-a100", python="/never/execute/python", torch_home=str(self.root / "torch"))

    def tearDown(self):
        self.temp.cleanup()

    def valid_terminal(self, **fields):
        return dict(dict(status="passed", returncode=0, charge={"gpu_seconds": 1.},
                         final_occupancy=dict(status="exclusive_observed"),
                         local_qualified=True, online_qualified=True), **fields)

    def publish(self, phase, value):
        atomic_json(self.output / phase / "terminal.json", value)

    def execute(self, *, returncode=0, result=None, lost_isolation=False, gone=True, retry_from=None,
                phase="sana_compile", online_replay_from=None):
        process = SimpleNamespace(pid=123456, wait=Mock(return_value=returncode))
        def popen(command, **kwargs):
            self.last_command = list(command)
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "test-a100")
            destination = Path(command[command.index("--output") + 1])
            atomic_json(destination / "worker.json", result or dict(status="passed", decision="值得继续",
                        local_qualified=True, online_qualified=True))
            return process
        exclusive = dict(status="exclusive_observed", owned=[], foreign=[])
        contended = dict(status="contended", owned=[], foreign=[dict(pid=777)])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, "PARENT", self.parent))
            stack.enter_context(patch.object(runner, "inspect", return_value=dict(status="ready")))
            stack.enter_context(patch.object(runner, "freeze", return_value=dict(
                execution_root=str(self.root), source_sha256="frozen", source_files={})))
            stack.enter_context(patch.object(runner.subprocess, "Popen", side_effect=popen))
            stack.enter_context(patch.object(runner, "gpu_occupancy", side_effect=[
                contended if lost_isolation else exclusive, exclusive]))
            cleanup = stack.enter_context(patch.object(runner, "stop_owned_group", return_value=gone))
            value = runner.execute(self.config, self.output, self.ledger, phase, retry_from=retry_from,
                                   online_replay_from=online_replay_from)
            cleanup.assert_called_once_with(process)
        return value

    def test_normal_execution_charges_and_starts_own_session(self):
        parent = self.parent.read_bytes()
        result = self.execute()
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["online_qualified"])
        budget = json.loads(self.ledger.read_text())
        self.assertIsNone(budget["active"])
        self.assertEqual(len(budget["sessions"]), 1)
        self.assertGreaterEqual(budget["used_gpu_seconds"], 0)
        self.assertEqual(self.parent.read_bytes(), parent)

    def test_foreign_occupancy_invalidates_without_targeting_foreign_process(self):
        result = self.execute(lost_isolation=True)
        self.assertFalse(result["online_qualified"])
        self.assertFalse(result["local_qualified"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["charge"]["status"], "failed")
        self.assertEqual(result["decision"], "未验证")

    def test_unconfirmed_cleanup_leaves_active_charge(self):
        with self.assertRaisesRegex(RuntimeError, "cleanup unconfirmed"):
            self.execute(gone=False)
        budget = json.loads(self.ledger.read_text())
        self.assertIsNotNone(budget["active"])
        self.assertEqual(budget["active"]["child_pgid"], 123456)

    def test_worker_success_file_cannot_outlive_nonzero_exit(self):
        result = self.execute(returncode=1)
        self.assertFalse(result["online_qualified"])
        self.assertFalse(result["local_qualified"])
        self.assertIn(result["status"], ("failed", "incomplete"))
        self.assertEqual(result["decision"], "未验证")

    def test_same_configuration_retry_preserves_unused_phase_allowance(self):
        first = self.execute(returncode=1)
        failure = Path(first["output"]) / "supervisor.json"
        original = failure.read_bytes()
        second = self.execute(retry_from=failure)
        self.assertEqual(second["charge"]["stage"], "sana_compile")
        self.assertEqual(failure.read_bytes(), original)
        self.assertEqual(len(json.loads(self.ledger.read_text())["sessions"]), 2)

    def test_reserve_only_backs_a_recorded_retry_when_phase_allowance_is_exhausted(self):
        first = self.execute(returncode=1)
        failure = Path(first["output"]) / "supervisor.json"
        with patch.object(runner.Allocation, "remaining", side_effect=lambda stage: 3600. if stage == "reserve" else 0.):
            second = self.execute(retry_from=failure)
        self.assertEqual(second["charge"]["stage"], "reserve")

    def test_prior_phase_is_required_before_admission(self):
        with patch.object(runner, "inspect") as inspect:
            with self.assertRaisesRegex(ValueError, "earlier"):
                runner.execute(self.config, self.output, self.ledger, "sana_batch")
        inspect.assert_not_called()

    def test_terminal_phase_is_not_silently_reexecuted(self):
        expected = self.valid_terminal()
        self.publish("sana_compile", expected)
        with patch.object(runner, "inspect") as inspect:
            self.assertEqual(runner.execute(self.config, self.output, self.ledger, "sana_compile"), expected)
        inspect.assert_not_called()

    def blocked_encoder_fallback(self, *, visible_v100=True, foreign=None):
        self.config["v100_uuid"]="test-v100"
        for phase in PHASES[:PHASES.index("text_encoder")]:
            self.publish(phase,self.valid_terminal())
        self.publish("text_encoder",dict(status="failed",needs_v100=True))
        terminal_path=self.output/"text_encoder/terminal.json"
        previous=terminal_path.read_bytes()
        devices=[dict(uuid="test-a100",name="NVIDIA A100-PCIE-40GB")]
        if visible_v100:
            devices.append(dict(uuid="test-v100",name="Tesla V100-PCIE-32GB"))
        original=dict(status="ready",gpu_idle=True,missing_models=[],
            devices=dict(status="available",devices=devices),
            occupancy=dict(status="exclusive_observed",foreign=[]))
        occupancy=dict(status="contended" if foreign else "exclusive_observed",foreign=foreign or [])
        with patch.object(runner,"_trigger",return_value=None), \
             patch.object(runner,"inspect",return_value=original), \
             patch.object(runner,"gpu_occupancy",return_value=occupancy), \
             patch.object(runner,"Allocation") as allocation, \
             patch.object(runner,"freeze") as freeze, \
             patch.object(runner.subprocess,"Popen") as popen:
            result=runner.execute(self.config,self.output,self.ledger,"text_encoder",auxiliary_encoder=True)
        allocation.assert_not_called()
        freeze.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(terminal_path.read_bytes(),previous)
        self.assertFalse(self.ledger.exists())
        self.assertFalse(result["gpu_work_started"])
        self.assertEqual(result["status"],"blocked")
        return result

    def test_fallback_block_identifies_busy_v100_while_a100_is_idle(self):
        result=self.blocked_encoder_fallback(foreign=[dict(uuid="test-v100",pid=3791985)])
        self.assertIn("V100",result["reason"])
        self.assertIn("busy",result["reason"])
        self.assertIn("3791985",result["reason"])
        self.assertNotIn("A100",result["reason"])
        evidence=result["admission_evidence"]
        self.assertEqual(evidence["occupancy_source"],"both_devices_occupancy")
        self.assertEqual(evidence["required_devices"][0]["foreign_pids"],[])
        self.assertEqual(evidence["required_devices"][1]["foreign_pids"],[3791985])

    def test_fallback_block_identifies_invisible_v100_without_calling_it_busy(self):
        result=self.blocked_encoder_fallback(visible_v100=False)
        self.assertIn("V100 (test-v100) unavailable: device not visible",result["reason"])
        self.assertNotIn("busy",result["reason"])
        self.assertNotIn("A100",result["reason"])
        self.assertFalse(result["admission_evidence"]["required_devices"][1]["visible"])

    def test_model_asset_block_does_not_invent_gpu_contention(self):
        observation=dict(status="blocked",missing_models=["wan"],
            devices=dict(status="available",devices=[dict(uuid="test-a100",name="A100")]),
            occupancy=dict(status="exclusive_observed",foreign=[]))
        with patch.object(runner,"inspect",return_value=observation), \
             patch.object(runner,"Allocation") as allocation:
            result=runner.execute(self.config,self.output,self.ledger,"sana_compile")
        allocation.assert_not_called()
        self.assertEqual(result["reason"],"model assets incomplete: wan")
        self.assertFalse(result["gpu_work_started"])
        self.assertEqual(result["admission_evidence"]["occupancy_source"],"occupancy")

    def baseline_evidence(self, *, terminal_status="passed", complete=True, online_status="completed", trigger=True):
        path = self.output / "sana_compile" / "attempt00" / "online.json"
        row = dict(arm="baseline", status="completed" if complete else "interrupted", complete=complete,
                   requests=[dict(status="completed" if complete else "interrupted")],
                   attribution=dict(encoding_trigger=trigger, vae_trigger=trigger))
        atomic_json(path, dict(status=online_status, rows=[row]))
        self.publish("sana_compile", self.valid_terminal(status=terminal_status, online_path=str(path),
                                                        online_qualified=False))

    def test_complete_baseline_only_can_trigger_without_candidate_qualification(self):
        self.baseline_evidence()
        value = json.loads((self.output / "sana_compile/terminal.json").read_text())
        value["online_qualified"] = False
        self.publish("sana_compile", value)
        self.assertIsNone(runner._trigger(self.output, "text_encoder"))

    def test_failed_supervisor_cannot_supply_bottleneck_evidence(self):
        self.baseline_evidence(terminal_status="failed")
        value = runner._trigger(self.output, "text_encoder")
        self.assertIsNotNone(value)
        self.assertEqual(value["decision"], "未验证")

    def test_partial_row_cannot_supply_positive_or_negative_bottleneck_evidence(self):
        for trigger in (True, False):
            self.baseline_evidence(complete=False, trigger=trigger)
            value = runner._trigger(self.output, "text_encoder")
            self.assertIsNotNone(value)
            self.assertEqual(value["decision"], "未验证")

    def test_incomplete_online_container_is_not_completed_baseline_evidence(self):
        self.baseline_evidence(online_status="incomplete")
        value = runner._trigger(self.output, "vae")
        self.assertIsNotNone(value)
        self.assertEqual(value["decision"], "未验证")

    def test_completed_negative_baseline_retains_conditional_scope(self):
        self.baseline_evidence(trigger=False)
        value = runner._trigger(self.output, "vae")
        self.assertEqual(value["decision"], "简单策略已足够")
        self.assertFalse(value["online_qualified"])
        self.assertIn("not proof", value["scope"])

    def test_newly_qualified_candidate_replaces_old_bottleneck(self):
        self.baseline_evidence(trigger=True)
        path = self.output / "wan_compile/attempt00/online.json"
        row = dict(status="completed", complete=True, requests=[dict(status="completed")])
        atomic_json(path, dict(status="completed", rows=[
            dict(row, arm="baseline", attribution=dict(encoding_trigger=True)),
            dict(row, arm="candidate", attribution=dict(encoding_trigger=False))]))
        self.publish("wan_compile", self.valid_terminal(online_path=str(path)))
        value = runner._trigger(self.output, "text_encoder")
        self.assertEqual(value["decision"], "简单策略已足够")

    def test_online_audit_invalidates_positive_and_negative_trigger_evidence(self):
        for trigger in (True, False):
            with self.subTest(trigger=trigger):
                self.baseline_evidence(trigger=trigger)
                audit = self.output / "sana_compile/attempt00/online-measurement-audit.json"
                atomic_json(audit, dict(invalidates_online_qualification=True))
                for phase in ("text_encoder", "vae"):
                    value = runner._trigger(self.output, phase)
                    self.assertEqual(value["decision"], "未验证")
                    self.assertFalse(value["gpu_work_started"])

    def test_invalid_latest_online_audit_falls_back_to_older_valid_baseline(self):
        self.baseline_evidence(trigger=False)
        attempt = self.output / "wan_compile/attempt00"
        path = attempt / "online.json"
        row = dict(status="completed", complete=True, requests=[dict(status="completed")],
                   attribution=dict(encoding_trigger=True, vae_trigger=True))
        atomic_json(path, dict(status="completed", rows=[dict(row, arm=arm) for arm in ("baseline", "candidate")]))
        self.publish("wan_compile", self.valid_terminal(output=str(attempt), online_path=str(path)))
        audit = attempt / "online-measurement-audit.json"
        for note in ('{"invalidates_online_qualification":true}', '{broken', '[]'):
            with self.subTest(audit=note):
                audit.write_text(note)
                value = runner._trigger(self.output, "text_encoder")
                self.assertEqual(value["decision"], "简单策略已足够")
                self.assertFalse(value["trigger_evidence"][0]["encoding_trigger"])

    def test_valid_online_audit_does_not_exclude_unqualified_candidate_baseline(self):
        self.baseline_evidence(trigger=True)
        atomic_json(self.output / "sana_compile/attempt00/online-measurement-audit.json",
                    dict(invalidates_online_qualification=False, reason="measurement valid"))
        self.assertIsNone(runner._trigger(self.output, "text_encoder"))

    def test_failed_terminal_or_partial_cannot_promote_backend(self):
        self.publish("sana_compile", self.valid_terminal(status="failed", selected_backend="compiled"))
        atomic_json(self.output / "sana_batch" / "partial.json", self.valid_terminal())
        self.publish("wan_compile", self.valid_terminal(status="incomplete", selected_backend="inductor-block1"))
        self.assertEqual(worker.selected(self.output), dict(sana_backend="eager", image_batch_size=1, wan_backend="eager"))

    def test_valid_supervisor_online_qualification_can_promote(self):
        self.publish("sana_compile", self.valid_terminal(selected_backend="compiled"))
        self.publish("sana_batch", self.valid_terminal())
        self.publish("wan_compile", self.valid_terminal(selected_backend="inductor-block1"))
        self.assertEqual(worker.selected(self.output), dict(sana_backend="compiled", image_batch_size=2,
                                                           wan_backend="inductor-block1"))

    def test_online_measurement_audit_prevents_default_promotion(self):
        attempt = self.output / "wan_compile/attempt02"
        self.publish("wan_compile", self.valid_terminal(output=str(attempt), selected_backend="inductor-block1"))
        atomic_json(attempt / "online-measurement-audit.json",
                    dict(invalidates_online_qualification=True, reason="observer history differs"))
        self.assertEqual(worker.selected(self.output)["wan_backend"], "eager")
        # An unreadable audit is not permission to restore a qualified default.
        (attempt / "online-measurement-audit.json").write_text("{broken")
        self.assertEqual(worker.selected(self.output)["wan_backend"], "eager")

    def test_online_replay_is_not_an_arbitrary_phase_or_retry_escape(self):
        for phase, retry, auxiliary in (("sana_compile", None, False),
                                         ("wan_compile", "failure.json", False),
                                         ("wan_compile", None, True)):
            with self.subTest(phase=phase, retry=retry, auxiliary=auxiliary), \
                 patch.object(runner, "inspect") as inspect, \
                 self.assertRaisesRegex(ValueError, "explicit Wan measurement repair"):
                runner.execute(self.config, self.output, self.ledger, phase,
                               online_replay_from="prior.json", retry_from=retry,
                               auxiliary_encoder=auxiliary)
            inspect.assert_not_called()

    def test_verified_online_replay_gets_new_charged_attempt_and_worker_binding(self):
        from ditserve.optimizations import replay
        for phase in ("sana_compile", "sana_batch"):
            self.publish(phase, self.valid_terminal())
        prior = self.output / "wan_compile/attempt00/supervisor.json"
        original = self.valid_terminal(output=str(prior.parent))
        atomic_json(prior, original)
        self.publish("wan_compile", original)
        binding = dict(path=str(prior), sha256="verified-fixture")
        with patch.object(replay, "inspect_local_replay", return_value=binding) as inspect:
            result = self.execute(phase="wan_compile", online_replay_from=prior)
        inspect.assert_called_once_with(prior, self.config, self.output.resolve())
        self.assertEqual(Path(result["output"]).name, "attempt01")
        self.assertEqual(result["online_replay_from"], binding)
        self.assertEqual(result["charge"]["identity"]["online_replay_from"], binding)
        self.assertEqual(result["charge"]["stage"], "wan_compile")
        self.assertEqual(self.last_command[self.last_command.index("--online-replay-from") + 1], str(prior))
        self.assertEqual(json.loads(prior.read_text()), original)


class OnlineMatrixTests(unittest.TestCase):
    def test_complete_ab_ba_matrix_and_budget_limited_no_new_unpaired_work(self):
        from ditserve.optimizations import online
        traces = [dict(seed=101 + i, load=.5, pattern=pattern, trace_sha256="trace%d" % i,
                       requests=[dict(request_id="r000", kind="video", arrival_ms=0., service_ms=1000000.)])
                  for i, pattern in enumerate(("burst", "ordinary"))]
        for remaining, expected in ((100000., 8), (5000., 0)):
            with self.subTest(remaining=remaining), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                guard = SimpleNamespace(remaining=lambda: remaining, ensure_unit=Mock())
                sana = SimpleNamespace(set_backend=Mock(), generate_batch=Mock(), steady_context=contextlib.nullcontext)
                context = SimpleNamespace(config=dict(old_online="unused"), output=root, run_root=root,
                    executor=object(), worker=object(), sana=sana, opts=dict(prompts=["a", "b"]),
                    check_budget=guard, metadata=dict(wan_warmed=True))
                result = {}
                def measured_trace(ctx, requests, options):
                    before = (root / "online.json").read_bytes()
                    for count in (1, 2):
                        ctx.save_partial(dict(status="running", complete=False,
                            requests=[dict(request_id="r000")], events=[dict(kind="fixture")] * count))
                        self.assertEqual((root / "online.json").read_bytes(), before)
                        current = json.loads((root / "partial.json").read_text())
                        self.assertEqual(len(current["events"]), count)
                        self.assertNotIn("rows", current)
                    return dict(status="completed", complete=True)
                with patch.object(online, "load_frozen_traces", return_value=traces), \
                     patch.object(online, "run_trace", side_effect=measured_trace) as run, \
                     patch.object(online, "paired_gate", return_value=dict(passed=True)) as gate:
                    worker._online(context, result, dict(image_batch_size=2))
                document = json.loads((root / "online.json").read_text())
                self.assertEqual(run.call_count, expected)
                self.assertEqual(len(document["rows"]), expected)
                self.assertEqual(result["online_qualified"], expected == 8)
                if expected == 8:
                    gate.assert_called_once()
                    self.assertEqual([(r["order_index"], r["arm"]) for r in document["rows"][:4]],
                                     [(0, "baseline"), (0, "candidate"), (1, "candidate"), (1, "baseline")])
                else:
                    gate.assert_not_called()
                    self.assertEqual(document["rows"], [])
                    self.assertEqual(document["status"], "not_run_budget_limited")


class WorkerReplayAdmissionTests(unittest.TestCase):
    def test_replay_flag_must_match_charged_original_supervisor_path_and_hash(self):
        from ditserve.directions.budget import BudgetStop
        from ditserve.research.common import digest, file_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'supervisor.json'
            atomic_json(source, dict(status='failed', reason='observer audit'))
            config = dict(gpu_uuid='fake-a100')
            binding = dict(path=str(source.resolve()), sha256=file_digest(source))
            args = SimpleNamespace(ledger=root/'ledger.json', output=root/'attempt03',
                session_id='session', phase='wan_compile', auxiliary_encoder=False, online_replay_from=source)
            active = dict(session_id='session', supervisor_pid=11, child_pgid=22, stage='wan_compile', gpu_count=1,
                identity=dict(phase='wan_compile', output=str(args.output.resolve()), config_sha256=digest(config),
                              gpu_uuids=['fake-a100'], online_replay_from=binding))
            with patch.object(worker, 'read_json', return_value=dict(active=active)), \
                 patch.object(worker.os, 'getppid', return_value=11), \
                 patch.object(worker.os, 'getpid', return_value=22), \
                 patch.object(worker.os, 'getpgrp', return_value=22), \
                 patch.dict(worker.os.environ, {'CUDA_VISIBLE_DEVICES': 'fake-a100'}):
                self.assertIs(worker.verify_admission(args, config), active)
                self.assertEqual(args.online_replay_binding, binding)
                for mutation in ('hash', 'path', 'missing_argument', 'missing_binding'):
                    with self.subTest(mutation=mutation):
                        active['identity']['online_replay_from'] = dict(binding)
                        args.online_replay_from = source
                        if mutation == 'hash':active['identity']['online_replay_from']['sha256'] = '0'*64
                        if mutation == 'path':active['identity']['online_replay_from']['path'] = str(root/'other.json')
                        if mutation == 'missing_argument':args.online_replay_from = None
                        if mutation == 'missing_binding':active['identity'].pop('online_replay_from')
                        with self.assertRaises(BudgetStop):worker.verify_admission(args, config)


if __name__ == "__main__":
    unittest.main()
