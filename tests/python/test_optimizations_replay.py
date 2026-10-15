"""Standard-library fixtures; no models, raw arrays, torch, or GPU work."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import replay
from ditserve.research.common import digest, file_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class ReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        patcher = patch.object(replay, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.run = self.root / "results/optimization-screen/run01"
        self.attempt = self.run / "wan_compile/attempt02"
        self.path = self.attempt / "supervisor.json"
        self.local_path = self.attempt / "wan-compile.json"
        self.audit_path = self.attempt / "online-measurement-audit.json"
        self.ledger_path = self.run.parent / "gpu-budget.json"
        self.snapshot = self.run / "sources/wan_compile-attempt02"
        self.config = dict(quality=dict(video_lpips=.05, rtol=1e-5, atol=1e-6),
                           direction1=dict(prompts=["kettle", "pendulum"]),
                           metrics=dict(minimum_local_gain=.05))
        self.config_sha = digest(self.config)
        files = replay._REQUIRED | {replay._PREFIX + "online.py", replay._PREFIX + "stage_phases.py",
                                   "configs/protocol.json", "tests/python/test_optimizations_old.py"}
        hashes = {}
        for name in files:
            text = "{}" if name.endswith(".json") else "# immutable tiny source\n"
            for base in (self.root, self.snapshot):
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            hashes[name] = file_digest(self.root / name)
        self.manifest = dict(execution_root=str(self.snapshot), source_files=hashes,
                             source_sha256=digest(hashes), config_sha256=self.config_sha,
                             phase="wan_compile", attempt="attempt02")
        write(self.snapshot / "manifest.json", self.manifest)
        self.local = self.make_local()
        write(self.local_path, self.local)
        self.audit = dict(invalidates_online_qualification=True, invalidates_local_qualification=False)
        write(self.audit_path, self.audit)
        self.charge = dict(session_id="settled", stage="wan_compile", gpu_seconds=1234.,
                           elapsed_seconds=1234., ended_at=5000., status="passed",
                           identity=dict(phase="wan_compile", output=str(self.attempt),
                                         run_root=str(self.run), config_sha256=self.config_sha,
                                         source_sha256=self.manifest["source_sha256"]))
        self.supervisor = dict(phase="wan_compile", output=str(self.attempt),
                               config_sha256=self.config_sha, source_sha256=self.manifest["source_sha256"],
                               status="passed", returncode=0, charge=self.charge,
                               local_qualified=True, final_occupancy=dict(status="exclusive_observed",
                                                                          owned=[], foreign=[]))
        self.ledger = dict(active=None, sessions=[self.charge], used_gpu_seconds=1234.)
        self.save_authority()

    def save_authority(self):
        write(self.path, self.supervisor)
        write(self.ledger_path, self.ledger)

    def artifact(self, name):
        # Intentionally absent: the verifier must not read/rehash large arrays.
        return dict(path=str(self.root / ".cache" / (name + ".npy")), shape=replay._SHAPE,
                    sha256="a" * 64, array_sha256="b" * 64, raw_available=True)

    def make_local(self):
        inputs = [dict(input_id="p%d-s%d" % (i, seed), prompt=prompt, seed=seed, split="dev")
                  for i, prompt in enumerate(self.config["direction1"]["prompts"]) for seed in (101, 102)]
        rows = []
        for backend in ("eager", "inductor-block1"):
            for item in inputs:
                rows.append(dict(kind="same_backend_resume", backend=backend, input_id=item["input_id"],
                    result=dict(passed=True,
                        steps=[dict(step=step, latent=True, solver=True, cache_state=True, rng=True, passed=True)
                               for step in range(1, 33)],
                        interruptions=[dict(step=step, image_service=dict(finite=True)) for step in (0, 8, 16)],
                        final=dict(passed=True, finite=True, shape=replay._SHAPE, rtol=1e-5, atol=1e-6),
                        artifacts=[self.artifact(backend + item["input_id"] + str(i)) for i in range(2)])))
        for order in range(2):
            for item in inputs:
                for arm in ("eager", "candidate"):
                    rows.append(dict(kind="performance", order=order, input_id=item["input_id"], repeat=0,
                        arm=arm, candidate_backend="inductor-block1", complete=True,
                        service=dict(ms=100. if arm == "eager" else 80., completed_steps=32,
                                     generation_completed=True, cache=False, native_cache=False,
                                     backend="eager" if arm == "eager" else "inductor-block1",
                                     output_sha256="b" * 64), artifact=self.artifact(str(len(rows))),
                        quality=dict(passed=True, metrics=dict(status="ok", finite=True, range_valid=True,
                            lpips_status="ok", shape=replay._SHAPE, frame_count=81,
                            lpips_mean=.001, lpips_per_frame=[.001]*81))))
        return dict(schema_version="ditserve.optimizations.wan.v1", phase="wan-compile", status="passed",
                    local_qualified=True, active_unit=None, selected_backend="inductor-block1", inputs=inputs,
                    quality_policy=dict(video_lpips=.05, same_backend_resume_rtol=1e-5, same_backend_resume_atol=1e-6),
                    rows=rows, metadata=dict(original="provenance"),
                    eager_baseline=dict(original_advance=True),
                    single_block=dict(complete=True, distinct_inputs=4, repeats=1, orders=2,
                        quality_passed=True, quality_checked=16, performance_passed=True, n_per_arm=8,
                        baseline_backend="eager", candidate_backend="inductor-block1", gains=[.2,.2]),
                    compilation={"inductor-block1":dict(compiled=True, group_size=1, fullgraph=True,
                        dynamic=False, cuda_graphs=False, external_callable=True, model_forward_replaced=False,
                        test_compiler=False, compilations=1, graph_invocations=30720,
                        compile_attempts=[dict(status="completed")])})

    def inspect(self):
        return replay.inspect_local_replay(self.path, self.config, self.run)

    def controlled_stop(self):
        self.supervisor.update(returncode=1, status="failed", local_qualified=False, reason="KeyboardInterrupt: ")
        self.charge["status"] = "failed"
        self.audit["controlled_stop"] = dict(signal="SIGINT", reason="history-dependent online observer",
            local_evidence_sha256=file_digest(self.local_path), session_id="settled")
        write(self.audit_path, self.audit)
        self.save_authority()

    def admit(self, binding):
        target = self.run / "wan_compile/attempt03"
        self.ledger["active"] = dict(session_id="new", stage="reserve",
            identity=dict(output=str(target), phase="wan_compile", run_root=str(self.run),
                          config_sha256=self.config_sha, online_replay_from=binding))
        write(self.ledger_path, self.ledger)
        return SimpleNamespace(output=target, config=self.config, run_root=self.run, replay_binding=binding,
                               metadata=dict(phase="wan_compile", wan_warmed=True, wan_warm_evidence={"stale":True}))

    def test_complete_local_metadata_validates_without_raw_array_reads(self):
        result = self.inspect()
        self.assertEqual(result["path"], str(self.path))
        self.assertEqual(result["sha256"], file_digest(self.path))
        self.assertEqual(result["local_validation"]["performance_rows"], 16)
        self.assertFalse(result["local_validation"]["arrays_read"])
        self.assertFalse(result["controlled_measurement_stop"])
        self.assertEqual(result["settled_session_sha256"], digest(self.charge))

    def test_controlled_interrupt_uses_independent_local_gate_not_supervisor_flag(self):
        self.controlled_stop()
        self.assertTrue(self.inspect()["controlled_measurement_stop"])
        self.local["local_qualified"] = False
        write(self.local_path, self.local)
        self.audit["controlled_stop"]["local_evidence_sha256"] = file_digest(self.local_path)
        write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, "local pilot"):
            self.inspect()

    def test_controlled_stop_requires_every_exact_audit_binding(self):
        self.controlled_stop()
        for key, value in [("signal","SIGTERM"), ("reason","budget"), ("session_id","other"),
                           ("local_evidence_sha256", "0"*64)]:
            with self.subTest(key=key):
                audit = copy.deepcopy(self.audit)
                audit["controlled_stop"][key] = value
                write(self.audit_path, audit)
                with self.assertRaisesRegex(ValueError, "nonzero exit"):
                    self.inspect()
        write(self.audit_path, self.audit)
        self.supervisor["reason"] = "RuntimeError: compiled graph failed"
        self.save_authority()
        with self.assertRaisesRegex(ValueError, "nonzero exit"):
            self.inspect()

    def test_other_nonzero_failed_and_supervisor_failures_are_rejected(self):
        for update in [dict(returncode=1), dict(returncode=2), dict(status="failed"),
                       dict(supervisor_failure="cleanup failed")]:
            with self.subTest(update=update):
                write(self.path, dict(self.supervisor, **update))
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_active_ledger_and_missing_settlement_rejected(self):
        write(self.ledger_path, dict(self.ledger, active={"session_id":"still-active"}))
        with self.assertRaisesRegex(ValueError, "ledger is active"):
            self.inspect()
        write(self.ledger_path, dict(self.ledger, sessions=[]))
        with self.assertRaisesRegex(ValueError, "settled"):
            self.inspect()
        write(self.ledger_path, dict(self.ledger, sessions=[self.charge, self.charge]))
        with self.assertRaisesRegex(ValueError, "settled"):
            self.inspect()

    def test_cleanup_and_audit_cannot_be_missing_or_invalidate_local(self):
        for invalid in [dict(invalidates_online_qualification=False),dict(invalidates_local_qualification=True)]:
            write(self.audit_path, dict(self.audit, **invalid))
            with self.assertRaisesRegex(ValueError, "invalidation"):
                self.inspect()
        write(self.audit_path, self.audit)
        self.supervisor["final_occupancy"]["owned"] = [{"pid":123}]
        self.save_authority()
        with self.assertRaisesRegex(ValueError, "cleanup"):
            self.inspect()

    def test_same_config_run_and_snapshot_are_mandatory(self):
        changed = copy.deepcopy(self.config)
        changed["quality"]["video_lpips"] = .1
        with self.assertRaisesRegex(ValueError, "config"):
            replay.inspect_local_replay(self.path, changed, self.run)
        wrong_run = self.root / "other"
        wrong_run.mkdir()
        with self.assertRaisesRegex(ValueError, "same run"):
            replay.inspect_local_replay(self.path, self.config, wrong_run)
        source = self.snapshot / (replay._PREFIX + "wan.py")
        source.write_text("# tampered old snapshot")
        with self.assertRaisesRegex(ValueError, "old snapshot source"):
            self.inspect()

    def test_math_change_new_unknown_code_or_missing_file_is_rejected(self):
        name = replay._PREFIX + "wan.py"
        source = self.root / name
        original = source.read_text()
        source.write_text("# changed math")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.inspect()
        source.write_text(original)
        unknown = self.root / (replay._PREFIX + "unclassified.py")
        unknown.write_text("# new code")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.inspect()
        unknown.unlink()
        source.unlink()
        with self.assertRaisesRegex(ValueError, "source files missing"):
            self.inspect()

    def test_explicit_observation_other_phase_and_new_replay_tests_are_bound(self):
        for name in ("online.py", "stage_phases.py", "cache_online.py", "replay.py"):
            (self.root / (replay._PREFIX + name)).write_text("# explicit reviewed changes\n")
        (self.root / "tests/python/test_optimizations_replay.py").write_text("# new tests\n")
        binding = self.inspect()
        changed = {x["path"] for x in binding["observation_only_changed_files"]}
        self.assertIn(replay._PREFIX + "replay.py", changed)
        self.assertIn(replay._PREFIX + "cache_online.py", changed)
        self.assertIn("tests/python/test_optimizations_replay.py", changed)
        self.assertIn(replay._PREFIX + "cache_executor.py", binding["mathematical_source_files"])

    def test_partial_duplicate_or_failed_trajectory_rejected(self):
        variants = []
        missing = copy.deepcopy(self.local); missing["rows"].pop(); variants.append(missing)
        duplicate = copy.deepcopy(self.local); duplicate["rows"][1] = duplicate["rows"][0]; variants.append(duplicate)
        incomplete = copy.deepcopy(self.local); incomplete["rows"][0]["result"]["steps"].pop(); variants.append(incomplete)
        bad_state = copy.deepcopy(self.local); bad_state["rows"][0]["result"]["steps"][8]["rng"] = False; variants.append(bad_state)
        missing_image = copy.deepcopy(self.local); missing_image["rows"][0]["result"]["interruptions"].pop(); variants.append(missing_image)
        bad_quality = copy.deepcopy(self.local); bad_quality["rows"][8]["quality"]["metrics"]["lpips_mean"] = .051; variants.append(bad_quality)
        partial = copy.deepcopy(self.local); partial["rows"][8]["complete"] = False; variants.append(partial)
        for index, local in enumerate(variants):
            with self.subTest(index=index):
                write(self.local_path, local)
                with self.assertRaises(ValueError):
                    self.inspect()

    def test_prompt_seed_compiler_and_recomputed_gain_are_not_trusted_flags(self):
        variants = []
        bad = copy.deepcopy(self.local); bad["inputs"][0]["seed"] = 999; variants.append(bad)
        bad = copy.deepcopy(self.local); bad["compilation"]["inductor-block1"]["graph_invocations"] = 0; variants.append(bad)
        bad = copy.deepcopy(self.local); bad["compilation"]["inductor-block1"]["test_compiler"] = True; variants.append(bad)
        bad = copy.deepcopy(self.local); bad["single_block"]["gains"] = [.3,.3]; variants.append(bad)
        bad = copy.deepcopy(self.local)
        for row in bad["rows"][8:]:
            if row["arm"] == "candidate": row["service"]["ms"] = 99.
        bad["single_block"]["gains"] = [.01,.01]; variants.append(bad)
        for index, local in enumerate(variants):
            with self.subTest(index=index):
                write(self.local_path, local)
                with self.assertRaises(ValueError): self.inspect()

    def test_load_preserves_sources_without_qualification_or_warmup_reuse(self):
        binding = self.inspect()
        context = self.admit(binding)
        result = replay.load_local_evidence(context, self.path)
        self.assertTrue(result["local_qualified"])
        self.assertFalse(result["local_evidence_reuse"]["remeasured"])
        self.assertEqual(result["rows"], self.local["rows"])
        self.assertEqual(result["local_evidence_reuse"]["source_metadata"], self.local["metadata"])
        self.assertTrue((context.output / "wan-compile.json").exists())
        self.assertEqual(json.loads(self.local_path.read_text()), self.local)
        self.assertNotIn("wan_warmed", context.metadata)
        self.assertNotIn("wan_warm_evidence", context.metadata)
        for key in ("online_qualified", "online_evaluation_complete", "scientific_qualified", "system_qualified"):
            self.assertIs(result[key], False)
        with self.assertRaisesRegex(ValueError, "already has"):
            replay.load_local_evidence(context, self.path)

    def test_load_uses_original_run_ledger_when_root_is_new_frozen_snapshot(self):
        binding = self.inspect()
        context = self.admit(binding)
        frozen = self.run / "sources/wan_compile-attempt03"
        for folder in ("python", "tests", "configs"):
            shutil.copytree(self.root / folder, frozen / folder)
        with patch.object(replay, "ROOT", frozen):
            result = replay.load_local_evidence(context, self.path)
        self.assertTrue(result["local_qualified"])

    def test_load_rejects_other_active_session_changed_binding_and_changed_source(self):
        binding = self.inspect()
        context = self.admit(binding)
        context.replay_binding = dict(binding, config_sha256="tampered")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            replay.load_local_evidence(context, self.path)
        context.replay_binding = binding
        self.ledger["active"]["identity"]["output"] = str(self.run / "wan_compile/attempt04")
        write(self.ledger_path, self.ledger)
        with self.assertRaisesRegex(ValueError, "newly admitted"):
            replay.load_local_evidence(context, self.path)
        context = self.admit(binding)
        (self.root / (replay._PREFIX + "online.py")).write_text("# even allowed code changed after binding\n")
        with self.assertRaisesRegex(ValueError, "binding changed"):
            replay.load_local_evidence(context, self.path)


if __name__ == "__main__":
    unittest.main()
