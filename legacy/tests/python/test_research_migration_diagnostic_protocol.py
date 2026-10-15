import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ditserve.research import cfg_protocol, dual_protocol
from ditserve.research import migration_diagnostic_protocol as protocol
from ditserve.research.common import atomic_json, digest, read_json
from ditserve.research.dual_evidence import UnitJournal
from ditserve.research.dual_matrix import seal


class MigrationDiagnosticProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.diagnostic = read_json(protocol.DEFAULT_CONFIG)
        self.config = read_json(cfg_protocol.DEFAULT_CONFIG)
        self.config_path = self.root / "configs/research_migration_diagnostic.json"
        atomic_json(self.config_path, self.diagnostic)
        atomic_json(self.root / "configs/research_cfg.json", self.config)
        self.source = self.root / "python/ditserve/fixture.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("implementation = 'native'")
        self.ledger = self.root / "budget.json"
        self.budget = dict(active=None, limit_gpu_seconds=28800., sessions=[],
                           stage_limits_gpu_seconds=dict(migration=5638.), used_gpu_seconds=14000.)
        atomic_json(self.ledger, self.budget)
        self.model = dict(sha256="test-model-manifest")
        for module, name, result in (
                (dual_protocol, "verify_model_manifest", self.model),
                (dual_protocol, "dependency_identity", {}), (dual_protocol, "quality_assets", {}),
                (protocol, "dependency_identity", {}), (protocol, "quality_assets", {})):
            mocked = patch.object(module, name, return_value=result)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch("ditserve.research.cfg_refinement.verify_cpu_binding", return_value={})
        mocked.start()
        self.addCleanup(mocked.stop)
        self.parent_root = self.root / "results/run03"
        parent = cfg_protocol.freeze_protocol(self.root / "configs/research_cfg.json", self.parent_root,
            self.ledger, model_manifest=self.model, root=self.root)
        # A diagnostic input has no old selection/holdout/prototype qualification
        # dependency. It needs only the closed migration failure and CPU input.
        parent.pop("sha256")
        parent["cpu_embeddings"] = dict(root=str(self.root / "cpu-cache"), manifest_sha256="cpu-input",
                                         model_manifest_sha256=self.model["sha256"])
        self.parent = seal(parent)
        atomic_json(self.parent_root / "protocol.json", self.parent)
        for name, status, extra in (
                ("terminal", "completed", {}),
                ("migration", "quality_stopped", dict(failed_reference_case="dev-kettle-102",
                    qualified_actions=[], reason="cache-off device-path reference failed common reference"))):
            atomic_json(self.parent_root / (name + ".json"), seal(dict(status=status,
                protocol_sha256=self.parent["sha256"], source_sha256=self.parent["source_sha256"], **extra)))
        atomic_json(self.parent_root / "pending-references.json", dict(pending=False))
        atomic_json(self.parent_root / "migration-attempt.json", seal(dict(status="completed", returncode=0,
            protocol_sha256=self.parent["sha256"])))
        atomic_json(self.parent_root / "migration-job.json", dict(phase="migration", budget_stage="migration",
            session_id="closed-migration", protocol_sha256=self.parent["sha256"], source_sha256=self.parent["source_sha256"],
            physical_devices=[dict(uuid="A100"), dict(uuid="V100")]))
        self.budget["sessions"].append(dict(session_id="closed-migration", status="completed", stage="migration",
            protocol_sha256=self.parent["sha256"], devices=["A100", "V100"], gpu_count=2, gpu_seconds=280.))
        atomic_json(self.ledger, self.budget)
        self.journal = UnitJournal(self.parent_root, self.parent, ["A100", "V100"],
                                  observe=lambda *args, **kwargs: dict(status="exclusive_observed"))
        for seed in (101, 102):
            row = self.reference_row(seed)
            with self.journal.unit("migration-kettle-" + str(seed), 1., time.monotonic() + 60):
                self.journal.append(row)
        self.output = self.root / "diagnostic/run01"

    def reference_row(self, seed):
        common = dict(recipe="off", implementation="native", boundary=16,
                      input=dict(prompt_id="kettle", seed=seed))
        left = dict(identity=dict(common, action="stay", path=["a100"], attention_path=["flash"]),
                    output_sha256="common-output-" + str(seed))
        right = dict(identity=dict(common, action="full_migrate", path=["v100", "a100"],
                                   attention_path=["efficient", "flash"]), output_sha256="path-output-" + str(seed))
        quality = dict(status="ok", lpips_status="ok", finite=True, range_valid=True,
            lpips_mean=.001 if seed == 101 else .076,
            reference_identity_sha256=digest(left["identity"]), reference_output_sha256=left["output_sha256"],
            candidate_identity_sha256=digest(right["identity"]), candidate_output_sha256=right["output_sha256"])
        return dict(type="cfg_migration_quality" if seed == 101 else "cfg_migration_reference",
            case_id="dev-kettle-" + str(seed) + ("-full_migrate" if seed == 101 else ""),
            status="completed" if seed == 101 else "failed", prompt_id="kettle", seed=seed,
            common_reference=left, path_reference=right, path_reference_quality=quality)

    def freeze(self):
        return protocol.freeze_protocol(self.config_path, self.output, self.ledger,
            parent_run=self.parent_root, model_manifest=self.model, root=self.root)

    def verify(self, **kwargs):
        return protocol.verify_protocol(self.output / "protocol.json", root=self.root, **kwargs)

    def test_new_freeze_retains_original_configuration_and_no_qualification(self):
        before_protocol = (self.parent_root / "protocol.json").read_bytes()
        before_ledger = self.ledger.read_bytes()
        frozen = self.freeze()
        self.assertEqual(self.verify(), frozen)
        self.assertEqual(frozen["config"], self.config)
        self.assertEqual(frozen["config"]["attention"], dict(A100="flash", V100="efficient"))
        self.assertEqual(frozen["diagnostic"], self.diagnostic)
        self.assertEqual(frozen["cpu_embeddings"], self.parent["cpu_embeddings"])
        self.assertEqual(frozen["budget_binding"]["stage"], "migration")
        for key in ("quality_qualified", "performance_qualified", "inherit_formal_evidence", "raw_reanalysis"):
            self.assertIs(frozen["parent_failure"][key], False)
        self.assertIn("configs/research_migration_diagnostic.json", frozen["source_files"])
        self.assertEqual(self.ledger.read_bytes(), before_ledger)
        self.assertEqual((self.parent_root / "protocol.json").read_bytes(), before_protocol)
        self.assertFalse((self.parent_root / "selection.json").exists())

    def test_fixed_diagnostic_scope_rejects_cache_shape_boundary_backend_or_cases(self):
        changes = dict(recipe_id="paired_008", transition_step=15, shadow_step=18,
                       budget_stage="reserve", cost_safety_factor=1.2,
                       comparisons=["original"], repetitions=["forward"],
                       cases=[dict(prompt_id="kettle", seed=103)])
        for key, value in changes.items():
            config = copy.deepcopy(self.diagnostic)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.validate_protocol(config)

    def test_estimates_are_adjustable_but_full_original_and_math_are_required(self):
        changed = copy.deepcopy(self.diagnostic)
        changed["estimate_wall_seconds"]["math"] = 800.
        self.assertEqual(protocol.validate_protocol(changed), changed)
        for value in (0., -1., float("inf"), float("nan")):
            changed["estimate_wall_seconds"]["math"] = value
            with self.assertRaises(ValueError):
                protocol.validate_protocol(changed)
        changed["estimate_wall_seconds"] = dict(original=500.)
        with self.assertRaises(ValueError):
            protocol.validate_protocol(changed)

    def test_attention_comparisons_do_not_silently_fallback(self):
        a = dict(name="NVIDIA A100", compute_capability="8.0")
        v = dict(name="Tesla V100", compute_capability="7.0")
        self.assertEqual(protocol.attention_for_device(a), "flash")
        self.assertEqual(protocol.attention_for_device(v), "efficient")
        self.assertEqual(protocol.attention_for_device(a, "math"), "math")
        self.assertEqual(protocol.attention_for_device(v, "math"), "math")
        for device, comparison in ((a, "efficient"), (dict(name="H100", compute_capability="9.0"), "math")):
            with self.assertRaises(ValueError):
                protocol.attention_for_device(device, comparison)

    def test_parent_pending_or_unsettled_work_prevents_freeze(self):
        atomic_json(self.parent_root / "pending-references.json", dict(pending=True))
        with self.assertRaisesRegex(ValueError, "active or pending"):
            self.freeze()
        atomic_json(self.parent_root / "pending-references.json", dict(pending=False))
        self.budget["active"] = dict(protocol_sha256=self.parent["sha256"])
        atomic_json(self.ledger, self.budget)
        with self.assertRaisesRegex(ValueError, "unsettled"):
            self.freeze()

    def test_missing_or_invalid_parent_unit_rejects_failure_anchor(self):
        path = self.parent_root / "units/migration-kettle-102.json"
        value = read_json(path)
        value.pop("sha256")
        value["performance_valid"] = False
        atomic_json(path, seal(value))
        with self.assertRaisesRegex(ValueError, "invalidated"):
            self.freeze()

    def test_changed_parent_source_or_model_manifest_is_not_accepted(self):
        archived = self.parent_root / "sources/python/ditserve/fixture.py"
        archived.write_text("tampered parent")
        with self.assertRaisesRegex(ValueError, "archived source"):
            self.freeze()

    def test_different_model_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "model manifest"):
            protocol.freeze_protocol(self.config_path, self.output, self.ledger, parent_run=self.parent_root,
                                     model_manifest=dict(sha256="other-model"), root=self.root)

    def test_frozen_run_refuses_changed_source_and_preserves_historical_archive(self):
        frozen = self.freeze()
        self.source.write_text("implementation = 'changed'")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.verify()
        self.assertEqual(self.verify(verify_current=False), frozen)
        with self.assertRaisesRegex(ValueError, "new output directory"):
            self.freeze()

    def test_stage_cap_cannot_change_but_settled_usage_can_increase(self):
        frozen = self.freeze()
        self.budget["used_gpu_seconds"] += 20.
        atomic_json(self.ledger, self.budget)
        self.assertEqual(self.verify(), frozen)
        self.budget["stage_limits_gpu_seconds"]["migration"] += 100.
        atomic_json(self.ledger, self.budget)
        with self.assertRaisesRegex(ValueError, "stage cap changed"):
            self.verify()

    def test_resealed_qualification_or_cpu_input_forgery_does_not_pass(self):
        original = self.freeze()
        for kind in ("qualification", "cpu"):
            frozen = copy.deepcopy(original)
            frozen.pop("sha256")
            if kind == "qualification":
                anchor = frozen["parent_failure"]
                anchor.pop("sha256")
                anchor["performance_qualified"] = True
                frozen["parent_failure"] = seal(anchor)
            else:
                frozen["cpu_embeddings"]["manifest_sha256"] = "different-input"
            atomic_json(self.output / "protocol.json", seal(frozen))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "failure anchor"):
                self.verify()


if __name__ == "__main__":
    unittest.main()
