import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ditserve.research import cfg_protocol, dual_protocol
from ditserve.research.cfg_continuation import build_continuation, screening_decisions, verify_continuation
from ditserve.research.cfg_matrix import CANDIDATES, select_candidate, stage_a_gate
from ditserve.research.common import atomic_json, read_json
from ditserve.research.dual_evidence import UnitJournal, read_records
from ditserve.research.dual_matrix import seal
from test_research_cfg_matrix import cache_rows, quality


class CfgContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = read_json(cfg_protocol.DEFAULT_CONFIG)
        self.config_path = self.root / "configs/research_cfg.json"
        atomic_json(self.config_path, self.config)
        self.source = self.root / "python/ditserve/fixture.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("implementation = 'oracle'\n")
        self.ledger = self.root / "gpu-budget.json"
        atomic_json(self.ledger, dict(active=None, used_gpu_seconds=12800))
        self.manifest = {"sha256": "fake-model-manifest"}
        for name, result in (("verify_model_manifest", self.manifest), ("dependency_identity", {}), ("quality_assets", {})):
            patched = patch.object(dual_protocol, name, return_value=result)
            patched.start()
            self.addCleanup(patched.stop)
        self.parent_root = self.root / "results/run01"
        self.parent = cfg_protocol.freeze_protocol(self.config_path, self.parent_root, self.ledger,
                                                   model_manifest=self.manifest, root=self.root)
        journal = UnitJournal(self.parent_root, self.parent, ["GPU-a", "GPU-v"],
                              observe=lambda *args, **kwargs: dict(status="exclusive_observed"))
        for row in sum((cache_rows(candidate, "stage_a") for candidate in CANDIDATES), []):
            with journal.unit(row["case_id"], 1., time.monotonic() + 60):
                journal.append(row)
        self.parent_rows = read_records(self.parent_root, self.parent)
        self.gates = {candidate: stage_a_gate(self.parent_rows, self.config, candidate, "ctx")
                      for candidate in CANDIDATES}
        atomic_json(self.parent_root / "probe.json", seal(dict(status="completed", stage_a=self.gates,
            protocol_sha256=self.parent["sha256"], source_sha256=self.parent["source_sha256"], context_sha256="ctx")))
        for role in ("a100", "v100"):
            atomic_json(self.parent_root / ("screen-" + role + "-attempt.json"),
                        seal(dict(status="completed", returncode=0, protocol_sha256=self.parent["sha256"])))

    def new_freeze(self):
        self.source.write_text("implementation = 'split_prefix'\n")
        output = self.root / "results/run02"
        return cfg_protocol.freeze_protocol(self.config_path, output, self.ledger,
            model_manifest=self.manifest, root=self.root, prototype_run=self.parent_root)

    def current_development(self):
        rows = sum((cache_rows(candidate) for candidate in CANDIDATES), [])
        for row in rows:
            row["context_sha256"] = "new-context"
            for key in ("candidate", "reference"):
                row[key]["identity"]["context_sha256"] = "new-context"
            row["quality"] = quality(row["candidate"], row["reference"])
        return rows

    def test_new_freeze_binds_both_parent_gates_without_rewriting_parent_or_ledger(self):
        before = {str(path.relative_to(self.parent_root)): path.read_bytes()
                  for path in self.parent_root.rglob("*") if path.is_file() and "sources" not in path.parts}
        ledger = self.ledger.read_bytes()
        child = self.new_freeze()
        proof = child["prototype_continuation"]
        self.assertEqual(proof["purpose"], "implementation_continuation_only")
        self.assertFalse(proof["quality_qualified"])
        self.assertFalse(proof["performance_qualified"])
        self.assertEqual(proof["stage_a"], self.gates)
        self.assertEqual(len(proof["screen_records"]), 12)
        self.assertEqual(proof["parent_protocol_sha256"], self.parent["sha256"])
        self.assertNotEqual(child["source_sha256"], self.parent["source_sha256"])
        self.assertEqual(cfg_protocol.verify_protocol(self.root / "results/run02/protocol.json", root=self.root), child)
        self.assertEqual(cfg_protocol.verify_protocol(self.parent_root / "protocol.json", verify_current=False), self.parent)
        self.assertEqual({name: (self.parent_root / name).read_bytes() for name in before}, before)
        self.assertEqual(self.ledger.read_bytes(), ledger)

    def test_parent_gate_only_allows_fresh_current_development_qualification(self):
        child = self.new_freeze()
        self.assertEqual(screening_decisions(child, [], "new-context"), self.gates)
        selected = select_candidate([], self.config, "new-context", frozen=child)
        self.assertIsNone(selected["selected"])
        self.assertEqual(selected["status"], "unverified")
        selected = select_candidate(self.current_development(), self.config, "new-context", frozen=child)
        self.assertEqual(selected["selected"], "independent_004")
        stale = sum((cache_rows(candidate) for candidate in CANDIDATES), [])
        self.assertIsNone(select_candidate(stale, self.config, "new-context", frozen=child)["selected"])
        # Parent screen records cannot be relabelled as the child's Stage A.
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            screening_decisions(child, self.parent_rows, "new-context")

    def test_arbitrary_gate_dict_and_resealed_forged_continuation_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "recomputed"):
            select_candidate([], self.config, "new-context", stage_a=self.gates)
        child = self.new_freeze()
        arbitrary = copy.deepcopy(self.gates)
        arbitrary["independent_004"]["eligible"] = False
        with self.assertRaisesRegex(ValueError, "recomputed"):
            select_candidate([], self.config, "new-context", stage_a=arbitrary, frozen=child)
        forged = copy.deepcopy(child)
        proof = forged["prototype_continuation"]
        proof["parent_source_sha256"] = "0" * 64
        proof.pop("sha256")
        forged["prototype_continuation"] = seal(proof)
        forged.pop("sha256")
        with self.assertRaisesRegex(ValueError, "sealed parent"):
            screening_decisions(seal(forged), [], "new-context")
        with self.assertRaisesRegex(ValueError, "checksum-bound"):
            screening_decisions(dict(child, sha256="wrong"), [], "new-context")

    def test_missing_parent_record_or_source_archive_prevents_continuation(self):
        proof = build_continuation(self.parent_root, config=self.config, ledger=self.ledger)
        path = self.parent_root / "records.jsonl"
        original = path.read_text()
        path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "complete sealed"):
            verify_continuation(proof, config=self.config, ledger=self.ledger)
        path.write_text(original)
        (self.parent_root / "sources/python/ditserve/fixture.py").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            verify_continuation(proof, config=self.config, ledger=self.ledger)

    def test_parent_incomplete_supervisor_active_unit_changed_recipe_and_ledger_rejected(self):
        with self.assertRaisesRegex(ValueError, "lifetime ledger"):
            build_continuation(self.parent_root, config=self.config, ledger=self.root / "new-ledger.json")
        with self.assertRaisesRegex(ValueError, "scientific recipe"):
            build_continuation(self.parent_root, config=dict(self.config, epsilon=.06), ledger=self.ledger)
        atomic_json(self.parent_root / "active-unit.json", dict(status="running"))
        with self.assertRaisesRegex(ValueError, "active paired unit"):
            build_continuation(self.parent_root, config=self.config, ledger=self.ledger)
        atomic_json(self.parent_root / "active-unit.json", dict(status="idle"))
        atomic_json(self.parent_root / "screen-v100-attempt.json",
                    seal(dict(status="failed", returncode=1, protocol_sha256=self.parent["sha256"])))
        with self.assertRaisesRegex(ValueError, "supervisor"):
            build_continuation(self.parent_root, config=self.config, ledger=self.ledger)

    def test_parent_unit_tamper_and_declared_qualification_are_rejected(self):
        proof = build_continuation(self.parent_root, config=self.config, ledger=self.ledger)
        forged = copy.deepcopy(proof)
        forged["quality_qualified"] = True
        forged.pop("sha256")
        with self.assertRaisesRegex(ValueError, "invalid prototype"):
            verify_continuation(seal(forged), config=self.config, ledger=self.ledger)
        path = self.parent_root / "units" / (proof["screen_records"][0]["unit_id"] + ".json")
        unit = read_json(path)
        unit["performance_valid"] = False
        unit.pop("sha256")
        atomic_json(path, seal(unit))
        with self.assertRaisesRegex(ValueError, "complete sealed"):
            verify_continuation(proof, config=self.config, ledger=self.ledger)


if __name__ == "__main__":
    unittest.main()
