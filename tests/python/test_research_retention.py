"""Destructive retention logic is exercised only against temporary fixtures."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.research.common import atomic_json, digest, file_digest, read_json
from ditserve.research.retention import (
    _seal, apply_plan, build_plan, discover_catalog, select_key_samples,
    verify_plan, verify_reference,
)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "results"
        self.run = self.root / "series" / "run01"
        self.run.mkdir(parents=True)
        self.store = Path(self.temp.name) / "key-artifacts"
        self.ledger = self.root / "gpu-budget.json"
        atomic_json(self.ledger, dict(active=None, used_gpu_seconds=100))
        self.metrics = self.run / "metrics.json"
        atomic_json(self.metrics, dict(samples=8, means=[.01, .02, .03, .04]))

    def artifact(self, name, data=None):
        path = self.run / name
        path.write_bytes(data if data is not None else name.encode())
        return dict(path=str(path), sha256=file_digest(path), kind="output")

    def case(self, name, quality=None, **kwargs):
        result = dict(series="series", run=str(self.run), ended_at=1, case_id=name,
                      kind="calibration", split="dev", policy="t008", selected_policy="t008",
                      quality=quality, failed=False, artifacts=[self.artifact(name + ".npy")],
                      reference_sha256s=[], verified=True, evaluated=True)
        result.update(kwargs)
        return result

    def plan(self, cases, **kwargs):
        return build_plan(self.root, self.store, catalog=dict(cases=cases, sources=[str(self.metrics)]), **kwargs)

    def test_selection_latest_median_ties_and_failures(self):
        cases = [self.case("old", .99, run="old", ended_at=0),
                 self.case("b", .01), self.case("a", .01), self.case("c", .03), self.case("d", .04),
                 self.case("hold", .02, split="holdout"), self.case("failure", .1, split="unknown", failed=True),
                 self.case("correct", kind="correctness", split="unknown")]
        selected = select_key_samples(list(reversed(cases)))
        by_id = {case["case_id"]: case for case in selected}
        self.assertEqual(set(by_id), {"a", "d", "hold", "failure", "correct"})
        self.assertIn("selected_policy_development_median", by_id["a"]["reasons"])
        self.assertEqual(selected, select_key_samples(cases))

    def test_development_worst_does_not_require_selection_but_median_does(self):
        cases = [self.case("a", .01), self.case("b", .03),
                 self.case("unselected", .04, policy="other")]
        by_id = {case["case_id"]: case for case in select_key_samples(cases)}
        self.assertEqual(set(by_id), {"a", "unselected"})
        self.assertEqual(by_id["a"]["reasons"], ["selected_policy_development_median"])
        self.assertEqual(by_id["unselected"]["reasons"], ["development_worst"])

    def test_key_reference_closure_dedup_and_statistics_unchanged(self):
        reference = self.case("reference", 0, policy="off")
        candidates = [self.case("a", .01), self.case("b", .02), self.case("c", .03), self.case("d", .04)]
        candidates[1]["reference_sha256s"] = [reference["artifacts"][0]["sha256"]]
        duplicate = self.artifact("same-key.npy", Path(candidates[1]["artifacts"][0]["path"]).read_bytes())
        candidates[1]["artifacts"].append(duplicate)
        before = self.metrics.read_bytes()
        plan = self.plan([reference] + candidates)
        result = apply_plan(plan, verify_plan(plan))
        self.assertEqual(before, self.metrics.read_bytes())
        self.assertTrue(result["statistics_unchanged"])
        self.assertEqual(result["unique_key_files"], 3)
        for case in (reference, candidates[1], candidates[3]):
            item = case["artifacts"][0]
            verified = verify_reference(self.run, item["path"], item["sha256"])
            self.assertTrue(verified["re_evaluable"])
            self.assertTrue(Path(verified["path"]).is_file())
        removed = candidates[0]["artifacts"][0]
        status = verify_reference(self.run, removed["path"], removed["sha256"])
        self.assertEqual(status["status"], "retained_metrics_only")
        self.assertFalse(status["re_evaluable"])
        self.assertGreater(result["net_released_blocks_bytes"], 0)
        # Applying the exact manifest twice cannot redelete or double-charge.
        self.assertEqual(result["net_released_blocks_bytes"], apply_plan(plan, verify_plan_mock(plan))["net_released_blocks_bytes"])

    def test_active_ledger_blocks_before_discovery_or_hashing(self):
        atomic_json(self.ledger, dict(active={"pid": 999999}))
        with patch("ditserve.research.retention.discover_catalog", side_effect=AssertionError("must not scan")):
            with self.assertRaisesRegex(RuntimeError, "unsettled"):
                build_plan(self.root, self.store)

    def test_shared_external_unknown_and_pending_are_protected(self):
        case = self.case("owned", .01)
        hard = self.run / "hard.npy"
        os.link(case["artifacts"][0]["path"], hard)
        external = Path(self.temp.name) / "shared.npy"
        external.write_bytes(b"shared")
        (self.run / "external.npy").symlink_to(external)
        unknown = self.artifact("unknown.npy")
        pending = self.case("pending", .02)
        plan = self.plan([case, pending], pending_references=[pending["artifacts"][0]["path"]])
        self.assertEqual(plan["files"], [])
        self.assertGreaterEqual(len(plan["exceptions"]), 4)
        self.assertTrue(Path(unknown["path"]).exists())
        self.assertTrue(external.exists())

    def test_external_array_directory_requires_explicit_ownership(self):
        ext = Path(self.temp.name) / "owned-external"
        ext.mkdir()
        arrays = self.run / "arrays"
        arrays.symlink_to(ext, target_is_directory=True)
        (ext / "sample.npy").write_bytes(b"sample")
        case = self.case("key", .01)
        case["artifacts"] = [dict(path=str(arrays / "sample.npy"), sha256=file_digest(ext / "sample.npy"), kind="output")]
        self.assertEqual(self.plan([case])["files"], [])
        plan = self.plan([case], project_external_roots=[ext])
        self.assertEqual(len(plan["files"]), 1)
        apply_plan(plan, verify_plan(plan))
        self.assertFalse((ext / "sample.npy").exists())

    def test_missing_unexplained_array_is_error(self):
        case = self.case("missing", .01)
        item = case["artifacts"][0]
        Path(item["path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "undeclared"):
            self.plan([case])
        with self.assertRaises(FileNotFoundError):
            verify_reference(self.run, item["path"], item["sha256"])

    def test_corrupt_reference_and_changed_metadata_block_before_delete(self):
        case = self.case("key", .01)
        plan = self.plan([case])
        Path(case["artifacts"][0]["path"]).write_bytes(b"mutated")
        with self.assertRaises(ValueError):
            verify_plan(plan)
        case = self.case("other", .02)
        plan = self.plan([case])
        verification = verify_plan(plan)
        self.metrics.write_text('{"samples": 99}')
        with self.assertRaisesRegex(ValueError, "structured metrics"):
            apply_plan(plan, verification)
        self.assertTrue(Path(case["artifacts"][0]["path"]).exists())

    def test_interrupt_after_unlink_resumes_from_intent(self):
        cases = [self.case("a", .01), self.case("b", .02), self.case("c", .03)]
        plan = self.plan(cases)
        verification = verify_plan(plan)
        original = Path.unlink
        did_interrupt = []
        def unlink(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path.suffix == ".npy" and not did_interrupt:
                did_interrupt.append(True)
                raise RuntimeError("injected cleanup interruption")
            return result
        with patch.object(Path, "unlink", unlink):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                apply_plan(plan, verification)
        result = apply_plan(plan, verification)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["statistics_unchanged"])

    def test_selection_requires_reference_completeness(self):
        case = self.case("a", .01)
        case["reference_sha256s"] = ["f" * 64]
        plan = self.plan([case])
        self.assertEqual(plan["files"], [])
        self.assertTrue(any("no identifiable" in e["reason"] for e in plan["exceptions"]))

    def test_legacy_implicit_reference_hash_is_bound_without_rewriting_rows(self):
        case = self.case("a", .01)
        arrays = self.run / "arrays"
        arrays.mkdir()
        reference = arrays / "research-mechanism-4-video.npy"
        reference.write_bytes(b"original reference")
        expected = file_digest(reference)
        case["reference_sha256s"] = [expected]
        plan = self.plan([case])
        self.assertEqual(len(plan["implicit_reference_bindings"]), 1)
        self.assertEqual(plan["implicit_reference_bindings"][0]["sha256"], expected)
        self.assertEqual(len([f for f in plan["files"] if f["action"] == "retain_key"]), 2)
        apply_plan(plan, verify_plan(plan))
        self.assertTrue(verify_reference(self.run, reference, expected)["re_evaluable"])

    def test_incremental_cleanup_recovers_nonkey_reference_and_pinned_latent(self):
        cases = [self.case("a", .01), self.case("b", .02), self.case("c", .03)]
        arrays = self.run / "arrays"
        arrays.mkdir()
        reference = arrays / "research-mechanism-4-video.npy"
        latent = arrays / "research-mechanism-4-latent.npy"
        reference.write_bytes(b"evaluated reference video")
        latent.write_bytes(b"paired final latent")
        source_relative = "python/ditserve/research/experiment.py"
        source = self.run / "sources" / source_relative
        source.parent.mkdir(parents=True)
        source.write_text('artifact = self.root / "arrays" / spec["request_id"]\n'
                          'np.save(str(artifact) + "-video.npy", output, allow_pickle=False)\n'
                          'np.save(str(artifact) + "-latent.npy", latent, allow_pickle=False)\n'
                          'request_id="research-%s-%d" % (self.job["stage"], self.counter)\n')
        atomic_json(self.run / "protocol.json", _seal(dict(source_files={source_relative: file_digest(source)})))
        first = self.plan(cases)
        result1 = apply_plan(first, verify_plan(first))
        self.assertTrue(reference.exists())
        self.assertTrue(latent.exists())
        completion_path = self.root / ".retention" / first["sha256"] / "completion.json"
        historical_completion = completion_path.read_bytes()
        cases[0]["reference_sha256s"] = [file_digest(reference)]
        second = self.plan(cases)
        self.assertEqual({Path(f["path"]).name for f in second["files"]}, {reference.name, latent.name})
        self.assertTrue(all(f["action"] == "prune" for f in second["files"]))
        self.assertTrue(any(b["binding"] == "pinned_generator_final_latent_pair" for b in second["implicit_reference_bindings"]))
        result2 = apply_plan(second, verify_plan(second))
        self.assertEqual(completion_path.read_bytes(), historical_completion)
        self.assertEqual(result1["unique_key_files"], result2["unique_key_files"])
        self.assertEqual(result2["key_files_processed_this_pass"], 0)
        for case in cases[1:]:
            item = case["artifacts"][0]
            self.assertTrue(verify_reference(self.run, item["path"], item["sha256"])["re_evaluable"])
        self.assertEqual(verify_reference(self.run, reference, cases[0]["reference_sha256s"][0])["status"], "retained_metrics_only")

    def test_replaced_external_link_refuses_apply(self):
        ext = Path(self.temp.name) / "external"
        ext.mkdir()
        arrays = self.run / "arrays"
        arrays.symlink_to(ext, target_is_directory=True)
        target = ext / "sample.npy"
        target.write_bytes(b"a")
        case = self.case("a", .01)
        case["artifacts"] = [dict(path=str(arrays / "sample.npy"), sha256=file_digest(target), kind="output")]
        plan = self.plan([case], project_external_roots=[ext])
        verification = verify_plan(plan)
        arrays.unlink()
        other = Path(self.temp.name) / "foreign"
        other.mkdir()
        (other / "sample.npy").write_bytes(b"a")
        arrays.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "changed or shared"):
            apply_plan(plan, verification)
        self.assertTrue((other / "sample.npy").exists())

    def test_legacy_joint_policy_has_its_own_development_samples(self):
        rows = []
        for kind, key, selected, losses in (("calibration", "recipe_id", "t008", [.001, .002, .003]),
                                           ("joint_quality", "candidate_id", "t008-hot-t008", [.01, .02, .03])):
            for index, loss in enumerate(losses):
                item = self.artifact(kind + str(index) + ".npy")
                rows.append(_seal(dict(type=kind, case_id=kind + str(index), status="completed", split="dev",
                    protocol_sha256="p", evidence=dict(metrics=dict(lpips_mean=loss)), **{key: selected},
                    run=dict(output_path=item["path"], output_sha256=item["sha256"]))))
        (self.run / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        atomic_json(self.run / "report.json", dict(status="completed", protocol_sha256="p"))
        atomic_json(self.run / "calibration.json", _seal(dict(selected=dict(recipe_id="t008"))))
        atomic_json(self.run / "action-selection.json", _seal(dict(protocol_sha256="p", selected=[dict(candidate_id="t008-hot-t008")])))
        selected = select_key_samples(discover_catalog(self.root)["cases"])
        self.assertEqual({c["case_id"] for c in selected}, {"calibration1", "calibration2", "joint_quality1", "joint_quality2"})
        self.assertEqual(len({c["series"] for c in selected}), 2)

    def test_dual_schema_selects_two_reference_migration_and_raw_state(self):
        atomic_json(self.run / "protocol.json", _seal(dict(schema_version="ditserve.research.dual.frozen.v1")))
        atomic_json(self.run / "report.json", dict(status="completed", protocol_sha256="p"))
        atomic_json(self.run / "migration-selection.json", _seal(dict(selected="cold_migrate")))
        atomic_json(self.run / "migration-attempt.json", dict(status="completed"))
        atomic_json(self.run / "active-unit.json", dict(status="idle"))
        units = self.run / "units"
        units.mkdir()
        rows = []
        def run_artifact(name):
            video = self.artifact(name + ".npy")
            state = self.artifact(name + ".pkl")
            latent = self.artifact(name + "-latent.npy")
            return dict(output_path=video["path"], output_sha256=video["sha256"],
                        latent_path=latent["path"], latent_file_sha256=latent["sha256"],
                        latent_sha256="0" * 64,
                        checkpoint=dict(state_path=state["path"], state_sha256=state["sha256"]))
        for index, split in enumerate(("dev", "dev", "holdout")):
            row = _seal(dict(type="migration_quality", case_id=str(index), unit_id=str(index),
                protocol_sha256="p", source_sha256="s", split=split, status="completed", action="cold_migrate",
                common_reference=run_artifact("common" + str(index)),
                path_reference=run_artifact("path" + str(index)), candidate=run_artifact("candidate" + str(index)),
                canonical_quality=dict(lpips_mean=.01 + index / 100), path_quality=dict(lpips_mean=.02)))
            rows.append(row)
            atomic_json(units / (str(index) + ".json"), _seal(dict(status="completed", records=[row["sha256"]],
                protocol_sha256="p", source_sha256="s")))
        failed = _seal(dict(type="probe", case_id="failed", unit_id="failed", protocol_sha256="p", source_sha256="s",
                           run=run_artifact("failed")))
        rows.append(failed)
        atomic_json(units / "failed.json", _seal(dict(status="failed", records=[failed["sha256"]],
                    protocol_sha256="p", source_sha256="s")))
        (self.run / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        catalog = discover_catalog(self.root)
        selected = select_key_samples(catalog["cases"])
        self.assertEqual({c["case_id"] for c in selected}, {"0", "2"})
        plan = build_plan(self.root, self.store, catalog=catalog)
        keys = [f for f in plan["files"] if f["action"] == "retain_key"]
        self.assertEqual(len(keys), 18)
        self.assertTrue(any(f["path"].endswith(".pkl") for f in keys))
        self.assertTrue(any("failed" in f["path"] for f in plan["exceptions"]))
        verify_plan(plan)  # latent payload hashes must not substitute file SHA.

    def test_dual_pending_stage_or_unit_protects_every_raw_file(self):
        atomic_json(self.run / "protocol.json", _seal(dict(schema_version="ditserve.research.dual.frozen.v1")))
        atomic_json(self.run / "report.json", dict(status="completed"))
        (self.run / "records.jsonl").write_text("")
        atomic_json(self.run / "active-unit.json", dict(status="running"))
        catalog = discover_catalog(self.root)
        self.assertIn(str(self.run), catalog["protected_roots"])
        atomic_json(self.run / "active-unit.json", dict(status="idle"))
        atomic_json(self.run / "migration-attempt.json", dict(status="running"))
        self.assertIn(str(self.run), discover_catalog(self.root)["protected_roots"])

    def cfg_artifact(self, name, recipe="independent_004", implementation="native"):
        video = self.artifact(name + ".npy")
        state = self.artifact(name + ".pkl")
        return dict(output_path=video["path"], output_sha256=video["sha256"],
                    checkpoint=dict(state_path=state["path"], state_sha256=state["sha256"]),
                    recipe_id=recipe, device_role="a100", identity=dict(implementation=implementation))

    def cfg_fixture(self, payloads, selected="independent_004"):
        protocol = _seal(dict(schema_version="ditserve.research.cfg.frozen.v1", source_sha256="source"))
        atomic_json(self.run / "protocol.json", protocol)
        atomic_json(self.run / "report.json", _seal(dict(status="completed", protocol_sha256=protocol["sha256"])))
        atomic_json(self.run / "selection.json", _seal(dict(status="selected" if selected else "direction_stopped",
            selected=selected, protocol_sha256=protocol["sha256"], source_sha256="source")))
        atomic_json(self.run / "active-unit.json", dict(status="idle"))
        atomic_json(self.run / "pending-references.json", dict(pending=False))
        atomic_json(self.run / "screen-attempt.json", dict(status="completed"))
        rows = []
        for index, payload in enumerate(payloads):
            row = _seal(dict(payload, unit_id=str(index), protocol_sha256=protocol["sha256"],
                             source_sha256="source", recorded_at=100 + index))
            rows.append(row)
            atomic_json(self.run / "units" / (str(index) + ".json"), _seal(dict(status="completed",
                records=[row["sha256"]], protocol_sha256=protocol["sha256"], source_sha256="source")))
        (self.run / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return rows

    def test_cfg_selected_samples_keep_real_case_ids_both_orders_and_all_references(self):
        payloads = [dict(type="cfg_correctness", case_id="a100-independent004", status="completed", equal=True,
            recipe_id="independent_004", device_role="a100", continuous=self.cfg_artifact("continuous"),
            restored=self.cfg_artifact("restored"))]
        for index, (case_id, loss) in enumerate((("b", .01), ("a", .02), ("c", .04), ("a", .06))):
            payloads.append(dict(type="cfg_calibration", phase="dev", split="dev", case_id=case_id,
                status="completed", recipe_id="independent_004", device_role="a100", quality=dict(lpips_mean=loss),
                candidate=self.cfg_artifact("candidate" + str(index)), reference=self.cfg_artifact("reference" + str(index), "off")))
        rows = self.cfg_fixture(payloads)
        catalog = discover_catalog(self.root)
        selected = select_key_samples(catalog["cases"])
        self.assertEqual(selected, select_key_samples(list(reversed(catalog["cases"]))))
        chosen = [c for c in selected if c["case_id"] == "a"]
        self.assertEqual({c["evidence_id"] for c in chosen}, {rows[2]["sha256"], rows[4]["sha256"]})
        self.assertTrue(any("quality_failure_worst" in c["reasons"] for c in chosen))
        originals = {name: (self.run / name).read_bytes() for name in ("records.jsonl", "protocol.json", "report.json")}
        plan = build_plan(self.root, self.store, catalog=catalog)
        keys = {Path(f["path"]).name for f in plan["files"] if f["action"] == "retain_key"}
        for name in ("continuous", "restored", "candidate1", "reference1", "candidate3", "reference3"):
            self.assertIn(name + ".npy", keys)
            self.assertIn(name + ".pkl", keys)
        apply_plan(plan, verify_plan(plan))
        for name, value in originals.items():
            self.assertEqual((self.run / name).read_bytes(), value)

    def test_cfg_no_selected_policy_keeps_formal_worst_and_reference_deterministically(self):
        payloads = []
        for index, (case_id, loss) in enumerate((("b", .04), ("a", .04), ("a", .04), ("c", .01))):
            payloads.append(dict(type="cfg_calibration", phase="dev", split="dev", case_id=case_id,
                status="completed", recipe_id="paired_008", device_role="a100", quality=dict(lpips_mean=loss),
                candidate=self.cfg_artifact("candidate" + str(index), "paired_008", "split_prefix"),
                reference=self.cfg_artifact("reference" + str(index), "off")))
        rows = self.cfg_fixture(payloads, selected=None)
        catalog = discover_catalog(self.root)
        selected = select_key_samples(catalog["cases"])
        self.assertEqual(selected, select_key_samples(list(reversed(catalog["cases"]))))
        self.assertEqual(len(selected), 1)
        expected = min(rows[:3], key=lambda row: (row["case_id"], row["sha256"]))
        self.assertEqual(selected[0]["evidence_id"], expected["sha256"])
        self.assertEqual(selected[0]["reasons"], ["development_worst"])
        self.assertFalse(selected[0]["failed"])
        originals = {name: (self.run / name).read_bytes() for name in
                     ("records.jsonl", "protocol.json", "selection.json", "report.json")}
        plan = build_plan(self.root, self.store, catalog=catalog)
        retained = {entry["path"] for entry in plan["files"] if entry["action"] == "retain_key"}
        expected_paths = {run["output_path"] for run in (expected["candidate"], expected["reference"])}
        expected_paths.update(run["checkpoint"]["state_path"] for run in (expected["candidate"], expected["reference"]))
        self.assertEqual(retained, expected_paths)
        completion = apply_plan(plan, verify_plan(plan))
        self.assertTrue(completion["statistics_unchanged"])
        for name, value in originals.items():
            self.assertEqual((self.run / name).read_bytes(), value)
        for row in rows:
            for run in (row["candidate"], row["reference"]):
                status = verify_reference(self.run, run["output_path"], run["output_sha256"])
                self.assertEqual(status["status"], "raw_available" if row is expected else "retained_metrics_only")

    def test_cfg_failed_screen_and_diagnostic_neutrality_are_kept_but_routine_diagnostic_can_prune(self):
        rows = self.cfg_fixture([
            dict(type="cfg_screen", phase="stage_a", split="dev", case_id="failed-screen", status="completed",
                 recipe_id="paired_008", device_role="a100", quality=dict(lpips_mean=.07),
                 candidate=self.cfg_artifact("oracle", "paired_008", "oracle"), reference=self.cfg_artifact("oracle-off", "off")),
            dict(type="cfg_diagnostic_neutrality", case_id="a100", status="completed", equal=True,
                 clean=self.cfg_artifact("clean", "independent_008"), instrumented=self.cfg_artifact("instrumented", "independent_008")),
            dict(type="cfg_diagnostic", phase="diagnostic", case_id="ordinary-diagnostic", status="completed",
                 recipe_id="independent_008", device_role="a100", quality=dict(lpips_mean=.02),
                 candidate=self.cfg_artifact("diagnostic", "independent_008"), reference=self.cfg_artifact("diagnostic-off", "off"))], selected=None)
        catalog = discover_catalog(self.root)
        selected = select_key_samples(catalog["cases"])
        self.assertEqual({c["case_id"] for c in selected}, {"failed-screen", "a100"})
        failed = next(c for c in selected if c["case_id"] == "failed-screen")
        self.assertEqual(failed["split"], "stage_a")
        self.assertIsNone(failed["selected_policy"])
        self.assertIn("quality_failure_worst", failed["reasons"])
        plan = build_plan(self.root, self.store, catalog=catalog)
        pruned = {Path(f["path"]).name for f in plan["files"] if f["action"] == "prune"}
        self.assertEqual(pruned, {"diagnostic.npy", "diagnostic.pkl", "diagnostic-off.npy", "diagnostic-off.pkl"})
        verify_plan(plan)

    def test_cfg_pending_references_and_failed_units_protect_raw_evidence(self):
        rows = self.cfg_fixture([dict(type="cfg_calibration", phase="dev", split="dev", case_id="one",
            status="completed", recipe_id="independent_004", device_role="a100", quality=dict(lpips_mean=.02),
            candidate=self.cfg_artifact("one"), reference=self.cfg_artifact("one-off", "off"))])
        atomic_json(self.run / "pending-references.json", dict(pending=True))
        self.assertIn(str(self.run), discover_catalog(self.root)["protected_roots"])
        atomic_json(self.run / "pending-references.json", dict(pending=False))
        unit_path = self.run / "units/0.json"
        unit = read_json(unit_path)
        unit["status"] = "failed"
        atomic_json(unit_path, _seal(unit))
        catalog = discover_catalog(self.root)
        self.assertFalse(catalog["cases"][0]["verified"])
        self.assertEqual(build_plan(self.root, self.store, catalog=catalog)["files"], [])

    def test_cfg_selection_cannot_borrow_another_source_and_oracle_is_separate_series(self):
        payloads = [dict(type="cfg_screen", phase="stage_a", split="dev", case_id="oracle", status="completed",
            recipe_id="paired_008", device_role="a100", quality=dict(lpips_mean=.01),
            candidate=self.cfg_artifact("oracle", "paired_008", "oracle")),
            dict(type="cfg_screen", phase="stage_a", split="dev", case_id="oracle-best", status="completed",
            recipe_id="paired_008", device_role="a100", quality=dict(lpips_mean=.005),
            candidate=self.cfg_artifact("oracle-best", "paired_008", "oracle")),
            dict(type="cfg_calibration", phase="dev", split="dev", case_id="split", status="completed",
            recipe_id="paired_008", device_role="a100", quality=dict(lpips_mean=.02),
            candidate=self.cfg_artifact("split", "paired_008", "split_prefix"))]
        self.cfg_fixture(payloads, selected="paired_008")
        catalog = discover_catalog(self.root)
        self.assertEqual(len({c["series"] for c in catalog["cases"]}), 2)
        chosen = {c["case_id"]: c for c in select_key_samples(catalog["cases"])}
        self.assertEqual(set(chosen), {"oracle", "split"})
        self.assertEqual(chosen["oracle"]["reasons"], ["prototype_screen_worst"])
        self.assertEqual(chosen["split"]["reasons"], ["selected_policy_development_median", "development_worst"])
        path = self.run / "selection.json"
        value = read_json(path)
        value["source_sha256"] = "other"
        atomic_json(path, _seal(value))
        with self.assertRaisesRegex(ValueError, "protocol/source"):
            discover_catalog(self.root)

    def test_cfg_implementation_equivalence_retains_both_implementations_and_state(self):
        self.cfg_fixture([dict(type="cfg_implementation_equivalence", phase="correctness", case_id="a100-paired",
            status="completed", equal=True, device_role="a100", recipe_id="paired_008",
            continuous=self.cfg_artifact("prototype", "paired_008", "oracle"),
            restored=self.cfg_artifact("efficient", "paired_008", "split_prefix"))], selected=None)
        catalog = discover_catalog(self.root)
        selected = select_key_samples(catalog["cases"])
        self.assertEqual(len(selected), 1)
        self.assertIn("implementation-equivalence", selected[0]["series"])
        self.assertEqual(selected[0]["kind"], "correctness")
        plan = build_plan(self.root, self.store, catalog=catalog)
        self.assertEqual({Path(f["path"]).name for f in plan["files"] if f["action"] == "retain_key"},
                         {"prototype.npy", "prototype.pkl", "efficient.npy", "efficient.pkl"})
        verify_plan(plan)

    def test_research_discovery_preserves_protocol_and_excludes_active(self):
        baseline = self.artifact("baseline.npy")
        candidate = self.artifact("candidate.npy")
        row = _seal(dict(type="correctness", case_id="hot", status="passed", action="hot", protocol_sha256="p",
                         baseline=dict(output_path=baseline["path"], output_sha256=baseline["sha256"]),
                         candidate=dict(output_path=candidate["path"], output_sha256=candidate["sha256"])))
        (self.run / "records.jsonl").write_text(json.dumps(row) + "\n")
        atomic_json(self.run / "report.json", dict(status="completed", protocol_sha256="p"))
        catalog = discover_catalog(self.root)
        self.assertEqual(len(catalog["cases"]), 1)
        self.assertEqual(catalog["cases"][0]["kind"], "correctness")
        cold = _seal(dict(row, case_id="cold", action="cold"))
        (self.run / "records.jsonl").write_text(json.dumps(cold) + "\n" + json.dumps(row) + "\n")
        selected = select_key_samples(discover_catalog(self.root)["cases"])
        self.assertEqual([r["case_id"] for r in selected], ["hot"])
        (self.run / "report.json").unlink()
        catalog = discover_catalog(self.root)
        self.assertEqual(catalog["cases"], [])
        self.assertIn(str(self.run), catalog["protected_roots"])


def verify_plan_mock(plan):
    # Recover the immutable predelete verification stored by the first apply.
    return read_json(Path(plan["results_root"]) / ".retention" / plan["sha256"] / "verification.json")


if __name__ == "__main__":
    unittest.main()
