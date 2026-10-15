"""Temporary metadata/Excel fixtures; no models, GPU calls or real results."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ditserve.directions.sana import RECIPE
from ditserve.optimizations import next_numerics_retry_report as report
from ditserve.research.common import atomic_json, digest, file_digest, read_json


def bind(path):
    return dict(path=str(path), sha256=file_digest(path))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    return bind(path)


def numeric(exact=True):
    return dict(shape_equal=True, dtype_equal=True, exact=exact, passed=exact, finite=True,
                rtol=1e-5, atol=1e-6, max_abs=0. if exact else .02,
                mean_abs=0. if exact else .01, rmse=0. if exact else .01,
                max_relative=0. if exact else .1)


def native(original=False):
    inputs = [dict(input_id="prompt%d-seed%d" % (index, seed), prompt=prompt, seed=seed)
              for index,prompt in enumerate(("one", "two")) for seed in (101,102)]
    rows = []
    for index,(arm,backend,members) in enumerate(report._units(inputs)):
        row = dict(unit_index=index, arm=arm, backend=backend, input_ids=members,
            status="completed", generation_completed=True, comparisons=[],
            execution=dict(generation_completed=True, requested_backend=backend, backend=backend,
                transformer_calls=2, scheduler_calls=2, same_cpu_generators_through_scm=True))
        if original and arm in ("eager_serial", "compiled_single"):
            row.update(status="failed", generation_completed=False,
                error="RuntimeError: self.stride(-1) must be 1 to view Float as Byte, but got 0")
        elif arm != "eager_serial":
            for key in members:
                if original:
                    row["comparisons"].append(dict(input_id=key, status="unavailable", reason="eager serial reference failed"))
                    continue
                components = []
                for boundary in report._boundaries():
                    ref = dict(sha256="a"*64, shape=[1], dtype="torch.bfloat16", finite=True)
                    components.append(dict(boundary=boundary, numerical=numeric(), reference=ref, candidate=ref.copy()))
                metrics = dict(status="ok", lpips_status="ok", finite=True, range_valid=True,
                    shape=[1,1,1024,1024,3], frame_count=1, lpips_per_frame=[.02], lpips_mean=.02, lpips_p95=.02)
                row["comparisons"].append(dict(input_id=key, status="completed",
                    trajectory=dict(complete=True, components=components, first_exact_drift=None,
                        first_strict_drift=None, all_exact=True, all_strict=True),
                    output=dict(epsilon=.01, passed=False, metrics=metrics, numerical=numeric(False),
                        reference_sha256="a"*64, candidate_sha256="b"*64)))
        rows.append(row)
    return dict(schema_version="ditserve.optimizations.sana_numerics.v1", status="passed",
        diagnostic_complete=True, inputs=inputs, recipe=dict(RECIPE),
        quality=dict(rtol=1e-5,atol=1e-6,image_lpips=.01), rows=rows, active_unit=None,
        required_units=12, required_generated_images=16, completed_units=4 if original else 12,
        failed_units=8 if original else 0, compared_images=0 if original else 12, lpips_passed_images=0,
        backend_restored=True, source={}, **{key:False for key in report.NO_CLAIMS})


class Fixture:
    def __init__(self, root):
        self.root = root
        self.output = root/"retry/run01"
        self.ledger = root/"retry/gpu-budget.json"
        self.attempt = self.output/"sana_numerics/attempt00"
        self.parents = [save(root/("parent%d"%i)/"gpu-budget.json",
            dict(active=None, used_gpu_seconds=10., sessions=[])) for i in range(4)]
        self.auth = save(root/"auth.json", dict(approved_by_user=True,budget_unlimited=True,authorized_gpu_seconds=None))
        self.config = dict(direction1=dict(prompts=["one","two"]), gpu_uuid="GPU-test")
        self.config_sha = digest(self.config)
        old_attempt = root/"parent3/run01/sana_numerics/attempt00"
        self.original_native = save(old_attempt/"sana-numerics.json", native(True))
        old_terminal = dict(phase="sana_numerics",output=str(old_attempt),status="passed")
        original = dict(native=self.original_native,terminal=save(old_attempt.parent/"terminal.json",old_terminal),
                        supervisor=save(old_attempt/"supervisor.json",old_terminal))
        chain_id = dict(pid=99999991,start_ticks=1,boot_id="fixture-previous-boot")
        finalizer_id = dict(pid=99999992,start_ticks=2,boot_id="fixture-previous-boot")
        self.chain_path = root/"parent3/run01/chain.json"
        chain = save(self.chain_path,dict(status="completed",started_unix=1.))
        main_report = save(root/"parent3/run01/report.json",dict(status="completed"))
        audit = save(root/"parent3/run01/audits/final-integrity.json",
            dict(status="passed",report_sha256=main_report["sha256"]))
        receipt = save(root/"parent3/run01/audits/finalize-attempt000/receipt.json",
            dict(status="completed",pid=finalizer_id["pid"],process_identity=chain_id,
                 chain_identity=dict(started_unix=1.),chain_final=chain,audit=audit))
        implementation = {}
        for key,name in (("numerics_source","next_numerics.py"),("numerics_test","test_optimizations_next_numerics.py"),
                         ("supervisor_source","next_study.py"),("runner_source","next_numerics_retry.py"),
                         ("reporter_source","next_numerics_retry_report.py"),
                         ("reporter_test","test_optimizations_next_numerics_retry_report.py")):
            directory = "tests/python" if key.endswith("_test") else "python/ditserve/optimizations"
            path = root/"source"/directory/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture "+name+"\n")
            implementation[key] = bind(path)
        self.binding = dict(schema_version=report.SCHEMA,config_sha256=self.config_sha,authorization=self.auth,
            parents=self.parents,original=original,original_failed_units=8,original_compared_images=0,
            required_units=12,required_generated_images=16,required_comparisons=12,
            parent_study=dict(chain=chain,finalizer_receipt=receipt,final_audit=audit,report=main_report),
            process_identities=dict(chain=chain_id,finalizer=finalizer_id),chain_started_unix=1.,
            implementation=implementation)
        admission = save(self.output/"retry-binding.json",self.binding)
        workloads = dict(frozen=True)
        save(self.output/"workloads.json",workloads)
        source_root = self.output/"sources/sana_numerics-attempt00"
        hashes = {}
        for key,value in implementation.items():
            relative = ("tests/python/" if key.endswith("_test") else "python/ditserve/optimizations/")+Path(value["path"]).name
            path = source_root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(value["path"]).read_bytes())
            hashes[relative] = value["sha256"]
        math_name = "python/ditserve/optimizations/next_study.py"
        mathematics = {math_name:hashes[math_name]}
        self.study = dict(schema_version="ditserve.optimizations.next_study.v1",phases=["sana_numerics"],
            config_sha256=self.config_sha,workloads_sha256=digest(workloads),authorization=self.auth,
            historical_ledgers=self.parents,cache_local_binding=dict(mathematical_source_files=mathematics,
                mathematical_source_sha256=digest(mathematics),bindings=dict(numerics_retry_admission=admission)))
        save(self.output/"study.json",self.study)
        save(self.attempt/"study.json",self.study)
        save(self.attempt/"config.json",self.config)
        self.manifest_path = source_root/"manifest.json"
        save(self.manifest_path,dict(source_files=hashes,source_sha256=digest(hashes),execution_root=str(source_root),
            config_sha256=self.config_sha,phase="sana_numerics",attempt="attempt00"))
        identity = dict(phase="sana_numerics",output=str(self.attempt),run_root=str(self.output),
            config_sha256=self.config_sha,source_sha256=digest(hashes),workloads_sha256=digest(workloads),
            cache_local_binding=self.study["cache_local_binding"],authorization_source=self.auth,gpu_uuids=["GPU-test"])
        self.session = dict(session_id="session",stage="sana_numerics",gpu_count=1,identity=identity,
            started_monotonic=100.,ended_monotonic=125.,elapsed_seconds=25.,gpu_seconds=25.,
            status="completed",child_pgid=900)
        self.budget = dict(stages=["sana_numerics"],budget_unlimited=True,limit_gpu_seconds=None,active=None,
            sessions=[self.session],authorization_binding=self.auth,parents={p["path"]:p for p in self.parents},
            gpu_uuid="GPU-test",used_gpu_seconds=25.,stage_used_gpu_seconds=dict(sana_numerics=25.))
        save(self.ledger,self.budget)
        self.terminal = dict(status="passed",phase="sana_numerics",charge=self.session,output=str(self.attempt),
            source_sha256=digest(hashes),config_sha256=self.config_sha,returncode=0,supervisor_failure=None,
            evidence_path=str(self.attempt/"sana-numerics.json"),
            final_occupancy=dict(status="exclusive_observed",owned=[],foreign=[]),
            **{key:False for key in report.NO_CLAIMS})
        self.save_terminal()
        self.native = native()
        save(self.attempt/"sana-numerics.json",self.native)
        (self.attempt/"isolation.jsonl").write_text(json.dumps(dict(status="exclusive_observed",foreign=[],owned=[],
            devices=["GPU-test"],allowed_pgid=900))+"\n")
        old_session = dict(session_id="old-session",stage="sana_numerics",identity=dict(cache_local_binding=
            dict(mathematical_source_files=mathematics,mathematical_source_sha256=digest(mathematics),bindings={})))
        old_terminal["charge"] = old_session
        self.binding["original"]["terminal"] = save(old_attempt.parent/"terminal.json",old_terminal)
        self.binding["original"]["supervisor"] = save(old_attempt/"supervisor.json",old_terminal)
        self.parents[-1].update(save(Path(self.parents[-1]["path"]),
            dict(active=None,used_gpu_seconds=10.,sessions=[old_session])))
        self.rebind()

    def save_terminal(self):
        save(self.output/"sana_numerics/terminal.json",self.terminal)
        save(self.attempt/"supervisor.json",self.terminal)

    def rebind(self):
        admission = save(self.output/"retry-binding.json",self.binding)
        self.study["cache_local_binding"]["bindings"]["numerics_retry_admission"] = admission
        save(self.output/"study.json",self.study)
        save(self.attempt/"study.json",self.study)
        save(self.ledger,self.budget)
        self.save_terminal()


class RetryReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def collect(self):
        return report.collect(self.fixture.output,self.fixture.ledger)

    def test_complete_comparisons_do_not_grant_serving_quality(self):
        value = self.collect()
        self.assertEqual(value["integrity_errors"],[])
        self.assertEqual(value["status"],"completed")
        self.assertTrue(value["diagnostic_complete"])
        self.assertEqual(value["retry"]["counts"]["verified_comparisons"],12)
        self.assertEqual(value["retry"]["counts"]["lpips_passed"],0)
        self.assertEqual(value["original"]["counts"]["failed_units"],8)
        self.assertEqual(value["original"]["counts"]["unverified_comparisons"],12)
        self.assertEqual(value["combined_denominators"]["required_comparisons"],24)
        self.assertFalse(value["online_qualified"])

    def test_passed_native_with_old_eight_failures_is_unverified(self):
        save(self.fixture.attempt/"sana-numerics.json",native(True))
        value = self.collect()
        self.assertEqual(value["integrity_status"],"passed")
        self.assertEqual(value["status"],"unverified")
        self.assertFalse(value["diagnostic_complete"])
        self.assertEqual(value["combined_denominators"]["failed_units"],16)
        self.assertEqual(value["combined_denominators"]["unverified_comparisons"],24)

    def test_partial_failure_retains_missing_comparison_denominator(self):
        value = self.fixture.native
        row = value["rows"][11]
        row.update(status="failed",comparisons=[],error="fixture compiler unavailable")
        value.update(completed_units=11,failed_units=1,compared_images=11)
        save(self.fixture.attempt/"sana-numerics.json",value)
        result = self.collect()
        self.assertEqual(result["status"],"partial")
        self.assertEqual(result["retry"]["counts"]["verified_comparisons"],11)
        self.assertEqual(result["retry"]["counts"]["unverified_comparisons"],1)

    def test_excel_readback_and_no_history_or_raw_changes(self):
        old = file_digest(self.fixture.original_native["path"])
        result = report.write_report(self.fixture.output,self.fixture.ledger)
        self.assertTrue(result["integrity_valid"])
        self.assertTrue(result["diagnostic_complete"])
        self.assertEqual(result["verified_comparisons"],12)
        self.assertEqual(result["excel"]["status"],"verified")
        self.assertEqual(file_digest(self.fixture.original_native["path"]),old)
        self.assertTrue((self.fixture.attempt/"sana-numerics.json").exists())
        self.assertEqual(read_json(result["audit"])["status"],"passed")
        self.assertEqual(result["originals_deleted"],0)

    def test_parent_mutation_and_active_parent_are_not_admitted(self):
        path = Path(self.fixture.parents[0]["path"])
        save(path,dict(active=dict(pid=123),used_gpu_seconds=10.))
        value = self.collect()
        self.assertEqual(value["status"],"unverified")
        self.assertTrue(value["integrity_errors"])
        self.assertEqual(value["original"]["counts"]["failed_units"],8)

    def test_active_or_missing_settlement_stays_unverified(self):
        self.fixture.budget["active"] = dict(session_id="active")
        save(self.fixture.ledger,self.fixture.budget)
        self.assertEqual(self.collect()["status"],"unverified")
        self.fixture.ledger.unlink()
        value = self.collect()
        self.assertEqual(value["status"],"unverified")
        self.assertEqual(value["retry"]["counts"]["verified_comparisons"],12)
        self.assertFalse(value["diagnostic_complete"])

    def test_duplicate_session_and_undercharge_rejected(self):
        self.fixture.budget["sessions"].append(copy.deepcopy(self.fixture.session))
        save(self.fixture.ledger,self.fixture.budget)
        self.assertEqual(self.collect()["status"],"unverified")
        self.fixture.budget["sessions"].pop()
        self.fixture.session["gpu_seconds"] = 1.
        self.fixture.budget["used_gpu_seconds"] = 1.
        self.fixture.budget["stage_used_gpu_seconds"]["sana_numerics"] = 1.
        save(self.fixture.ledger,self.fixture.budget)
        self.fixture.save_terminal()
        self.assertEqual(self.collect()["status"],"unverified")

    def test_terminal_divergence_source_tamper_and_claims_rejected(self):
        terminal = dict(self.fixture.terminal,status="completed")
        save(self.fixture.output/"sana_numerics/terminal.json",terminal)
        self.assertEqual(self.collect()["status"],"unverified")
        self.fixture.save_terminal()
        self.fixture.native["online_qualified"] = True
        save(self.fixture.attempt/"sana-numerics.json",self.fixture.native)
        self.assertEqual(self.collect()["status"],"unverified")
        self.fixture.native["online_qualified"] = False
        save(self.fixture.attempt/"sana-numerics.json",self.fixture.native)
        path = self.fixture.manifest_path.parent/"python/ditserve/optimizations/next_numerics.py"
        path.write_text("changed source")
        self.assertEqual(self.collect()["status"],"unverified")

    def test_isolation_and_cleanup_failures_rejected(self):
        (self.fixture.attempt/"isolation.jsonl").write_text(json.dumps(dict(status="contended",foreign=[1]))+"\n")
        self.assertEqual(self.collect()["status"],"unverified")
        self.fixture.terminal["final_occupancy"]["owned"] = [123]
        self.fixture.save_terminal()
        self.assertEqual(self.collect()["status"],"unverified")

    def test_narrow_final_audit_only_chain_failure_is_accepted(self):
        def change(reason):
            chain = save(self.fixture.chain_path,dict(status="needs_attention",reason=reason,started_unix=1.))
            self.fixture.binding["parent_study"]["chain"] = chain
            path = Path(self.fixture.binding["parent_study"]["finalizer_receipt"]["path"])
            receipt = read_json(path)
            receipt["chain_final"] = chain
            self.fixture.binding["parent_study"]["finalizer_receipt"] = save(path,receipt)
            self.fixture.rebind()
        change("RuntimeError: final independent integrity audit did not pass")
        self.assertEqual(self.collect()["status"],"completed")
        change("RuntimeError: unknown worker failure")
        self.assertEqual(self.collect()["status"],"unverified")

    def test_wrong_matrix_counter_boundary_or_lpips_flag_rejected(self):
        value = self.fixture.native
        value["compared_images"] = 0
        save(self.fixture.attempt/"sana-numerics.json",value)
        self.assertEqual(self.collect()["status"],"unverified")
        value["compared_images"] = 12
        comparison = value["rows"][4]["comparisons"][0]
        comparison["trajectory"]["components"][0]["boundary"] = "invented"
        save(self.fixture.attempt/"sana-numerics.json",value)
        self.assertEqual(self.collect()["status"],"unverified")
        comparison["trajectory"]["components"][0]["boundary"] = report._boundaries()[0]
        comparison["output"]["passed"] = True
        save(self.fixture.attempt/"sana-numerics.json",value)
        self.assertEqual(self.collect()["status"],"unverified")

    def test_inconsistent_single_frame_quality_is_not_verified(self):
        comparison = self.fixture.native["rows"][4]["comparisons"][0]
        comparison["output"]["metrics"]["lpips_per_frame"] = [.03]
        save(self.fixture.attempt/"sana-numerics.json",self.fixture.native)
        value = self.collect()
        self.assertEqual(value["integrity_errors"],[])
        self.assertEqual(value["status"],"partial")
        self.assertEqual(value["retry"]["counts"]["completed_comparisons"],12)
        self.assertEqual(value["retry"]["counts"]["verified_comparisons"],11)

    def test_real_cpu_producer_boundary_and_quality_schema_compatibility(self):
        import test_optimizations_next_numerics as fixtures
        from ditserve.optimizations import next_numerics
        directory = Path(self.temp.name)/"cpu-producer"
        directory.mkdir()
        context = SimpleNamespace(sana=fixtures.Adapter(),output=directory,
            config=dict(direction1=dict(prompts=["one","two"]),quality=dict(rtol=1e-5,atol=1e-6,image_lpips=.01)),
            check_budget=lambda:None,quality_evaluator=fixtures.quality)
        data = next_numerics.run_numerics(context)
        value = report._matrix(data,data["inputs"],"cpu_fixture")
        self.assertEqual(value["counts"]["completed_units"],12)
        self.assertEqual(value["counts"]["pipeline_generated_images"],16)
        self.assertEqual(value["counts"]["completed_comparisons"],12)
        self.assertEqual(len(value["boundaries"]),12*29)
        # Tiny CPU image fixture deliberately lacks production 1024² evidence.
        self.assertEqual(value["counts"]["verified_comparisons"],0)
        self.assertFalse(value["full_comparison"])

    def test_real_signed_zero_value_equality_does_not_imply_hash_equality(self):
        import torch
        from ditserve.optimizations import next_numerics
        positive = torch.tensor([0.],dtype=torch.bfloat16)
        negative = torch.tensor([-0.],dtype=torch.bfloat16)
        number = next_numerics._error(positive,negative,1e-5,1e-6)
        self.assertTrue(number["exact"])
        reference = next_numerics._tensor_summary(positive)
        candidate = next_numerics._tensor_summary(negative)
        self.assertNotEqual(reference["sha256"],candidate["sha256"])
        component = self.fixture.native["rows"][4]["comparisons"][0]["trajectory"]["components"][0]
        component.update(numerical=number,reference=reference,candidate=candidate)
        save(self.fixture.attempt/"sana-numerics.json",self.fixture.native)
        value = self.collect()
        self.assertEqual(value["integrity_errors"],[])
        self.assertTrue(value["retry"]["boundaries"][0]["exact"])
        self.assertFalse(value["retry"]["boundaries"][0]["bitwise_sha_equal"])


if __name__ == "__main__":
    unittest.main()
