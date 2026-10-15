"""CPU-only finite phase orchestration, budget and evidence contracts."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.directions.budget import BudgetStop
from ditserve.optimizations import wan_phases as phases
REAL_ARTIFACT = phases._artifact
REAL_RESUME_CHECK = phases._resume_check


def cache_events(reuse_indices=(8, 10)):
    return [dict(kind="step", step_index=index, completed_steps=index+1,
                 path="reuse" if index in reuse_indices else "compute",
                 reused_forwards=2 if index in reuse_indices else 0) for index in range(32)]


def cache_resume_result(events=None):
    events = cache_events() if events is None else events
    interruptions = [dict(step=index) for index in (0, 8, 16)]
    return dict(passed=True, interruptions=interruptions,
                cache_paths=phases._cache_resume_paths(events, events, interruptions),
                dispatch_observation=dict(host_dispatch_fraction=None))


class Budget:
    def __init__(self, remaining=100000):
        self.left, self.units = remaining, []
    def __call__(self):
        pass
    def remaining(self):
        return self.left
    def ensure_unit(self, seconds):
        self.units.append(seconds)


class Array:
    def __init__(self, value):
        self.value = value
        self.shape = (1, 81, 480, 832, 3)
    def copy(self):
        return Array(self.value)


class Backend:
    def __init__(self, worker, *, compiled=False, group_size=1):
        self.compiled, self.group_size = compiled, group_size
        self.name = "inductor-block%d" % group_size if compiled else "eager"
        self.frozen = False
    def freeze_compilation(self):
        self.frozen = True
    def report(self):
        return dict(name=self.name, frozen=self.frozen)


class Executor:
    def __init__(self, worker, backend=None):
        self.worker, self.backend = worker, backend or Backend(worker)


class Bank:
    def __init__(self, worker):
        self.entries = {}


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.context = SimpleNamespace(output=Path(self.temp.name)/"phase", run_root=Path(self.temp.name),
            config=dict(direction1=dict(prompts=["kettle", "pendulum"]), repeats=3),
            worker=SimpleNamespace(attention="flash"), check_budget=Budget(), metadata={})
        self.calls, self.resumes = [], []
        def generate(context, evidence, bank, item, executor, *, cache=False, native_cache=False):
            self.calls.append(dict(input_id=item["input_id"], split=item["split"],cache=cache,
                native=native_cache, backend=executor.backend.name,executor_class=type(executor).__name__))
            value = (item["input_id"],cache)
            service = dict(ms=80000 if cache or executor.backend.compiled else 100000,
                output_sha256=str(value), generation_completed=True, reuse_steps=[9,11] if cache else [])
            if cache:
                service["cache_paths"] = phases._cache_paths(cache_events())
            return Array(value), service
        def resume(context, evidence, bank, item, executor, **kwargs):
            self.resumes.append((executor.backend.name,item["input_id"]))
            return (cache_resume_result() if kwargs.get("cache") else
                    dict(passed=True,dispatch_observation=dict(host_dispatch_fraction=None)))
        replacements = dict(FrozenPromptBank=Bank, BlockBackend=Backend,
            NativeWanExecutor=Executor, WanCompiledExecutor=Executor, PairedCacheExecutor=Executor,
            _generate=generate, _resume_check=resume,
            _artifact=lambda context,name,array: dict(path=name,raw_available=True),
            _quality=lambda reference,candidate: dict(passed=True,metrics=dict(lpips_mean=.01)),
            numerical_error=lambda actual,expected: dict(passed=actual.value==expected.value))
        for name,value in replacements.items():
            change=patch.object(phases,name,value)
            change.start()
            self.addCleanup(change.stop)

    def test_four_inputs_ab_ba_one_repeat_per_arm_and_no_host_gate_invention(self):
        result=phases.run_wan(self.context)
        self.assertTrue(result["local_qualified"])
        self.assertFalse(result["scientific_qualified"])
        self.assertEqual(result["single_block"]["n_per_arm"],8)
        self.assertEqual(result["single_block"]["repeats"],1)
        self.assertEqual(len(self.calls),16)
        self.assertEqual(len(self.resumes),8)
        expected_inputs=["p0-s101","p0-s102","p1-s101","p1-s102"]
        self.assertEqual(self.resumes,
            [(backend,item) for backend in ("inductor-block1","eager") for item in expected_inputs])
        self.assertEqual(result["resume_backend_order"],["inductor-block1","eager"])
        self.assertEqual({v["input_id"] for v in self.calls},
                         {"p0-s101","p0-s102","p1-s101","p1-s102"})
        self.assertEqual(set(self.context.executors),{"eager","inductor-block1"})
        self.assertTrue(self.context.executors["inductor-block1"].backend.frozen)
        self.assertEqual(result["optional_group4"]["reason"],"host_control_fraction_not_measured")
        self.assertIsNone(result["optional_group4"]["host_dispatch_fraction"])

    def test_group4_requires_actual_host_lower_bound_and_reuses_eager_quality_reference(self):
        original=phases._generate
        def generate(*args,**kwargs):
            array,row=original(*args,**kwargs)
            if args[4].backend.group_size==4:
                row["ms"]=60000
            return array,row
        check=dict(passed=True,dispatch_observation=dict(pure_host_control_ms=10.,quantum_total_wall_ms=100.))
        with patch.object(phases,"_resume_check",return_value=check),patch.object(phases,"_generate",side_effect=generate):
            result=phases.run_wan(self.context)
        self.assertEqual(result["selected_backend"],"inductor-block4")
        self.assertTrue(result["optional_group4"]["qualified"])
        summary=result["optional_group4"]["result"]
        self.assertEqual(summary["baseline_backend"],"inductor-block1")
        self.assertEqual(summary["quality_reference"],"original eager same-input outputs")
        self.assertEqual(summary["n_per_arm"],8)
        self.assertEqual(len(self.calls),32)

    def test_group4_host_lower_bound_below_five_percent_is_not_triggered(self):
        check=dict(passed=True,dispatch_observation=dict(pure_host_control_ms=4.,quantum_total_wall_ms=100.))
        with patch.object(phases,"_resume_check",return_value=check):
            result=phases.run_wan(self.context)
        self.assertEqual(result["selected_backend"],"inductor-block1")
        self.assertEqual(result["optional_group4"]["reason"],"host_control_lower_bound_below_trigger")
        self.assertNotIn("inductor-block4",self.context.executors)

    def test_optional_group4_budget_limit_preserves_primary_candidate(self):
        self.context.check_budget.left=100
        check=dict(passed=True,dispatch_observation=dict(pure_host_control_ms=10.,quantum_total_wall_ms=100.))
        with patch.object(phases,"_resume_check",return_value=check):
            result=phases.run_wan(self.context)
        self.assertTrue(result["local_qualified"])
        self.assertEqual(result["selected_backend"],"inductor-block1")
        self.assertEqual(result["optional_group4"]["status"],"not_measured_budget_limited")

    def test_optional_group4_implementation_failure_preserves_primary_and_failed_unit(self):
        def resume(context,evidence,bank,item,executor,**kwargs):
            if executor.backend.group_size==4:
                raise RuntimeError("group4 compile error")
            return dict(passed=True,dispatch_observation=dict(pure_host_control_ms=10.,quantum_total_wall_ms=100.))
        with patch.object(phases,"_resume_check",side_effect=resume):
            result=phases.run_wan(self.context)
        self.assertTrue(result["local_qualified"])
        self.assertEqual(result["selected_backend"],"inductor-block1")
        self.assertEqual(result["optional_group4"]["status"],"failed")
        self.assertEqual(result["optional_group4"]["active_unit"]["backend"],"inductor-block4")
        self.assertIn("group4 compile error",result["optional_group4"]["error"])

    def test_optional_group4_quality_failure_preserves_primary_candidate(self):
        count=[0]
        def quality(*unused):
            count[0]+=1
            return dict(passed=count[0]<=16,metrics=dict(lpips_mean=.01 if count[0]<=16 else .1))
        check=dict(passed=True,dispatch_observation=dict(pure_host_control_ms=10.,quantum_total_wall_ms=100.))
        with patch.object(phases,"_resume_check",return_value=check),patch.object(phases,"_quality",side_effect=quality):
            result=phases.run_wan(self.context)
        self.assertTrue(result["local_qualified"])
        self.assertEqual(result["selected_backend"],"inductor-block1")
        self.assertEqual(result["optional_group4"]["status"],"failed")
        self.assertFalse(result["optional_group4"]["result"]["quality_passed"])

    def test_artifact_namespace_includes_phase_and_attempt(self):
        import numpy as np
        self.context.config["artifact_root"]=str(Path(self.temp.name)/"artifacts")
        paths=[]
        for phase in ("wan_compile","cache_k4"):
            self.context.output=self.context.run_root/phase/"attempt00"
            paths.append(REAL_ARTIFACT(self.context,"same-failure-name",np.zeros((1,2,2,3)))["path"])
        self.assertNotEqual(paths[0],paths[1])
        self.assertIn("/wan_compile/attempt00/",paths[0])
        self.assertIn("/cache_k4/attempt00/",paths[1])
        self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_failed_resume_is_retained_and_not_promoted(self):
        with patch.object(phases,"_resume_check",return_value=dict(passed=False)):
            result=phases.run_wan(self.context)
        self.assertEqual(result["status"],"failed")
        self.assertFalse(result["local_qualified"])
        self.assertEqual(len(result["rows"]),1)
        self.assertFalse(result["rows"][0]["result"]["passed"])
        self.assertEqual(self.calls,[])

    def test_phase_entry_points_use_native_for_every_cache_off_baseline(self):
        class Original(Executor):
            pass
        class Wrapped(Executor):
            pass
        self.context.check_budget.left=100
        with patch.object(phases,"NativeWanExecutor",Original), \
             patch.object(phases,"WanCompiledExecutor",Wrapped):
            for name,run in (("wan",phases.run_wan),("cache",phases.run_cache)):
                self.context.output=Path(self.temp.name)/name
                self.calls.clear()
                run(self.context)
                baseline=[row for row in self.calls if not row["cache"] and row["backend"]=="eager"]
                self.assertTrue(baseline)
                self.assertTrue(all(row["executor_class"]=="Original" for row in baseline))
            self.assertIsInstance(self.context.executors["eager"],Original)

    def test_budget_exception_preserves_completed_generation_and_active_arm(self):
        original=phases._generate
        def generate(*args,**kwargs):
            if self.calls:
                raise BudgetStop("between paired generations")
            return original(*args,**kwargs)
        with patch.object(phases,"_generate",side_effect=generate):
            with self.assertRaises(BudgetStop):
                phases.run_wan(self.context)
        result=json.loads((self.context.output/"wan-compile.json").read_text())
        self.assertEqual(result["status"],"stopped")
        self.assertEqual(result["active_unit"]["arm"],"candidate")
        self.assertTrue(result["rows"][-1]["service"]["generation_completed"])
        self.assertNotIn("complete",result["rows"][-1])
        self.assertFalse(result["local_qualified"])

    def test_cache_three_sensitive_inputs_fixed_eager_and_unverified_full_matrix(self):
        self.context.check_budget.left=100
        result=phases.run_cache(self.context)
        self.assertTrue(result["local_qualified"])
        self.assertFalse(result["cache_quality_qualified"])
        self.assertFalse(result["full_qualification_completed"])
        self.assertEqual(result["full_qualification"]["status"],"not_measured_budget_limited")
        self.assertEqual({v["input_id"] for v in self.calls},{"origami-101","kettle-101","kettle-102"})
        self.assertEqual(len(self.calls),18)  # 6 original/K4, then 12 off/paired.
        self.assertEqual(result["screen"]["quality_checked"],12)
        self.assertEqual(len(self.resumes),3)
        self.assertTrue(all(v["backend"]=="eager" for v in self.calls))
        self.assertFalse(self.context.cache_executor.backend.compiled)
        coverage=result["cache_path_coverage"]
        self.assertTrue(coverage["passed"])
        self.assertEqual(coverage["counts"]["resumed_k4"],dict(full=90,reuse=6,mixed=0,unknown=0))
        self.assertEqual(len(coverage["samples"]),3)

    def test_empty_equal_reuse_lists_do_not_qualify_missing_path_observations(self):
        original=phases._generate
        def generate(*args,**kwargs):
            array,service=original(*args,**kwargs)
            service.update(reuse_steps=[],cache_paths=phases._cache_paths([]))
            return array,service
        with patch.object(phases,"_generate",side_effect=generate), \
             patch.object(phases,"_resume_check",side_effect=lambda *a,**kw: cache_resume_result([])):
            result=phases.run_cache(self.context)
        self.assertEqual((result["status"],result["decision"]),("stopped","未验证"))
        self.assertFalse(result["local_qualified"])
        self.assertFalse(result["cache_quality_qualified"])
        self.assertFalse(result["cache_path_coverage"]["complete"])
        comparisons=[row for row in result["rows"] if row["kind"]=="original_split_comparison"]
        self.assertEqual(len(comparisons),3)
        self.assertTrue(all(row["numerical"]["passed"] and row["cache_paths_equal"] for row in comparisons))
        self.assertTrue(all(not row["path_records_valid"] for row in comparisons))
        self.assertEqual(len(self.calls),6)
        self.assertNotIn("screen",result)

    def test_all_full_resumed_trajectory_remains_unverified_even_if_original_reused(self):
        with patch.object(phases,"_resume_check",side_effect=lambda *a,**kw: cache_resume_result(cache_events(()))):
            result=phases.run_cache(self.context)
        coverage=result["cache_path_coverage"]
        self.assertTrue(coverage["complete"])
        self.assertFalse(coverage["passed"])
        self.assertEqual(coverage["missing_resumed_paths"],["reuse"])
        self.assertGreater(coverage["counts"]["original_atomic"]["reuse"],0)
        self.assertEqual(coverage["counts"]["resumed_k4"]["reuse"],0)
        self.assertEqual(result["decision"],"未验证")
        self.assertFalse(result["local_qualified"])
        self.assertEqual(len(self.calls),6)

    def test_sana_insertion_path_cross_product_is_descriptive_not_an_extra_gate(self):
        self.context.check_budget.left=100
        # Actual reuse exists elsewhere in the fixed trajectory, outside 0/8/16.
        with patch.object(phases,"_resume_check",side_effect=lambda *a,**kw: cache_resume_result(cache_events((9,11)))):
            result=phases.run_cache(self.context)
        self.assertTrue(result["cache_path_coverage"]["passed"])
        self.assertTrue(result["local_qualified"])
        insertion=result["cache_path_coverage"]["sana_interruptions"]
        self.assertEqual(insertion["counts"],dict(full=9,reuse=0,mixed=0,unknown=0))
        self.assertEqual(insertion["unverified_paths"],["reuse"])
        self.assertFalse(insertion["full_reuse_cross_product_required"])
        self.assertEqual(len(self.calls),18)

    def test_mismatched_resume_paths_cannot_pass_on_matching_final_numbers(self):
        def resume(*args,**kwargs):
            result=cache_resume_result()
            result["cache_paths"]=phases._cache_resume_paths(cache_events((8,10)),cache_events((9,11)),
                                                            result["interruptions"])
            return result
        with patch.object(phases,"_resume_check",side_effect=resume):
            result=phases.run_cache(self.context)
        self.assertFalse(result["cache_path_coverage"]["passed"])
        self.assertFalse(result["local_qualified"])
        self.assertEqual(result["decision"],"未验证")
        self.assertTrue(all(row["result"]["passed"] for row in result["rows"] if row["kind"]=="same_backend_resume"))
        self.assertEqual(len(self.calls),6)

    def test_full_cache_qualification_preserves_every_frozen_pair(self):
        result=phases.run_cache(self.context)
        self.assertTrue(result["cache_quality_qualified"])
        self.assertTrue(result["full_qualification_completed"])
        self.assertFalse(result["scientific_qualified"])
        rows=[r for r in result["rows"] if r["kind"]=="cache_qualification_pair"]
        self.assertEqual(len(rows),20)
        self.assertEqual(sum(r["split"]=="dev" for r in rows),16)
        self.assertEqual(sum(r["split"]=="holdout" for r in rows),4)
        for order in (0,1):
            self.assertEqual(len({r["input_id"] for r in rows if r["split"]=="dev" and r["order"]==order}),8)
        self.assertTrue(all(r["order"]==0 for r in rows if r["split"]=="holdout"))
        self.assertEqual(len(self.calls),58)
        self.assertEqual(result["full_qualification"]["required_pairs"],20)
        self.assertEqual(result["full_qualification"]["remaining_pairs"],0)

    def test_cache_budget_admits_only_complete_pairs_and_keeps_required_twenty(self):
        self.assertEqual(phases.DEFAULTS["cache_full_qualification"],"complete_pairs_until_budget_limit")
        original_generate, original_screen = phases._generate, phases._cache_performance
        def screen(*args,**kwargs):
            result=original_screen(*args,**kwargs)
            self.context.check_budget.left=650.  # Two 180s pairs fit a 310s admission bound.
            return result
        def generate(*args,**kwargs):
            array,service=original_generate(*args,**kwargs)
            if args[1].data["active_unit"]["kind"]=="cache_qualification":
                self.context.check_budget.left-=service["ms"]/1000
            return array,service
        with patch.object(phases,"_cache_performance",side_effect=screen), \
             patch.object(phases,"_generate",side_effect=generate):
            result=phases.run_cache(self.context)
        full=result["full_qualification"]
        self.assertTrue(result["local_qualified"])
        self.assertEqual(result["status"],"passed")
        self.assertTrue(result["screen_local_qualified"])
        self.assertEqual(result["decision"],"未验证")
        self.assertEqual(full["status"],"partial")
        self.assertEqual((full["pairs"],full["required_pairs"],full["remaining_pairs"]),(2,20,18))
        self.assertEqual((full["dev_pairs"],full["heldout_pairs"]),(2,0))
        self.assertEqual((full["required_dev_pairs"],full["required_heldout_pairs"]),(16,4))
        self.assertEqual(full["required_seconds"],310.)
        self.assertEqual(full["remaining_seconds"],290.)
        self.assertFalse(full["complete"])
        self.assertFalse(full["qualified"])
        self.assertFalse(result["cache_quality_qualified"])
        self.assertFalse(result["full_qualification_completed"])
        rows=[r for r in result["rows"] if r["kind"]=="cache_qualification"]
        pairs=[r for r in result["rows"] if r["kind"]=="cache_qualification_pair"]
        self.assertEqual(len(rows),4)
        self.assertEqual(len(pairs),2)
        self.assertTrue(all(r["complete"] for r in rows+pairs))
        self.assertEqual(full["next_unit"]["input_id"],phases._cache_inputs("dev")[2]["input_id"])
        self.assertIsNone(result["active_unit"])
        self.assertEqual(len(self.calls),22)
        from ditserve.optimizations.cache_online import run_cache_online
        with patch("ditserve.optimizations.cache_online.load_frozen_traces",
                   side_effect=AssertionError("partial quality cannot start online")):
            result=run_cache_online(self.context,result)
        self.assertFalse(result["online_qualified"])

    def test_cache_exact_deadline_between_pairs_preserves_completed_screen(self):
        original=phases._cache_performance
        def screen(*args,**kwargs):
            result=original(*args,**kwargs)
            self.context.check_budget.remaining=lambda: (_ for _ in ()).throw(BudgetStop("phase allowance reached"))
            return result
        with patch.object(phases,"_cache_performance",side_effect=screen):
            result=phases.run_cache(self.context)
        self.assertTrue(result["local_qualified"])
        self.assertEqual(result["full_qualification"]["remaining_seconds"],0.)
        self.assertEqual(result["full_qualification"]["status"],"not_measured_budget_limited")
        self.assertEqual(result["full_qualification"]["required_pairs"],20)
        self.assertEqual(result["full_qualification"]["remaining_pairs"],20)
        self.assertEqual(len(self.calls),18)

    def test_cache_interrupt_inside_admitted_pair_keeps_original_hard_stop_path(self):
        original=phases._generate
        def generate(*args,**kwargs):
            unit=args[1].data["active_unit"]
            if unit["kind"]=="cache_qualification" and unit["arm"]=="paired":
                raise BudgetStop("phase allowance reached")
            return original(*args,**kwargs)
        with patch.object(phases,"_generate",side_effect=generate):
            with self.assertRaises(BudgetStop):
                phases.run_cache(self.context)
        result=json.loads((self.context.output/"cache-k4.json").read_text())
        self.assertEqual(result["status"],"stopped")
        self.assertTrue(result["local_gate_passed_before_interruption"])
        self.assertFalse(result["cache_quality_qualified"])
        rows=[r for r in result["rows"] if r["kind"]=="cache_qualification"]
        self.assertEqual(len(rows),1)
        self.assertTrue(rows[0]["service"]["generation_completed"])
        self.assertNotIn("complete",rows[0])
        self.assertEqual(result["active_unit"]["arm"],"paired")

    def test_cache_supervisor_loss_is_not_reclassified_as_pair_budget_limit(self):
        original=phases._cache_performance
        def screen(*args,**kwargs):
            result=original(*args,**kwargs)
            self.context.check_budget.remaining=lambda: (_ for _ in ()).throw(BudgetStop("supervisor exited"))
            return result
        with patch.object(phases,"_cache_performance",side_effect=screen):
            with self.assertRaisesRegex(BudgetStop,"supervisor exited"):
                phases.run_cache(self.context)
        result=json.loads((self.context.output/"cache-k4.json").read_text())
        self.assertEqual(result["status"],"stopped")
        self.assertTrue(result["local_gate_passed_before_interruption"])
        self.assertNotIn("full_qualification",result)

    def test_full_cache_quality_failure_stays_in_denominator(self):
        count=[0]
        def quality(*unused):
            count[0]+=1
            return dict(passed=count[0]!=32,metrics=dict(lpips_mean=.1 if count[0]==32 else .01))
        with patch.object(phases,"_quality",side_effect=quality):
            result=phases.run_cache(self.context)
        self.assertTrue(result["full_qualification_completed"])
        self.assertFalse(result["cache_quality_qualified"])
        self.assertTrue(result["screen_local_qualified"])
        self.assertFalse(result["local_qualified"])
        self.assertEqual(result["decision"],"当前配置无收益")
        self.assertIn("qualification failed: quality",result["reason"])
        self.assertEqual(result["selected_backend"],"eager")
        self.assertEqual(result["full_qualification"]["pairs"],20)
        rows=[r for r in result["rows"] if r["kind"]=="cache_qualification_pair"]
        self.assertEqual(sum(not r["quality"]["passed"] for r in rows),1)
        self.assertFalse(rows[-1]["quality"]["passed"])

    def test_complete_cache_speed_failure_is_negative_while_screen_is_retained(self):
        original=phases._generate
        def generate(*args,**kwargs):
            array,service=original(*args,**kwargs)
            unit=args[1].data["active_unit"]
            if unit["kind"]=="cache_qualification" and unit["arm"]=="paired":
                service["ms"]=120000.
            return array,service
        with patch.object(phases,"_generate",side_effect=generate):
            result=phases.run_cache(self.context)
        self.assertTrue(result["screen_local_qualified"])
        self.assertTrue(result["screen"]["performance_passed"])
        self.assertTrue(result["full_qualification"]["quality_passed"])
        self.assertFalse(result["full_qualification"]["performance_passed"])
        self.assertEqual(result["full_qualification"]["pairs"],20)
        self.assertTrue(result["full_qualification_completed"])
        self.assertFalse(result["local_qualified"])
        self.assertFalse(result["cache_quality_qualified"])
        self.assertEqual(result["decision"],"当前配置无收益")
        self.assertIn("qualification failed: performance",result["reason"])
        self.assertEqual(result["selected_backend"],"eager")

    def test_between_unit_stop_does_not_relabel_completed_row(self):
        evidence=phases._Evidence(self.context,"between")
        evidence.begin(kind="first")
        row=dict(kind="first",complete=True,passed=True)
        evidence.append(row)
        evidence.interrupted(BudgetStop("next unit does not fit"))
        data=json.loads(evidence.path.read_text())
        self.assertEqual(data["rows"],[row])
        self.assertIsNone(data["active_unit"])
        self.assertEqual(data["status"],"stopped")

    def test_begin_persists_admitted_unit_before_execution(self):
        evidence=phases._Evidence(self.context,"admission")
        evidence.begin(kind="correctness",input_id="p0-s101")
        saved=json.loads(evidence.path.read_text())
        self.assertTrue(saved["active_unit"]["started"])


class CachePathEvidenceTests(unittest.TestCase):
    def test_records_require_unique_ordered_commits_and_consistent_paired_paths(self):
        valid=phases._cache_paths(cache_events())
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["counts"],dict(full=30,reuse=2,mixed=0,unknown=0))
        malformed=[]
        malformed.append(cache_events()[:-1])
        duplicate=cache_events(); duplicate[10]=dict(duplicate[9]); malformed.append(duplicate)
        reordered=cache_events(); reordered[0],reordered[1]=reordered[1],reordered[0]; malformed.append(reordered)
        mixed=cache_events(); mixed[8].update(path="mixed",reused_forwards=1); malformed.append(mixed)
        unknown=cache_events(); unknown[8].pop("path"); malformed.append(unknown)
        inconsistent=cache_events(); inconsistent[8]["reused_forwards"]=0; malformed.append(inconsistent)
        for events in malformed:
            with self.subTest(events=events):
                observed=phases._cache_paths(events)
                self.assertFalse(observed["valid"])
                self.assertFalse(observed["full_reuse_observed"])

    def test_interruption_is_resolved_only_after_observed_commit(self):
        interruptions=[dict(step=8)]
        partial=phases._cache_resume_paths(cache_events()[:8],cache_events()[:8],interruptions,8)
        self.assertTrue(partial["valid"])
        self.assertEqual(interruptions[0]["resolved_step_path"],"unknown")
        committed=phases._cache_resume_paths(cache_events()[:9],cache_events()[:9],interruptions,9)
        self.assertTrue(committed["valid"])
        self.assertEqual(interruptions[0]["resolved_step_path"],"reuse")
        self.assertIn("mid-prefix before cache decision",interruptions[0]["path_scope"])

    def make_runtime(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        states=[]
        def prepare(spec):
            state=SimpleNamespace(step_index=0,latents=Array("same"),scheduler=None,
                generator=SimpleNamespace(get_state=lambda:101),residency="gpu",events=[])
            states.append(state)
            return state
        bank=SimpleNamespace(prepare=prepare,warm=lambda spec:None)
        torch=SimpleNamespace(isfinite=lambda value:SimpleNamespace(all=lambda:True),
                              allclose=lambda *a,**kw:True,equal=lambda *a:True)
        worker=SimpleNamespace(torch=torch,finish=lambda state:Array("same"),_sync=lambda:None,
                               release=lambda state:setattr(state,"residency","released"))
        class FakeExecutor:
            backend=SimpleNamespace(name="eager-paired")
            supports_host_observation=False
            def attach(self,state):
                return SimpleNamespace(base=state,closed=False,branch="cond",phase="prefix",next_block=0)
            def advance(self,continuation,quantum,**unused):
                state=continuation.base
                if quantum==60 or continuation.next_block==4:
                    state.events.append(cache_events()[state.step_index])
                    state.step_index+=1
                    continuation.next_block=0
                else:
                    continuation.next_block=4
                return dict(ms=1.,block_cuda_ms=[],extra_state_bytes=0)
            def close(self,continuation):
                continuation.closed=True
        calls=[]
        def image(*args,**kwargs):
            calls.append(states[1].step_index)
            return None,dict(ms=1.)
        context=SimpleNamespace(worker=worker,sana=SimpleNamespace(generate=image),
            check_budget=Budget(),config={},metadata={},output=Path(temp.name)/"phase",run_root=Path(temp.name))
        evidence=phases._Evidence(context,"paths")
        evidence.begin(kind="correctness")
        item=dict(input_id="fake101",prompt="fake",seed=101)
        return context,evidence,bank,item,FakeExecutor(),calls

    def test_real_resume_observer_records_each_commit_and_fixed_sana_resolution(self):
        context,evidence,bank,item,executor,calls=self.make_runtime()
        with patch.object(phases,"solver_matches",return_value=True), \
             patch.object(phases,"_cache_equal",return_value=True), \
             patch.object(phases,"_artifact",side_effect=lambda c,n,a:dict(path=n)), \
             patch.object(phases,"numerical_error",return_value=dict(passed=True)):
            result=REAL_RESUME_CHECK(context,evidence,bank,item,executor,cache=True)
        self.assertTrue(result["passed"])
        self.assertEqual(calls,[0,8,16])
        self.assertEqual(len(result["steps"]),32)
        self.assertTrue(all(row["cache_paths_equal"] for row in result["steps"]))
        self.assertEqual(result["cache_paths"]["k4"]["counts"],dict(full=30,reuse=2,mixed=0,unknown=0))
        self.assertEqual([row["resolved_step_path"] for row in result["interruptions"]],["full","reuse","full"])
        self.assertTrue(all(row["next_block"]==4 and row["phase"]=="prefix" for row in result["interruptions"]))
        saved=json.loads(evidence.path.read_text())
        self.assertEqual(saved["active_unit"]["cache_paths"],result["cache_paths"])
        self.assertEqual(len(saved["active_unit"]["step_checks"]),32)

    def test_real_generate_observer_uses_committed_events(self):
        context,evidence,bank,item,executor,unused=self.make_runtime()
        with patch.object(phases,"_hash",return_value="fake-array-hash"):
            array,service=phases._generate(context,evidence,bank,item,executor,cache=True)
        self.assertTrue(service["cache_paths"]["valid"])
        self.assertEqual(service["cache_paths"]["observed_steps"],32)
        self.assertEqual(service["cache_paths"]["counts"],dict(full=30,reuse=2,mixed=0,unknown=0))
        self.assertEqual(service["reuse_steps"],[9,11])


if __name__=="__main__":
    unittest.main()
