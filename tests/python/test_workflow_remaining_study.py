"""CPU contracts for the real remaining matrix, frozen registration and GPU mapping."""
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_remaining_study as remaining
from ditserve.evaluation import workflow_next_study as api
from ditserve.evaluation.workflow_overlap_trace import capture_range
from test_workflow_remaining_online import inputs

REPO=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('remaining_suite',REPO/'docker/run-workflow-suite.py')
suite=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(suite)


class RemainingStudyTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/'workflow/run03';self.root.mkdir(parents=True)
        self.study,self.profile,self.anchors=inputs()
        source=self.root/'sources/r018';(source/'configs/env').mkdir(parents=True)
        (source/'bin').mkdir();(source/'bin/ditserve').write_text('CPU fixture binary')
        shutil.copyfile(REPO/'configs/env/rtx4090-2x24g.json',source/'configs/env/rtx4090-2x24g.json')
        revision=api.seal(source)
        self.study.update(root=str(self.root),run_id='run03',image='fixture',**revision,
                          source_revisions=[revision],decisions=[],active_job=None,budget_seconds=None,
                          gpu_mapping=[dict(ordinal=i,uuid='GPU-12345678-1234-1234-1234-123456789ab'+str(i)) for i in (0,1)])
        repaired=dict(stage='resize',sp_linear_policy='full_rows_except_ffn1',
                      cpu_masks={'cuda:0':list(range(8)),'cuda:1':list(range(60,68))})
        path=self.root/'inputs/elastic-repaired-resize.json';api.atomic_json(path,repaired)
        self.study['jobs']=[dict(id='elastic-repaired-resize',logical_id='elastic-repaired-resize',input=str(path))]
        self.calibration=dict(profile=self.profile,capacities={'video':.1,'mixed':.4})
        api.atomic_json(self.root/'calibration.json',self.calibration)
        self.context=dict(profile=self.profile,anchors=self.anchors,input_bindings=[],inheritance='CPU fixture',
                          history=dict(jobs=1,attempts=0))
        patcher=patch.object(remaining,'_context',return_value=self.context);patcher.start();self.addCleanup(patcher.stop)
        self.nsys=self.root/'tools/nsys';self.nsys.parent.mkdir();self.nsys.write_bytes(b'CPU profiler binding fixture')
        patcher=patch.object(remaining,'NSYS_HOST',str(self.nsys));patcher.start();self.addCleanup(patcher.stop)

    def plan(self):return remaining.build_plan(self.study)

    def short_plan(self):
        plan=self.plan()
        chosen=['remaining-cpu-depth-s5301','remaining-vae-performance-a17-b33','remaining-vae-trace-a17-b33']
        plan['jobs']=[j for name in chosen for j in plan['jobs'] if j['id']==name]
        return plan

    def test_real_36_job_plan_is_pure_and_topological(self):
        before=copy.deepcopy(self.study);plan=self.plan()
        self.assertEqual(self.study,before);self.assertEqual(plan,self.plan())
        self.assertEqual(len(plan['jobs']),36)
        self.assertEqual([j['metadata']['seed'] for j in plan['jobs'][:2]],[4600,4700])
        identifiers={j['id'] for j in plan['jobs']};seen=set()
        for job in plan['jobs']:
            self.assertTrue((set(job['requires'])&identifiers)<=seen)
            seen.add(job['id'])
            self.assertGreater(job['timeout_seconds'],job['estimated_seconds'])
            self.assertEqual(job['metadata']['monitor_cpu_affinity'],[58])
        self.assertFalse(plan['default_change'])
        self.assertEqual(sum(j['metadata']['role']=='remaining_sp_comparison' for j in plan['jobs']),3)
        self.assertEqual(sum(j['metadata']['role']=='remaining_vae_trace' for j in plan['jobs']),4)

    def test_depth_zero_one_two_and_eight_core_controls_are_preserved(self):
        rows=[j for j in self.plan()['jobs'] if j['metadata']['role']=='remaining_cpu_depth']
        self.assertEqual(len(rows),3)
        for repeat,row in enumerate(rows):
            phases=row['config']['phases'];expected=[0,1,2][repeat:]+[0,1,2][:repeat]
            self.assertEqual([p['text_prefetch_depth'] for p in phases],expected)
            self.assertTrue(all(p['workload']==phases[0]['workload'] for p in phases))
            self.assertEqual(len(phases[0]['workload']),16)
            self.assertTrue(all(r['kind']=='video' for r in phases[0]['workload']))
            for phase in phases:
                self.assertEqual(phase['controller_cpu_affinity'],[59])
                self.assertEqual(phase['occupancy_backend'],'nvml')
                self.assertFalse(phase['dispatch_refresh'])
                for rank in (0,1):
                    worker=phase['worker_overrides']['wan'+str(rank)]
                    self.assertEqual(worker['cpu_threads'],8)
                    self.assertEqual(worker['cpu_affinity'],list(range(rank*60,rank*60+8)))

    def test_lifecycle_wave_uuid_and_normal_timing_remap(self):
        rows=self.plan()['jobs'];bundle=next(j for j in rows if j['metadata']['role']=='remaining_vae_lifecycle_bundle')
        self.assertEqual(bundle['devices'],[0,1]);self.assertEqual(bundle['timeout_seconds'],6000)
        self.assertEqual([c['device_ordinal'] for c in bundle['config']['cases']],[0,1,0,1])
        for case in bundle['config']['cases']:
            ordinal=case['device_ordinal'];c=case['config']
            self.assertEqual(case['gpu_uuid'],self.study['gpu_mapping'][ordinal]['uuid'])
            self.assertEqual(c['cpu_masks'],{'cuda:0':list(range(ordinal*60,ordinal*60+8))})
            self.assertEqual((c['warmup_pairs'],c['lifecycle_pairs'],c['arm_timeout_seconds']),(2,3,900))
        for row in rows:
            if row['metadata']['role'] in ('remaining_vae_performance','remaining_vae_trace'):
                self.assertEqual(row['config']['device'],'cuda:0')
                self.assertEqual(row['config']['gpu_uuid'],self.study['gpu_mapping'][row['devices'][0]]['uuid'])

    def test_registration_is_idempotent_and_preserves_existing_history(self):
        plan=self.short_plan();before=copy.deepcopy(self.study['jobs'])
        with patch.object(remaining,'build_plan',return_value=plan):
            rows=remaining.register(self.study);snapshot=copy.deepcopy(self.study)
            self.assertEqual(remaining.register(self.study),rows)
        self.assertEqual(self.study,snapshot);self.assertEqual(self.study['jobs'][:1],before)
        for row in rows:
            self.assertEqual(suite.selected_devices(row),row['devices'])
            self.assertEqual(row['source'],self.study['source'])
            self.assertTrue(any(b['path'].endswith('remaining-validation-plan.json') for b in row['input_bindings']))
        trace=rows[-1]
        self.assertEqual(trace['command'][0],remaining.NSYS)
        self.assertIn('--capture-range=nvtx',trace['command'])
        self.assertIn('--capture-range-end=stop',trace['command'])
        self.assertIn('--nvtx-capture='+capture_range(plan['jobs'][-1]['config']),trace['command'])
        self.assertIn(api.binding(self.nsys),trace['input_bindings'])
        self.assertFalse(trace['counters'])

    def test_registered_command_timeout_source_and_mapping_drift_rejected(self):
        plan=self.short_plan()
        with patch.object(remaining,'build_plan',return_value=plan):
            remaining.register(self.study)
            baseline=copy.deepcopy(self.study)
            for field,value in [('command',['/bin/true']),('timeout_seconds',99999),('estimated_seconds',1),
                                ('source','/wrong'),('source_digest','0'*64),('command_type','native'),
                                ('gpu_mapping',[self.study['gpu_mapping'][1]])]:
                self.study=copy.deepcopy(baseline);self.study['jobs'][-1][field]=value
                with self.subTest(field=field),self.assertRaises(ValueError):remaining.register(self.study)
            self.study=copy.deepcopy(baseline)
            path=Path(self.study['jobs'][-1]['input']);path.write_text('{}')
            with self.assertRaises(ValueError):remaining.register(self.study)

    def test_every_python_job_maps_actual_physical_devices(self):
        for template in self.plan()['jobs']:
            if template.get('native'):continue
            job=api.add_job(self.study,template['id'],'extension',template['config'],module=template['module'],
                            devices=template['devices'],requires=template['requires'],metadata=template['metadata'])
            job.update(command_type='remaining_gpu',gpu_mapping=[self.study['gpu_mapping'][d] for d in job['devices']],
                       command=remaining.expected_command(self.study,job['id'],template['config'],role=job['metadata']['role']))
            with self.subTest(job=job['id']):self.assertEqual(suite.selected_devices(job),job['devices'])
            altered=copy.deepcopy(job);altered['command'][-1]='/data/wrong'
            with self.assertRaises(ValueError):suite.selected_devices(altered)
            altered=copy.deepcopy(job);altered['gpu_mapping'][0]['uuid']='GPU-ffffffff-ffff-ffff-ffff-ffffffffffff'
            if job['metadata']['role']!='remaining_interleave':
                with self.assertRaises(ValueError):suite.selected_devices(altered)

    def test_topological_cycle_and_duplicate_ids_rejected(self):
        for rows in ([dict(id='a',requires=['b']),dict(id='b',requires=['a'])],
                     [dict(id='a',requires=[]),dict(id='a',requires=[])]):
            with self.assertRaises(ValueError):remaining.topological_order(rows)


if __name__=='__main__':unittest.main()
