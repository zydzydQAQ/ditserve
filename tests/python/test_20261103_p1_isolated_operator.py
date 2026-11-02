"""P0-gated operator preparation keeps calibration and ABBA on one allocation."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import p1_isolated_job as job
import p1_isolated_batch as isolated
import formal_benchmark as formal
import test_20261103_p1_isolated_batch as fixtures
import test_20261102_stage_gates as gates
from ditserve.evaluation.common import canonical_hash


class OperatorPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IsolatedBatchTest.setUpClass()

    def fixture(self, root, model='wan21'):
        first = copy.deepcopy(fixtures.IsolatedBatchTest.examples[model + '-0'])
        package = root / 'python'
        for name in ('ditserve', 'xfuser'):
            (package / name).mkdir(parents=True)
            (package / name / '__init__.py').write_text('# frozen installed fixture\n')
        core = root / 'ditserve'; core.write_bytes(b'fixture native executable')
        operator = root / 'release/scripts/kairos_formal_benchmark.sh'
        operator.parent.mkdir(parents=True); operator.write_text('# frozen operator\n')
        build = dict(commit='a'*40, core_sha256=job.file_hash(core))
        env = dict(DITSERVE_BUILD='a'*7, DITSERVE_TEST_REVISION='a'*40,
            DITSERVE_RUN_ID='d20261103-p1iso-' + model + '-aaaaaaa-r1',
            DITSERVE_PROTOCOL_SHA256=canonical_hash(first['protocol']),
            DITSERVE_P1_PLAN_SHA256=canonical_hash(first['pair_plan']),
            DITSERVE_FORMAL_ACTION='p1_isolated', DITSERVE_STAGE_ACTION='p1_diagnostics',
            DITSERVE_BENCHMARK_MODE='formal',
            DITSERVE_P0_GATE_SHA256='b'*64, DITSERVE_IMAGE_DIGEST='sha256:'+'d'*64,
            DITSERVE_EXPECTED_CORE_SHA256=build['core_sha256'],
            DITSERVE_EXPECTED_PYTHON_SHA256=canonical_hash(job.inventory(package)),
            DITSERVE_P1_HELPER_SHA256=job.file_hash(job.__file__),
            DITSERVE_P1_OPERATOR_SHA256=job.file_hash(operator),
            DITSERVE_EXPECTED_BENCHMARKS_SHA256=canonical_hash(job.benchmark_inventory(ROOT/'benchmarks')),
            DITSERVE_KAIROS_RUNTIME_ROOT=str(operator.parent.parent),
            DITSERVE_JOB_SPEC_PATH=str(root/'job_spec.json'),
            DITSERVE_MODEL_MANIFESTS=str(root/'model_manifests.json'),
            DITSERVE_MODEL_MANIFESTS_SHA256=canonical_hash(first['model_manifests']),
            WAN_MODEL='wan' if model == 'wan21' else 'wan2.2-ti2v-5b')
        spec = dict(name=env['DITSERVE_RUN_ID'], image='registry/ditserve@'+env['DITSERVE_IMAGE_DIGEST'],
            workers=1, gpus_per_worker=2, mount_shared=True, command='bash '+str(operator),
            env=[dict(name=k, value=v) for k,v in env.items()])
        docs = {key: first[key] for key in ('protocol', 'pair_plan', 'model_manifests')}
        docs.update(build=build, job_spec=spec, **{'pool_'+k: v for k,v in first['pools'].items()})
        sources = {}
        for key, value in docs.items():
            sources[key] = root / (key+'.json'); sources[key].write_bytes(job.encoded(value))
        records, paths = [], []
        for i in range(2):
            record = copy.deepcopy(fixtures.IsolatedBatchTest.examples[model+'-'+str(i)]['calibrations'][0])
            record.update(run_id=env['DITSERVE_RUN_ID']+'-cal'+str(i),
                source=dict(core_sha256=build['core_sha256'], files=job.inventory(package)))
            path = root/f'calibration-{i}.json'; path.write_bytes(job.encoded(record))
            records.append(record); paths.append(path)
        bundle = formal.assemble(first['protocol'], model, records, first['runtime_facts'])
        (root/'bundle.json').write_bytes(job.encoded(bundle))
        (root/'runtime-facts.json').write_bytes(job.encoded(first['runtime_facts']))
        kwargs = dict(model=model, sources=sources, package=package, core=core,
            operator_script=operator, gpu_uuids=first['gpu_uuids'], environment=env)
        checked = job.preflight(**kwargs)
        (root/'preflight.json').write_bytes(job.encoded(checked))
        prepare = dict(bundle_path=root/'bundle.json', runtime_facts_path=root/'runtime-facts.json',
            calibration_paths=paths, package=package, core=core, gpu_uuids=first['gpu_uuids'],
            output_dir=root/'prepared')
        return kwargs, prepare

    def test_both_models_prepare_both_complete_pairs_with_portable_sources(self):
        for model in ('wan21', 'wan22'):
            with self.subTest(model=model), TemporaryDirectory() as directory:
                root = Path(directory); _, kwargs = self.fixture(root, model)
                preflight = json.loads((root/'preflight.json').read_bytes())
                self.assertEqual(preflight['expected_requests'], 192)
                self.assertEqual([v['expected_requests'] for v in preflight['planned_cases']], [96, 96])
                result = job.prepare_inputs(root/'preflight.json', **kwargs)
                self.assertEqual(result['expected_requests'], 192)
                self.assertEqual(len(result['cases']), 2)
                self.assertFalse(result['qualification_eligible'])
                self.assertFalse(result['dependencies_closed'])
                mirror = root/'relocated'; shutil.copytree(root/'prepared', mirror)
                for case in result['cases']:
                    inputs = isolated.load_inputs(mirror/case['inputs'], case['inputs_sha256'])
                    config = json.loads((mirror/case['config']).read_bytes())
                    isolated.audit_config(config, inputs)
                    coverage = json.loads((mirror/(case['pair_id']+'-planned-coverage.json')).read_bytes())
                    self.assertEqual(coverage['unexecuted'], 96)
                    self.assertEqual(inputs['gpu_uuids'], ['GPU-a', 'GPU-b'])

    def test_preflight_checks_allocation_operator_and_submitted_environment(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); kwargs, _ = self.fixture(root)
            for mutate in (
                lambda x: x.update(gpu_uuids=['GPU-a','GPU-b','GPU-c','GPU-d']),
                lambda x: x['environment'].update(DITSERVE_P1_HELPER_SHA256='f'*64),
                lambda x: x['environment'].update(DITSERVE_EXPECTED_BENCHMARKS_SHA256='f'*64),
                lambda x: x['environment'].update(DITSERVE_STAGE_ACTION='shape_calibration')):
                changed = copy.deepcopy(kwargs); mutate(changed)
                with self.assertRaises(ValueError): job.preflight(**changed)
            kwargs['operator_script'].write_text('# mutated\n')
            with self.assertRaisesRegex(ValueError, 'operator'): job.preflight(**kwargs)

    def test_missing_or_relabelled_formal_mode_cannot_bypass_the_submission_gate(self):
        for mode in (None, 'development'):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root=Path(directory); kwargs,_=self.fixture(root)
                spec=json.loads(kwargs['sources']['job_spec'].read_bytes())
                spec['env']=[item for item in spec['env'] if item['name']!='DITSERVE_BENCHMARK_MODE']
                kwargs['environment'].pop('DITSERVE_BENCHMARK_MODE')
                if mode is not None:
                    spec['env'].append(dict(name='DITSERVE_BENCHMARK_MODE',value=mode))
                    kwargs['environment']['DITSERVE_BENCHMARK_MODE']=mode
                kwargs['sources']['job_spec'].write_bytes(job.encoded(spec))
                with self.assertRaisesRegex(ValueError,'frozen P0-gated'): job.preflight(**kwargs)

    def test_submitted_spec_rejects_a_changed_model_map_before_calibration(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); kwargs,_=self.fixture(root)
            path=kwargs['sources']['model_manifests']
            value=json.loads(path.read_bytes()); value['wan21']['sha256']='f'*64
            path.write_bytes(job.encoded(value))
            with self.assertRaisesRegex(ValueError,'frozen P0-gated'): job.preflight(**kwargs)

    def test_rejects_changed_runtime_sources_or_uuid_without_preparing_partial_output(self):
        changes = (
            lambda root,k: k.update(gpu_uuids=['GPU-b','GPU-a']),
            lambda root,k: k['core'].write_bytes(b'changed'),
            lambda root,k: (k['package']/'xfuser/__init__.py').write_text('changed'),
            lambda root,k: (root/'release/scripts/kairos_formal_benchmark.sh').write_text('changed'),
            lambda root,k: (root/'pool_video.json').write_text('{}'))
        for i, change in enumerate(changes):
            with self.subTest(change=i), TemporaryDirectory() as directory:
                root=Path(directory); _, kwargs=self.fixture(root); change(root,kwargs)
                with self.assertRaises(ValueError): job.prepare_inputs(root/'preflight.json', **kwargs)
                self.assertFalse(kwargs['output_dir'].exists())

    def test_missing_failed_reordered_or_foreign_calibration_does_not_prepare(self):
        for case in ('missing', 'order', 'foreign', 'row', 'failed', 'reassembled_incomplete'):
            with self.subTest(case=case), TemporaryDirectory() as directory:
                root=Path(directory); pre, kwargs=self.fixture(root)
                paths=kwargs['calibration_paths']
                if case=='missing': paths.pop()
                elif case=='order': paths.reverse()
                else:
                    value=json.loads(paths[0].read_bytes())
                    if case=='foreign': value['run_id']='another-job-cal0'
                    elif case=='failed': value['status']='failed'
                    else: value['requests'].pop()
                    paths[0].write_bytes(job.encoded(value))
                    if case=='reassembled_incomplete':
                        bundle=formal.assemble(json.loads(pre['sources']['protocol'].read_bytes()), 'wan21',
                            [json.loads(p.read_bytes()) for p in paths], json.loads(kwargs['runtime_facts_path'].read_bytes()))
                        kwargs['bundle_path'].write_bytes(job.encoded(bundle))
                with self.assertRaises(ValueError): job.prepare_inputs(root/'preflight.json', **kwargs)
                self.assertFalse(kwargs['output_dir'].exists())

    def test_spec_preserves_gpu_build_and_requires_explicit_p0_before_unique_spec(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); pre, _=self.fixture(root)
            protocol,plan=(json.loads(pre['sources'][k].read_bytes()) for k in ('protocol','pair_plan'))
            base=json.loads(pre['sources']['job_spec'].read_bytes()); env=job.env_values(base)
            env.update(DITSERVE_FORMAL_ACTION='calibration', DITSERVE_STAGE_ACTION='shape_calibration')
            env.pop('DITSERVE_BENCHMARK_MODE')
            base['env']=[dict(name=k,value=v) for k,v in env.items()]
            kwargs=dict(model='wan21',date='20261103',operator_root='/shared/data/DITSERVE/scripts/releases/new-operator',
                operator_script=pre['operator_script'],benchmark_root=ROOT/'benchmarks')
            receipt=root/'receipt.json'
            with self.assertRaises(FileNotFoundError): job.prepare_spec(base,protocol,plan,receipt,**kwargs)
            receipt.write_text('{}')
            with self.assertRaisesRegex(ValueError,'invalid stage receipt'):
                job.prepare_spec(base,protocol,plan,receipt,**kwargs)
            fresh=dict(build=dict(commit='a'*40,core_sha256=env['DITSERVE_EXPECTED_CORE_SHA256'],image_digest=env['DITSERVE_IMAGE_DIGEST']),
                stages=dict(P0=dict(status='passed',allows=['p1_diagnostics']),CONTRACT=dict(python_sources_sha256=env['DITSERVE_EXPECTED_PYTHON_SHA256'])))
            mapping=json.loads(pre['sources']['model_manifests'].read_bytes())
            records=[dict(model='wan21',role=role,**gates.write(root,role+'.json',dict(
                config=dict(benchmark=dict(model_manifests=mapping))))) for role in ('preemption','no_preemption')]
            receipt.write_bytes(job.encoded(dict(source_manifest=gates.write(root,'p0-manifest.json',dict(runs=records)))))
            with patch.object(job,'verify_receipt',return_value=fresh) as verify:
                result=job.prepare_spec(base,protocol,plan,receipt,**kwargs)
            verify.assert_called_once_with(receipt,build_commit='a'*7,protocol_sha256=canonical_hash(protocol),
                required_stage='P0',action='p1_diagnostics')
            self.assertEqual(result['name'],'d20261103-p1iso-wan21-aaaaaaa-r1')
            actual=job.env_values(result)
            self.assertEqual(actual['DITSERVE_BUILD'],env['DITSERVE_BUILD'])
            self.assertEqual(actual['DITSERVE_STAGE_ACTION'],'p1_diagnostics')
            self.assertEqual(actual['DITSERVE_BENCHMARK_MODE'],'formal')
            self.assertEqual(actual['DITSERVE_P0_GATE_SHA256'],job.file_hash(receipt))
            self.assertEqual(actual['DITSERVE_P1_HELPER_SHA256'],job.file_hash(job.__file__))
            self.assertEqual(actual['DITSERVE_MODEL_MANIFESTS_SHA256'],canonical_hash(mapping))
            self.assertNotEqual(result['command'],base['command'])
            # Even with a rehashed manifest, inconsistent P0 role mappings
            # cannot authorize a differently weighted diagnostic.
            altered=copy.deepcopy(mapping);altered['wan21']['sha256']='f'*64
            records[0].update(gates.write(root,'preemption.json',dict(config=dict(benchmark=dict(model_manifests=altered)))))
            receipt.write_bytes(job.encoded(dict(source_manifest=gates.write(root,'p0-manifest.json',dict(runs=records)))))
            with patch.object(job,'verify_receipt',return_value=fresh), self.assertRaisesRegex(ValueError,'single frozen'):
                job.prepare_spec(base,protocol,plan,receipt,**kwargs)
            base['gpus_per_worker']=4
            with self.assertRaisesRegex(ValueError,'two-GPU'): job.prepare_spec(base,protocol,plan,receipt,**kwargs)

    def test_recomputed_blocked_p0_is_never_authorized(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); manifest=gates.fixture(root); manifest['runs']=[]
            receipt=gates.stage_gates.evaluate(manifest,root)
            receipt['source_manifest']=gates.write(root,'manifest.json',manifest)
            path=root/'receipt.json'; path.write_bytes(job.encoded(receipt))
            # Formal plan consistency is covered above; use the stage fixture's
            # small protocol here to exercise the real receipt recomputation.
            protocol=json.loads((root/'protocol.json').read_bytes())
            base=dict(workers=1,gpus_per_worker=2,mount_shared=True,env=[dict(name=k,value=v) for k,v in dict(
                DITSERVE_BUILD='a'*7, DITSERVE_FORMAL_ACTION='calibration', DITSERVE_STAGE_ACTION='shape_calibration',
                WAN_MODEL='wan', DITSERVE_PROTOCOL_SHA256=canonical_hash(protocol)).items()])
            with patch.object(job,'validate_plan'), self.assertRaisesRegex(ValueError,'required stage has not passed: P0'):
                job.prepare_spec(base,protocol,{},path,model='wan21',date='20261103',
                    operator_root='/shared/data/DITSERVE/scripts/releases/new',operator_script=root/'unused',benchmark_root=root/'unused')


if __name__ == '__main__':
    unittest.main()
