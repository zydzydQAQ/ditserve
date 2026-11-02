"""Finite offline AB/BA compilation is distinct from P1 low-load evidence."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
import formal_benchmark as formal
import p1_isolated_batch as isolated
import test_20261102_formal_benchmark as fixtures
from ditserve.evaluation.common import canonical_hash, read_json, seal
from ditserve.runtime import PACKAGE_ROOT
from ditserve.runtime.client import CoreBridge


class IsolatedBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.FormalBenchmarkTest.setUpClass()
        cls.f = fixtures.FormalBenchmarkTest()
        cls.plan = read_json(ROOT / 'configs/benchmarks/p1-qualification-v1.json')
        cls.examples = {pair['id']: cls.make_inputs(pair) for pair in cls.plan['shape_pairs']}

    @classmethod
    def make_inputs(cls, pair):
        uuids = ['GPU-a', 'GPU-b']
        manifests = {m: dict(path=f'/shared/data/DITSERVE/models/{m}.json', sha256='a' * 64,
                            model_path=f'/shared/data/DITSERVE/models/{m}-official')
                     for m in ('wan21', 'wan22', 'sd35')}
        runtime = dict(python='fixture', cuda='fixture', packages={'torch': 'fixture'})
        identity = dict(schema='ditserve.p1-isolated-identity.v1', core_sha256='c' * 64,
            python_sources={'ditserve/runtime/worker.py': 'd' * 64},
            runtime_identity_sha256=canonical_hash(runtime), model_manifests_sha256=canonical_hash(manifests))
        facts = dict(models={formal.MODELS[pair['model']]['name']: dict(
            transformer_config=dict(patch_size=[1, 2, 2], num_attention_heads=24,
                attention_head_dim=128, num_layers=30, out_channels=48),
            vae_config=dict(scale_factor_temporal=4, scale_factor_spatial=16))})
        config = formal.calibration(cls.f.p, pair['model'], uuids, cls.f.pools, manifests,
                                    pair['video_shape'], pair['image_shape'])
        phase = config['phases'][0]
        rows, cells = [], []
        for index, policy in enumerate(phase['matrix']['/policy']):
            cells.append(dict(cell=index, policy=policy, order='forward', status='completed'))
            for source in phase['workload']:
                row = copy.deepcopy(source)
                video = row['kind'] == 'video'
                model = pair['model'] if video else 'sd35'
                row['spec']['request_id'] = row['request_id']
                spec = row['spec']
                generation = dict(spec, model_path=manifests[model]['model_path'],
                    model_manifest_sha256=manifests[model]['sha256'], model_integrity='manifest_verified',
                    runtime_identity=runtime)
                artifact = dict(finite=True, shape=[1] + ([spec['frames']] if video else [])
                                + [spec['height'], spec['width'], 3])
                row.update(cell=index, phase='cluster', attempt=1, order='forward',
                    observation_id=f'cluster/1/{index}', status='completed',
                    model_ms=1000 * spec['steps'] + 100 if video else 250,
                    outputs=[dict(model_ms=100 if video else 0, generation=generation, output=artifact)])
                if video:
                    group = [0] if policy == 'static_sp1' else [0, 1]
                    row['forms'] = [policy.removeprefix('static_')]
                    row.update(encode_model_ms=20, transfer_export_ms=2, dispatch_gap_ms=1, timed_quanta=1,
                        advance_samples=[dict(steady=True, model_ms=1000 if policy == 'static_sp1' else 600,
                                              group=group, form=row['forms'][0])])
                rows.append(row)
        record = dict(schema='ditserve.run.v1', experiment=config['experiment'], run_id='fixture-' + pair['id'],
            status='completed', failures=[], config=config, config_sha256=canonical_hash(config),
            source=dict(core_sha256=identity['core_sha256'], files=identity['python_sources']),
            phases=[dict(id='cluster', attempt=1, status='completed', cells=cells)], requests=rows,
            budget=dict(sessions=[dict(identity=dict(gpu_uuids=uuids, config_sha256=canonical_hash(config)))]))
        bundle = formal.assemble(cls.f.p, pair['model'], [record], facts)
        return dict(protocol=cls.f.p, pair_plan=cls.plan, pair_id=pair['id'], gpu_uuids=uuids,
            pools=cls.f.pools, model_manifests=manifests, identity=identity, bundle=bundle,
            calibrations=[record], runtime_facts=facts, inputs_manifest_sha256='e' * 64)

    def inputs(self, pair='wan21-0'):
        return copy.deepcopy(self.examples[pair])

    def test_all_four_pairs_compile_three_same_input_abba_phases(self):
        for pair, inputs in self.examples.items():
            with self.subTest(pair=pair):
                cfg = isolated.compile_case(inputs)
                self.assertEqual(len(cfg['phases']), 3)
                self.assertEqual(len(isolated.planned_cells(cfg)), 12)
                self.assertEqual(sum(c['expected_requests'] for c in isolated.planned_cells(cfg)), 96)
                for phase in cfg['phases']:
                    self.assertTrue(phase['paired'])
                    self.assertEqual(phase['matrix']['/policy'], ['edf-cfg2', 'block-cfg2'])
                    rows = phase['matrix']['/workload'][0]
                    self.assertEqual(len(rows), 8)
                    self.assertEqual(len({r['dataset_sample_id'] for r in rows}), 8)
                    for r in rows:
                        self.assertEqual((r['kind'], r['arrival_ms'], r['output_mode']), ('video', 0, 'digest'))
                        self.assertEqual(formal.input_shape(r['spec']), formal.shape_spec(inputs['protocol'], r['class']))
                        self.assertEqual(r['deadline_ms'], r['service_ms'] * 1.5)
                    selected = formal.split_pool(inputs['pools']['video'], inputs['protocol'], 'video', 'development')
                    self.assertTrue({r['spec']['prompt'] for r in rows} <= {r['prompt'] for r in selected['samples']})
                meta = cfg['benchmark']
                self.assertTrue(meta['diagnostic_only'])
                self.assertFalse(meta['qualification_eligible'])
                self.assertEqual(meta['formal_protocol']['role'], 'p1_isolated_batch')
                self.assertNotIn('p1', meta)
                self.assertNotIn('preemption_diagnostic', meta)
                self.assertNotIn('load', meta)
                self.assertEqual(meta['isolated_batch']['requires_stage'], 'P0')
                self.assertEqual(cfg['budget']['limit_gpu_seconds'],
                                 sum(p['watchdog_seconds'] * 2 for p in cfg['phases']))

    @unittest.skipUnless(CoreBridge().available(), 'build the native core first')
    def test_native_matrix_and_parser_preserve_policy_inputs_and_deadlines(self):
        cfg = isolated.compile_case(self.inputs())
        for phase in cfg['phases']:
            expanded = CoreBridge().call('control', dict(op='matrix', phase=phase))
            self.assertEqual([c['policy'] for c in expanded], isolated.POLICIES)
            for cell in expanded:
                self.assertTrue(cell['paired'])
                rows = CoreBridge().call('control', dict(op='workload', phase=cell))
                for parsed, original in zip(rows, phase['matrix']['/workload'][0]):
                    for key in ('spec', 'arrival_ms', 'deadline_ms', 'dataset_sample_id'):
                        self.assertEqual(parsed[key], original[key])
            self.assertEqual(expanded[0]['workload'], expanded[1]['workload'])

    @unittest.skipUnless(CoreBridge().available(), 'build the native core first')
    def test_actual_native_paired_execution_is_abba_and_preserves_all_96_rows(self):
        from ditserve import api
        from test_study import FixtureBridge
        # A fast mock projection exercises the real controller's paired flag,
        # cell identities, worker restart and publication. It is not GPU evidence.
        cfg = isolated.compile_case(self.inputs())
        cfg['workers'] = {name: dict(backend='mock', config=dict(options=dict(
            costs=dict(step_ms=1, finish_ms=0)))) for name in cfg['workers']}
        for phase in cfg['phases']:
            phase['warmup'] = []
            phase['text_prefetch'] = 0
            for klass in phase['profile']['classes'].values():
                if klass['kind'] == 'video':
                    klass.update(steps=1, blocks_per_step=2, step_compute_ms=1)
            for row in phase['matrix']['/workload'][0]:
                row['spec'].update(steps=1, blocks_per_step=2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'python'
            for name in ('ditserve', 'xfuser'):
                # CONTRACT tests are staged separately from the installed
                # Python package; a source checkout's workers/ is not present.
                sources = list((PACKAGE_ROOT/name).rglob('*.py'))
                self.assertTrue(sources, 'installed fixture package is missing: ' + name)
                for source in sources:
                    target = package/name/source.relative_to(PACKAGE_ROOT/name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            # Freeze the CPU fixture identity before execution. This projection
            # deliberately cannot pass audit_config as a GPU experiment.
            identity = cfg['benchmark']['isolated_batch']['identity']
            identity['core_sha256'] = hashlib.sha256(CoreBridge().binary.read_bytes()).hexdigest()
            identity['python_sources'] = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in package.rglob('*.py')}
            cfg['benchmark']['profile_placement']['gpu_uuids'] = []
            value = api.run(cfg, run_id='abba', work_dir=root/'work', results_root=root/'results',
                            ledger=root/'budget.json', core=FixtureBridge(package))
            self.assertEqual(value['status'], 'completed', value.get('failures'))
            for phase in value['phases']:
                self.assertEqual([(c['policy'], c['cell'], c['order']) for c in phase['cells']],
                    [('edf-cfg2', 0, 'forward'), ('block-cfg2', 1, 'forward'),
                     ('block-cfg2', 1, 'reverse'), ('edf-cfg2', 0, 'reverse')])
            self.assertEqual(len(value['requests']), 96)
            self.assertTrue(all(r['status'] == 'completed' for r in value['requests']))
            result = isolated.coverage(cfg, value)
            self.assertEqual((result['completed'], result['abba_cells_completed'], result['unexecuted']), (96, 12, 0))
            self.assertFalse(result['qualification_eligible'])
            failed = copy.deepcopy(cfg)
            failed['workers']['wan1']['config']['options']['fail_ops'] = ['block_import']
            value = api.run(failed, run_id='failed', work_dir=root/'failed', results_root=root/'results',
                            ledger=root/'failed-budget.json', core=FixtureBridge(package))
            self.assertEqual(value['status'], 'failed')
            self.assertEqual(len(value['requests']), 8)
            self.assertTrue(all(r['status'] != 'completed' for r in value['requests']))
            self.assertEqual(sum(c['expected_requests'] for c in isolated.planned_cells(failed)), 96)
            result = isolated.coverage(failed, value)
            self.assertEqual((result['failed_or_incomplete'], result['unexecuted'], result['abba_cells_completed']), (8, 88, 0))
            self.assertEqual(result['failures'], value['failures'])

    def test_compilation_is_reproducible_and_does_not_change_inputs(self):
        inputs = self.inputs()
        before = copy.deepcopy(inputs)
        first = isolated.compile_case(inputs)
        self.assertEqual(first, isolated.compile_case(inputs))
        self.assertEqual(before, inputs)
        self.assertEqual(len({p['matrix']['/trace_provenance'][0]['sha256'] for p in first['phases']}), 3)
        isolated.audit_config(first, inputs)

    def test_rehashed_bundle_cannot_replace_actual_shape_cost_or_sources(self):
        for field in ('shape', 'cost', 'source', 'gpu'):
            inputs = self.inputs()
            entry = inputs['bundle']['classes']['wan21-480p']
            if field == 'shape':
                entry['spec']['frames'] = 33
            elif field == 'cost':
                entry['cost']['step_compute_ms'] *= 2
            elif field == 'source':
                entry['source']['run_sha256'] = 'f' * 64
            else:
                inputs['bundle']['gpu_uuids'].reverse()
            inputs['bundle'] = seal(inputs['bundle'])
            with self.subTest(field=field), self.assertRaises(ValueError):
                isolated.compile_case(inputs)

    def test_model_runtime_and_failed_calibration_cannot_be_relabelled(self):
        changes = [lambda x: x['model_manifests']['wan21'].__setitem__('sha256', 'f'*64),
                   lambda x: x['calibrations'][0]['source'].__setitem__('core_sha256', 'f'*64),
                   lambda x: x['calibrations'][0]['source'].__setitem__('files', {'ditserve/x.py': 'd'*64}),
                   lambda x: x['calibrations'][0]['requests'].pop(),
                   lambda x: x['calibrations'][0]['requests'][0]['outputs'][0]['generation'].__setitem__('runtime_identity', {'python':'changed'}),
                   lambda x: x['calibrations'][0]['requests'][0]['outputs'][0]['output'].__setitem__('finite', False)]
        for change in changes:
            inputs = self.inputs(); change(inputs)
            with self.subTest(change=change), self.assertRaises(ValueError):
                isolated.compile_case(inputs)

    def test_reassembled_calibration_must_match_executed_policy_status_and_planned_prompt(self):
        def prompt(x):
            row = x['calibrations'][0]['requests'][0]
            row['spec']['prompt'] = row['outputs'][0]['generation']['prompt'] = 'changed prompt'
        def swap(x):
            cells = x['calibrations'][0]['phases'][0]['cells']
            cells[1]['policy'], cells[2]['policy'] = cells[2]['policy'], cells[1]['policy']
        changes = [prompt, swap,
            lambda x: x['calibrations'][0]['phases'][0]['cells'][0].__setitem__('status', 'failed'),
            lambda x: x['calibrations'][0]['requests'][2].__setitem__('forms', ['sp2']),
            lambda x: x['calibrations'][0]['requests'][2]['advance_samples'][0].__setitem__('form', 'sp2'),
            lambda x: x['calibrations'][0]['requests'][2]['advance_samples'][0].__setitem__('group', [1, 2])]
        for change in changes:
            inputs = self.inputs(); change(inputs)
            inputs['bundle'] = formal.assemble(inputs['protocol'], 'wan21', inputs['calibrations'], inputs['runtime_facts'])
            inputs['inputs_manifest_sha256'] = canonical_hash(inputs)
            with self.subTest(change=change), self.assertRaises(ValueError):
                isolated.compile_case(inputs)

    def test_short_development_pool_is_not_cycled_or_supplemented_with_test_prompts(self):
        inputs = self.inputs()
        selected = formal.split_pool(inputs['pools']['video'], inputs['protocol'], 'video', 'development')
        excluded = {r['sample_id'] for r in selected['samples'][7:]}
        pool = inputs['pools']['video']
        pool['samples'] = [r for r in pool['samples'] if r['sample_id'] not in excluded]
        inputs['pools']['video'] = seal(pool)
        pair = inputs['pair_plan']['shape_pairs'][0]
        record = inputs['calibrations'][0]
        record['config'] = formal.calibration(inputs['protocol'], pair['model'], inputs['gpu_uuids'],
            inputs['pools'], inputs['model_manifests'], pair['video_shape'], pair['image_shape'])
        record['config_sha256'] = canonical_hash(record['config'])
        record['budget']['sessions'][0]['identity']['config_sha256'] = record['config_sha256']
        inputs['bundle'] = formal.assemble(inputs['protocol'], pair['model'], [record], inputs['runtime_facts'])
        with self.assertRaisesRegex(ValueError, 'insufficient distinct development'):
            isolated.compile_case(inputs)

    def test_cpu_review_rejects_changed_order_repeat_input_or_deadline(self):
        inputs = self.inputs(); config = isolated.compile_case(inputs)
        changes = [lambda c: c['phases'][0]['matrix']['/policy'].reverse(),
                   lambda c: c['phases'][0].__setitem__('paired', False),
                   lambda c: c['phases'][0]['matrix']['/workload'][0][0].__setitem__('deadline_ms', 1),
                   lambda c: c['phases'][0]['matrix']['/workload'][0][0]['spec'].__setitem__('seed', 999),
                   lambda c: c['benchmark'].__setitem__('diagnostic_only', False)]
        for change in changes:
            cfg = copy.deepcopy(config); change(cfg)
            with self.subTest(change=change), self.assertRaises(ValueError):
                isolated.audit_config(cfg, inputs)

    def test_partial_run_keeps_all_never_started_cells_and_repeats(self):
        config = isolated.compile_case(self.inputs())
        row = copy.deepcopy(config['phases'][0]['matrix']['/workload'][0][0])
        row.update(phase='seed-201', cell=0, order='forward', attempt=1,
                   observation_id='seed-201/1/0', status='failed')
        identity = config['benchmark']['isolated_batch']['identity']
        record = dict(config=config, config_sha256=canonical_hash(config), requests=[row],
            source=dict(core_sha256=identity['core_sha256'], files=identity['python_sources']),
            budget=dict(sessions=[dict(identity=dict(gpu_uuids=['GPU-a', 'GPU-b'], config_sha256=canonical_hash(config)))]),
            phases=[dict(id='seed-201', attempt=1, status='failed', cells=[dict(cell=0, status='failed')])])
        result = isolated.coverage(config, record)
        self.assertEqual((result['expected_requests'], result['failed_or_incomplete'], result['unexecuted']), (96, 1, 95))
        self.assertFalse(result['qualification_eligible'])
        record['requests'].append(copy.deepcopy(row))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            isolated.coverage(config, record)

    def test_rehashed_run_cannot_fake_native_abba_observations_or_cell_policies(self):
        config = isolated.compile_case(self.inputs())
        identity = config['benchmark']['isolated_batch']['identity']
        record = dict(config=config, config_sha256=canonical_hash(config), requests=[], phases=[],
            source=dict(core_sha256=identity['core_sha256'], files=identity['python_sources']),
            budget=dict(sessions=[dict(identity=dict(gpu_uuids=['GPU-a', 'GPU-b'], config_sha256=canonical_hash(config)))]))
        for phase in config['phases']:
            executed = dict(id=phase['id'], attempt=1, status='completed', cells=[])
            for plan in [p for p in isolated.planned_cells(config) if p['phase'] == phase['id']]:
                executed['cells'].append(dict(cell=plan['cell'], order=plan['order'], policy=plan['policy'], status='completed'))
                for source in phase['matrix']['/workload'][0]:
                    row = copy.deepcopy(source)
                    row.update(phase=phase['id'], attempt=1, cell=plan['cell'], order=plan['order'],
                               observation_id=f"{phase['id']}/1/{plan['position']}", status='completed')
                    record['requests'].append(row)
            record['phases'].append(executed)
        self.assertEqual(isolated.coverage(config, record)['abba_cells_completed'], 12)
        changes = [lambda r: r.pop('phases'),
            lambda r: r['requests'][0].pop('observation_id'),
            lambda r: r['requests'][0].__setitem__('policy', 'wrong-policy'),
            lambda r: r['requests'][0].__setitem__('kind', 'image'),
            lambda r: r['requests'][0].__setitem__('class', 'wan21-720p'),
            lambda r: r['requests'][16].__setitem__('observation_id', 'seed-201/1/1'),
            lambda r: r['requests'][0].__setitem__('attempt', 2),
            lambda r: r['phases'][0]['cells'][0].__setitem__('policy', 'block-cfg2'),
            lambda r: r['phases'][0]['cells'][2].__setitem__('order', 'forward'),
            lambda r: r['phases'][0]['cells'][2].__setitem__('attempt', 2),
            lambda r: r['phases'][0].__setitem__('attempt', 2)]
        for change in changes:
            value = copy.deepcopy(record); change(value)
            value['sha256'] = canonical_hash(value)  # Rehashing cannot repair an execution contradiction.
            with self.subTest(change=change), self.assertRaises(ValueError):
                isolated.coverage(config, value)

    def test_hash_bound_manifest_and_cli_never_probe_gpu_or_overwrite(self):
        inputs = self.inputs()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, value):
                path = root / (name + '.json'); path.write_text(json.dumps(value))
                return dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest = dict(schema=isolated.SCHEMA, pair_id=inputs['pair_id'], gpu_uuids=inputs['gpu_uuids'])
            for key in ('protocol', 'pair_plan', 'bundle', 'model_manifests', 'identity', 'runtime_facts'):
                manifest[key] = write(key, inputs[key])
            manifest['pools'] = {k: write(k, v) for k, v in inputs['pools'].items()}
            manifest['calibrations'] = [write('calibration', inputs['calibrations'][0])]
            ref = write('inputs', manifest); output = root / 'config.json'
            args = ['p1_isolated_batch.py', '--inputs', str(root/'inputs.json'), '--inputs-sha256', ref['sha256'], '--output', str(output)]
            with patch.object(sys, 'argv', args), patch('cluster_config.allocated_uuids', side_effect=AssertionError('no GPU probe')):
                isolated.main()
                with self.assertRaises(FileExistsError):
                    isolated.main()
            (root/'identity.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                isolated.load_inputs(root/'inputs.json', ref['sha256'])


if __name__ == '__main__':
    unittest.main()
