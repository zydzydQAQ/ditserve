"""CPU process fixtures for native elastic orchestration, never CUDA execution.

A temporary package replaces both Wan and GPU inventory with explicit fixtures.
This exercises the C++ protocol/ownership path, not tensor or NCCL correctness.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from ditserve.runtime import PACKAGE_ROOT, ROOT
from ditserve.runtime.client import CoreBridge


FIXTURE = r'''
import json
import time
from pathlib import Path
from ditserve.models.mock import MockBackend

class WanBackend(MockBackend):
    def __init__(self, config):
        super().__init__(config)
        self.rank = None

    def prepare(self, request):
        result = super().prepare(request)
        state = self.states[result['state_id']]
        state.update(group=[self.rank], ownership_epoch=0, quantum_seq=0)
        result['progress'] = self._progress(state)
        return result

    @staticmethod
    def _progress(state):
        return dict(MockBackend._progress(state), group=state.get('group'),
                    ownership_epoch=state.get('ownership_epoch', 0), quantum_seq=state.get('quantum_seq', 0))

    def advance(self, request):
        state = self.state(request)
        if request['expected_epoch'] != state['ownership_epoch'] or request['quantum_seq'] != state['quantum_seq']:
            raise ValueError('fixture advance epoch/sequence mismatch')
        result = super().advance(request)
        state['quantum_seq'] += 1
        result['progress'] = self._progress(state)
        if self.config.get('options', {}).get('mismatch_cursor') and self.rank == 1:
            result['progress']['quantum_seq'] += 1
        return result

    def distributed(self, request):
        if request['action'] == 'init':
            self.rank = request['rank']
            return dict(initialized=True, gpu_validated=False, fixture_only=True)
        if request['action'] != 'transition':
            raise ValueError('unsupported fixture action')
        source, target = request['source_group'], request['target_group']
        key, path = request['state_id'], Path(request['path'])
        if self.rank == min(source):
            state = self.states[key]
            if state['ownership_epoch'] != request['expected_epoch'] or state['unit'] % state['blocks']:
                raise ValueError('fixture transfer outside committed step')
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(json.dumps(state))
            tmp.replace(path)
        until = time.monotonic() + 2
        while not path.exists():
            if time.monotonic() > until:
                raise RuntimeError('fixture transfer did not receive payload')
            time.sleep(.001)
        if self.config.get('options', {}).get('fail_transition') and self.rank == 1:
            raise RuntimeError('injected fixture transfer failure')
        state = json.loads(path.read_text())
        state.update(group=target, ownership_epoch=request['expected_epoch'] + 1)
        if self.rank in target:
            self.states[key] = state
        else:
            self.states.pop(key, None)
        return dict(state_id=key, group=target, ownership_epoch=state['ownership_epoch'],
                    progress=self._progress(state) if self.rank in target else None)
'''


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class NativeElasticProcessTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.package = self.root / 'python'
        shutil.copytree(PACKAGE_ROOT / 'ditserve', self.package / 'ditserve',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (self.package / 'ditserve/models/wan/backend.py').write_text(FIXTURE)
        path = self.package / 'ditserve/runtime/worker.py'
        source = path.read_text()
        source = source.replace('if __name__ == "__main__":', '''
# CPU TEST FIXTURE ONLY: no actual GPU inventory or CUDA calls occur.
def occupancy(request, mock=False):
    return dict(status="exclusive_observed", owned=[], foreign=[], fixture_only=True,
                occupancy_backend=request.get("occupancy_backend", "nvidia_smi"))

if __name__ == "__main__":''')
        path.write_text(source)
        env = mock.patch.dict(os.environ, {'PYTHONPATH': os.pathsep.join(
            [str(PACKAGE_ROOT), str(ROOT / '.cache/tools'), os.environ.get('PYTHONPATH', '')])})
        env.start()
        self.addCleanup(env.stop)

    def run_case(self, failure=None, dynamic=False):
        workers = {}
        for rank in (0, 1):
            workers['wan%d' % rank] = dict(backend='wan', device='cuda:0',
                gpu_uuid='GPU-00000000-0000-0000-0000-%012d' % rank,
                config=dict(options=dict(execution='block', **({failure: True} if failure else {}))))
            workers['image%d' % rank] = dict(backend='mock')
        profile = dict(provenance='synthetic', links={'2': dict(gbps=10, latency_ms=0)}, sp_efficiency=1, resize_ms=0,
            classes=dict(video=dict(kind='video', steps=3, blocks_per_step=4, step_compute_ms=20,
                prepare_ms=0, encode_ms=0, comm_mb_per_step=0, collectives_per_step=0, finish_ms=1),
                image=dict(kind='image', service_ms=1)))
        phase = dict(id='elastic', kind='cluster', policy='block_elastic', profile=profile,
            real_sp=dict(enabled=True, degrees=[1, 2], transport='host_file_v1', timeout_seconds=5),
            options=dict(quantum_ms=7), text_prefetch_depth=0,
            gpus=[dict(video='wan%d' % rank, image='image%d' % rank) for rank in (0, 1)],
            workload=[dict(request_id='video', kind='video', **{'class': 'video'}, arrival_ms=0, slo_ms=10000,
                           spec=dict(steps=3, blocks_per_step=4, seed=7, prompt='fixture'))])
        if dynamic:
            phase['policy'] = 'step_elastic'
            profile['classes']['video']['step_compute_ms'] = 80
            phase['workload'][0]['spec']['steps'] = 8
            phase['workload'][0]['steps_total'] = 8
            phase['workload'].append(dict(request_id='second', kind='video', **{'class': 'video'},
                arrival_ms=30, slo_ms=11000, steps_total=1,
                spec=dict(steps=1, blocks_per_step=4, seed=8, prompt='second fixture')))
            for rank in (0, 1):
                workers['wan%d' % rank]['config']['options']['costs'] = dict(step_ms=80)
        config = dict(schema='ditserve.study.v1', experiment='elastic-fixture', workers=workers,
            budget=dict(limit_gpu_seconds=100, watchdog_seconds=10), phases=[phase])
        return CoreBridge().call('run', dict(config=config, run_id='fixture', root=str(self.root),
            package_root=str(self.package), work_dir=str(self.root / 'run'), results_dir=str(self.root / 'results'),
            ledger=str(self.root / 'budget.json'), allow_gpu=True, prune=False))

    def test_initial_replication_block_cursor_and_leader_decode(self):
        result = self.run_case()
        self.assertEqual(result['status'], 'completed', result.get('failures'))
        row = result['requests'][0]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual([(t['source_group'], t['target_group'], t['ownership_epoch']) for t in row['transitions']],
                         [([0], [0, 1], 1), ([0, 1], [0], 2)])
        self.assertEqual(row['operation_counts']['prepare'], 1)
        self.assertEqual(row['operation_counts']['finish'], 1)
        self.assertGreater(row['operation_counts']['advance'], 6)
        self.assertIsNone(result['budget']['active'])
        self.assertTrue(all(s['cleanup_verified'] for s in result['budget']['sessions']))

    def test_transfer_failure_settles_and_records_request_failure(self):
        result = self.run_case('fail_transition')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('injected fixture transfer failure', json.dumps(result['failures']))
        self.assertEqual(result['requests'][0]['status'], 'failed')
        self.assertIsNone(result['budget']['active'])
        self.assertTrue(all(s['cleanup_verified'] for s in result['budget']['sessions']))

    def test_arriving_request_shrinks_then_regrows_committed_state(self):
        result = self.run_case(dynamic=True)
        self.assertEqual(result['status'], 'completed', result.get('failures'))
        rows = {row['request_id']: row for row in result['requests']}
        transitions = [(t['source_group'], t['target_group']) for t in rows['video']['transitions']]
        self.assertEqual(transitions, [([0], [0, 1]), ([0, 1], [0]), ([0], [0, 1]), ([0, 1], [0])])
        self.assertEqual(rows['video']['operation_counts']['prepare'], 1)
        self.assertEqual(rows['second']['operation_counts']['prepare'], 1)
        self.assertEqual(rows['video']['operation_counts']['finish'], 1)
        self.assertLess(rows['second']['start_ms'], rows['video']['finish_ms'])
        self.assertIsNone(result['budget']['active'])
        self.assertTrue(all(s['cleanup_verified'] for s in result['budget']['sessions']))

    def test_rank_cursor_disagreement_rejects_result(self):
        result = self.run_case('mismatch_cursor')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('ownership/sequence mismatch', json.dumps(result['failures']))
        self.assertEqual(result['requests'][0]['status'], 'failed')
        self.assertIsNone(result['budget']['active'])


if __name__ == '__main__':
    unittest.main()
