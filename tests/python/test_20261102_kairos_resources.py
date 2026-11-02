"""Resource admission must respect live quota, fragmentation and missing evidence."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

LOCAL = Path(os.environ.get('DITSERVE_KAIROS_RUNTIME_ROOT',
    Path(__file__).resolve().parents[3] / '.local/ditserve-kairos-rewrite-runtime'))


@unittest.skipUnless((LOCAL / 'tools/kairos_queue.py').is_file(), 'local operator scripts')
class ResourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('resource_queue_test', LOCAL / 'tools/kairos_queue.py')
        cls.q = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.q)

    def node(self, name, free, **kwargs):
        return dict(name=name, ready=True, cordoned=False, gpu_healthy=True,
                    gpu_allocatable=8, gpu_allocated=8-free, **kwargs)

    def capacity(self, nodes=None, jobs=None, quota=16, cap=None):
        return self.q.capacity_from_records(dict(namespace='user-jjjkairos', gpu_quota=quota),
            nodes or [self.node('gpu01', 8), self.node('gpu02', 8)], jobs or [], 'user-jjjkairos', cap)

    def test_free_gpu_total_cannot_replace_single_node_fit(self):
        snap = self.capacity([self.node('gpu01', 4), self.node('gpu02', 4)])
        self.assertEqual(snap['cluster_free_gpus'], 8)
        self.assertFalse(self.q.capacity_fits(dict(workers=1, gpus_per_worker=8), snap))
        self.assertTrue(self.q.capacity_fits(dict(workers=2, gpus_per_worker=4), snap))

    def test_quota_override_can_only_reduce_live_quota(self):
        snap = self.capacity(quota=4, cap=16)
        self.assertEqual(snap['quota'], 4)
        self.assertFalse(self.q.capacity_fits(dict(workers=1, gpus_per_worker=8), snap))
        self.assertEqual(self.capacity(quota=16, cap=8)['quota'], 8)

    def test_other_projects_in_same_account_consume_quota(self):
        jobs = [dict(name='a05-other-project', namespace='user-jjjkairos', state='Running', total_gpus=12)]
        self.assertEqual(self.capacity(jobs=jobs)['account_free_gpus'], 4)

    def test_pending_job_cannot_double_spend_unreported_node_slots(self):
        snap = self.capacity(jobs=[dict(name='another-job', namespace='user-jjjkairos', state='Created', total_gpus=8)])
        self.assertFalse(self.q.capacity_fits(dict(workers=1, gpus_per_worker=8), snap))

    def test_zero_utilization_on_allocated_or_cordoned_node_is_not_free(self):
        busy = self.node('gpu01', 0, gpu_util_avg=0)
        cordoned = self.node('gpu02', 8)
        cordoned['cordoned'] = True
        snap = self.capacity([busy, cordoned])
        self.assertEqual(snap['cluster_free_gpus'], 0)

    def test_unknown_schema_and_health_fail_closed(self):
        with self.assertRaises(ValueError):
            self.q.capacity_from_records(dict(namespace='user-jjjkairos'), [], [], 'user-jjjkairos')
        node = self.node('gpu01', 8)
        del node['gpu_healthy']
        with self.assertRaises(ValueError):
            self.capacity([node])
        with self.assertRaises(ValueError):
            self.capacity(jobs=[dict(name='job', namespace='user-jjjkairos', state='Running')])

    def test_formal_without_original_evidence_is_rejected(self):
        spec = dict(env=[dict(name='DITSERVE_BENCHMARK_MODE', value='formal')])
        with self.assertRaisesRegex(ValueError, 'evidence receipt'):
            self.q.stage_gate(spec, None, None)

    def test_storage_exact_limits_and_df_h_check(self):
        for used, pct, stops in [(790, 79, False), (791, 79, True), (756, 80, True)]:
            calls = []
            def run(cmd, **kwargs):
                calls.append(cmd)
                return SimpleNamespace(returncode=0, stderr='', stdout=f'Filesystem 1G-blocks Used Available Use% Mounted on\n/dev/sdb 1024G {used}G 200G {pct}% /data\n')
            with patch.object(self.q.subprocess, 'run', side_effect=run):
                if stops:
                    with self.assertRaisesRegex(RuntimeError, 'stop threshold'):
                        self.q.storage_gate()
                else:
                    self.assertEqual(self.q.storage_gate()['used_gib'], used)
            self.assertEqual(calls[0][-3:], ['df', '-h', '/data'])


if __name__ == '__main__':
    unittest.main()
