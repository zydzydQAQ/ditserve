import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.research.dual_matrix import seal
from ditserve.research.migration_diagnostic_experiment import relative_error, roundtrip_equal, recompute_checks
from ditserve.research.migration_diagnostic_runner import pending_plan


class DiagnosticExecutionTests(unittest.TestCase):
    def trajectory(self):
        return dict(traces=[dict(step=i,state_sha256=str(i)) for i in range(1,33)],
                    signatures={'16':{'state_sha256':'16'},'17':{'state_sha256':'17'}},
                    latent_sha256='latent',output_sha256='output')

    def test_local_restore_requires_entire_trajectory_and_history(self):
        baseline=self.trajectory(); restored=copy.deepcopy(baseline)
        self.assertTrue(roundtrip_equal(baseline,restored))
        restored['traces'][18]['state_sha256']='wrong history'
        self.assertFalse(roundtrip_equal(baseline,restored))
        restored=copy.deepcopy(baseline); restored['signatures']['16']['state_sha256']='wrong'
        self.assertFalse(roundtrip_equal(baseline,restored))

    def test_prediction_errors_are_same_shape_finite_and_directional(self):
        a=np.array([3.,4.],dtype=np.float32); b=np.array([0.,4.],dtype=np.float32)
        value=relative_error(a,b)
        self.assertEqual(value['relative_l2'],.75); self.assertEqual(value['max_abs'],3.)
        self.assertTrue(relative_error(a,a)['exact_equal'])
        for invalid in (a.astype(np.float64),a.reshape(1,2),np.array([np.nan,4.],dtype=np.float32)):
            with self.assertRaises(ValueError):relative_error(invalid,b)

    def test_recompute_checks_rejects_mutated_shadow_source(self):
        runs={key:self.trajectory() for key in ('a100_stay','v100_stay','full_migrate','a100_local_restore','v100_local_restore')}
        shadow=dict(source=dict(before={'state_sha256':'16'},after={'state_sha256':'17'}),
            target=dict(before={'state_sha256':'16'},after={'state_sha256':'17'}),
            source_status=dict(residency='diagnostic_paused',signature={'state_sha256':'16'}))
        self.assertTrue(all(recompute_checks(runs,shadow).values()))
        shadow['source_status']['signature']['state_sha256']='mutated'
        self.assertFalse(recompute_checks(runs,shadow)['source_unchanged_and_paused'])

    def test_recompute_checks_does_not_require_cross_device_step_equality(self):
        runs={key:self.trajectory() for key in ('a100_stay','v100_stay','full_migrate','a100_local_restore','v100_local_restore')}
        runs['full_migrate']['signatures']['17']['state_sha256']='cross-device-numeric-difference'
        shadow=dict(source=dict(before={'state_sha256':'16'},after={'state_sha256':'17'}),
            target=dict(before={'state_sha256':'16'},after={'state_sha256':'cross-device-numeric-difference'}),
            source_status=dict(residency='diagnostic_paused',signature={'state_sha256':'16'}))
        self.assertTrue(all(recompute_checks(runs,shadow).values()))

    def test_pending_plan_cannot_borrow_lifetime_balance(self):
        frozen=dict(sha256='protocol',source_sha256='source')
        planning=seal(dict(status='fits',protocol_sha256='protocol',comparisons={
            'original':dict(total_gpu_seconds=100.),'math':dict(total_gpu_seconds=200.)}))
        ledger=dict(active=None,limit_gpu_seconds=28800,used_gpu_seconds=100,
            stage_used_gpu_seconds={'migration':90},reallocations=[dict(
                stage_allowances_gpu_seconds={'migration':350},baseline_stage_used={'migration':0})])
        with patch('ditserve.research.migration_diagnostic_runner.completed',return_value=False):
            result=pending_plan(Path('/tmp'),frozen,ledger,planning)
        self.assertEqual(result['status'],'budget_stopped')
        self.assertEqual(result['remaining_migration_gpu_seconds'],260)
        with patch('ditserve.research.migration_diagnostic_runner.completed',side_effect=lambda root,f,p:p=='original'):
            self.assertEqual(pending_plan(Path('/tmp'),frozen,ledger,planning)['status'],'fits')
        ledger['active']={'stage':'migration'}
        with self.assertRaises(ValueError):pending_plan(Path('/tmp'),frozen,ledger,planning)


if __name__=='__main__':unittest.main()
