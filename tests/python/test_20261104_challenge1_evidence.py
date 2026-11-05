#!/usr/bin/env python3
"""Unit and integration tests for Challenge 1 evidence modules (2026-11-04)."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / 'benchmarks', ROOT / 'benchmarks' / 'src', ROOT / 'workers', ROOT / 'tests' / 'python'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import challenge1_contrasts as c1
import cluster_config as ccfg
import cross_modal_memory_study as cmms
import external_xdit_scaling as xscale
import fit_native_profile as fnp
import native_serving as nserv
import pcie_collective_characterization as pcie
from test_native_serving import pool as _serving_pool, profile as _serving_profile
from ditserve.models.encoder_pool import ProactiveEncoderPool
from ditserve.models.sd35 import SD35Backend


CAL8_ROOT = (
    ROOT.parent
    / '.local'
    / 'ditserve-kairos-rewrite-runtime'
    / 'evidence'
    / '20261103-cal8-51740c0'
    / 'calibration-local'
)
WAN21_BUNDLE = CAL8_ROOT / 'wan21' / 'bundle' / 'source.json'
WAN22_BUNDLE = CAL8_ROOT / 'wan22' / 'bundle' / 'source.json'
IMAGE_POOL = CAL8_ROOT / 'sources' / 'image.json'
VIDEO_POOL = CAL8_ROOT / 'sources' / 'video.json'


class Challenge1EvidenceTests(unittest.TestCase):
    def test_form_tuple_and_legal_forms(self) -> None:
        self.assertEqual(pcie.form_tuple('sp1'), (1, 1))
        self.assertEqual(pcie.form_tuple('sp4'), (1, 4))
        self.assertEqual(pcie.form_tuple('cfg2'), (2, 1))
        self.assertEqual(pcie.form_tuple('cfg2sp4'), (2, 4))

        wan21_forms = pcie.legal_forms_for_shape('wan21-480p', max_gpus=8)
        self.assertNotIn('sp8', wan21_forms)
        self.assertIn('cfg2sp4', wan21_forms)

        wan22_forms = pcie.legal_forms_for_shape('wan22-small', max_gpus=8)
        self.assertIn('sp8', wan22_forms)
        self.assertIn('cfg2sp4', wan22_forms)

        sd35_forms = pcie.legal_forms_for_shape('sd35-1024', max_gpus=8)
        self.assertIn('sp8', sd35_forms)
        self.assertIn('cfg2sp4', sd35_forms)

    def test_per_step_comm_stats_wan_and_sd35(self) -> None:
        # Wan2.1 480p: 240 collectives in pure SP vs 120 in CFG2xSP
        sp4 = pcie.per_step_comm_stats('wan21-480p', 'sp4')
        cfg2sp2 = pcie.per_step_comm_stats('wan21-480p', 'cfg2sp2')
        self.assertEqual(sp4['gpus'], 4)
        self.assertEqual(cfg2sp2['gpus'], 4)
        self.assertEqual(sp4['a2a_collectives_per_rank'], 240)
        self.assertEqual(cfg2sp2['a2a_collectives_per_rank'], 120)
        self.assertLess(cfg2sp2['total_pcie_bus_mb_per_rank'], sp4['total_pcie_bus_mb_per_rank'])

        # SD3.5 1024: 192 collectives in pure SP vs 96 in CFG2xSP
        sd_sp8 = pcie.per_step_comm_stats('sd35-1024', 'sp8')
        sd_cfg2sp4 = pcie.per_step_comm_stats('sd35-1024', 'cfg2sp4')
        self.assertEqual(sd_sp8['gpus'], 8)
        self.assertEqual(sd_cfg2sp4['gpus'], 8)
        self.assertEqual(sd_sp8['a2a_collectives_per_rank'], 192)
        self.assertEqual(sd_cfg2sp4['a2a_collectives_per_rank'], 96)
        self.assertLess(sd_cfg2sp4['total_pcie_bus_mb_per_rank'], sd_sp8['total_pcie_bus_mb_per_rank'])

    def test_storage_gate_thresholds(self) -> None:
        self.assertEqual(pcie.STORAGE_MAX_GIB, 800)
        self.assertEqual(pcie.STORAGE_MAX_PCT, 80)
        # Non-existent local mount returns safe zero usage
        res = pcie.check_storage('/nonexistent-mount-for-test')
        self.assertEqual(res['used_gib'], 0)
        self.assertEqual(res['percent'], 0)

    def test_dominance_pruning_on_8gpu_bundles(self) -> None:
        if not (WAN21_BUNDLE.exists() and WAN22_BUNDLE.exists()):
            self.skipTest('8-GPU calibration bundles not present locally')
        report = pcie.build_decomposition_report([WAN21_BUNDLE, WAN22_BUNDLE], include_historical_dev=True)
        expected_pareto = {
            'wan21-480p': ['cfg2', 'cfg2sp3', 'cfg2sp4'],
            'wan21-720p': ['cfg2', 'cfg2sp2', 'cfg2sp3', 'cfg2sp4'],
            'wan22-small': ['cfg2', 'cfg2sp3', 'cfg2sp4'],
            'wan22-720p': ['cfg2', 'cfg2sp2', 'cfg2sp3', 'cfg2sp4'],
        }
        for sid, exp_forms in expected_pareto.items():
            sdata = report['video_shapes'][sid]
            self.assertEqual(
                sdata['retained_pareto_forms'],
                exp_forms,
                msg=f'Unexpected Pareto forms for {sid}: {sdata["retained_pareto_forms"]}',
            )
            for pure_f in ('sp2', 'sp3', 'sp4', 'sp6'):
                self.assertIn(pure_f, sdata['pruned_dominated_forms'])
            if sid.startswith('wan22'):
                self.assertIn('sp8', sdata['pruned_dominated_forms'])
            for contrast in sdata['same_gpu_cfg_vs_pure_sp']:
                self.assertGreater(contrast['cfg_sp_speedup_over_pure_sp'], 1.10)

    def test_external_xdit_scaling_topologies_and_step_summary(self) -> None:
        # Valid topologies for Wan2.1, Wan2.2, SD3.5
        self.assertEqual(xscale.validate_parallel_degrees('wan21', 1, 1, 1), 'sp1')
        self.assertEqual(xscale.validate_parallel_degrees('wan21', 8, 2, 4), 'cfg2sp4')
        self.assertEqual(xscale.validate_parallel_degrees('sd35', 8, 1, 8), 'sp8')
        self.assertEqual(xscale.validate_parallel_degrees('sd35', 8, 2, 4), 'cfg2sp4')

        # Wan2.1 (12 per-head groups in protocol) cannot divide by 8
        with self.assertRaises(ValueError):
            xscale.validate_parallel_degrees('wan21', 8, 1, 8)
        with self.assertRaises(ValueError):
            xscale.validate_parallel_degrees('sd35', 4, 2, 4)

        # Step timing summary (1 warmup + 27 steady steps for 28-step SD3.5)
        steps_ms = [120.0] + [40.0] * 27
        summary = xscale.summarize_step_timings(steps_ms, expected_steps=28, calls_per_step=1)
        self.assertEqual(summary['total_steps'], 28)
        self.assertEqual(summary['steady_sample_count'], 27)
        self.assertAlmostEqual(summary['median_steady_step_ms'], 40.0, places=4)

    def test_challenge1_contrasts_resolution_and_simulation(self) -> None:
        self.assertEqual(
            c1.model_forms('wan21', 8, 'pure_sp'),
            ['sp1', 'sp2', 'sp3', 'sp4', 'sp6'],
        )
        self.assertEqual(
            c1.model_forms('wan22', 8, 'pure_sp'),
            ['sp1', 'sp2', 'sp3', 'sp4', 'sp6', 'sp8'],
        )
        self.assertEqual(
            c1.model_forms('wan21', 8, 'pruned_cfg2xsp'),
            ['cfg2', 'cfg2sp2', 'cfg2sp3', 'cfg2sp4'],
        )

        if not (WAN21_BUNDLE.exists() and IMAGE_POOL.exists() and VIDEO_POOL.exists() and c1.DEFAULT_BINARY.exists()):
            self.skipTest('8-GPU bundle or compiled ditserve binary not present')

        suite = c1.run_challenge1_simulation_suite(
            WAN21_BUNDLE,
            IMAGE_POOL,
            VIDEO_POOL,
            arms=('pure-sp-elastic', 'edf-cfg2', 'static-cfg2sp4', 'cfg2-first', 'ditserve'),
            video_slo_scales=(0.45,),
            loads=(0.8,),
            seeds=(301,),
        )
        arms_045 = suite['slo_sweep'][0]['arms']
        # At tight SLO (0.45x SP1 ~ 0.88x CFG2), static CFG2 cannot meet video deadlines,
        # whereas DitServe and CFG2-First actively widen to CFG2SP3/CFG2SP4.
        self.assertGreater(arms_045['ditserve']['widened_video_requests_total'], 0)
        self.assertGreater(
            arms_045['ditserve']['overall_slo_attainment'],
            arms_045['pure-sp-elastic']['overall_slo_attainment'],
        )
        self.assertGreater(
            arms_045['ditserve']['video_slo_attainment'],
            arms_045['edf-cfg2']['video_slo_attainment'],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = c1.prepare_gpu_study_configs(
                WAN21_BUNDLE,
                IMAGE_POOL,
                VIDEO_POOL,
                Path(tmpdir),
                arms=('pure-sp-elastic', 'edf-cfg2', 'ditserve'),
                video_slo_scale=0.65,
                load=0.8,
                seed=301,
            )
            self.assertEqual(set(manifest['configs'].keys()), {'pure-sp-elastic', 'edf-cfg2', 'ditserve'})

    def test_s0_to_s4_cross_modal_memory_study_and_shm_pool(self) -> None:
        report = cmms.evaluate_all_strategies(gpus=8, shm_iterations=5)
        self.assertEqual(report['schema'], 'ditserve.cross-modal-memory-study.v1')
        strategies = report['strategies']
        self.assertEqual(
            list(strategies.keys()),
            [
                's0_naive_gpu',
                's1_weight_swap',
                's2_wan_cpu_sd35_gpu',
                's3_both_encoders_cpu',
                's4_disaggregated_pool',
            ],
        )

        # S0: 100% hard OOM on wan21-720p, wan22-small, wan22-720p; fragmentation OOM on wan21-480p
        s0 = strategies['s0_naive_gpu']
        self.assertFalse(s0['oom_free'])
        self.assertEqual(set(s0['hard_oom_shapes']), {'wan21-720p', 'wan22-small', 'wan22-720p'})
        self.assertEqual(s0['redundant_heavy_t5_umt5_8gpu_gib'], 155.6)

        # S1: Avoids OOM, but incurs >2,700 ms PCIe weight swap stall per preemption
        s1 = strategies['s1_weight_swap']
        self.assertGreater(s1['weight_swap_ms_range'][0], 2700.0)
        self.assertGreater(s1['weight_swap_ms_range'][1], 3900.0)

        # S2: Current default has ~15,000 ms Wan CPU encode and fragmentation OOM risk on wan22-720p
        s2 = strategies['s2_wan_cpu_sd35_gpu']
        self.assertIn('wan22-720p', s2['fragmentation_oom_shapes'])
        self.assertGreaterEqual(s2['wan_prompt_encode_ms_range'][0], 14000.0)

        # S3: Both heavy encoders on CPU -> zero OOM (11.9-22.4 GiB headroom), but ~3,800 ms SD3.5 CPU encode
        s3 = strategies['s3_both_encoders_cpu']
        self.assertTrue(s3['oom_free'])
        self.assertEqual(s3['static_gpu_weights_gib']['wan21_plus_sd35'], 9.17)
        self.assertEqual(s3['static_gpu_weights_gib']['wan22_plus_sd35'], 16.92)
        self.assertGreaterEqual(s3['sd35_prompt_encode_ms_range'][0], 3600.0)

        # S4: Disaggregated/sharded GPU pool + /dev/shm pinned prefetch -> zero OOM, <80 ms encode, 0 ms weight swap
        s4 = strategies['s4_disaggregated_pool']
        self.assertTrue(s4['oom_free'])
        self.assertEqual(s4['static_gpu_weights_gib']['wan21_plus_sd35'], 9.17)
        self.assertEqual(s4['static_gpu_weights_gib']['wan22_plus_sd35'], 16.92)
        self.assertAlmostEqual(s4['saved_static_hbm_per_gpu_vs_s0_gib'], 19.45, delta=0.02)
        self.assertAlmostEqual(s4['saved_static_hbm_8gpu_vs_s0_gib'], 155.6, delta=0.15)
        self.assertEqual(s4['weight_swap_ms_range'], [0.0, 0.0])
        self.assertLessEqual(s4['wan_prompt_encode_ms_range'][1], 80.0)
        self.assertLessEqual(s4['sd35_prompt_encode_ms_range'][1], 45.0)
        self.assertLess(s4['pinned_shm_h2d_fetch_ms_range'][1], 0.5)

        # Verify live /dev/shm microbenchmark hit rate
        shm_bench = report['shm_encoder_pool_microbenchmark']
        self.assertEqual(shm_bench['pool_hit_rate'], 1.0)
        self.assertEqual(shm_bench['sd35_embedding_bundle']['cfg_pair_total_bytes'], 4202496)
        self.assertEqual(shm_bench['wan_embedding_bundle']['total_bytes'], 4194304)

    def test_s4_cluster_config_and_sd35_shm_hit_without_restoring_t5(self) -> None:
        uuids = [f'GPU-{i}' for i in range(4)]
        # Memory strategies are opt-in: the default preflight keeps the legacy worker options.
        self.assertIsNone(ccfg.DEFAULT_MEMORY_STRATEGY)
        cfg_default = ccfg.build_config('wan22', uuids)
        self.assertNotIn('memory_strategy', cfg_default['phases'][0]['options'])
        self.assertNotIn('memory_strategy', cfg_default['benchmark'])
        self.assertFalse(cfg_default['phases'][0]['options']['tiered_memory'])
        self.assertEqual(set(cfg_default['workers']['wan0']['config']['options']), {'execution', 'cfg_fold'})
        self.assertEqual(cfg_default['workers']['image0']['config']['options'], {'profile': False, 'cache': False})
        # S4 claims a CPU SD3.5 T5 encoder and tiered memory, which no worker implements:
        # it stays analytical (cross_modal_memory_study.py) and cannot reach a GPU config.
        with self.assertRaisesRegex(ValueError, 'cross_modal_memory_study'):
            ccfg.build_config('wan22', uuids, memory_strategy='s4_disaggregated_pool')
        self.assertEqual(ccfg.memory_strategy_descriptor('s4_disaggregated_pool')['code'], 'S4')

        # Verify SD35Backend._ensure_staged hits a /dev/shm slab staged by another process
        # WITHOUT calling restore_encoders() (keeping T5-XXL off the denoising GPU).
        had_torch = 'torch' in sys.modules
        torch_mod, _ = cmms._ensure_torch_or_shim()
        try:
            with tempfile.TemporaryDirectory() as tmp_shm:
                producer = ProactiveEncoderPool({'shm_dir': tmp_shm, 'write_shm': True, 'pin_memory': False})
                consumer = ProactiveEncoderPool({'shm_dir': tmp_shm, 'write_shm': True, 'pin_memory': False})
                dummy = object.__new__(SD35Backend)
                dummy.config = {'model_path': 'sd35', 'dtype': 'bfloat16', 'options': {}}
                dummy.encoder_pool = consumer
                dummy.encoders_offloaded = True
                restore_called = False

                def _forbid_restore(*_args, **_kwargs):
                    nonlocal restore_called
                    restore_called = True
                    raise AssertionError('restore_encoders() should not be called on /dev/shm hit!')

                dummy.restore_encoders = _forbid_restore  # type: ignore[method-assign]
                key = dummy._prompt_key('a lighthouse at dusk', 'blurry', 256, True)
                producer.stage_embeddings(
                    key,
                    {'prompt_embeds': torch_mod.empty((1, 256, 4096), dtype=torch_mod.bfloat16, device='cpu')},
                    pin_memory=False,
                    write_shm=True,
                )
                self.assertNotIn(key, consumer._entries)
                self.assertTrue(consumer.has_key(key))
                stage_info = dummy._ensure_staged('a lighthouse at dusk', 'blurry', 256, True)
                self.assertTrue(stage_info['cached'])
                self.assertFalse(stage_info['staged'])
                self.assertFalse(restore_called)
        finally:
            if not had_torch and isinstance(sys.modules.get('torch'), cmms._CpuTorchModuleShim):
                sys.modules.pop('torch', None)

    def test_continuous_batch_multi_value_config_and_serving(self) -> None:
        self.assertEqual(ccfg.CONTINUOUS_BATCH_SIZES, (1, 2, 3, 4, 6, 8))
        # Continuous batching is explicit opt-in; the default is off.
        self.assertEqual(ccfg.DEFAULT_CONTINUOUS_BATCH, 0)

        uuids = [f'GPU-{i}' for i in range(4)]
        default_cfg = ccfg.build_config('wan22', uuids)
        for key in ('continuous_batch', 'sequence_packing', 'bubble_free'):
            self.assertNotIn(key, default_cfg['phases'][0]['options'])
            self.assertNotIn(key, default_cfg['workers']['image0']['config']['options'])
        self.assertNotIn('continuous_batch', default_cfg['benchmark'])
        self.assertNotIn('batch_step_ms', default_cfg['phases'][0]['profile']['classes']['sd35-preflight'])

        cfg_off = ccfg.build_config('wan21', uuids, continuous_batch=0)
        self.assertNotIn('continuous_batch', cfg_off['phases'][0]['options'])
        self.assertEqual(cfg_off['workers']['image0']['config']['options'], {'profile': False, 'cache': False})
        with self.assertRaises(ValueError):
            ccfg.build_config('wan21', uuids, continuous_batch=0, sequence_packing=True)

        for cb in (1, 2, 3, 4, 6, 8):
            cfg_cb = ccfg.build_config(
                'wan21',
                uuids,
                continuous_batch=cb,
                sequence_packing=(cb != 2),
                bubble_free=(cb != 3),
            )
            self.assertEqual(cfg_cb['phases'][0]['options']['continuous_batch'], cb)
            self.assertEqual(cfg_cb['phases'][0]['options']['sequence_packing'], cb != 2)
            self.assertEqual(cfg_cb['phases'][0]['options']['bubble_free'], cb != 3)
            image_options = cfg_cb['workers']['image0']['config']['options']
            self.assertTrue(image_options['continuous_batch'])
            self.assertEqual(image_options['sequence_packing'], cb != 2)
            self.assertEqual(image_options['bubble_free'], cb != 3)
            self.assertEqual(cfg_cb['benchmark']['continuous_batch']['max_requests'], cb)
            self.assertEqual(
                set(cfg_cb['phases'][0]['profile']['classes']['sd35-preflight']['batch_step_ms'].keys()),
                {'1', '2', '3', '4', '6', '8'},
            )

        for invalid_cb in (-1, 9, True):
            with self.assertRaises(ValueError):
                ccfg.build_config('wan21', uuids, continuous_batch=invalid_cb)

        pools = {'image': _serving_pool('diffusiondb'), 'video': _serving_pool('vidprom')}
        # Legacy fixture without batch_step_ms defaults to eff_cb=0 unless explicitly requested
        legacy_srv = nserv.build_serving_config(
            'wan22',
            [f'GPU-{i}' for i in range(8)],
            _serving_profile(),
            pools,
            seed=201,
            load=1.0,
            pattern='poisson',
            video_fraction=0.2,
            count=20,
        )
        self.assertNotIn('continuous_batch', legacy_srv['phases'][0]['options'])

        # A profile with batch_step_ms never enables continuous batching by itself.
        cb_prof = _serving_profile()
        cb_prof['classes']['sd35-preflight'].update(
            steps=28,
            prepare_ms=15.0,
            finish_ms=65.0,
            batch_step_ms=ccfg.default_sd35_batch_step_ms(250.0, 28, resolution=1024),
        )
        auto_cb_srv = nserv.build_serving_config(
            'wan22',
            [f'GPU-{i}' for i in range(8)],
            cb_prof,
            pools,
            seed=201,
            load=1.0,
            pattern='poisson',
            video_fraction=0.2,
            count=20,
        )
        self.assertNotIn('continuous_batch', auto_cb_srv['phases'][0]['options'])
        self.assertNotIn('continuous_batch', auto_cb_srv['benchmark'])
        self.assertEqual(auto_cb_srv['workers']['image0']['config']['options'], {'profile': False, 'cache': False})
        opt_in_srv = nserv.build_serving_config(
            'wan22',
            [f'GPU-{i}' for i in range(8)],
            cb_prof,
            pools,
            seed=201,
            load=1.0,
            pattern='poisson',
            video_fraction=0.2,
            count=20,
            continuous_batch=4,
        )
        self.assertEqual(opt_in_srv['phases'][0]['options']['continuous_batch'], 4)
        self.assertTrue(opt_in_srv['phases'][0]['options']['sequence_packing'])
        self.assertTrue(opt_in_srv['phases'][0]['options']['bubble_free'])
        self.assertEqual(opt_in_srv['benchmark']['continuous_batch']['max_requests'], 4)
        self.assertEqual(set(opt_in_srv['benchmark']['continuous_batch']['cost_provenance'].values()), {'profile'})

        # Explicit continuous_batch across 1..8 fills a missing batch_step_ms with a labelled synthetic table
        for cb in (1, 2, 3, 4, 6, 8):
            srv_cb = nserv.build_serving_config(
                'wan22',
                [f'GPU-{i}' for i in range(8)],
                _serving_profile(),
                pools,
                seed=201,
                load=1.0,
                pattern='poisson',
                video_fraction=0.2,
                count=20,
                continuous_batch=cb,
                sequence_packing=False,
                bubble_free=True,
            )
            self.assertEqual(srv_cb['phases'][0]['options']['continuous_batch'], cb)
            self.assertFalse(srv_cb['phases'][0]['options']['sequence_packing'])
            self.assertTrue(srv_cb['phases'][0]['options']['bubble_free'])
            self.assertIn('batch_step_ms', srv_cb['phases'][0]['profile']['classes']['sd35-preflight'])
            self.assertEqual(
                srv_cb['benchmark']['continuous_batch']['cost_provenance']['batch_step_ms'],
                'synthetic_default_table',
            )
            self.assertEqual(
                srv_cb['phases'][0]['profile']['source']['image_cost_provenance']['batch_step_ms'],
                'synthetic_default_table',
            )

        # Mutual exclusion between continuous_batch > 0 and atomic image_batch > 1
        measured_atomic = _serving_profile()
        measured_atomic['classes']['sd35-preflight']['batch_ms'] = {'2': 400.0}
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            nserv.build_serving_config(
                'wan22',
                [f'GPU-{i}' for i in range(8)],
                measured_atomic,
                pools,
                seed=201,
                load=1.0,
                pattern='poisson',
                video_fraction=0.2,
                count=20,
                image_batch=2,
                continuous_batch=4,
            )

        # fit_native_profile writes only measured continuous-batch costs (no synthetic padding)
        spec = dict(frames=49, steps=20, height=512, width=896, guidance_scale=5.0)
        model_cfg = {
            'transformer_config': {
                'patch_size': [1, 2, 2],
                'num_attention_heads': 24,
                'attention_head_dim': 128,
                'num_layers': 30,
                'out_channels': 48,
            },
            'vae_config': {'scale_factor_temporal': 4, 'scale_factor_spatial': 16},
        }
        arch = fnp.architecture(model_cfg, spec)
        policies = list(fnp.FIT_POLICIES)
        requests = []
        for idx, pol in enumerate(policies):
            step_ms = fnp.predict(500.0, arch, *fnp.form_of(pol), 12.0, 1.0, 0.05)
            requests.append({
                'cell': idx,
                'kind': 'video',
                'status': 'completed',
                'model_ms': 20 * step_ms + 100.0,
                'outputs': [{'model_ms': 100.0}],
                'spec': spec,
                'encode_model_ms': 20.0,
                'advance_samples': [{'steady': True, 'model_ms': step_ms}],
            })
        requests.append({
            'cell': 0,
            'kind': 'image',
            'status': 'completed',
            'model_ms': 1000.0,
            'spec': {'steps': 28, 'height': 1024, 'width': 1024},
        })
        record = {
            'experiment': 'kairos-wan22-native-calibration',
            'run_id': 'cb-fit-test',
            'status': 'completed',
            'phases': [{
                'cells': [
                    {
                        'config': {'policy': pol},
                        'continuous_batch': {
                            'samples': [
                                {'batch_size': 1, 'model_ms': 31.5},
                                {'batch_size': 4, 'model_ms': 82.0},
                            ]
                        } if idx == 0 else {},
                    }
                    for idx, pol in enumerate(policies)
                ]
            }],
            'requests': requests,
        }
        fitted = fnp.build_profile(record, {'models': {'Wan2.2-TI2V-5B': model_cfg}}, 'Wan2.2-TI2V-5B')
        sd_fitted = fitted['classes']['sd35-preflight']
        self.assertEqual(sd_fitted['steps'], 28)
        self.assertNotIn('prepare_ms', sd_fitted)
        self.assertNotIn('finish_ms', sd_fitted)
        self.assertEqual(sd_fitted['batch_step_ms'], {'1': 31.5, '4': 82.0})
        self.assertEqual(
            fitted['source']['image_cost_provenance']['batch_step_ms'], 'measured_continuous_batch_rounds'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ccfg_out = root / 'preflight-cb6.json'
            with patch.object(
                sys,
                'argv',
                [
                    'cluster_config.py',
                    '--model',
                    'wan21',
                    '--gpu-uuids',
                    ','.join(uuids),
                    '--continuous-batch',
                    '6',
                    '--no-sequence-packing',
                    '--bubble-free',
                    '--output',
                    str(ccfg_out),
                ],
            ):
                ccfg.main()
            ccfg_loaded = json.loads(ccfg_out.read_text(encoding='utf-8'))
            self.assertEqual(ccfg_loaded['phases'][0]['options']['continuous_batch'], 6)
            self.assertFalse(ccfg_loaded['phases'][0]['options']['sequence_packing'])
            self.assertTrue(ccfg_loaded['phases'][0]['options']['bubble_free'])

            (root / 'profile.json').write_text(json.dumps(_serving_profile()), encoding='utf-8')
            (root / 'diffusiondb.json').write_text(json.dumps(pools['image']), encoding='utf-8')
            (root / 'vidprom.json').write_text(json.dumps(pools['video']), encoding='utf-8')
            nserv_out = root / 'serving-cb8.json'
            with patch.object(
                sys,
                'argv',
                [
                    'native_serving.py',
                    '--model',
                    'wan22',
                    '--profile',
                    str(root / 'profile.json'),
                    '--datasets',
                    str(root),
                    '--gpu-uuids',
                    ','.join(f'GPU-{i}' for i in range(8)),
                    '--count',
                    '10',
                    '--continuous-batch',
                    '8',
                    '--sequence-packing',
                    '--no-bubble-free',
                    '--output',
                    str(nserv_out),
                ],
            ):
                nserv.main()
            nserv_loaded = json.loads(nserv_out.read_text(encoding='utf-8'))
            self.assertEqual(nserv_loaded['phases'][0]['options']['continuous_batch'], 8)
            self.assertTrue(nserv_loaded['phases'][0]['options']['sequence_packing'])
            self.assertFalse(nserv_loaded['phases'][0]['options']['bubble_free'])


if __name__ == '__main__':
    unittest.main()
