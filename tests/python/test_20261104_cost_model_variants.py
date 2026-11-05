#!/usr/bin/env python3
"""Tests for benchmarks/cost_model_variants.py (2026-11-04, item cost_model_refit).

Covers: exact agreement with fit_native_profile.predict/fit_links, recovery of
known parameters from synthetic data, leave-one-form-out isolation, and
read-only reproduction of the reported cal8 held-out errors.
"""

from pathlib import Path
import json
import math
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / 'benchmarks',):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cost_model_variants as cmv  # noqa: E402

CAL8_ROOT = (ROOT.parent / '.local' / 'ditserve-kairos-rewrite-runtime' / 'evidence'
             / '20261103-cal8-51740c0' / 'calibration-local')
WAN21_BUNDLE = CAL8_ROOT / 'wan21' / 'bundle' / 'source.json'
WAN22_BUNDLE = CAL8_ROOT / 'wan22' / 'bundle' / 'source.json'

ARCH = {"tokens": 32760, "dim": 1536, "layers": 30, "comm_mb_per_step": 24153.2928,
        "collectives_per_step": 240, "cfg_exchange_mb": 4.19328}
FORMS = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 6), (1, 8)]


def policy_of(cfg, sp):
    if cfg == 2:
        return 'static_cfg2' if sp == 1 else f'static_cfg2sp{sp}'
    return f'static_sp{sp}'


def synthetic_rows(model, params, t1=4000.0, arch=ARCH, forms=FORMS, dataset='synthetic'):
    return [{"policy": policy_of(c, s), "cfg": c, "sp": s, "t1": t1, "arch": arch, "dataset": dataset,
             "step_ms": cmv.model_predict(model, params, t1, arch, c, s)} for c, s in forms]


class BaselineReplicaTest(unittest.TestCase):
    def test_predict_matches_fit_native_profile(self):
        import fit_native_profile as fnp
        rng = random.Random(7)
        for _ in range(300):
            cfg = rng.choice([1, 2])
            sp = rng.choice([1, 2, 3, 4, 6, 8])
            args = (rng.uniform(500, 20000), ARCH, cfg, sp, rng.uniform(0.25, 50), rng.choice(cmv.GRID_GAMMA),
                    rng.choice(cmv.GRID_LATENCY))
            self.assertEqual(cmv.baseline_predict(*args), fnp.predict(*args))

    def test_grid_fit_matches_fit_links_on_synthetic_rows(self):
        import fit_native_profile as fnp
        truth = {"bandwidth_inf": 6.0, "gamma": 1.0, "latency": 0.2}
        rows = synthetic_rows('baseline_grid', truth)
        rng = random.Random(3)
        videos = [{"policy": r["policy"], "form": (r["cfg"], r["sp"]),
                   "step_ms": r["step_ms"] * (1.0 + rng.uniform(-0.03, 0.03))} for r in rows]
        videos[0]["step_ms"] = 4000.0
        step_compute, expected, _ = fnp.fit_links(videos, ARCH)
        fit_rows = [{"policy": v["policy"], "cfg": v["form"][0], "sp": v["form"][1], "t1": step_compute,
                     "arch": ARCH, "dataset": "s", "step_ms": v["step_ms"]}
                    for v in videos if v["policy"] in cmv.FIT_POLICIES]
        params, _ = cmv.fit_model('baseline_grid', fit_rows)
        self.assertEqual((params["bandwidth_inf"], params["gamma"], params["latency"]), tuple(expected))
        self.assertEqual(tuple(cmv.FIT_POLICIES), tuple(sorted(fnp.FIT_POLICIES, key=cmv.FIT_POLICIES.index)))

    def test_architecture_matches_reported_bundle(self):
        transformer = {"patch_size": [1, 2, 2], "num_attention_heads": 12, "attention_head_dim": 128,
                       "num_layers": 30, "out_channels": 16}
        arch = cmv.architecture(transformer, {}, {"frames": 81, "height": 480, "width": 832})
        for key, value in ARCH.items():
            self.assertAlmostEqual(arch[key], value, places=6)


class SyntheticRecoveryTest(unittest.TestCase):
    def assert_recovers(self, model, truth, tolerance):
        rows = synthetic_rows(model, truth)
        params, loss = cmv.fit_model(model, rows, objective='rel_mse')
        self.assertLess(loss, 1e-8)
        for key, value in truth.items():
            self.assertAlmostEqual(params[key], value, delta=tolerance[key], msg=f'{model}.{key}')
        for row in rows:
            predicted = cmv.model_predict(model, params, row["t1"], row["arch"], row["cfg"], row["sp"])
            self.assertLess(abs(predicted - row["step_ms"]) / row["step_ms"], 1e-3)

    def test_amdahl_sp_recovers_known_parameters(self):
        self.assert_recovers('amdahl_sp', {"bandwidth_inf": 5.0, "gamma": 0.0, "latency": 1.5, "serial": 0.12},
                             {"bandwidth_inf": 0.1, "gamma": 0.05, "latency": 0.05, "serial": 0.005})

    def test_bus_recovers_known_parameters(self):
        self.assert_recovers('bus', {"link_gbps": 8.0, "bus_gbps": 30.0, "latency": 0.8},
                             {"link_gbps": 0.2, "bus_gbps": 1.0, "latency": 0.05})

    def test_baseline_continuous_recovers_off_grid_latency(self):
        truth = {"bandwidth_inf": 3.3, "gamma": 0.0, "latency": 1.7}
        rows = synthetic_rows('baseline_cont', truth)
        params, _ = cmv.fit_model('baseline_cont', rows, objective='abs_mse')
        self.assertAlmostEqual(params["latency"], 1.7, delta=0.05)
        self.assertAlmostEqual(params["bandwidth_inf"], 3.3, delta=0.1)
        grid, _ = cmv.fit_model('baseline_grid', rows)
        self.assertEqual(grid["latency"], max(cmv.GRID_LATENCY))  # grid cannot represent 1.7 ms

    def test_identifiability_flags_parameter_count_against_points(self):
        truth = {"bandwidth_inf": 5.0, "gamma": 0.0, "latency": 1.5, "serial": 0.12}
        few = synthetic_rows('amdahl_sp', truth, forms=[(1, 1), (2, 1), (1, 2), (2, 2), (1, 4)])
        diag = cmv.identifiability('amdahl_sp', truth, few)
        self.assertEqual(diag["informative_points"], 4)
        self.assertEqual(diag["status"], 'unidentifiable')
        many = synthetic_rows('bus', {"link_gbps": 8.0, "bus_gbps": 30.0, "latency": 0.8})
        self.assertIn(cmv.identifiability('bus', {"link_gbps": 8.0, "bus_gbps": 30.0, "latency": 0.8},
                                          many)["status"], ('ok', 'weak'))


class LofoIsolationTest(unittest.TestCase):
    def test_left_out_row_is_never_seen_by_fitter(self):
        rows = synthetic_rows('bus', {"link_gbps": 8.0, "bus_gbps": 30.0, "latency": 0.8})
        seen = []

        def spy(name, training, objective='abs_mse', **kwargs):
            seen.append([(r["dataset"], r["policy"]) for r in training])
            return cmv.fit_model(name, training, objective=objective, starts=3, max_iter=200)

        folds = cmv.leave_one_form_out('bus', rows, objective='rel_mse', fit=spy)
        self.assertEqual(len(folds), len(rows) - 1)  # SP1 anchor never left out
        for fold, training in zip(folds, seen):
            self.assertNotIn((fold["dataset"], fold["policy"]), training)
            self.assertIn(('synthetic', 'static_sp1'), training)
            self.assertEqual(fold["train_n"], len(rows) - 1)

    def test_corrupting_left_out_measurement_does_not_change_its_prediction(self):
        rows = synthetic_rows('baseline_cont', {"bandwidth_inf": 4.0, "gamma": 0.5, "latency": 0.7})
        base = {f["policy"]: f for f in cmv.leave_one_form_out('baseline_cont', rows, starts=3, max_iter=300)}
        for target in ('static_sp6', 'static_cfg2sp3'):
            corrupted = [dict(r, step_ms=r["step_ms"] * 10.0) if r["policy"] == target else r for r in rows]
            folds = {f["policy"]: f for f in cmv.leave_one_form_out('baseline_cont', corrupted, starts=3,
                                                                    max_iter=300)}
            self.assertAlmostEqual(folds[target]["predicted_step_ms"], base[target]["predicted_step_ms"], places=9)
            self.assertNotAlmostEqual(folds[target]["measured_step_ms"], base[target]["measured_step_ms"])
            other = 'static_sp2'
            self.assertNotAlmostEqual(folds[other]["predicted_step_ms"], base[other]["predicted_step_ms"], places=3)

    def test_ratio_transfer_uses_only_source_ratios(self):
        source = {"t1": 1000.0, "points": [{"policy": "static_sp1", "step_ms": 1000.0},
                                           {"policy": "static_sp2", "step_ms": 700.0}]}
        target = [{"policy": "static_sp1", "step_ms": 2000.0}, {"policy": "static_sp2", "step_ms": 1500.0}]
        out = cmv.ratio_transfer(source, 2000.0, target, 't')
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["predicted_step_ms"], 1400.0)
        self.assertAlmostEqual(out[0]["signed_error_pct"], (1400.0 - 1500.0) / 1500.0 * 100.0)


class RealCal8ReproductionTest(unittest.TestCase):
    """Read-only: reproduces the reported fit/held-out errors from the verified bundles."""

    EXPECTED = {
        'wan21:wan21-480p': (8.389620835164498, 10.596772284034827, 17.740258723801098),
        'wan21:wan21-720p': (5.529577715343422, 7.034934718427106, 12.546909930007208),
        'wan22:wan22-small': (7.519959260729671, 12.776516322515672, 19.182161822701413),
        'wan22:wan22-720p': (6.327511268954323, 11.99787770151751, 17.83589665250801),
    }

    def test_reported_errors_reproduce_exactly(self):
        datasets = cmv.load_bundle(WAN21_BUNDLE) + cmv.load_bundle(WAN22_BUNDLE)
        self.assertEqual(sorted(d["key"] for d in datasets), sorted(self.EXPECTED))
        for d in datasets:
            rows = cmv.rows_of(d)
            fit_rows = [r for r in rows if r["policy"] in cmv.FIT_POLICIES]
            params, _ = cmv.fit_model('baseline_grid', fit_rows)
            self.assertEqual(params["bandwidth_inf"], d["reported"]["bandwidth_inf"])
            self.assertEqual(params["gamma"], d["reported"]["gamma"])
            self.assertEqual(params["latency"], d["reported"]["latency"])
            errors = cmv.relative_errors('baseline_grid', params, rows)
            fit = cmv.mean(e["relative_error_pct"] for e in errors if e["policy"] in cmv.FIT_POLICIES)
            held = [e["relative_error_pct"] for e in errors if e["policy"] not in cmv.FIT_POLICIES]
            exp_fit, exp_mean, exp_max = self.EXPECTED[d["key"]]
            self.assertAlmostEqual(fit, exp_fit, places=9)
            self.assertAlmostEqual(cmv.mean(held), exp_mean, places=9)
            self.assertAlmostEqual(max(held), exp_max, places=9)
            for e, p in zip(errors, d["points"]):
                self.assertAlmostEqual(e["predicted_step_ms"], p["reported_predicted_ms"], places=6)
            self.assertEqual(params["latency"], max(cmv.GRID_LATENCY))

    def test_placement_bandwidth_not_identifiable_from_cal8(self):
        datasets = cmv.load_bundle(WAN21_BUNDLE) + cmv.load_bundle(WAN22_BUNDLE)
        diag = cmv.placement_identifiability(datasets)
        self.assertFalse(diag["identifiable"])
        self.assertEqual(diag["placements_by_form"]["cfg2sp4"], ["0-7"])


class CliSmokeTest(unittest.TestCase):
    def test_cli_writes_json_with_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'r.json'
            result = cmv.main(['--bundle', str(WAN21_BUNDLE), '--models', 'baseline_grid,bus',
                               '--objectives', 'abs_mse', '--output', str(out)])
            data = json.loads(out.read_text())
            self.assertEqual(data["schema"], cmv.SCHEMA)
            for value in data["reproduction"].values():
                self.assertEqual(value["max_abs_diff_pct_points"], 0.0)
            self.assertIn('wan21:wan21-480p', result["per_dataset"])
            row = [r for r in data["summary_table"] if r["variant"] == 'bus|abs_mse'][0]
            self.assertTrue(math.isfinite(row["held_out_mean_pct"]))


if __name__ == '__main__':
    unittest.main()
