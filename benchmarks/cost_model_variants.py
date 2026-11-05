#!/usr/bin/env python3
"""Offline comparison of analytical step-cost model variants on native calibration data.

This module is a read-only analysis tool (2026-11-04, item ``cost_model_refit``).
It re-implements the video step model of ``fit_native_profile.py`` verbatim
(``baseline_predict`` / ``baseline_grid_fit``; copied, not imported, because the
working-tree copy of that file carries uncommitted edits and imports
``cluster_config``) and compares it with candidate variants:

* ``baseline_grid``    -- exact replica: 3 params, fixed grid, abs-MSE objective.
* ``baseline_cont``    -- same formula, bounded continuous optimisation (wider
                          latency range; the original grid caps latency at 0.5 ms).
* ``amdahl_sp``        -- replicated/serial compute fraction inside an SP group:
                          compute = T1/cfg * (s + (1-s)/sp).
* ``amdahl_all``       -- compute = T1 * (s + (1-s)/(cfg*sp)).
* ``overlap``          -- compute/communication overlap:
                          step = P + C - rho*min(P, C).
* ``bus``              -- shared PCIe host-bridge (PHB) contention, no gamma:
                          per-rank all-to-all bytes v pay v/B_link + n_active*v/B_bus,
                          n_active = cfg*sp concurrently communicating ranks.
* ``bus_amdahl``       -- ``bus`` plus the ``amdahl_sp`` serial fraction.

Placement-specific bandwidth (variant (c) of the request) is reported as not
identifiable when every measured group is the consecutive block starting at
rank 0 and the recorded topology is uniform (all PHB).

All step times are per-denoising-step milliseconds; ``T1`` is anchored to the
measured SP1 median exactly as in ``fit_native_profile.fit_links``.  Data volumes
are MB and bandwidths MB/ms (= GB/s).  Only the Python standard library is used.

Usage::

    python3 benchmarks/cost_model_variants.py \
        --bundle <cal8>/wan21/bundle/source.json --bundle <cal8>/wan22/bundle/source.json \
        [--extracted run_summary.json ...] [--replicate run_summary.json ...] \
        --output results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

SCHEMA = "ditserve.cost-model-variants.v1"
FIT_POLICIES = ("static_sp1", "static_cfg2", "static_sp2", "static_cfg2sp2", "static_sp4")


# ---------------------------------------------------------------------------
# Helpers copied from fit_native_profile.py (behaviour must stay identical).
# ---------------------------------------------------------------------------

def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else 0.0


def median(values):
    values = sorted(values)
    if not values:
        raise ValueError("median requires at least one sample")
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def form_of(policy):
    name = policy.removeprefix("static_")
    if name.startswith("cfg2"):
        rest = name[4:]
        return 2, 1 if rest == "" else int(rest.removeprefix("sp"))
    return 1, int(name.removeprefix("sp"))


def tokens_and_dim(transformer, vae, spec):
    patch_t, patch_h, patch_w = transformer["patch_size"]
    temporal = vae.get("scale_factor_temporal", 4)
    spatial = vae.get("scale_factor_spatial", 8)
    frames = (spec["frames"] - 1) // temporal + 1
    tokens = ((frames // patch_t) * (spec["height"] // (spatial * patch_h))
              * (spec["width"] // (spatial * patch_w)))
    dim = transformer["num_attention_heads"] * transformer["attention_head_dim"]
    return tokens, dim, transformer["num_layers"], transformer["out_channels"]


def architecture(transformer, vae, spec):
    tokens, dim, layers, out_channels = tokens_and_dim(transformer, vae, spec)
    patch_volume = math.prod(transformer["patch_size"])
    return {
        "tokens": tokens,
        "dim": dim,
        "layers": layers,
        "comm_mb_per_step": 4.0 * tokens * dim * 2.0 * layers * 2.0 / 1e6,
        "collectives_per_step": layers * 8,
        "cfg_exchange_mb": tokens * patch_volume * out_channels * 2.0 / 1e6,
    }


def baseline_predict(step_compute, arch, cfg, sp, bandwidth_inf, gamma, latency):
    """Verbatim copy of fit_native_profile.predict."""
    compute = step_compute / (cfg * sp)
    if sp == 1:
        communication = 0.0
    else:
        bandwidth = bandwidth_inf * sp / (sp + gamma)
        communication = (arch["comm_mb_per_step"] / cfg) * (sp - 1) / (sp * sp) / bandwidth
        communication += (arch["collectives_per_step"] / cfg) * latency
    exchange = 0.0
    if cfg == 2:
        bandwidth2 = bandwidth_inf * 2 / (2 + gamma)
        exchange = arch["cfg_exchange_mb"] / bandwidth2 + latency
    return compute + communication + exchange


GRID_BANDWIDTH = tuple(0.25 * index for index in range(1, 201))
GRID_GAMMA = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
GRID_LATENCY = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5)


def baseline_grid_fit(step_compute, arch, training):
    """Grid search of fit_native_profile.fit_links on ``training`` rows.

    ``training`` rows need ``form`` (cfg, sp) and ``step_ms``.  Returns
    (bandwidth_inf, gamma, latency) chosen by the identical loop order and
    strict-improvement tie rule.
    """
    best = None
    for bandwidth_inf in GRID_BANDWIDTH:
        for gamma in GRID_GAMMA:
            for latency in GRID_LATENCY:
                error = mean([
                    (baseline_predict(step_compute, arch, *row["form"], bandwidth_inf, gamma, latency)
                     - row["step_ms"]) ** 2
                    for row in training
                ])
                if best is None or error < best[0]:
                    best = (error, bandwidth_inf, gamma, latency)
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# Model variants.
# ---------------------------------------------------------------------------

def _alltoall_rank_mb(arch, cfg, sp):
    """Per-rank Ulysses all-to-all MB per step (baseline volume convention)."""
    return (arch["comm_mb_per_step"] / cfg) * (sp - 1) / (sp * sp)


def _predict_baseline(p, t1, arch, cfg, sp):
    return baseline_predict(t1, arch, cfg, sp, p["bandwidth_inf"], p["gamma"], p["latency"])


def _baseline_comm(p, arch, cfg, sp):
    return baseline_predict(0.0, arch, cfg, sp, p["bandwidth_inf"], p["gamma"], p["latency"])


def _predict_amdahl_sp(p, t1, arch, cfg, sp):
    compute = t1 / cfg * (p["serial"] + (1.0 - p["serial"]) / sp)
    return compute + _baseline_comm(p, arch, cfg, sp)


def _predict_amdahl_all(p, t1, arch, cfg, sp):
    compute = t1 * (p["serial"] + (1.0 - p["serial"]) / (cfg * sp))
    return compute + _baseline_comm(p, arch, cfg, sp)


def _predict_overlap(p, t1, arch, cfg, sp):
    compute = t1 / (cfg * sp)
    comm = _baseline_comm(p, arch, cfg, sp)
    return compute + comm - p["overlap"] * min(compute, comm)


def _bus_comm(p, arch, cfg, sp):
    comm = 0.0
    if sp > 1:
        volume = _alltoall_rank_mb(arch, cfg, sp)
        comm = volume / p["link_gbps"] + volume * (cfg * sp) / p["bus_gbps"]
        comm += (arch["collectives_per_step"] / cfg) * p["latency"]
    if cfg == 2:
        comm += arch["cfg_exchange_mb"] / p["link_gbps"] + p["latency"]
    return comm


def _predict_bus(p, t1, arch, cfg, sp):
    return t1 / (cfg * sp) + _bus_comm(p, arch, cfg, sp)


def _predict_bus_amdahl(p, t1, arch, cfg, sp):
    compute = t1 / cfg * (p["serial"] + (1.0 - p["serial"]) / sp)
    return compute + _bus_comm(p, arch, cfg, sp)


# Parameter specs: (name, lower, upper, scale) with scale "log" or "lin".
_BASE = [("bandwidth_inf", 0.01, 1000.0, "log"), ("gamma", 0.0, 64.0, "lin"), ("latency", 0.0, 20.0, "lin")]
MODELS = {
    "baseline_grid": {"params": _BASE, "predict": _predict_baseline, "grid": True,
                      "description": "fit_native_profile replica: 3-param saturation model, fixed grid"},
    "baseline_cont": {"params": _BASE, "predict": _predict_baseline,
                      "description": "same formula, continuous bounded fit (latency up to 20 ms)"},
    "amdahl_sp": {"params": _BASE + [("serial", 0.0, 0.95, "lin")], "predict": _predict_amdahl_sp,
                  "description": "baseline comm + replicated compute fraction inside SP group"},
    "amdahl_all": {"params": _BASE + [("serial", 0.0, 0.95, "lin")], "predict": _predict_amdahl_all,
                   "description": "baseline comm + Amdahl over all cfg*sp ranks"},
    "overlap": {"params": _BASE + [("overlap", 0.0, 1.0, "lin")], "predict": _predict_overlap,
                "description": "baseline + compute/comm overlap step=P+C-rho*min(P,C)"},
    "bus": {"params": [("link_gbps", 0.01, 1000.0, "log"), ("bus_gbps", 0.01, 10000.0, "log"),
                       ("latency", 0.0, 20.0, "lin")],
            "predict": _predict_bus,
            "description": "shared PHB host-bridge contention: v/B_link + cfg*sp*v/B_bus"},
    "bus_amdahl": {"params": [("link_gbps", 0.01, 1000.0, "log"), ("bus_gbps", 0.01, 10000.0, "log"),
                              ("latency", 0.0, 20.0, "lin"), ("serial", 0.0, 0.95, "lin")],
                   "predict": _predict_bus_amdahl,
                   "description": "bus contention + replicated compute fraction inside SP group"},
}


def model_predict(name, params, t1, arch, cfg, sp):
    return MODELS[name]["predict"](params, t1, arch, cfg, sp)


# ---------------------------------------------------------------------------
# Optimiser (deterministic multi-start Nelder-Mead in a bounded transform).
# ---------------------------------------------------------------------------

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _decode(spec, z):
    out = {}
    for (name, lo, hi, scale), value in zip(spec, z):
        u = _sigmoid(value)
        if scale == "log":
            out[name] = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * u)
        else:
            out[name] = lo + (hi - lo) * u
    return out


def nelder_mead(func, x0, step=0.8, max_iter=800, tol=1e-12):
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        x[i] += step
        simplex.append(x)
    values = [func(x) for x in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda k: values[k])
        simplex = [simplex[k] for k in order]
        values = [values[k] for k in order]
        if abs(values[-1] - values[0]) <= tol * (abs(values[0]) + tol):
            break
        centroid = [math.fsum(simplex[k][i] for k in range(n)) / n for i in range(n)]
        worst = simplex[-1]
        reflected = [c + (c - w) for c, w in zip(centroid, worst)]
        fr = func(reflected)
        if fr < values[0]:
            expanded = [c + 2.0 * (c - w) for c, w in zip(centroid, worst)]
            fe = func(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = [c + 0.5 * (w - c) for c, w in zip(centroid, worst)]
            fc = func(contracted)
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:
                best = simplex[0]
                simplex = [best] + [[b + 0.5 * (x - b) for b, x in zip(best, s)] for s in simplex[1:]]
                values = [values[0]] + [func(s) for s in simplex[1:]]
    k = min(range(n + 1), key=lambda j: values[j])
    return simplex[k], values[k]


def objective_value(name, params, rows, objective):
    total = 0.0
    for row in rows:
        predicted = model_predict(name, params, row["t1"], row["arch"], row["cfg"], row["sp"])
        if objective == "abs_mse":
            total += (predicted - row["step_ms"]) ** 2
        elif objective == "rel_mse":
            total += ((predicted - row["step_ms"]) / row["step_ms"]) ** 2
        else:
            raise ValueError(f"unknown objective {objective}")
    return total / len(rows)


def fit_model(name, rows, objective="abs_mse", starts=10, seed=20261104, max_iter=800):
    """Fit model ``name`` on ``rows`` only (each row carries t1/arch/cfg/sp/step_ms)."""
    if not rows:
        raise ValueError("fit requires at least one training row")
    model = MODELS[name]
    if model.get("grid"):
        if objective != "abs_mse":
            raise ValueError("baseline_grid is defined with the abs-MSE objective only")
        arches = {json.dumps(row["arch"], sort_keys=True) for row in rows}
        t1s = {row["t1"] for row in rows}
        if len(arches) != 1 or len(t1s) != 1:
            raise ValueError("baseline_grid replicates a single-shape fit only")
        training = [{"form": (row["cfg"], row["sp"]), "step_ms": row["step_ms"]} for row in rows]
        b, g, lat = baseline_grid_fit(rows[0]["t1"], rows[0]["arch"], training)
        params = {"bandwidth_inf": b, "gamma": g, "latency": lat}
        return params, objective_value(name, params, rows, objective)
    spec = model["params"]
    rng = random.Random(seed)
    candidates = [[0.0] * len(spec)] + [[rng.uniform(-3.0, 3.0) for _ in spec] for _ in range(starts - 1)]
    best = None
    for z0 in candidates:
        z, value = nelder_mead(lambda z: objective_value(name, _decode(spec, z), rows, objective), z0,
                               max_iter=max_iter)
        z, value = nelder_mead(lambda z: objective_value(name, _decode(spec, z), rows, objective), z,
                               step=0.2, max_iter=max_iter)
        if best is None or value < best[1] - 1e-15:
            best = (z, value)
    return _decode(spec, best[0]), best[1]


# ---------------------------------------------------------------------------
# Identifiability diagnostics.
# ---------------------------------------------------------------------------

def _sym_eigenvalues(matrix, sweeps=100):
    a = [list(row) for row in matrix]
    n = len(a)
    for _ in range(sweeps):
        off = math.fsum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-30:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
    return sorted(a[i][i] for i in range(n))


def identifiability(name, params, rows, rel_step=1e-4):
    """Local sensitivity of relative residuals to log-parameters.

    Returns singular values of the Jacobian d(pred/meas)/d(log p) (or d p for
    params whose value is ~0) and a status:
    ``unidentifiable`` if n_params >= informative points or the smallest
    singular value is < 1e-6 of the largest; ``weak`` if n_params ==
    informative points - 1 or condition number > 1e3; else ``ok``.
    """
    names = [spec[0] for spec in MODELS[name]["params"]]
    informative = sum(1 for row in rows if not (row["cfg"] == 1 and row["sp"] == 1))
    columns = []
    for key in names:
        value = params[key]
        delta = abs(value) * rel_step if abs(value) > 1e-9 else rel_step
        up = dict(params, **{key: value + delta})
        down = dict(params, **{key: max(value - delta, 0.0)})
        span = up[key] - down[key]
        scale = abs(value) if abs(value) > 1e-9 else 1.0
        columns.append([
            (model_predict(name, up, r["t1"], r["arch"], r["cfg"], r["sp"])
             - model_predict(name, down, r["t1"], r["arch"], r["cfg"], r["sp"])) / span * scale / r["step_ms"]
            for r in rows
        ])
    gram = [[math.fsum(a * b for a, b in zip(ci, cj)) for cj in columns] for ci in columns]
    eig = _sym_eigenvalues(gram)
    singular = [math.sqrt(max(e, 0.0)) for e in reversed(eig)]
    largest = singular[0] if singular else 0.0
    smallest = singular[-1] if singular else 0.0
    condition = largest / smallest if smallest > 0 else math.inf
    if len(names) >= informative or smallest < 1e-6 * max(largest, 1e-300):
        status = "unidentifiable"
    elif len(names) == informative - 1 or condition > 1e3:
        status = "weak"
    else:
        status = "ok"
    return {"n_params": len(names), "informative_points": informative,
            "singular_values": singular, "condition_number": condition if math.isfinite(condition) else None,
            "status": status}


# ---------------------------------------------------------------------------
# Data.
# ---------------------------------------------------------------------------

def load_bundle(path):
    """Read a ditserve.shape-profiles bundle and return per-shape datasets."""
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    datasets = []
    for class_name, entry in bundle["classes"].items():
        cost = entry["cost"]
        if cost.get("kind") != "video":
            continue
        fit = entry["source"]["fit"]
        arch = fit["architecture"]
        points = []
        for row in fit["predictions"]:
            cfg, sp = row["form"]
            placements = cost.get("forms", {}).get(row["policy"].removeprefix("static_"), {}).get("placements", {})
            points.append({"policy": row["policy"], "cfg": cfg, "sp": sp,
                           "step_ms": row["measured_step_ms"],
                           "placements": sorted(placements),
                           "reported_predicted_ms": row["predicted_step_ms"],
                           "reported_partition": row["partition"]})
        datasets.append({
            "key": f'{bundle["model"]}:{class_name}',
            "model": bundle["model"],
            "class": class_name,
            "run_id": fit["run_id"],
            "run_sha256": fit["run_sha256"],
            "bundle_path": str(path),
            "bundle_file_sha256": hashlib.sha256(raw).hexdigest(),
            "spec": {k: entry["spec"].get(k) for k in ("frames", "height", "width", "steps")},
            "arch": arch,
            "t1": cost["step_compute_ms"],
            "points": points,
            "reported": {
                "bandwidth_inf": fit["bandwidth_saturation_gbps"],
                "gamma": fit["bandwidth_gamma"],
                "latency": cost["links"]["2"]["latency_ms"],
                "fit_mean_relative_error_pct": fit["fit_mean_relative_error_pct"],
                "held_out_mean_relative_error_pct": fit["held_out_mean_relative_error_pct"],
                "held_out_max_relative_error_pct": fit["held_out_max_relative_error_pct"],
            },
        })
    return datasets


def rows_of(dataset, policies=None):
    return [{"policy": p["policy"], "cfg": p["cfg"], "sp": p["sp"], "step_ms": p["step_ms"],
             "t1": dataset["t1"], "arch": dataset["arch"], "dataset": dataset["key"]}
            for p in dataset["points"] if policies is None or p["policy"] in policies]


def relative_errors(name, params, rows):
    out = []
    for row in rows:
        predicted = model_predict(name, params, row["t1"], row["arch"], row["cfg"], row["sp"])
        out.append({"dataset": row["dataset"], "policy": row["policy"], "measured_step_ms": row["step_ms"],
                    "predicted_step_ms": predicted,
                    "signed_error_pct": (predicted - row["step_ms"]) / row["step_ms"] * 100.0,
                    "relative_error_pct": abs(predicted - row["step_ms"]) / row["step_ms"] * 100.0})
    return out


def _err_summary(errors, exclude_sp1=False):
    values = [e["relative_error_pct"] for e in errors if not (exclude_sp1 and e["policy"] == "static_sp1")]
    if not values:
        return {"n": 0, "mean_pct": None, "max_pct": None}
    return {"n": len(values), "mean_pct": mean(values), "max_pct": max(values)}


def leave_one_form_out(name, rows, objective="abs_mse", fit=fit_model, **kwargs):
    """LOFO over non-SP1 forms; SP1 is the T1 anchor and always stays in training.

    The fitter receives only the training rows; the left-out row's measured
    value is never visible to it.  Returns per-fold predictions.
    """
    folds = []
    for left in rows:
        if left["cfg"] == 1 and left["sp"] == 1:
            continue
        training = [row for row in rows if row is not left
                    and not (row["policy"] == left["policy"] and row["dataset"] == left["dataset"])]
        params, _ = fit(name, training, objective=objective, **kwargs)
        predicted = model_predict(name, params, left["t1"], left["arch"], left["cfg"], left["sp"])
        folds.append({"dataset": left["dataset"], "policy": left["policy"], "measured_step_ms": left["step_ms"],
                      "predicted_step_ms": predicted, "train_n": len(training),
                      "relative_error_pct": abs(predicted - left["step_ms"]) / left["step_ms"] * 100.0,
                      "signed_error_pct": (predicted - left["step_ms"]) / left["step_ms"] * 100.0})
    return folds


def decision_check(errors):
    """Same-GPU-count form choice agreement (argmin predicted vs measured)."""
    by_gpus = {}
    for e in errors:
        cfg, sp = form_of(e["policy"])
        by_gpus.setdefault((e["dataset"], cfg * sp), []).append(e)
    out = []
    for (dataset, gpus), group in sorted(by_gpus.items()):
        if len(group) < 2:
            continue
        chosen = min(group, key=lambda e: e["predicted_step_ms"])
        best = min(group, key=lambda e: e["measured_step_ms"])
        out.append({"dataset": dataset, "gpus": gpus, "forms": sorted(e["policy"] for e in group),
                    "predicted_best": chosen["policy"], "measured_best": best["policy"],
                    "agree": chosen["policy"] == best["policy"],
                    "regret_pct": (chosen["measured_step_ms"] / best["measured_step_ms"] - 1.0) * 100.0})
    return out


def placement_identifiability(datasets, topology_note=None):
    groups = {}
    for d in datasets:
        for p in d["points"]:
            groups.setdefault(p["policy"].removeprefix("static_"), set()).update(p["placements"])
    multiple = {form: sorted(v) for form, v in groups.items() if len(v) > 1}
    starts_nonzero = sorted({pl for v in groups.values() for pl in v if not pl.startswith("0-")})
    identifiable = bool(multiple or starts_nonzero)
    return {"identifiable": identifiable, "placements_by_form": {k: sorted(v) for k, v in groups.items()},
            "reason": ("multiple placements per form available" if identifiable else
                       "every form measured on exactly one consecutive placement starting at rank 0; "
                       "per-placement bandwidth is confounded with the form itself"),
            "topology_note": topology_note}


# ---------------------------------------------------------------------------
# Experiment driver.
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset, models, objectives):
    rows_all = rows_of(dataset)
    rows_fit = [r for r in rows_all if r["policy"] in FIT_POLICIES]
    rows_held = [r for r in rows_all if r["policy"] not in FIT_POLICIES]
    results = {}
    for name in models:
        for objective in objectives:
            if MODELS[name].get("grid") and objective != "abs_mse":
                continue
            params, loss = fit_model(name, rows_fit, objective=objective)
            errors = relative_errors(name, params, rows_all)
            fit_err = [e for e in errors if e["policy"] in FIT_POLICIES]
            held_err = [e for e in errors if e["policy"] not in FIT_POLICIES]
            lofo_all = leave_one_form_out(name, rows_all, objective=objective)
            lofo_fit = leave_one_form_out(name, rows_fit, objective=objective)
            results[f"{name}|{objective}"] = {
                "model": name, "objective": objective, "params": params, "train_loss": loss,
                "n_params": len(MODELS[name]["params"]),
                "fit": _err_summary(fit_err), "fit_excluding_sp1_anchor": _err_summary(fit_err, True),
                "held_out": _err_summary(held_err),
                "lofo_all_forms": _err_summary(lofo_all),
                "lofo_within_fit_set": _err_summary(lofo_fit),
                "identifiability_fit_set": identifiability(name, params, rows_fit),
                "identifiability_all_forms": identifiability(name, fit_model(name, rows_all, objective=objective)[0],
                                                             rows_all),
                "predictions": errors, "lofo_all_folds": lofo_all, "lofo_fit_folds": lofo_fit,
                "decisions_full_fit_partition": decision_check(errors),
                "decisions_lofo": decision_check(lofo_all),
            }
    return {"dataset": {k: v for k, v in dataset.items() if k != "points"},
            "points": dataset["points"], "n_fit": len(rows_fit), "n_held_out": len(rows_held),
            "results": results}


def ratio_transfer(source, target_t1, target_points, target_key):
    """Table-based shape transfer: step_target(form) = T1_target * step_source(form) / T1_source."""
    ratios = {p["policy"]: p["step_ms"] / source["t1"] for p in source["points"]}
    out = []
    for p in target_points:
        if p["policy"] == "static_sp1" or p["policy"] not in ratios:
            continue
        predicted = target_t1 * ratios[p["policy"]]
        out.append({"dataset": target_key, "policy": p["policy"], "measured_step_ms": p["step_ms"],
                    "predicted_step_ms": predicted,
                    "signed_error_pct": (predicted - p["step_ms"]) / p["step_ms"] * 100.0,
                    "relative_error_pct": abs(predicted - p["step_ms"]) / p["step_ms"] * 100.0})
    return out


def evaluate_ratio_transfer(datasets):
    out = {}
    for target in datasets:
        for source in datasets:
            if source is target or source["model"] != target["model"]:
                continue
            errors = ratio_transfer(source, target["t1"], target["points"], target["key"])
            out[f'{source["key"]}->{target["key"]}'] = {"errors": _err_summary(errors), "predictions": errors}
    return out


def evaluate_joint(datasets, models, objective):
    """Shared link/serial parameters across shapes of the same model (one job, one GPU set)."""
    by_model = {}
    for d in datasets:
        by_model.setdefault(d["model"], []).append(d)
    out = {}
    for model_key, group in by_model.items():
        if len(group) < 2:
            continue
        rows_all = [r for d in group for r in rows_of(d)]
        rows_fit = [r for r in rows_all if r["policy"] in FIT_POLICIES]
        for name in models:
            if MODELS[name].get("grid"):
                continue
            params, loss = fit_model(name, rows_fit, objective=objective)
            errors = relative_errors(name, params, rows_all)
            per_shape = {}
            for d in group:
                held = [e for e in errors if e["dataset"] == d["key"] and e["policy"] not in FIT_POLICIES]
                fit = [e for e in errors if e["dataset"] == d["key"] and e["policy"] in FIT_POLICIES]
                per_shape[d["key"]] = {"fit": _err_summary(fit), "held_out": _err_summary(held)}
            lofo = leave_one_form_out(name, rows_all, objective=objective)
            transfer = {}
            for target in group:
                source_rows = [r for r in rows_all if r["dataset"] != target["key"]]
                p_src, _ = fit_model(name, source_rows, objective=objective)
                t_err = relative_errors(name, p_src, rows_of(target))
                transfer[target["key"]] = {"fitted_on": sorted({r["dataset"] for r in source_rows}),
                                           "errors": _err_summary(t_err, True), "predictions": t_err}
            out[f"{model_key}|{name}|{objective}"] = {
                "model_family": model_key, "model": name, "objective": objective, "params": params,
                "n_params": len(MODELS[name]["params"]), "n_fit_rows": len(rows_fit),
                "held_out_all": _err_summary([e for e in errors if e["policy"] not in FIT_POLICIES]),
                "per_shape": per_shape, "lofo_all_forms": _err_summary(lofo),
                "identifiability_fit_set": identifiability(name, params, rows_fit),
                "cross_shape_transfer": transfer, "predictions": errors,
            }
    return out


def noise_summary(extracted_paths):
    out = {}
    for path in extracted_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        forms = {}
        for v in data["videos"]:
            if v.get("step_mean_ms"):
                forms[v["policy"]] = {"steady_n": v["steady_n"], "median_ms": v["step_median_ms"],
                                      "cv_pct": v["step_std_ms"] / v["step_mean_ms"] * 100.0,
                                      "p10_ms": v["step_p10_ms"], "p90_ms": v["step_p90_ms"],
                                      "groups": v["groups"]}
        out[data["run_id"]] = {"run_file_sha256": data["run_file_sha256"],
                               "run_canonical_sha256": data["run_canonical_sha256"],
                               "gpu_uuids": data["gpu_uuids"], "forms": forms}
    return out


def replicate_summary(replicate_paths, joint_params=None, datasets=None):
    """Between-job replicates (e.g. 2-GPU P0 cal0 runs): spread per form, optional transfer check."""
    runs = []
    for path in replicate_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        videos = {v["policy"]: v for v in data["videos"] if v.get("step_median_ms")}
        if "static_sp1" not in videos:
            continue
        spec = videos["static_sp1"]["spec"]
        arch = architecture(data["transformer_config"], data["vae_config"], spec)
        runs.append({"run_id": data["run_id"], "model_name": data["model_name"], "spec": spec,
                     "gpu_uuids": data["gpu_uuids"], "arch": arch,
                     "run_file_sha256": data["run_file_sha256"],
                     "steps": {k: v["step_median_ms"] for k, v in videos.items()},
                     "within_cv_pct": {k: v["step_std_ms"] / v["step_mean_ms"] * 100.0 for k, v in videos.items()}})
    spread = {}
    for run in runs:
        key = (run["model_name"], json.dumps(run["spec"], sort_keys=True))
        for policy, value in run["steps"].items():
            spread.setdefault(key, {}).setdefault(policy, []).append(value)
    spread_out = []
    for (model_name, spec), forms in sorted(spread.items()):
        for policy, values in sorted(forms.items()):
            spread_out.append({"model_name": model_name, "spec": json.loads(spec), "policy": policy,
                               "n_runs": len(values), "min_ms": min(values), "max_ms": max(values),
                               "median_ms": median(values),
                               "range_over_median_pct": (max(values) - min(values)) / median(values) * 100.0})
    transfer = []
    if joint_params:
        for run in runs:
            family = "wan21" if "2.1" in run["model_name"] else "wan22"
            for key, entry in joint_params.items():
                if entry["model_family"] != family:
                    continue
                t1 = run["steps"]["static_sp1"]
                for policy, value in run["steps"].items():
                    if policy == "static_sp1":
                        continue
                    cfg, sp = form_of(policy)
                    predicted = model_predict(entry["model"], entry["params"], t1, run["arch"], cfg, sp)
                    transfer.append({"run_id": run["run_id"], "model": entry["model"], "policy": policy,
                                     "measured_step_ms": value, "predicted_step_ms": predicted,
                                     "signed_error_pct": (predicted - value) / value * 100.0})
    for run in runs if datasets else []:
        family = "wan21" if "2.1" in run["model_name"] else "wan22"
        same = [d for d in datasets if d["model"] == family]
        if not same:
            continue
        source = min(same, key=lambda d: abs(d["arch"]["tokens"] - run["arch"]["tokens"]))
        points = [{"policy": k, "step_ms": v} for k, v in run["steps"].items()]
        for e in ratio_transfer(source, run["steps"]["static_sp1"], points, run["run_id"]):
            transfer.append({"run_id": run["run_id"], "model": f'ratio_from:{source["key"]}',
                             "policy": e["policy"], "measured_step_ms": e["measured_step_ms"],
                             "predicted_step_ms": e["predicted_step_ms"],
                             "signed_error_pct": e["signed_error_pct"]})
    return {"runs": [{k: v for k, v in r.items() if k != "arch"} for r in runs],
            "spread": spread_out, "cross_job_shape_transfer": transfer}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--extracted", type=Path, action="append", default=[],
                        help="compact per-run summaries of the same calibration runs (noise floor)")
    parser.add_argument("--replicate", type=Path, action="append", default=[],
                        help="compact per-run summaries of other jobs (between-job spread)")
    parser.add_argument("--topology-note", default=None)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--objectives", default="abs_mse,rel_mse")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    models = [m for m in args.models.split(",") if m]
    objectives = [o for o in args.objectives.split(",") if o]
    datasets = [d for path in args.bundle for d in load_bundle(path)]
    per_dataset = {d["key"]: evaluate_dataset(d, models, objectives) for d in datasets}
    reproduction = {}
    for key, entry in per_dataset.items():
        res = entry["results"]["baseline_grid|abs_mse"]
        rep = entry["dataset"]["reported"]
        reproduction[key] = {
            "reported": rep, "reproduced_params": res["params"],
            "reproduced_fit_mean_pct": res["fit"]["mean_pct"],
            "reproduced_held_out_mean_pct": res["held_out"]["mean_pct"],
            "reproduced_held_out_max_pct": res["held_out"]["max_pct"],
            "max_abs_diff_pct_points": max(
                abs(res["fit"]["mean_pct"] - rep["fit_mean_relative_error_pct"]),
                abs(res["held_out"]["mean_pct"] - rep["held_out_mean_relative_error_pct"]),
                abs(res["held_out"]["max_pct"] - rep["held_out_max_relative_error_pct"])),
            "latency_at_grid_upper_bound": res["params"]["latency"] == max(GRID_LATENCY),
        }
    joint = {}
    for objective in objectives:
        joint.update(evaluate_joint(datasets, [m for m in models if not MODELS[m].get("grid")], objective))
    summary_table = []
    for key, entry in per_dataset.items():
        for label, res in entry["results"].items():
            summary_table.append({
                "dataset": key, "variant": label, "n_params": res["n_params"],
                "fit_mean_pct": res["fit"]["mean_pct"], "held_out_mean_pct": res["held_out"]["mean_pct"],
                "held_out_max_pct": res["held_out"]["max_pct"],
                "lofo_all_mean_pct": res["lofo_all_forms"]["mean_pct"],
                "lofo_all_max_pct": res["lofo_all_forms"]["max_pct"],
                "lofo_fit_set_mean_pct": res["lofo_within_fit_set"]["mean_pct"],
                "identifiability_fit_set": res["identifiability_fit_set"]["status"],
                "identifiability_all_forms": res["identifiability_all_forms"]["status"],
                "decision_agreement_lofo": (sum(d["agree"] for d in res["decisions_lofo"]),
                                            len(res["decisions_lofo"])),
            })
    joint_for_transfer = {k: v for k, v in joint.items() if k.endswith("|abs_mse")}
    result = {
        "schema": SCHEMA,
        "scope": ("offline re-analysis of one 8-GPU calibration per model (single prompt, seed 101, "
                  "one request per form, median of 49 steady steps); held-out forms only, not held-out "
                  "prompts, placements, nodes or workloads"),
        "fit_policies": list(FIT_POLICIES),
        "models": {k: {"params": [s[0] for s in v["params"]], "description": v["description"]}
                   for k, v in MODELS.items() if k in models},
        "reproduction": reproduction,
        "summary_table": summary_table,
        "per_dataset": per_dataset,
        "joint_across_shapes": joint,
        "ratio_shape_transfer": evaluate_ratio_transfer(datasets),
        "placement_identifiability": placement_identifiability(datasets, args.topology_note),
        "noise_floor_within_request": noise_summary(args.extracted),
        "between_job_replicates": replicate_summary(args.replicate, joint_for_transfer, datasets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "datasets": list(per_dataset),
                      "reproduction": {k: v["max_abs_diff_pct_points"] for k, v in reproduction.items()}},
                     indent=1))
    return result


if __name__ == "__main__":
    main()
