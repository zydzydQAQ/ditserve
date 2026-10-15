"""Bounded same-card state costs; engineering admission is not research qualification."""

import hashlib
from pathlib import Path
import time

from . import direction2 as d2


PROTOCOL_PATH = d2.ROOT / "configs/research_direction2_state_cost.json"
PROTOCOL = dict(schema_version="ditserve.directions.state-cost.protocol.v1", phase="state_cost",
    prerequisite="qualify", frames=81, height=480, width=832, steps=32, guidance_scale=5.,
    prompt_id="origami", seed=101, split="dev", pause_step=16,
    orders=[["keep", "full_cpu", "flush"], ["flush", "full_cpu", "keep"]],
    quality_mean_lpips_max=.05, full_cpu_requires_exact_keep=True,
    max_retained_outputs=7, descriptive_only=True, urgent_requests=0)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint(experiment, row):
    experiment.report["active_unit"] = dict(unit_id=row["unit_id"], stage=row["stage"],
                                             last_completed_step=row.get("last_completed_step"))
    d2._write(experiment.output / "partial_units" / ("state_cost-" + row["unit_id"] + ".json"), row)
    experiment.save()


def _store(experiment, key, policy, output):
    """At most seven fixed outputs; keep every completed output until adjudicated."""
    import numpy as np
    base = experiment.settings.get("artifact_root") or experiment.config.get("artifact_root") or str(experiment.output / "key_samples")
    directory = (Path(base) / "direction2" / d2._digest(str(experiment.output.resolve()))[:16]
                 / experiment.identity["config_sha256"][:12] / "state_cost" / key)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (policy + ".npy")
    if path.exists():
        raise ValueError("refusing to overwrite state-cost output: " + str(path))
    array = np.ascontiguousarray(output)
    sha = hashlib.sha256(array.tobytes()).hexdigest()
    np.save(path, array, allow_pickle=False)
    retention = dict(raw_available=True, files=[dict(path=str(path), policy=policy, output_sha256=sha)])
    d2._write(directory / "manifest.json", dict(identity=experiment.identity,
        cost_identity=experiment.report["cost_identity"], shape=list(array.shape), dtype=str(array.dtype), **retention))
    return sha, retention


def _timed(runtime, operation):
    runtime.worker._sync()
    started = time.perf_counter()
    value = operation()
    runtime.worker._sync()
    return value, (time.perf_counter() - started) * 1000


def _trial(experiment, runtime, item, row):
    state = None
    worker = runtime.worker
    try:
        experiment.check_budget()
        row["stage"] = "setup"
        _checkpoint(experiment, row)
        started = time.perf_counter()
        runtime.bank.warm(runtime.spec(item, 81, "paired_008"))
        row["timings"]["encoding_setup_ms"] = (time.perf_counter() - started) * 1000
        row["stage"] = "prefix"
        worker._sync()
        started = time.perf_counter()
        state = runtime.prepare(item, 81, "paired_008")
        while state.step_index < 16:
            experiment.check_budget()
            runtime.step(state, "paired_008")
            row["last_completed_step"] = state.step_index
        worker._sync()
        row["timings"]["prefix_ms"] = (time.perf_counter() - started) * 1000
        row["state_inventory"]["before"] = runtime.state_adapter.inventory(state)
        row["memory"]["before"] = worker._memory()
        row["stage"] = "action"
        _checkpoint(experiment, row)
        experiment.check_budget()
        action = row["action"]
        if action == "full_cpu":
            event, ms = _timed(runtime, lambda: runtime.state_adapter.snapshot(state, "hot"))
            row["snapshot"] = event
            row["timings"]["d2h_ms"] = ms
            row["transfer"]["d2h_bytes"] = event["transfer_bytes"]
            if state.residency != "cpu":
                raise RuntimeError("full_cpu snapshot did not offload the real state")
        elif action == "flush":
            def flush():
                for context in state.contexts.values():
                    context.buffers.clear()
                state.needs_anchor = True
            _, row["timings"]["flush_ms"] = _timed(runtime, flush)
            row["forced_recompute"] = True
        elif action != "keep":
            raise ValueError("unknown state-cost action")
        row["memory"]["after_action"] = worker._memory()
        row["state_inventory"]["after_action"] = runtime.state_adapter.inventory(state)
        _checkpoint(experiment, row)
        if action == "full_cpu":
            row["stage"] = "restore"
            _checkpoint(experiment, row)
            experiment.check_budget()
            event, ms = _timed(runtime, lambda: runtime.state_adapter.restore(state))
            row["restore"] = event
            row["timings"]["h2d_ms"] = ms
            row["transfer"]["h2d_bytes"] = event["transfer_bytes"]
            if state.residency != "gpu":
                raise RuntimeError("full_cpu restore did not restore the real state")
        row["timings"]["action_ms"] = sum(row["timings"][k] for k in ("d2h_ms", "h2d_ms", "flush_ms"))
        row["memory"]["after_restore"] = worker._memory()
        row["state_inventory"]["after_restore"] = runtime.state_adapter.inventory(state)
        row["stage"] = "next_step"
        _checkpoint(experiment, row)
        experiment.check_budget()
        worker._sync()
        suffix_started = time.perf_counter()
        _, row["timings"]["next_step_ms"] = _timed(runtime, lambda: runtime.step(state, "paired_008"))
        row["last_completed_step"] = state.step_index
        row["stage"] = "suffix"
        while state.step_index < 32:
            experiment.check_budget()
            runtime.step(state, "paired_008")
            row["last_completed_step"] = state.step_index
        row["stage"] = "decode"
        experiment.check_budget()
        output = worker.finish(state)
        worker._sync()
        row["timings"]["suffix_complete_ms"] = (time.perf_counter() - suffix_started) * 1000
        row["timings"]["post_pause_work_ms"] = row["timings"]["action_ms"] + row["timings"]["suffix_complete_ms"]
        # Persist a completed candidate before any budget check or reference comparison.
        row["output_sha256"], row["retention"] = _store(experiment, row["unit_id"], action, output)
        row.update(generation_completed=True, stage="quality_pending")
        _checkpoint(experiment, row)
    finally:
        if state is not None and state.residency != "released":
            worker.release(state)
        try:
            row["memory"]["after_release"] = worker._memory()
        except Exception as error:
            row["memory"]["after_release"] = dict(status="unavailable", error=str(error))


def _evaluate_round(experiment, runtime, rows, reference):
    import numpy as np
    keep = next(row for row in rows if row["action"] == "keep")
    keep_output = np.load(keep["retention"]["files"][0]["path"], mmap_mode="r", allow_pickle=False)
    for row in rows:
        row["stage"] = "quality"
        _checkpoint(experiment, row)
        experiment.check_budget()
        output = np.load(row["retention"]["files"][0]["path"], mmap_mode="r", allow_pickle=False)
        quality = runtime.quality(reference, output)
        exact = bool(np.array_equal(keep_output, output))
        row.update(quality=quality, exact_to_keep=exact)
        _checkpoint(experiment, row)
        experiment.check_budget()
        passed = d2._quality_known(quality, 81, .05) and (row["action"] != "full_cpu" or exact)
        row.update(quality=quality, quality_passed=passed, exact_to_keep=exact,
                   keep_reference_sha256=keep["output_sha256"], complete=True,
                   status="passed" if passed else "failed", stage="completed")
        row["comparison"] = {name: row["timings"][key] - keep["timings"][key] for name, key in (
            ("next_step_over_keep_ms", "next_step_ms"), ("suffix_over_keep_ms", "suffix_complete_ms"),
            ("post_pause_work_over_keep_ms", "post_pause_work_ms"))}
        d2._write(experiment.output / "units" / ("state_cost-" + row["unit_id"] + ".json"), row)
        _checkpoint(experiment, row)


def run_phase(config, output: Path, check_budget, phase="state_cost"):
    """Return a finite cost report; the supervisor owns accounting and isolation."""
    if phase != "state_cost":
        raise ValueError("this module only implements state_cost")
    protocol = d2._read(PROTOCOL_PATH)
    if protocol != PROTOCOL:
        raise ValueError("state-cost protocol differs from the fixed supplement")
    experiment = d2._Experiment(config, output, check_budget, phase)
    cost_identity = dict(protocol_sha256=_hash(PROTOCOL_PATH), implementation_sha256=_hash(__file__))
    if experiment.path.exists():
        old = d2._read(experiment.path)
        if old.get("identity") != experiment.identity or old.get("cost_identity") != cost_identity:
            raise ValueError("state-cost evidence identity differs")
        return old
    report = experiment.report
    report.update(schema_version="ditserve.directions.state-cost.v1", cost_identity=cost_identity,
        cost_protocol=dict(path=str(PROTOCOL_PATH), sha256=cost_identity["protocol_sha256"], value=protocol),
        completed_cost_matrix=False, descriptive_only=True, capacity_qualified=False,
        selected_frames=81, required_trials=6, reference_off=None, repeated_keep_exact=None,
        qualification_scope="single dev input, two orders; no capacity, state policy, online or SP qualification",
        timing_scope="synchronized operation wall times; next_step is17; suffix is17..32+decode; post_pause_work is action+restore+suffix sum, excluding inventory/evidence IO; encoding/prefix are charged setup")
    experiment.save()
    active = None
    try:
        previous = experiment.previous("qualify")
        authority_path = experiment.output / "supervisor-qualify.json"
        authority = d2._read(authority_path) if authority_path.is_file() else {}
        if not (previous and previous.get("status") == "passed" and previous.get("selected_frames") == 81
                and previous.get("cache_quality_qualified") is True and previous.get("cache_performance_qualified") is True
                and authority.get("status") == "passed" and authority.get("eligible") is True
                and authority.get("selected_frames") == 81 and authority.get("cache_quality_qualified") is True
                and authority.get("cache_performance_qualified") is True
                and authority.get("identity") == experiment.identity and not authority.get("supervisor_failure")
                and authority.get("returncode", 0) == 0):
            report.update(status="blocked", reason="supervised 81-frame cache qualification is required")
            return report
        report["prerequisite"] = dict(path=str(authority_path), sha256=_hash(authority_path))
        item = next(i for i in d2._inputs("dev") if i["prompt_id"] == "origami" and i["seed"] == 101)
        runtime = experiment.get_runtime()
        check_budget()
        report["active_unit"] = dict(unit_id="reference-off", stage="reference_generation")
        experiment.save()
        reference = runtime.generate(item, 81, "off")
        sha, retention = _store(experiment, "reference-off", "off", reference["output"])
        report["reference_off"] = dict(output_sha256=sha, retention=retention,
            generation_ms=reference["generation_ms"], item=item, frames=81)
        experiment.save()
        for index, order in enumerate(PROTOCOL["orders"]):
            round_rows = []
            for position, action in enumerate(order):
                # Admission belongs to the next unit, never to the preceding
                # completed row (including the last adjudicated row of a round).
                active = None
                report["active_unit"] = dict(unit_id="round%d-%s" % (index, action),
                    stage="between_units", started=False, round=index, position=position, action=action)
                experiment.save()
                check_budget()
                active = dict(unit_id="round%d-%s" % (index, action), round=index, position=position,
                    action=action, status="incomplete", complete=False, generation_completed=False,
                    stage="setup", last_completed_step=0, item=item, frames=81, pause_step=16,
                    reference_off_sha256=sha,
                    timings=dict(d2h_ms=None if action == "full_cpu" else 0.,
                                 h2d_ms=None if action == "full_cpu" else 0.,
                                 flush_ms=None if action == "flush" else 0.),
                    transfer=dict(d2h_bytes=None if action == "full_cpu" else 0,
                                  h2d_bytes=None if action == "full_cpu" else 0,
                                  transport="direct_tensor_CPU_copies", no_pickle=True),
                    state_inventory={}, memory={})
                report["units"].append(active)
                round_rows.append(active)
                _trial(experiment, runtime, item, active)
                _checkpoint(experiment, active)
            _evaluate_round(experiment, runtime, round_rows, reference["output"])
        complete = len(report["units"]) == 6 and all(r.get("complete") for r in report["units"])
        keeps = [r for r in report["units"] if r["action"] == "keep"]
        repeat_exact = len(keeps) == 2 and keeps[0]["output_sha256"] == keeps[1]["output_sha256"]
        passed = complete and repeat_exact and all(r.get("quality_passed") is True for r in report["units"])
        report.update(status="passed" if passed else "stopped", eligible=passed,
            completed_cost_matrix=complete, repeated_keep_exact=repeat_exact, active_unit=None,
            reason="finite descriptive state costs completed" if passed else
                   "repeated keep output mismatch; no qualification" if not repeat_exact else
                   "state-cost correctness check failed; no qualification")
    except BaseException as error:
        budget = "budget" in type(error).__name__.lower()
        report.update(status="stopped" if budget else "incomplete" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed",
            eligible=False, budget_stopped=budget, reason="%s: %s" % (type(error).__name__, error))
        if (report.get("active_unit") or {}).get("stage") == "between_units":
            report["interruption"] = dict(report["active_unit"], reason=report["reason"])
        elif active is not None:
            # The active row may be a prior completed trial while evaluating its round.
            current = next((r for r in report["units"] if r["unit_id"] == (report.get("active_unit") or {}).get("unit_id")), active)
            current.update(error=report["reason"], status="incomplete" if budget else "failed")
            _checkpoint(experiment, current)
        if not isinstance(error, Exception):
            raise
    finally:
        if experiment.runtime is not None:
            try:
                experiment.runtime.close()
            except Exception as error:
                report.update(status="failed", eligible=False, cleanup_error=str(error))
        experiment.save()
    return report
