"""Frozen, budgeted run02 jobs. No run01 writes and no unmetered GPU entry point."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import time
import traceback

from ..util import atomic_json, read_json, utc_now
from .previs2_base import BudgetStop, Experiment as LegacyExperiment
from .previs2_policy import (FULL, build_recovery_table, capacity_feasibility,
    choose_action, enumerate_recovery_actions, select_fixed_state, select_initial_policy)


def shape_of(spec):
    return {k: spec[k] for k in ("height", "width", "frames", "steps")}


def arrival_accounting(trace, cutoff_ms, completed_ids):
    """Derive arrivals at the terminal cutoff, including arrivals during a failed step."""
    offered = [r for r in trace if r["arrival_ms"] <= cutoff_ms]
    done = set(completed_ids)
    return dict(offered=len(offered), scheduled=len(trace), completed=len(done),
                offered_unfinished=[r["request_id"] for r in offered if r["request_id"] not in done],
                not_yet_arrived=[r["request_id"] for r in trace if r["arrival_ms"] > cutoff_ms])


def choose_shape(rows):
    safe = [r for r in rows if r.get("type") == "shape_probe" and r.get("status") == "completed"
            and r.get("safe_to_run") is True and r.get("device") == "a100"]
    if not safe:
        raise RuntimeError("no safe measured long shape")
    return max(safe, key=lambda r: r["shape"]["frames"])


class Experiment(LegacyExperiment):
    def __init__(self, job):
        super().__init__(job)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = job["source_identity"]["sha256"]
        self.protocol_sha = job["protocol_sha256"]
        self.model_config = dict(model_id=self.config["model_id"], revision=self.config["revision"],
                                 dtype="float16", vae_dtype="float32", text_encoder_dtype="float32",
                                 attention="math", cache_algorithm="DBCache", strategy_version="previs2-m1-v1")
        self.selected_probe = None
        self.active_case = None

    def record(self, value):
        return super().record(dict(value, source_identity_sha256=self.identity, protocol_sha256=self.protocol_sha))

    def setup(self):
        allowed = set(os.sched_getaffinity(0))
        numa = allowed.intersection(set(range(20, 40)) | set(range(60, 80)))
        if numa:
            os.sched_setaffinity(0, numa)
        import torch
        from ..gpu.worker import WanStepWorker
        torch.set_num_threads(8)
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() != self.job["uuid"].lower():
            raise RuntimeError("wrong GPU visibility")
        self.worker = WanStepWorker(device="cuda:0")
        self.record(dict(type="environment", status="completed", measured=True,
            gpu=torch.cuda.get_device_name(0), uuid=self.job["uuid"],
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            versions={n: importlib.metadata.version(n) for n in ("torch", "diffusers", "cache-dit", "transformers")},
            source_hashes=self.job["source_identity"]["source_hashes"], protocol=self.config,
            elapsed_ms=(time.monotonic() - self.started) * 1000, memory=self.worker._memory()))

    def spec(self, case_id, kind="long", prompt=None, seed=101, cache=False, threshold=.08, max_continuous=1):
        if max_continuous != 1:
            raise ValueError("run02 only admits m=1")
        return super().spec(case_id, kind, prompt, seed, cache, threshold if cache else 0., max_continuous)

    def save_array(self, case_id, output, label="output"):
        from .previs2_storage import save_array
        return save_array(self.suite, self.root, case_id, output, label,
            self.config.get("raw_storage_root", "/mnt/flexllmgen_ssd/preempt-dual-fp16-20260914/previs-run02-overflow"))

    def previous_records(self):
        result = []
        for path in sorted(self.suite.glob("*/records.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("source_identity_sha256") == self.identity and row.get("protocol_sha256") == self.protocol_sha:
                    result.append(row)
        return result

    def adopt_shape(self):
        self.selected_probe = choose_shape(self.previous_records())
        self.config = copy.deepcopy(self.config)
        self.config["shapes"]["long"] = dict(self.selected_probe["shape"])
        self.config["shapes"]["formal"] = dict(self.selected_probe["shape"])
        return self.selected_probe

    def generate(self, spec, pause=None, mode="keep", interleave=False, resume_config=None):
        import torch
        self.check_budget()
        self.active_case = spec["request_id"]
        started = time.monotonic()
        torch.cuda.synchronize()
        baseline = self.worker._memory()
        torch.cuda.reset_peak_memory_stats()
        state = self.worker.prepare(spec)
        switches, boundaries, other_result = [], [], None
        try:
            while state.step_index < spec["steps"]:
                self.check_budget()
                self.worker.step(state)
                if state.step_index in self.config.get("pause_steps", [8, 16, 24]):
                    boundaries.append(dict(completed_steps=state.step_index, state_bytes=self.worker._inventory(state),
                                           memory=self.worker._memory()))
                if pause is not None and state.step_index == pause:
                    switches.append(self.worker.snapshot(state, mode))
                    if interleave:
                        other_spec = self.spec(spec["request_id"] + "-interloper", "short",
                            self.config["prompts"][0], seed=102, cache=False)
                        # The nested call records real successful compute/decoding while victim state is absent.
                        other_result = self.generate(other_spec)
                        self.active_case = spec["request_id"]
                    switches.append(self.worker.restore(state))
                    if resume_config is not None:
                        if set(resume_config) != {"cache", "threshold", "max_continuous"}:
                            raise ValueError("invalid suffix policy")
                        state.spec.update(resume_config)
            torch.cuda.synchronize()
            latent = state.latents.detach().cpu().clone()
            output = self.worker.finish(state)
            torch.cuda.synchronize()
            result = dict(output=output, latents=latent, elapsed_ms=(time.monotonic() - started) * 1000,
                events=copy.deepcopy(state.events), summary=self.worker.state_summary(state), switches=switches,
                boundaries=boundaries, instrumented=True, memory_baseline=baseline,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                memory_final=self.worker._memory())
            if other_result is not None:
                result["interloper"] = other_result
            return result
        finally:
            self.worker.release(state)

    def correctness(self):
        self.config = copy.deepcopy(self.config)
        small = self.config["correctness_shape"]
        self.config["shapes"]["long"] = dict(small)
        self.config["shapes"]["short"] = dict(small, frames=9)
        # Legacy's first eight official-pipeline/keep/hot contracts, with m1 default.
        super().correctness()

    def _cold_policy_interleave_correctness(self):
        import numpy as np
        import torch
        case = "cold-policy-interleave-16"
        spec = self.spec(case, cache=True, threshold=.08)
        reference = self.generate(dict(spec, request_id=case + "-reference"), pause=16, mode="cold", resume_config=FULL)
        candidate = self.generate(spec, pause=16, mode="cold", interleave=True, resume_config=FULL)
        def valid(run):
            steps = [e for e in run["events"] if e["kind"] == "step"]
            switches = run["switches"]
            return (len(steps) == 32 and [e["step_index"] for e in steps] == list(range(32))
                and all(e["finite"] and e["max_continuous"] == 1 and e["threshold"] == (.08 if i < 16 else 0.)
                        for i, e in enumerate(steps))
                and len(switches) == 2 and all(e["mode"] == "cold" for e in switches)
                and switches[0]["memory_before"]["allocated_bytes"] - switches[0]["memory_after"]["allocated_bytes"]
                    >= .95 * switches[0]["state_bytes"]["cuda_bytes"]
                and all(e["reused_forwards"] == 0 for e in steps[16:]))
        output_equal = bool(np.array_equal(reference["output"], candidate["output"]))
        latent_equal = bool(torch.equal(reference["latents"], candidate["latents"]))
        path_equal = self.paths(reference) == self.paths(candidate)
        trajectory = valid(reference) and valid(candidate)
        passed = output_equal and latent_equal and path_equal and trajectory
        self.record(dict(type="correctness", case_id=case, status="completed" if passed else "failed",
            output_exact_match=output_equal, latent_exact_match=latent_equal, path_exact_match=path_equal,
            trajectory_verified=trajectory, memory_release_verified=trajectory, mode="cold", interleave=True, cache=True,
            elapsed_ms=candidate["elapsed_ms"], reference_elapsed_ms=reference["elapsed_ms"],
            baseline_scope="same_cold_prefix_suffix_trajectory", suffix_config=dict(FULL),
            reference=self.save_array(case + "-reference", reference["output"]),
            candidate=self.save_array(case, candidate["output"]),
            latent_reference=self.save_array(case + "-reference", reference["latents"].numpy(), "latent"),
            latent_candidate=self.save_array(case, candidate["latents"].numpy(), "latent"),
            events=candidate["events"], reference_events=reference["events"], switches=candidate["switches"],
            comparison_scope="same cold/suffix trajectory; not equality to continuous cached trajectory"))
        if not passed:
            raise RuntimeError("cold suffix/interleaving contract failed")

    def probe(self):
        import torch
        frames = self.job["frame_probe"]
        shape = dict(self.config["shapes"]["formal"], frames=frames, steps=32)
        self.config = copy.deepcopy(self.config)
        self.config["shapes"]["long"] = shape
        cases = []
        for cache in (False, True):
            case = "probe-%df-%s" % (frames, "cache" if cache else "full")
            result = self.generate(self.spec(case, cache=cache, threshold=.08))
            margin = max(512 * 1024 ** 2, int(result["memory_baseline"]["device_total_bytes"] * .05))
            baseline = result["memory_baseline"]
            incremental = result["peak_allocated_bytes"] - baseline["allocated_bytes"]
            headroom = baseline["driver_free_bytes"] + baseline["reserved_bytes"] - baseline["allocated_bytes"]
            safe = incremental + margin <= headroom
            row = self.record(dict(type="shape_probe_case", case_id=case, status="completed", shape=shape,
                cache_config=dict(cache=cache, threshold=.08 if cache else 0., max_continuous=1),
                safe_to_run=safe, next_peak_bytes=incremental, safety_margin_bytes=margin,
                peak_allocated_bytes=result["peak_allocated_bytes"], peak_reserved_bytes=result["peak_reserved_bytes"],
                memory_baseline=baseline, memory_final=result["memory_final"], boundaries=result["boundaries"],
                elapsed_ms=result["elapsed_ms"], events=result["events"],
                output=self.save_array(case, result["output"]), latent=self.save_array(case, result["latents"].numpy(), "latent")))
            cases.append(row)
            torch.cuda.empty_cache()
        self.record(dict(type="shape_probe", case_id="probe-%df" % frames, status="completed", shape=shape,
            safe_to_run=all(r["safe_to_run"] for r in cases), next_peak_bytes=max(r["next_peak_bytes"] for r in cases),
            peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in cases), cases=[r["case_id"] for r in cases],
            full_compute_service_ms=cases[0]["elapsed_ms"], memory=cases[0]["memory_baseline"]))

    def quality_row(self, case, spec, result, prompt, reference=None, **extra):
        from ..gpu.quality import evaluate_quality
        import numpy as np
        output = self.save_array(case, result["output"])
        latent = self.save_array(case, result["latents"].numpy(), "latent")
        if reference is None:
            # A self-reference has exactly zero metric distance, without a model call.
            finite = bool(np.isfinite(result["output"]).all())
            in_range = bool(finite and result["output"].min() >= 0 and result["output"].max() <= 1)
            metrics = dict(status="ok" if in_range else "invalid", finite=finite, range_valid=in_range,
                lpips_mean=0. if in_range else None, lpips_per_frame=[0.] * spec["frames"] if in_range else [],
                temporal_delta_rmse=0., psnr_db=None, psnr_is_infinite=True, exact_match=True,
                lpips_status="identity_reference", quality_scope="uncached_reference_identity")
        else:
            array = np.load(reference["path"], mmap_mode="r", allow_pickle=False)
            metrics = evaluate_quality(array, result["output"], lpips_device="cpu")
            del array
        return self.record(dict(type="quality_reference" if reference is None else "quality",
            case_id=case, status="completed", split=prompt["split"], prompt_id=prompt["id"], seed=spec["seed"],
            shape=shape_of(spec), config=self.model_config, model_config=self.model_config,
            cache_config={k: spec[k] for k in FULL}, elapsed_ms=result["elapsed_ms"], events=result["events"],
            switches=result["switches"], boundaries=result.get("boundaries", []), memory_baseline=result["memory_baseline"],
            peak_allocated_bytes=result["peak_allocated_bytes"], peak_reserved_bytes=result["peak_reserved_bytes"],
            output=output, latent=latent, reference=reference or output, quality=metrics,
            **dict(dict(quality_scope="whole_request_fixed_policy", measurement_scope="fixed"), **extra)))

    def calibration(self):
        self.adopt_shape()
        atomic_json(self.suite / "selected-shape.json", dict(schema_version="previs2-selected-shape.v1",
            source_identity_sha256=self.identity, protocol_sha256=self.protocol_sha, selected=self.selected_probe,
            selection_rule="largest measured frame count passing real peak plus safety margin"))
        dev = [p for p in self.config["prompts"] if p["split"] == "dev"]
        seeds = self.config["dev_seeds"]
        configs = [dict(FULL)] + [dict(cache=True, threshold=t, max_continuous=1) for t in self.config["thresholds"]]
        refs = {}
        # Complete each 4x2 cell before the next cell. No partial cell qualifies a policy.
        for cfg in configs:
            for prompt in dev:
                for seed in seeds:
                    case = "dev-long-%s-%d-t%g" % (prompt["id"], seed, cfg["threshold"])
                    spec = self.spec(case, prompt=prompt, seed=seed, **cfg)
                    result = self.generate(spec)
                    row = self.quality_row(case, spec, result, prompt, refs.get((prompt["id"], seed)))
                    if not cfg["cache"]:
                        refs[prompt["id"], seed] = row["output"]
        rows = self.previous_records()
        initial = select_initial_policy(rows, device=self.device, shape=self.config["shapes"]["long"],
            dev_prompt_ids=[p["id"] for p in dev], seeds=seeds, epsilon=.05, model_config=self.model_config)
        self.record(dict(type="initial_policy", case_id="long-policy", status="completed", selection=initial))
        atomic_json(self.suite / "initial-policy.json", initial)
        prefix = initial["config"]
        self.freeze_controller(initial, dev)
        # Each action gets all four development prompts before progressing.
        # Continue/cold first answers whether state choice has measurable value.
        for pause in self.config["pause_steps"]:
            for suffix in ("continue", "full_compute"):
                if not prefix["cache"] and suffix == "full_compute":
                    continue
                target = prefix if suffix == "continue" else FULL
                for mode in ("keep", "hot", "cold"):
                    for prompt in dev:
                        seed = seeds[0]
                        case = "resume-%s-%d-%d-%s-%s" % (prompt["id"], seed, pause, mode, suffix)
                        spec = self.spec(case, prompt=prompt, seed=seed, **prefix)
                        result = self.generate(spec, pause=pause, mode=mode, resume_config=target)
                        self.quality_row(case, dict(spec, **target), result, prompt, refs[prompt["id"], seed],
                            quality_scope="resume_policy", measurement_scope="resume", prefix_config=prefix,
                            pause_step=pause, pause_completed_steps=pause, mode=mode, state_mode=mode, suffix=suffix)
                    self.freeze_controller(initial, dev)
        controller = self.freeze_controller(initial, dev)
        # Fresh held-out long quality, after controller freezing. Never used for fitting.
        for prompt in [p for p in self.config["prompts"] if p["split"] == "test"]:
            for seed in self.config["test_seeds"]:
                case = "test-long-%s-%d" % (prompt["id"], seed)
                spec = self.spec(case + "-reference", prompt=prompt, seed=seed)
                reference = self.quality_row(spec["request_id"], spec, self.generate(spec), prompt)
                cached = self.spec(case + "-selected", prompt=prompt, seed=seed, **prefix)
                self.quality_row(cached["request_id"], cached, self.generate(cached), prompt, reference["output"],
                    predicted_trajectory_lpips=initial["predicted_trajectory_lpips"], controller_sha256=controller["sha256"])

    def freeze_controller(self, initial, dev):
        table = build_recovery_table(self.previous_records(), device=self.device, shape=self.config["shapes"]["long"],
            prefix_config=initial["config"], dev_prompt_ids=[p["id"] for p in dev],
            first_seed=self.config["dev_seeds"][0], epsilon=.05, model_config=self.model_config)
        value = dict(schema_version="previs2-controller.v1", source_identity_sha256=self.identity,
            protocol_sha256=self.protocol_sha, initial=initial, recovery=table, fixed_state=select_fixed_state(table),
            frozen_before_test=True, development_only=True, shape=self.config["shapes"]["long"])
        value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        atomic_json(self.suite / "controller.json", value)
        return value

    def load_controller(self):
        value = read_json(self.suite / "controller.json")
        if value["source_identity_sha256"] != self.identity or value["protocol_sha256"] != self.protocol_sha:
            raise RuntimeError("controller identity mismatch")
        digest = value.pop("sha256")
        if hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() != digest:
            raise RuntimeError("controller content changed")
        value["sha256"] = digest
        return value

    def capacity(self, state, next_peak, logical_budget=None):
        m = self.worker._memory()
        size = self.worker._inventory(state)["cuda_bytes"]
        return capacity_feasibility(total_bytes=m["device_total_bytes"], driver_free_bytes=m["driver_free_bytes"],
            allocated_bytes=m["allocated_bytes"], reserved_bytes=m["reserved_bytes"], victim_storage_bytes=size,
            next_peak_bytes=next_peak, logical_budget_bytes=logical_budget,
            logical_live_bytes=size if logical_budget is not None else None,
            next_state_bytes=size if logical_budget is not None else None)

    def allocation_check(self, next_peak, already_resident=0):
        """Re-read actual headroom after completed copies, before another request allocates."""
        memory = self.worker._memory()
        margin = max(512 * 1024 ** 2, int(memory["device_total_bytes"] * .05))
        available = memory["driver_free_bytes"] + memory["reserved_bytes"] - memory["allocated_bytes"] - margin
        required = max(0, next_peak - already_resident)
        return dict(feasible=available >= required, memory=memory, margin_bytes=margin,
            available_bytes=available, required_incremental_bytes=required,
            reason="actual_headroom_sufficient" if available >= required else "actual_headroom_insufficient")

    def enumerate(self, state, controller, capacity, changes=0, committed_action=None):
        initial = controller["initial"]
        costs = {}
        table = controller["recovery"]
        for mode in ("keep", "hot"):
            measured = [a for a in table["actions"] if a["mode"] == mode and a["suffix"] == "continue"]
            costs[mode] = dict(snapshot_ms=statistics.median(a["snapshot_ms"] for a in measured) if measured else None,
                restore_ms=statistics.median(a["restore_ms"] for a in measured) if measured else None)
        prediction = initial["predicted_trajectory_lpips"] if committed_action is None else committed_action["predicted_trajectory_lpips"]
        curve = (initial.get("remaining_by_completed_step_ms") or {}) if committed_action is None else committed_action.get("remaining_by_completed_step_ms", {})
        request = dict(device=self.device, shape=shape_of(state.spec), model_config=self.model_config,
            selected_cache_config={k: state.spec[k] for k in FULL}, initial_cache_config=initial["config"],
            predicted_trajectory_lpips=prediction, step_index=state.step_index,
            remaining_ms=curve.get(str(state.step_index)),
            state_cost_ms=costs, policy_change_count=changes, lossy_choices_used=changes)
        return enumerate_recovery_actions(request, table, capacity, epsilon=.05)

    def mechanism(self):
        self.adopt_shape()
        controller = self.load_controller()
        # Measure an actual short full-compute allocation peak before capacity decisions.
        prompt = [p for p in self.config["prompts"] if p["split"] == "dev"][0]
        short_spec = self.spec("mechanism-short-reference", "short", prompt=prompt, seed=101)
        short = self.generate(short_spec)
        short_row = self.quality_row(short_spec["request_id"], short_spec, short, prompt,
            quality_scope="mechanism_reference", measurement_scope="mechanism_reference")
        short_peak = short["peak_allocated_bytes"] - short["memory_baseline"]["allocated_bytes"]
        atomic_json(self.suite / "short-profile.json", dict(next_peak_bytes=short_peak,
            service_ms=short["elapsed_ms"], shape=shape_of(short_spec), record=short_row,
            source_identity_sha256=self.identity, protocol_sha256=self.protocol_sha))
        prefix = controller["initial"]["config"]
        references = [r for r in self.previous_records() if r.get("type") == "quality_reference"
            and r.get("shape") == self.config["shapes"]["long"] and r.get("prompt_id") == prompt["id"] and r.get("seed") == 101]
        if not references:
            raise RuntimeError("missing mechanism reference")
        # Prefer a fully covered boundary, in the preregistered order 16,8,24.
        pause = next((s for s in (16, 8, 24) if all(any(a["step_index"] == s and a["mode"] == m
                    for a in controller["recovery"]["actions"]) for m in ("keep", "hot", "cold"))), None)
        if pause is None:
            self.write_gate(False, False, ["no_boundary_with_complete_three_mode_quality_coverage"])
            return
        cases = []
        for repetition in range(2):
            # Reverse on repeat to avoid assigning every warm-up effect to keep.
            for mode in (("keep", "hot", "cold") if repetition == 0 else ("cold", "hot", "keep")):
                case = "mechanism-%d-%s" % (repetition, mode)
                cases.append(self.mechanism_case(case, prompt, prefix, pause, mode, controller, short_peak, references[0]["output"], repetition))
        executed = [r for r in cases if r["status"] == "completed"]
        success = bool(executed) and all(r["quality"].get("status") == "ok"
            and r["quality"].get("lpips_mean", 1) <= .05 and r["capacity_reusable_verified"] for r in executed)
        success = success and all(any(r["repetition"] == rep and r["mode"] in ("hot", "cold") for r in executed) for rep in (0, 1))
        selected_cache = prefix["cache"] is True
        capacity_pressure = any(r["natural_capacity"]["modes"]["keep"]["feasible"] is False and
            r["natural_capacity"]["modes"][r["mode"]]["feasible"] for r in executed)
        crossovers = []
        for mode in ("hot", "cold"):
            ratios = []
            for rep in (0, 1):
                base = next((r for r in executed if r["mode"] == "keep" and r["repetition"] == rep), None)
                candidate = next((r for r in executed if r["mode"] == mode and r["repetition"] == rep), None)
                if base is not None and candidate is not None:
                    ratios.append(candidate["resume_critical_ms"] / base["resume_critical_ms"])
            # A repeatable >5% change, excluding text encoding/interloper variance.
            if len(ratios) == 2 and max(ratios) < .95:
                crossovers.append(dict(mode=mode, paired_cost_ratios=ratios))
        natural = bool(capacity_pressure or crossovers)
        mechanism = success and selected_cache
        reasons = ([] if selected_cache else ["development_selected_full_compute_no_cached_admission"]) + (
            [] if natural else ["keep_feasible_and_no_repeatable_state_cost_crossover"])
        self.write_gate(mechanism, natural, reasons, cases=[r["case_id"] for r in cases], crossovers=crossovers)
        atomic_json(self.suite / "confirm-case.json", dict(prompt_id=prompt["id"], seed=101, pause_step=pause,
            shape=self.config["shapes"]["long"], prefix_config=prefix, modes=["keep", "hot", "cold"],
            suffix_by_mode={r["mode"]: r["actual_action"]["suffix"] for r in executed if r["repetition"] == 0},
            selection_rule="first development prompt, first seed, covered boundary order16/8/24; all modes regardless of winner"))

    def mechanism_case(self, case, prompt, prefix, pause, mode, controller, next_peak, reference, repetition):
        import torch
        from ..gpu.quality import evaluate_quality
        import numpy as np
        spec = self.spec(case, prompt=prompt, seed=101, **prefix)
        state = self.worker.prepare(spec)
        try:
            while state.step_index < pause:
                self.check_budget(); self.worker.step(state)
            urgent_arrival = time.monotonic()
            natural = self.capacity(state, next_peak)
            candidates = self.enumerate(state, controller, natural)
            fixed = choose_action("fixed_state_qc", candidates["actions"], fixed_state=controller["fixed_state"]["mode"])
            joint = choose_action("joint_state_qc", candidates["actions"], fixed_state=controller["fixed_state"]["mode"])
            selected = choose_action("joint_state_qc", [a for a in candidates["actions"] if a["mode"] == mode]).get("action")
            self.record(dict(type="decision", case_id=case, status="enumerated", completed_steps=pause,
                prefix_config=prefix, prefix_matches=prefix == controller["recovery"]["prefix_config"],
                capacity=natural, candidates=candidates["actions"], excluded=candidates["excluded"],
                fixed_selection=fixed, joint_selection=joint, selected_action=selected,
                selection_scope="controlled_mode_probe", natural_capacity=natural))
            if selected is None:
                return self.record(dict(type="mechanism_case", case_id=case, status="infeasible",
                    mode=mode, state_mode=mode, repetition=repetition, natural_capacity=natural,
                    reason="no_legal_quality_capacity_action", excluded=candidates["excluded"],
                    measurement_scope="controlled_point_arrival", resource_scope="natural", logical_budget=False))
            snapshot = self.worker.snapshot(state, mode)
            # snapshot is synchronous: copying has completed before this next request allocates.
            allocation = self.allocation_check(next_peak)
            if not allocation["feasible"]:
                return self.record(dict(type="mechanism_case", case_id=case, status="infeasible",
                    mode=mode, state_mode=mode, repetition=repetition, natural_capacity=natural,
                    actual_capacity=allocation, reason="post_snapshot_headroom_insufficient", snapshot=snapshot,
                    measurement_scope="controlled_point_arrival", resource_scope="natural", logical_budget=False))
            other_spec = self.spec(case + "-urgent", "short", prompt=prompt, seed=102)
            urgent = self.generate(other_spec)
            self.active_case = case
            urgent_response_ms = (time.monotonic() - urgent_arrival) * 1000
            restore = self.worker.restore(state)
            state.spec.update(selected["config"])
            suffix_start = time.monotonic()
            while state.step_index < spec["steps"]:
                self.check_budget(); self.worker.step(state)
            latent = state.latents.cpu().clone()
            output = self.worker.finish(state)
            suffix_wall_ms = (time.monotonic() - suffix_start) * 1000
            events = copy.deepcopy(state.events)
            released = snapshot["memory_before"]["allocated_bytes"] - snapshot["memory_after"]["allocated_bytes"]
            release_ok = mode == "keep" or released >= .95 * snapshot["state_bytes"]["cuda_bytes"]
            reference_array = np.load(reference["path"], mmap_mode="r", allow_pickle=False)
            quality = evaluate_quality(reference_array, output)
            del reference_array
            critical = snapshot["ms"] + restore["ms"] + sum(e["ms"] for e in events if e["kind"] == "decode"
                or e["kind"] == "step" and e["step_index"] >= pause)
            return self.record(dict(type="mechanism_case", case_id=case, status="completed", shape=shape_of(spec),
                repetition=repetition, prompt_id=prompt["id"], seed=101, pause_step=pause, mode=mode, state_mode=mode,
                quality_scope="whole_prefix_suffix_trajectory", measurement_scope="controlled_point_arrival", resource_scope="natural",
                quality=quality, output=self.save_array(case, output), latent=self.save_array(case, latent.numpy(), "latent"),
                reference=reference, urgent_output=self.save_array(case + "-urgent", urgent["output"]),
                events=events, snapshot=snapshot, restore=restore, switches=[snapshot, restore],
                state_bytes=snapshot["state_bytes"], memory_before=snapshot["memory_before"], memory_after=snapshot["memory_after"],
                natural_capacity=natural, actual_capacity=allocation, logical_budget=False, capacity_reusable_verified=release_ok,
                next_request_allocation_and_completion_verified=True, copy_complete_before_release=True,
                selected_action=selected, actual_action=dict(mode=mode, suffix=selected["suffix"], config=selected["config"]),
                estimated_resume_critical_ms=selected["estimated_total_ms"], resume_critical_ms=critical,
                cost_error_ms=critical - selected["estimated_total_ms"] if selected["estimated_total_ms"] is not None else None,
                urgent_e2e_ms=urgent_response_ms, urgent_service_ms=urgent["elapsed_ms"],
                urgent_deadline_ms=1.5 * read_json(self.suite / "short-profile.json")["service_ms"],
                urgent_deadline_met=urgent_response_ms <= 1.5 * read_json(self.suite / "short-profile.json")["service_ms"],
                suffix_wall_ms=suffix_wall_ms, elapsed_ms=sum(e["ms"] for e in events) + urgent["elapsed_ms"]))
        finally:
            self.worker.release(state)

    def write_gate(self, mechanism, natural, reasons, **extra):
        gate = dict(schema_version="previs2-gate.v1", status="passed" if mechanism and natural else "failed",
            source_identity_sha256=self.identity, protocol_sha256=self.protocol_sha,
            mechanism_gate=bool(mechanism), natural_resource_gate=bool(natural), reasons=reasons, **extra)
        atomic_json(self.suite / "gate.json", gate)
        self.record(dict(type="gate", case_id="mechanism-natural", gate_id="natural_resource", required=True,
            passed=bool(mechanism and natural), scope="natural", **gate))

    def online(self):
        from .run_previs2 import gate_passed
        from ..gpu.quality import evaluate_quality
        if not gate_passed(self.suite, self.job["config"], self.job["source_identity"])[0]:
            raise RuntimeError("online mechanism/natural gate is not passed")
        self.adopt_shape()
        controller = self.load_controller()
        dev = [p for p in self.config["prompts"] if p["split"] == "dev"]
        test = [p for p in self.config["prompts"] if p["split"] == "test"]
        references = {}
        for row in self.previous_records():
            if row.get("type") == "quality_reference" and row.get("device") == self.device:
                references[(json.dumps(row["shape"], sort_keys=True), row["prompt_id"], row["seed"])] = row
        # The short shape has independent calibration; it may not inherit a long-shape cache prediction.
        for cfg in [dict(FULL)] + [dict(cache=True, threshold=t, max_continuous=1) for t in self.config["thresholds"]]:
            for prompt in dev:
                for seed in self.config["dev_seeds"]:
                    case = "online-dev-short-%s-%d-t%g" % (prompt["id"], seed, cfg["threshold"])
                    spec = self.spec(case, "short", prompt=prompt, seed=seed, **cfg)
                    key = (json.dumps(shape_of(spec), sort_keys=True), prompt["id"], seed)
                    row = self.quality_row(case, spec, self.generate(spec), prompt,
                        references[key]["output"] if cfg["cache"] else None)
                    if not cfg["cache"]:
                        references[key] = row
        short_initial = select_initial_policy(self.previous_records(), device=self.device,
            shape=self.config["shapes"]["short"], dev_prompt_ids=[p["id"] for p in dev],
            seeds=self.config["dev_seeds"], epsilon=.05, model_config=self.model_config)
        short_controller = dict(controller, initial=short_initial, recovery=dict(status="unavailable", actions=[],
            device=self.device, shape=self.config["shapes"]["short"], prefix_config=short_initial["config"]))
        controllers = dict(short=short_controller, long=controller)
        atomic_json(self.root / "controllers-frozen-before-test.json", controllers)
        profiles, peaks = {}, {}
        for kind in ("short", "long"):
            rows = [r for r in self.previous_records() if r.get("type") == "quality_reference"
                and r.get("device") == self.device and r.get("split") == "dev"
                and r.get("shape") == self.config["shapes"][kind] and r.get("measurement_scope") == "fixed"]
            profiles[kind] = statistics.median(r["elapsed_ms"] for r in rows)
            all_shape = [r for r in self.previous_records() if r.get("device") == self.device
                and r.get("shape") == self.config["shapes"][kind] and "peak_allocated_bytes" in r and "memory_baseline" in r]
            peaks[kind] = max(r["peak_allocated_bytes"] - r["memory_baseline"]["allocated_bytes"] for r in all_shape)
        # Independent per-input reference E2E, with CPU encoding, for deadlines and offline quality.
        for kind in ("short", "long"):
            for prompt in test:
                for seed in self.config["test_seeds"]:
                    key = (json.dumps(self.config["shapes"][kind], sort_keys=True), prompt["id"], seed)
                    if key not in references:
                        case = "online-ref-%s-%s-%d" % (kind, prompt["id"], seed)
                        spec = self.spec(case, kind, prompt=prompt, seed=seed)
                        references[key] = self.quality_row(case, spec, self.generate(spec), prompt)
        traces = []
        for trace_seed in self.config["trace_seeds"]:
            rng = random.Random(trace_seed)
            ordered = [(kind, prompt, seed) for kind in ("short", "long") for prompt, seed in zip(test, [201, 202, 201, 202])]
            rng.shuffle(ordered)
            trace, arrival = [], 0.
            # Four bursts of two; exponential inter-burst spacing normalized to one shared offered load.
            gaps = [rng.expovariate(1.) for _ in range(3)]
            span = sum(profiles[k] for k, _, _ in ordered) / 1.2
            for index, (kind, prompt, seed) in enumerate(ordered):
                if index and index % 2 == 0:
                    arrival += gaps[index // 2 - 1] / sum(gaps) * span
                ref = references[json.dumps(self.config["shapes"][kind], sort_keys=True), prompt["id"], seed]
                trace.append(dict(request_id="r%d" % index, kind=kind, shape=dict(self.config["shapes"][kind]), prompt_id=prompt["id"], seed=seed,
                    prompt=prompt["text"], arrival_ms=arrival,
                    deadline_ms=arrival + (1.5 if kind == "short" else 4.) * ref["elapsed_ms"],
                    uncached_reference_service_ms=ref["elapsed_ms"], reference=ref["output"],
                    initial_cache_config=controllers[kind]["initial"]["config"]))
            traces.append(dict(trace_seed=trace_seed, trace=trace))
        atomic_json(self.root / "traces-frozen.json", dict(load=1.2, pattern="bursty", traces=traces,
            source_identity_sha256=self.identity, protocol_sha256=self.protocol_sha))
        for item in traces:
            # Reserve both arms; a partial pair is retained but excluded by reporting.
            self.check_budget(2.3 * sum(profiles[r["kind"]] for r in item["trace"]) / 1000)
            policies = ("fixed_state_qc", "joint_state_qc") if item["trace_seed"] % 2 == 0 else ("joint_state_qc", "fixed_state_qc")
            for policy in policies:
                self.run_trace(item["trace"], policy, item["trace_seed"], controllers, peaks)

    def run_trace(self, trace, policy, trace_seed, controllers, peaks):
        import numpy as np
        from ..gpu.quality import evaluate_quality
        start = time.monotonic()
        now = lambda: (time.monotonic() - start) * 1000
        states, results, changes, pending_actions, committed_actions = {}, {}, {}, {}, {}
        active = None
        funnel = dict(opportunities=0, eligible_candidates=0, selected_nonkeep=0, executed_nonkeep=0, reasons={})
        failure = None
        trace_id = "load1.2-bursty-%d" % trace_seed
        inputs = copy.deepcopy(trace)
        deadline_cutoff = None
        try:
            while len(results) < len(trace):
                self.check_budget()
                offered = [r for r in trace if r["arrival_ms"] <= now() and r["request_id"] not in results]
                if not offered:
                    delay = min(r["arrival_ms"] for r in trace if r["request_id"] not in results) - now()
                    time.sleep(max(0., min(.05, delay / 1000)))
                    continue
                next_row = min(offered, key=lambda r: (r["deadline_ms"], r["arrival_ms"], r["request_id"]))
                nxt = next_row["request_id"]
                if active is not None and active != nxt:
                    victim = states[active]
                    victim_row = next(r for r in trace if r["request_id"] == active)
                    ctrl = controllers[victim_row["kind"]]
                    # Subtract only the next request's state already included in current allocated memory.
                    already_resident = self.worker._inventory(states[nxt])["cuda_bytes"] if nxt in states and states[nxt].residency == "gpu" else 0
                    capacity = self.capacity(victim, max(0, peaks[next_row["kind"]] - already_resident))
                    candidates = self.enumerate(victim, ctrl, capacity, changes.get(active, 0), committed_actions.get(active))
                    funnel["opportunities"] += 1
                    funnel["eligible_candidates"] += len(candidates["actions"])
                    selection = choose_action(policy, candidates["actions"], fixed_state=ctrl["fixed_state"]["mode"])
                    for excluded in candidates["excluded"]:
                        reason = excluded["reason"]
                        funnel["reasons"][reason] = funnel["reasons"].get(reason, 0) + 1
                    self.record(dict(type="decision", case_id=trace_id + "-" + policy + "-" + active + "-%d" % victim.step_index,
                        status="enumerated", trace_id=trace_id, policy=policy, request_id=active,
                        completed_steps=victim.step_index, capacity=capacity, candidates=candidates["actions"],
                        excluded=candidates["excluded"], selected_action=selection.get("action"), selection=selection,
                        initial_cache_config=ctrl["initial"]["config"], prefix_config={k: victim.spec[k] for k in FULL}))
                    action = selection.get("action")
                    if action is None:
                        # EDF desired switch cannot be executed safely: continue the current victim and record it.
                        next_row, nxt = victim_row, active
                    else:
                        snapshot = self.worker.snapshot(victim, action["mode"])
                        pending_actions[active] = action
                        if action["mode"] != "keep":
                            funnel["selected_nonkeep"] += 1; funnel["executed_nonkeep"] += 1
                        if action.get("consumes_recovery_choice"):
                            changes[active] = changes.get(active, 0) + 1
                            committed_actions[active] = action
                        self.record(dict(type="action_execution", case_id=trace_id + "-" + active,
                            status="completed", trace_id=trace_id, policy=policy, selected_action=action,
                            actual_action=dict(mode=action["mode"], suffix=action["suffix"]), snapshot=snapshot,
                            copy_complete_before_release=True))
                        active = None
                if active != nxt:
                    resident = self.worker._inventory(states[nxt])["cuda_bytes"] if nxt in states and states[nxt].residency == "gpu" else 0
                    allocation = self.allocation_check(peaks[next_row["kind"]], resident)
                    self.record(dict(type="capacity_check", case_id=trace_id + "-" + policy + "-" + nxt,
                        status="completed", trace_id=trace_id, policy=policy, request_id=nxt,
                        allocation=allocation, after_snapshot_and_before_next_request=True))
                    if not allocation["feasible"]:
                        results[nxt] = dict(next_row, status="resource_rejected", finish_ms=now(),
                            quality=dict(status="unknown", reason="actual_capacity_insufficient"),
                            deadline_met=False, thirty_second_met=False, allocation=allocation)
                        if nxt in states:
                            self.worker.release(states.pop(nxt))
                        pending_actions.pop(nxt, None)
                        active = None
                        continue
                    if nxt not in states:
                        spec = self.spec(trace_id + "-" + policy + "-" + nxt, next_row["kind"],
                            prompt=dict(text=next_row["prompt"]), seed=next_row["seed"], **next_row["initial_cache_config"])
                        states[nxt] = self.worker.prepare(spec)
                    else:
                        restore = self.worker.restore(states[nxt])
                        action = pending_actions.pop(nxt, None)
                        if action is not None:
                            states[nxt].spec.update(action["config"])
                    active = nxt
                state = states[active]
                self.worker.step(state)
                if state.step_index < state.spec["steps"]:
                    ctrl = controllers[next_row["kind"]]
                    self.record(dict(type="decision_boundary", case_id=trace_id + "-" + policy + "-" + active + "-%d" % state.step_index,
                        status="observed", trace_id=trace_id, policy=policy, request_id=active,
                        completed_steps=state.step_index, now_ms=now(),
                        prefix_matches=changes.get(active, 0) == 0 and {k: state.spec[k] for k in FULL} == ctrl["initial"]["config"],
                        quality_covered_actions=[dict(mode=a["mode"], suffix=a["suffix"]) for a in ctrl["recovery"]["actions"]
                            if a["step_index"] == state.step_index and changes.get(active, 0) == 0],
                        memory=self.worker._memory(),
                        ready_request_ids=[r["request_id"] for r in trace if r["arrival_ms"] <= now() and r["request_id"] not in results],
                        reason="EDF_and_capacity_reconsidered_before_next_step", actual_action="continue_to_completed_boundary"))
                if state.step_index == state.spec["steps"]:
                    latent = state.latents.cpu().clone()
                    output = self.worker.finish(state)
                    finish = now()  # Main E2E ends at decoded output, before output persistence/quality scoring.
                    results[active] = dict(next_row, status="completed", finish_ms=finish,
                        e2e_ms=finish - next_row["arrival_ms"], deadline_met=finish <= next_row["deadline_ms"],
                        thirty_second_met=finish - next_row["arrival_ms"] < 30000,
                        events=copy.deepcopy(state.events), _output=output, _latent=latent.numpy(),
                        actual_cache_config={k: state.spec[k] for k in FULL})
                    self.worker.release(state); del states[active]; active = None
            deadline_cutoff = now()
        except Exception as exc:
            deadline_cutoff = now()
            failure = exc
        finally:
            cleanup_errors = []
            for state in list(states.values()):
                try:
                    self.worker.release(state)
                except Exception as exc:
                    cleanup_errors.append(str(exc))
        # Scoring holds the GPU process and is charged, but never delays an arrival inside this trace.
        for request_id, row in results.items():
            if row["status"] != "completed":
                continue
            stem = trace_id + "-" + policy + "-" + request_id
            try:
                row["output"] = self.save_array(stem, row["_output"])
                row["latent"] = self.save_array(stem, row["_latent"], "latent")
                self.check_budget()
                reference = np.load(row["reference"]["path"], mmap_mode="r", allow_pickle=False)
                candidate = np.load(row["output"]["path"], mmap_mode="r", allow_pickle=False)
                row["quality"] = evaluate_quality(reference, candidate)
                del reference, candidate
            except BudgetStop:
                row["quality"] = dict(status="unknown", reason="budget_ended_before_scoring")
            except Exception as exc:
                row["quality"] = dict(status="unknown", reason="artifact_or_quality_failure", error=str(exc))
                row["artifact_error"] = str(exc)
                if failure is None:
                    failure = exc
            finally:
                row.pop("_output", None); row.pop("_latent", None)
        accounting = arrival_accounting(trace, deadline_cutoff, [rid for rid, r in results.items() if r["status"] == "completed"])
        accounting["resource_rejected"] = sum(r["status"] == "resource_rejected" for r in results.values())
        unfinished = [dict(r, status="aborted" if r["arrival_ms"] <= deadline_cutoff else "not_arrived",
            quality=dict(status="unknown")) for r in trace if r["request_id"] not in results]
        self.record(dict(type="online_summary", case_id=trace_id + "-" + policy,
            status="completed" if failure is None else "partial", scope="natural", trace_id=trace_id, pair_id=trace_id,
            policy=policy, load=1.2, pattern="bursty", trace_seed=trace_seed, epsilon=.05,
            horizon_ms=deadline_cutoff, elapsed_ms=deadline_cutoff, offered=accounting["offered"],
            arrival_accounting=accounting, requests=list(results.values()) + unfinished, inputs=inputs,
            initial_cache_config={k: c["initial"]["config"] for k, c in controllers.items()},
            funnel=funnel, cleanup_errors=cleanup_errors, error=None if failure is None else str(failure),
            timing_scope="arrival through CPU encoding and decode; raw I/O and offline scoring after trace",
            predicted_quality_is_guarantee=False))
        if failure is not None:
            raise failure

    def confirm(self):
        path = self.suite / "confirm-case.json"
        if not path.exists():
            self.record(dict(type="stage_skip", case_id="v100-confirm", status="not_run", reason="no_completed_A100_mechanism_case"))
            return
        config = read_json(path)
        self.config = copy.deepcopy(self.config)
        self.config["shapes"]["long"] = config["shape"]
        prompt = next(p for p in self.config["prompts"] if p["id"] == config["prompt_id"])
        ref_spec = self.spec("v100-reference", prompt=prompt, seed=config["seed"])
        reference = self.quality_row(ref_spec["request_id"], ref_spec, self.generate(ref_spec), prompt)
        for repetition in (0, 1):
            for mode in ("keep", "hot", "cold"):
                case = "v100-%d-%s" % (repetition, mode)
                spec = self.spec(case, prompt=prompt, seed=config["seed"], **config["prefix_config"])
                suffix = config.get("suffix_by_mode", {}).get(mode, "continue")
                target = config["prefix_config"] if suffix == "continue" else FULL
                run = self.generate(spec, pause=config["pause_step"], mode=mode, resume_config=target)
                self.quality_row(case, dict(spec, **target), run, dict(prompt, split="device_confirmation"), reference["output"],
                    quality_scope="resume_policy", measurement_scope="device_confirmation", mode=mode,
                    state_mode=mode, pause_step=config["pause_step"], suffix=suffix, repetition=repetition,
                    prefix_config=config["prefix_config"])

    def run(self):
        outcome = dict(status="running", stage=self.job["stage"], measured=False, started_at=utc_now())
        try:
            self.setup()
            getattr(self, self.job["stage"])()
            outcome.update(status="completed", measured=True)
        except BudgetStop as exc:
            outcome.update(status="budget_exhausted", reason=str(exc), measured=bool(self.records), active_case=self.active_case)
            self.record(dict(type="unfinished", case_id=self.active_case or self.job["stage"], status="budget_exhausted", reason=str(exc)))
        except Exception as exc:
            outcome.update(status="failed", error=type(exc).__name__ + ": " + str(exc), measured=bool(self.records), active_case=self.active_case)
            traceback.print_exc()
            self.record(dict(type="failure", status="failed", case_id=self.active_case or self.job["stage"], error=outcome["error"]))
        finally:
            outcome.update(elapsed_seconds=time.monotonic() - self.started, records=len(self.records), finished_at=utc_now())
            atomic_json(self.root / "outcome.json", outcome)
        return 0 if outcome["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    return Experiment(read_json(args.job)).run()


if __name__ == "__main__":
    raise SystemExit(main())
