"""Finite same-A100 cache-state experiment. No distributed elasticity claims.

The caller owns exclusive GPU admission, process lifetime and the cumulative
GPU ledger. Importing this module never imports torch or initializes CUDA.
Only newly completed, identity-bound units may open a subsequent phase.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import time


DEFAULTS = dict(frames=[49, 81], height=480, width=832, steps=32,
                guidance_scale=5.0, epsilon=0.05, minimum_gain=0.05,
                pressure_counts=[1, 2, 4, 8, 16], boundary=16,
                artifact_root=None, max_key_samples=4)
PHASES = ("screen", "qualify", "pressure", "actions", "report")
ACTIONS = ("keep", "full_cpu", "flush", "wait_refresh")
ROOT = Path(__file__).resolve().parents[3]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _settings(config):
    settings = dict(DEFAULTS, **config.get("direction2", {}))
    if "quality_epsilon" in settings:
        settings["epsilon"] = settings["quality_epsilon"]
    if settings.get("recipe", "paired_008") != "paired_008":
        raise ValueError("direction2 has exactly one frozen candidate: paired_008")
    fixed = dict(frames=[49, 81], height=480, width=832, steps=32,
                 guidance_scale=5.0, pressure_counts=[1, 2, 4, 8, 16], boundary=16)
    if any(settings[key] != value for key, value in fixed.items()):
        raise ValueError("direction2 finite shape, pressure and sampler matrix is fixed")
    if settings["epsilon"] != .05 or settings["minimum_gain"] != .05:
        raise ValueError("direction2 quality and gain gates are fixed at 0.05")
    if type(settings["max_key_samples"]) is not int or settings["max_key_samples"] < 1:
        raise ValueError("at least one representative key sample must be retained")
    model = config["models"]["wan"]
    if (model["id"] != "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" or
            model["revision"] != "0fad780a534b6463e45facd96134c9f345acfa5b"):
        raise ValueError("direction2 requires the previously validated Wan1.3B revision")
    return settings


def _identity(config, settings):
    files = [Path(__file__), ROOT / "python/ditserve/gpu/worker.py",
             ROOT / "python/ditserve/gpu/quality.py", ROOT / "configs/research_cfg.json"]
    files += [ROOT / ("python/ditserve/research/" + name + ".py") for name in
              ("cfg_adapter", "cfg_split_prefix", "cfg_empty_context", "adapter",
               "storage", "state", "prompt_bank", "cfg_policy")]
    return dict(config_sha256=_digest(config), settings_sha256=_digest(settings),
                implementation_sha256=_digest({str(path.relative_to(ROOT)):
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in files}),
                model=copy.deepcopy(config["models"]["wan"]), device=config.get("device", "cuda:0"))


def historical_evidence():
    """Read existing sealed conclusions; never import their qualifications."""
    paths = dict(pilot="results/research-pilot/run05/report.json",
                 cfg="results/research-cfg/run03/report.json",
                 migration="results/research-migration-diagnostic/run01/summary.json")
    result = dict(inherited_quality_qualification=False, inherited_performance_qualification=False,
                  scope="historical observations only; no 17/33-frame hot-versus-keep rerun",
                  sp_resize=dict(status="blocked", qualified=False,
                    reason="only heterogeneous A100/V100 available; homogeneous SP GPU validation absent"))
    for name, relative in paths.items():
        path = ROOT / relative
        if not path.exists():
            result[name] = dict(status="incomplete", path=str(path))
            continue
        data = _read(path)
        selected = {key: data[key] for key in
                    ("gates", "hypotheses", "migration_qualified", "performance_qualified",
                     "quality_qualified", "cost_measurements_valid") if key in data}
        if name == "cfg":
            selected["qualification_scope"] = "256x448x17, 32 steps, CFG5; each device SP1 only"
        result[name] = dict(path=str(path), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            observations=selected)
    return result


def _inputs(split):
    # The existing finite prompt bank is data, not permission inherited from a
    # previous protocol. Each new trajectory uses this run's implementation.
    catalog = _read(ROOT / "configs/research_cfg.json")
    seeds = [101, 102] if split == "dev" else [201, 202]
    return [dict(prompt_id=p["id"], prompt=p["text"], seed=seed, split=split)
            for p in catalog["prompts"] if p["split"] == split for seed in seeds]


def _screen_inputs():
    required = (("origami", 101), ("kettle", 101), ("kettle", 102))
    values = {(v["prompt_id"], v["seed"]): v for v in _inputs("dev")}
    return [values[key] for key in required]


def _case(item):
    return "%s-%s" % (item["prompt_id"], item["seed"])


def _quality_known(metrics, frames, epsilon=.05):
    value = metrics.get("lpips_mean")
    values = metrics.get("lpips_per_frame", [])
    return (metrics.get("status") == "ok" and metrics.get("finite") is True
            and metrics.get("range_valid") is True and metrics.get("frame_count") == frames
            and metrics.get("shape") == [1, frames, 480, 832, 3]
            and metrics.get("lpips_status") == "ok"
            and len(values) == frames
            and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values)
            and type(value) in (int, float) and math.isfinite(value) and 0 <= value <= epsilon)


def _gain(rows):
    for row in rows:
        for run in row["runs"].values():
            value = run.get("generation_ms")
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    or run.get("performance_eligible") is not True):
                raise ValueError("gain requires complete, positive, non-diagnostic generation timings")
    reference = sum(row["runs"]["off"]["generation_ms"] for row in rows)
    candidate = sum(row["runs"]["paired_008"]["generation_ms"] for row in rows)
    if not reference or not math.isfinite(reference + candidate):
        raise ValueError("generation times must be finite and positive")
    return 1.0 - candidate / reference


def _path_statistics(events):
    paths = [dict(step=e["completed_steps"], reused_forwards=e["reused_forwards"],
                  total_forwards=e.get("total_forwards", 2))
             for e in events if e.get("kind") == "step" and "reused_forwards" in e]
    next_reads = [dict(boundary=prior["step"], next_step=following["step"],
                       bn_feature_read_branches=following["reused_forwards"])
                  for prior, following in zip(paths, paths[1:])]
    return dict(steps=paths, observed_next_step_bn_reads=next_reads,
                reuse_steps=sum(p["reused_forwards"] == 2 for p in paths),
                mixed_steps=sum(p["reused_forwards"] == 1 for p in paths),
                future_reads_scope="Bn features are read on a later accepted-reuse branch; not a general liveness proof")


class _Runtime:
    """Thin adapter over the existing pinned Wan implementation."""

    def __init__(self, config, settings, check_budget):
        from ..gpu.worker import WanStepWorker
        from ..research.adapter import ResearchWanAdapter
        from ..research.cfg_adapter import CfgResearchAdapter
        from ..research.cfg_policy import POLICIES, resolve_policy
        from ..research.prompt_bank import FrozenPromptBank
        self.settings, self.check_budget = settings, check_budget
        self.worker = WanStepWorker(device=config.get("device", "cuda:0"),
                                    model_path=config["models"]["wan"]["path"], attention="flash")
        if "A100" not in self.worker.torch.cuda.get_device_name(self.worker.device):
            raise ValueError("direction2 GPU execution requires the selected A100")
        self.adapter = CfgResearchAdapter(self.worker)
        self.state_adapter = ResearchWanAdapter(worker=self.worker, validation_only=True)
        self.bank = FrozenPromptBank(self.worker)
        self.policies = {name: resolve_policy(dict(recipes=list(POLICIES)), name,
                         "split_prefix" if name == "paired_008" else "native")
                         for name in ("off", "paired_008")}
        self.counter = 0

    def spec(self, item, frames, policy):
        self.counter += 1
        recipe = self.policies[policy]
        return dict(request_id="d2-%08d" % self.counter, prompt=item["prompt"], negative_prompt="",
                    seed=item["seed"], height=self.settings["height"], width=self.settings["width"],
                    frames=frames, steps=32, guidance_scale=5.0, cache=recipe["cache"],
                    threshold=recipe["threshold"], max_continuous=1)

    def step(self, state, policy):
        self.check_budget()
        return self.adapter.step(state, policy=self.policies[policy])

    def prepare(self, item, frames, policy):
        self.check_budget()
        return self.bank.prepare(self.spec(item, frames, policy))

    def generate(self, item, frames, policy, *, diagnostics=False):
        import numpy as np
        from ..research.state import state_inventory
        spec = self.spec(item, frames, policy)
        self.check_budget()
        self.bank.warm(spec)
        self.worker._sync()
        started = time.perf_counter()
        state = None
        boundaries = []
        try:
            state = self.bank.prepare(spec)
            while state.step_index < 32:
                self.step(state, policy)
                if diagnostics and state.step_index < 32:
                    inventory = state_inventory(state, torch_module=self.worker.torch)
                    certificate = self.state_adapter.certificate(state)
                    boundaries.append(dict(step=state.step_index, inventory=inventory,
                                           refresh_eligible=certificate["eligible"],
                                           refresh_reasons=certificate["reasons"]))
            output = self.worker.finish(state)
            elapsed = (time.perf_counter() - started) * 1000
            result = dict(item=copy.deepcopy(item), frames=frames, policy=policy,
                          policy_sha256=self.policies[policy]["sha256"], generation_ms=elapsed,
                          output_sha256=hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
                          path_statistics=_path_statistics(state.events), boundaries=boundaries,
                          performance_eligible=not diagnostics,
                          timing_scope="frozen-text prepare, denoise and decode; encoding setup charged separately",
                          output=output)
            return result
        finally:
            if state is not None and state.request_id in self.worker._states:
                self.worker.release(state)

    def quality(self, reference, candidate):
        from ..gpu.quality import evaluate_quality
        return evaluate_quality(reference, candidate, with_lpips=True, lpips_device="cpu")

    def is_oom(self, error):
        return isinstance(error, self.worker.torch.cuda.OutOfMemoryError)

    def close(self):
        for state in list(self.worker._states.values()):
            self.worker.release(state)
        self.bank.entries.clear()
        self.worker.torch.cuda.empty_cache()

    def pressure_trial(self, items, frames, count, urgent_item):
        """Only genuine request prefixes consume GPU memory; no filler tensors."""
        states, stage = [], "background_prefix"
        inventories, memory = [], None
        started = time.perf_counter()
        def footprint():
            return dict(requests=inventories, paused_requests=len(inventories),
                        necessary_storage_bytes=sum(r["inventory"]["necessary_storage_bytes"] for r in inventories),
                        cache_feature_storage_bytes=sum(r["inventory"]["feature_only_bytes"] for r in inventories),
                        necessary_cuda_bytes=sum(r["inventory"]["cuda_bytes"] - r["inventory"]["feature_only_cuda_bytes"]
                                                 for r in inventories),
                        cache_feature_cuda_bytes=sum(r["inventory"]["feature_only_cuda_bytes"] for r in inventories),
                        scope="sum of actual paused request-owned storages; excludes weights and temporary workspace")
        try:
            for index in range(count):
                item = items[index % len(items)]
                state = self.prepare(item, frames, "paired_008")
                states.append(state)
                while state.step_index < 16:
                    self.step(state, "paired_008")
                inventories.append(dict(index=index, item=copy.deepcopy(item),
                                        inventory=self.state_adapter.inventory(state)))
            stage = "urgent"
            memory = self.worker._memory()
            urgent = self.generate(urgent_item, frames, "off")
            return dict(count=count, status="passed", pressure_found=False,
                        urgent_output_sha256=urgent["output_sha256"], memory_before_urgent=memory,
                        paused_state_bytes=footprint(),
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        scope="capacity probe only; background suffixes are not performance-qualified")
        except Exception as error:
            if not self.is_oom(error):
                raise
            try:
                failure_memory = self.worker._memory()
            except Exception as memory_error:
                failure_memory = dict(status="unavailable", error=str(memory_error))
            return dict(count=count, status="failed", pressure_found=stage == "urgent",
                        oom_stage=stage, completed_prefixes=len(inventories), error=str(error),
                        paused_state_bytes=footprint(), memory_before_urgent=memory,
                        memory_at_failure=failure_memory,
                        elapsed_ms=(time.perf_counter() - started) * 1000)
        finally:
            self.close()

    def action_trial(self, items, frames, count, urgent_item, action, references, *, retain_failure=None):
        """Four same-device actions; exact actions require reference hashes.

        ``references`` contains newly generated off and paired arrays for every
        participating input. All surviving background requests must finish.
        """
        import numpy as np
        states, completed, actions = [], [], []
        urgent = None
        stage, stage_detail = "setup", "urgent_prompt_preencode"
        arrival = memory_before = memory_after = urgent_ms = all_ms = None
        urgent_encoding_setup_ms = None
        def preserve(details, background_indices=None, include_urgent=True):
            if retain_failure is None:
                return dict(raw_available=False, reason="no evidence sink supplied")
            runs = {}
            def add(name, output):
                runs[name] = dict(output=output,
                    output_sha256=hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest())
            selected = range(len(completed)) if background_indices is None else background_indices
            for index in selected:
                item, output = completed[index]
                case = _case(item)
                add("candidate-%03d-%s" % (index, case), output)
                for policy in ("off", "paired_008"):
                    add("reference-%s-%s" % (policy, case), references[case][policy])
            if include_urgent and urgent is not None:
                add("urgent-candidate-" + _case(urgent_item), urgent["output"])
                add("urgent-reference-" + _case(urgent_item), references[_case(urgent_item)]["off"])
            return retain_failure(runs, details)
        try:
            # Every trial has the same frozen-text timing scope, even when
            # references left a warm bank for the first arm and close() cleared
            # it before later arms. CPU encoding remains charged phase setup.
            self.check_budget()
            setup_started = time.perf_counter()
            self.bank.warm(self.spec(urgent_item, frames, "off"))
            urgent_encoding_setup_ms = (time.perf_counter() - setup_started) * 1000
            stage, stage_detail = "prefix", "background_prepare_and_prefix"
            for index in range(count):
                item = items[index % len(items)]
                state = self.prepare(item, frames, "paired_008")
                states.append((state, item))
                while state.step_index < 16:
                    self.step(state, "paired_008")
            memory_before = self.worker._memory()
            self.worker._sync()
            arrival = time.perf_counter()
            stage, stage_detail = "action", action
            for state, item in states:
                stage, stage_detail = "action", action
                self.check_budget()
                if action == "full_cpu":
                    actions.append(self.state_adapter.snapshot(state, "hot"))
                elif action == "flush":
                    before = self.worker._memory()
                    for context in state.contexts.values():
                        context.buffers.clear()
                    state.needs_anchor = True
                    self.worker._sync()
                    actions.append(dict(action="flush", before=before, after=self.worker._memory(),
                                        forced_recompute=True))
                elif action == "wait_refresh":
                    first_step = state.step_index
                    while state.step_index < 32:
                        certificate = self.state_adapter.certificate(state)
                        if certificate["eligible"]:
                            actions.append(self.state_adapter.snapshot(state, "exact_refresh", certificate=certificate))
                            break
                        self.step(state, "paired_008")
                    actions.append(dict(action="wait_refresh", waited_steps=state.step_index-first_step))
                    if state.step_index == 32:
                        stage, stage_detail = "finish", "background_decode_during_wait_refresh"
                        output = self.worker.finish(state)
                        completed.append((item, output))
                        self.worker.release(state)
                        stage, stage_detail = "action", action
                elif action != "keep":
                    raise ValueError("unknown action")
            memory_after = self.worker._memory()
            stage, stage_detail = "urgent", "urgent_prepare_denoise_decode"
            urgent = self.generate(urgent_item, frames, "off")
            urgent_ms = (time.perf_counter() - arrival) * 1000
            for state, item in states:
                if state.residency == "released":
                    continue
                stage, stage_detail = "resume", "background_restore"
                if state.residency == "cpu":
                    actions.append(self.state_adapter.restore(state))
                stage_detail = "background_denoise_suffix"
                while state.step_index < 32:
                    self.step(state, "paired_008")
                stage, stage_detail = "finish", "background_decode"
                completed.append((item, self.worker.finish(state)))
                self.worker.release(state)
            all_ms = (time.perf_counter() - arrival) * 1000
            stage, stage_detail = "finish", "quality_evaluation"
            qualities, exact = [], []
            for item, output in completed:
                reference = references[_case(item)]
                qualities.append(self.quality(reference["off"], output))
                exact.append(bool(np.array_equal(reference["paired_008"], output)))
            urgent_equal = bool(np.array_equal(references[_case(urgent_item)]["off"], urgent["output"]))
            qualified = (len(completed) == count and urgent_equal and
                         all(_quality_known(q, frames) for q in qualities)
                         and (action == "flush" or all(exact)))
            result = dict(action=action, count=count, status="passed" if qualified else "failed",
                        quality_passed=qualified, background_completed=len(completed),
                        all_requests_completed=True, urgent_ms=urgent_ms, all_requests_ms=all_ms,
                        memory_before=memory_before, memory_after_action=memory_after,
                        actions=actions, qualities=qualities, cached_reference_exact=exact,
                        urgent_reference_exact=urgent_equal, transfer_transport="direct_tensor_CPU_copies",
                        no_pickle=True, urgent_encoding_setup_ms=urgent_encoding_setup_ms,
                        timing_scope="arrival after frozen-text encoding and background prefixes; all setup charged to phase budget")
            if not qualified:
                failed = [index for index, quality in enumerate(qualities)
                          if not _quality_known(quality, frames) or action != "flush" and not exact[index]]
                result["retention"] = preserve(result, failed, include_urgent=not urgent_equal)
            return result
        except Exception as error:
            details = dict(action=action, count=count, complete=False, stage=stage, stage_detail=stage_detail,
                background_completed=len(completed), urgent_completed=urgent is not None,
                all_requests_completed=urgent is not None and len(completed) == count,
                actions=actions, memory_before=memory_before, memory_after_action=memory_after,
                urgent_ms=urgent_ms, all_requests_ms=all_ms,
                urgent_encoding_setup_ms=urgent_encoding_setup_ms,
                error="%s: %s" % (type(error).__name__, error))
            retention = preserve(details)
            if not self.is_oom(error):
                raise
            return dict(details, status="failed", quality_passed=False, oom=True,
                        oom_stage=stage, oom_stage_detail=stage_detail, retention=retention)
        finally:
            self.close()


_runtime_factory = _Runtime


class _Experiment:
    def __init__(self, config, output, check_budget, phase):
        self.config, self.output = config, Path(output)
        self.settings, self.check_budget, self.phase = _settings(config), check_budget, phase
        self.identity = _identity(config, self.settings)
        self.started = time.perf_counter()
        self.runtime = None
        self.path = self.output / (phase + ".json")
        self.report = dict(schema_version="ditserve.directions.direction2.v1", phase=phase,
            identity=self.identity, status="incomplete", eligible=False, units=[], active_unit=None,
            cache_quality_qualified=False, cache_performance_qualified=False,
            state_mechanism_qualified=False, sp_resize_qualified=False, hardware_blocked=True,
            scope="same-A100 SP1; frozen-text mechanism, not online goodput or SP resize",
            historical_evidence=historical_evidence())
        self.key_samples = 0

    def save(self):
        self.report["elapsed_wall_seconds"] = time.perf_counter() - self.started
        _write(self.path, self.report)

    def previous(self, phase):
        path = self.output / (phase + ".json")
        if not path.exists():
            return None
        report = _read(path)
        if report.get("identity") != self.identity:
            raise ValueError("prerequisite config/model/implementation differs: " + phase)
        return report

    def unit(self, key, operation):
        self.check_budget()
        self.report["active_unit"] = dict(key=key, status="incomplete", started_at=time.time())
        self.save()
        value = operation()
        value = dict(value, unit_id=key, complete=True)
        _write(self.output / "units" / (self.phase + "-" + key + ".json"), value)
        self.report["units"].append(value)
        self.report["active_unit"] = None
        self.save()
        return value

    def get_runtime(self):
        if self.runtime is None:
            self.check_budget()
            self.runtime = _runtime_factory(self.config, self.settings, self.check_budget)
        return self.runtime

    def retain(self, key, runs, metrics, *, failure=False):
        if not failure and self.key_samples >= self.settings["max_key_samples"]:
            return dict(raw_available=False, reason="evaluated pair retained as metrics and hashes only")
        import numpy as np
        base = (self.settings.get("artifact_root") or self.config.get("artifact_root")
                or str(self.output / "key_samples"))
        run_key = _digest(str(self.output.resolve()))[:16]
        directory = Path(base) / "direction2" / run_key / self.identity["config_sha256"][:12] / self.phase / key
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for policy, run in runs.items():
            path = directory / (policy + ".npy")
            np.save(path, run["output"], allow_pickle=False)
            files.append(dict(path=str(path), policy=policy, output_sha256=run["output_sha256"]))
        self.key_samples += 1
        result = dict(raw_available=True, files=files, failed_quality=failure)
        _write(directory / "manifest.json", dict(identity=self.identity, metrics=metrics, **result))
        return result

    def pair(self, item, frames, order, label, *, captured_runs=None):
        runtime = self.get_runtime()
        runs = {} if captured_runs is None else captured_runs
        try:
            for policy in order:
                runs[policy] = runtime.generate(item, frames, policy)
                partial = dict(item=item, frames=frames, order=list(order), complete=False,
                               runs={name: {k: v for k, v in run.items() if k != "output"}
                                     for name, run in runs.items()},
                               quality_evaluated=False, qualification=False)
                _write(self.output / "partial_units" / (self.phase + "-" + label + ".json"), partial)
            self.check_budget()
            metrics = runtime.quality(runs["off"]["output"], runs["paired_008"]["output"])
            self.check_budget()
        except Exception as error:
            retention = self.retain(label, runs, {}, failure=True) if runs else None
            _write(self.output / "partial_units" / (self.phase + "-" + label + ".json"),
                   dict(item=item, frames=frames, complete=False, qualification=False,
                        completed_trajectories=list(runs), retention=retention,
                        error="%s: %s" % (type(error).__name__, error)))
            if runtime.is_oom(error):
                return dict(item=item, frames=frames, order=list(order), quality_passed=False,
                            runs={name: {k: v for k, v in run.items() if k != "output"}
                                  for name, run in runs.items()},
                            capacity_failed=True, error=str(error), retention=retention)
            raise
        passed = _quality_known(metrics, frames, self.settings["epsilon"])
        retention = self.retain(label, runs, metrics, failure=not passed)
        scalars = {name: {k: v for k, v in run.items() if k != "output"} for name, run in runs.items()}
        return dict(item=item, frames=frames, order=list(order), runs=scalars, quality=metrics,
                    quality_passed=passed, retention=retention)

    def screen(self):
        results = []
        for frames in self.settings["frames"]:
            rows = []
            for item in _screen_inputs():
                key = "%df-%s" % (frames, _case(item))
                row = self.unit(key, lambda item=item, frames=frames, key=key:
                    self.pair(item, frames, ("off", "paired_008"), key))
                rows.append(row)
                if not row["quality_passed"]:
                    break
            passed = len(rows) == 3 and all(r["quality_passed"] for r in rows)
            complete_pairs = [r for r in rows if set(r["runs"]) == {"off", "paired_008"}]
            results.append(dict(frames=frames, quality_screen_passed=passed, observed=len(rows),
                                required=3, descriptive_gain=_gain(complete_pairs) if complete_pairs else None))
        eligible = [r for r in results if r["quality_screen_passed"] and r["descriptive_gain"] >= .05]
        chosen = max(eligible, key=lambda r: r["frames"]) if eligible else None
        self.report.update(status="passed" if chosen else "stopped", eligible=chosen is not None,
                           selected_frames=chosen["frames"] if chosen else None, shapes=results,
                           selection_rule="largest shape passing all three quality screens and descriptive gain>=5%",
                           performance_scope="screen order only; not a performance qualification",
                           reason=None if chosen else "no large shape passed the finite screen; no threshold retuning")

    def qualify(self):
        previous = self.previous("screen")
        if not previous or not previous.get("eligible"):
            self.report.update(status="blocked", reason="current-run screen prerequisite not passed")
            return
        frames = previous["selected_frames"]
        self.report["selected_frames"] = frames
        development = []
        for order in (("off", "paired_008"), ("paired_008", "off")):
            for item in _inputs("dev"):
                key = "%s-%s" % (order[0], _case(item))
                row = self.unit(key, lambda item=item, order=order, key=key:
                                self.pair(item, frames, order, key))
                development.append(row)
                if not row["quality_passed"]:
                    self.report.update(status="stopped", reason="development quality failed; no retuning")
                    return
        gains = [_gain([r for r in development if r["order"][0] == first]) for first in ("off", "paired_008")]
        self.report["development_gains"] = gains
        if not all(value >= .05 for value in gains):
            self.report.update(status="stopped", reason="both development orders must improve at least 5%")
            return
        for item in _inputs("holdout"):
            key = "holdout-" + _case(item)
            row = self.unit(key, lambda item=item, key=key: self.pair(item, frames, ("off", "paired_008"), key))
            if not row["quality_passed"]:
                self.report.update(status="stopped", reason="held-out quality failed; selection remains frozen")
                return
        self.report.update(status="passed", eligible=True, cache_quality_qualified=True,
                           cache_performance_qualified=True, selected_frames=frames,
                           qualification_scope="this shape, A100/Flash/SP1, 8 dev x 2 orders and 4 held-out inputs")

    def pressure(self):
        previous = self.previous("qualify")
        if not previous or not previous.get("cache_performance_qualified") or not previous.get("cache_quality_qualified"):
            self.report.update(status="blocked", reason="new-shape cache qualification is required")
            return
        frames, runtime = previous["selected_frames"], self.get_runtime()
        self.report["selected_frames"] = frames
        def diagnostic():
            run = runtime.generate(_screen_inputs()[0], frames, "paired_008", diagnostics=True)
            boundaries = run["boundaries"]
            refresh = [r["step"] for r in boundaries if r["refresh_eligible"]]
            waits = [next((step-r["step"] for step in refresh if step >= r["step"]), None) for r in boundaries]
            return dict(boundaries=boundaries, refresh_steps=refresh, wait_steps_to_observed_refresh=waits,
                        path_statistics=run["path_statistics"], performance_eligible=False,
                        claim="observed next-read paths and pinned refresh certificates; not general cache liveness")
        self.unit("refresh-diagnostic", diagnostic)
        items, urgent_item = _inputs("dev"), _inputs("holdout")[0]
        reference = self.unit("urgent-reference", lambda: {k: v for k, v in
            runtime.generate(urgent_item, frames, "off").items() if k != "output"})
        prior_count = None
        for count in self.settings["pressure_counts"]:
            row = self.unit("k%d" % count, lambda count=count:
                            runtime.pressure_trial(items, frames, count, urgent_item))
            if row.get("pressure_found"):
                self.report.update(status="passed", eligible=True, pressure_count=count,
                                   below_pressure_count=prior_count, capacity_qualified=False,
                                   reason="urgent OOM with real paused request states; action efficacy unverified")
                return
            if row["status"] != "passed":
                self.report.update(status="stopped", reason="background-prefix OOM does not establish urgent admission opportunity")
                return
            if row["urgent_output_sha256"] != reference["output_sha256"]:
                self.report.update(status="failed", reason="urgent output differs from isolated same-input reference")
                return
            prior_count = count
        self.report.update(status="stopped", eligible=False, reason="keep admits urgent through K=16; no observed capacity opportunity",
                           no_artificial_memory_filling=True)

    def actions(self):
        pressure, qualification = self.previous("pressure"), self.previous("qualify")
        if (not pressure or not pressure.get("eligible") or not qualification
                or not qualification.get("cache_quality_qualified")):
            self.report.update(status="blocked", reason="qualified cache and actual urgent capacity failure are required")
            return
        frames, runtime = pressure["selected_frames"], self.get_runtime()
        self.report["selected_frames"] = frames
        # Fresh references are evaluated once, then shared only by identical
        # inputs in this phase. Failure samples remain on disk via pair().
        references = {}
        items, urgent_item = _inputs("dev"), _inputs("holdout")[0]
        for item in items + [urgent_item]:
            def reference(item=item):
                runs = {}
                row = self.pair(item, frames, ("off", "paired_008"), "reference-" + _case(item),
                                captured_runs=runs)
                if row["quality_passed"]:
                    references[_case(item)] = {p: runs[p]["output"] for p in runs}
                return row
            row = self.unit("reference-" + _case(item), reference)
            if not row["quality_passed"]:
                self.report.update(status="stopped", reason="new action reference quality failed")
                return
        counts = [value for value in (pressure.get("below_pressure_count"), pressure["pressure_count"]) if value is not None]
        trials = []
        for count in counts:
            for order_id, order in enumerate((ACTIONS, tuple(reversed(ACTIONS)))):
                for action in order:
                    key = "k%d-order%d-%s" % (count, order_id, action)
                    def retain_failure(runs, details, key=key):
                        retention = (self.retain("action-" + key, runs, details, failure=True) if runs else
                                     dict(raw_available=False, reason="no decoded outputs completed before failure"))
                        _write(self.output / "partial_units" / (self.phase + "-" + key + ".json"),
                               dict(complete=False, qualification=False, details=details, retention=retention))
                        return retention
                    row = self.unit(key, lambda count=count, action=action:
                        runtime.action_trial(items, frames, count, urgent_item, action, references,
                                             retain_failure=retain_failure))
                    row["order_id"] = order_id
                    trials.append(row)
        outcomes = []
        for count in counts:
            for action in ACTIONS:
                rows = [r for r in trials if r["count"] == count and r["action"] == action]
                passed = len(rows) == 2 and all(r.get("quality_passed") and r.get("all_requests_completed") for r in rows)
                gains = []
                for row in rows:
                    keep = next(r for r in trials if r["count"] == count and r["action"] == "keep"
                                and r["order_id"] == row["order_id"])
                    if passed and keep.get("quality_passed") and keep.get("all_requests_ms", 0) > 0:
                        gains.append(1 - row["all_requests_ms"] / keep["all_requests_ms"])
                outcomes.append(dict(count=count, action=action, observed=len(rows), required=2,
                                     all_completions_and_quality_passed=passed,
                                     gains_vs_keep_when_keep_completes=gains,
                                     repeated_gain_observed=len(gains) == 2 and min(gains) >= .05))
        pressure_keeps = [r for r in trials if r["count"] == pressure["pressure_count"] and r["action"] == "keep"]
        keep_fails = len(pressure_keeps) == 2 and all(r.get("oom") and r.get("oom_stage") == "urgent"
                                                   for r in pressure_keeps)
        admission = [r["action"] for r in outcomes if r["count"] == pressure["pressure_count"]
                     and r["action"] != "keep" and r["all_completions_and_quality_passed"] and keep_fails]
        self.report.update(status="passed", eligible=False, completed_action_matrix=True,
                           state_mechanism_qualified=False, action_trials=trials,
                           baseline_outcomes=outcomes, observed_admission_actions=admission,
                           reason="finite mechanism measurements complete; flush lacks full 8-dev/4-heldout action qualification",
                           qualification_scope="no new-policy or SP claim; failures remain in the comparison denominator")


def run_phase(config, output: Path, check_budget, phase: str):
    """Execute one finite phase. Budget exceptions preserve incomplete units.

    `check_budget()` is called before every unit, model load and denoising step.
    The root must serialize this process with all other GPU experiments.
    """
    if phase not in PHASES:
        raise ValueError("unknown direction2 phase: " + str(phase))
    experiment = _Experiment(config, output, check_budget, phase)
    if experiment.path.exists():
        previous = _read(experiment.path)
        if previous.get("identity") != experiment.identity:
            raise ValueError("refusing to overwrite evidence from a different implementation/config")
        # A failed or interrupted phase is never implicitly rerun for free.
        return previous
    experiment.save()
    try:
        if phase == "report":
            phases = {name: experiment.previous(name) for name in PHASES if name != "report"}
            experiment.report.update(status="passed", phases=phases, eligible=False,
                                     cache_quality_qualified=bool((phases["qualify"] or {}).get("cache_quality_qualified")),
                                     cache_performance_qualified=bool((phases["qualify"] or {}).get("cache_performance_qualified")),
                                     state_mechanism_qualified=False, overall_validation="partial_same_card_only")
        else:
            getattr(experiment, phase)()
    except Exception as error:
        budget = "budget" in type(error).__name__.lower()
        experiment.report.update(status="stopped" if budget else "failed", eligible=False,
                                 reason="%s: %s" % (type(error).__name__, error),
                                 interrupted_unit=experiment.report.get("active_unit"))
    finally:
        if experiment.runtime is not None:
            try:
                experiment.runtime.close()
            except Exception as error:
                experiment.report.update(status="failed", eligible=False,
                                         cleanup_error="%s: %s" % (type(error).__name__, error))
        experiment.save()
    return experiment.report
