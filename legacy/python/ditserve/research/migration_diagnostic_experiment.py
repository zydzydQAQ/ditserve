"""Cache-off numerical controls, never performance or migration qualification."""
import contextlib
import time

from .common import digest, read_json
from .dual_experiment import DualExperiment
from .dual_runtime import ResidentWorker, transfer
from .cfg_policy import resolve_policy


ACTIONS = ("a100_stay", "v100_stay", "full_migrate", "a100_local_restore", "v100_local_restore")


def relative_error(candidate, reference):
    import numpy as np
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise ValueError("diagnostic tensors differ in shape or dtype")
    left, right = candidate.astype(np.float64), reference.astype(np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite diagnostic tensor")
    difference = left - right
    norm = float(np.linalg.norm(right.reshape(-1)))
    return dict(relative_l2=float(np.linalg.norm(difference.reshape(-1))) / max(norm, 1e-12),
                max_abs=float(np.max(np.abs(difference))), reference_l2=norm,
                exact_equal=bool(np.array_equal(candidate, reference)), epsilon=1e-12)


def roundtrip_equal(continuous, restored):
    return (len(continuous.get("traces", [])) == 32
            and continuous["traces"] == restored.get("traces")
            and continuous["signatures"] == restored.get("signatures")
            and all(continuous[key] == restored.get(key) for key in ("latent_sha256", "output_sha256")))


def recompute_checks(runs, shadow):
    source, target = shadow["source"], shadow["target"]
    continuous, migrated = runs["v100_stay"], runs["full_migrate"]
    return dict(a100_local_restore=roundtrip_equal(runs["a100_stay"], runs["a100_local_restore"]),
        v100_local_restore=roundtrip_equal(continuous, runs["v100_local_restore"]),
        migrated_prefix_equal=migrated["traces"][:16] == continuous["traces"][:16],
        same_input=source["before"] == target["before"] == continuous["signatures"]["16"],
        source_shadow_matches_native=source["after"] == continuous["signatures"]["17"],
        target_shadow_matches_migration=target["after"] == migrated["signatures"]["17"],
        source_unchanged_and_paused=shadow["source_status"].get("residency") == "diagnostic_paused"
            and shadow["source_status"].get("signature") == source["before"])


class MigrationDiagnosticExperiment(DualExperiment):
    def __init__(self, job):
        super().__init__(job)
        self.diagnostic = self.frozen["diagnostic"]
        self.comparison = job["phase"]
        self.backends = dict(self.diagnostic["attention_by_comparison"][self.comparison])
        self.policy = resolve_policy(self.config, "off")
        self.planning = read_json(self.root / "execution-plan.json")
        self.unit_estimate = self.planning["comparisons"][self.comparison]["unit_wall_seconds"]

    def load(self):
        for device in self.job["physical_devices"]:
            role = "a100" if "A100" in device["name"] else "v100"
            self.workers[role] = ResidentWorker(device["uuid"], self.backends[role], self.config["model_path"],
                self.root, self.deadline, research_config=self.config, diagnostic_mode=True)
            binding = self.frozen["cpu_embeddings"]
            self.workers[role].call("install_encoded", cache_root=binding["root"],
                cache_manifest_sha256=binding["manifest_sha256"], model_manifest_sha256=binding["model_manifest_sha256"])

    def spec(self, item):
        self.counter += 1
        return dict(self.shape, request_id="migration-diagnostic-%s-%06d" % (self.comparison, self.counter),
                    prompt=item["prompt"], negative_prompt="", seed=item["seed"], guidance_scale=5.,
                    cache=False, threshold=0., max_continuous=1)

    def generate(self, item, action):
        if action not in ACTIONS:
            raise ValueError("unknown diagnostic action")
        role = "v100" if action == "full_migrate" or action.startswith("v100") else "a100"
        source, target = self.workers[role], self.workers["a100"]
        owner, live = source, True
        spec = self.spec(item)
        request_id = spec["request_id"]
        source.call("warm", spec=spec)
        started = time.monotonic()
        prepared = source.call("prepare", spec=spec, policy=self.policy)
        signatures, traces, inventories = {}, [], []
        try:
            for step in (1, 16):
                progress = source.call("advance", request_id=request_id, until=step, instrument=True)
                traces.extend(progress["traces"]); inventories.extend(progress["inventories"])
                signatures[str(step)] = source.call("diagnostic_signature", request_id=request_id)
            checkpoint = source.call("checkpoint", request_id=request_id)
            migration = None
            if action.endswith("local_restore"):
                migration = source.call("local_roundtrip", request_id=request_id, action="full_migrate")
            elif action == "full_migrate":
                migration = transfer(source, target, request_id, "full_migrate", request_id + "-transfer")
                owner = target
            boundary_signature = owner.call("diagnostic_signature", request_id=request_id)
            boundary_equal = boundary_signature == signatures["16"]
            if not boundary_equal:
                raise RuntimeError("complete mathematical state differs before first restored computation")
            for step in (17, 32):
                progress = owner.call("advance", request_id=request_id, until=step, instrument=True)
                traces.extend(progress["traces"]); inventories.extend(progress["inventories"])
                signatures[str(step)] = owner.call("diagnostic_signature", request_id=request_id)
            result = owner.call("finish", request_id=request_id, retain_latent=True, defer_artifacts=True)
            live = False
            elapsed = (time.monotonic() - started) * 1000
            result.update(owner.call("flush", request_ids=[request_id])[request_id])
            result.update(action=action, action_step=16, generation_id=request_id, recipe_id="off", policy=self.policy,
                device_role=role, prompt_id=item["prompt_id"], seed=item["seed"], split="dev", checkpoint=checkpoint,
                traces=traces, signatures=signatures, boundary_signature=boundary_signature, boundary_equal=boundary_equal,
                migration=migration, text_encoding=prepared["text_encoding"], inventories=inventories,
                elapsed_ms=elapsed, timing_eligible=False, diagnostic=True,
                timing_scope="instrumented diagnostic only; includes state hashing and transport; not performance qualification",
                identity=dict(context_sha256=self.context(), input=item, recipe="off", action=action,
                    path=[role, "a100"] if action == "full_migrate" else [role],
                    attention_path=[self.backends[role], self.backends["a100"]] if action == "full_migrate" else [self.backends[role]],
                    policy_sha256=self.policy["sha256"], implementation="native", boundary=16))
            return result
        finally:
            if live:
                with contextlib.suppress(Exception):
                    owner.call("release", request_id=request_id)

    def shadow(self, item, native_source, migrated):
        """Fork one paused source into isolated one-step diagnostic copies."""
        source, target = self.workers["v100"], self.workers["a100"]
        spec = self.spec(item)
        request_id = spec["request_id"]
        source.call("prepare", spec=spec, policy=self.policy)
        copies = []
        try:
            source.call("advance", request_id=request_id, until=16)
            packet = source.call("diagnostic_export", request_id=request_id, snapshot_id=request_id + "-snapshot")
            results = {}
            for role, resident in (("v100", source), ("a100", target)):
                diagnostic_id = request_id + "-" + role
                resident.call("diagnostic_import", snapshot=packet["snapshot"], tensors=packet["tensors"], diagnostic_id=diagnostic_id)
                copies.append((resident, diagnostic_id))
                results[role] = resident.call("diagnostic_step", diagnostic_id=diagnostic_id, return_tensors=True)
                if (results[role].get("target_device") != resident.device
                        or results[role].get("backend") != self.backends[role]
                        or results[role].get("snapshot_sha256") != packet["snapshot"]["sha256"]):
                    raise RuntimeError("shadow result belongs to another device, backend or snapshot")
            left, right = results["a100"], results["v100"]
            source_status = source.call("diagnostic_status", request_id=request_id)
            errors = {key: relative_error(left["tensors"][key], right["tensors"][key])
                      for key in ("cond", "uncond", "cfg", "latent_before", "latent_after")}
            checks = dict(same_input=left["before"] == right["before"] == native_source["signatures"]["16"],
                source_shadow_matches_native=right["after"] == native_source["signatures"]["17"],
                target_shadow_matches_migration=left["after"] == migrated["signatures"]["17"],
                source_unchanged_and_paused=source_status.get("residency") == "diagnostic_paused"
                    and source_status.get("signature") == packet["snapshot"]["state_signature"])
            return dict(checks=checks, errors=errors,
                source={k: v for k, v in right.items() if k != "tensors"},
                target={k: v for k, v in left.items() if k != "tensors"},
                snapshot_sha256=packet["snapshot"]["sha256"], step=17, source_status=source_status,
                production_ownership=False, timing_eligible=False)
        finally:
            for resident, diagnostic_id in copies:
                with contextlib.suppress(Exception):
                    resident.call("diagnostic_drop", diagnostic_id=diagnostic_id)
            with contextlib.suppress(Exception):
                source.call("release", request_id=request_id)

    def execute(self):
        items = []
        for case in self.diagnostic["cases"]:
            prompt = next(p for p in self.config["prompts"] if p["id"] == case["prompt_id"])
            items.append(dict(prompt_id=case["prompt_id"], seed=case["seed"], prompt=prompt["text"], split="dev"))
        rows, previous, maximum = [], {}, self.unit_estimate
        for order in self.diagnostic["repetitions"]:
            for item in items:
                unit_id = "%s-%s-%s" % (order, item["prompt_id"], item["seed"])
                start = time.monotonic()
                with self.unit(unit_id, maximum):
                    actions = list(ACTIONS) if order == "forward" else list(reversed(ACTIONS))
                    runs = {action: self.generate(item, action) for action in actions}
                    shadow = self.shadow(item, runs["v100_stay"], runs["full_migrate"])
                    checks = recompute_checks(runs, shadow)
                    pairs = (("v100_vs_a100", "v100_stay", "a100_stay"),
                             ("migration_vs_a100", "full_migrate", "a100_stay"),
                             ("migration_vs_v100", "full_migrate", "v100_stay"))
                    qualities = {name: self.quality(runs[candidate], runs[reference]) for name, candidate, reference in pairs}
                    repeatability = {}
                    key = (item["prompt_id"], item["seed"])
                    if key in previous:
                        repeatability = {action: dict(
                            output_equal=runs[action]["output_sha256"] == previous[key][action]["output_sha256"],
                            latent_equal=runs[action]["latent_sha256"] == previous[key][action]["latent_sha256"],
                            quality=self.quality(runs[action], previous[key][action])) for action in ACTIONS}
                    previous[key] = runs
                    row = self.record("migration_diagnostic", unit_id, status="completed", comparison=self.comparison,
                        prompt_id=item["prompt_id"], seed=item["seed"], split="dev", recipe_id="off", order=order,
                        checks=checks, checks_passed=all(checks.values()), runs=runs, qualities=qualities, shadow=shadow,
                        repeatability=repeatability, timing_eligible=False, performance_qualified=False,
                        quality_qualified=False, scope="numerical attribution controls only")
                    rows.append(row)
                maximum = max(maximum, (time.monotonic() - start) * 1.3)
                if not all(checks.values()):
                    return self.write(self.comparison, dict(status="correctness_failed", checks_passed=False,
                        completed_units=len(rows), expected_units=4, failed_unit=unit_id,
                        reason="state restore or shadow replay differed; do not attribute to hardware"))
        return self.write(self.comparison, dict(status="completed", checks_passed=True,
            completed_units=len(rows), expected_units=4, row_sha256s=[r["sha256"] for r in rows],
            performance_qualified=False, quality_qualified=False))


def run(job):
    experiment = MigrationDiagnosticExperiment(job)
    try:
        experiment.load()
        return experiment.execute()
    finally:
        experiment.close()
