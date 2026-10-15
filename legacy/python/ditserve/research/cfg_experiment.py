"""Finite CFG research stages; no qualification is inherited from the old pilot."""
import contextlib
import time
from pathlib import Path

from .common import read_json, digest
from .dual_experiment import DualExperiment
from .dual_evidence import read_records
from .dual_matrix import inputs, known_quality
from .dual_runtime import ResidentWorker, transfer
from .cfg_policy import resolve_policy


class CfgExperiment(DualExperiment):
    def __init__(self, job):
        super().__init__(job)
        self.backends = dict(a100="flash", v100="efficient")
        self.shape = self.config["shapes"]["main"]
        self.costs = dict(a100=6., v100=15.)
        self.implementations = {"paired_008": "split_prefix" if self.frozen.get("prototype_continuation") else "oracle"}
        if self.frozen.get("prototype_continuation"):
            from .cfg_continuation import screening_decisions
            screening_decisions(self.frozen, [], self.context())
        for role in ("a100", "v100"):
            p = self.root / ("correctness-" + role + ".json")
            if p.exists():
                self.costs[role] = read_json(p).get("off_seconds", self.costs[role])
        if self.frozen.get("implementation_refinement"):
            planning = read_json(self.root / "execution-plan.json")
            for role in ("a100", "v100"):
                self.costs[role] = planning["per_device"][role]["measured_off_trajectory_wall_seconds"]

    def load(self):
        for device in self.job["physical_devices"]:
            role = "a100" if "A100" in device["name"] else "v100"
            self.workers[role] = ResidentWorker(device["uuid"], self.backends[role], self.config["model_path"],
                self.root, self.deadline, research_config=self.config)
            if self.frozen.get("implementation_refinement"):
                encoded = self.frozen["cpu_embeddings"]
                self.workers[role].call("install_encoded", cache_root=encoded["root"],
                    cache_manifest_sha256=encoded["manifest_sha256"],
                    model_manifest_sha256=encoded["model_manifest_sha256"])

    def policy(self, recipe):
        return resolve_policy(self.config, recipe, self.implementations.get(recipe))

    def spec(self, item, recipe="off"):
        policy = self.policy(recipe)
        self.counter += 1
        return dict(self.shape, request_id="cfg-%s-%06d" % (self.job["phase"], self.counter),
                    prompt=item["prompt"], negative_prompt="", seed=item["seed"], guidance_scale=5.,
                    cache=policy["cache"], threshold=policy["threshold"], max_continuous=policy["max_continuous"])

    def generate(self, item, recipe="off", *, role="a100", action="stay", instrument=False,
                 diagnostic=False, diagnostic_reference=True, queue=0):
        if not diagnostic_reference and (not diagnostic or not instrument
                or not self.frozen.get("implementation_refinement")
                or not self.job["phase"].startswith("correctness-") or recipe != "paired_008"):
            raise ValueError("same-input-only shadow is restricted to refinement correctness")
        if action not in ("stay", "full_migrate", "cold_migrate", "local_full_migrate"):
            raise ValueError("CFG research action is not declared")
        if diagnostic and action != "stay":
            raise ValueError("diagnostics require an uninterrupted same-device trajectory")
        if queue and role != "v100":
            raise ValueError("real target queue requires a distinct V100 source")
        if action in ("full_migrate", "cold_migrate") and role != "v100":
            raise ValueError("only V100 to A100 migration is declared")
        policy = self.policy(recipe)
        spec = self.spec(item, recipe)
        source = self.workers[role]
        target = self.workers["a100"] if action in ("full_migrate", "cold_migrate") or queue else source
        source.call("warm", spec=spec)
        queue_specs = [self.spec(inputs(self.config)[0], "off") for _ in range(queue)]
        for queued in queue_specs:
            target.call("warm", spec=queued)
        reference_id = None
        request_id = spec["request_id"]
        owner, live, queued_outputs = source, True, []
        try:
            if diagnostic and diagnostic_reference:
                reference_spec = self.spec(item, "off")
                reference_id = reference_spec["request_id"]
                source.call("prepare", spec=reference_spec, policy=self.policy("off"))
            start = time.monotonic()
            prepared = source.call("prepare", spec=spec, policy=policy)
            prefix = source.call("advance", request_id=request_id, until=16, instrument=instrument,
                                 diagnostic=diagnostic, reference_request_id=reference_id)
            checkpoint = source.call("checkpoint", request_id=request_id) if instrument else {}
            boundary = time.monotonic()
            if queue:
                target.start("batch", specs=queue_specs)
            migration, wait_ms = None, 0.
            if action == "local_full_migrate":
                migration = source.call("local_roundtrip", request_id=request_id, action="full_migrate")
            elif action != "stay":
                if role != "v100" or diagnostic:
                    raise ValueError("only complete-boundary V100→A100 non-diagnostic migration is supported")
                if queue:
                    queued_outputs = target.result()
                wait_ms = (time.monotonic() - boundary) * 1000
                migration = transfer(source, target, request_id, action, request_id + "-migration", None)
                owner = target
            suffix = owner.call("advance", request_id=request_id, until=32, instrument=instrument,
                                diagnostic=diagnostic, reference_request_id=reference_id)
            run = owner.call("finish", request_id=request_id, retain_latent=instrument, defer_artifacts=True)
            live = False
            if queue and action == "stay":
                queued_outputs = target.result()
            queue_item = inputs(self.config)[0]
            off_policy = self.policy("off")
            for queued, queued_spec in zip(queued_outputs, queue_specs):
                queued.update(generation_id=queued_spec["request_id"], policy=off_policy,
                    recipe_id="off", device_role="a100", action="stay", action_step=16,
                    prompt_id=queue_item["prompt_id"], seed=queue_item["seed"], split="dev",
                    identity=dict(context_sha256=self.context(), input=queue_item, recipe="off", action="stay",
                                  path=["a100"], attention_path=["flash"], boundary=16,
                                  policy_sha256=off_policy["sha256"], implementation="native"))
            decoded = max(r["decoded_at_monotonic"] for r in [run] + queued_outputs)
            elapsed, critical = (decoded - start) * 1000, (decoded - boundary) * 1000
            for resident in dict.fromkeys([owner] + ([target] if queue else [])):
                pending = [r for r in [run] + queued_outputs if r.get("artifacts_pending")
                           and (resident is owner if r is run else resident is target)]
                if pending:
                    saved = resident.call("flush", request_ids=[r["request_id"] for r in pending])
                    for r in pending:
                        r.update(saved[r["request_id"]])
            run.update(elapsed_ms=elapsed, critical_path_ms=critical, target_wait_ms=wait_ms,
                action=action, action_step=16, recipe_id=recipe, device_role=role,
                prompt_id=item["prompt_id"], seed=item["seed"], split=item["split"],
                generation_id=request_id, policy=policy, migration=migration, queued_outputs=queued_outputs,
                all_requests_decoded=True, text_encoding=prepared["text_encoding"],
                traces=prefix["traces"] + suffix["traces"], inventories=prefix["inventories"] + suffix["inventories"],
                checkpoint=checkpoint, diagnostic=diagnostic,
                diagnostic_scope=("same_input_and_off_trajectory" if diagnostic_reference else
                                  "same_input_prediction_only_correctness") if diagnostic else None,
                timing_scope="frozen_text_mechanism_including_RPC_wait_transfer_decode; artifact_IO_after_measurement",
                identity=dict(context_sha256=self.context(), input=item, recipe=recipe, action=action,
                    path=[role] if action in ("stay", "local_full_migrate") else ["v100", "a100"],
                    attention_path=[source.backend] + ([owner.backend] if owner is not source else []), boundary=16,
                    policy_sha256=policy["sha256"], implementation=policy["implementation"]))
            if policy["implementation"] == "oracle" and not diagnostic:
                details = [e.get("cfg_research", {}) for e in run["events"] if e.get("kind") == "step"]
                if len(details) != 32 or any(e.get("implementation") != "oracle"
                        or e.get("policy_sha256") != policy["sha256"]
                        or any(key not in e for key in ("snapshot_ms", "discarded_probe_ms", "restore_ms"))
                        for e in details):
                    raise ValueError("oracle cost bound requires all 32 measured overhead records")
                snapshots = sum(e.get("snapshot_ms", 0.) for e in details)
                discarded = sum(e.get("discarded_probe_ms", 0.) + e.get("restore_ms", 0.) for e in details)
                run["optimistic_bound"] = dict(method="oracle_excluding_snapshot_and_discarded_trial",
                    elapsed_ms=elapsed, snapshot_ms=snapshots, discarded_trial_ms=discarded,
                    optimistic_ms=elapsed - snapshots - discarded)
            return run
        finally:
            if reference_id and not source.pending:
                with contextlib.suppress(Exception):
                    source.call("release", request_id=reference_id)
            if live and not owner.pending:
                with contextlib.suppress(Exception):
                    owner.call("release", request_id=request_id)

    def paired_row(self, item, candidate, role, phase, order, *, diagnostic=False):
        unit_id = "%s-%s-%s-%s-%s-%d" % (phase, role, candidate, "forward" if order[0] == "off" else "reverse",
                                         item["prompt_id"], item["seed"])
        multiplier = 5 if diagnostic else 4 if self.policy(candidate)["implementation"] == "oracle" else 2
        with self.unit(unit_id, self.costs[role] * multiplier * 1.3 + 10):
            runs = {recipe: self.generate(item, recipe, role=role, diagnostic=diagnostic and recipe != "off")
                    for recipe in order}
            reference, run = runs["off"], runs[candidate]
            quality = self.quality(run, reference)
            self.record("cfg_screen" if phase == "stage_a" else "cfg_diagnostic" if diagnostic else "cfg_calibration",
                role + "-" + item["prompt_id"] + "-" + str(item["seed"]),
                phase=phase, split=item["split"], prompt_id=item["prompt_id"], seed=item["seed"],
                recipe_id=candidate, device_role=role, order=order, pair_id=unit_id, fresh_pair=True,
                status="completed", timing_eligible=phase in ("dev", "holdout") and not diagnostic,
                candidate=run, reference=reference, quality=quality)
        self.costs[role] = max(self.costs[role], reference["elapsed_ms"] / 1000)
        return quality

    def correctness(self, role):
        if getattr(self, "frozen", {}).get("implementation_refinement"):
            return self._refinement_correctness(role)
        item = next(x for x in inputs(self.config) if x["prompt_id"] == "kettle" and x["seed"] == 101)
        checks = []
        for recipe in ("off", "independent_008", "paired_008"):
            split = recipe == "paired_008" and self.implementations[recipe] == "split_prefix"
            with self.unit("correctness-%s-%s" % (role, recipe), self.costs[role] * (6 if split else 5) * 1.3 + 10):
                continuous = self.generate(item, recipe, role=role, instrument=True)
                restored = self.generate(item, recipe, role=role, action="local_full_migrate", instrument=True)
                equal = (continuous["latent_sha256"] == restored["latent_sha256"]
                         and continuous["output_sha256"] == restored["output_sha256"]
                         and continuous["traces"] == restored["traces"])
                # Digests contain only physical state, not action or event timings.
                row = self.record("cfg_correctness", role + "-" + recipe, status="completed", device_role=role,
                    recipe_id=recipe, continuous=continuous, restored=restored, equal=equal,
                    timing_eligible=False, phase="correctness")
                checks.append(row)
                if not equal:
                    raise RuntimeError("same-device policy-bound full restore differs: " + recipe)
                if split:
                    self.implementations[recipe] = "oracle"
                    try:
                        oracle = self.generate(item, recipe, role=role, instrument=True)
                    finally:
                        self.implementations[recipe] = "split_prefix"
                    same = (continuous["latent_sha256"] == oracle["latent_sha256"]
                            and continuous["output_sha256"] == oracle["output_sha256"]
                            and continuous["traces"] == oracle["traces"])
                    self.record("cfg_implementation_equivalence", role + "-paired_008", status="completed",
                        device_role=role, recipe_id=recipe, continuous=oracle, restored=continuous,
                        equal=same, timing_eligible=False, phase="correctness",
                        comparison="current-source oracle versus split_prefix; every complete step")
                    if not same:
                        raise RuntimeError("split-prefix state or output differs from the current oracle")
                if recipe == "off":
                    self.costs[role] = continuous["elapsed_ms"] / 1000
        with self.unit("diagnostic-neutrality-" + role, self.costs[role] * 8 * 1.3 + 10):
            clean = self.generate(item, "independent_008", role=role, instrument=True)
            instrumented = self.generate(item, "independent_008", role=role, instrument=True, diagnostic=True)
            equal = (clean["latent_sha256"] == instrumented["latent_sha256"]
                     and clean["output_sha256"] == instrumented["output_sha256"]
                     and clean["traces"] == instrumented["traces"])
            self.record("cfg_diagnostic_neutrality", role, status="completed", device_role=role,
                        clean=clean, instrumented=instrumented, equal=equal, timing_eligible=False)
            if not equal:
                raise RuntimeError("diagnostic instrumentation changes the trajectory")
        return self.write("correctness-" + role, dict(status="completed", off_seconds=self.costs[role],
                          shape=self.shape, backends=self.backends, checks_passed=True))

    def _refinement_correctness(self, role):
        """Three new trajectories triangulate restore, shadow and oracle state.

        Shadow still runs a full same-input prediction whenever cache is reused.
        It does not need a second off trajectory for this state-equivalence
        check; no denoising-error curve or performance qualification is claimed.
        """
        item = next(x for x in inputs(self.config) if x["prompt_id"] == "kettle" and x["seed"] == 101)
        planning = read_json(self.root / "execution-plan.json")
        estimate = planning["per_device"][role]["correctness_unit_estimate_wall_seconds"]
        def equal(left, right):
            return (left["latent_sha256"] == right["latent_sha256"]
                    and left["output_sha256"] == right["output_sha256"]
                    and len(left["traces"]) == len(right["traces"]) == 32
                    and left["traces"] == right["traces"])
        with self.unit("refinement-correctness-" + role, estimate):
            diagnostic = self.generate(item, "paired_008", role=role, instrument=True,
                                       diagnostic=True, diagnostic_reference=False)
            restored = self.generate(item, "paired_008", role=role,
                                     action="local_full_migrate", instrument=True)
            self.implementations["paired_008"] = "oracle"
            try:
                oracle = self.generate(item, "paired_008", role=role, instrument=True)
            finally:
                self.implementations["paired_008"] = "split_prefix"
            restored_equal, oracle_equal = equal(diagnostic, restored), equal(oracle, diagnostic)
            neutral = restored_equal and oracle_equal and equal(oracle, restored)
            scope = "three current-source trajectories: oracle, split shadow continuous, split clean full restore"
            self.record("cfg_correctness", role + "-paired_008", status="completed", device_role=role,
                recipe_id="paired_008", continuous=diagnostic, restored=restored, equal=restored_equal,
                timing_eligible=False, phase="correctness", comparison=scope)
            self.record("cfg_implementation_equivalence", role + "-paired_008", status="completed",
                device_role=role, recipe_id="paired_008", continuous=oracle, restored=diagnostic,
                equal=oracle_equal, timing_eligible=False, phase="correctness", comparison=scope)
            self.record("cfg_diagnostic_neutrality", role + "-paired_008", status="completed",
                device_role=role, recipe_id="paired_008", clean=restored, instrumented=diagnostic,
                equal=neutral, timing_eligible=False, phase="correctness", comparison=scope,
                off_trajectory_latent_error_measured=False)
            if not neutral:
                raise RuntimeError("refinement oracle, diagnostic or restored trajectory differs")
        return self.write("correctness-" + role, dict(status="completed", off_seconds=self.costs[role],
            shape=self.shape, backends=self.backends, checks_passed=True,
            validation_mode="refinement_three_trajectories", diagnostic_gpu_validated=True))

    def screen(self, role):
        if self.frozen.get("prototype_continuation"):
            raise ValueError("continuation uses separately bound prototype decisions; do not mix implementation screens")
        selected = [x for x in inputs(self.config) if (x["prompt_id"], x["seed"]) in
                    (("origami", 101), ("origami", 102), ("kettle", 101))]
        # Diagnosis is separate from the non-shadow candidate screening costs.
        for item in selected:
            self.paired_row(item, "independent_008", role, "diagnostic", ["off", "independent_008"], diagnostic=True)
        for candidate in ("independent_004", "paired_008"):
            for item in selected:
                self.paired_row(item, candidate, role, "stage_a", ["off", candidate])
        return self.write("screen-" + role, dict(status="completed", phase="stage_a", device_role=role))

    def calibrate(self, role):
        from .cfg_continuation import screening_decisions
        from .cfg_refinement import development_candidates
        rows = read_records(self.root, self.frozen)
        screens = screening_decisions(self.frozen, rows, self.context())
        approved = []
        for candidate in development_candidates(self.frozen, screens):
            gate = screens[candidate]
            if gate.get("eligible"):
                if candidate == "paired_008" and self.implementations[candidate] != "split_prefix":
                    raise RuntimeError("paired gate passed: implement and qualify split-prefix adapter before formal timing")
                approved.append(candidate)
        if not approved:
            return self.write("calibrate-" + role, dict(status="direction_stopped", reason="no screened candidate eligible"))
        for candidate in approved:
            for order in (["off", candidate], [candidate, "off"]):
                for item in inputs(self.config):
                    self.paired_row(item, candidate, role, "dev", order)
        return self.write("calibrate-" + role, dict(status="completed", candidates=approved))

    def holdout(self, role):
        selected = read_json(self.root / "selection.json")["selected"]
        if selected == "paired_008":
            self.implementations[selected] = "split_prefix"
        for item in inputs(self.config, "holdout"):
            self.paired_row(item, selected, role, "holdout", ["off", selected])
        return self.write("holdout-" + role, dict(status="completed", selected=selected))


def run(job):
    experiment = CfgExperiment(job)
    try:
        experiment.load()
        if job["phase"] in ("migration", "mechanism"):
            from . import cfg_migration
            return getattr(cfg_migration, job["phase"])(experiment)
        phase, role = job["phase"].rsplit("-", 1)
        method = {"correctness": experiment.correctness, "screen": experiment.screen,
                  "calibrate": experiment.calibrate, "holdout": experiment.holdout}[phase]
        return method(role)
    finally:
        experiment.close()
