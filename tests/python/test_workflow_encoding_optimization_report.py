"""CPU-shaped online fixtures exercise provenance, pairing and capacity gates."""

import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_encoding_optimization_report as encoding_report
from ditserve.evaluation import workflow_encoding_optimization_study as encoding
from ditserve.evaluation import workflow_encoding_qualification as qualification
from ditserve.evaluation import workflow_next_study as api
from ditserve.models.sana.backend import SanaBackend
from test_workflow_image_cache_audit import FakeAdapter, receipt
from test_workflow_encoding_qualification import arms as numeric_arms, config as numeric_config, row as numeric_row
import test_workflow_encoding_optimization_study as study_fixtures


class FusedFakeAdapter(FakeAdapter):
    def encode_and_generate(self, prompt, *, seed, on_encoded):
        encoded = self.encode(prompt)
        on_encoded(encoded)
        output, _ = self.generate(prompt, seed=seed, encoded=encoded)
        return encoded, output, dict(ms=30., encode_ms=10., generation_ms=20., finite=True,
                                    timing_scope="fused_miss_host_wall")


def fingerprint(identifier):
    return dict(sha256=hashlib.sha256(identifier.encode()).hexdigest(), finite=True)


def fixture(optimization="sana_prompt_cache", distribution="repeated", repeat=0):
    base = study_fixtures.EncodingOptimizationStudyTests()
    base.setUp()
    config = encoding.native_online_config(base.profile, base.rows, optimization=optimization,
                                           prompt_distribution=distribution, repeat=repeat)
    phases = []
    with patch("ditserve.models.sana.runtime.SanaAdapter", FusedFakeAdapter):
        for phase_index, phase in enumerate(config["phases"]):
            runtime, backends, warmed, wan_states = {}, {}, [], [{}, {}]
            for name, settings in phase["worker_overrides"].items():
                runtime[name] = dict(pid=1000 + phase_index * 10 + len(runtime),
                    backend="sana" if name.startswith("sana") else "wan", measured=True,
                    cpu_binding=dict(configured=True, requested_affinity=settings["cpu_affinity"],
                        effective_affinity=settings["cpu_affinity"], requested_threads=settings["cpu_threads"],
                        torch_threads=settings["cpu_threads"]))
                if name.startswith("sana"):
                    backends[name] = SanaBackend(dict(settings, model_path="/fake"))

            def attach_wan(target, slot, identifier, spec, prefetch):
                target.update(cpu_prompt_encode_ms=10.)
                cfg = dict(model_path="/models/wan2.1-t2v-1.3b")
                packet = numeric_row("wan", dict(request_id=identifier, spec=spec), phase["id"], wan_states[slot], cfg)["cache"]
                if packet is not None:
                    packet.update(instance_id=(str(phase_index + 1) + str(slot + 1)) * 16,
                        cumulative_includes_warmup=True, cpu_prompt_encode_ms=10.,
                        encode_call_ms=9., lookup_ms=.1, clone_ms=.2, store_ms=.3)
                    target["negative_prompt_cache"] = packet
                event = dict(op="prepare", worker=f"wan{slot}", pid=runtime[f"wan{slot}"]["pid"],
                             cpu_prompt_encode_ms=10.)
                if packet is not None:
                    event["negative_prompt_cache"] = copy.deepcopy(packet)
                if prefetch:
                    event["ticket"] = "ticket-" + identifier
                    target["encoding_events"] = [dict(copy.deepcopy(event), op="encode_poll"), event]
                else:
                    target["encoding_events"] = [event]

            for index, item in enumerate(phase["warmup"]):
                name = item["worker"]
                spec = dict(item["spec"], request_id=f"warmup-{index}")
                warm = dict(worker=name, worker_pid=runtime[name]["pid"], spec=spec,
                            state_id=f"warmup-{index}", load_ms=1., elapsed_ms=100.)
                if index >= 2:
                    warm["prompt_cache"] = receipt(backends[name], f"warmup-{index}", spec["prompt"])
                elif optimization == "wan_negative_cache":
                    attach_wan(warm, index, f"warmup-{index}", spec, False)
                warmed.append(warm)
            rows = copy.deepcopy(phase["workload"])
            for index, row in enumerate(rows):
                slot = index % 2
                name = ("sana" if row["kind"] == "image" else "wan") + str(slot)
                start = 100000. + index * 10000.
                row.update(status="completed", start_ms=start, finish_ms=start+5000,
                           worker_pids={name: runtime[name]["pid"]})
                row["spec"]["request_id"] = row["request_id"]
                final = dict(output=fingerprint(row["request_id"]))
                if row["kind"] == "image":
                    packet = receipt(backends[name], row["request_id"], row["spec"]["prompt"], row["spec"]["seed"])
                    final["prompt_cache"] = packet
                    row.update(operation_counts=dict(prepare=1, advance=1, finish=1, release=1),
                        quanta=1, degrees=[1], model_ms=packet["service_ms"],
                        dispatch_timeline=[dict(group=[slot], quantum="atomic", start_ms=start+1, finish_ms=start+49)])
                else:
                    cost = phase["profile"]["classes"][row["class"]]
                    quantum = max(1, min(math.floor(300 / (cost["step_compute_ms"] / cost["blocks_per_step"])),
                                         cost["blocks_per_step"]))
                    blocks, timeline = 20 * cost["blocks_per_step"], []
                    while blocks:
                        executed = min(quantum, blocks)
                        timeline.append(dict(group=[slot], quantum=quantum, block_count=quantum,
                            executed_blocks=executed, chosen_quantum_ms=300,
                            start_ms=start+len(timeline), finish_ms=start+len(timeline)+1))
                        blocks -= executed
                    for _ in range(cost["decode_chunks"]):
                        timeline.append(dict(group=[slot], quantum="finish", decode_frames=1,
                            start_ms=start+len(timeline), finish_ms=start+len(timeline)+1))
                    row["dispatch_timeline"] = timeline
                    row["operation_counts"] = dict(prepare=1, release=1)
                    final["latent"] = fingerprint("latent" + row["request_id"])
                    if optimization == "wan_negative_cache":
                        attach_wan(row, slot, row["request_id"], row["spec"], True)
                row["outputs"] = [final]
            cell = dict(status="completed", gpus=2, policy="block_elastic", real_backends_pinned_sp1=True,
                real_sp=None, text_prefetch_depth=1, elapsed_ms=400000., requests=rows, warmup=warmed,
                worker_runtime=runtime, scheduler_options=dict(phase["options"], degrees=[1], pin_started=True))
            phases.append(dict(id=phase["id"], status="completed", cells=[cell]))
            for backend in backends.values():
                backend.close()
    seed = (encoding.SANA_SEEDS if optimization == "sana_prompt_cache" else encoding.WAN_SEEDS)[repeat]
    job = dict(id="fixture", configuration_sha256=api.digest(config), devices=[0, 1],
        gpu_mapping=[dict(ordinal=i, uuid=f"GPU-{i}") for i in (0, 1)],
        metadata=dict(role="encoding_online", optimization=optimization, prompt_distribution=distribution,
                      repeat=repeat, seed=seed))
    result = dict(schema="ditserve.run.v1", status="completed", config=copy.deepcopy(config), phases=phases)
    return result, job, config


def launch(job):
    samples, memory = [], {}
    for mapping in job["gpu_mapping"]:
        device = mapping["ordinal"]
        for timestamp in (100, 200):
            samples.append(dict(device=device, uuid=mapping["uuid"], timestamp_ns=timestamp,
                                memory_total_bytes=24*1024**3, memory_free_bytes=4*1024**3,
                                memory_used_bytes=20*1024**3))
        memory[str(device)] = dict(samples=2, total_bytes=24*1024**3,
                                  peak_used_bytes=20*1024**3, minimum_free_bytes=4*1024**3)
    return dict(gpu_monitor=dict(samples=samples, memory=memory, interval_seconds=2))


class EncodingReportTests(unittest.TestCase):
    def test_sana_three_arms_unique_and_repeated_use_actual_cache_producer(self):
        for distribution in ("unique", "repeated"):
            result, job, config = fixture(distribution=distribution)
            checked = encoding_report.audit_result(result, job, config)
            self.assertTrue(checked["passed"], checked["errors"])
            self.assertEqual(set(checked["arms"]), {"disabled", "staged", "fused"})
            self.assertTrue(all(pair["passed"] for pair in checked["exact"].values()))
            hits = sum(w["measured"]["hits"] for w in checked["arms"]["fused"]["sana"].values())
            self.assertEqual(hits, 0 if distribution == "unique" else 17)

    def test_wan_absent_disabled_receipts_and_prefetch_prepare_are_accepted(self):
        result, job, config = fixture("wan_negative_cache", "mixed_negative")
        self.assertNotIn("negative_prompt_cache", result["phases"][0]["cells"][0]["warmup"][0])
        checked = encoding_report.audit_result(result, job, config)
        self.assertTrue(checked["passed"], checked["errors"])
        workers = checked["arms"]["enabled"]["wan"]
        self.assertEqual(sum(w["measured"]["requests"] for w in workers.values()), 4)
        self.assertEqual(sum(w["measured"]["hits"] for w in workers.values()), 2)
        self.assertLessEqual(max(w["peak_retained_bytes"] for w in workers.values()), 64*1024**2)

    def test_missing_warmup_cache_and_forged_cpu_worker_are_rejected(self):
        result, job, config = fixture()
        mutations = [lambda r: r["phases"][1]["cells"][0]["warmup"][2].pop("prompt_cache"),
                     lambda r: r["phases"][1]["cells"][0]["worker_runtime"]["wan0"]["cpu_binding"].update(torch_threads=16),
                     lambda r: r["config"]["phases"][1].update(text_prefetch_depth=2),
                     lambda r: r["phases"][2]["cells"][0]["requests"][4]["outputs"][0]["prompt_cache"].pop("sync_mode")]
        for mutation in mutations:
            changed = copy.deepcopy(result)
            mutation(changed)
            self.assertFalse(encoding_report.audit_result(changed, job, config)["passed"])

    def test_wan_missing_prefetch_provenance_or_wrong_negative_key_is_rejected(self):
        result, job, config = fixture("wan_negative_cache", "mixed_negative")
        mutations = [lambda row: row["encoding_events"][1].update(ticket="wrong-ticket"),
                     lambda row: row["encoding_events"][0].update(pid=999),
                     lambda row: row.pop("negative_prompt_cache"),
                     lambda row: row["negative_prompt_cache"].update(negative_prompt_sha256="f"*64),
                     lambda row: row.update(cpu_prompt_encode_ms=0)]
        for mutation in mutations:
            changed = copy.deepcopy(result)
            mutation(changed["phases"][1]["cells"][0]["requests"][0])
            self.assertFalse(encoding_report.audit_result(changed, job, config)["passed"])

    def test_final_video_latents_and_image_hashes_must_match(self):
        result, job, config = fixture()
        for index, field in ((0, "latent"), (4, "output")):
            changed = copy.deepcopy(result)
            changed["phases"][1]["cells"][0]["requests"][index]["outputs"][0][field]["sha256"] = "f"*64
            self.assertFalse(encoding_report.audit_result(changed, job, config)["passed"])
        self.assertFalse(encoding_report.audit_result({}, job, config)["passed"])

    def test_numerical_is_recomputed_from_full_matrix_and_not_performance(self):
        config = numeric_config("wan")
        job = dict(configuration_sha256=api.digest(config), metadata=dict(role="encoding_numeric"))
        result = dict(schema=qualification.SCHEMA, status="completed", kind="wan", config=config,
                      arms=numeric_arms("wan", config), performance_evidence=False)
        checked = encoding_report.audit_result(result, job, config)
        self.assertTrue(checked["passed"], checked["errors"])
        self.assertFalse(checked["performance_evidence"])
        result["arms"][1]["result"]["requests"][0]["step_latents"].pop()
        self.assertFalse(encoding_report.audit_result(result, job, config)["passed"])

    def test_raw_capacity_needs_uuid_chronology_and_exact_aggregate(self):
        _, job, _ = fixture()
        original = launch(job)
        self.assertTrue(encoding_report.raw_capacity(original, job)["passed"])
        mutations = [lambda m: m.pop("samples"), lambda m: m["samples"][0].update(uuid="wrong"),
                     lambda m: m["samples"][1].update(timestamp_ns=1),
                     lambda m: m["memory"]["0"].update(samples=3),
                     lambda m: m["samples"][0].update(memory_free_bytes=0)]
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed["gpu_monitor"])
            self.assertFalse(encoding_report.raw_capacity(changed, job)["passed"])

    def test_three_seed_median_uses_paired_changes_and_missing_gate_stays_unqualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs, observations = [], {}
            for repeat in range(3):
                result, job, config = fixture(repeat=repeat)
                job.update(id=f"seed-{repeat}", input=str(Path(tmp)/f"{repeat}.json"), requires=["numeric-gate"])
                api.atomic_json(job["input"], config)
                jobs.append(job)
                observations[job["id"]] = dict(result=result, launch=launch(job), guarded=True, qualified=True, capacity=True)
            gate = dict(id="numeric-gate", metadata={})
            jobs.append(gate)
            observations["numeric-gate"] = dict(result={}, launch={}, guarded=True, qualified=False, capacity=True)
            study = dict(jobs=jobs)
            with patch.object(encoding_report.report, "evidence", side_effect=lambda s, j: observations[j["id"]]):
                failed = encoding_report.summarize(study)
                self.assertFalse(failed["groups"]["sana_prompt_cache/repeated"]["qualified"])
                self.assertIsNone(failed["groups"]["sana_prompt_cache/repeated"]["median"])
                observations["numeric-gate"]["qualified"] = True
                passed = encoding_report.summarize(study)
                group = passed["groups"]["sana_prompt_cache/repeated"]
                self.assertTrue(group["qualified"])
                self.assertEqual(set(group["paired_changes_median"]), {"disabled/staged", "disabled/fused", "staged/fused"})
                self.assertFalse(passed["default_change"])
                self.assertEqual(group["paired_changes_median"]["staged/fused"]["image"]["latency_p95_ms"]["improvement_percent"], 0)


if __name__ == "__main__":
    unittest.main()
