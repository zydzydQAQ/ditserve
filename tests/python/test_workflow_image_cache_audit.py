"""Native-shaped provenance fixtures with real cache producer and no CUDA.

The cell and image response layout is reduced from run03/r010
adaptive-quantum-s3801: notably gpus is an integer and simulated is an
additional cost-model object, not a flag describing the measured requests.
"""

import copy
import types
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_image_cache_audit as cache_audit
from ditserve.evaluation import workflow_next_study as api
from ditserve.models.sana.backend import SanaBackend


class FakeAdapter:
    storage_bytes = 64

    def __init__(self, path, device):
        self.owner = object()

    def encode(self, prompt):
        tensors = [dict(name=name, shape=[1, 4], numel=4, element_size=2,
                        logical_bytes=8, storage_bytes=self.storage_bytes,
                        accounted_bytes=max(8, self.storage_bytes))
                   for name in ("embeddings", "attention_mask")]
        return types.SimpleNamespace(prompt=prompt, owner=self.owner, encode_ms=10.,
                                     storage_details=lambda: copy.deepcopy(tensors))

    def generate(self, prompt, *, seed=0, encoded=None):
        if encoded is not None:
            assert encoded.owner is self.owner and encoded.prompt == prompt
        return [[[float(seed)]]], dict(ms=30. if encoded is None else 20., finite=True)

    def close(self):
        pass


def receipt(backend, identifier, prompt, seed=7):
    backend.prepare(dict(state_id=identifier, spec=dict(prompt=prompt, seed=seed)))
    advanced = backend.advance(dict(state_id=identifier))
    backend.release(dict(state_id=identifier))
    return advanced["prompt_cache"]


def fixture():
    """Keep native field types while using small deterministic model receipts."""
    profile = dict(classes={"sana-1024": dict(kind="image", service_ms=30.)})
    workload = []
    for i, frames in enumerate((33, 33, 17, 17)):
        klass = f"wan1.3b-480x832x{frames}"
        profile["classes"][klass] = dict(kind="video", steps=20, blocks_per_step=60,
            step_compute_ms=1200., decode_chunks=9 if frames == 33 else 5)
        workload.append(dict(request_id=f"v{i}", kind="video", **{"class": klass}, arrival_ms=0.,
            deadline_ms=100000., spec=dict(prompt=api.PROMPTS[i], seed=390100+i, height=480,
                width=832, frames=frames, steps=20, guidance_scale=5., negative_prompt="")))
    for i in range(20):
        workload.append(dict(request_id=f"i{i}", kind="image", **{"class": "sana-1024"},
            arrival_ms=i*10., deadline_ms=10000., spec=dict(prompt=api.PROMPTS[i % 4], seed=390104+i)))
    policies = [dict(id=name, policy="block_elastic", **cache_audit.OPTIONS) for name in ("reference", "candidate")]
    config = api.cluster_config(profile, workload, policies)
    phases = []
    with patch("ditserve.models.sana.runtime.SanaAdapter", FakeAdapter):
        for phase in config["phases"]:
            phase["record_dispatch_timeline"] = True
            backends, warmed, runtime = {}, [], {}
            for slot in range(2):
                name = f"sana{slot}"
                settings = copy.deepcopy(config["workers"][name]["config"])
                settings["options"] = {} if phase["id"] == "reference" else dict(
                    prompt_cache_entries=4, prompt_cache_bytes=16*1024**2)
                phase["worker_overrides"][name] = settings
                backends[name] = SanaBackend(dict(settings, model_path="/fake"))
                runtime[name] = dict(pid=1000+slot, backend="sana", measured=True,
                    cpu_binding=dict(configured=True, requested_affinity=settings["cpu_affinity"],
                        effective_affinity=settings["cpu_affinity"], requested_threads=4, torch_threads=4))
            for index, item in enumerate(phase["warmup"]):
                warm = dict(worker=item["worker"], load_ms=5., elapsed_ms=100.)
                if index >= 2:
                    warm["prompt_cache"] = receipt(backends[item["worker"]], f"warmup-{index}", item["spec"]["prompt"])
                warmed.append(warm)
            rows = copy.deepcopy(phase["workload"])
            for row in rows:
                row["status"] = "completed"
                row["spec"]["request_id"] = row["request_id"]
                if row["kind"] == "video":
                    row["dispatch_timeline"] = [dict(group=[0], quantum=15, block_count=15,
                        chosen_quantum_ms=300., executed_blocks=15, start_ms=float(i), finish_ms=float(i+1))
                        for i in range(80)]
                    row["dispatch_timeline"] += [dict(group=[0], quantum="finish", decode_frames=1,
                        start_ms=100.+i, finish_ms=101.+i) for i in range(profile["classes"][row["class"]]["decode_chunks"])]
                    continue
                index = int(row["request_id"][1:])
                slot = index % 2
                name = f"sana{slot}"
                event = receipt(backends[name], row["request_id"], row["spec"]["prompt"], row["spec"]["seed"])
                start = 1000. + 100 * index
                row.update(start_ms=start, finish_ms=start+50, worker_pids={name: 1000+slot},
                    operation_counts=dict(prepare=1, advance=1, finish=1, release=1), quanta=1, degrees=[1],
                    model_ms=event["service_ms"], outputs=[dict(prompt_cache=event)],
                    dispatch_timeline=[dict(group=[slot], quantum="atomic", start_ms=start+1, finish_ms=start+49)])
            phases.append(dict(id=phase["id"], status="completed", cells=[dict(status="completed",
                gpus=2, policy="block_elastic", real_backends_pinned_sp1=True, real_sp=None,
                simulated=dict(scope="cost-model replay of this cell's trace; not a measurement"),
                scheduler_options=dict(cache_audit.OPTIONS, degrees=[1], pin_started=True),
                text_prefetch_depth=1, requests=rows, warmup=warmed, worker_runtime=runtime)]))
            for backend in backends.values():
                backend.close()
    return dict(schema="ditserve.run.v1", status="completed", config=config, phases=phases)


def images(result, phase=1):
    return [row for row in result["phases"][phase]["cells"][0]["requests"] if row["kind"] == "image"]


class ImageCacheAuditTests(unittest.TestCase):
    def setUp(self):
        self.result = fixture()

    def rejected(self, result):
        checked = cache_audit.audit(result)
        self.assertFalse(checked["passed"], checked)
        self.assertTrue(checked["errors"])

    def test_native_layout_replays_actual_producer_and_keeps_warmup_separate(self):
        checked = cache_audit.audit(self.result)
        self.assertTrue(checked["passed"], checked["errors"])
        self.assertEqual(checked["arms"]["candidate"]["measured_hits"], 17)
        self.assertEqual(checked["arms"]["reference"]["measured_hits"], 0)
        candidate = checked["arms"]["candidate"]["workers"]
        self.assertEqual([candidate[name]["warmup_requests"] for name in candidate], [1, 1])
        self.assertEqual([candidate[name]["measured"]["requests"] for name in candidate], [10, 10])
        self.assertEqual(candidate["sana0"]["counters"]["misses"], 2)
        self.assertEqual(candidate["sana0"]["measured"]["misses"], 1)
        self.assertEqual(candidate["sana0"]["peak_retained_bytes"], 256)
        self.assertIn("four recurring prompts", checked["scope"])
        reverse = copy.deepcopy(self.result)
        reverse["config"]["phases"].reverse()
        reverse["phases"].reverse()
        self.assertTrue(cache_audit.audit(reverse)["passed"])

    def test_missing_warmup_and_hidden_prewarming_fail_closed(self):
        for mutation in (lambda r: r["phases"][1]["cells"][0]["warmup"][2].pop("prompt_cache"),
                         lambda r: r["phases"][1]["cells"][0]["warmup"][2]["prompt_cache"].update(sequence=2),
                         lambda r: r["phases"][1]["cells"][0]["warmup"][2]["prompt_cache"]["before"].update(entries=1)):
            value = copy.deepcopy(self.result); mutation(value); self.rejected(value)

    def test_cannot_trust_self_reported_hit_order_prompt_or_owner(self):
        for field, value in (("outcome", "miss"), ("sequence", 99), ("prompt_sha256", "0"*64),
                             ("adapter_id", "a"*32), ("request_id", "other")):
            changed = copy.deepcopy(self.result)
            images(changed)[0]["outputs"][0]["prompt_cache"][field] = value
            self.rejected(changed)
        changed = copy.deepcopy(self.result)
        first, third = images(changed)[0], images(changed)[2]
        for key in ("start_ms", "finish_ms", "dispatch_timeline"):
            first[key], third[key] = third[key], first[key]
        self.rejected(changed)

    def test_miss_encoding_cost_and_nonfinite_or_forged_totals_rejected(self):
        for field, value in (("encode_ms", 0.), ("service_ms", 20.), ("generation_ms", float("nan")),
                             ("encode_called", False), ("lookup_ms", -1.)):
            changed = copy.deepcopy(self.result)
            images(changed)[1]["outputs"][0]["prompt_cache"][field] = value
            self.rejected(changed)
        changed = copy.deepcopy(self.result)
        images(changed)[1]["outputs"][0]["prompt_cache"]["cold_service"]["total_ms"] += 1.
        self.rejected(changed)

    def test_tensor_storage_bytes_limits_and_lru_state_are_independently_checked(self):
        for field, value in (("shape", [1, 8]), ("numel", 5), ("logical_bytes", 9),
                             ("accounted_bytes", 8), ("storage_bytes", True)):
            changed = copy.deepcopy(self.result)
            images(changed)[0]["outputs"][0]["prompt_cache"]["entry_tensors"][0][field] = value
            self.rejected(changed)
        for field, value in (("entry_bytes", 1), ("limits", dict(entries=5, bytes=16*1024**2)),
                             ("after", dict(entries=0, bytes=0, lru=[]))):
            changed = copy.deepcopy(self.result)
            images(changed)[0]["outputs"][0]["prompt_cache"][field] = value
            self.rejected(changed)

    def test_configuration_full_set_backend_and_actual_work_cannot_be_relaxed(self):
        mutations = [lambda r: r["config"]["phases"][1]["options"].update(quantum_ms=100),
                     lambda r: r["phases"].append(copy.deepcopy(r["phases"][0])),
                     lambda r: r["phases"][1]["cells"][0]["requests"].pop(),
                     lambda r: r["phases"][1]["cells"][0].update(real_backends_pinned_sp1=False),
                     lambda r: r["phases"][1]["cells"][0]["requests"][0]["dispatch_timeline"].pop(),
                     lambda r: r["phases"][1]["cells"][0]["worker_runtime"]["sana0"]["cpu_binding"].update(torch_threads=8),
                     lambda r: images(r)[0]["worker_pids"].update(sana0=999),
                     lambda r: images(r, 0)[0]["outputs"][0]["prompt_cache"].update(enabled=True)]
        for mutation in mutations:
            changed = copy.deepcopy(self.result); mutation(changed); self.rejected(changed)
        for value in ({}, None, dict(schema="ditserve.run.v1", status="completed", config=None)):
            self.rejected(value)

    def test_real_lru_evictions_and_oversized_bypass_replay(self):
        with patch("ditserve.models.sana.runtime.SanaAdapter", FakeAdapter):
            backend = SanaBackend(dict(model_path="/fake", options=dict(prompt_cache_entries=4,
                                                                       prompt_cache_bytes=16*1024**2)))
            records = []
            # Five unique prompts force an entry eviction; revisiting A proves it
            # was evicted instead of trusting the producer's hit flag.
            for index, prompt in enumerate(("A", "B", "C", "D", "E", "A", "A")):
                value = receipt(backend, str(index), prompt)
                records.append(dict(receipt=value, request_id=str(index), prompt=prompt, warmup=index == 0))
            audited = cache_audit._replay(records, True)
            self.assertEqual(audited["counters"]["evictions"], 2)
            self.assertEqual(audited["counters"]["hits"], 1)
            changed = copy.deepcopy(records)
            changed[4]["receipt"]["evicted"][0]["key"] = changed[1]["receipt"]["prompt_sha256"]
            with self.assertRaisesRegex(ValueError, "LRU replay"):
                cache_audit._replay(changed, True)
            backend.close()
            backend = SanaBackend(dict(model_path="/fake", options=dict(prompt_cache_entries=4,
                                                                       prompt_cache_bytes=16*1024**2)))
            with patch.object(FakeAdapter, "storage_bytes", 9*1024**2):
                value = receipt(backend, "large", "large")
            audited = cache_audit._replay([dict(receipt=value, request_id="large", prompt="large", warmup=False)], True)
            self.assertEqual(audited["counters"]["bypasses"], 1)
            self.assertEqual(audited["peak_retained_bytes"], 0)
            backend.close()


if __name__ == "__main__":
    unittest.main()
