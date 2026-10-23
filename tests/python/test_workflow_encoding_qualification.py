"""CPU fixtures test qualification failures; no fixture is GPU acceptance."""

from concurrent.futures import Future
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_encoding_qualification as q
from ditserve.evaluation.workflow_serving_overlap import memory_margin_gate

UUID = "GPU-00000000-0000-0000-0000-000000000001"


def config(kind="sana", **overrides):
    value = dict(model_path="/models/" + kind, gpu_uuid=UUID, cpu_mask=list(range(60, 68)))
    value.update(dict(prompt_distribution="repeated") if kind == "sana" else dict(guidance_scale=5.))
    return q.normalize_config(kind, dict(value, **overrides))


def fingerprint(label, shape):
    return dict(sha256=hashlib.sha256(label.encode()).hexdigest(), shape=shape, dtype="float32", finite=True)


def samples():
    return [dict(timestamp_ns=t, uuid=UUID, memory_free_bytes=4 * 1024**3,
                 memory_total_bytes=24 * 1024**3) for t in (100, 200)]


def row(kind, case, arm="disabled", state=None, cfg=None):
    state = {} if state is None else state
    cfg = config(kind) if cfg is None else cfg
    value = dict(deepcopy(case), status="completed", start_ns=100, end_ns=200,
                 nvml_samples=samples(), nvml_errors=[], cpu_embeddings=None, step_latents=[], cache=None)
    value["capacity"] = memory_margin_gate(value["nvml_samples"], [], 100, 200)
    if kind == "sana":
        value["outputs"] = dict(image=fingerprint(case["request_id"], [1, 1024, 1024, 3]))
    else:
        frames = case["spec"]["frames"]
        shape = [1, 16, (frames - 1) // 4 + 1, 60, 104]
        value["cpu_embeddings"] = dict(prompt=fingerprint("positive", [1, 512, 4096]),
            negative=fingerprint(case["spec"]["negative_prompt"], [1, 512, 4096]) if case["spec"]["guidance_scale"] > 1 else None)
        value["step_latents"] = [fingerprint(case["request_id"] + str(step), shape) for step in range(20)]
        value["outputs"] = dict(video=fingerprint(case["request_id"], [1, frames, 480, 832, 3]),
                                latent=deepcopy(value["step_latents"][-1]))
    if arm == "disabled" and kind == "wan":
        return value
    lru = state.setdefault("lru", OrderedDict())
    counters = state.setdefault("counters", dict.fromkeys(("lookups", "hits", "misses", "insertions", "evictions", "bypasses", "errors", "clears"), 0))
    def snapshot():
        return dict(entries=len(lru), bytes=sum(lru.values()), lru=[dict(key=k, bytes=v) for k, v in lru.items()])
    before = snapshot()
    state["sequence"] = state.get("sequence", 0) + 1
    text_hash = hashlib.sha256(case["spec"]["prompt" if kind == "sana" else "negative_prompt"].encode()).hexdigest()
    key = text_hash if kind == "sana" else hashlib.sha256((q._digest(q._wan_recipe(cfg)) + ":" + text_hash).encode()).hexdigest()
    outcome = ("disabled" if arm == "disabled" else "bypass" if kind == "wan" and case["spec"]["guidance_scale"] <= 1
               else "hit" if key in lru else "miss")
    size = 24 if kind == "sana" else 512 * 4096 * 4
    evicted = []
    if outcome in ("hit", "miss"):
        counters["lookups"] += 1
        counters["hits" if outcome == "hit" else "misses"] += 1
        if outcome == "hit":
            lru.move_to_end(key)
        else:
            if len(lru) == 4:
                old, size_old = lru.popitem(last=False)
                evicted.append(dict(key=old, bytes=size_old))
                counters["evictions"] += 1
            lru[key] = size
            counters["insertions"] += 1
    elif outcome == "bypass":
        counters["bypasses"] += 1
    receipt = dict(enabled=arm != "disabled", outcome=outcome, admitted=outcome == "miss",
        schema="ditserve.sana.prompt-cache.v1" if kind == "sana" else "ditserve.wan.negative-prompt-cache.v1",
        request_id=case["request_id"], sequence=state["sequence"], before=before, after=snapshot(),
        counters=dict(counters), evicted=evicted, bypass_reason="cfg_disabled" if outcome == "bypass" else None,
        limits=dict(entries=4, bytes=(16 if kind == "sana" else 64) * 1024**2) if arm != "disabled" else dict(entries=0, bytes=0),
        entry_bytes=size if outcome in ("hit", "miss") else 0)
    receipt["encode_called" if kind == "sana" else "negative_encode_called"] = outcome == "miss"
    if kind == "sana":
        receipt.update(adapter_id="a" * 32, prompt_sha256=text_hash, owner_scope="adapter_instance", key_scope="exact_prompt_utf8",
            entry_tensors=[dict(shape=[1, 2, 4], numel=8, element_size=2, storage_bytes=16, logical_bytes=16, accounted_bytes=16),
                           dict(shape=[1, 2], numel=2, element_size=4, storage_bytes=8, logical_bytes=8, accounted_bytes=8)])
    else:
        recipe = q._wan_recipe(cfg)
        receipt.update(instance_id="b" * 32, negative_prompt_sha256=text_hash, recipe=recipe, recipe_sha256=q._digest(recipe), key=key,
            owner_scope="worker_instance", key_scope="exact_negative_prompt_utf8_and_recipe",
            entry_tensor=dict(shape=[1, 512, 4096], dtype="torch.float32", device="cpu", storage_bytes=size, logical_bytes=size, accounted_bytes=size))
    if arm == "fused":
        receipt.update(sync_mode="fused", timing_scope="fused_miss_host_wall" if outcome == "miss" else "synchronized_preencoded_generation")
    value["cache"] = receipt
    return value


def arms(kind, cfg):
    result = []
    for name in q.ARMS[kind]:
        state = {}
        expected = q.cases(kind, cfg)
        warmup = dict(request_id="warmup", spec=dict(expected[0]["spec"], prompt="Warmup: a blue cup on a table."))
        if kind == "wan":
            warmup["spec"]["request_id"] = "warmup"
        warmup_row = row(kind, warmup, name, state, cfg)
        if state:
            state["lru"].clear()
            state["counters"]["clears"] += 1
        child = dict(schema=q.SCHEMA, kind=kind, arm=name, config=cfg, config_sha256=q._digest(cfg),
                     status="completed", requests=[row(kind, case, name, state, cfg) for case in expected],
                     warmup=warmup_row, cache_initial=None if kind == "wan" and name == "disabled"
                     else dict(entries=0, bytes=0, lru=[]),
                     runtime=dict(matmul_tf32=False, cudnn_tf32=True, interop_threads=1, intraop_threads=8),
                     cpu_affinity=dict(bound_cpus=cfg["cpu_mask"], physical_cores=[dict(cpu=cpu, socket=0, core=cpu)
                                                                                 for cpu in cfg["cpu_mask"]]))
        result.append(dict(id=name, returncode=0, result=child))
    return result


class QualificationTests(unittest.TestCase):
    def test_fixed_matrices_and_explicit_runtime_contract(self):
        unique = q.cases("sana", config(prompt_distribution="unique"))
        repeated = q.cases("sana", config())
        self.assertEqual(len(unique), 8)
        self.assertEqual(len({r["spec"]["prompt"] for r in unique}), 8)
        self.assertEqual([r["spec"]["prompt"] for r in repeated[:4]], [r["spec"]["prompt"] for r in repeated[4:]])
        for guidance in (1., 5.):
            cfg = config("wan", guidance_scale=guidance)
            wan = q.cases("wan", cfg)
            self.assertEqual(len(wan), 8)
            self.assertEqual({r["spec"]["frames"] for r in wan}, {17, 33})
            self.assertEqual({r["spec"]["steps"] for r in wan}, {20})
            self.assertEqual(len({r["spec"]["prompt"] for r in wan}), 8)
            self.assertEqual({r["spec"]["guidance_scale"] for r in wan}, {guidance})
            self.assertEqual(q.normalize_config("wan", cfg), cfg)
        for changes in (dict(cpu_threads=4), dict(cpu_mask=list(range(4))), dict(device="cuda:1"),
                        dict(cpu_mask=[0] * 8), dict(gpu_uuid="1"), dict(sample_interval=0),
                        dict(seed=True), dict(prompt_distribution="random"), dict(arm_specs=[])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)

    def test_complete_fixture_qualifies_but_explicitly_is_not_performance_evidence(self):
        for kind, variants in (("sana", [dict(prompt_distribution="unique"), dict(prompt_distribution="repeated")]),
                               ("wan", [dict(guidance_scale=1.), dict(guidance_scale=5.)])):
            for variant in variants:
                cfg = config(kind, **variant)
                result = q.qualify(kind, cfg, arms(kind, cfg))
                self.assertTrue(result["passed"], result)
                self.assertFalse(result["performance_evidence"])

    def test_missing_capacity_cannot_be_replaced_by_a_passed_flag(self):
        mutations = [lambda r: r.update(nvml_samples=[]),
                     lambda r: r.update(nvml_samples=r["nvml_samples"][:1]),
                     lambda r: r["nvml_samples"][0].pop("memory_free_bytes"),
                     lambda r: r["nvml_samples"][0].update(memory_free_bytes=100),
                     lambda r: r["nvml_samples"][0].update(uuid="GPU-wrong"),
                     lambda r: r.update(nvml_errors=["query failed"])]
        cfg = config()
        for mutate in mutations:
            evidence = arms("sana", cfg)
            mutate(evidence[1]["result"]["requests"][0])
            self.assertFalse(q.qualify("sana", cfg, evidence)["passed"])

    def test_missing_or_changed_numerical_evidence_fails_closed(self):
        cfg = config("wan")
        mutations = [lambda r: r["cpu_embeddings"].pop("prompt"),
                     lambda r: r["cpu_embeddings"]["negative"].update(sha256="a" * 64),
                     lambda r: r["step_latents"].pop(3),
                     lambda r: r["step_latents"][3].update(sha256="b" * 64),
                     lambda r: r["step_latents"][3].update(finite=False),
                     lambda r: r["outputs"].pop("video"),
                     lambda r: r["outputs"]["latent"].update(sha256="c" * 64),
                     lambda r: r["outputs"]["video"].update(dtype="float16"),
                     lambda r: r.update(status="failed"),
                     lambda r: r["spec"].update(seed=99)]
        for mutate in mutations:
            evidence = arms("wan", cfg)
            mutate(evidence[1]["result"]["requests"][0])
            self.assertFalse(q.qualify("wan", cfg, evidence)["passed"])

    def test_warmup_and_request_count_are_also_required(self):
        cfg = config()
        for mutate in (lambda child: child["requests"].pop(), lambda child: child.pop("warmup"),
                       lambda child: child["warmup"].update(nvml_samples=[]),
                       lambda child: child["cache_initial"].update(entries=1),
                       lambda child: child.update(config_sha256="changed"),
                       lambda child: child["runtime"].update(matmul_tf32=True),
                       lambda child: child["cpu_affinity"]["physical_cores"][1].update(core=60)):
            evidence = arms("sana", cfg)
            mutate(evidence[1]["result"])
            self.assertFalse(q.qualify("sana", cfg, evidence)["passed"])

    def test_cache_expectations_unique_repeated_guided_and_bypass_are_strict(self):
        cfg = config()
        evidence = arms("sana", cfg)
        self.assertEqual([r["cache"]["outcome"] for r in evidence[1]["result"]["requests"]], ["miss"] * 4 + ["hit"] * 4)
        for key, value in (("outcome", "hit"), ("admitted", False), ("encode_called", False), ("timing_scope", "omitted")):
            changed = deepcopy(evidence)
            changed[2]["result"]["requests"][0]["cache"][key] = value
            self.assertFalse(q.qualify("sana", cfg, changed)["passed"])
        cfg = config("wan")
        evidence = arms("wan", cfg)
        self.assertEqual([r["cache"]["outcome"] for r in evidence[1]["result"]["requests"]],
                         ["miss", "hit", "miss", "hit", "hit", "hit", "hit", "hit"])
        cfg = config("wan", guidance_scale=1.)
        evidence = arms("wan", cfg)
        self.assertEqual({r["cache"]["outcome"] for r in evidence[1]["result"]["requests"]}, {"bypass"})
        evidence[1]["result"]["requests"][0]["cache"]["negative_encode_called"] = True
        self.assertFalse(q.qualify("wan", cfg, evidence)["passed"])

    def test_cache_provenance_storage_and_lru_are_independently_replayed(self):
        for kind in ("sana", "wan"):
            cfg = config(kind)
            mutations = [lambda r: r.update(schema="other"), lambda r: r.update(request_id="wrong"),
                         lambda r: r["after"].update(bytes=0), lambda r: r["after"]["lru"].clear(),
                         lambda r: r["before"].update(entries=1), lambda r: r.update(sequence=99),
                         lambda r: r["counters"].update(hits=99), lambda r: r.update(entry_bytes=1)]
            if kind == "sana":
                mutations += [lambda r: r.update(adapter_id="c" * 32), lambda r: r.update(prompt_sha256="d" * 64),
                              lambda r: r["entry_tensors"][0].update(accounted_bytes=1)]
            else:
                mutations += [lambda r: r.update(instance_id="c" * 32), lambda r: r.update(negative_prompt_sha256="d" * 64),
                              lambda r: r["recipe"].update(dtype="torch.float16"),
                              lambda r: r["entry_tensor"].update(storage_bytes=1)]
            for mutate in mutations:
                with self.subTest(kind=kind, mutate=mutate):
                    evidence = arms(kind, cfg)
                    mutate(evidence[1]["result"]["requests"][0]["cache"])
                    self.assertFalse(q.qualify(kind, cfg, evidence)["passed"])

    def test_receipt_replay_accepts_actual_sana_backend_cpu_standin_receipts(self):
        from ditserve.models.sana.backend import SanaBackend
        from test_sana_backend import FakeAdapter
        cfg = config()
        for arm in q.ARMS["sana"]:
            with self.subTest(arm=arm), patch("ditserve.models.sana.runtime.SanaAdapter", FakeAdapter):
                backend = SanaBackend(dict(model_path="/models/sana", device="cuda:0", options=q._options("sana", arm)))
                def execute(case):
                    request = dict(state_id=case["request_id"], spec=case["spec"])
                    backend.prepare(request)
                    receipt = backend.advance(request)["prompt_cache"]
                    backend.release(request)
                    return dict(case, cache=receipt)
                try:
                    warmup = execute(dict(request_id="warmup", spec=dict(prompt="warmup", seed=0)))
                    previous = q._check_receipt("sana", arm, warmup, cfg)
                    backend._clear_cache()
                    for index, case in enumerate(q.cases("sana", cfg)):
                        previous = q._check_receipt("sana", arm, execute(case), cfg, previous, reset=index == 0)
                finally:
                    backend.close()

    def test_receipt_replay_accepts_actual_wan_cache_cpu_standin_receipts(self):
        from ditserve.models.wan.prompt_cache import NegativePromptCache
        class Tensor:
            shape, dtype, device = (1, 512, 4096), "torch.float32", "cpu"
            def numel(self): return 512 * 4096
            def element_size(self): return 4
            def untyped_storage(self): return SimpleNamespace(nbytes=lambda: self.numel() * 4)
            def detach(self): return self
            def clone(self): return Tensor()
        for guidance in (1., 5.):
            cfg = config("wan", guidance_scale=guidance)
            cache = NegativePromptCache(entries=4, byte_limit=64 * 1024**2, recipe=q._wan_recipe(cfg))
            def execute(case):
                args = dict(prompt=case["spec"]["prompt"], negative_prompt=case["spec"]["negative_prompt"],
                            do_classifier_free_guidance=guidance > 1)
                def encode(**arguments):
                    return Tensor(), (arguments.get("negative_prompt_embeds", Tensor()) if guidance > 1 else None)
                _, _, receipt = cache.encode(encode, args, request_id=case["request_id"], validate=lambda *unused: None)
                return dict(case, cache=receipt)
            warmup = dict(request_id="warmup", spec=dict(q.cases("wan", cfg)[0]["spec"], request_id="warmup"))
            previous = q._check_receipt("wan", "enabled", execute(warmup), cfg)
            cache.clear()
            for index, case in enumerate(q.cases("wan", cfg)):
                previous = q._check_receipt("wan", "enabled", execute(case), cfg, previous, reset=index == 0)

    def test_parent_spawns_distinct_arms_and_refuses_to_overwrite(self):
        cfg = config()
        fixtures = {arm["id"]: arm["result"] for arm in arms("sana", cfg)}
        commands = []
        def child(command, **kwargs):
            commands.append(command)
            arm = command[command.index("--arm") + 1]
            out = Path(command[command.index("--out") + 1])
            out.mkdir()
            (out / "run.json").write_text(json.dumps(fixtures[arm]))
            self.assertEqual(kwargs["timeout"], cfg["arm_timeout_seconds"])
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory, patch.object(q.subprocess, "run", side_effect=child):
            out = Path(directory) / "qualification"
            result = q.run("sana", cfg, out)
            self.assertTrue(result["qualified"], result)
            self.assertEqual(len(commands), 3)
            self.assertEqual(len({c[c.index("--out") + 1] for c in commands}), 3)
            self.assertTrue(all(c[1:3] == ["-m", "ditserve.evaluation.workflow_encoding_qualification"] for c in commands))
            with self.assertRaises(FileExistsError):
                q.run("sana", cfg, out)
            self.assertEqual(len(commands), 3)

    def test_child_failure_or_timeout_never_qualifies(self):
        cfg = config()
        for error in (subprocess.TimeoutExpired("child", 1), RuntimeError("child crashed")):
            with tempfile.TemporaryDirectory() as directory, patch.object(q.subprocess, "run", side_effect=error):
                result = q.run("sana", cfg, Path(directory) / "qualification")
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["qualified"])
                self.assertFalse(result["qualification"]["passed"])
        evidence = arms("sana", cfg)
        self.assertFalse(q.qualify("sana", cfg, evidence[:2])["passed"])
        evidence[1]["returncode"] = 1
        self.assertFalse(q.qualify("sana", cfg, evidence)["passed"])

    def test_wan_execution_hashes_all_steps_and_releases_state(self):
        cfg = config("wan")
        case = q.cases("wan", cfg)[0]
        shape = [1, 16, 5, 60, 104]
        class Backend:
            states, tickets = {}, {}
            def encode_start(self, request):
                future = Future()
                future.set_result(SimpleNamespace(prompt_embeds="positive", negative_prompt_embeds="negative"))
                self.tickets = {"t": (request["spec"], future)}
                return dict(ticket="t")
            def prepare(self, request):
                self.states[request["state_id"]] = SimpleNamespace(step_index=0, latents=None)
                return {}
            def advance(self, request):
                state = self.states[request["state_id"]]
                state.step_index += 1
                state.latents = "latent%d" % state.step_index
            def finish(self, request):
                self.steps = self.states[request["state_id"]].step_index
                def output(label, dimensions):
                    value = fingerprint(label, dimensions)
                    value["array_sha256"] = value.pop("sha256")
                    return value
                return dict(output=output("video", [1, 17, 480, 832, 3]), latent=output("latent20", shape))
            def release(self, request):
                del self.states[request["state_id"]]
        class Sampler:
            def __init__(self, **kwargs):
                self.samples, self.errors = samples(), []
            def start(self):
                return self
            def stop(self):
                return self
        backend = Backend()
        with patch("ditserve.evaluation.workflow_metrics.NvmlSampler", Sampler), \
             patch.object(q.time, "perf_counter_ns", side_effect=[100, 200]), \
             patch.object(q, "_fingerprint", side_effect=lambda value: fingerprint(value, shape if value.startswith("latent") else [1, 512, 4096])):
            observed = q._execute_request("wan", backend, case, cfg)
        self.assertEqual(observed["status"], "completed")
        self.assertEqual(backend.steps, 20)
        self.assertEqual(len(observed["step_latents"]), 20)
        self.assertEqual(backend.states, {})
        self.assertTrue(observed["capacity"]["eligible"])
        q._numeric("wan", observed, case, cfg)


if __name__ == "__main__":
    unittest.main()
