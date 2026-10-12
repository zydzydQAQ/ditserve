"""Execution adapters. Importing this module never imports torch or touches CUDA."""

import copy
import hashlib
import math
import random
import time
from types import SimpleNamespace


WAN_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
CORE_POLICIES = ("cache_only", "batch_only", "naive", "edf", "path_aware", "state_aware")


class VirtualClock:
    virtual = True

    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def advance(self, milliseconds):
        self.value += float(milliseconds)

    def wait_until(self, milliseconds):
        self.value = max(self.value, milliseconds)


class RealClock:
    virtual = False

    def __init__(self):
        self.origin = time.monotonic()

    def now(self):
        return (time.monotonic() - self.origin) * 1000.0

    def wait_until(self, milliseconds):
        time.sleep(max(0.0, min(.02, (milliseconds - self.now()) / 1000)))


def request_spec(request):
    return dict(request_id=request["request_id"], prompt=request["prompt"],
                negative_prompt=request.get("negative_prompt", ""), seed=request["seed"],
                height=request["height"], width=request["width"], frames=request["frames"],
                steps=request["steps"], guidance_scale=request.get("guidance_scale", 5.0),
                cache=request.get("cache_enabled", False),
                threshold=request.get("cache_threshold", .08),
                max_continuous=request.get("max_continuous", 3))


class FixtureBackend:
    """Deterministic numerical fixture with an explicitly synthetic virtual clock.

    The output has a small fixed spatial extent: it is not a generated video or
    a Wan cost approximation. Independent RNG and solver/cache state make
    pause/interleaving parity tests meaningful without importing GPU libraries.
    """

    name = "cpu-test"
    measurement_kind = "synthetic_virtual_fixture"
    virtual = True

    def __init__(self, clock=None, capacity_mb=math.inf, **unused):
        self.clock = clock or VirtualClock()
        self.capacity_mb = float(capacity_mb)
        self.states = {}

    @staticmethod
    def costs(request):
        return dict(prepare_ms=2.0, step_ms=[3.0] * request["steps"], decode_ms=2.0,
                    peak_increment_mb=2.0, state_mb=1.0,
                    service_ms=4.0 + 3.0 * request["steps"])

    def prepare(self, request):
        import numpy as np
        rng = random.Random(request["seed"])
        digest = hashlib.sha256(request["prompt"].encode("utf-8")).digest()
        initial = [rng.random() + digest[i] / 2550 for i in range(12)]
        state = SimpleNamespace(spec=request_spec(request), request=copy.deepcopy(request),
            latents=np.array(initial, dtype=np.float64), step_index=0, previous=0.0,
            rng=rng, cache_value=None, events=[], residency="gpu", finished=False)
        self.states[request["request_id"]] = state
        self.clock.advance(2)
        state.events.append(dict(kind="prepare", ms=2.0, cpu_prompt_encode_ms=2.0))
        return state

    def step(self, state):
        reuse = state.spec["cache"] and state.step_index % 3 == 2
        if not reuse:
            state.cache_value = sum(state.latents) / len(state.latents)
        noise = state.rng.random() / 100
        state.latents = .91 * state.latents + .03 * state.cache_value + .02 * state.previous + noise
        state.previous = float(sum(state.latents))
        state.step_index += 1
        self.clock.advance(3)
        event = dict(kind="step", ms=3.0, step_index=state.step_index - 1,
                     path="reuse" if reuse else "compute")
        state.events.append(event)
        return event

    def finish(self, state):
        import numpy as np
        if state.step_index != state.spec["steps"]:
            raise ValueError("decode before all steps")
        self.clock.advance(2)
        state.finished = True
        state.events.append(dict(kind="decode", ms=2.0))
        pixels = np.clip(state.latents.reshape(1, 1, 2, 2, 3), 0, 1).astype(np.float32)
        return np.repeat(pixels, min(state.spec["frames"], 5), axis=1)

    def latent(self, state):
        return state.latents.copy()

    def snapshot(self, state):
        if state.residency != "gpu":
            raise ValueError("snapshot requires resident state")
        # Copies the complete numerical state, including independent RNG.
        state.latents = state.latents.copy()
        state.rng.setstate(copy.deepcopy(state.rng.getstate()))
        state.residency = "cpu"
        self.clock.advance(1)
        event = dict(kind="snapshot", mode="hot", ms=1.0)
        state.events.append(event)
        return event

    def restore(self, state):
        state.residency = "gpu"
        self.clock.advance(1)
        event = dict(kind="restore", mode="hot", ms=1.0)
        state.events.append(event)
        return event

    def available_mb(self):
        return self.capacity_mb - sum(s.residency == "gpu" for s in self.states.values())

    def state_mb(self, state):
        return 1.0 if state.residency == "gpu" else 0.0

    def memory(self):
        return dict(scope="synthetic_fixture", resident_state_mb=sum(s.residency == "gpu" for s in self.states.values()))

    def execution_identity(self):
        return dict(backend=self.name, gpu=None, clock="virtual",
                    precision=dict(numerical_state="float64", output="float32"),
                    attention="not_applicable", scope="CPU numerical fixture, not model execution")

    def begin_measurement(self):
        pass

    def peak_increment_mb(self):
        return 2.0

    def release(self, state):
        self.states.pop(state.spec["request_id"], None)
        state.residency = "released"

    def smoke(self, request, budget=None):
        return dict(status="passed", scope="numerical_fixture_only_not_gpu")

    def close(self):
        self.states.clear()


class WanBackend:
    name = "wan"
    measurement_kind = "measured_gpu"
    virtual = False

    def __init__(self, device="cuda:0", model_path=None, clock=None, **unused):
        from ..gpu.worker import WanStepWorker
        self.clock = clock or RealClock()
        self.worker = WanStepWorker(device=device, model_path=model_path)
        self.measurement_baseline = 0

    def prepare(self, request):
        if request.get("model") != WAN_MODEL or request.get("revision") != WAN_REVISION:
            raise ValueError("Wan adapter supports only the pinned Wan2.1 1.3B snapshot")
        return self.worker.prepare(request_spec(request))

    def step(self, state):
        return self.worker.step(state)

    def finish(self, state):
        return self.worker.finish(state)

    def latent(self, state):
        return state.latents.detach().cpu().numpy().copy()

    def snapshot(self, state):
        return self.worker.snapshot(state, "hot")

    def restore(self, state):
        return self.worker.restore(state)

    def available_mb(self):
        m = self.worker._memory()
        margin = max(512 * 1024 ** 2, .05 * m["device_total_bytes"])
        return max(0, m["driver_free_bytes"] + m["reserved_bytes"] - m["allocated_bytes"] - margin) / 1024 ** 2

    def state_mb(self, state):
        return self.worker._inventory(state)["cuda_bytes"] / 1024 ** 2

    def memory(self):
        return self.worker._memory()

    def execution_identity(self):
        """Inspect the already selected device, never infer a GPU from its rank."""
        import subprocess
        import uuid

        torch = self.worker.torch
        properties = torch.cuda.get_device_properties(self.worker.device)
        raw_uuid = str(getattr(properties, "uuid", ""))
        raw_uuid = raw_uuid[4:] if raw_uuid.startswith("GPU-") else raw_uuid
        try:
            physical_uuid = "GPU-" + str(uuid.UUID(raw_uuid))
        except (ValueError, AttributeError) as error:
            raise RuntimeError("selected CUDA device has no usable physical UUID") from error
        driver, driver_error = None, None
        try:
            query = subprocess.run(["nvidia-smi", "--id=" + physical_uuid,
                "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
            lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
            if query.returncode == 0 and len(lines) == 1:
                driver = lines[0]
            else:
                driver_error = "nvidia-smi driver query unavailable"
        except (OSError, subprocess.TimeoutExpired) as error:
            driver_error = type(error).__name__
        return dict(backend=self.name, logical_device=str(self.worker.device),
            gpu=dict(uuid=physical_uuid, name=properties.name,
                     total_memory_bytes=int(properties.total_memory),
                     compute_capability=[int(properties.major), int(properties.minor)]),
            driver_version=driver, driver_query_error=driver_error,
            torch_version=str(torch.__version__), cuda_runtime_version=torch.version.cuda,
            precision=dict(dit="float16", vae="float32", text_encoder="float32", latents="float32"),
            text_encoder_device="cpu", attention=dict(processor="native", sdpa_backend="MATH"),
            scheduler="UniPCMultistepScheduler", model_id=WAN_MODEL, revision=WAN_REVISION)

    def begin_measurement(self):
        self.worker._sync()
        self.measurement_baseline = self.worker._memory()["allocated_bytes"]
        self.worker.torch.cuda.reset_peak_memory_stats(self.worker.device)

    def peak_increment_mb(self):
        value = self.worker.torch.cuda.max_memory_allocated(self.worker.device)
        return max(0, value - self.measurement_baseline) / 1024 ** 2

    def release(self, state):
        self.worker.release(state)

    def smoke(self, request, budget=None):
        import numpy as np
        check = budget.check if budget else lambda *args: None
        # Preserve cache and sampler semantics, with an explicitly small smoke shape.
        smoke = dict(request, request_id="smoke-official", height=256, width=256, frames=9,
                     steps=max(10, min(request["steps"], 12)))
        check()
        baseline = self.worker.reference(request_spec(smoke))
        check()
        state = self.prepare(dict(smoke, request_id="smoke-restored"))
        try:
            while state.step_index < state.spec["steps"]:
                check()
                self.step(state)
                if state.step_index == state.spec["steps"] // 2:
                    self.snapshot(state)
                    self.restore(state)
            latent = self.latent(state)
            output = self.finish(state)
            actual_path = [e["path"] for e in state.events if e["kind"] == "step"]
            expected_path = [e["path"] for e in baseline["path_history"]]
            exact = np.array_equal(output, baseline["output"])
            latent_exact = np.array_equal(latent, baseline["latents"].numpy())
            output_close = bool(np.allclose(output, baseline["output"], rtol=1e-5, atol=1e-6))
            latent_close = bool(np.allclose(latent, baseline["latents"].numpy(), rtol=1e-5, atol=1e-6))
            passed = output_close and latent_close and actual_path == expected_path
            result = dict(status="passed" if passed else "failed", output_exact=exact,
                          latent_exact=latent_exact, path_exact=actual_path == expected_path,
                          output_close=output_close, latent_close=latent_close, rtol=1e-5, atol=1e-6,
                          scope="official_loop_vs_stepwise_full_hot_restore", generation=smoke)
            if not passed:
                raise RuntimeError("Wan smoke failed: " + str(result))
            return result
        finally:
            self.release(state)

    def close(self):
        for state in list(self.worker._states.values()):
            self.worker.release(state)


def canonical_backend(name):
    return {"fixture": "cpu-test", "cpp": "core"}.get(name, name)


def make_backend(name, **kwargs):
    name = canonical_backend(name)
    if name == "cpu-test":
        return FixtureBackend(**kwargs)
    if name == "wan":
        return WanBackend(**kwargs)
    raise ValueError("No step adapter for backend " + str(name))


def core_policy(name):
    policy = name[5:] if name.startswith("core:") else name
    if policy not in CORE_POLICIES:
        raise ValueError("core backend requires explicit core policy: " + ", ".join(CORE_POLICIES))
    return policy


def run_core(requests, policy, config=None):
    from ..core import CoreBridge
    config = config or {}
    workload = [{key: r[key] for key in ("request_id", "arrival_ms", "deadline_ms", "model",
                "height", "width", "frames", "steps", "seed", "cache_enabled")} for r in requests]
    for spec, request in zip(workload, requests):
        spec["cache_threshold"] = request.get("cache_threshold", .08)
    return CoreBridge().call("simulate", dict(schema_version=2, workload=workload,
        cost_model=config.get("cost_model", {}), config=dict(policy=core_policy(policy),
        num_gpus=config.get("num_gpus", 1), max_batch=config.get("max_batch", 1),
        memory_mb=config.get("memory_mb", 1024), host_memory_mb=config.get("host_memory_mb", 65536),
        verify_outputs=True)))
