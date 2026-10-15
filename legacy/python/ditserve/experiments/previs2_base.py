"""Shared run-protocol base for the PreViS2 experiment worker.

Inlined from the retired first-generation experiment so run02 has no
dependency on run01 modules. Only the members the run02 Experiment actually
inherits are kept: bookkeeping, the budget guard, spec/path helpers and the
correctness gate (which dispatches to subclass implementations of
generate/save_array/_cold_policy_interleave_correctness).
"""

import json
from pathlib import Path
import time

from ..util import utc_now


class BudgetStop(RuntimeError):
    pass


class Experiment:
    def __init__(self, job):
        self.job, self.config = job, job["config"]
        self.root, self.suite = Path(job["output"]), Path(job["suite_root"])
        self.started = time.monotonic()
        self.deadline = self.started + job["allowed_seconds"]
        self.records = []
        self.worker = None
        self.device = job["device"]

    def check_budget(self, estimate=5):
        if time.monotonic() + estimate >= self.deadline:
            raise BudgetStop("stage budget reached; completed measurements retained")

    def record(self, value):
        value = dict(value, device=self.device, recorded_at=utc_now())
        with (self.root / "records.jsonl").open("a") as file:
            file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
        self.records.append(value)
        print(json.dumps({k: value.get(k) for k in ("type", "case_id", "status", "elapsed_ms")}, ensure_ascii=False), flush=True)
        return value

    def spec(self, case_id, kind="long", prompt=None, seed=0, cache=False, threshold=.08, max_continuous=3):
        prompt = prompt or self.config["prompts"][0]
        return dict(self.config["shapes"][kind], request_id=case_id, prompt=prompt["text"],
                    negative_prompt="", seed=seed, cache=cache, threshold=threshold,
                    max_continuous=max_continuous, guidance_scale=5.0)

    @staticmethod
    def paths(run):
        events = run.get("path_history", run.get("events", []))
        return [(e.get("step_index"), e.get("path"),
                 [(c.get("cond_reused"), c.get("uncond_reused"), c.get("executed_steps"),
                   c.get("transformer_executed_steps")) for c in e.get("cache_contexts", [])])
                for e in events if e.get("kind") == "step" or "step_index" in e]

    def correctness(self):
        import numpy as np
        import torch
        for cache in (False, True):
            label = "cache" if cache else "none"
            spec = self.spec("correct-" + label, cache=cache)
            self.check_budget(60)
            reference = self.worker.reference(spec)
            baseline = self.generate(spec)
            output_equal = bool(np.array_equal(reference["output"], baseline["output"]))
            latent_equal = bool(torch.equal(reference["latents"], baseline["latents"]))
            path_equal = self.paths(reference) == self.paths(baseline)
            self.record(dict(type="correctness", case_id="official-" + label, status="completed" if output_equal and latent_equal and path_equal else "failed",
                             cache=cache, official_pipeline=True, output_exact_match=output_equal, latent_exact_match=latent_equal,
                             path_exact_match=path_equal, instrumented=True,
                             elapsed_ms=baseline["elapsed_ms"], reference_elapsed_ms=reference["elapsed_ms"],
                             output=self.save_array("official-" + label, reference["output"]),
                             latent_reference=self.save_array("official-" + label, reference["latents"].numpy(), "latent"),
                             latent_candidate=self.save_array("stepwise-" + label, baseline["latents"].numpy(), "latent"),
                             candidate=self.save_array("stepwise-" + label, baseline["output"]), events=baseline["events"]))
            if not (output_equal and latent_equal and path_equal):
                raise RuntimeError("stepwise worker differs from official pipeline: " + label)
            for fraction, mode, interleave in ((.25, "keep", False), (.5, "hot", False), (.75, "hot", True)):
                case_id = "%s-%s-%s-%d" % (label, mode, "interleave" if interleave else "pause", fraction * 100)
                candidate = self.generate(dict(spec, request_id=case_id), pause=int(spec["steps"] * fraction), mode=mode, interleave=interleave)
                output_equal = bool(np.array_equal(baseline["output"], candidate["output"]))
                latent_equal = bool(torch.equal(baseline["latents"], candidate["latents"]))
                paths_equal = self.paths(baseline) == self.paths(candidate)
                snapshot = candidate["switches"][0]
                released = snapshot["memory_before"]["allocated_bytes"] - snapshot["memory_after"]["allocated_bytes"]
                cuda_bytes = snapshot.get("state_bytes", {}).get("cuda_bytes", 0)
                release_verified = mode == "keep" or (cuda_bytes > 0 and released >= .95 * cuda_bytes)
                self.record(dict(type="correctness", case_id=case_id,
                    status="completed" if output_equal and latent_equal and paths_equal and release_verified else "failed",
                    cache=cache, mode=mode, fraction=fraction, interleave=interleave,
                    output_exact_match=output_equal, latent_exact_match=latent_equal, path_exact_match=paths_equal,
                    released_allocated_bytes=released, state_cuda_bytes=cuda_bytes, memory_release_verified=release_verified,
                    elapsed_ms=candidate["elapsed_ms"], switches=candidate["switches"], events=candidate["events"],
                    latent=self.save_array(case_id, candidate["latents"].numpy(), "latent"),
                    output=self.save_array(case_id, candidate["output"])))
                if not (output_equal and latent_equal and paths_equal and release_verified):
                    raise RuntimeError("restoration correctness failed: " + case_id)

        self._cold_policy_interleave_correctness()
