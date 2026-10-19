"""Mistral and Sana request ownership with optional Sana pipeline candidates."""

from ..base import Backend, artifact, positive_int, progress


class SessionBackend(Backend):
    operations = Backend.operations + ("snapshot", "restore", "inventory", "reference", "optimize", "generate_batch")

    def __init__(self, config):
        super().__init__(config)
        self.kind = config["backend"]
        self.pipeline_mode = self.kind == "sana" and config.get("options", {}).get("execution") == "pipeline"
        device, path = config.get("device", "cuda:0"), config["model_path"]
        if self.pipeline_mode:
            from ..sana.graph import NextSanaAdapter
            from ..sana.compile import OptimizedSanaAdapter
            candidate = config.get("options", {}).get("candidate", "eager")
            self.runtime = (OptimizedSanaAdapter(path, device) if candidate == "transformer_compile"
                            else NextSanaAdapter(path, device))
            if candidate == "transformer_compile":
                self.runtime.compile_transformer()
                self.runtime.set_backend("compiled")
            else:
                self.runtime.set_candidate(candidate)
        else:
            from .runtime import MistralRuntime, SanaRuntime
            cls = MistralRuntime if self.kind == "mistral" else SanaRuntime
            self.runtime = cls(path, device)

    def _progress(self, state):
        if self.pipeline_mode:
            return progress(int(state["done"]), 1)
        if self.kind == "mistral":
            return progress(len(state.generated_ids), state.max_new_tokens, done=state.done)
        return progress(state.step_index, state.num_steps)

    def prepare(self, request):
        key = self.new_id(request)
        spec = dict(self.config.get("spec", {}), **request.get("spec", {}))
        if self.pipeline_mode:
            state = dict(spec=spec, done=False, output=None)
        elif self.kind == "mistral":
            previous = request.get("previous_state_id")
            if previous and previous not in self.states:
                raise ValueError("unknown previous_state_id")
            state = self.runtime.prepare(spec.get("messages", [{"role": "user", "content": spec.get("prompt", "")}]),
                max_new_tokens=positive_int(spec.get("max_new_tokens", 64)),
                previous=self.states.get(previous), request_id=key)
            if previous:
                del self.states[previous]
        else:
            state = self.runtime.prepare(spec["prompt"], seed=spec.get("seed", 0), request_id=key)
        self.states[key] = state
        return dict(state_id=key, progress=self._progress(state), model_ms=getattr(state, "prepare_ms", 0.))

    def advance(self, request):
        state = self.state(request)
        if self._progress(state)["done"]:
            raise ValueError("request has no remaining model work")
        events = []
        if self.pipeline_mode:
            state["output"], service = self.runtime.generate(state["spec"]["prompt"], seed=state["spec"].get("seed", 0))
            state["done"] = True
            events.append(service)
        else:
            for unused in range(positive_int(request.get("steps", 1))):
                if state.done:
                    break
                events.append(self.runtime.advance(state))
        return dict(state_id=request["state_id"], progress=self._progress(state), events=events,
                    model_ms=sum(row.get("ms", 0.) for row in events))

    def finish(self, request):
        state = self.state(request)
        if not self._progress(state)["done"]:
            raise ValueError("finish requires completed model work")
        value = state["output"] if self.pipeline_mode else self.runtime.finish(state)
        suffix = ".txt" if self.kind == "mistral" else ".npy"
        result = dict(state_id=request["state_id"], progress=self._progress(state),
                      output=self.persist(value, request, suffix),
                      model_ms=getattr(state, "decode_ms", 0.))
        self.release_cache()
        return result

    def snapshot(self, request):
        if self.pipeline_mode:
            raise ValueError("Sana full-pipeline candidates do not expose preemptible state")
        mode = request.get("mode", "hot")
        if mode not in ("keep", "hot"):
            raise ValueError("exact session state supports keep/hot only; cold KV discard is unsupported")
        state = self.state(request)
        result = self.runtime.inventory(state) if mode == "keep" else self.runtime.offload_to_cpu(state)
        return dict(result, snapshot_id=request["state_id"], state_id=request["state_id"], mode=mode)

    def restore(self, request):
        if self.pipeline_mode or request.get("path"):
            raise ValueError("session restore requires an existing in-process exact state")
        state = self.state(request)
        return dict(self.runtime.restore(state), progress=self._progress(state),
                    state_id=request.get("state_id", request.get("snapshot_id")))

    def inventory(self, request):
        if self.pipeline_mode:
            return dict(memory=self.runtime.memory(), preemptible=False)
        return self.runtime.inventory(self.state(request))

    def reference(self, request):
        if self.pipeline_mode:
            raise ValueError("use an independently configured eager worker for pipeline references")
        spec = request.get("spec", {})
        if self.kind == "mistral":
            result = self.runtime.reference(spec.get("messages", [{"role": "user", "content": spec.get("prompt", "")}]),
                max_new_tokens=spec.get("max_new_tokens", 64))
            result.pop("cache", None)
            return dict(result, output=self.persist(result["text"], request, ".txt"))
        result = self.runtime.reference(spec["prompt"], seed=spec.get("seed", 0))
        result.pop("step_latents", None)
        output = result.pop("output")
        return dict(result, output=self.persist(output, request))

    def optimize(self, request):
        if not self.pipeline_mode or self.states:
            raise ValueError("Sana optimization requires an idle pipeline-mode worker")
        if not hasattr(self.runtime, "set_candidate"):
            raise ValueError("transformer_compile is fixed for this worker; launch a new worker for another candidate")
        return self.runtime.set_candidate(request["candidate"])

    def generate_batch(self, request):
        if not self.pipeline_mode:
            raise ValueError("batch execution requires Sana pipeline mode")
        output, service = self.runtime.generate_batch(request["prompts"], seeds=request["seeds"])
        return dict(output=self.persist(output, request), service=service, qualified=False)

    def release(self, request):
        state = self.state(request)
        if not self.pipeline_mode:
            self.runtime.release(state)
        return super().release(request)

    def close(self):
        super().close()
        self.runtime.close()
