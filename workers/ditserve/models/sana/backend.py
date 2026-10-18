"""SANA-Sprint 0.6B image requests: one eager full-pipeline call per request."""

from ..base import Backend, progress


class SanaBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        if config.get("options"):
            raise ValueError("the Sana backend runs the eager official pipeline and takes no options")
        from .runtime import SanaAdapter
        self.runtime = SanaAdapter(config["model_path"], config.get("device", "cuda:0"))

    def prepare(self, request):
        key = self.new_id(request)
        spec = dict(self.config.get("spec", {}), **request.get("spec", {}))
        if not isinstance(spec.get("prompt"), str) or type(spec.get("seed", 0)) is not int:
            raise ValueError("Sana requests need a string prompt and an integer seed")
        self.states[key] = dict(spec=spec, output=None)
        return dict(state_id=key, progress=progress(0, 1), model_ms=0.)

    def advance(self, request):
        state = self.state(request)
        if state["output"] is not None:
            raise ValueError("request has no remaining model work")
        state["output"], service = self.runtime.generate(state["spec"]["prompt"], seed=state["spec"].get("seed", 0))
        return dict(state_id=request["state_id"], progress=progress(1, 1), events=[service], model_ms=service["ms"])

    def finish(self, request):
        state = self.state(request)
        if state["output"] is None:
            raise ValueError("finish requires completed model work")
        return dict(state_id=request["state_id"], progress=progress(1, 1),
                    output=self.persist(state["output"], request), model_ms=0.)

    def close(self):
        super().close()
        self.runtime.close()
