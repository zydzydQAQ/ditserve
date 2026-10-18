"""Deterministic CPU fixture, explicitly distinct from model measurements."""

import hashlib
import json
import time

from .base import Backend, positive_int, progress


class MockBackend(Backend):
    operations = Backend.operations + ("distributed", "encode_start", "encode_poll")

    def __init__(self, config):
        super().__init__(config)
        self.tickets = {}

    def encode_start(self, request):
        ticket = "encode-%d" % request["id"]
        spec = dict(request.get("spec", {}))
        self.tickets[ticket] = (spec, time.monotonic() + self.config.get("options", {}).get("encode_delay_ms", 0) / 1000)
        return dict(ticket=ticket, ready=False, measured=False)

    def encode_poll(self, request):
        if request["ticket"] not in self.tickets:
            raise ValueError("unknown encode ticket")
        spec, ready_at = self.tickets[request["ticket"]]
        return dict(ticket=request["ticket"], ready=time.monotonic() >= ready_at, measured=False)

    def _inject(self, op, request):
        options = dict(self.config.get("options", {}))
        options.update(request.get("inject", {}))
        delay = options.get("delay_ms", 0)
        if type(delay) not in (float, int) or not 0 <= delay <= 60000:
            raise ValueError("mock delay_ms must be in [0, 60000]")
        if delay:
            time.sleep(delay / 1000)
        if options.get("print"):
            print(options["print"])
        if op == options.get("fail_on") or op in options.get("fail_ops", []):
            raise RuntimeError("injected mock " + op + " failure")

    def prepare(self, request):
        self._inject("prepare", request)
        key = self.new_id(request)
        spec = dict(request.get("spec", {}))
        if request.get("encode_ticket"):
            ticket = request["encode_ticket"]
            if ticket not in self.tickets:
                raise ValueError("unknown encode ticket")
            encoded, ready_at = self.tickets[ticket]
            if encoded != spec or time.monotonic() < ready_at:
                raise ValueError("encode ticket is not ready or belongs to another spec")
            del self.tickets[ticket]
        steps = positive_int(spec.get("steps", spec.get("blocks", 2)))
        blocks = positive_int(spec.get("blocks_per_step", 1), "blocks_per_step")
        seed = spec.get("seed", 0)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        value = hashlib.sha256(json.dumps({"prompt": spec.get("prompt", ""), "seed": seed},
                                         sort_keys=True).encode()).hexdigest()
        # Cluster regrouping resumes a request on new ranks from its position.
        resume = request.get("resume", {})
        unit = int(resume.get("step", 0)) * blocks + int(resume.get("block", 0))
        if not 0 <= unit < steps * blocks:
            raise ValueError("resume position is outside the request")
        self.states[key] = dict(step=unit // blocks, steps=steps, unit=unit, blocks=blocks, value=value)
        self._sleep("prepare_ms")
        return dict(state_id=key, done=False, completed_steps=0, remaining_steps=steps,
                    model_ms=0., measured=False, progress=self._progress(self.states[key]))

    def _sleep(self, key, scale=1.):
        """Optional fixture timing from options.costs; never a measurement."""
        value = self.config.get("options", {}).get("costs", {}).get(key, 0) * scale
        if type(value) not in (float, int) or not 0 <= value <= 60000:
            raise ValueError("mock costs must be in [0, 60000] ms")
        if value:
            time.sleep(value / 1000)

    @staticmethod
    def _progress(state):
        blocks = state.get("blocks", 1)
        if blocks == 1:
            return progress(state["step"], state["steps"])
        return dict(progress(state["step"], state["steps"]), block=state["unit"] % blocks,
                    transfer_ready=state["unit"] % blocks == 0)

    def advance(self, request):
        self._inject("advance", request)
        state = self.state(request)
        if state["step"] == state["steps"]:
            raise ValueError("request has no remaining steps")
        degree = positive_int(request.get("sp_degree", 1), "sp_degree")
        if state["blocks"] == 1:
            count = positive_int(request.get("steps", request.get("blocks", 1)))
            executed = min(count, state["steps"] - state["step"])
            for unused in range(executed):
                state["value"] = hashlib.sha256(state["value"].encode()).hexdigest()
                state["step"] += 1
            state["unit"] = state["step"]
            self._sleep("step_ms", executed / degree)
        else:
            # Block quanta stop at the next solver-step boundary, like Wan.
            boundary = state["blocks"] - state["unit"] % state["blocks"]
            executed = boundary if request.get("quantum") == "step" else min(positive_int(request.get("blocks", 1)), boundary)
            state["unit"] += executed
            if state["unit"] % state["blocks"] == 0:
                state["value"] = hashlib.sha256(state["value"].encode()).hexdigest()
            state["step"] = state["unit"] // state["blocks"]
            self._sleep("step_ms", executed / state["blocks"] / degree)
        return dict(state_id=request["state_id"], completed_steps=state["step"], executed_units=executed,
                    remaining_steps=state["steps"] - state["step"],
                    done=state["step"] == state["steps"], model_ms=0., measured=False,
                    progress=self._progress(state))

    def finish(self, request):
        if request.get("decode_frames") is not None:
            state = self.state(request)
            chunks = int(self.config.get("options", {}).get("decode_chunks", 1))
            frames = positive_int(request["decode_frames"], "decode_frames")
            state["decoded"] = min(chunks, state.get("decoded", 0) + frames)
            self._sleep("finish_ms", frames / chunks)
            info = dict(frames_done=state["decoded"], frames=chunks, done=state["decoded"] == chunks)
            if not info["done"]:
                return dict(state_id=request["state_id"], decode=info, model_ms=0., measured=False)
            return dict(self._finish(dict(request, _chunked=True)), decode=info)
        return self._finish(request)

    def _finish(self, request):
        self._inject("finish", request)
        state = self.state(request)
        if state["step"] != state["steps"]:
            raise ValueError("finish requires a complete state")
        if not request.get("_chunked"):
            self._sleep("finish_ms")
        output = self.persist(json.dumps({"value": state["value"]}, sort_keys=True), request, ".json")
        return dict(state_id=request["state_id"], output=output,
                    done=True, model_ms=0., measured=False)

    def distributed(self, request):
        action = request.get("action", "init")
        if action == "prepare":
            return self.prepare(request)
        if action == "step":
            return self.advance(request)
        if action not in ("init", "commit"):
            raise ValueError("unknown mock distributed action")
        return dict(status="fixture_only", initialized=True, gpu_validated=False,
                    rank=request.get("rank", 0), world_size=request.get("world_size", 2))
