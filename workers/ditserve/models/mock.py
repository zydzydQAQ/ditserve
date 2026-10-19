"""Deterministic CPU fixture, explicitly distinct from model measurements."""

import hashlib
import json
import time
from pathlib import Path

from .base import Backend, artifact, file_hash, positive_int, progress


class MockBackend(Backend):
    operations = Backend.operations + ("snapshot", "restore", "inventory", "update_prompt", "distributed",
                                      "migration_release", "migration_abort", "encode_start", "encode_poll",
                                      "decode_start", "decode_poll", "block_export", "block_import")

    def __init__(self, config):
        super().__init__(config)
        self.migrations = {}
        self.tickets, self.decodes = {}, {}
        # Like Wan's layout handle under SP groups: one request inside a solver step per worker.
        self.mid_step = None

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

    def decode_start(self, request):
        state = self.state(request)
        if state["step"] != state["steps"] or state["paused"]:
            raise ValueError("decode requires completed resident state")
        if self.decodes:
            raise ValueError("collect the previous decode first")
        delay = self.config.get("options", {}).get("decode_delay_ms", 0)
        if type(delay) not in (float, int) or not 0 <= delay <= 60000:
            raise ValueError("mock decode_delay_ms must be in [0, 60000]")
        ticket = "decode-%d" % request["id"]
        self.decodes[ticket] = (request["state_id"], time.monotonic() + delay / 1000)
        return dict(ticket=ticket, ready=False, measured=False)

    def decode_poll(self, request):
        if request["ticket"] not in self.decodes:
            raise ValueError("unknown decode ticket")
        key, ready_at = self.decodes[request["ticket"]]
        remaining = ready_at - time.monotonic()
        if remaining > 0:
            if not request.get("wait"):
                return dict(ticket=request["ticket"], ready=False, measured=False)
            time.sleep(remaining)
        result = self._finish(dict(request, state_id=key))
        del self.decodes[request["ticket"]]
        return dict(result, ticket=request["ticket"], ready=True)

    def _check_not_decoding(self, key):
        if any(owner == key for owner, ready_at in self.decodes.values()):
            raise ValueError("state is owned by an outstanding async decode")

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
        if request.get("previous_state_id"):
            previous = self.state(dict(state_id=request["previous_state_id"]))
            if previous["step"] != previous["steps"] or previous["paused"]:
                raise ValueError("previous state must be complete and resident")
            del self.states[request["previous_state_id"]]
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
        self.states[key] = dict(step=unit // blocks, steps=steps, unit=unit, blocks=blocks, value=value, residency="cpu",
                               paused=False, prompt_version=spec.get("prompt_version", 0), text=spec.get("prompt", "fixture output"))
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
        if state["paused"]:
            raise ValueError("restore the mock state before advancing")
        if state["step"] == state["steps"]:
            raise ValueError("request has no remaining steps")
        degree = positive_int(request.get("sp_degree", 1), "sp_degree")
        if request.get("group") is not None and len(request["group"]) != degree:
            raise ValueError("advance group size differs from sp_degree")
        grouped = "epoch" in request and state["blocks"] > 1
        if grouped and self.mid_step not in (None, request["state_id"]):
            raise ValueError("another request is inside a solver step on this worker")
        # Like Wan's _adopt: a request inside a step keeps its group and epoch.
        # A list, so the value survives the JSON round trip of exports and snapshots.
        placement = [list(request.get("group") or []), request.get("epoch")]
        if grouped and state["unit"] % state["blocks"] and state.get("placement") not in (None, placement):
            raise ValueError("SP group changes only at a solver-step boundary")
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
            if grouped:
                self.mid_step = request["state_id"] if state["unit"] % state["blocks"] else None
                state["placement"] = placement
            self._sleep("step_ms", executed / state["blocks"] / degree)
        return dict(state_id=request["state_id"], completed_steps=state["step"], executed_units=executed,
                    remaining_steps=state["steps"] - state["step"],
                    done=state["step"] == state["steps"], model_ms=0., measured=False,
                    frames=4, fps=16., duration_ms=250., prompt_version=state["prompt_version"],
                    progress=self._progress(state))

    def block_export(self, request):
        """Fixture transport: the request position as JSON, like Wan's step-boundary packet."""
        self._inject("block_export", request)
        key = request["state_id"]
        state = self.state(request)
        if state["unit"] % state["blocks"] or state["paused"]:
            raise ValueError("mock export requires a resident solver-step boundary")
        destination = Path(request["path"]).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True)
        released = bool(request.get("release", True))
        if released:
            del self.states[key]
        return dict(path=str(destination), sha256=file_hash(destination), bytes=destination.stat().st_size,
                    released=released, export_ms=0., write_ms=0., measured=False)

    def block_import(self, request):
        self._inject("block_import", request)
        key = self.new_id(request)
        path = Path(request["path"])
        if request.get("sha256") and file_hash(path) != request["sha256"]:
            raise ValueError("mock packet hash mismatch")
        self.states[key] = json.loads(path.read_text(encoding="utf-8"))
        return dict(state_id=key, progress=self._progress(self.states[key]), measured=False)

    def snapshot(self, request):
        self._inject("snapshot", request)
        state = self.state(request)
        self._check_not_decoding(request["state_id"])
        mode = request.get("mode", "hot")
        if mode not in ("keep", "hot", "cold"):
            raise ValueError("snapshot mode must be keep, hot, or cold")
        if state["paused"]:
            raise ValueError("state is already paused")
        if request.get("target_device"):
            migration_id = request["migration_id"]
            if migration_id in self.migrations:
                raise ValueError("duplicate migration_id")
            payload = dict(state=dict(state, paused=True), migration_id=migration_id, state_id=request["state_id"])
            record = artifact(json.dumps(payload, sort_keys=True), request["path"])
            self.migrations[migration_id] = ("source", request["state_id"])
            state["paused"] = True
            return dict(record, snapshot_id=migration_id, phase="paused", measured=False)
        state["paused"] = mode != "keep"
        return dict(state_id=request["state_id"], snapshot_id=request["state_id"],
                    mode=mode, model_ms=0., measured=False)

    def restore(self, request):
        self._inject("restore", request)
        if request.get("path"):
            if file_hash(request["path"]) != request.get("sha256"):
                raise ValueError("mock migration hash mismatch")
            payload = json.loads(Path(request["path"]).read_text())
            if payload["migration_id"] in self.migrations:
                raise ValueError("duplicate migration_id")
            key = self.new_id(request)
            self.states[key] = payload["state"]
            self.states[key]["paused"] = True
            self.migrations[payload["migration_id"]] = ("target", key)
            return dict(state_id=key, phase="prepared", runnable=False,
                        acknowledgement=dict(migration_id=payload["migration_id"], phase="prepared"))
        if request.get("receipt"):
            migration_id = request["migration_id"]
            if request["receipt"] != dict(migration_id=migration_id, phase="released"):
                raise ValueError("mock migration requires release receipt")
            role, key = self.migrations[migration_id]
            if role != "target":
                raise ValueError("only the prepared target can commit a migration")
            del self.migrations[migration_id]
            request = dict(request, state_id=key)
        elif any(key == request.get("state_id", request.get("snapshot_id")) for role, key in self.migrations.values()):
            raise ValueError("migration needs commit or abort, not a local restore")
        self.state(request)["paused"] = False
        state = self.state(request)
        return dict(state_id=request.get("state_id", request.get("snapshot_id")), model_ms=0.,
                    measured=False, progress=self._progress(state))

    def inventory(self, request):
        return dict(self.state(request), measured=False)

    def finish(self, request):
        self._check_not_decoding(request["state_id"])
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
        if state["step"] != state["steps"] or state["paused"]:
            raise ValueError("finish requires a complete resident state")
        if not request.get("_chunked"):
            self._sleep("finish_ms")
        output = self.persist(json.dumps({"value": state["value"]}, sort_keys=True), request, ".json")
        output["text"] = state["text"]
        return dict(state_id=request["state_id"], output=output,
                    done=True, model_ms=0., measured=False)

    def release(self, request):
        self.state(request)
        self._check_not_decoding(request["state_id"])
        if any(key == request["state_id"] for role, key in self.migrations.values()):
            raise ValueError("migration needs commit or abort before release")
        del self.states[request["state_id"]]
        return {"released": request["state_id"]}

    def update_prompt(self, request):
        state = self.state(request)
        version = request["prompt_version"]
        if type(version) is not int or version <= state["prompt_version"]:
            raise ValueError("prompt_version must increase")
        state["prompt_version"] = version
        state["text"] = request["prompt"]
        return dict(prompt_version=version, measured=False)

    def distributed(self, request):
        action = request.get("action", "init")
        if action == "prepare":
            return self.prepare(request)
        if action == "step":
            return self.advance(request)
        if action not in ("init", "commit"):
            raise ValueError("unknown mock distributed action")
        return dict(status="fixture_only", initialized=True, gpu_validated=False,
                    rank=request.get("rank", 0), world_size=request.get("world_size", 4))

    def migration_release(self, request):
        migration_id = request["migration_id"]
        if request["acknowledgement"] != dict(migration_id=migration_id, phase="prepared"):
            raise ValueError("mock target acknowledgement differs")
        role, key = self.migrations[migration_id]
        if role != "source":
            raise ValueError("only the paused source can release a migration")
        del self.migrations[migration_id]
        self.release(dict(state_id=key))
        return dict(receipt=dict(migration_id=migration_id, phase="released"), measured=False)

    def migration_abort(self, request):
        role, key = self.migrations.pop(request["migration_id"])
        if role == "source":
            self.states[key]["paused"] = False
        else:
            del self.states[key]
        return dict(aborted=request["migration_id"], measured=False)

    def close(self):
        self.decodes.clear()
        for migration_id in list(self.migrations):
            self.migration_abort(dict(migration_id=migration_id))
        super().close()
