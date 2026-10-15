"""Frozen CPU text embeddings for mechanism measurements, never online E2E."""

import hashlib

from ..gpu.worker import EncodedPrompt, normalize_spec
from .common import digest


class FrozenPromptBank:
    def __init__(self, worker):
        self.worker = worker
        self.owner = worker._owner
        self.entries = {}

    def _hash(self, tensor):
        if tensor is None:
            return None
        if tensor.device.type != "cpu" or tensor.dtype != self.worker.torch.float32:
            raise ValueError("frozen text embeddings must be CPU float32")
        return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest())

    def warm(self, spec):
        spec = normalize_spec(spec)
        if self.worker._owner is not self.owner:
            raise ValueError("frozen prompt bank belongs to another worker")
        key = digest({k: spec[k] for k in ("prompt", "negative_prompt", "guidance_scale")})
        if key not in self.entries:
            payload = self.worker.encode_prompt(spec)
            if payload.owner is not self.owner or payload.consumed:
                raise ValueError("invalid fresh prompt encoding")
            self.entries[key] = dict(payload=payload, prompt=self._hash(payload.prompt_embeds),
                                     negative=self._hash(payload.negative_prompt_embeds))
        entry = self.entries[key]
        payload = entry["payload"]
        if (entry["prompt"] != self._hash(payload.prompt_embeds) or
                entry["negative"] != self._hash(payload.negative_prompt_embeds)):
            raise ValueError("frozen text embedding content changed")
        return dict(input_sha256=key, prompt=entry["prompt"], negative=entry["negative"],
                    mode="frozen_cpu_embeddings_mechanism_only")

    def prepare(self, spec):
        spec = normalize_spec(spec)
        identity = self.warm(spec)
        original = self.entries[identity["input_sha256"]]["payload"]
        # Each installation consumes its own payload. Copies also protect the
        # bank from an implementation mutating CPU input during installation.
        payload = EncodedPrompt(spec, original.prompt_embeds.clone(),
            None if original.negative_prompt_embeds is None else original.negative_prompt_embeds.clone(),
            cpu_prompt_encode_ms=0., owner=self.owner)
        state = self.worker.prepare_encoded(spec, payload)
        state.events.append(dict(kind="frozen_text_install", identity=identity,
                                 timing_scope="encoding_charged_once_in_stage_setup"))
        return state

    def manifest(self):
        return [dict(input_sha256=key, prompt=value["prompt"], negative=value["negative"],
                     encode_ms=value["payload"].cpu_prompt_encode_ms)
                for key, value in sorted(self.entries.items())]
