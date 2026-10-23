"""SANA-Sprint requests, optionally reusing bounded adapter-owned prompt tensors."""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import math
import time
import uuid

from ..base import Backend, progress


class SanaBackend(Backend):
    def __init__(self, config):
        super().__init__(config)
        options = config.get("options", {})
        limits = {"prompt_cache_entries", "prompt_cache_bytes"}
        allowed = limits | {"prompt_cache_sync_mode"}
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError("the Sana backend takes no options except prompt_cache_entries, prompt_cache_bytes and prompt_cache_sync_mode")
        if set(options) & limits and not limits <= set(options):
            raise ValueError("prompt_cache_entries and prompt_cache_bytes must be supplied together")
        for name, maximum in (("prompt_cache_entries", 64), ("prompt_cache_bytes", 256 * 1024 * 1024)):
            if name in options and (type(options[name]) is not int or not 1 <= options[name] <= maximum):
                raise ValueError("%s must be an integer in [1, %d]" % (name, maximum))
        self._sync_mode = options.get("prompt_cache_sync_mode", "staged")
        if not isinstance(self._sync_mode, str) or self._sync_mode not in ("staged", "fused"):
            raise ValueError("prompt_cache_sync_mode must be staged or fused")
        self._cache_enabled = limits <= set(options)
        if self._sync_mode == "fused" and not self._cache_enabled:
            raise ValueError("fused prompt_cache_sync_mode requires enabled prompt cache limits")
        self._limits = dict(entries=options.get("prompt_cache_entries", 0),
                            bytes=options.get("prompt_cache_bytes", 0))
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_sequence = 0
        self._adapter_id = uuid.uuid4().hex
        self._counters = dict(lookups=0, hits=0, misses=0, insertions=0,
                              evictions=0, bypasses=0, errors=0, clears=0)
        self._cold_service = dict(count=0, total_ms=0.)
        self._warm_service = dict(count=0, total_ms=0.)
        from .runtime import SanaAdapter
        self.runtime = SanaAdapter(config["model_path"], config.get("device", "cuda:0"))

    def _snapshot(self):
        return dict(entries=len(self._cache), bytes=self._cache_bytes,
                    lru=[dict(key=entry["key"], bytes=entry["bytes"]) for entry in self._cache.values()])

    def _clear_cache(self):
        self._cache.clear()
        self._cache_bytes = 0
        self._counters["clears"] += 1

    @staticmethod
    def _milliseconds(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Sana service timing must be finite and nonnegative")
        return float(value)

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
        prompt = state["spec"]["prompt"]
        self._cache_sequence += 1
        before = self._snapshot()
        outcome, encode_called, admitted, bypass_reason = "disabled", False, False, None
        entry_bytes, entry_tensors, evicted = 0, [], []
        encode_ms, lookup_ms = 0., 0.
        encoded = None
        output, service = None, None

        def admit(packet):
            nonlocal entry_tensors, entry_bytes, bypass_reason, admitted
            entry_tensors = packet.storage_details()
            entry_bytes = sum(tensor["accounted_bytes"] for tensor in entry_tensors)
            if entry_bytes > self._limits["bytes"]:
                bypass_reason = "entry_exceeds_byte_limit"
                self._counters["bypasses"] += 1
                return
            while (len(self._cache) >= self._limits["entries"]
                   or self._cache_bytes + entry_bytes > self._limits["bytes"]):
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= old["bytes"]
                evicted.append(dict(key=old["key"], bytes=old["bytes"]))
                self._counters["evictions"] += 1
                del old
            self._cache[prompt] = dict(encoded=packet, bytes=entry_bytes, tensors=entry_tensors,
                                      key=hashlib.sha256(prompt.encode("utf-8")).hexdigest())
            self._cache_bytes += entry_bytes
            self._counters["insertions"] += 1
            admitted = True

        try:
            if self._cache_enabled:
                self._counters["lookups"] += 1
                started = time.perf_counter()
                entry = self._cache.get(prompt)
                if entry is not None:
                    self._cache.move_to_end(prompt)
                lookup_ms = (time.perf_counter() - started) * 1000
                if entry is not None:
                    outcome = "hit"
                    self._counters["hits"] += 1
                    encoded, entry_bytes, entry_tensors = entry["encoded"], entry["bytes"], entry["tensors"]
                else:
                    outcome, encode_called = "miss", True
                    self._counters["misses"] += 1
                    if self._sync_mode == "fused":
                        encoded, output, service = self.runtime.encode_and_generate(
                            prompt, seed=state["spec"].get("seed", 0), on_encoded=admit)
                        encode_ms = self._milliseconds(service["encode_ms"])
                        generation_ms = self._milliseconds(service["generation_ms"])
                        if self._milliseconds(service["ms"]) != encode_ms + generation_ms:
                            raise ValueError("fused Sana service timing must equal its disjoint wall intervals")
                    else:
                        encoded = self.runtime.encode(prompt)
                        encode_ms = self._milliseconds(encoded.encode_ms)
                        admit(encoded)
            if service is None:
                arguments = dict(seed=state["spec"].get("seed", 0))
                if self._cache_enabled:
                    arguments["encoded"] = encoded
                # With the default disabled cache, preserve the original full-pipeline call.
                output, service = self.runtime.generate(prompt, **arguments)
                generation_ms = self._milliseconds(service["ms"])
            service_ms = encode_ms + generation_ms
            if self._cache_enabled:
                aggregate = self._warm_service if outcome == "hit" else self._cold_service
                aggregate["count"] += 1
                aggregate["total_ms"] += service_ms
                # The adapter packet stores its original encoding cost. Charge it
                # only on this request's miss, never again on a later hit.
                service = dict(service, ms=service_ms, generation_ms=generation_ms, encode_ms=encode_ms)
            receipt = dict(schema="ditserve.sana.prompt-cache.v1", enabled=self._cache_enabled,
                           sequence=self._cache_sequence, request_id=request["state_id"],
                           prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                           adapter_id=self._adapter_id, outcome=outcome, encode_called=encode_called,
                           admitted=admitted, bypass_reason=bypass_reason, entry_bytes=entry_bytes,
                           entry_tensors=deepcopy(entry_tensors), before=before, after=self._snapshot(),
                           evicted=evicted, limits=dict(self._limits), counters=dict(self._counters),
                           encode_ms=encode_ms, generation_ms=generation_ms, service_ms=service_ms,
                           lookup_ms=lookup_ms, cold_service=dict(self._cold_service),
                           warm_service=dict(self._warm_service), cumulative_includes_warmup=True,
                           owner_scope="adapter_instance", key_scope="exact_prompt_utf8")
            if self._sync_mode == "fused":
                receipt.update(sync_mode="fused", timing_scope=service.get(
                    "timing_scope", "synchronized_preencoded_generation"))
                for name in ("encode_timing_scope", "generation_timing_scope"):
                    if name in service:
                        receipt[name] = service[name]
        except Exception:
            self._counters["errors"] += 1
            self._clear_cache()
            raise
        state["output"], state["prompt_cache"] = output, receipt
        return dict(state_id=request["state_id"], progress=progress(1, 1), events=[service],
                    prompt_cache=deepcopy(receipt), model_ms=service_ms)

    def finish(self, request):
        state = self.state(request)
        if state["output"] is None:
            raise ValueError("finish requires completed model work")
        mode = request.get("output_mode", "file")
        output = self.persist(state["output"], dict(request, output_mode="digest") if mode == "check" else request)
        if mode in ("check", "digest"):
            output["output_mode"] = mode
            output["sha256"] = output["array_sha256"]
        return dict(state_id=request["state_id"], progress=progress(1, 1), output=output,
                    prompt_cache=deepcopy(state["prompt_cache"]), model_ms=0.)

    def close(self):
        try:
            super().close()
        finally:
            self._clear_cache()
            self.runtime.close()
