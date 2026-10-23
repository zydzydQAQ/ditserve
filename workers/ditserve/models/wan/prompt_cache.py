"""Bounded worker-owned CPU negative embeddings; never caches request state.

The caller serializes access with its encoder lock. Capacity covers retained
backing storage, not transient clones or the encoder's working memory.
"""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import time
import uuid


def cache_options(options):
    """Validate the optional pair before a backend loads model weights."""
    if not isinstance(options, dict):
        raise ValueError("Wan options must be an object")
    names = ("negative_prompt_cache_entries", "negative_prompt_cache_bytes")
    present = [name in options for name in names]
    if any(present) and not all(present):
        raise ValueError("negative_prompt_cache_entries and negative_prompt_cache_bytes must be supplied together")
    if not any(present):
        return {}
    for name, maximum in zip(names, (64, 256 * 1024 * 1024)):
        if type(options[name]) is not int or not 1 <= options[name] <= maximum:
            raise ValueError("%s must be an integer in [1, %d]" % (name, maximum))
    return {name: options[name] for name in names}


def tensor_storage(tensor):
    """Count the whole backing allocation, including bytes outside a view."""
    logical = int(tensor.numel()) * int(tensor.element_size())
    storage = int(tensor.untyped_storage().nbytes())
    if logical < 0 or storage < 0:
        raise ValueError("invalid negative embedding storage size")
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), device=str(tensor.device),
                logical_bytes=logical, storage_bytes=storage, accounted_bytes=storage)


class NegativePromptCache:
    def __init__(self, *, entries, byte_limit, recipe):
        cache_options(dict(negative_prompt_cache_entries=entries, negative_prompt_cache_bytes=byte_limit))
        self.limits = dict(entries=entries, bytes=byte_limit)
        self.recipe = deepcopy(recipe)
        self.recipe_sha256 = hashlib.sha256(json.dumps(
            self.recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.instance_id = uuid.uuid4().hex
        self.entries = OrderedDict()
        self.bytes = 0
        self.sequence = 0
        self.counters = dict(lookups=0, hits=0, misses=0, insertions=0, evictions=0,
                             bypasses=0, errors=0, clears=0)

    def snapshot(self):
        return dict(entries=len(self.entries), bytes=self.bytes,
                    lru=[dict(key=key, bytes=entry["bytes"]) for key, entry in self.entries.items()])

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        self.counters["clears"] += 1

    def encode(self, encode, arguments, *, request_id, validate):
        """Use the unchanged official call on misses, supplying only a hit clone.

        ``validate`` runs before insertion and on hits. A request receives an
        independent tensor even if the official function returns its argument
        unchanged. Cache errors are fatal to the request and drop all entries.
        """
        self.sequence += 1
        before = self.snapshot()
        negative_sha256 = hashlib.sha256(arguments["negative_prompt"].encode("utf-8")).hexdigest()
        key = hashlib.sha256((self.recipe_sha256 + ":" + negative_sha256).encode("ascii")).hexdigest()
        outcome, bypass_reason, admitted = "bypass", None, False
        entry_bytes, entry_tensor, evicted = 0, None, []
        lookup_ms = clone_ms = store_ms = 0.
        call_arguments = dict(arguments)
        try:
            if arguments["do_classifier_free_guidance"]:
                self.counters["lookups"] += 1
                started = time.perf_counter()
                entry = self.entries.get(key)
                lookup_ms = (time.perf_counter() - started) * 1000
                if entry is None:
                    outcome = "miss"
                    self.counters["misses"] += 1
                else:
                    outcome = "hit"
                    self.counters["hits"] += 1
                    self.entries.move_to_end(key)
                    entry_bytes, entry_tensor = entry["bytes"], entry["tensor_info"]
                    started = time.perf_counter()
                    call_arguments["negative_prompt_embeds"] = entry["tensor"].clone()
                    clone_ms = (time.perf_counter() - started) * 1000
            else:
                bypass_reason = "cfg_disabled"
                self.counters["bypasses"] += 1
            started = time.perf_counter()
            prompt, negative = encode(**call_arguments)
            encode_call_ms = (time.perf_counter() - started) * 1000
            validate(prompt, negative)
            if outcome == "miss":
                started = time.perf_counter()
                retained = negative.detach().clone()
                entry_tensor = tensor_storage(retained)
                entry_bytes = entry_tensor["accounted_bytes"]
                if entry_bytes > self.limits["bytes"]:
                    bypass_reason = "entry_exceeds_byte_limit"
                    self.counters["bypasses"] += 1
                else:
                    while (len(self.entries) >= self.limits["entries"] or
                           self.bytes + entry_bytes > self.limits["bytes"]):
                        old_key, old = self.entries.popitem(last=False)
                        self.bytes -= old["bytes"]
                        self.counters["evictions"] += 1
                        evicted.append(dict(key=old_key, bytes=old["bytes"]))
                        del old
                    self.entries[key] = dict(tensor=retained, tensor_info=entry_tensor, bytes=entry_bytes)
                    self.bytes += entry_bytes
                    self.counters["insertions"] += 1
                    admitted = True
                store_ms = (time.perf_counter() - started) * 1000
            receipt = dict(schema="ditserve.wan.negative-prompt-cache.v1", enabled=True,
                           sequence=self.sequence, request_id=request_id, instance_id=self.instance_id,
                           owner_scope="worker_instance", key_scope="exact_negative_prompt_utf8_and_recipe",
                           negative_prompt_sha256=negative_sha256, recipe=deepcopy(self.recipe),
                           recipe_sha256=self.recipe_sha256, key=key, outcome=outcome,
                           negative_encode_called=outcome == "miss", admitted=admitted,
                           bypass_reason=bypass_reason, entry_bytes=entry_bytes,
                           entry_tensor=deepcopy(entry_tensor), before=before, after=self.snapshot(),
                           evicted=evicted, limits=dict(self.limits), counters=dict(self.counters),
                           lookup_ms=lookup_ms, clone_ms=clone_ms, store_ms=store_ms,
                           encode_call_ms=encode_call_ms, cumulative_includes_warmup=True)
            return prompt, negative, receipt
        except BaseException:
            self.counters["errors"] += 1
            self.clear()
            raise
