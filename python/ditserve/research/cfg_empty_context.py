"""Fail-closed proof that the pinned diffusers cache contexts are empty.

An empty native registry stores no context name: ``cache_context`` only creates
the root registry, then walks the module tree to notify stateful hooks. Preserve
that creation once and check an immutable module topology at each step. This is
not a replacement for cache-dit's separate, request-owned CFG cache contexts.

The worker lock excludes supported concurrent mutations. Changes made between
steps are rejected; mutating model internals concurrently outside that lock is
unsupported. No dependency method or module registration API is patched.
"""

import diffusers.hooks as _diffusers_hooks
from diffusers.hooks import HookRegistry
from diffusers.models.cache_utils import CacheMixin
from torch.nn.modules import module as _torch_module


_MISSING = object()
_CACHE_CONTEXT = CacheMixin.cache_context
_SET_CONTEXT = HookRegistry._set_context
_INITIALIZE = HookRegistry.check_if_exists_or_initialize.__func__
_GLOBAL_HOOK_NAMES = ("_global_forward_hooks", "_global_forward_pre_hooks",
                      "_global_forward_hooks_always_called", "_global_forward_hooks_with_kwargs")
_CLASS_ATTRIBUTES = ("forward", "_diffusers_hook", "_hf_hook", "_orig_mod", "_compiled_call_impl")


class EmptyCacheContextGuard:
    """Keep a flat model table and prove the native context remains a no-op."""

    mode = "noop_guarded_v1"

    def __init__(self, model):
        self.model = model
        self._check_native_methods()
        # This is the sole observable effect of the first empty native context.
        if type(HookRegistry.check_if_exists_or_initialize(model)) is not HookRegistry:
            raise ValueError("empty cache-context guard requires an exact native root registry")
        entries, classes = [], {}
        for module in model.modules():
            attributes = vars(module)
            children = attributes.get("_modules")
            if type(children) is not dict:
                raise ValueError("empty cache-context guard requires native module registration dictionaries")
            cls = type(module)
            if cls not in classes:
                if getattr(cls, "_diffusers_hook", _MISSING) is not _MISSING:
                    raise ValueError("empty cache-context guard refuses class-level registries")
                classes[cls] = tuple(getattr(cls, name, _MISSING) for name in _CLASS_ATTRIBUTES)
            entries.append((module, attributes, children, tuple(children.items()),
                            getattr(module, "_diffusers_hook", _MISSING), attributes.get("forward", _MISSING)))
        self._entries = tuple(entries)
        self._classes = tuple(classes.items())
        self.validate()

    def _check_native_methods(self):
        method = self.model.cache_context
        if (getattr(method, "__self__", None) is not self.model or
                getattr(method, "__func__", None) is not _CACHE_CONTEXT or
                CacheMixin.cache_context is not _CACHE_CONTEXT or
                _diffusers_hooks.HookRegistry is not HookRegistry or
                HookRegistry._set_context is not _SET_CONTEXT or
                getattr(HookRegistry.check_if_exists_or_initialize, "__func__", None) is not _INITIALIZE):
            raise ValueError("empty cache-context guard requires unmodified native context methods")
        if any(getattr(_torch_module, name, None) for name in _GLOBAL_HOOK_NAMES):
            raise ValueError("empty cache-context guard refuses global forward hooks")

    @staticmethod
    def _check_registry(registry, module):
        if registry is _MISSING:
            return
        if type(registry) is not HookRegistry:
            raise ValueError("empty cache-context guard requires exact native registries")
        values = vars(registry)
        if (values.keys() != {"hooks", "_module_ref", "_hook_order", "_fn_refs"} or
                values["_module_ref"] is not module or
                type(values["hooks"]) is not dict or values["hooks"] or
                type(values["_hook_order"]) is not list or values["_hook_order"] or
                type(values["_fn_refs"]) is not list or values["_fn_refs"]):
            raise ValueError("empty cache-context guard refuses nonempty or modified hook registries")

    def validate(self):
        """One flat guard replaces four context enter/exit tree walks per step."""
        self._check_native_methods()
        for cls, expected in self._classes:
            if any(getattr(cls, name, _MISSING) is not value
                   for name, value in zip(_CLASS_ATTRIBUTES, expected)):
                raise ValueError("empty cache-context guard detected changed module class methods or hooks")
        for module, attributes, children, edges, registry, forward in self._entries:
            if (vars(module) is not attributes or attributes.get("_modules") is not children or
                    len(children) != len(edges) or
                    any(children.get(name, _MISSING) is not child for name, child in edges)):
                raise ValueError("empty cache-context guard detected changed module structure")
            if (attributes.get("forward", _MISSING) is not forward or
                    attributes.get("_diffusers_hook", _MISSING) is not registry or
                    "_diffusers_hook" in attributes.get("_buffers", {}) or
                    "_diffusers_hook" in attributes.get("_parameters", {})):
                raise ValueError("empty cache-context guard detected changed forward or registry")
            if (attributes.get("training") is not False or attributes.get("_hf_hook") is not None or
                    attributes.get("_orig_mod") is not None or attributes.get("_compiled_call_impl") is not None or
                    attributes.get("_forward_hooks") or attributes.get("_forward_pre_hooks") or
                    attributes.get("_forward_hooks_with_kwargs") or attributes.get("_forward_hooks_always_called") or
                    attributes.get("_forward_pre_hooks_with_kwargs")):
                raise ValueError("empty cache-context guard refuses training, hooks, offload or compilation")
            self._check_registry(registry, module)
