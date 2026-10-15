"""The empty-context shortcut must reject changes instead of hiding hook state."""

import contextlib
import importlib.util
import unittest
from unittest.mock import patch

from ditserve.research.cfg_empty_context import EmptyCacheContextGuard


class EmptyCacheContextGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("torch", "diffusers")):
            raise unittest.SkipTest("pinned CPU dependencies required")
        import torch
        from diffusers.hooks import HookRegistry
        from diffusers.models.cache_utils import CacheMixin
        cls.torch, cls.Registry, cls.CacheMixin = torch, HookRegistry, CacheMixin
        torch.set_num_threads(1)

    def model(self, *, counted=False):
        torch, mixin = self.torch, self.CacheMixin

        class Tiny(torch.nn.Module, mixin):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
                self.walks = 0

            def forward(self, *args):
                raise AssertionError("context validation must not execute a forward")

        if counted:
            class CountedTiny(Tiny):
                def named_modules(self, *args, **kwargs):
                    self.walks += 1
                    return super().named_modules(*args, **kwargs)
            return CountedTiny().eval()
        return Tiny().eval()

    def test_initialization_preserves_native_root_registry_side_effect(self):
        model = self.model()
        self.assertFalse(hasattr(model, "_diffusers_hook"))
        guard = EmptyCacheContextGuard(model)
        registry = model._diffusers_hook
        self.assertIs(type(registry), self.Registry)
        self.assertIs(registry._module_ref, model)
        self.assertEqual(vars(registry), dict(hooks={}, _module_ref=model, _hook_order=[], _fn_refs=[]))
        guard.validate()
        self.assertIs(model._diffusers_hook, registry)
        self.assertTrue(all(not hasattr(m, "_diffusers_hook") for m in model.modules() if m is not model))

    def test_native_empty_context_names_leave_model_and_rng_unchanged(self):
        model = self.model()
        for module in (model, model.layers, model.layers[0]):
            self.Registry.check_if_exists_or_initialize(module)
        guard = EmptyCacheContextGuard(model)
        modules = tuple(model.modules())
        before_attrs = [{key: id(value) for key, value in vars(module).items()} for module in modules]
        before_registry = [dict(vars(module._diffusers_hook)) if hasattr(module, "_diffusers_hook") else None
                           for module in modules]
        before_tensors = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        before_rng = self.torch.get_rng_state().clone()
        for name in ("cond", "uncond", "another-request", None):
            with model.cache_context(name):
                guard.validate()
            guard.validate()
        self.assertEqual(before_attrs, [{key: id(value) for key, value in vars(module).items()}
                                        for module in modules])
        self.assertEqual(before_registry, [dict(vars(module._diffusers_hook))
                         if hasattr(module, "_diffusers_hook") else None for module in modules])
        self.assertTrue(self.torch.equal(before_rng, self.torch.get_rng_state()))
        for name, tensor in model.state_dict().items():
            self.assertTrue(self.torch.equal(before_tensors[name], tensor), name)

    def test_validation_reuses_flat_module_table_without_tree_walks(self):
        model = self.model(counted=True)
        guard = EmptyCacheContextGuard(model)
        walks = model.walks
        for _ in range(32):
            guard.validate()
        self.assertEqual(model.walks, walks)

    def test_non_native_cache_context_is_rejected_at_creation_and_later(self):
        for after in (False, True):
            with self.subTest(after=after):
                model = self.model()
                guard = EmptyCacheContextGuard(model) if after else None
                model.cache_context = lambda name: contextlib.nullcontext()
                with self.assertRaises(ValueError):
                    guard.validate() if after else EmptyCacheContextGuard(model)

    def test_cache_context_class_method_change_is_rejected(self):
        model = self.model()
        guard = EmptyCacheContextGuard(model)
        with patch.object(self.CacheMixin, "cache_context", lambda self, name: contextlib.nullcontext()):
            with self.assertRaises(ValueError):
                guard.validate()

    def test_registry_type_and_owner_are_strict(self):
        class DerivedRegistry(self.Registry):
            pass
        for replacement in (lambda model: DerivedRegistry(model),
                            lambda model: self.Registry(model.layers[0])):
            for after in (False, True):
                with self.subTest(replacement=replacement, after=after):
                    model = self.model()
                    guard = EmptyCacheContextGuard(model) if after else None
                    model._diffusers_hook = replacement(model)
                    with self.assertRaises(ValueError):
                        guard.validate() if after else EmptyCacheContextGuard(model)

    def test_explicit_none_registry_is_not_treated_as_missing(self):
        for target_name in ("root", "child"):
            for after in (False, True):
                with self.subTest(target=target_name, after=after):
                    model = self.model()
                    guard = EmptyCacheContextGuard(model) if after else None
                    target = model if target_name == "root" else model.layers[0]
                    target._diffusers_hook = None
                    with self.assertRaises(ValueError):
                        guard.validate() if after else EmptyCacheContextGuard(model)

    def test_class_registry_attribute_is_rejected_even_when_none(self):
        for target_name in ("root", "child"):
            for after in (False, True):
                with self.subTest(target=target_name, after=after):
                    model = self.model()
                    guard = EmptyCacheContextGuard(model) if after else None
                    target = type(model) if target_name == "root" else type(model.layers[0])
                    with patch.object(target, "_diffusers_hook", None, create=True):
                        with self.assertRaises(ValueError):
                            guard.validate() if after else EmptyCacheContextGuard(model)

    def test_registry_name_cannot_hide_in_torch_registered_state(self):
        for kind in ("module", "buffer", "parameter"):
            for after in (False, True):
                model = self.model()
                guard = EmptyCacheContextGuard(model) if after else None
                child = model.layers[0]
                if kind == "module":
                    child.add_module("_diffusers_hook", self.torch.nn.Identity().eval())
                elif kind == "buffer":
                    child.register_buffer("_diffusers_hook", None)
                else:
                    child.register_parameter("_diffusers_hook", None)
                with self.subTest(kind=kind, after=after), self.assertRaises(ValueError):
                    guard.validate() if after else EmptyCacheContextGuard(model)

    def test_registry_containers_require_exact_empty_types(self):
        class DictSubclass(dict):
            pass
        class ListSubclass(list):
            pass
        for name, value in (("hooks", {"injected": object()}), ("hooks", []),
                            ("hooks", DictSubclass()), ("_hook_order", ["injected"]),
                            ("_hook_order", ()), ("_hook_order", ListSubclass()),
                            ("_fn_refs", [object()]), ("_fn_refs", ()),
                            ("_fn_refs", ListSubclass())):
            for after in (False, True):
                with self.subTest(name=name, type=type(value), after=after):
                    model = self.model()
                    self.Registry.check_if_exists_or_initialize(model.layers[0])
                    guard = EmptyCacheContextGuard(model) if after else None
                    setattr(model.layers[0]._diffusers_hook, name, value)
                    with self.assertRaises(ValueError):
                        guard.validate() if after else EmptyCacheContextGuard(model)

    def test_registry_instance_and_class_context_methods_cannot_change(self):
        for instance in (False, True):
            with self.subTest(instance=instance):
                model = self.model()
                guard = EmptyCacheContextGuard(model)
                target = model._diffusers_hook if instance else self.Registry
                with patch.object(target, "_set_context", lambda *args: None):
                    with self.assertRaises(ValueError):
                        guard.validate()

    def test_registry_initialization_method_cannot_change(self):
        model = self.model()
        guard = EmptyCacheContextGuard(model)
        with patch.object(self.Registry, "check_if_exists_or_initialize", lambda *args: None):
            with self.assertRaises(ValueError):
                guard.validate()

    def test_late_child_registry_is_rejected(self):
        model = self.model()
        guard = EmptyCacheContextGuard(model)
        self.Registry.check_if_exists_or_initialize(model.layers[0])
        with self.assertRaises(ValueError):
            guard.validate()

    def test_local_torch_hooks_cannot_be_added(self):
        for register in ("register_forward_hook", "register_forward_pre_hook"):
            model = self.model()
            guard = EmptyCacheContextGuard(model)
            handle = getattr(model.layers[1], register)(lambda *args: None)
            try:
                with self.subTest(register=register), self.assertRaises(ValueError):
                    guard.validate()
            finally:
                handle.remove()
            guard.validate()

    def test_global_torch_hooks_cannot_exist_or_be_added(self):
        from torch.nn.modules import module as module_api
        for register in ("register_module_forward_hook", "register_module_forward_pre_hook"):
            for after in (False, True):
                model = self.model()
                guard = EmptyCacheContextGuard(model) if after else None
                handle = getattr(module_api, register)(lambda *args: None)
                try:
                    with self.subTest(register=register, after=after), self.assertRaises(ValueError):
                        guard.validate() if after else EmptyCacheContextGuard(model)
                finally:
                    handle.remove()

    def test_module_forward_changes_are_rejected(self):
        for class_change in (False, True):
            with self.subTest(class_change=class_change):
                model = self.model()
                guard = EmptyCacheContextGuard(model)
                target = type(model.layers[0]) if class_change else model.layers[0]
                with patch.object(target, "forward", lambda *args: None):
                    with self.assertRaises(ValueError):
                        guard.validate()

    def test_structure_add_remove_replace_and_mapping_replacement_are_rejected(self):
        mutations = (
            lambda model: model.layers.add_module("new", self.torch.nn.Identity()),
            lambda model: model.layers._modules.pop("0"),
            lambda model: model.layers._modules.__setitem__("0", self.torch.nn.Linear(2, 2)),
            lambda model: setattr(model.layers, "_modules", dict(model.layers._modules)),
        )
        for mutate in mutations:
            model = self.model()
            guard = EmptyCacheContextGuard(model)
            mutate(model)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                guard.validate()


if __name__ == "__main__":
    unittest.main()
