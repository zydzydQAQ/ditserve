"""CPU object-lifetime tests for native D3 session/oracle cache ownership."""

import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from ditserve.directions import direction3
from ditserve.directions.budget import BudgetStop


class Cache:
    pass


class VAE:
    def __init__(self):
        self._feat_map = [Cache()]
        self.clears = 0

    def clear_cache(self):
        self.clears += 1
        self._feat_map = []


def session(name):
    return SimpleNamespace(session_id=name, kv=Cache(), cross=Cache(),
                           conditional=object(), noise=object(), history=[object()],
                           vae={"_feat_map": [Cache()]})


class OracleTensor:
    """Inert shape carrier sufficient to reach the next allocation boundary."""

    def __init__(self, transfer_error=None):
        self.transfer_error = transfer_error

    def clone(self):
        return OracleTensor()

    def detach(self):
        return self

    def cpu(self):
        if self.transfer_error:
            raise self.transfer_error("oracle host transfer failed")
        return self

    def __getitem__(self, index):
        return self

    def permute(self, *dimensions):
        return self

    def float(self):
        return self

    def numpy(self):
        return []


class Direction3CacheLifetimeTests(unittest.TestCase):
    def runtime(self):
        value = direction3.NativeLongLive.__new__(direction3.NativeLongLive)
        value.sessions = {}
        value.pipe = SimpleNamespace(kv_cache1=None, crossattn_cache=None,
                                     vae=SimpleNamespace(model=VAE()))
        return value

    def test_release_inactive_session_preserves_active_kv_cross_and_vae(self):
        runtime = self.runtime()
        active, inactive = session("active"), session("inactive")
        runtime.sessions = {value.session_id: value for value in (active, inactive)}
        runtime.pipe.kv_cache1, runtime.pipe.crossattn_cache = active.kv, active.cross
        runtime.pipe.vae.model._feat_map = active.vae["_feat_map"]
        active_vae = runtime.pipe.vae.model._feat_map
        runtime.release(inactive)
        self.assertIs(runtime.pipe.kv_cache1, active.kv)
        self.assertIs(runtime.pipe.crossattn_cache, active.cross)
        self.assertIs(runtime.pipe.vae.model._feat_map, active_vae)
        self.assertIs(active.vae["_feat_map"], active_vae)
        self.assertEqual(runtime.pipe.vae.model.clears, 0)
        self.assertEqual(list(runtime.sessions), ["active"])
        self.assertIsNone(inactive.noise)
        self.assertEqual(inactive.history, [])

    def test_release_active_session_drops_shared_pointers_and_preserves_other_state(self):
        runtime = self.runtime()
        active, other = session("active"), session("other")
        runtime.sessions = {value.session_id: value for value in (active, other)}
        runtime.pipe.kv_cache1, runtime.pipe.crossattn_cache = active.kv, active.cross
        saved_other = (other.kv, other.cross, other.vae["_feat_map"])
        runtime.release(active)
        self.assertIsNone(runtime.pipe.kv_cache1)
        self.assertIsNone(runtime.pipe.crossattn_cache)
        self.assertEqual(runtime.pipe.vae.model.clears, 1)
        self.assertEqual(list(runtime.sessions), ["other"])
        self.assertIs(other.kv, saved_other[0])
        self.assertIs(other.cross, saved_other[1])
        self.assertIs(other.vae["_feat_map"], saved_other[2])

    def oracle_runtime(self, oracle_error=None, transfer_error=None):
        runtime = self.runtime()
        runtime.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        runtime.check_budget = lambda: None
        runtime.rng = lambda state: contextlib.nullcontext()
        runtime.oracle_refs = []
        runtime.created = 0

        def new_session(sid, seed, blocks):
            runtime.created += 1
            if runtime.created == 2:
                self.assertIsNone(runtime.pipe.kv_cache1)
                self.assertIsNone(runtime.pipe.crossattn_cache)
                self.assertTrue(all(reference() is None for reference in runtime.oracle_refs))
                self.assertEqual(runtime.sessions, {})
                raise LookupError("reached clean next-session allocation")
            state = session(sid)
            state.noise = OracleTensor()
            runtime.sessions[sid] = state
            runtime.pipe.kv_cache1, runtime.pipe.crossattn_cache = state.kv, state.cross
            return state

        def inference(**kwargs):
            runtime.pipe.kv_cache1, runtime.pipe.crossattn_cache = Cache(), Cache()
            runtime.oracle_refs = [weakref.ref(runtime.pipe.kv_cache1), weakref.ref(runtime.pipe.crossattn_cache)]
            if oracle_error:
                raise oracle_error("oracle interrupted")
            return OracleTensor(transfer_error), OracleTensor()

        runtime.new_session = new_session
        runtime.pipe.inference = inference
        return runtime

    def run_correctness(self, runtime):
        config = {"direction3": dict(correctness_blocks=6, prompt_pairs=[["old", "new"]], seeds=[101])}
        with tempfile.TemporaryDirectory() as directory, patch.object(direction3, "_artifact", return_value={}):
            direction3._correctness(runtime, config, Path(directory), dict(samples=[]))

    def test_oracle_caches_are_dead_before_adapter_session_allocation(self):
        runtime = self.oracle_runtime()
        with self.assertRaisesRegex(LookupError, "clean next-session allocation"):
            self.run_correctness(runtime)
        self.assertEqual(runtime.created, 2)
        self.assertEqual(runtime.pipe.vae.model.clears, 1)

    def test_oracle_failure_or_host_transfer_failure_always_drops_caches(self):
        for oracle_error, transfer_error in ((RuntimeError, None), (BudgetStop, None),
                                             (KeyboardInterrupt, None), (None, RuntimeError)):
            with self.subTest(oracle_error=oracle_error, transfer_error=transfer_error):
                runtime = self.oracle_runtime(oracle_error, transfer_error)
                with self.assertRaises(oracle_error or transfer_error):
                    self.run_correctness(runtime)
                self.assertIsNone(runtime.pipe.kv_cache1)
                self.assertIsNone(runtime.pipe.crossattn_cache)
                self.assertEqual(runtime.sessions, {})
                self.assertTrue(all(reference() is None for reference in runtime.oracle_refs))
                self.assertEqual(runtime.pipe.vae.model.clears, 1)
                self.assertEqual(runtime.created, 1)
