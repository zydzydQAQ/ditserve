import unittest
from types import SimpleNamespace

import torch

from ditserve.gpu.worker import EncodedPrompt
from ditserve.research.prompt_bank import FrozenPromptBank


class FakeWorker:
    def __init__(self):
        self._owner = object()
        self.torch = torch
        self.encodings = 0
        self.payloads = []

    def encode_prompt(self, spec):
        self.encodings += 1
        return EncodedPrompt(spec, torch.ones(1, 4, 8), torch.zeros(1, 4, 8), 17., self._owner)

    def prepare_encoded(self, spec, payload):
        assert not payload.consumed and payload.spec == spec and payload.owner is self._owner
        payload.consumed = True
        self.payloads.append(payload)
        return SimpleNamespace(events=[])


class FrozenPromptTests(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.bank = FrozenPromptBank(self.worker)
        self.spec = dict(request_id="a", prompt="a kettle", height=256, width=256,
                         frames=17, steps=32, seed=101, guidance_scale=5.)

    def test_reuse_across_seed_shape_policy_has_new_consumable_payload(self):
        first = self.bank.warm(self.spec)
        self.bank.prepare(self.spec)
        self.bank.prepare(dict(self.spec, request_id="b", frames=33, seed=102, cache=True, threshold=.08))
        self.assertEqual(self.worker.encodings, 1)
        self.assertEqual(self.bank.warm(self.spec), first)
        self.assertTrue(all(p.consumed and p.cpu_prompt_encode_ms == 0 for p in self.worker.payloads))
        self.assertNotEqual(self.worker.payloads[0].prompt_embeds.data_ptr(), self.worker.payloads[1].prompt_embeds.data_ptr())
        self.assertEqual(self.bank.manifest()[0]["encode_ms"], 17.)

    def test_prompt_negative_and_cfg_are_distinct(self):
        for values in ({}, {"prompt": "fish"}, {"negative_prompt": "blur"}, {"guidance_scale": 1.}):
            self.bank.warm(dict(self.spec, **values))
        self.assertEqual(self.worker.encodings, 4)

    def test_mutation_and_worker_change_are_rejected(self):
        self.bank.warm(self.spec)
        next(iter(self.bank.entries.values()))["payload"].prompt_embeds[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.bank.prepare(self.spec)
        self.worker._owner = object()
        with self.assertRaisesRegex(ValueError, "another worker"):
            self.bank.warm(self.spec)


if __name__ == "__main__":
    unittest.main()
