"""CPU codec/ownership and real Gloo handoff tests; no GPU qualification."""

import copy
from concurrent.futures import Future
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def adapter(rank, *, owned=True, boundary=2, groups=None):
    from ditserve.models.base import Backend
    from ditserve.models.wan.backend import WanBackend
    from ditserve.models.wan.block import WanBlockExecutor
    from ditserve.models.wan.elastic import ElasticRequests
    try:
        from wan_fixture import fixture
    except ImportError:
        from tests.python.wan_fixture import fixture
    worker, state = fixture(boundary=boundary)
    worker.pipe = SimpleNamespace(transformer=SimpleNamespace(blocks=[None] * 30))
    backend = WanBackend.__new__(WanBackend)
    Backend.__init__(backend, {})
    backend.runtime, backend.execution = worker, "block"
    backend.parallel = SimpleNamespace(groups=groups or SimpleNamespace(rank=rank))
    with patch.object(WanBlockExecutor, "_validate_model"):
        backend.executor = WanBlockExecutor(worker)
    backend.continuations = {}
    backend.elastic = ElasticRequests(backend)
    if owned:
        backend.states[state.request_id] = state
        if boundary < state.spec["steps"]:
            backend.continuations[state.request_id] = backend.executor.attach(state)
        backend.elastic.register(state.request_id)
    else:
        worker.release(state)
    return backend


def command(path, *, source=(0,), target=(1,), epoch=0, identity="transfer-1"):
    return dict(transfer_id=identity, state_id="request-a", expected_epoch=epoch,
                source_group=list(source), target_group=list(target), path=str(path))


def gloo_rank(rank, port, directory, policy="native"):
    import torch
    from ditserve.models.wan.parallel import build_fixed_groups, pack_request
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method="tcp://127.0.0.1:%d" % port,
                                        rank=rank, world_size=2)
    try:
        groups = build_fixed_groups(backend="gloo", timeout_seconds=15)
        backend = adapter(rank, owned=rank == 0, groups=groups)
        backend.parallel.linear_policy = policy
        if rank == 0:
            backend.elastic.records["request-a"].quantum_seq = 7
        common = command(Path(directory) / "expand.pt", target=(0, 1), identity="expand")
        expanded = backend.elastic.transition(common)
        assert expanded["ownership_epoch"] == 1
        assert expanded["progress"]["quantum_seq"] == 7
        first_digest = pack_request(backend.runtime, backend.states["request-a"])[0]["state_sha256"]
        replay = backend.elastic.transition(common)
        assert replay["ownership_epoch"] == 1
        # Move the retained replica to rank 1; rank 0 must lose ownership.
        shrunk = backend.elastic.transition(command(Path(directory) / "shrink.pt", source=(0, 1),
                                                    target=(1,), epoch=1, identity="shrink"))
        assert shrunk["ownership_epoch"] == 2
        if rank == 0:
            assert "request-a" not in backend.states
        else:
            assert pack_request(backend.runtime, backend.states["request-a"])[0]["state_sha256"] == first_digest
            assert backend.elastic.records["request-a"].quantum_seq == 7
        Path(directory, "rank%d.json" % rank).write_text(json.dumps(dict(expanded=expanded["ownership_epoch"],
                                                                        shrunk=shrunk["ownership_epoch"])))
        for key in list(backend.states):
            backend.release(dict(state_id=key))
    finally:
        torch.distributed.destroy_process_group()


class ElasticTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import diffusers
            import cache_dit
        except ImportError:
            raise unittest.SkipTest("pinned CPU dependencies required")
        torch.set_num_threads(1)
        cls.torch = torch

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def stage(self, source, target, req):
        exported = source.transfer(dict(req, action="source_prepare"))
        incoming = dict(req, action="target_prepare", sha256=exported["sha256"],
                        state_sha256=exported["state_sha256"], quantum_seq=exported["quantum_seq"])
        target.transfer(incoming)
        return exported, incoming

    def test_actual_codec_stages_unregistered_state_and_requires_activation_receipts(self):
        from ditserve.models.wan.parallel import pack_request
        source, target = adapter(0), adapter(1, owned=False)
        req = command(self.root / "state.pt")
        original = pack_request(source.runtime, source.states["request-a"])[0]["state_sha256"]
        exported, incoming = self.stage(source, target, req)
        self.assertNotIn("request-a", target.states)
        self.assertNotIn("request-a", target.runtime._states)
        with self.assertRaisesRegex(ValueError, "before transfer"):
            source.release(dict(state_id="request-a"))
        with self.assertRaisesRegex(Exception, "frozen"):
            source.elastic.before_advance(dict(state_id="request-a"))
        source.transfer(dict(req, action="commit"))
        with self.assertRaisesRegex(Exception, "activation receipt"):
            source.transfer(dict(req, action="finalize"))
        activated = target.transfer(dict(req, action="commit"))
        self.assertIn("request-a", source.states)
        self.assertEqual(pack_request(target.runtime, target.states["request-a"])[0]["state_sha256"], original)
        self.assertEqual(target.elastic.describe("request-a")["ownership_epoch"], 1)
        source.transfer(dict(req, action="finalize", activation_receipts=[activated]))
        self.assertNotIn("request-a", source.runtime._states)
        source.transfer(dict(req, action="finalize", activation_receipts=[activated]))
        target.transfer(dict(req, action="finalize", activation_receipts=[activated]))
        with self.assertRaisesRegex(Exception, "roll back committed"):
            target.transfer(dict(req, action="abort"))
        self.assertEqual(exported["bytes"], Path(req["path"]).stat().st_size)

    def test_corrupt_packet_rolls_back_without_releasing_source(self):
        source, target = adapter(0), adapter(1, owned=False)
        req = command(self.root / "state.pt")
        exported = source.transfer(dict(req, action="source_prepare"))
        Path(req["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(Exception, "checksum"):
            target.transfer(dict(req, action="target_prepare", sha256=exported["sha256"],
                                 state_sha256=exported["state_sha256"], quantum_seq=0))
        source.transfer(dict(req, action="abort"))
        target.transfer(dict(req, action="abort"))
        self.assertEqual(source.elastic.records["request-a"].status, "active")
        self.assertNotIn("request-a", target.states)
        with self.assertRaisesRegex(Exception, "aborted"):
            source.transfer(dict(req, action="source_prepare"))

    def test_transfer_packet_binds_linear_policy_and_rejects_mismatch(self):
        source, target = adapter(0), adapter(1, owned=False)
        source.parallel.linear_policy = "full_rows_except_ffn1"
        target.parallel.linear_policy = "native"
        req = command(self.root / "policy.pt")
        exported = source.transfer(dict(req, action="source_prepare"))
        payload = self.torch.load(req["path"], map_location="cpu", weights_only=True)
        self.assertEqual(payload["execution_contract"]["sp_linear_policy"], "full_rows_except_ffn1")
        incoming = dict(req, action="target_prepare", sha256=exported["sha256"],
                        state_sha256=exported["state_sha256"], quantum_seq=exported["quantum_seq"])
        with self.assertRaisesRegex(Exception, "execution contract"):
            target.transfer(incoming)
        self.assertNotIn("request-a", target.states)
        self.assertNotIn("request-a", target.runtime._states)
        source.transfer(dict(req, action="abort"))
        target.transfer(dict(req, action="abort"))

    def test_matching_repair_contract_survives_transfer_and_receipts(self):
        source, target = adapter(0), adapter(1, owned=False)
        source.parallel.linear_policy = target.parallel.linear_policy = "full_rows_except_ffn1"
        req = command(self.root / "policy-match.pt")
        _, incoming = self.stage(source, target, req)
        with self.assertRaisesRegex(Exception, "execution contract"):
            target.transfer(dict(incoming, execution_contract=dict(schema="ditserve.wan-sp-execution.v1",
                                                                    sp_linear_policy="native")))
        activated = target.transfer(dict(req, action="commit"))
        source.transfer(dict(req, action="commit"))
        changed = copy.deepcopy(activated)
        changed["execution_contract"]["sp_linear_policy"] = "native"
        with self.assertRaisesRegex(Exception, "activation receipt"):
            source.transfer(dict(req, action="finalize", activation_receipts=[changed]))
        source.transfer(dict(req, action="finalize", activation_receipts=[activated]))
        self.assertEqual(activated["execution_contract"]["sp_linear_policy"], "full_rows_except_ffn1")

    def test_stale_epoch_changed_transfer_and_midstep_refused(self):
        source = adapter(0)
        req = command(self.root / "state.pt", epoch=1)
        with self.assertRaisesRegex(Exception, "ownership/layout"):
            source.transfer(dict(req, action="source_prepare"))
        with self.assertRaisesRegex(Exception, "different contents"):
            source.transfer(dict(req, expected_epoch=0, action="source_prepare"))
        continuation = source.continuations["request-a"]
        continuation.next_block = 5
        with self.assertRaisesRegex(ValueError, "solver-step boundary"):
            source.transfer(dict(command(self.root / "mid.pt", identity="mid"), action="source_prepare"))

    def test_completed_dit_state_can_move_for_leader_decode(self):
        source, target = adapter(0, boundary=32), adapter(1, owned=False)
        req = command(self.root / "done.pt")
        self.stage(source, target, req)
        receipt = target.transfer(dict(req, action="commit"))
        self.assertEqual(target.states["request-a"].step_index, 32)
        self.assertNotIn("request-a", target.continuations)
        self.assertTrue(target._progress(target.states["request-a"])["transfer_ready"])
        source.transfer(dict(req, action="commit"))
        source.transfer(dict(req, action="finalize", activation_receipts=[receipt]))

    def test_quantum_replay_epoch_and_request_handles_are_independent(self):
        source = adapter(0)
        req = dict(state_id="request-a", expected_epoch=0, quantum_seq=0, blocks=4)
        record, sequence, cached = source.elastic.before_advance(req)
        result = dict(progress={})
        source.elastic.after_advance(req, result)
        self.assertIs(source.elastic.before_advance(req)[2], result)
        with self.assertRaisesRegex(Exception, "changed"):
            source.elastic.before_advance(dict(req, blocks=5))
        with self.assertRaisesRegex(Exception, "stale request"):
            source.elastic.before_advance(dict(req, expected_epoch=1))
        source.elastic.register("another")
        self.assertIsNot(record.handle, source.elastic.records["another"].handle)

    def test_check_mode_hashes_actual_output_and_latent_without_files(self):
        import hashlib
        backend = adapter(0)
        backend.runtime.model_path = Path("/models/fixture")
        backend.runtime.attention = "flash"
        state = backend.states["request-a"]
        request = dict(id=1, state_id="request-a", output_mode="check")
        output = backend._persist_checked(state.latents, request)
        evidence = backend._evidence(request, state.spec, state.latents)["latent"]
        expected = hashlib.sha256(state.latents.numpy().tobytes(order="C")).hexdigest()
        self.assertEqual(output["sha256"], expected)
        self.assertEqual(evidence["sha256"], expected)
        self.assertEqual(output["output_mode"], "check")
        self.assertNotIn("path", output)
        state.latents = state.latents + 1
        self.assertNotEqual(backend._persist_checked(state.latents, request)["sha256"], expected)

    def test_two_process_gloo_expand_shrink_and_idempotent_ack(self):
        import torch.multiprocessing as mp
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(gloo_rank, args=(port, str(self.root)), nprocs=2, join=True)
        for rank in (0, 1):
            self.assertEqual(json.loads((self.root / ("rank%d.json" % rank)).read_text()),
                             dict(expanded=1, shrunk=2))

    def test_two_process_repaired_contract_expand_shrink_and_idempotent_ack(self):
        import torch.multiprocessing as mp
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(gloo_rank, args=(port, str(self.root), "full_rows_except_ffn1"), nprocs=2, join=True)
        for rank in (0, 1):
            self.assertEqual(json.loads((self.root / ("rank%d.json" % rank)).read_text()),
                             dict(expanded=1, shrunk=2))


class EncoderTicketTests(unittest.TestCase):
    def backend(self, depth):
        from ditserve.models.wan.backend import WanBackend
        backend = WanBackend.__new__(WanBackend)
        backend.ticket_limit, backend.ticket_peak, backend.ticket_counter = depth, 0, 0
        backend.tickets, backend.cancelled_tickets = {}, set()
        backend._spec = lambda request: request["spec"]
        self.futures = []
        def submit(*unused):
            future = Future()
            self.futures.append(future)
            return future
        backend.encoder = SimpleNamespace(submit=submit)
        backend.runtime = SimpleNamespace(encode_prompt=lambda value: value)
        return backend

    def test_ready_ticket_still_consumes_capacity_and_failed_future_releases(self):
        backend = self.backend(1)
        first = backend.encode_start(dict(spec={"request_id": "a"}))
        self.futures[0].set_result(SimpleNamespace(cpu_prompt_encode_ms=1))
        self.assertTrue(backend.encode_poll(dict(ticket=first["ticket"]))["ready"])
        with self.assertRaisesRegex(ValueError, "capacity"):
            backend.encode_start(dict(spec={"request_id": "b"}))
        backend.encode_cancel(dict(ticket=first["ticket"]))
        second = backend.encode_start(dict(spec={"request_id": "b"}))
        self.futures[1].set_exception(ValueError("encoding failed"))
        with self.assertRaisesRegex(ValueError, "encoding failed"):
            backend.encode_poll(dict(ticket=second["ticket"]))
        self.assertFalse(backend.tickets)
        self.assertEqual(backend.ticket_peak, 1)

    def test_running_cancel_remains_counted_and_depth_zero_refuses_submission(self):
        backend = self.backend(1)
        first = backend.encode_start(dict(spec={"request_id": "a"}))
        self.futures[0].set_running_or_notify_cancel()
        result = backend.encode_cancel(dict(ticket=first["ticket"]))
        self.assertEqual(result["outstanding_tickets"], 1)
        with self.assertRaisesRegex(ValueError, "capacity"):
            backend.encode_start(dict(spec={"request_id": "b"}))
        self.futures[0].set_result(None)
        backend.encode_start(dict(spec={"request_id": "b"}))
        disabled = self.backend(0)
        with self.assertRaisesRegex(ValueError, "capacity"):
            disabled.encode_start(dict(spec={"request_id": "x"}))


if __name__ == "__main__":
    unittest.main()
