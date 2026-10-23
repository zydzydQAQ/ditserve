"""Experimental two-rank handoff; tensors never travel through the JSON RPC.

Sources remain frozen and resident until every destination has acknowledged
activation. A failed collective after commit makes the world unusable: callers
must terminate the cell, rather than retrying an ambiguous model operation.
"""

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .parallel import FixedLayout, LayoutHandle, ParallelContractError, pack_request, unpack_request


@dataclasses.dataclass
class ElasticRequest:
    handle: object
    quantum_seq: int = 0
    status: str = "active"
    last_result: object = None


class ElasticRequests:
    transport = "host_file_v1"

    def __init__(self, backend):
        self.backend = backend
        self.rank = backend.parallel.groups.rank
        self.records = {}
        self.transactions = {}
        self.failed = False

    def register(self, key):
        if key in self.records:
            raise ParallelContractError("duplicate elastic request")
        self.records[key] = ElasticRequest(LayoutHandle(self.rank))

    def fail_world(self):
        self.failed = True
        for record in self.records.values():
            record.handle.fail()
            record.status = "failed"

    def describe(self, key):
        record = self.records[key]
        return dict(group=list(record.handle.current.ranks),
                    ownership_epoch=record.handle.current.epoch,
                    quantum_seq=record.quantum_seq, elastic_status=record.status)

    def before_advance(self, request):
        if self.failed:
            raise ParallelContractError("elastic world failed; terminate the cell")
        record = self.records[request["state_id"]]
        if record.status != "active" or record.handle.failed:
            raise ParallelContractError("request is frozen, staged, or failed")
        epoch = request.get("expected_epoch", record.handle.current.epoch)
        sequence = request.get("quantum_seq", record.quantum_seq)
        if type(epoch) is not int or epoch != record.handle.current.epoch:
            raise ParallelContractError("stale request ownership epoch")
        if type(sequence) is not int or sequence < 0:
            raise ParallelContractError("invalid quantum sequence")
        signature = {key: request.get(key) for key in ("quantum", "blocks", "steps", "observe")}
        if sequence == record.quantum_seq - 1 and record.last_result:
            previous, result = record.last_result
            if signature != previous:
                raise ParallelContractError("replayed quantum changed its contents")
            return record, sequence, result
        if sequence != record.quantum_seq:
            raise ParallelContractError("stale or skipped quantum sequence")
        return record, sequence, None

    def after_advance(self, request, result):
        record = self.records[request["state_id"]]
        record.quantum_seq += 1
        result["progress"].update(self.describe(request["state_id"]))
        record.last_result = ({key: request.get(key) for key in ("quantum", "blocks", "steps", "observe")}, result)

    @staticmethod
    def _groups(request):
        source = tuple(request.get("source_group", request.get("from_group", ())))
        target = tuple(request.get("target_group", request.get("to_group", ())))
        allowed = ((0,), (1,), (0, 1))
        if source not in allowed or target not in allowed:
            raise ParallelContractError("elastic v1 supports aligned groups on two ranks only")
        return source, target

    def _transaction(self, request, *, create=False):
        identity = request.get("transfer_id")
        key = request.get("state_id")
        epoch = request.get("expected_epoch")
        if not isinstance(identity, str) or not identity or not isinstance(key, str) or not key:
            raise ParallelContractError("transfer_id and state_id must be nonempty")
        if type(epoch) is not int or epoch < 0:
            raise ParallelContractError("transfer expected_epoch must be nonnegative")
        source, target = self._groups(request)
        if self.rank not in set(source) | set(target):
            raise ParallelContractError("rank is outside transfer participants")
        expected = dict(state_id=key, expected_epoch=epoch, source_group=list(source),
                        target_group=list(target), path=str(Path(request["path"]).resolve()),
                        execution_contract=dict(schema="ditserve.wan-sp-execution.v1",
                            sp_linear_policy=getattr(self.backend.parallel, "linear_policy", "native")))
        fingerprint = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        transaction = self.transactions.get(identity)
        if transaction is not None:
            if transaction["fingerprint"] != fingerprint:
                raise ParallelContractError("transfer ID reused with different contents")
            return transaction
        if not create:
            raise ParallelContractError("unknown transfer ID")
        transaction = dict(expected, transfer_id=identity, fingerprint=fingerprint, phase="new",
                           target_epoch=epoch + 1, source_ready=False, target_ready=False,
                           committed=False, staged=None, result=None)
        self.transactions[identity] = transaction
        return transaction

    def _summary(self, transaction):
        if transaction["phase"] == "finalized" and transaction["result"] is not None:
            return copy.deepcopy(transaction["result"])
        key = transaction["state_id"]
        result = dict(transfer_id=transaction["transfer_id"], state_id=key, rank=self.rank,
                      phase=transaction["phase"], transport=self.transport,
                      group=transaction["target_group"],
                      ownership_epoch=transaction["target_epoch"],
                      source_ready=transaction["source_ready"],
                      target_ready=transaction["target_ready"],
                      committed=transaction["committed"])
        if key in self.backend.states and key in self.records:
            result["progress"] = self.backend._progress(self.backend.states[key])
        else:
            result["progress"] = None
        for name in ("path", "sha256", "state_sha256", "bytes", "quantum_seq", "artifacts", "execution_contract"):
            if name in transaction:
                result[name] = transaction[name]
        return result

    def source_prepare(self, request):
        tx = self._transaction(request, create=True)
        if tx["phase"] == "aborted":
            raise ParallelContractError("aborted transfer needs a new ID")
        if tx["source_ready"]:
            return self._summary(tx)
        key = tx["state_id"]
        if self.rank not in tx["source_group"]:
            raise ParallelContractError("only source ranks can freeze a request")
        record = self.records[key]
        if (record.status != "active" or record.handle.failed or record.handle.active is not None
                or list(record.handle.current.ranks) != tx["source_group"]
                or record.handle.current.epoch != tx["expected_epoch"]):
            raise ParallelContractError("source ownership/layout does not match transfer")
        state = self.backend._boundary(dict(state_id=key))
        if state.request_id != key:
            raise ParallelContractError("state key differs from mathematical request identity")
        self.backend.runtime._sync()
        manifest, tensors = pack_request(self.backend.runtime, state, allow_completed=True)
        tx["state_sha256"] = manifest["state_sha256"]
        tx["quantum_seq"] = record.quantum_seq
        record.status = "frozen_source"
        tx["source_ready"], tx["phase"] = True, "source_prepared"
        try:
            if request.get("write", True):
                destination = Path(tx["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".handoff-", delete=False) as stream:
                        temporary = Path(stream.name)
                        self.backend.runtime.torch.save(dict(manifest=manifest, tensors=tensors,
                            execution_contract=tx["execution_contract"]), stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Exclusive publication: never replace a previous attempt.
                    os.link(temporary, destination)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                from ..base import file_hash
                tx.update(sha256=file_hash(destination), bytes=destination.stat().st_size)
                tx["artifacts"] = [dict(path=str(destination), sha256=tx["sha256"],
                                        bytes=tx["bytes"], kind="raw")]
            return self._summary(tx)
        except BaseException:
            record.status = "active"
            tx["phase"] = "aborted"
            raise

    def target_prepare(self, request):
        tx = self._transaction(request, create=True)
        if tx["phase"] == "aborted":
            raise ParallelContractError("aborted transfer needs a new ID")
        if request.get("execution_contract", tx["execution_contract"]) != tx["execution_contract"]:
            raise ParallelContractError("transfer execution contract mismatch")
        if tx["target_ready"]:
            if (request.get("state_sha256") != tx["state_sha256"]
                    or request.get("quantum_seq") != tx["quantum_seq"]):
                raise ParallelContractError("replayed target preparation changed its contents")
            return self._summary(tx)
        if self.rank not in tx["target_group"]:
            raise ParallelContractError("only destination ranks can stage a request")
        key = tx["state_id"]
        state_digest = request.get("state_sha256")
        if not isinstance(state_digest, str) or not state_digest:
            raise ParallelContractError("destination requires mathematical state digest")
        sequence = request.get("quantum_seq")
        if type(sequence) is not int or sequence < 0:
            raise ParallelContractError("destination requires quantum sequence")
        if key in self.records:
            record = self.records[key]
            if (not tx["source_ready"] or record.status != "frozen_source"
                    or tx["state_sha256"] != state_digest or record.quantum_seq != sequence):
                raise ParallelContractError("destination already owns a different request")
        else:
            if key in self.backend.states or key in self.backend.runtime._states:
                raise ParallelContractError("destination request is already registered")
            from ..base import file_hash
            if not request.get("sha256") or file_hash(tx["path"]) != request["sha256"]:
                raise ParallelContractError("transfer file checksum mismatch")
            payload = self.backend.runtime.torch.load(tx["path"], map_location="cpu", weights_only=True)
            if payload.get("execution_contract") != tx["execution_contract"]:
                raise ParallelContractError("transfer packet execution contract mismatch")
            if payload["manifest"]["state_sha256"] != state_digest:
                raise ParallelContractError("mathematical state digest mismatch")
            state = unpack_request(self.backend.runtime, payload["manifest"], payload["tensors"])
            if state.request_id != key:
                raise ParallelContractError("transported request identity mismatch")
            # Remains unregistered and cannot be advanced before commit.
            tx["staged"] = state
        tx.update(state_sha256=state_digest, quantum_seq=sequence, target_ready=True, phase="target_prepared")
        return self._summary(tx)

    def commit(self, request):
        tx = self._transaction(request)
        if tx["committed"]:
            return self._summary(tx)
        if tx["phase"] == "aborted":
            raise ParallelContractError("cannot commit an aborted transfer")
        key = tx["state_id"]
        if self.rank in tx["target_group"]:
            if not tx["target_ready"]:
                raise ParallelContractError("destination has not acknowledged its state")
            if key not in self.records:
                state = tx["staged"]
                if state is None:
                    raise ParallelContractError("missing staged request")
                runtime = self.backend.runtime
                runtime._states[key] = state
                self.backend.states[key] = state
                try:
                    if state.step_index < state.spec["steps"]:
                        self.backend.continuations[key] = self.backend.executor.attach(state)
                    handle = LayoutHandle(self.rank, tx["target_group"])
                    handle.current = FixedLayout(tuple(tx["target_group"]), tx["target_epoch"])
                    self.records[key] = ElasticRequest(handle, tx["quantum_seq"])
                except BaseException:
                    self.backend.states.pop(key, None)
                    runtime._states.pop(key, None)
                    raise
                tx["staged"] = None
            else:
                record = self.records[key]
                record.handle.commit(tx["target_group"], expected_epoch=tx["expected_epoch"])
                record.status, record.last_result = "active", None
        else:
            if not tx["source_ready"]:
                raise ParallelContractError("leaving rank was not frozen")
            self.records[key].status = "committed_leaving"
        tx["committed"], tx["phase"] = True, "committed"
        return self._summary(tx)

    def abort(self, request):
        tx = self._transaction(request)
        if tx["committed"]:
            raise ParallelContractError("cannot roll back committed ownership; abort the world")
        key = tx["state_id"]
        if tx["source_ready"] and key in self.records:
            self.records[key].status = "active"
        tx["staged"], tx["phase"] = None, "aborted"
        return self._summary(tx)

    def finalize(self, request):
        tx = self._transaction(request)
        if tx["phase"] == "finalized":
            return self._summary(tx)
        if not tx["committed"]:
            raise ParallelContractError("cannot release source before committed activation")
        receipts = request.get("activation_receipts", [])
        if (len(receipts) != len(tx["target_group"])
                or {receipt.get("rank") for receipt in receipts} != set(tx["target_group"])
                or any(receipt.get("transfer_id") != tx["transfer_id"]
                       or receipt.get("state_id") != tx["state_id"]
                       or receipt.get("ownership_epoch") != tx["target_epoch"]
                       or receipt.get("state_sha256") != tx.get("state_sha256")
                       or receipt.get("execution_contract") != tx["execution_contract"]
                       or not receipt.get("committed")
                       or not receipt.get("target_ready")
                       for receipt in receipts)):
            raise ParallelContractError("source release requires every destination activation receipt")
        key = tx["state_id"]
        if self.rank not in tx["target_group"]:
            self.backend._release_state(key)
        tx["phase"] = "finalized"
        tx["result"] = self._summary(tx)
        return copy.deepcopy(tx["result"])

    def dispatch(self, request):
        action = request.get("action")
        if action == "status":
            return self._summary(self._transaction(request))
        if action not in ("source_prepare", "target_prepare", "commit", "abort", "finalize"):
            raise ParallelContractError("unknown transfer action")
        if self.failed:
            raise ParallelContractError("elastic world failed; terminate the cell")
        return getattr(self, action)(request)

    def transition(self, request):
        """Collective convenience RPC called concurrently on the old/new union."""
        if self.failed:
            raise ParallelContractError("elastic world failed; terminate the cell")
        source, target = self._groups(request)
        key = request["state_id"]
        if source == target:
            record = self.records[key]
            if record.handle.current.epoch != request["expected_epoch"]:
                raise ParallelContractError("stale unchanged-group transition")
            return dict(state_id=key, **self.describe(key),
                        transfer_ready=self.backend._progress(self.backend.states[key])["transfer_ready"],
                        progress=self.backend._progress(self.backend.states[key]), unchanged=True)
        union = tuple(sorted(set(source) | set(target)))
        group = self.backend.parallel.groups.group(union)
        torch = self.backend.runtime.torch
        committed = False
        tx = None

        def phase(name, operation):
            try:
                value = operation()
                local = dict(rank=self.rank, phase=name, ok=True, value=value)
            except Exception as error:
                local = dict(rank=self.rank, phase=name, ok=False,
                             error=type(error).__name__ + ": " + str(error))
            peers = [None] * len(union)
            try:
                torch.distributed.all_gather_object(peers, local, group=group)
            except BaseException:
                self.fail_world()
                raise
            if any(row["phase"] != name or not row["ok"] for row in peers):
                raise ParallelContractError("handoff %s failed: %s" % (name, peers))
            return peers

        try:
            identities = phase("identity", lambda: self._transaction(request, create=True)["fingerprint"])
            if len({row["value"] for row in identities}) != 1:
                raise ParallelContractError("ranks disagree on handoff identity")
            tx = self._transaction(request)
            # A completed transition can be acknowledged without replaying it.
            statuses = phase("status", lambda: tx["phase"])
            if all(row["value"] == "finalized" for row in statuses):
                return self._summary(tx)
            if any(row["value"] != "new" for row in statuses):
                raise ParallelContractError("incomplete collective handoff cannot be replayed")
            prepared = phase("source_prepare", lambda: self.source_prepare(dict(request, write=self.rank == source[0]))
                             if self.rank in source else None)
            sources = [row["value"] for row in prepared if row["rank"] in source]
            if len({(row["state_sha256"], row["quantum_seq"]) for row in sources}) != 1:
                raise ParallelContractError("source replicas disagree on mathematical state")
            leader = next(row["value"] for row in prepared if row["rank"] == source[0])
            incoming = dict(request, sha256=leader["sha256"],
                            state_sha256=leader["state_sha256"], quantum_seq=leader["quantum_seq"],
                            execution_contract=leader["execution_contract"])
            phase("target_prepare", lambda: self.target_prepare(incoming) if self.rank in target else None)
            # Any failure at or after this point may have activated a destination.
            committed = True
            acknowledgments = phase("commit", lambda: self.commit(request))
            receipts = [row["value"] for row in acknowledgments if row["rank"] in target]
            phase("finalize", lambda: self.finalize(dict(request, activation_receipts=receipts)))
            result = self._summary(tx)
            result["transfer_ready"] = True
            return result
        except BaseException:
            if committed:
                self.fail_world()
            elif tx is not None and not tx["committed"]:
                self.abort(request)
            raise
