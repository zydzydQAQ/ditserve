"""Resident SP1 workers and byte-serialized CPU-only transactional transport."""

import copy
import hashlib
import multiprocessing
import os
from pathlib import Path
import pickle
import time
import traceback
from types import SimpleNamespace

from .common import file_digest


def correctness_state_digest(worker, state):
    """Value digest with only UniPC's set-derived bookkeeping canonicalized.

    The wire codec sorts this field because Python process hash seeds can alter
    its order. Copy the two metadata dictionaries before sorting; live scheduler
    configuration and ordered sampler histories remain untouched.
    """
    from .experiment import state_digest
    payload = dict(worker._payload(state))
    scheduler = dict(payload["scheduler"])
    config = dict(scheduler["_internal_dict"])
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    scheduler["_internal_dict"] = config
    payload["scheduler"] = scheduler
    view = SimpleNamespace(torch=worker.torch, _payload=lambda _: payload)
    return state_digest(view, state)


class RemoteError(RuntimeError):
    def __init__(self, value):
        super().__init__(value["error"])
        self.remote_type = value.get("error_type")
        self.traceback = value.get("traceback")


def _serve(connection, physical_device, backend, model_path, root, deadline, research_config=None,
           diagnostic_mode=False):
    os.environ.update(CUDA_VISIBLE_DEVICES=physical_device, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from ..gpu.worker import WanStepWorker
    from .dual_state import MigrationEndpoint
    from .prompt_bank import FrozenPromptBank
    from .state import state_inventory, refresh_certificate, assert_transfer_supported, DEPENDENCY_VERSION
    from .storage import snapshot_timestep_views
    import numpy as np

    worker = None
    try:
        worker = WanStepWorker(device="cuda:0", model_path=model_path, attention=backend)
        adapter = None
        if research_config is not None:
            from .cfg_adapter import CfgResearchAdapter
            from .cfg_policy import resolve_policy
            adapter = CfgResearchAdapter(worker)
            worker._cfg_research_config = research_config
            worker._cfg_policies = {}
        endpoint = MigrationEndpoint(worker, device_id=physical_device)
        diagnostics = None
        if diagnostic_mode:
            from .cache_off_diagnostics import CacheOffDiagnostics
            diagnostics = CacheOffDiagnostics(worker, physical_device, enabled=True)
        bank = FrozenPromptBank(worker)
        pending_artifacts = {}
        connection.send_bytes(pickle.dumps(dict(ok=True, result=dict(pid=os.getpid(), device=physical_device))))
        def state(request_id):
            return worker._states[request_id]

        def execute(command, args):
            nonlocal bank
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU lifecycle deadline reached")
            if command.startswith("diagnostic_"):
                if diagnostics is None:
                    raise ValueError("diagnostic RPCs require explicit diagnostic-only worker mode")
                methods = dict(diagnostic_signature=diagnostics.signature, diagnostic_status=diagnostics.status,
                    diagnostic_export=diagnostics.export,
                    diagnostic_import=diagnostics.import_copy, diagnostic_step=diagnostics.step,
                    diagnostic_drop=diagnostics.drop)
                if command not in methods:
                    raise ValueError("unknown diagnostic RPC")
                return methods[command](**args)
            if command == "install_encoded":
                if research_config is None or worker._states or bank.entries:
                    raise ValueError("CPU embedding import requires a fresh, explicitly enabled CFG research worker")
                from .cfg_embeddings import PreencodedPromptBank
                if isinstance(bank, PreencodedPromptBank):
                    raise ValueError("CPU embedding cache is already installed")
                replacement = PreencodedPromptBank(worker, research_config, args["cache_root"],
                    cache_manifest_sha256=args["cache_manifest_sha256"],
                    model_manifest_sha256=args["model_manifest_sha256"])
                bank = replacement
                return dict(cache_manifest_sha256=bank.cache_manifest_sha256, entries=len(bank.entries),
                            mode="preencoded_cpu_embeddings_mechanism_only")
            if command == "backend":
                if worker._states:
                    raise RuntimeError("backend may change only without live requests")
                if args["attention"] not in ("math", "flash", "efficient"):
                    raise ValueError("unknown native attention backend")
                worker.attention = args["attention"]
                return worker.attention
            if command == "warm":
                return bank.warm(args["spec"])
            if command == "prepare":
                policy = None
                if adapter is not None:
                    supplied = args.get("policy")
                    if not isinstance(supplied, dict):
                        raise ValueError("CFG request requires an explicit resolved policy")
                    policy = resolve_policy(research_config, supplied["id"], supplied["implementation"])
                    if supplied != policy or any(args["spec"][key] != policy[key]
                                                 for key in ("cache", "threshold", "max_continuous")):
                        raise ValueError("request policy or scientific spec mismatch")
                start = time.perf_counter()
                s = bank.prepare(args["spec"])
                if policy is not None:
                    worker._cfg_policies[s.request_id] = policy
                worker.torch.cuda.reset_peak_memory_stats(worker.device)
                return dict(request_id=s.request_id, prepare_ms=(time.perf_counter() - start) * 1000,
                            text_encoding=bank.warm(args["spec"]))
            if command == "advance":
                s = state(args["request_id"])
                traces, inventories = [], []
                while s.step_index < args["until"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("unfinished trajectory exceeded estimate")
                    if adapter is None:
                        worker.step(s)
                    else:
                        reference = state(args["reference_request_id"]) if args.get("reference_request_id") else None
                        if reference is not None:
                            if reference.step_index != s.step_index:
                                raise ValueError("diagnostic reference step mismatch")
                            adapter.step(reference, policy=worker._cfg_policies[reference.request_id])
                        adapter.step(s, policy=worker._cfg_policies[s.request_id],
                                     diagnostic=args.get("diagnostic", False), reference_state=reference)
                    if args.get("instrument"):
                        traces.append(dict(step=s.step_index, state_sha256=correctness_state_digest(worker, s)))
                    if s.step_index in (8, 16, 24, 32):
                        inventories.append(dict(step=s.step_index, state=state_inventory(s, torch_module=worker.torch),
                                                memory=worker._memory()))
                return dict(step=s.step_index, traces=traces, inventories=inventories,
                            events=copy.deepcopy(s.events))
            if command == "finish":
                s = state(args["request_id"])
                worker._check(s, "gpu")
                raw_latent = s.latents.detach().cpu().contiguous().numpy()
                latent_hash = hashlib.sha256(raw_latent.tobytes()).hexdigest()
                output = worker.finish(s)
                finished_at = time.monotonic()
                events = copy.deepcopy(s.events)
                peak = dict(allocated=worker.torch.cuda.max_memory_allocated(worker.device),
                            reserved=worker.torch.cuda.max_memory_reserved(worker.device))
                path = Path(root) / "arrays" / (s.request_id + "-video.npy")
                result = dict(request_id=s.request_id, output_path=str(path.relative_to(root)),
                              latent_sha256=latent_hash, events=events, peak_memory_bytes=peak,
                              decoded_at_monotonic=finished_at, status="completed", artifact_io_ms=0., artifacts_pending=True)
                pending_artifacts[s.request_id] = (output, raw_latent if args.get("retain_latent") else None)
                worker.release(s)
                if adapter is not None:
                    worker._cfg_policies.pop(s.request_id, None)
                if not args.get("defer_artifacts"):
                    result.update(execute("flush", dict(request_ids=[s.request_id]))[s.request_id])
                return result
            if command == "flush":
                results = {}
                for request_id in args.get("request_ids", list(pending_artifacts)):
                    output, latent = pending_artifacts[request_id]
                    started = time.perf_counter()
                    path = Path(root) / "arrays" / (request_id + "-video.npy")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, output, allow_pickle=False)
                    item = dict(output_sha256=file_digest(path), artifacts_pending=False)
                    if latent is not None:
                        latent_path = path.with_name(request_id + "-latent.npy")
                        np.save(latent_path, latent, allow_pickle=False)
                        item.update(latent_path=str(latent_path.relative_to(root)), latent_file_sha256=file_digest(latent_path))
                    item["artifact_io_ms"] = (time.perf_counter() - started) * 1000
                    results[request_id] = item
                    del pending_artifacts[request_id]
                return results
            if command == "certificate":
                s = state(args["request_id"])
                return refresh_certificate(s, layout=dict(sp_size=1, ranks=[0], device=str(s.latents.device), version=1),
                                           dependency_version=DEPENDENCY_VERSION, torch_module=worker.torch)
            if command == "pause_export":
                s = state(args.pop("request_id"))
                return endpoint.pause_export(s, **args)
            if command == "local_roundtrip":
                from .dual_state import local_roundtrip
                s = state(args["request_id"])
                certificate = execute("certificate", dict(request_id=s.request_id)) if args["action"] == "certified_omit" else None
                restored, evidence = local_roundtrip(worker, s, action=args["action"],
                    migration_id=args.get("migration_id", s.request_id + "-local-roundtrip"), certificate=certificate)
                return dict(request_id=restored.request_id, **evidence)
            if command in ("migration_prepare", "acknowledge", "release_source", "commit", "abort"):
                method = "prepare" if command == "migration_prepare" else command
                value = getattr(endpoint, method)(**args)
                return dict(request_id=value.request_id) if command == "commit" else value
            if command == "checkpoint":
                s = state(args["request_id"])
                from .dual_state import pack_request
                with worker._lock:
                    manifest, tensors = pack_request(worker, s, action="full_migrate", target_device=physical_device,
                                                      migration_id=args["request_id"] + "-checkpoint")
                path = Path(root) / "arrays" / (s.request_id + "-state.pkl")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    pickle.dump((manifest, tensors), stream, protocol=5)
                return dict(state_path=str(path.relative_to(root)), state_sha256=file_digest(path),
                            checkpoint_schema=manifest["schema"], checkpoint_state_sha256=manifest["source_state_sha256"],
                            checkpoint_manifest_sha256=manifest["sha256"])
            if command == "release":
                worker.release(state(args["request_id"]))
                if adapter is not None:
                    worker._cfg_policies.pop(args["request_id"], None)
                return None
            if command == "offload":
                s = state(args["request_id"])
                with worker._lock:
                    worker._check(s, "gpu")
                    inventory = state_inventory(s, torch_module=worker.torch)
                    assert_transfer_supported(inventory)
                    return snapshot_timestep_views(worker, s, "hot", inventory)
            if command == "restore":
                from .storage import restore_timestep_views
                s = state(args["request_id"])
                with worker._lock:
                    worker._check(s, "cpu")
                    if s.pending_snapshot is None or not s.pending_snapshot.get("research_storage_codec"):
                        raise ValueError("restore requires the matching known-storage snapshot")
                    return restore_timestep_views(worker, s)
            if command == "reclaim":
                from .state import validate_refresh_certificate
                s = state(args["request_id"])
                with worker._lock:
                    worker._check(s, "gpu")
                    inventory = state_inventory(s, torch_module=worker.torch)
                    assert_transfer_supported(inventory)
                    if inventory["shared_role_bytes"]:
                        raise ValueError("reclaim cannot drop shared feature storage")
                    cert = execute("certificate", args)
                    validate_refresh_certificate(s, cert,
                        layout=dict(sp_size=1, ranks=[0], device=str(s.latents.device), version=1),
                        dependency_version=DEPENDENCY_VERSION, torch_module=worker.torch)
                    for context in s.contexts.values():
                        context.buffers.clear()
                return cert
            if command == "batch":
                results = []
                for spec in args["specs"]:
                    start = time.perf_counter()
                    prepare_args = dict(spec=spec)
                    if adapter is not None:
                        prepare_args["policy"] = resolve_policy(research_config, "off")
                    execute("prepare", prepare_args)
                    progress = execute("advance", dict(request_id=spec["request_id"], until=32))
                    result = execute("finish", dict(request_id=spec["request_id"], defer_artifacts=True))
                    result.update(elapsed_ms=(time.perf_counter() - start) * 1000 - result["artifact_io_ms"],
                                  inventories=progress["inventories"])
                    results.append(result)
                return results
            if command == "memory":
                return worker._memory()
            raise ValueError("unknown worker command: " + command)

        while True:
            command, args = pickle.loads(connection.recv_bytes())
            if command == "close":
                break
            try:
                result = execute(command, args)
                connection.send_bytes(pickle.dumps(dict(ok=True, result=result), protocol=5))
            except Exception as error:
                traceback.print_exc()
                connection.send_bytes(pickle.dumps(dict(ok=False, error=str(error), error_type=type(error).__name__,
                                                        traceback=traceback.format_exc()), protocol=5))
    except BaseException as error:
        traceback.print_exc()
        try:
            connection.send_bytes(pickle.dumps(dict(ok=False, error=str(error), error_type=type(error).__name__,
                                                    traceback=traceback.format_exc())))
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ResidentWorker:
    def __init__(self, device, backend, model_path, root, deadline, *, research_config=None,
                 diagnostic_mode=False):
        if type(diagnostic_mode) is not bool or diagnostic_mode and research_config is None:
            raise ValueError("diagnostic mode requires an explicit research configuration")
        ctx = multiprocessing.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.device, self.backend, self.deadline = device, backend, deadline
        self.process = ctx.Process(target=_serve, args=(child, device, backend, str(model_path), str(root), deadline,
                                                        research_config, diagnostic_mode))
        self.process.start()
        child.close()
        self.pending = True
        try:
            self.result()
        except BaseException:
            self.close()
            raise

    def start(self, command, **args):
        if self.pending:
            raise RuntimeError("resident worker already has an outstanding operation")
        self.connection.send_bytes(pickle.dumps((command, args), protocol=5))
        self.pending = True

    def result(self):
        if not self.pending:
            raise RuntimeError("no outstanding operation")
        while not self.connection.poll(.2):
            if time.monotonic() >= self.deadline or not self.process.is_alive():
                raise RuntimeError("resident worker died or exceeded lifecycle deadline")
        value = pickle.loads(self.connection.recv_bytes())
        self.pending = False
        if not value["ok"]:
            raise RemoteError(value)
        return value["result"]

    def call(self, command, **args):
        self.start(command, **args)
        result = self.result()
        if command == "backend":
            self.backend = result
        return result

    def close(self):
        if self.process.is_alive():
            if not self.pending:
                try:
                    self.connection.send_bytes(pickle.dumps(("close", {})))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)
        self.connection.close()
        if self.process.is_alive():
            raise RuntimeError("resident GPU worker cleanup unconfirmed")


def transfer(source, target, request_id, action, migration_id, certificate=None, epoch=0):
    start = time.perf_counter()
    manifest, tensors, timing = source.call("pause_export", request_id=request_id, action=action,
                                           target_device=target.device, migration_id=migration_id,
                                           certificate=certificate, epoch=epoch)
    released, prepared = False, False
    phases = {}
    try:
        phase = time.perf_counter()
        ack = target.call("migration_prepare", manifest=manifest, tensors=tensors)
        prepared = True
        phases["target_ipc_prepare_ack_ms"] = (time.perf_counter() - phase) * 1000
        phase = time.perf_counter()
        source.call("acknowledge", migration_id=migration_id, acknowledgement=ack)
        receipt = source.call("release_source", migration_id=migration_id)
        released = True
        phases["source_ack_release_ms"] = (time.perf_counter() - phase) * 1000
        phase = time.perf_counter()
        target.call("commit", migration_id=migration_id, release_receipt=receipt)
        phases["target_commit_ms"] = (time.perf_counter() - phase) * 1000
    except BaseException as error:
        cleanup_errors = []
        # Prepared copies are never runnable; discard one even if commit failed
        # after the source had been released. A lost ACK may also leave a copy.
        try:
            target.call("abort", migration_id=migration_id)
        except Exception as cleanup_error:
            if prepared:
                cleanup_errors.append("target: " + str(cleanup_error))
        if not released:
            try:
                source.call("abort", migration_id=migration_id)
            except Exception as cleanup_error:
                cleanup_errors.append("source: " + str(cleanup_error))
        if cleanup_errors:
            raise RuntimeError("migration failed with incomplete rollback: " + "; ".join(cleanup_errors)) from error
        raise
    return dict(transfer_ms=(time.perf_counter() - start) * 1000, export=timing,
                phases=phases, migration_id=migration_id, action=action,
                state_sha256=manifest["source_state_sha256"], manifest_sha256=manifest["sha256"],
                transfer_bytes=manifest["transfer_bytes"], omitted_feature_bytes=manifest["omitted_feature_bytes"],
                ownership_epoch=epoch + 1, transport="D2H_CPU_pickle_bytes_IPC_H2D_ACK_release_commit",
                context_name_mapping=ack["context_name_mapping"],
                byte_exact_before_first_compute=True,
                exactness_scope="all transported tensors; omitted features are absent and not compared")
