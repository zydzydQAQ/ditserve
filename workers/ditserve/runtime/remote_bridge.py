"""Multi-node TCP RPC, doorbell, and artifact bridge for 16-GPU DitServe execution.

Connects the C++ controller on Node 0 (`remote_worker_proxy`) to GPU worker
processes on Node 1 (`remote_worker_daemon`) over low-latency `TCP_NODELAY`
sockets:
1. Line-oriented JSON RPC forwarding (`stdin` <-> TCP <-> `stdout`).
2. Sub-millisecond doorbell counter mirroring (`8-byte uint64` mmap file on
   Node 0 -> TCP frame -> local `/dev/shm` doorbell file on Node 1).
3. Cross-node `/dev/shm` `ProactiveEncoderPool` prompt slab synchronization,
   on-demand `block_import` `.pt` state transfer, and output artifact syncing.
4. Remote GPU occupancy probing (`nvml` and `nvidia_smi`) with automatic
   Node 0 proxy PID/PGID -> Node 1 worker PID/PGID translation and synchronous
   teardown verification.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


HEADER_STRUCT = struct.Struct("!II")
DOORBELL_STRUCT = struct.Struct("<Q")


def _set_nodelay(sock: socket.socket) -> None:
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(min(65536, remaining))
        if not chunk:
            raise EOFError("socket closed during frame read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(
    sock: socket.socket,
    lock: threading.Lock,
    header: dict[str, Any],
    blob: bytes = b"",
) -> None:
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = HEADER_STRUCT.pack(len(raw_header), len(blob))
    with lock:
        sock.sendall(prefix + raw_header + blob)


def recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    prefix = _recv_exact(sock, HEADER_STRUCT.size)
    header_len, blob_len = HEADER_STRUCT.unpack(prefix)
    raw_header = _recv_exact(sock, header_len)
    blob = _recv_exact(sock, blob_len) if blob_len > 0 else b""
    return json.loads(raw_header.decode("utf-8")), blob


def _atomic_write_bytes(path_str: str, data: bytes) -> None:
    target = Path(path_str)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    tmp.write_bytes(data)
    os.replace(tmp, target)


def _read_doorbell_u64(path_str: str) -> int:
    try:
        with open(path_str, "rb") as f:
            raw = f.read(8)
        if len(raw) == 8:
            return DOORBELL_STRUCT.unpack(raw)[0]
    except OSError:
        pass
    return 0


def _write_doorbell_fd(fd: int, value: int) -> None:
    os.pwrite(fd, DOORBELL_STRUCT.pack(int(value) & 0xFFFFFFFFFFFFFFFF), 0)


def _extract_shm_dir(config: dict[str, Any] | None) -> Path | None:
    if not isinstance(config, dict):
        return None
    opts = config.get("options")
    if not isinstance(opts, dict):
        return None
    pool = opts.get("encoder_pool")
    if not isinstance(pool, dict):
        return None
    shm_dir = pool.get("shm_dir")
    if isinstance(shm_dir, str) and shm_dir and shm_dir != "auto":
        return Path(shm_dir)
    return None


def _collect_artifact_paths(result: Any) -> list[str]:
    paths: list[str] = []
    if not isinstance(result, dict):
        return paths

    def _add_obj(obj: Any) -> None:
        if isinstance(obj, dict) and isinstance(obj.get("path"), str):
            paths.append(obj["path"])

    for item in result.get("artifacts") or ():
        _add_obj(item)
    _add_obj(result.get("output"))
    _add_obj(result.get("latent"))
    for sub in result.get("results") or ():
        if isinstance(sub, dict):
            for item in sub.get("artifacts") or ():
                _add_obj(item)
            _add_obj(sub.get("output"))
            _add_obj(sub.get("latent"))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, sep, port_str = endpoint.rpartition(":")
    if not sep or not host or not port_str.isdigit():
        raise ValueError(f"invalid remote endpoint {endpoint!r}; expected host:port")
    return host, int(port_str)


def ping_daemon(endpoint: str, *, timeout: float = 3.0) -> dict[str, Any]:
    host, port = parse_endpoint(endpoint)
    lock = threading.Lock()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        _set_nodelay(sock)
        send_frame(sock, lock, {"_bridge_op": "ping"})
        reply, _ = recv_frame(sock)
        return reply


def release_daemon(endpoint: str, *, timeout: float = 10.0) -> dict[str, Any]:
    host, port = parse_endpoint(endpoint)
    lock = threading.Lock()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        _set_nodelay(sock)
        send_frame(sock, lock, {"_bridge_op": "release_daemon"})
        reply, _ = recv_frame(sock)
        return reply


def query_remote_occupancy(
    endpoint: str,
    *,
    occupancy_backend: str,
    gpu_uuids: list[str],
    allowed_pids: list[int],
    allowed_pgids: list[int],
    timeout: float = 10.0,
) -> dict[str, Any]:
    host, port = parse_endpoint(endpoint)
    lock = threading.Lock()
    with socket.create_connection((host, port), timeout=timeout) as sock:
        _set_nodelay(sock)
        send_frame(
            sock,
            lock,
            {
                "_bridge_op": "occupancy",
                "occupancy_backend": occupancy_backend,
                "gpu_uuids": list(gpu_uuids),
                "allowed_pids": [int(x) for x in allowed_pids],
                "allowed_pgids": [int(x) for x in allowed_pgids],
            },
        )
        reply, _ = recv_frame(sock)
        return reply


# ==============================================================================
# Node 0 Proxy (`remote_worker_proxy`)
# ==============================================================================


def run_proxy(
    *,
    backend: str | None = None,
    config_path: str | None = None,
    target_script: str | None = None,
    target_argv: list[str] | None = None,
    endpoint: str | None = None,
) -> int:
    endpoint = endpoint or os.environ.get("DITSERVE_REMOTE_ENDPOINT", "")
    if not endpoint:
        raise RuntimeError("DITSERVE_REMOTE_ENDPOINT is required for remote_worker_proxy")
    host, port = parse_endpoint(endpoint)

    config_dict: dict[str, Any] = {}
    if config_path and Path(config_path).is_file():
        config_dict = json.loads(Path(config_path).read_text(encoding="utf-8"))

    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", config_dict.get("gpu_uuid", ""))
    slot_str = os.environ.get("DITSERVE_REMOTE_SLOT", "0")
    slot = int(slot_str) if slot_str.lstrip("-").isdigit() else 0
    target_script = target_script if target_script is not None else os.environ.get("DITSERVE_REMOTE_TARGET_SCRIPT", "")
    forwarded_env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(
            (
                "DITSERVE_",
                "HF_",
                "TORCH",
                "TRITON_",
                "WAN_",
                "PYTHONPATH",
                "PYTHONDONTWRITEBYTECODE",
                "TRANSFORMERS_",
            )
        )
        and k
        not in (
            "DITSERVE_REMOTE_WORKER",
            "DITSERVE_REMOTE_ENDPOINT",
            "DITSERVE_REMOTE_SLOT",
            "DITSERVE_REMOTE_TARGET_SCRIPT",
        )
    }

    sock = socket.create_connection((host, port), timeout=30.0)
    _set_nodelay(sock)
    sock.settimeout(None)
    send_lock = threading.Lock()
    stop_event = threading.Event()

    handshake = {
        "_bridge_op": "spawn_worker",
        "backend": backend or config_dict.get("backend", "mock"),
        "config": config_dict,
        "gpu_uuid": gpu_uuid,
        "slot": slot,
        "client_pid": os.getpid(),
        "client_pgid": os.getpgrp(),
        "python": sys.executable,
        "target_script": target_script,
        "target_argv": list(target_argv or []),
        "env": forwarded_env,
    }
    env_doorbell = os.environ.get("DITSERVE_DOORBELL_FILE", "")
    if env_doorbell:
        handshake["env_doorbell_initial"] = _read_doorbell_u64(env_doorbell)

    send_frame(sock, send_lock, handshake)
    ack, _ = recv_frame(sock)
    if not ack.get("ok"):
        raise RuntimeError(f"remote_worker_daemon failed to spawn worker: {ack}")

    # Optional continuous watcher for DITSERVE_DOORBELL_FILE env var
    def _watch_doorbell_file(bell_path: str, cancel_evt: threading.Event, initial_val: int) -> None:
        last_val = initial_val
        while not cancel_evt.is_set() and not stop_event.is_set():
            cur_val = _read_doorbell_u64(bell_path)
            if cur_val != last_val:
                last_val = cur_val
                try:
                    send_frame(sock, send_lock, {"_bridge_op": "doorbell", "value": cur_val})
                except OSError:
                    return
            time.sleep(0.001)

    env_bell_thread: threading.Thread | None = None
    if env_doorbell:
        env_bell_thread = threading.Thread(
            target=_watch_doorbell_file,
            args=(env_doorbell, stop_event, handshake.get("env_doorbell_initial", 0)),
            daemon=True,
        )
        env_bell_thread.start()

    shm_dir = _extract_shm_dir(config_dict)
    sent_shm_files: set[tuple[str, int, int]] = set()

    def _sync_local_shm_to_remote() -> None:
        if shm_dir is None or not shm_dir.is_dir():
            return
        for p in sorted(shm_dir.glob("*.pt")):
            try:
                st = p.stat()
                ident = (str(p), st.st_size, st.st_mtime_ns)
                if ident in sent_shm_files:
                    continue
                raw = p.read_bytes()
                send_frame(sock, send_lock, {"_bridge_op": "file_put", "path": str(p)}, raw)
                sent_shm_files.add(ident)
            except OSError:
                continue

    def _handle_sigterm(_signum: int, _frame: Any) -> None:
        stop_event.set()
        try:
            send_frame(sock, send_lock, {"_bridge_op": "close"})
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        for raw_line in sys.stdin:
            if not raw_line.strip():
                continue
            try:
                req = json.loads(raw_line)
            except json.JSONDecodeError:
                req = None

            op = req.get("op") if isinstance(req, dict) else None
            if op in ("prepare", "advance_batch", "generate_batch"):
                _sync_local_shm_to_remote()

            call_cancel = threading.Event()
            call_watcher: threading.Thread | None = None
            frame_msg: dict[str, Any] = {"_bridge_op": "rpc", "raw_line": raw_line.rstrip("\n")}
            if isinstance(req, dict) and op == "advance" and isinstance(req.get("doorbell"), str):
                bell_path = req["doorbell"]
                init_val = _read_doorbell_u64(bell_path)
                frame_msg["doorbell_initial_value"] = init_val
                call_watcher = threading.Thread(
                    target=_watch_doorbell_file,
                    args=(bell_path, call_cancel, init_val),
                    daemon=True,
                )
                call_watcher.start()

            send_frame(sock, send_lock, frame_msg)

            try:
                while True:
                    reply_hdr, reply_blob = recv_frame(sock)
                    rop = reply_hdr.get("_bridge_op")
                    if rop == "file_put":
                        _atomic_write_bytes(reply_hdr["path"], reply_blob)
                    elif rop == "need_file":
                        req_path = reply_hdr["path"]
                        try:
                            data = Path(req_path).read_bytes()
                            send_frame(
                                sock,
                                send_lock,
                                {"_bridge_op": "file_put", "path": req_path, "found": True},
                                data,
                            )
                        except OSError as exc:
                            send_frame(
                                sock,
                                send_lock,
                                {"_bridge_op": "file_put", "path": req_path, "found": False, "error": str(exc)},
                            )
                    elif rop == "rpc_response":
                        out_line = reply_hdr["line"]
                        sys.stdout.write(out_line if out_line.endswith("\n") else out_line + "\n")
                        sys.stdout.flush()
                        break
                    else:
                        raise RuntimeError(f"unexpected frame from remote_worker_daemon: {reply_hdr}")
            finally:
                call_cancel.set()
                if call_watcher is not None:
                    call_watcher.join(timeout=0.05)

            if op == "shutdown":
                break
    except EOFError:
        pass
    finally:
        stop_event.set()
        try:
            send_frame(sock, send_lock, {"_bridge_op": "close"})
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
    return 0


# ==============================================================================
# Node 1 Daemon (`remote_worker_daemon`)
# ==============================================================================


def _detect_host_ip() -> str:
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=5)
        for token in out.split():
            if token and not token.startswith("127.") and ":" not in token:
                return token
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def _query_local_gpus() -> tuple[list[str], list[str]]:
    uuids: list[str] = []
    names: list[str] = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        for line in out.splitlines():
            if "," in line:
                u, n = line.split(",", 1)
                uuids.append(u.strip())
                names.append(n.strip())
    except Exception:
        pass
    return uuids, names


def _group_gone(pgid: int) -> bool:
    if pgid <= 0:
        return True
    try:
        os.killpg(pgid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


class _WorkerSession:
    def __init__(
        self,
        session_id: int,
        sock: socket.socket,
        client_pid: int,
        client_pgid: int,
        gpu_uuid: str,
        proc: subprocess.Popen[bytes],
        bell_path: Path,
        bell_fd: int,
        session_dir: Path,
    ) -> None:
        self.session_id = session_id
        self.sock = sock
        self.client_pid = int(client_pid)
        self.client_pgid = int(client_pgid)
        self.gpu_uuid = gpu_uuid
        self.proc = proc
        self.worker_pid = int(proc.pid)
        try:
            self.worker_pgid = int(os.getpgid(proc.pid))
        except OSError:
            self.worker_pgid = int(proc.pid)
        self.bell_path = bell_path
        self.bell_fd = bell_fd
        self.session_dir = session_dir
        self.closing = threading.Event()
        self.closed = threading.Event()
        self._cleanup_lock = threading.Lock()

    def terminate_and_reap(self, timeout: float = 30.0) -> None:
        self.closing.set()
        with self._cleanup_lock:
            if self.closed.is_set():
                return
            try:
                pgid = self.worker_pgid
                if not _group_gone(pgid):
                    try:
                        os.killpg(pgid, signal.SIGTERM)
                    except OSError:
                        pass
                deadline = time.monotonic() + timeout
                grace = time.monotonic() + 0.25
                killed = False
                while time.monotonic() < deadline:
                    if self.proc.poll() is not None and _group_gone(pgid):
                        break
                    if not killed and time.monotonic() >= grace:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except OSError:
                            pass
                        killed = True
                    time.sleep(0.01)
                try:
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
            finally:
                if self.bell_fd >= 0:
                    try:
                        os.close(self.bell_fd)
                    except OSError:
                        pass
                    self.bell_fd = -1
                try:
                    self.bell_path.unlink(missing_ok=True)
                except OSError:
                    pass
                shutil.rmtree(self.session_dir, ignore_errors=True)
                self.closed.set()


class RemoteWorkerDaemon:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        coord_dir: Path | None = None,
    ) -> None:
        self.bind_host = host
        self.bind_port = port
        self.coord_dir = coord_dir
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.bind_host, self.bind_port))
        self.server_sock.listen(128)
        self.actual_port = int(self.server_sock.getsockname()[1])
        self.host_ip = _detect_host_ip() if self.bind_host == "0.0.0.0" else self.bind_host
        self.endpoint = f"{self.host_ip}:{self.actual_port}"
        self.gpu_uuids, self.gpu_names = _query_local_gpus()

        self._lock = threading.Lock()
        self._session_seq = 0
        self._sessions: dict[int, _WorkerSession] = {}
        self._file_locks: dict[str, threading.Lock] = {}
        self._active_shm_dir: str | None = None
        self.stop_event = threading.Event()
        self._accept_thread: threading.Thread | None = None

        base_tmp = Path("/dev/shm") if Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK) else Path(tempfile.gettempdir())
        self.runtime_dir = base_tmp / f"ditserve-daemon-{os.getpid()}"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        if self.coord_dir is not None:
            self.write_coord_file(self.coord_dir / "node1.json")

    def write_coord_file(self, path: Path) -> None:
        info = {
            "host": self.host_ip,
            "port": self.actual_port,
            "endpoint": self.endpoint,
            "gpu_uuids": self.gpu_uuids,
            "gpu_names": self.gpu_names,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        }
        _atomic_write_bytes(str(path), (json.dumps(info, indent=2) + "\n").encode("utf-8"))

    def start_background(self) -> None:
        self._accept_thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._accept_thread.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        try:
            self.server_sock.close()
        except OSError:
            pass
        with self._lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            try:
                s.sock.close()
            except OSError:
                pass
            s.terminate_and_reap(timeout=5.0)
        shutil.rmtree(self.runtime_dir, ignore_errors=True)

    def serve_forever(self) -> None:
        while not self.stop_event.is_set():
            try:
                client_sock, _ = self.server_sock.accept()
            except OSError:
                if self.stop_event.is_set():
                    break
                continue
            _set_nodelay(client_sock)
            t = threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True)
            t.start()

    def _file_lock(self, path_str: str) -> threading.Lock:
        with self._lock:
            lock = self._file_locks.get(path_str)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[path_str] = lock
            return lock

    def _handle_client(self, sock: socket.socket) -> None:
        send_lock = threading.Lock()
        try:
            hdr, _ = recv_frame(sock)
            op = hdr.get("_bridge_op")
            if op == "ping":
                send_frame(
                    sock,
                    send_lock,
                    {
                        "ok": True,
                        "endpoint": self.endpoint,
                        "gpu_uuids": self.gpu_uuids,
                        "gpu_names": self.gpu_names,
                        "hostname": socket.gethostname(),
                    },
                )
                return
            if op == "release_daemon":
                send_frame(sock, send_lock, {"ok": True})
                threading.Thread(target=self.shutdown, daemon=True).start()
                return
            if op == "occupancy":
                res = self._handle_occupancy(hdr)
                send_frame(sock, send_lock, res)
                return
            if op == "spawn_worker":
                self._handle_worker_session(sock, send_lock, hdr)
                return
            send_frame(sock, send_lock, {"ok": False, "error": f"unknown _bridge_op: {op}"})
        except Exception:
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _poll_disconnected_sessions(self) -> list[_WorkerSession]:
        with self._lock:
            sessions = list(self._sessions.values())
        closing_sessions: list[_WorkerSession] = []
        for s in sessions:
            if s.closing.is_set():
                closing_sessions.append(s)
                continue
            try:
                r, _, _ = select.select([s.sock], [], [], 0.0)
                if r:
                    peeked = s.sock.recv(1, socket.MSG_PEEK)
                    if not peeked:
                        s.closing.set()
                        closing_sessions.append(s)
            except OSError:
                s.closing.set()
                closing_sessions.append(s)
        return closing_sessions

    def _handle_occupancy(self, req: dict[str, Any]) -> dict[str, Any]:
        backend = req.get("occupancy_backend", "nvidia_smi")
        uuids = list(req.get("gpu_uuids") or [])
        allowed_pids = [int(x) for x in (req.get("allowed_pids") or [])]
        allowed_pgids = [int(x) for x in (req.get("allowed_pgids") or [])]

        # Ensure any disconnected proxy sessions finish reaping their GPU workers first
        closing = self._poll_disconnected_sessions()
        for s in closing:
            s.terminate_and_reap(timeout=15.0)

        with self._lock:
            active_sessions = [s for s in self._sessions.values() if not s.closing.is_set() and not s.closed.is_set()]

        local_pids = set(allowed_pids)
        local_pgids = set(allowed_pgids)
        allowed_pid_set = set(allowed_pids)
        allowed_pgid_set = set(allowed_pgids)
        for s in active_sessions:
            if s.client_pid in allowed_pid_set or s.client_pgid in allowed_pgid_set:
                local_pids.add(s.worker_pid)
                local_pgids.add(s.worker_pgid)

        local_req = {
            "occupancy_backend": backend,
            "gpu_uuids": uuids,
            "allowed_pids": sorted(local_pids),
            "allowed_pgids": sorted(local_pgids),
            "_local_only": True,
        }
        if backend == "nvml":
            from . import occupancy_nvml

            return occupancy_nvml.observe(local_req)
        from . import worker as worker_mod

        return worker_mod.occupancy(local_req)

    def _handle_worker_session(
        self,
        sock: socket.socket,
        send_lock: threading.Lock,
        hdr: dict[str, Any],
    ) -> None:
        with self._lock:
            self._session_seq += 1
            session_id = self._session_seq

        session_dir = self.runtime_dir / f"session-{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        bell_path = self.runtime_dir / f"doorbell-{session_id}.bin"
        bell_fd = os.open(str(bell_path), os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, 0o644)
        _write_doorbell_fd(bell_fd, int(hdr.get("env_doorbell_initial", 0)))

        config = dict(hdr.get("config") or {})
        backend = str(hdr.get("backend") or config.get("backend") or "mock")
        gpu_uuid = str(hdr.get("gpu_uuid") or config.get("gpu_uuid") or "")
        client_pid = int(hdr.get("client_pid") or 0)
        client_pgid = int(hdr.get("client_pgid") or 0)

        work_dir_str = config.get("work_dir")
        if isinstance(work_dir_str, str) and work_dir_str:
            Path(work_dir_str).mkdir(parents=True, exist_ok=True)

        shm_dir = _extract_shm_dir(config)
        if shm_dir is not None:
            with self._lock:
                if str(shm_dir) != self._active_shm_dir:
                    if self._active_shm_dir is not None:
                        shutil.rmtree(self._active_shm_dir, ignore_errors=True)
                    self._active_shm_dir = str(shm_dir)
                    shm_dir.mkdir(parents=True, exist_ok=True)

        local_cfg_path = session_dir / "worker.json"
        local_cfg_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        env = os.environ.copy()
        for k, v in (hdr.get("env") or {}).items():
            env[str(k)] = str(v)
        for k in (
            "DITSERVE_REMOTE_WORKER",
            "DITSERVE_REMOTE_ENDPOINT",
            "DITSERVE_REMOTE_SLOT",
            "DITSERVE_REMOTE_TARGET_SCRIPT",
        ):
            env.pop(k, None)
        if gpu_uuid:
            env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        env["DITSERVE_DOORBELL_FILE"] = str(bell_path)

        python_exe = str(hdr.get("python") or sys.executable)
        if not Path(python_exe).is_file():
            python_exe = sys.executable

        target_script = str(hdr.get("target_script") or "")
        target_argv = list(hdr.get("target_argv") or [])
        if target_script:
            cmd = [python_exe, "-u", target_script, *target_argv]
        else:
            cmd = [
                python_exe,
                "-u",
                "-m",
                "ditserve.runtime.worker",
                "--backend",
                backend,
                "--config",
                str(local_cfg_path),
            ]

        log_path = session_dir / "worker.stderr.log"
        log_file = open(log_path, "ab")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
                env=env,
                start_new_session=True,
            )
        except Exception as exc:
            log_file.close()
            os.close(bell_fd)
            send_frame(sock, send_lock, {"ok": False, "error": str(exc)})
            return

        session = _WorkerSession(
            session_id=session_id,
            sock=sock,
            client_pid=client_pid,
            client_pgid=client_pgid,
            gpu_uuid=gpu_uuid,
            proc=proc,
            bell_path=bell_path,
            bell_fd=bell_fd,
            session_dir=session_dir,
        )
        with self._lock:
            self._sessions[session_id] = session

        send_frame(
            sock,
            send_lock,
            {
                "ok": True,
                "remote_pid": session.worker_pid,
                "remote_pgid": session.worker_pgid,
                "doorbell_path": str(bell_path),
            },
        )

        sent_shm_files: set[tuple[str, int, int]] = set()

        def _sync_remote_shm_to_proxy() -> None:
            if shm_dir is None or not shm_dir.is_dir():
                return
            for p in sorted(shm_dir.glob("*.pt")):
                try:
                    st = p.stat()
                    ident = (str(p), st.st_size, st.st_mtime_ns)
                    if ident in sent_shm_files:
                        continue
                    raw = p.read_bytes()
                    send_frame(sock, send_lock, {"_bridge_op": "file_put", "path": str(p)}, raw)
                    sent_shm_files.add(ident)
                except OSError:
                    continue

        try:
            assert proc.stdin is not None and proc.stdout is not None
            while not self.stop_event.is_set():
                msg_hdr, msg_blob = recv_frame(sock)
                mop = msg_hdr.get("_bridge_op")
                if mop == "close":
                    break
                if mop == "doorbell":
                    if session.bell_fd >= 0:
                        _write_doorbell_fd(session.bell_fd, int(msg_hdr.get("value", 0)))
                    continue
                if mop == "file_put":
                    _atomic_write_bytes(msg_hdr["path"], msg_blob)
                    continue
                if mop != "rpc":
                    continue

                raw_line = str(msg_hdr.get("raw_line", ""))
                try:
                    req = json.loads(raw_line)
                except json.JSONDecodeError:
                    req = None

                op = req.get("op") if isinstance(req, dict) else None
                if isinstance(req, dict):
                    if op == "advance" and "doorbell" in req:
                        if "doorbell_initial_value" in msg_hdr and session.bell_fd >= 0:
                            _write_doorbell_fd(session.bell_fd, int(msg_hdr["doorbell_initial_value"]))
                        req["doorbell"] = str(bell_path)
                        raw_line = json.dumps(req, separators=(",", ":"))
                    elif op == "block_import" and isinstance(req.get("path"), str):
                        imp_path = req["path"]
                        flock = self._file_lock(imp_path)
                        with flock:
                            if not Path(imp_path).is_file():
                                send_frame(sock, send_lock, {"_bridge_op": "need_file", "path": imp_path})
                                while True:
                                    fhdr, fblob = recv_frame(sock)
                                    fop = fhdr.get("_bridge_op")
                                    if fop == "doorbell":
                                        if session.bell_fd >= 0:
                                            _write_doorbell_fd(session.bell_fd, int(fhdr.get("value", 0)))
                                        continue
                                    if fop == "file_put":
                                        if fhdr.get("found", True):
                                            _atomic_write_bytes(fhdr["path"], fblob)
                                        break

                proc.stdin.write((raw_line + "\n").encode("utf-8"))
                proc.stdin.flush()

                resp_bytes = proc.stdout.readline()
                if not resp_bytes:
                    raise EOFError("remote worker exited before replying")
                resp_line = resp_bytes.decode("utf-8", errors="replace").rstrip("\n")

                try:
                    resp_obj = json.loads(resp_line)
                except json.JSONDecodeError:
                    resp_obj = None

                if isinstance(resp_obj, dict) and resp_obj.get("ok") and isinstance(resp_obj.get("result"), dict):
                    res_dict = resp_obj["result"]
                    if op == "encode_poll" and res_dict.get("ready"):
                        _sync_remote_shm_to_proxy()
                    if op != "block_export":
                        for art_path in _collect_artifact_paths(res_dict):
                            p = Path(art_path)
                            if p.is_file():
                                try:
                                    send_frame(
                                        sock,
                                        send_lock,
                                        {"_bridge_op": "file_put", "path": str(p)},
                                        p.read_bytes(),
                                    )
                                except OSError:
                                    pass

                send_frame(sock, send_lock, {"_bridge_op": "rpc_response", "line": resp_line})
                if op == "shutdown":
                    break
        except (EOFError, OSError):
            pass
        finally:
            session.terminate_and_reap(timeout=30.0)
            log_file.close()
            with self._lock:
                self._sessions.pop(session_id, None)


def main_daemon(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DitServe multi-node remote worker TCP daemon.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DITSERVE_DAEMON_PORT", "29550")))
    parser.add_argument(
        "--coord-dir",
        type=Path,
        default=Path(os.environ["DITSERVE_MULTINODE_DIR"]) if os.environ.get("DITSERVE_MULTINODE_DIR") else None,
    )
    parser.add_argument("--coord-file", type=Path, default=None)
    args = parser.parse_args(argv)

    daemon = RemoteWorkerDaemon(host=args.host, port=args.port, coord_dir=args.coord_dir)
    if args.coord_file is not None:
        daemon.write_coord_file(args.coord_file)

    def _sig_handler(_signum: int, _frame: Any) -> None:
        daemon.shutdown()

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)
    print(
        json.dumps(
            {
                "status": "listening",
                "endpoint": daemon.endpoint,
                "gpu_count": len(daemon.gpu_uuids),
                "gpu_uuids": daemon.gpu_uuids,
            }
        ),
        flush=True,
    )
    daemon.serve_forever()
    return 0
