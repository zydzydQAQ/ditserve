"""Shared pytest configuration and deterministic CPU NumPy-backed Torch shim for DITSERVE domain tests.

Ensures all 5 functional domain test suites (`Control`, `BlockExecutor`, `Memory`,
`Collectives`, `Archive`) execute rapidly and deterministically on CPU even when
a full GPU PyTorch installation is absent in the host test environment.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import io
import math
import pathlib
import pickle
import sys
import types
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _rel in ("workers", "python", "benchmarks", "tests/python"):
    _cand = str(ROOT / _rel)
    if _cand not in sys.path:
        sys.path.insert(0, _cand)


class _DType:
    def __init__(self, name: str, np_dtype: Any, itemsize: int):
        self.name = name
        self.np_dtype = np.dtype(np_dtype)
        self.itemsize = int(itemsize)

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    def __str__(self) -> str:
        return f"torch.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DType):
            return self.name == other.name
        if isinstance(other, str):
            return other in (self.name, f"torch.{self.name}")
        return False

    def __hash__(self) -> int:
        return hash(self.name)


uint8 = _DType("uint8", np.uint8, 1)
int8 = _DType("int8", np.int8, 1)
int32 = _DType("int32", np.int32, 4)
int64 = _DType("int64", np.int64, 8)
long = int64
float16 = _DType("float16", np.float16, 2)
bfloat16 = _DType("bfloat16", np.float16, 2)
float32 = _DType("float32", np.float32, 4)
float64 = _DType("float64", np.float64, 8)
bool_dtype = _DType("bool", np.bool_, 1)

_DTYPE_MAP = {
    "uint8": uint8,
    "torch.uint8": uint8,
    "int8": int8,
    "torch.int8": int8,
    "int32": int32,
    "torch.int32": int32,
    "int64": int64,
    "torch.int64": int64,
    "long": int64,
    "torch.long": int64,
    "float16": float16,
    "torch.float16": float16,
    "bfloat16": bfloat16,
    "torch.bfloat16": bfloat16,
    "float32": float32,
    "torch.float32": float32,
    "float64": float64,
    "torch.float64": float64,
    "bool": bool_dtype,
    "torch.bool": bool_dtype,
}


def _resolve_dtype(dt: Any, default: _DType = float32) -> _DType:
    if dt is None:
        return default
    if isinstance(dt, _DType):
        return dt
    if isinstance(dt, str) and dt in _DTYPE_MAP:
        return _DTYPE_MAP[dt]
    if dt is int:
        return int64
    if dt is float:
        return float32
    if dt is bool:
        return bool_dtype
    return default


class _Device:
    __module__ = "torch"

    def __init__(self, spec: Any = "cpu"):
        if isinstance(spec, _Device):
            self.spec = spec.spec
            self.type = spec.type
            self.index = spec.index
            return
        if isinstance(spec, int):
            self.spec = f"cuda:{spec}"
            self.type = "cuda"
            self.index = spec
            return
        s = str(spec or "cpu")
        if s.startswith("cuda"):
            self.type = "cuda"
            self.index = int(s.split(":")[1]) if ":" in s else 0
            self.spec = s if ":" in s else "cuda:0"
        else:
            self.type = "cpu"
            self.index = None
            self.spec = "cpu"

    def __repr__(self) -> str:
        return f"device(type='{self.type}')" if self.index is None else f"device(type='{self.type}', index={self.index})"

    def __str__(self) -> str:
        return self.spec

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _Device):
            return self.spec == other.spec or (self.type == "cpu" and other.type == "cpu")
        if isinstance(other, str):
            return self.spec == other or (self.type == "cpu" and other == "cpu")
        return False

    def __hash__(self) -> int:
        return hash(self.spec)


_Device.__name__ = "device"


class _Layout:
    __module__ = "torch"

    def __init__(self, name: str = "torch.strided"):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


_Layout.__name__ = "layout"
strided = _Layout("torch.strided")


class _UntypedStorage:
    def __init__(self, raw_bytes: bytearray):
        self._buf = raw_bytes

    def nbytes(self) -> int:
        return len(self._buf)

    def data_ptr(self) -> int:
        return id(self._buf)


class CpuTensor:
    """NumPy-backed tensor double supporting views, storage sharing, dtypes, and neural ops."""

    def __init__(
        self,
        arr: np.ndarray,
        *,
        dtype: _DType | None = None,
        device: Any = "cpu",
        pinned: bool = False,
        storage: _UntypedStorage | None = None,
        storage_offset: int = 0,
    ):
        resolved_dt = _resolve_dtype(dtype, float32)
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=resolved_dt.np_dtype)
        elif arr.dtype != resolved_dt.np_dtype:
            arr = arr.astype(resolved_dt.np_dtype)
        self._arr = np.ascontiguousarray(arr) if storage is None else arr
        self.dtype = resolved_dt
        self.device = _Device(device)
        self._pinned = bool(pinned)
        self.layout = strided
        if storage is None:
            raw = bytearray(self._arr.tobytes())
            self._storage = _UntypedStorage(raw)
            self._arr = np.frombuffer(self._storage._buf, dtype=resolved_dt.np_dtype).reshape(self._arr.shape)
            self._storage_offset = 0
        else:
            self._storage = storage
            self._storage_offset = int(storage_offset)
        self._recorded_streams: list[Any] = []
        self._base = None

    def is_conj(self) -> bool:
        return False

    def is_neg(self) -> bool:
        return False

    def _sync_from_arr(self) -> None:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self._arr.shape)

    @property
    def ndim(self) -> int:
        return int(self._arr.ndim)

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def nbytes(self) -> int:
        return self.numel() * self.element_size()

    def numel(self) -> int:
        return int(self._arr.size)

    def element_size(self) -> int:
        return int(self.dtype.itemsize)

    def stride(self) -> tuple[int, ...]:
        item = max(1, self._arr.itemsize)
        return tuple(int(s // item) for s in self._arr.strides)

    def storage_offset(self) -> int:
        return int(self._storage_offset)

    def untyped_storage(self) -> _UntypedStorage:
        return self._storage

    def is_pinned(self) -> bool:
        return self._pinned

    def pin_memory(self) -> "CpuTensor":
        return self.clone(pinned=True)

    def is_contiguous(self) -> bool:
        return bool(self._arr.flags["C_CONTIGUOUS"])

    def contiguous(self) -> "CpuTensor":
        if self.is_contiguous() and self._storage_offset == 0 and len(self._storage._buf) == self.nbytes:
            return self
        return CpuTensor(np.ascontiguousarray(self._arr).copy(), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def detach(self) -> "CpuTensor":
        return self

    def cpu(self) -> "CpuTensor":
        if self.device.type == "cpu":
            return self
        return self.to(device="cpu")

    def clone(self, *, pinned: bool | None = None) -> "CpuTensor":
        return CpuTensor(
            np.ascontiguousarray(self._arr).copy(),
            dtype=self.dtype,
            device=self.device,
            pinned=self._pinned if pinned is None else pinned,
        )

    def to(self, *args: Any, **kwargs: Any) -> "CpuTensor":
        target_device = kwargs.get("device", self.device)
        target_dtype = kwargs.get("dtype", self.dtype)
        for arg in args:
            if isinstance(arg, (_DType, type)) or (isinstance(arg, str) and arg in _DTYPE_MAP):
                target_dtype = _resolve_dtype(arg, self.dtype)
            elif isinstance(arg, (_Device, str)):
                target_device = _Device(arg)
            elif isinstance(arg, CpuTensor):
                target_device = arg.device
                target_dtype = arg.dtype
        resolved_dt = _resolve_dtype(target_dtype, self.dtype)
        resolved_dev = _Device(target_device)
        if resolved_dt == self.dtype and resolved_dev == self.device and not kwargs.get("copy", False):
            return self
        return CpuTensor(
            self._arr.astype(resolved_dt.np_dtype).copy(),
            dtype=resolved_dt,
            device=resolved_dev,
            pinned=self._pinned and resolved_dev.type == "cpu",
        )

    def type_as(self, other: "CpuTensor") -> "CpuTensor":
        return self.to(dtype=other.dtype, device=other.device)

    def float(self) -> "CpuTensor":
        return self.to(dtype=float32)

    def set_(
        self,
        storage: _UntypedStorage,
        storage_offset: int = 0,
        size: tuple[int, ...] | list[int] = (),
        stride: tuple[int, ...] | list[int] | None = None,
    ) -> "CpuTensor":
        self._storage = storage
        self._storage_offset = int(storage_offset)
        shape = tuple(int(s) for s in size)
        elem_size = self.dtype.itemsize
        byte_offset = int(storage_offset) * elem_size
        if stride is not None and len(stride) == len(shape):
            np_strides = tuple(int(st) * elem_size for st in stride)
            self._arr = np.ndarray(
                shape=shape,
                dtype=self.dtype.np_dtype,
                buffer=storage._buf,
                offset=byte_offset,
                strides=np_strides,
            )
        else:
            count = math.prod(shape) if shape else 1
            flat = np.frombuffer(storage._buf, dtype=self.dtype.np_dtype, count=count, offset=byte_offset)
            self._arr = flat.reshape(shape)
        return self

    def copy_(self, src: "CpuTensor", non_blocking: bool = False) -> "CpuTensor":
        src_arr = src._arr if isinstance(src, CpuTensor) else np.asarray(src, dtype=self.dtype.np_dtype)
        np.copyto(self._arr, src_arr.astype(self.dtype.np_dtype))
        return self

    def fill_(self, value: float | int) -> "CpuTensor":
        self._arr.fill(value)
        return self

    def zero_(self) -> "CpuTensor":
        self._arr.fill(0)
        return self

    def record_stream(self, stream: Any) -> None:
        self._recorded_streams.append(stream)

    def view(self, *args: Any) -> "CpuTensor":
        if len(args) == 1 and (isinstance(args[0], _DType) or (isinstance(args[0], str) and args[0] in _DTYPE_MAP)):
            target_dt = _resolve_dtype(args[0])
            contig = self.contiguous()
            viewed_arr = contig._arr.view(target_dt.np_dtype)
            return CpuTensor(
                viewed_arr,
                dtype=target_dt,
                device=self.device,
                pinned=self._pinned,
                storage=contig._storage,
                storage_offset=0,
            )
        shape = args[0] if (len(args) == 1 and isinstance(args[0], (tuple, list))) else args
        reshaped = self._arr.reshape(shape)
        return CpuTensor(
            reshaped,
            dtype=self.dtype,
            device=self.device,
            pinned=self._pinned,
            storage=self._storage,
            storage_offset=self._storage_offset,
        )

    def reshape(self, *shape: Any) -> "CpuTensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self.view(*shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "CpuTensor":
        ndim = self.ndim
        if ndim == 0:
            return self.reshape(1)
        s = start_dim if start_dim >= 0 else ndim + start_dim
        e = end_dim if end_dim >= 0 else ndim + end_dim
        new_shape = self.shape[:s] + (-1,) + self.shape[e + 1 :]
        return self.reshape(*new_shape)

    def unflatten(self, dim: int, sizes: tuple[int, ...] | list[int]) -> "CpuTensor":
        d = dim if dim >= 0 else self.ndim + dim
        new_shape = self.shape[:d] + tuple(sizes) + self.shape[d + 1 :]
        return self.reshape(*new_shape)

    def unsqueeze(self, dim: int) -> "CpuTensor":
        return CpuTensor(np.expand_dims(self._arr, axis=dim), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def squeeze(self, dim: int | None = None) -> "CpuTensor":
        return CpuTensor(np.squeeze(self._arr, axis=dim), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def transpose(self, dim0: int, dim1: int) -> "CpuTensor":
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return CpuTensor(np.transpose(self._arr, axes), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def permute(self, *dims: int) -> "CpuTensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return CpuTensor(np.transpose(self._arr, dims), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def chunk(self, chunks: int, dim: int = 0) -> tuple["CpuTensor", ...]:
        parts = np.array_split(self._arr, chunks, axis=dim)
        return tuple(CpuTensor(p.copy(), dtype=self.dtype, device=self.device, pinned=self._pinned) for p in parts)

    def unbind(self, dim: int = 0) -> tuple["CpuTensor", ...]:
        return tuple(
            CpuTensor(np.take(self._arr, i, axis=dim).copy(), dtype=self.dtype, device=self.device, pinned=self._pinned)
            for i in range(self.shape[dim])
        )

    def expand(self, *sizes: int) -> "CpuTensor":
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        padded_shape = (1,) * (len(sizes) - self.ndim) + self.shape
        target = tuple(p if s == -1 else s for s, p in zip(sizes, padded_shape))
        return CpuTensor(np.broadcast_to(self._arr, target).copy(), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def repeat_interleave(self, repeats: int | "CpuTensor", dim: int | None = None, output_size: int | None = None) -> "CpuTensor":
        rep = repeats._arr if isinstance(repeats, CpuTensor) else repeats
        return CpuTensor(np.repeat(self._arr, rep, axis=dim), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def repeat(self, *sizes: int) -> "CpuTensor":
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        return CpuTensor(np.tile(self._arr, sizes).copy(), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def flip(self, dim: int = 0) -> "CpuTensor":
        return CpuTensor(np.flip(self._arr, axis=dim).copy(), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def abs(self) -> "CpuTensor":
        return CpuTensor(np.abs(self._arr), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def amax(self, dim: int | None = None, keepdim: bool = False) -> "CpuTensor":
        return CpuTensor(np.amax(self._arr, axis=dim, keepdims=keepdim), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def max(self, dim: int | None = None, keepdim: bool = False) -> "CpuTensor":
        return self.amax(dim=dim, keepdim=keepdim)

    def min(self, dim: int | None = None, keepdim: bool = False) -> "CpuTensor":
        return CpuTensor(np.amin(self._arr, axis=dim, keepdims=keepdim), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def sum(self, dim: int | None = None, keepdim: bool = False) -> "CpuTensor":
        out_dt = int64 if self.dtype in (bool_dtype, int8, int32, int64) else float32
        return CpuTensor(np.sum(self._arr, axis=dim, keepdims=keepdim), dtype=out_dt, device=self.device)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> "CpuTensor":
        return CpuTensor(np.mean(self._arr.astype(np.float32), axis=dim, keepdims=keepdim), dtype=float32, device=self.device)

    def clamp(self, min: float | None = None, max: float | None = None) -> "CpuTensor":
        return CpuTensor(np.clip(self._arr, min, max), dtype=self.dtype, device=self.device, pinned=self._pinned)

    def gt(self, other: Any) -> "CpuTensor":
        val = other._arr if isinstance(other, CpuTensor) else other
        return CpuTensor(self._arr > val, dtype=bool_dtype, device=self.device)

    def item(self) -> float | int | bool:
        return self._arr.item()

    def tolist(self) -> Any:
        return self._arr.tolist()

    def numpy(self) -> np.ndarray:
        return self._arr

    def __len__(self) -> int:
        return int(self._arr.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: Any) -> "CpuTensor":
        if isinstance(idx, CpuTensor):
            idx = idx._arr
        if isinstance(idx, (int, np.integer)) and self.ndim == 1:
            norm_idx = int(idx) if int(idx) >= 0 else self.shape[0] + int(idx)
            offset = self._storage_offset + norm_idx
            byte_offset = offset * self.dtype.itemsize
            sub = np.ndarray(shape=(), dtype=self.dtype.np_dtype, buffer=self._storage._buf, offset=byte_offset)
            return CpuTensor(sub, dtype=self.dtype, device=self.device, pinned=self._pinned, storage=self._storage, storage_offset=offset)
        sub = self._arr[idx]
        if not isinstance(sub, np.ndarray):
            sub = np.asarray(sub, dtype=self.dtype.np_dtype)
        return CpuTensor(sub, dtype=self.dtype, device=self.device, pinned=self._pinned)

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(idx, CpuTensor):
            idx = idx._arr
        v = val._arr if isinstance(val, CpuTensor) else val
        self._arr[idx] = v

    def _binop(self, other: Any, op: Any, out_dtype: _DType | None = None) -> "CpuTensor":
        rhs = other._arr if isinstance(other, CpuTensor) else other
        res = op(self._arr.astype(np.float32) if self.dtype in (float16, bfloat16) else self._arr,
                 rhs.astype(np.float32) if isinstance(other, CpuTensor) and other.dtype in (float16, bfloat16) else rhs)
        dt = out_dtype or self.dtype
        return CpuTensor(np.asarray(res, dtype=dt.np_dtype), dtype=dt, device=self.device)

    def __add__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: b + a)

    def __sub__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: b * a)

    def __truediv__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> "CpuTensor":
        return self._binop(other, lambda a, b: b / a)

    def __neg__(self) -> "CpuTensor":
        return CpuTensor(-self._arr, dtype=self.dtype, device=self.device, pinned=self._pinned)

    def __gt__(self, other: Any) -> "CpuTensor":
        return self.gt(other)

    def __lt__(self, other: Any) -> "CpuTensor":
        val = other._arr if isinstance(other, CpuTensor) else other
        return CpuTensor(self._arr < val, dtype=bool_dtype, device=self.device)

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        if other is None or not isinstance(other, (CpuTensor, np.ndarray, int, float, bool)):
            return False
        val = other._arr if isinstance(other, CpuTensor) else other
        return CpuTensor(self._arr == val, dtype=bool_dtype, device=self.device)

    def all(self) -> "CpuTensor":
        return CpuTensor(np.asarray(bool(np.all(self._arr)), dtype=np.bool_), dtype=bool_dtype, device=self.device)

    def any(self) -> "CpuTensor":
        return CpuTensor(np.asarray(bool(np.any(self._arr)), dtype=np.bool_), dtype=bool_dtype, device=self.device)

    def __bool__(self) -> bool:
        return bool(self._arr.item() if self._arr.size == 1 else np.all(self._arr))

    def __int__(self) -> int:
        return int(self._arr.item())

    def __float__(self) -> float:
        return float(self._arr.item())


class _Generator:
    def __init__(self, device: Any = "cpu"):
        self.device = _Device(device)
        self._seed = 0
        self._rng = np.random.RandomState(0)

    def manual_seed(self, seed: int) -> "_Generator":
        self._seed = int(seed)
        self._rng = np.random.RandomState(int(seed) % (2**32 - 1))
        return self

    def initial_seed(self) -> int:
        return self._seed

    def get_state(self) -> CpuTensor:
        state_bytes = pickle.dumps((self._seed, self._rng.get_state()))
        buf = np.zeros(5056, dtype=np.uint8)
        n = len(state_bytes)
        buf[:4] = np.frombuffer(int(n).to_bytes(4, "little"), dtype=np.uint8)
        buf[4 : 4 + n] = np.frombuffer(state_bytes, dtype=np.uint8)
        return CpuTensor(buf, dtype=uint8, device="cpu")

    def set_state(self, state: CpuTensor) -> None:
        raw = state._arr.tobytes()
        n = int.from_bytes(raw[:4], "little")
        seed, rng_state = pickle.loads(raw[4 : 4 + n])
        self._seed = seed
        self._rng.set_state(rng_state)


class _Module:
    def __init__(self):
        self.training = False
        self._forward_hooks: dict[int, Any] = {}
        self._forward_pre_hooks: dict[int, Any] = {}
        self._hook_counter = 0

    def eval(self) -> "_Module":
        self.training = False
        for m in self.modules():
            m.training = False
        return self

    def to(self, *args: Any, **kwargs: Any) -> "_Module":
        for k, v in list(vars(self).items()):
            if isinstance(v, CpuTensor):
                setattr(self, k, v.to(*args, **kwargs))
            elif isinstance(v, _Module):
                v.to(*args, **kwargs)
        return self

    def parameters(self):
        for k, v in vars(self).items():
            if isinstance(v, CpuTensor):
                yield v
            elif isinstance(v, _Module):
                yield from v.parameters()

    def modules(self):
        yield self
        for v in vars(self).values():
            if isinstance(v, _Module) and v is not self:
                yield from v.modules()

    def register_forward_hook(self, fn: Any):
        self._hook_counter += 1
        hid = self._hook_counter
        self._forward_hooks[hid] = fn
        return types.SimpleNamespace(remove=lambda: self._forward_hooks.pop(hid, None))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.forward(*args, **kwargs)
        for hook in list(self._forward_hooks.values()):
            hook(self, args, out)
        return out


class _ModuleList(_Module):
    def __init__(self, items=()):
        super().__init__()
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]

    def modules(self):
        yield self
        for item in self._items:
            if isinstance(item, _Module):
                yield from item.modules()

    def parameters(self):
        for item in self._items:
            if isinstance(item, _Module):
                yield from item.parameters()


class _Sequential(_ModuleList):
    def __init__(self, *layers: Any):
        super().__init__(layers)

    def forward(self, x: CpuTensor) -> CpuTensor:
        for layer in self._items:
            x = layer(x)
        return x


class _Linear(_Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        rng = np.random.RandomState((in_features * 31 + out_features * 17) % 10007)
        scale = 1.0 / math.sqrt(max(1, in_features))
        self.weight = CpuTensor(rng.uniform(-scale, scale, (out_features, in_features)).astype(np.float32))
        self.bias = CpuTensor(np.zeros((out_features,), dtype=np.float32)) if bias else None

    def forward(self, x: CpuTensor) -> CpuTensor:
        w = self.weight._arr.astype(np.float32).T
        out = x._arr.astype(np.float32) @ w
        if self.bias is not None:
            out = out + self.bias._arr.astype(np.float32)
        return CpuTensor(out.astype(x.dtype.np_dtype), dtype=x.dtype, device=x.device)


class _LayerNorm(_Module):
    def __init__(self, normalized_shape: int, elementwise_affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: CpuTensor) -> CpuTensor:
        arr = x._arr.astype(np.float32)
        mean = arr.mean(axis=-1, keepdims=True)
        var = arr.var(axis=-1, keepdims=True)
        normed = (arr - mean) / np.sqrt(var + self.eps)
        return CpuTensor(normed.astype(x.dtype.np_dtype), dtype=x.dtype, device=x.device)


class _SiLU(_Module):
    def forward(self, x: CpuTensor) -> CpuTensor:
        arr = x._arr.astype(np.float32)
        out = arr / (1.0 + np.exp(-arr))
        return CpuTensor(out.astype(x.dtype.np_dtype), dtype=x.dtype, device=x.device)


class _GELU(_Module):
    def forward(self, x: CpuTensor) -> CpuTensor:
        arr = x._arr.astype(np.float32)
        out = 0.5 * arr * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (arr + 0.044715 * (arr**3))))
        return CpuTensor(out.astype(x.dtype.np_dtype), dtype=x.dtype, device=x.device)


class _Identity(_Module):
    def forward(self, x: CpuTensor) -> CpuTensor:
        return x


class _Embedding(_Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        rng = np.random.RandomState((num_embeddings * 13 + embedding_dim) % 9973)
        self.weight = CpuTensor(rng.normal(0.0, 0.1, (num_embeddings, embedding_dim)).astype(np.float32))

    def forward(self, input_ids: CpuTensor) -> CpuTensor:
        idx = input_ids._arr.astype(np.int64)
        out = self.weight._arr[idx]
        return CpuTensor(out, dtype=self.weight.dtype, device=input_ids.device)


class _Conv2d(_Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        rng = np.random.RandomState((in_channels * 19 + out_channels * 23) % 9973)
        self.weight = CpuTensor(
            rng.normal(0.0, 0.05, (out_channels, in_channels * self.kernel_size * self.kernel_size)).astype(np.float32)
        )

    def forward(self, x: CpuTensor) -> CpuTensor:
        b, c, h, w = x.shape
        k = self.kernel_size
        oh, ow = h // k, w // k
        patches = x._arr.astype(np.float32).reshape(b, c, oh, k, ow, k).transpose(0, 2, 4, 1, 3, 5).reshape(b, oh, ow, -1)
        out = (patches @ self.weight._arr.T).transpose(0, 3, 1, 2)
        return CpuTensor(out.astype(x.dtype.np_dtype), dtype=x.dtype, device=x.device)


def _scaled_dot_product_attention(
    q: CpuTensor, k: CpuTensor, v: CpuTensor, dropout_p: float = 0.0, is_causal: bool = False
) -> CpuTensor:
    qf = q._arr.astype(np.float32)
    kf = k._arr.astype(np.float32)
    vf = v._arr.astype(np.float32)
    scale = 1.0 / math.sqrt(max(1, qf.shape[-1]))
    scores = np.matmul(qf, np.swapaxes(kf, -1, -2)) * scale
    scores = scores - np.amax(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.maximum(1e-12, np.sum(weights, axis=-1, keepdims=True))
    out = np.matmul(weights, vf)
    return CpuTensor(out.astype(q.dtype.np_dtype), dtype=q.dtype, device=q.device)


def _cosine_similarity(x1: CpuTensor, x2: CpuTensor, dim: int = 1, eps: float = 1e-8) -> CpuTensor:
    a = x1._arr.astype(np.float32)
    b = x2._arr.astype(np.float32)
    num = np.sum(a * b, axis=dim)
    den = np.maximum(eps, np.linalg.norm(a, axis=dim) * np.linalg.norm(b, axis=dim))
    return CpuTensor(num / den, dtype=float32, device=x1.device)


def _pad(x: CpuTensor, pad_spec: tuple[int, ...], value: float = 0.0) -> CpuTensor:
    widths = [(0, 0)] * x.ndim
    for i in range(0, len(pad_spec), 2):
        axis = x.ndim - 1 - (i // 2)
        widths[axis] = (pad_spec[i], pad_spec[i + 1])
    out = np.pad(x._arr, widths, mode="constant", constant_values=value)
    return CpuTensor(out, dtype=x.dtype, device=x.device)


def _build_cpu_torch_module() -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__version__ = "2.10.0+cpu.shim"
    mod.version = types.SimpleNamespace(cuda="12.6", hip=None)
    mod.uint8 = uint8
    mod.int8 = int8
    mod.int32 = int32
    mod.int64 = int64
    mod.long = long
    mod.float16 = float16
    mod.bfloat16 = bfloat16
    mod.float32 = float32
    mod.float64 = float64
    mod.float8_e4m3fn = uint8
    mod.float8_e4m3fnuz = uint8
    mod.float8_e5m2 = uint8
    mod.float8_e5m2fnuz = uint8
    mod.bool = bool_dtype
    mod.strided = strided
    mod.device = _Device
    mod.dtype = _DType
    mod.Size = tuple
    mod.Tensor = CpuTensor
    mod.Generator = _Generator

    _global_rng = np.random.RandomState(42)

    def manual_seed(seed: int) -> None:
        _global_rng.seed(int(seed) % (2**32 - 1))

    mod.manual_seed = manual_seed
    mod.set_num_threads = lambda _n: None
    mod.get_num_threads = lambda: 1
    mod.inference_mode = nullcontext
    mod.no_grad = nullcontext
    mod.is_tensor = lambda obj: isinstance(obj, CpuTensor)
    mod.compile = lambda fn=None, *a, **k: (fn if fn is not None else lambda f: f)
    mod.compiler = types.SimpleNamespace(disable=lambda fn=None, *a, **k: (fn if fn is not None else lambda f: f))

    def tensor(data: Any, *, dtype: Any = None, device: Any = "cpu") -> CpuTensor:
        if isinstance(data, CpuTensor):
            return data.to(dtype=dtype, device=device).clone()
        arr = np.asarray(data)
        if dtype is None:
            dt = int64 if arr.dtype.kind in ("i", "u") else (bool_dtype if arr.dtype.kind == "b" else float32)
        else:
            dt = _resolve_dtype(dtype)
        return CpuTensor(arr.astype(dt.np_dtype), dtype=dt, device=device)

    def _shape_arg(shape: tuple[Any, ...]) -> tuple[int, ...]:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            return tuple(int(x) for x in shape[0])
        return tuple(int(x) for x in shape)

    def empty(*shape: Any, dtype: Any = float32, device: Any = "cpu", pin_memory: bool = False) -> CpuTensor:
        dt = _resolve_dtype(dtype)
        return CpuTensor(np.zeros(_shape_arg(shape), dtype=dt.np_dtype), dtype=dt, device=device, pinned=pin_memory)

    def empty_like(other: CpuTensor, *, dtype: Any = None, device: Any = None, pin_memory: bool = False) -> CpuTensor:
        dt = _resolve_dtype(dtype, other.dtype)
        return CpuTensor(
            np.zeros(other.shape, dtype=dt.np_dtype),
            dtype=dt,
            device=other.device if device is None else device,
            pinned=pin_memory,
        )

    def zeros(*shape: Any, dtype: Any = float32, device: Any = "cpu") -> CpuTensor:
        dt = _resolve_dtype(dtype)
        return CpuTensor(np.zeros(_shape_arg(shape), dtype=dt.np_dtype), dtype=dt, device=device)

    def ones(*shape: Any, dtype: Any = float32, device: Any = "cpu") -> CpuTensor:
        dt = _resolve_dtype(dtype)
        return CpuTensor(np.ones(_shape_arg(shape), dtype=dt.np_dtype), dtype=dt, device=device)

    def full(shape: Any, fill_value: Any, *, dtype: Any = float32, device: Any = "cpu") -> CpuTensor:
        dt = _resolve_dtype(dtype)
        return CpuTensor(np.full(_shape_arg((shape,)), fill_value, dtype=dt.np_dtype), dtype=dt, device=device)

    def full_like(other: CpuTensor, fill_value: Any, *, dtype: Any = None, device: Any = None) -> CpuTensor:
        dt = _resolve_dtype(dtype, other.dtype)
        return CpuTensor(
            np.full(other.shape, fill_value, dtype=dt.np_dtype),
            dtype=dt,
            device=other.device if device is None else device,
        )

    def arange(*args: Any, dtype: Any = int64, device: Any = "cpu") -> CpuTensor:
        dt = _resolve_dtype(dtype)
        return CpuTensor(np.arange(*args, dtype=dt.np_dtype), dtype=dt, device=device)

    def linspace(start: float, end: float, steps: int, *, dtype: Any = float32, device: Any = "cpu") -> CpuTensor:
        dt = _resolve_dtype(dtype)
        return CpuTensor(np.linspace(start, end, int(steps), dtype=dt.np_dtype), dtype=dt, device=device)

    def randn(*shape: Any, generator: _Generator | None = None, dtype: Any = float32, device: Any = "cpu") -> CpuTensor:
        dt = _resolve_dtype(dtype)
        rng = generator._rng if generator is not None else _global_rng
        arr = rng.standard_normal(_shape_arg(shape)).astype(dt.np_dtype)
        return CpuTensor(arr, dtype=dt, device=device)

    def cat(tensors: list[CpuTensor] | tuple[CpuTensor, ...], dim: int = 0) -> CpuTensor:
        first = tensors[0]
        arr = np.concatenate([t._arr for t in tensors], axis=dim)
        return CpuTensor(arr, dtype=first.dtype, device=first.device)

    def stack(tensors: list[CpuTensor] | tuple[CpuTensor, ...], dim: int = 0) -> CpuTensor:
        first = tensors[0]
        arr = np.stack([t._arr for t in tensors], axis=dim)
        return CpuTensor(arr, dtype=first.dtype, device=first.device)

    def repeat_interleave(
        input_t: CpuTensor, repeats: int | CpuTensor, dim: int | None = None, output_size: int | None = None
    ) -> CpuTensor:
        return input_t.repeat_interleave(repeats, dim=dim)

    def einsum(equation: str, *operands: CpuTensor) -> CpuTensor:
        first = operands[0]
        res = np.einsum(equation, *[op._arr.astype(np.float32) for op in operands])
        return CpuTensor(res.astype(first.dtype.np_dtype), dtype=first.dtype, device=first.device)

    def clamp(x: CpuTensor, min: float | None = None, max: float | None = None) -> CpuTensor:
        return x.clamp(min=min, max=max)

    def round_fn(x: CpuTensor) -> CpuTensor:
        return CpuTensor(np.round(x._arr), dtype=x.dtype, device=x.device)

    def equal(a: CpuTensor, b: CpuTensor) -> bool:
        return a.shape == b.shape and bool(np.array_equal(a._arr, b._arr))

    def allclose(a: CpuTensor, b: CpuTensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return a.shape == b.shape and bool(np.allclose(a._arr, b._arr, rtol=rtol, atol=atol))

    def _serialize_tree(obj: Any) -> Any:
        if isinstance(obj, CpuTensor):
            return {"__cpu_tensor__": True, "arr": obj._arr.copy(), "dtype": obj.dtype.name, "pinned": obj._pinned}
        if isinstance(obj, dict):
            return {k: _serialize_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize_tree(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_serialize_tree(v) for v in obj)
        return obj

    def _deserialize_tree(obj: Any, map_location: str = "cpu") -> Any:
        if isinstance(obj, dict) and obj.get("__cpu_tensor__") is True:
            return CpuTensor(obj["arr"], dtype=_resolve_dtype(obj["dtype"]), device=map_location, pinned=obj["pinned"])
        if isinstance(obj, dict):
            return {k: _deserialize_tree(v, map_location) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_deserialize_tree(v, map_location) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_deserialize_tree(v, map_location) for v in obj)
        return obj

    def save(obj: Any, f: str | pathlib.Path | io.BytesIO) -> None:
        payload = _serialize_tree(obj)
        if isinstance(f, (str, pathlib.Path)):
            with open(f, "wb") as fp:
                pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(f: str | pathlib.Path | io.BytesIO, map_location: str = "cpu", weights_only: bool = True) -> Any:
        if isinstance(f, (str, pathlib.Path)):
            with open(f, "rb") as fp:
                payload = pickle.load(fp)
        else:
            payload = pickle.load(f)
        return _deserialize_tree(payload, str(map_location))

    mod.tensor = tensor
    mod.empty = empty
    mod.empty_like = empty_like
    mod.zeros = zeros
    mod.ones = ones
    mod.full = full
    mod.full_like = full_like
    mod.arange = arange
    mod.linspace = linspace
    mod.randn = randn
    mod.cat = cat
    mod.stack = stack
    mod.repeat_interleave = repeat_interleave
    mod.einsum = einsum
    mod.clamp = clamp
    mod.round = round_fn
    mod.isfinite = lambda x: CpuTensor(np.isfinite(x._arr), dtype=bool_dtype, device=x.device)
    mod.sum = lambda x, dim=None, keepdim=False: x.sum(dim=dim, keepdim=keepdim)
    mod.tanh = lambda x: CpuTensor(np.tanh(x._arr.astype(np.float32)).astype(x.dtype.np_dtype), dtype=x.dtype, device=x.device)
    mod.equal = equal
    mod.allclose = allclose
    mod.save = save
    mod.load = load

    # Submodules: linalg, nn, nn.functional, cuda, distributed, testing
    linalg_mod = types.ModuleType("torch.linalg")
    linalg_mod.vector_norm = lambda x, ord=2, dim=None, keepdim=False: CpuTensor(
        np.asarray(np.linalg.norm(x._arr.astype(np.float32).reshape(-1) if dim is None else x._arr.astype(np.float32), ord=ord, axis=dim), dtype=np.float32),
        dtype=float32,
        device=x.device,
    )
    mod.linalg = linalg_mod

    nn_mod = types.ModuleType("torch.nn")
    nn_mod.__path__ = []  # type: ignore[attr-defined]
    nn_func_mod = types.ModuleType("torch.nn.functional")
    nn_func_mod.scaled_dot_product_attention = _scaled_dot_product_attention
    nn_func_mod.cosine_similarity = _cosine_similarity
    nn_func_mod.pad = _pad
    nn_mod.functional = nn_func_mod
    nn_mod.Module = _Module
    nn_mod.ModuleList = _ModuleList
    nn_mod.Sequential = _Sequential
    nn_mod.Linear = _Linear
    nn_mod.LayerNorm = _LayerNorm
    nn_mod.GroupNorm = lambda num_groups, num_channels, eps=1e-5, affine=True: _LayerNorm(num_channels, eps=eps)
    nn_mod.SiLU = _SiLU
    nn_mod.GELU = _GELU
    nn_mod.Identity = _Identity
    nn_mod.Embedding = _Embedding
    nn_mod.Conv2d = _Conv2d
    mod.nn = nn_mod

    backends_mod = types.ModuleType("torch.backends")
    backends_cuda_mod = types.ModuleType("torch.backends.cuda")
    backends_cuda_mod.math_sdp_enabled = lambda: True
    backends_cuda_mod.flash_sdp_enabled = lambda: False
    backends_cuda_mod.mem_efficient_sdp_enabled = lambda: False
    backends_cuda_mod.cudnn_sdp_enabled = lambda: False
    backends_mod.cuda = backends_cuda_mod
    backends_mod.mps = types.SimpleNamespace(is_available=lambda: False)
    mod.backends = backends_mod

    class _CudaStream:
        def __init__(self, device: Any = None, priority: int = 0):
            self.device = device
            self.priority = priority
            self.cuda_stream = id(self)

        def synchronize(self) -> None:
            pass

        def wait_stream(self, other: Any) -> None:
            pass

        def wait_event(self, event: Any) -> None:
            pass

    class _CudaEvent:
        def __init__(self, enable_timing: bool = False):
            self.enable_timing = enable_timing

        def record(self, stream: Any = None) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other: Any) -> float:
            return 0.25

    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: False
    cuda_mod.is_initialized = lambda: False
    cuda_mod.current_device = lambda: 0
    cuda_mod.device_count = lambda: 1
    cuda_mod.set_device = lambda dev: None
    cuda_mod.manual_seed = manual_seed
    cuda_mod.manual_seed_all = manual_seed
    cuda_mod.synchronize = lambda dev=None: None
    cuda_mod.current_stream = lambda device=None: _CudaStream(device)
    cuda_mod.stream = lambda s: nullcontext()
    cuda_mod.device = lambda d: nullcontext()
    cuda_mod.Stream = _CudaStream
    cuda_mod.Event = _CudaEvent
    cuda_mod.ExternalStream = lambda ptr, device=None: _CudaStream(device)
    cuda_mod.empty_cache = lambda: None
    cuda_mod.memory_allocated = lambda dev=None: 0
    cuda_mod.memory_reserved = lambda dev=None: 0
    cuda_mod.max_memory_allocated = lambda dev=None: 0
    cuda_mod.max_memory_reserved = lambda dev=None: 0
    cuda_mod.reset_peak_memory_stats = lambda dev=None: None
    cuda_mod.get_device_properties = lambda dev=0: types.SimpleNamespace(multi_processor_count=108, name="NVIDIA A100-PCIE-40GB")
    mod.cuda = cuda_mod

    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.__path__ = []  # type: ignore[attr-defined]
    dist_mod.ProcessGroup = object
    dist_mod.Backend = str
    dist_mod.ReduceOp = types.SimpleNamespace(SUM="sum", MAX="max", MIN="min", AVG="avg")
    dist_mod.is_available = lambda: True
    dist_mod.is_initialized = lambda: False
    dist_mod.get_rank = lambda group=None: 0
    dist_mod.get_world_size = lambda group=None: 1
    dist_mod.get_backend = lambda group=None: "gloo"
    dist_mod.new_group = lambda *a, **k: object()
    mod._C = types.SimpleNamespace(_distributed_c10d=types.SimpleNamespace(ReduceOp=dist_mod.ReduceOp))
    dist_mod.init_process_group = lambda *a, **k: None
    dist_mod.destroy_process_group = lambda *a, **k: None
    dist_mod.barrier = lambda *a, **k: None
    dist_mod.all_reduce = lambda *a, **k: None
    dist_mod.all_gather = lambda *a, **k: None
    dist_mod.all_to_all_single = lambda *a, **k: None
    dist_mod.broadcast = lambda *a, **k: None
    dist_mod.isend = "isend"
    dist_mod.irecv = "irecv"
    dist_mod.P2POp = lambda op, tensor, peer, group=None: types.SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

    ft_c_mod = types.ModuleType("torch.distributed._functional_collectives")

    class _AsyncCollectiveTensor:
        def __init__(self, tensor: CpuTensor):
            self._tensor = tensor

        def wait(self) -> CpuTensor:
            return self._tensor

    ft_c_mod.AsyncCollectiveTensor = _AsyncCollectiveTensor
    ft_c_mod.all_to_all_single = lambda flat, output_split_sizes=None, input_split_sizes=None, group=None: flat.clone()
    dist_mod._functional_collectives = ft_c_mod

    symm_mod = types.ModuleType("torch.distributed._symmetric_memory")
    symm_mod.empty = empty
    symm_mod.enable_symm_mem_for_group = lambda group_name: None
    symm_mod.rendezvous = lambda tensor, group=None: types.SimpleNamespace(
        rank=0,
        world_size=1,
        get_buffer=lambda peer, sizes, dtype: tensor.view(*sizes),
        barrier=lambda channel=0: None,
    )
    dist_mod._symmetric_memory = symm_mod
    dist_tensor_mod = types.ModuleType("torch.distributed.tensor")
    dist_tensor_exp_mod = types.ModuleType("torch.distributed.tensor.experimental")
    dist_tensor_attn_mod = types.ModuleType("torch.distributed.tensor.experimental._attention")
    dist_tensor_attn_mod._templated_ring_attention = lambda *a, **k: (a[0] if a else None, None)
    dist_tensor_exp_mod._attention = dist_tensor_attn_mod
    dist_tensor_mod.experimental = dist_tensor_exp_mod
    dist_mod.tensor = dist_tensor_mod

    fsdp_mod = types.ModuleType("torch.distributed.fsdp")
    fsdp_mod.__path__ = []  # type: ignore[attr-defined]
    fsdp_mod.ShardingStrategy = types.SimpleNamespace(FULL_SHARD="FULL_SHARD", SHARD_GRAD_OP="SHARD_GRAD_OP", NO_SHARD="NO_SHARD", HYBRID_SHARD="HYBRID_SHARD")
    fsdp_mod.FullyShardedDataParallel = _Module
    fsdp_wrap_mod = types.ModuleType("torch.distributed.fsdp.wrap")
    fsdp_wrap_mod.lambda_auto_wrap_policy = lambda *a, **k: None
    fsdp_mod.wrap = fsdp_wrap_mod
    dist_mod.fsdp = fsdp_mod

    device_mesh_mod = types.ModuleType("torch.distributed.device_mesh")
    device_mesh_mod.DeviceMesh = object
    dist_mod.device_mesh = device_mesh_mod
    mod.distributed = dist_mod

    testing_mod = types.ModuleType("torch.testing")

    def assert_close(actual: CpuTensor, expected: CpuTensor, rtol: float = 1e-5, atol: float = 1e-5) -> None:
        if not allclose(actual, expected, rtol=rtol, atol=atol):
            max_diff = float(np.max(np.abs(actual._arr.astype(np.float32) - expected._arr.astype(np.float32))))
            raise AssertionError(f"Tensors not close: max_abs_diff={max_diff} (rtol={rtol}, atol={atol})")

    testing_mod.assert_close = assert_close
    mod.testing = testing_mod

    sys.modules["torch"] = mod
    sys.modules["torch.linalg"] = linalg_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = nn_func_mod
    sys.modules["torch.cuda"] = cuda_mod
    sys.modules["torch.distributed"] = dist_mod
    sys.modules["torch.distributed._functional_collectives"] = ft_c_mod
    sys.modules["torch.distributed._symmetric_memory"] = symm_mod
    sys.modules["torch.distributed.tensor"] = dist_tensor_mod
    sys.modules["torch.distributed.tensor.experimental"] = dist_tensor_exp_mod
    sys.modules["torch.distributed.tensor.experimental._attention"] = dist_tensor_attn_mod
    sys.modules["torch.distributed.fsdp"] = fsdp_mod
    sys.modules["torch.distributed.fsdp.wrap"] = fsdp_wrap_mod
    sys.modules["torch.distributed.device_mesh"] = device_mesh_mod
    sys.modules["torch.testing"] = testing_mod
    return mod


try:
    import torch as _installed_torch  # noqa: F401
except ImportError:
    _build_cpu_torch_module()


def _build_diffusers_shim() -> None:
    diff_mod = types.ModuleType("diffusers")
    diff_mod.__version__ = "0.36.0"
    diff_mod.DiffusionPipeline = object

    cfg_mod = types.ModuleType("diffusers.configuration_utils")

    class FrozenDict(dict):
        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

    cfg_mod.FrozenDict = FrozenDict
    diff_mod.configuration_utils = cfg_mod

    models_mod = types.ModuleType("diffusers.models")
    ae_mod = types.ModuleType("diffusers.models.autoencoders")
    ae_wan_mod = types.ModuleType("diffusers.models.autoencoders.autoencoder_kl_wan")

    def unpatchify(video: CpuTensor, patch_size: int = 1) -> CpuTensor:
        if patch_size == 1:
            return video
        b, c_p2, f, h, w = video.shape
        c = c_p2 // (patch_size * patch_size)
        return (
            video.reshape(b, c, patch_size, patch_size, f, h, w)
            .permute(0, 1, 4, 5, 2, 6, 3)
            .reshape(b, c, f, h * patch_size, w * patch_size)
        )

    ae_wan_mod.unpatchify = unpatchify
    ae_mod.autoencoder_kl_wan = ae_wan_mod
    models_mod.autoencoders = ae_mod
    diff_mod.models = models_mod

    pipes_mod = types.ModuleType("diffusers.pipelines")
    wan_mod = types.ModuleType("diffusers.pipelines.wan")
    pipe_wan_mod = types.ModuleType("diffusers.pipelines.wan.pipeline_wan")
    pipe_wan_mod.prompt_clean = lambda text: " ".join(str(text).strip().split())
    wan_mod.pipeline_wan = pipe_wan_mod
    pipes_mod.wan = wan_mod
    diff_mod.pipelines = pipes_mod

    sys.modules["diffusers"] = diff_mod
    sys.modules["diffusers.configuration_utils"] = cfg_mod
    sys.modules["diffusers.models"] = models_mod
    sys.modules["diffusers.models.autoencoders"] = ae_mod
    sys.modules["diffusers.models.autoencoders.autoencoder_kl_wan"] = ae_wan_mod
    sys.modules["diffusers.pipelines"] = pipes_mod
    sys.modules["diffusers.pipelines.wan"] = wan_mod
    sys.modules["diffusers.pipelines.wan.pipeline_wan"] = pipe_wan_mod


try:
    import diffusers as _installed_diffusers  # noqa: F401
except ImportError:
    _build_diffusers_shim()


def _build_triton_shim() -> None:
    triton_mod = types.ModuleType("triton")
    triton_mod.__path__ = []  # type: ignore[attr-defined]
    tl_mod = types.ModuleType("triton.language")
    tl_mod.__path__ = []  # type: ignore[attr-defined]
    core_mod = types.ModuleType("triton.language.core")
    core_mod.extern = lambda fn: fn
    tl_mod.core = core_mod

    class _JitKernel:
        def __init__(self, fn: Any):
            self.fn = fn

        def __getitem__(self, grid: Any):
            return lambda *args, **kwargs: None

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.fn(*args, **kwargs)

    triton_mod.jit = lambda fn=None, **kwargs: (_JitKernel(fn) if fn is not None else lambda f: _JitKernel(f))
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    tl_mod.constexpr = Any
    for name in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "float16", "bfloat16", "float32", "float64"):
        setattr(tl_mod, name, name)
    tl_mod.pointer_type = lambda dt: f"ptr<{dt}>"
    triton_mod.language = tl_mod
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = tl_mod
    sys.modules["triton.language.core"] = core_mod


try:
    import triton as _installed_triton  # noqa: F401
except ImportError:
    _build_triton_shim()
