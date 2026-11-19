#!/bin/bash
# Multi-GPU / 1x1 A100 GPU validation for commit 65ed8f3:
#   1. In-cluster C++ build + ctest + doctor (via kairos_common.sh)
#   2. All 5 canonical Python domain test suites (Control, BlockExecutor, Memory, Collectives, Archive)
#   3. Real SD3.5-Medium A100 microbenchmarks (sharded across GPUs 0..2 when >=3 GPUs available):
#      - Exp A: Packed Joint Attention (flash_attn_varlen_func vs grouped_sdpa) & resolution-spread step-dilation boundary
#      - Exp B: FlashPS Fig. 10 3-stream async pipeline (encode_stream + compute_stream + vae_stream overlap)
#      - Exp C: Strict Tier-1 lossless CPU T5-XXL split t2i_pos/t2i_neg caching + cpu_threads=8
#   4. Optional chained 8-GPU formal-mixed-v2 serving evaluation when DITSERVE_CHAIN_FORMAL=1.
source "$(dirname "$0")/kairos_common.sh"

SIM_TMP=/tmp/ditserve-simtests-$DITSERVE_RUN_ID
PYTEST_JSON=/tmp/ditserve-pytest-$DITSERVE_RUN_ID.json
VAL_SHM=/dev/shm/ditserve-val-sd35-$DITSERVE_RUN_ID

trap 'rm -rf "$SIM_TMP" "$PYTEST_JSON" "$VAL_SHM" /dev/shm/ditserve-hetero-embeds "$DITSERVE_TRANSFER_DIR" "$RUN_DIR/ledgers" "$LEDGERS" 2>/dev/null || true; chmod -R a+rX "$RUN_DIR" 2>/dev/null || true' EXIT

nvidia-smi -L
df -h /dev/shm /shared
echo "memlock=$(ulimit -l) nproc=$(nproc) cgroup_memory_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"

# Operator storage gate (stricter than AGENTS 810G / 82% stop line).
used=$(df -BG /shared | awk 'NR==2{gsub("G","",$3); print $3}')
pct=$(df -P /shared | awk 'NR==2{gsub("%","",$5); print $5}')
if [ "$used" -gt 790 ] || [ "$pct" -ge 80 ]; then
  echo "shared storage gate: used=${used}G pct=${pct}%"
  exit 4
fi

# 1. Build simulation-tests helper binary in /tmp and run all 5 Python domain test suites in an isolated child process
SRC_STAGE=$CODE/builds/.src-$DITSERVE_BUILD
mkdir -p "$SIM_TMP"
/opt/venv/bin/python - "$SRC_STAGE" "$SIM_TMP/simulation-tests" <<'PY'
import pathlib, shutil, subprocess, sys
src_dir = pathlib.Path(sys.argv[1]).resolve()
out_exe = pathlib.Path(sys.argv[2]).resolve()
gxx = shutil.which("g++")
assert gxx, "g++ missing in container"
shim_inc = out_exe.parent / "shim_include"
if not pathlib.Path("/usr/include/openssl/sha.h").is_file():
    (shim_inc / "openssl").mkdir(parents=True, exist_ok=True)
    (shim_inc / "openssl" / "sha.h").write_text(
        '#pragma once\n#include <cstddef>\nextern "C" {\n'
        "#define SHA_LBLOCK 16\n#define SHA256_CBLOCK (SHA_LBLOCK*4)\n#define SHA256_DIGEST_LENGTH 32\n"
        "typedef unsigned int SHA_LONG;\n"
        "typedef struct SHA256state_st { SHA_LONG h[8]; SHA_LONG Nl, Nh; SHA_LONG data[SHA_LBLOCK]; unsigned int num, md_len; } SHA256_CTX;\n"
        "int SHA256_Init(SHA256_CTX *c);\nint SHA256_Update(SHA256_CTX *c, const void *data, size_t len);\n"
        "int SHA256_Final(unsigned char *md, SHA256_CTX *c);\n"
        "unsigned char *SHA256(const unsigned char *d, size_t n, unsigned char *md);\n}\n"
    )
inc = ["-I", str(src_dir / "include"), "-I", str(src_dir / "benchmarks/include"), "-isystem", str(src_dir / ".cache/deps/nlohmann3.11.3")]
if shim_inc.exists():
    inc.extend(["-isystem", str(shim_inc)])
flags = [
    "-std=c++17", "-O3", "-DNDEBUG", "-ffp-contract=off", "-pthread",
    f'-DDITSERVE_PROJECT_ROOT="{src_dir}"', '-DDITSERVE_VERSION="0.4.0"',
]
rels = [
    "src/ditserve/types.cpp",
    "src/ditserve/runtime/random.cpp",
    "src/ditserve/runtime/process.cpp",
    "src/ditserve/control/budget.cpp",
    "src/ditserve/control/dispatch.cpp",
    "src/ditserve/control/cluster.cpp",
    "src/ditserve/control/policy.cpp",
    "benchmarks/src/cluster_sim.cpp",
    "benchmarks/tests/test_cluster.cpp",
]
subprocess.run([gxx, *flags, *inc, *(str(src_dir / r) for r in rels), "-pthread", "-o", str(out_exe)], check=True)
PY

echo "== running 5 Python functional domain test suites"
RUNNER_PY="$SIM_TMP/run_domain_suites.py"
cat <<'PY' > "$RUNNER_PY"
from __future__ import annotations

import builtins
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import traceback
import types

# Force conftest.py's deterministic NumPy-backed shims for unit tests even inside
# the GPU container so all 5 domain suites run identically to host CPU pytest.
_SHIM_ROOTS = {"torch", "diffusers", "triton", "flash_attn"}


class _ForceConftestShims(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in _SHIM_ROOTS and root not in sys.modules:
            raise ModuleNotFoundError(f"Deferring {fullname} to conftest deterministic CPU shim")
        return None


sys.meta_path.insert(0, _ForceConftestShims())

_NOTSET = object()


class _ApproxScalar:
    def __init__(self, expected, rel=None, abs=None, nan_ok=False):
        self.expected = expected
        self.rel = 1e-6 if rel is None else float(rel)
        self.abs = 1e-12 if abs is None else float(abs)
        self.nan_ok = nan_ok

    def _matches(self, actual, expected) -> bool:
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                return False
            return all(self._matches(a, e) for a, e in zip(actual, expected))
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual.keys()) != set(expected.keys()):
                return False
            return all(self._matches(actual[k], expected[k]) for k in expected)
        actual_f = float(actual)
        expected_f = float(expected)
        if math.isnan(expected_f) and math.isnan(actual_f):
            return bool(self.nan_ok)
        tol = max(self.abs, self.rel * builtins.abs(expected_f))
        return builtins.abs(actual_f - expected_f) <= tol

    def __eq__(self, actual) -> bool:
        try:
            return self._matches(actual, self.expected)
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


class _ExcInfo:
    def __init__(self):
        self.value = None
        self.type = None
        self.tb = None


class _RaisesContext:
    def __init__(self, expected_exc, match=None):
        self.expected_exc = expected_exc
        self.match = match
        self.excinfo = _ExcInfo()

    def __enter__(self):
        return self.excinfo

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected_exc!r}")
        if not issubclass(exc_type, self.expected_exc):
            return False
        self.excinfo.type = exc_type
        self.excinfo.value = exc_val
        self.excinfo.tb = exc_tb
        if self.match is not None and not re.search(self.match, str(exc_val)):
            raise AssertionError(f"Regex pattern {self.match!r} does not match {str(exc_val)!r}")
        return True


class MonkeyPatch:
    def __init__(self):
        self._setattr = []
        self._setitem = []
        self._env = []
        self._syspath = None
        self._cwd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.undo()

    @classmethod
    def context(cls):
        return cls()

    def _resolve_dotted(self, target: str):
        parts = target.split(".")
        for i in range(len(parts) - 1, 0, -1):
            mod_name = ".".join(parts[:i])
            try:
                obj = importlib.import_module(mod_name)
                for attr in parts[i:-1]:
                    obj = getattr(obj, attr)
                return obj, parts[-1]
            except Exception:
                continue
        raise ImportError(f"Cannot resolve monkeypatch target {target!r}")

    def setattr(self, target, name, value=_NOTSET, raising=True):
        if value is _NOTSET:
            if not isinstance(target, str):
                raise TypeError("use setattr(target, name, value) or setattr('dotted.path', value)")
            value = name
            target, name = self._resolve_dotted(target)
        oldval = getattr(target, name, _NOTSET)
        if raising and oldval is _NOTSET:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._setattr.append((target, name, oldval))
        setattr(target, name, value)

    def delattr(self, target, name=_NOTSET, raising=True):
        if name is _NOTSET:
            target, name = self._resolve_dotted(target)
        oldval = getattr(target, name, _NOTSET)
        if raising and oldval is _NOTSET:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._setattr.append((target, name, oldval))
        if oldval is not _NOTSET:
            delattr(target, name)

    def setitem(self, dic, name, value):
        self._setitem.append((dic, name, dic.get(name, _NOTSET)))
        dic[name] = value

    def delitem(self, dic, name, raising=True):
        if name not in dic:
            if raising:
                raise KeyError(name)
            return
        self._setitem.append((dic, name, dic[name]))
        del dic[name]

    def setenv(self, name: str, value: str, prepend=None):
        if prepend and name in os.environ:
            value = f"{value}{prepend}{os.environ[name]}"
        self._env.append((name, os.environ.get(name, _NOTSET)))
        os.environ[name] = str(value)

    def delenv(self, name: str, raising=True):
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        self._env.append((name, os.environ[name]))
        del os.environ[name]

    def syspath_prepend(self, path):
        if self._syspath is None:
            self._syspath = list(sys.path)
        sys.path.insert(0, str(path))

    def chdir(self, path):
        if self._cwd is None:
            self._cwd = os.getcwd()
        os.chdir(path)

    def undo(self):
        while self._setattr:
            target, name, oldval = self._setattr.pop()
            if oldval is _NOTSET:
                try:
                    delattr(target, name)
                except AttributeError:
                    pass
            else:
                setattr(target, name, oldval)
        while self._setitem:
            dic, name, oldval = self._setitem.pop()
            if oldval is _NOTSET:
                dic.pop(name, None)
            else:
                dic[name] = oldval
        while self._env:
            name, oldval = self._env.pop()
            if oldval is _NOTSET:
                os.environ.pop(name, None)
            else:
                os.environ[name] = oldval
        if self._syspath is not None:
            sys.path[:] = self._syspath
            self._syspath = None
        if self._cwd is not None:
            os.chdir(self._cwd)
            self._cwd = None


try:
    import pytest as _pytest_mod  # type: ignore
except ModuleNotFoundError:
    _pytest_mod = types.ModuleType("pytest")
    _pytest_mod.__spec__ = importlib.machinery.ModuleSpec("pytest", loader=None)
    _pytest_mod.approx = lambda expected, rel=None, abs=None, nan_ok=False: _ApproxScalar(
        expected, rel=rel, abs=abs, nan_ok=nan_ok
    )
    _pytest_mod.raises = lambda expected_exc, *, match=None: _RaisesContext(expected_exc, match=match)
    _pytest_mod.MonkeyPatch = MonkeyPatch
    sys.modules["pytest"] = _pytest_mod


def _call_with_fixtures(fn, test_label: str) -> None:
    sig = inspect.signature(fn)
    mp = _pytest_mod.MonkeyPatch()
    tmp_dir = None
    saved_argv = list(sys.argv)
    kwargs = {}
    try:
        for pname in sig.parameters:
            if pname == "tmp_path":
                tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"ditserve-ut-{test_label[:24]}-"))
                kwargs["tmp_path"] = tmp_dir
            elif pname == "monkeypatch":
                kwargs["monkeypatch"] = mp
            else:
                raise RuntimeError(f"Unsupported fixture {pname!r} in {test_label}")
        fn(**kwargs)
    finally:
        sys.argv[:] = saved_argv
        mp.undo()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    src_dir = pathlib.Path(sys.argv[1]).resolve()
    out_json = pathlib.Path(sys.argv[2]).resolve()
    tests_dir = src_dir / "tests" / "python"

    conftest_path = tests_dir / "conftest.py"
    if conftest_path.is_file():
        cspec = importlib.util.spec_from_file_location("conftest", conftest_path)
        assert cspec is not None and cspec.loader is not None
        cmod = importlib.util.module_from_spec(cspec)
        sys.modules["conftest"] = cmod
        cspec.loader.exec_module(cmod)

    suites = [
        tests_dir / "test_domain_control.py",
        tests_dir / "test_domain_block_executor.py",
        tests_dir / "test_domain_memory.py",
        tests_dir / "test_domain_collectives.py",
        tests_dir / "test_domain_archive.py",
    ]
    sys.argv = ["domain-runner", *(str(s) for s in suites)]

    t0 = time.perf_counter()
    passed = 0
    failed: list[str] = []
    per_suite: dict[str, dict] = {}

    for spath in suites:
        st0 = time.perf_counter()
        spassed = 0
        mspec = importlib.util.spec_from_file_location(spath.stem, spath)
        assert mspec is not None and mspec.loader is not None
        mod = importlib.util.module_from_spec(mspec)
        sys.modules[spath.stem] = mod
        mspec.loader.exec_module(mod)

        for name, obj in list(mod.__dict__.items()):
            if inspect.isfunction(obj) and name.startswith("test_") and obj.__module__ == mod.__name__:
                try:
                    _call_with_fixtures(obj, f"{spath.stem}.{name}")
                    passed += 1
                    spassed += 1
                except BaseException as exc:
                    failed.append(f"{spath.name}::{name}: {exc}\n{traceback.format_exc()}")
            elif inspect.isclass(obj) and name.startswith("Test") and obj.__module__ == mod.__name__:
                if hasattr(obj, "setup_class"):
                    obj.setup_class()
                try:
                    seen_methods: list[str] = []
                    seen_set: set[str] = set()
                    for base in reversed(obj.__mro__):
                        if base is object:
                            continue
                        for mname, mobj in base.__dict__.items():
                            if mname.startswith("test_") and callable(mobj) and mname not in seen_set:
                                seen_set.add(mname)
                                seen_methods.append(mname)
                    for mname in seen_methods:
                        inst = obj()
                        bound = getattr(inst, mname)
                        if hasattr(inst, "setup_method"):
                            sm = inst.setup_method
                            sm(bound) if len(inspect.signature(sm).parameters) >= 1 else sm()
                        try:
                            _call_with_fixtures(bound, f"{name}.{mname}")
                            passed += 1
                            spassed += 1
                        except BaseException as exc:
                            failed.append(f"{spath.name}::{name}::{mname}: {exc}\n{traceback.format_exc()}")
                        finally:
                            if hasattr(inst, "teardown_method"):
                                tm = inst.teardown_method
                                tm(bound) if len(inspect.signature(tm).parameters) >= 1 else tm()
                finally:
                    if hasattr(obj, "teardown_class"):
                        obj.teardown_class()

        per_suite[spath.name] = {"passed": spassed, "elapsed_s": round(time.perf_counter() - st0, 3)}
        print(f"  {spath.name}: {spassed} passed ({per_suite[spath.name]['elapsed_s']}s)", flush=True)

    elapsed_s = round(time.perf_counter() - t0, 3)
    rc = 0 if not failed else 1
    summary_line = f"{passed} passed in {elapsed_s}s" if rc == 0 else f"{len(failed)} failed, {passed} passed in {elapsed_s}s"
    payload = {
        "returncode": rc,
        "elapsed_s": elapsed_s,
        "passed": passed,
        "failed_count": len(failed),
        "suites": [s.name for s in suites],
        "per_suite": per_suite,
        "summary_lines": [summary_line],
    }
    if failed:
        payload["failures"] = failed[:10]
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("domain_suites:", json.dumps({k: v for k, v in payload.items() if k != "failures"}), flush=True)
    if failed:
        for item in failed:
            print("FAIL:", item, file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
PY

env -u DITSERVE_ROOT \
  CUDA_VISIBLE_DEVICES="" \
  DITSERVE_CORE="$BUILD/bin/ditserve" \
  DITSERVE_BIN="$BUILD/bin/ditserve" \
  DITSERVE_SIMULATION_TESTS="$SIM_TMP/simulation-tests" \
  PYTHONPATH="$PACKAGE/python:$PACKAGE/benchmarks:$SRC_STAGE/tests/python" \
  OMP_NUM_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  /opt/venv/bin/python "$RUNNER_PY" "$SRC_STAGE" "$PYTEST_JSON"

# 2. Run real A100 GPU SD3.5-Medium microbenchmark & validation (auto-shards across GPUs 0..2 when >=3 GPUs present)
export PYTHONPATH=$PACKAGE/python:$PACKAGE/benchmarks
export DITSERVE_CORE=$BUILD/bin/ditserve
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPORT_NAME=${DITSERVE_REPORT_NAME:-20261115-sd35-cb-flashps-validation.md}
timeout "${DITSERVE_VAL_TIMEOUT_S:-2400}s" /opt/venv/bin/python -u "$SCRIPTS/run_val_sd35_cb_flashps.py" \
  --output-dir "$RUN_DIR" \
  --report-name "$REPORT_NAME" \
  --pytest-receipt "$PYTEST_JSON" \
  --work-root "$VAL_SHM"

rm -rf "$VAL_SHM" /dev/shm/ditserve-hetero-embeds 2>/dev/null || true

# 3. Optional chained 8-GPU formal-mixed-v2 serving evaluation on the same 8x A100 allocation
if [ "${DITSERVE_CHAIN_FORMAL:-0}" = "1" ]; then
  echo "== chaining 8-GPU formal-mixed-v2 serving evaluation (${DITSERVE_RUN_ID}-serve)"
  DITSERVE_RUN_ID="${DITSERVE_RUN_ID}-serve" bash "$SCRIPTS/kairos_formal_benchmark.sh"
fi
