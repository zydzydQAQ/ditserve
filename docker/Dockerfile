# Lean runtime: C++ controller + resident Python workers on the locked torch 2.10.0+cu126 stack.
# Historical recipe: its offline container build guide was removed. Building this image locally (WSL)
# or on jjjkairos is prohibited by /mnt/e/000_code/zyd-2026/AGENTS.md; new images need a user-approved build source.
ARG PYTHON_BASE=python:3.10.20-slim-bookworm@sha256:019e31cc1f52e4139d338bca44f27b71339a9da261738fd80fab9cf4595152fc
ARG BASE_IMAGE
FROM ${PYTHON_BASE} AS runtime-system
# GCC/G++ and /usr/local/include/python3.10 are required by torch.compile.
# CUDA libraries come from the locked PyTorch wheels; no CUDA SDK/nvcc is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates gcc g++ libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

FROM runtime-system AS dependencies
ARG DITSERVE_WHEELHOUSE_URL=
COPY docker/artifacts-runtime.json /tmp/artifacts-runtime.json
COPY docker/install_dependencies.py /tmp/install_dependencies.py
# Only Triton's static CUPTI archives are pruned; package tests/ stay (numpy.testing imports numpy._core.tests).
RUN python /tmp/install_dependencies.py /opt/venv /tmp/artifacts-runtime.json \
    && rm -f /opt/venv/lib/python3.10/site-packages/triton/backends/nvidia/lib/cupti/*.a \
    && /opt/venv/bin/python -m pip check

FROM runtime-system AS core-builder
RUN apt-get update && apt-get install -y --no-install-recommends cmake make libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build/source
COPY . .
RUN cmake -S . -B /build/core -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build /build/core --parallel 4 \
    && ctest --test-dir /build/core --output-on-failure \
    && cmake --install /build/core --prefix /opt/ditserve --strip

FROM runtime-system AS runtime-base
ARG GIT_REVISION=unknown
LABEL org.opencontainers.image.title=ditserve org.opencontainers.image.revision=${GIT_REVISION}
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=core-builder /opt/ditserve /opt/ditserve
# DITSERVE_ROOT holds results/, .state/ and .cache/results-work/; mount it to keep them.
ENV PATH=/opt/ditserve/bin:/opt/venv/bin:$PATH \
    DITSERVE_PYTHON=/opt/venv/bin/python DITSERVE_ROOT=/data \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/data/cache/huggingface TORCH_HOME=/models/torch \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    TRITON_CACHE_DIR=/data/cache/triton \
    MPLCONFIGDIR=/data/cache/matplotlib \
    HOME=/data USER=ditserve LOGNAME=ditserve \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN ditserve --version && ditserve doctor \
    && mkdir -p /data/results /data/cache /models && chmod -R a+rwX /data
WORKDIR /data
CMD ["ditserve", "--help"]

FROM runtime-base AS development
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake make libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && cmake --version && g++ --version && pkg-config --modversion openssl
WORKDIR /src
CMD ["bash"]

# Target: external-serving (historical superseded recipe)
# 状态：历史配方，已被取代，不得构建。
# - 原默认基础镜像 xdit-dev-016e1ca-slim（sha256:2009ea26…）已于 2026-11-04 经授权删除，
#   AGENTS.md 规定不得再用于后续实验，因此这里不再给出任何默认 BASE_IMAGE。
# - 本配方补齐的 common-control-plane-v1 共 22 个轻量依赖（fastapi/uvicorn/pandas/pyzmq/aenum 等）
#   与 13 个冻结核心包，已按相同版本全部包含在当前唯一镜像
#   192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3
#   （sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05，另含 vLLM 0.17.0+cu126.sm80）
#   的构建基线中（r3 构建记录 archive-r3-baseline.json）。r3 的依赖闭包与本配方不同，
#   不能把本配方改指 r3 当作新配方使用；依赖变更须由用户指定新的构建来源与锁定方案。
# - 禁止在本机 WSL、Docker Desktop WSL 后端、jjjkairos 登录节点或任何集群节点/作业容器构建。
#   详见 /mnt/e/000_code/zyd-2026/AGENTS.md 与 configs/benchmarks/external-image-build-handoff-v1.json。
# BASE_IMAGE 故意没有默认值：未显式提供时 FROM 的基础镜像名为空，构建在解析阶段即失败。
# 单文件：不需要 COPY 项目源码、requirements 或其他构建上下文文件。
# 未包含 vLLM：其原生 CUDA12.9 路线与现有 cu126 的 ABI 尚待单独验证。
FROM ${BASE_IMAGE:-runtime-base} AS external-serving
# Re-declared so the guard can read the value that was actually supplied.
ARG BASE_IMAGE
RUN case "${BASE_IMAGE}" in \
      *2009ea26f63bcf0360ce7b4d71ed508c91c0702c97dd97e372cecfb672a818a1*|*xdit-dev-016e1ca-slim*) \
        echo "Refusing deleted slim base ${BASE_IMAGE}; it was deleted 2026-11-04 and must not be reused (see AGENTS.md)." >&2; exit 1 ;; \
      *@sha256:????????????????????????????????????????????????????????????????) ;; \
      *) echo "BASE_IMAGE must be an explicit digest-pinned reference approved by the user; this recipe is superseded (see AGENTS.md and configs/benchmarks/external-image-build-handoff-v1.json)." >&2; exit 1 ;; \
    esac
LABEL org.ditserve.dependency-profile="20261103-common-control-plane-v1"

RUN /opt/venv/bin/python - <<'PYDEPS'
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import urlsplit

# 保持现有核心计算栈；下面是新增/补齐的服务依赖及其固定闭包。
CORE = dict(line.split("==", 1) for line in """torch==2.10.0+cu126
triton==3.6.0
numpy==2.2.6
diffusers==0.36.0
cache-dit==1.5.0
transformers==4.57.6
accelerate==1.12.0
safetensors==0.8.0
distvae==0.1.0
protobuf==7.36.2
sentencepiece==0.2.2
einops==0.8.2
huggingface-hub==0.36.2""".splitlines())
LIGHT = dict(line.split("==", 1) for line in """fastapi==0.162.0
uvicorn==0.54.0
pandas==2.3.3
pyzmq==27.2.0
aenum==3.1.16
starlette==1.7.0
pydantic==2.14.2
pydantic-core==2.50.2
typing-extensions==4.16.0
typing-inspection==0.4.4
annotated-types==0.8.0
anyio==4.16.0
click==8.5.0
h11==0.16.0
python-dateutil==2.9.0.post0
pytz==2026.6
tzdata==2026.6
six==1.17.0
idna==3.20
exceptiongroup==1.3.1
annotated-doc==0.0.4
opentelemetry-api==1.44.0""".splitlines())

def require(ok, message):
    if not ok:
        raise SystemExit(message)

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()

def inventory():
    result = {}
    for dist in metadata.distributions():
        name = normalize(dist.metadata["Name"])
        require(name not in result, "Duplicate installed distribution: " + name)
        result[name] = dist.version
    return result

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)

require(sys.prefix == "/opt/venv", "Expected the existing /opt/venv interpreter")
require(platform.python_version() == "3.10.20", "Expected base Python 3.10.20")
require(sys.platform == "linux" and platform.machine() == "x86_64", "Expected Linux amd64")
libc, version = platform.libc_ver()
require(libc == "glibc" and tuple(map(int, version.split(".")[:2])) >= (2, 26), "Selected wheels require glibc >= 2.26")
before = inventory()
for name, version in CORE.items():
    require(before.get(name) == version, "Base core version mismatch: " + name)
import torch
require(torch.version.cuda == "12.6", "Expected existing Torch CUDA 12.6")
pip("check")

out = Path("/opt/ditserve-image-deps/20261103-common-v1")
out.mkdir(parents=True, exist_ok=False)
(out / "baseline.json").write_text(json.dumps(before, indent=2, sort_keys=True) + "\n")
requirements = out / "requirements.txt"
constraints = out / "constraints.txt"
report = out / "resolution.json"
lock = out / "install.lock"
requirements.write_text("".join(f"{name}=={version}\n" for name, version in LIGHT.items()))
protected = {name: version for name, version in before.items() if name not in LIGHT}
protected.update(CORE)
constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(protected.items())))

# 先解析并核对实际变更；不允许隐式更新核心包或引入清单外依赖。
pip("install", "--dry-run", "--no-cache-dir", "--only-binary=:all:",
    "--report", str(report), "-r", str(requirements), "-c", str(constraints))
changes = {}
for item in json.loads(report.read_text()).get("install", []):
    name = normalize(item["metadata"]["name"])
    version = item["metadata"]["version"]
    require(name not in changes, "Duplicate installation in plan: " + name)
    require(name in LIGHT and LIGHT[name] == version, "Unapproved dependency: " + name)
    info = item.get("download_info", {})
    require(urlsplit(info.get("url", "")).path.endswith(".whl"), "Binary wheel required: " + name)
    digest = info.get("archive_info", {}).get("hashes", {}).get("sha256", "")
    require(re.fullmatch(r"[0-9a-fA-F]{64}", digest) is not None, "Missing wheel SHA256: " + name)
    changes[name] = {"version": version, "wheel_sha256": digest.lower()}
lock.write_text("".join(f"{name}=={item['version']} --hash=sha256:{item['wheel_sha256']}\n"
                        for name, item in sorted(changes.items())))
report.unlink()  # 镜像不保留可能含私有下载地址的原始解析报告。
if changes:
    # --no-deps 仅用于上面已经完整解析、核对且固定哈希的变更集合。
    pip("install", "--no-cache-dir", "--only-binary=:all:", "--no-deps",
        "--require-hashes", "-r", str(lock))

# 校验依赖、原环境保留情况和实际导入；构建时不运行 GPU 任务。
after = inventory()
for name, version in protected.items():
    require(after.get(name) == version, "Protected package changed: " + name)
for name, version in LIGHT.items():
    require(after.get(name) == version, "Missing or wrong light dependency: " + name)
require(set(before) <= set(after), "An existing package disappeared")
require(set(after) - set(before) <= set(LIGHT), "Unexpected new dependency")
actual = {name: version for name, version in after.items() if before.get(name) != version}
require(actual == {name: item["version"] for name, item in changes.items()}, "Installed changes differ from approved plan")
pip("check")
for module in ("torch", "triton", "numpy", "diffusers", "transformers", "accelerate",
               "safetensors", "fastapi", "uvicorn", "starlette", "pydantic", "pandas",
               "zmq", "aenum", "exceptiongroup"):
    importlib.import_module(module)
from fastapi import FastAPI
import pandas as pd
import uvicorn
import zmq
uvicorn.Server(uvicorn.Config(FastAPI(), host="127.0.0.1", port=8080))
require(pd.DataFrame({"value": [1, 2]}).value.sum() == 3, "pandas/NumPy check failed")
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.close(linger=0)
context.term()
(out / "installed.json").write_text(json.dumps(after, indent=2, sort_keys=True) + "\n")
(out / "summary.json").write_text(json.dumps({
    "profile": "common-control-plane-v1",
    "changes": changes,
    "core_unchanged": True,
    "pip_check": "passed",
    "cpu_imports": "passed",
    "gpu_validation": "pending",
    "full_service_validation": "pending"
}, indent=2, sort_keys=True) + "\n")
print("Dependencies installed; frozen core preserved. GPU/full-service validation remains pending.")
PYDEPS

FROM runtime-base AS runtime
