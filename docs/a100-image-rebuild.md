# 单张 A100 40GB：重建 DITSERVE xDiT 运行镜像

本文用于在另一台只有一张 A100 40GB 的 Linux x86_64 服务器上，从 GitHub `xdit` 分支直接重建最新运行镜像。镜像包含 C++ 控制器、常驻 Python workers、固定版本的 xDiT/xFuser 源码及 Python 运行环境；不包含模型权重、实验结果或宿主机的 CUDA Toolkit。

## 1. 不要从旧镜像增量修补

另一台服务器上的 `ditserve:rewrite-runtime`（镜像 ID `ecb1055e440e`）和集群的 `ditserve:8a100-224099f` 都不是新构建的基础镜像。当前源码已经把 xDiT `16c34e7f070de65d9d2c67a6d74e0007dd8b426a` 的 `xfuser` 包纳入 `workers/xfuser/`，并更新了 Python 锁文件。

不要通过 `docker commit`、只替换 `/opt/ditserve` 或在旧容器中临时 `pip install` 来升级，这会使代码和依赖失去一致性。保留旧镜像作为回退，从 `xdit` 源码完整构建并使用新标签即可；构建前不需要删除旧镜像或清理 Docker 缓存。

## 2. 宿主机的最小要求

| 项目 | 要求 |
| --- | --- |
| 系统与架构 | Linux，`uname -m` 输出 `x86_64`；锁文件使用 CPython 3.10 的 x86_64 manylinux wheel |
| Git | 能克隆 GitHub 仓库 |
| Docker | Docker Engine，支持 `docker build`；不需要 Docker Compose |
| NVIDIA 驱动 | 支持 CUDA 12.6，建议 560 系列或更新；`nvidia-smi` 能正常显示 A100 |
| NVIDIA Container Toolkit | 已配置 Docker 的 GPU 运行支持；仅 GPU 冒烟和正式运行需要，镜像构建本身不需要 GPU |
| 网络 | 能访问 Docker Hub、`files.pythonhosted.org`、`download-r2.pytorch.org` 和 `raw.githubusercontent.com`，包括其重定向地址 |
| 磁盘 | 建议在 Docker 根目录所在文件系统保留至少 25 GiB 可用空间，用于基础镜像、中间阶段和构建缓存 |

宿主机不需要安装 Python、CMake、GCC/G++、OpenSSL 开发包、PyTorch、xDiT、CUDA Toolkit 或 `nvcc`；这些都由镜像构建阶段提供，或由锁定的 wheel 提供。单卡与多卡的镜像依赖相同，不需要为了单卡另做一份依赖清单。

检查 Docker 磁盘空间和 GPU：

```bash
df -h "$(docker info --format '{{.DockerRootDir}}')"
nvidia-smi
```

## 3. 镜像内的固定最简依赖

### 构建工具和基础系统

`Dockerfile` 自动处理以下依赖，不需要在宿主机执行 `apt install`：

- 固定 digest 的 `python:3.10.20-slim-bookworm`。
- C++ 构建：C++17、CMake 3.16+、Make、GCC/G++、Threads、OpenSSL（`libssl-dev`）。
- 运行系统库：`ca-certificates`、`gcc`、`g++`、`libgomp1`、`libssl3`。保留编译器和 Python 头文件是因为 `torch.compile`/Triton 在运行时仍需要它们。精简 `runtime` 不含 CMake 和 OpenSSL 开发头文件；需要在容器内重新编译 C++ 时，用第 8 节的 `development` target 额外安装 `cmake`、`make`、`libssl-dev`、`pkg-config`。
- `nlohmann/json 3.11.3` 单头文件，SHA256 固定为 `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6`。全新克隆首次构建时由 CMake 下载并校验。
- 不需要 CUDA Toolkit、`nvcc` 或单独的 NCCL 安装；CUDA 运行库和 NCCL 来自锁定的 PyTorch/NVIDIA wheel。

### Python 运行环境

关键直接依赖如下。这里用于说明环境边界，**不要手工复制成新的 requirements 文件来替代锁文件**。

| 类别 | 固定版本 |
| --- | --- |
| 张量计算 | `torch==2.10.0+cu126`、`torchvision==0.25.0+cu126`、`triton==3.6.0` |
| 模型框架 | `diffusers==0.36.0`、`transformers==4.57.6`、`accelerate==1.12.0`、`cache-dit==1.5.0` |
| xDiT 新增直接依赖 | `yunchang==0.6.4`、`distvae==0.1.0`、`einops==0.8.2`、`beautifulsoup4==4.15.0`、`av==17.1.0`、`peft==0.21.2` |
| 文本与模型文件 | `tokenizers==0.22.2`、`sentencepiece==0.2.2`、`protobuf==7.36.2`、`safetensors==0.8.0`、`ftfy==6.3.1` |
| 质量、数值与报告 | `numpy==2.2.6`、`scipy==1.15.3`、`lpips==0.1.4`、`openpyxl==3.1.5`、`matplotlib==3.10.8` |

相对 `224099f` 的锁文件，xDiT 迁移新增 7 个 wheel：上表中的 6 个直接依赖，加上 `beautifulsoup4` 的传递依赖 `soupsieve==2.10`。`xfuser` 源码已经随仓库和镜像安装，不需要再执行 `pip install xfuser` 或克隆 xDiT。

完整且唯一的安装清单是 [`docker/artifacts-runtime.json`](../docker/artifacts-runtime.json)：共 80 个 wheel，约 3.49 GiB 下载量，包含全部直接/传递依赖、固定 URL、文件名、字节数和 SHA256。[`docker/install_dependencies.py`](../docker/install_dependencies.py) 使用 `--no-deps --require-hashes` 逐项安装，最后执行 `pip check`。`configs/requirements-v100.txt` 和 `configs/requirements-previs.txt` 是其他环境用的清单，不用于重建此运行镜像。

当前路径使用 PyTorch SDPA，不要求安装 `flash-attn`、`sageattention`、`aiter`、`torchao`、`onediff`、Ray、Mistral 或 LongLive 专用环境。不要为了构建最简运行镜像额外安装这些可选后端。运行镜像也不包含 `imageio`、`imageio-ffmpeg`、测试、文档、结果和模型权重。

## 4. 直接联网构建

在同一个 shell 中执行：

```bash
git clone --single-branch --branch xdit \
  https://github.com/zydzydQAQ/ditserve.git
cd ditserve

REV="$(git rev-parse HEAD)"
IMAGE="ditserve:xdit-$(git rev-parse --short=7 "$REV")"

docker build --target runtime \
  --build-arg GIT_REVISION="$REV" \
  -t "$IMAGE" .

printf 'image=%s\nrevision=%s\n' "$IMAGE" "$REV"
```

不需要传入旧镜像、`PYTHON_BASE` 或 `DITSERVE_WHEELHOUSE_URL`。默认流程会拉取 `Dockerfile` 中固定 digest 的 Python 基础镜像，按锁文件下载并校验全部 wheel，编译 C++ 控制器并运行 `ctest`，最后执行 `ditserve --version`、`pip check` 和 `ditserve doctor`。

`REV` 和 `IMAGE` 只保存在当前 shell；换终端后需重新设置，或把命令中的 `$IMAGE` 替换为构建时打印的完整镜像标签。

## 5. 构建后核验

先确认镜像标签、大小和源码 revision：

```bash
docker image inspect "$IMAGE" \
  --format '{{.Id}} {{.Size}} {{index .Config.Labels "org.opencontainers.image.revision"}}'
docker run --rm "$IMAGE" ditserve --version
docker run --rm --gpus all "$IMAGE" ditserve doctor
```

再在真实容器内验证关键 Python 包、固定 xDiT 适配层和单张 GPU：

```bash
docker run --rm -i --gpus all \
  -e PYTHONPATH=/opt/ditserve/share/ditserve/python \
  --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY'
import torch
import accelerate
import av
import bs4
import cache_dit
import diffusers
import distvae
import einops
import peft
import torchvision
import transformers
import xfuser
import yunchang
from ditserve.models.xdit import engine_info

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "expected exactly one visible GPU"
assert engine_info()["revision"] == "16c34e7f070de65d9d2c67a6d74e0007dd8b426a"
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("xDiT:", engine_info())
PY
```

该冒烟不加载模型权重，也不代表所有模型、分辨率和 batch 配置都能在 40GB 显存中运行。模型文件需在容器外准备好并挂载到 `/models`，容器默认保持 Hugging Face 离线模式。

## 6. 单卡正式运行时的必要参数

把 `/host/ditserve-data`、`/host/models` 和 `/data/study.json` 换成实际路径。配置必须已经填写真实模型路径、GPU UUID 等字段，不能直接运行占位模板。

```bash
docker run --rm --gpus all --pid=host --user "$(id -u):$(id -g)" \
  -v /host/ditserve-data:/data \
  -v /host/models:/models:ro \
  "$IMAGE" ditserve run \
  --config /data/study.json --run-id trial-01 --allow-gpu
```

`--pid=host` 是 GPU 隔离检查所需参数。单卡运行不需要多卡 SP/NCCL 使用的 `--ipc=host --shm-size=32g --ulimit memlock=-1`。运行结果、状态和缓存位于挂载的 `/data`，不会因容器删除而丢失。

## 7. 离线或受限网络构建

仅有 wheelhouse 并不等于完全离线。还需准备：

1. `Dockerfile` 固定 digest 对应的 Python 基础镜像，并通过 `PYTHON_BASE` 指向本地标签。
2. 锁文件中的全部 80 个 wheel，文件名必须与锁文件一致，通过 `DITSERVE_WHEELHOUSE_URL` 提供；安装时仍会校验 SHA256。
3. 若构建环境不能访问 `raw.githubusercontent.com`，预置 `.cache/deps/nlohmann3.11.3/nlohmann/json.hpp`，其 SHA256 必须与上文一致。

wheelhouse 服务及旧版 Docker builder 的命令见 [通用容器说明](container.md)。不要用未锁定的 `pip install -U`、重新解析最新版依赖或删除 NVIDIA wheel 来绕过网络/空间问题。

## 8. 带 C++ 工具链的持久开发镜像

默认 `runtime` 保持精简，不含 CMake 和 OpenSSL 开发头文件。需要在容器内修改并重新编译 C++ 时，在这台允许构建镜像的 Linux 服务器上使用 `development` target；它会额外安装 `cmake`、`make`、`libssl-dev` 和 `pkg-config`，并保留完整 Python 运行环境。不要用临时容器里的 `pip install` 代替该构建。

```bash
REV="$(git rev-parse HEAD)"
DEV_IMAGE="ditserve:xdit-dev-$(git rev-parse --short=7 "$REV")"
docker build --target development \
  --build-arg GIT_REVISION="$REV" \
  -t "$DEV_IMAGE" .

docker run --rm "$DEV_IMAGE" bash -lc \
  'cmake --version && g++ --version && pkg-config --modversion openssl && ditserve --version'
```

开发时把源码挂载到 `/src`，构建目录放到已持久化的 `/data`。CMake 还会在源码的 `.cache/deps` 下载并校验 nlohmann/json，因此这里不把源码挂成只读；离线构建需按第 7 节预置该依赖。把 `/host/ditserve-data` 换为实际数据目录：

```bash
docker run --rm -v "$PWD":/src -v /host/ditserve-data:/data \
  "$DEV_IMAGE" bash -lc \
  'cmake -S /src -B /data/core-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON &&
   cmake --build /data/core-build --parallel 4 &&
   ctest --test-dir /data/core-build --output-on-failure'
```

`development` 是源码构建出的持久镜像，不依赖对运行中容器执行 `docker commit`。该镜像包含工具链，不等于已经验证新的模型、调度策略或 GPU 结果。需要转移到 Kairos 时，仍按项目规定在控制台上传已准备好的 `.tar` 并核验 manifest digest；不要在 Kairos 作业容器内构建或 commit。
