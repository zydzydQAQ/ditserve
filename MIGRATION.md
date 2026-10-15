# 单机 8×A100 迁移操作手册

新环境 agent 按下面顺序执行。交付包含可构建源码、默认 `runtime` 和可选 `longlive` Docker 目标、固定依赖/模型清单及历史轻量快照。目标为 Linux x86_64、单机 8 张完整 A100 40GB 或 80GB；要求八卡采用相同型号、容量与规格。本文执行环境准备和有限验收，不自动启动研究队列。

迁移在隔离副本中完成，原实验目录和正在运行的队列保持原样。实际构建及测试进度以 [VALIDATION.md](VALIDATION.md) 为准；未记录通过的镜像、模型或目标机八卡检查均视为未验证。

## 1. 接收交付并校验源码

优先从 GitHub 获取源码。宿主需要 Git；仓库为私有时，先在新环境配置有读取权限的 GitHub 身份。下面变量沿用到后续步骤；可以提前指定新的绝对 `MIGRATION_ROOT` 和 `MIGRATION_VOLUME_ROOT`。Git 仓库不包含模型、历史结果或源机活动实验状态，这些都不阻止构建和环境验收。

```bash
set -euo pipefail
command -v git
export MIGRATION_ROOT="$(realpath -m "${MIGRATION_ROOT:-$HOME/code/ditserve-a100-migration}")"
mkdir -p "$MIGRATION_ROOT"
if [ ! -d "$MIGRATION_ROOT/a100/.git" ]; then
  test ! -e "$MIGRATION_ROOT/a100"
  git clone https://github.com/zydzydQAQ/ditserve.git "$MIGRATION_ROOT/a100"
fi
cd "$MIGRATION_ROOT/a100"
git rev-parse HEAD
python3 scripts/verify_delivery.py
```

记录实际 commit。已有目录时只做校验，不自动覆盖或更新；差异需要先核对本地修改或重新克隆到新目录。未安装 Git 时先通过宿主发行版包管理器安装 Git 和 CA 证书。

也可以使用单独收到的压缩交付包。此方式替代上面的克隆步骤；压缩包有自己的版本和清单，不保证与 GitHub 最新 commit 一致。将 `ditserve-a100-source.tar.gz` 和对应 `SHA256SUMS` 放在同一目录；若清单还列出历史包，则一并接收该历史包。源码包顶层为 `a100/`，可选历史包顶层为 `history/`。

```bash
set -euo pipefail
export DELIVERY_DIR="$(pwd -P)"
sha256sum -c SHA256SUMS
if [ -f "$DELIVERY_DIR/ditserve-a100-history.tar.gz" ]; then
  export HISTORY_ARCHIVE="$DELIVERY_DIR/ditserve-a100-history.tar.gz"
fi
export MIGRATION_ROOT="$(realpath -m "${MIGRATION_ROOT:-$HOME/code/ditserve-a100-migration}")"
mkdir -p "$MIGRATION_ROOT"
if [ ! -d "$MIGRATION_ROOT/a100" ]; then
  tar -xzf "$DELIVERY_DIR/ditserve-a100-source.tar.gz" -C "$MIGRATION_ROOT"
fi
cd "$MIGRATION_ROOT/a100"
python3 scripts/verify_delivery.py
```

预期源码显示 `Verified ... files (...)`，使用压缩包时校验还应显示 `OK`。后续命令沿用上述目录布局及变量；不要跳过初始化。接收端不要用 `--write` 重新生成清单绕过差异；损坏时重新获取对应版本。源码由 `DELIVERY_MANIFEST.json` 校验；可选历史文件的逐项 SHA256 在其历史工作簿中。

## 2. 检查宿主与容器运行条件

```bash
test "$(uname -m)" = x86_64
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,compute_cap --format=csv
nvidia-smi -L
free -h
df -h "$MIGRATION_ROOT"
docker version || true
docker buildx version || true
nvidia-ctk --version || true
```

应看到 8 个不同物理 UUID、A100/SM80，完整卡容量约 40GB 或 80GB。MIG 切片不属于此次验收。项目建议驱动采用原机使用的 `570.181` 或更新的受支持版本；这是项目迁移基线，不是 CUDA 官方最低驱动声明。驱动由宿主管理；容器使用 PyTorch wheel 自带的 CUDA 12.6 用户态库，不安装完整 CUDA SDK。

建议宿主内存至少 128 GiB，模型/构建/工作目录合计预留 200 GiB 可用磁盘，并单独核对 Docker data-root 的空间。模型精确字节数见第 4 节；镜像解压、构建层、下载临时文件及编译缓存还需要空间。

继续执行下面代码块；已安装的组件会自动跳过安装，末尾始终初始化后续使用的 `DOCKER` 命令数组。安装命令面向新的 Ubuntu 22.04/24.04 宿主；其他发行版按 [Docker 官方安装文档](https://docs.docker.com/engine/install/) 和 [NVIDIA Toolkit 安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 使用相应包管理器。已有 Docker 不在此流程中卸载、替换或清空。

```bash
if ! command -v docker >/dev/null; then
  test "$(. /etc/os-release; printf '%s' "$ID")" = ubuntu
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -d -m 0755 /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  DOCKER_CODENAME="$(. /etc/os-release; printf '%s' "${UBUNTU_CODENAME:-$VERSION_CODENAME}")"
  printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' "$DOCKER_CODENAME" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
  sudo systemctl enable --now docker
fi
if ! command -v nvidia-ctk >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y curl gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi
"${DOCKER[@]}" info --format '{{.DockerRootDir}}'
"${DOCKER[@]}" version
```

Toolkit 已安装但尚未配置 Docker 时，在新宿主运行 `sudo nvidia-ctk runtime configure --runtime=docker` 后重启 Docker。它会修改宿主 Docker 配置，已有容器任务时应安排维护窗口；配置作用见 [NVIDIA 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)。

## 3. 构建默认与可选镜像

```bash
cd "$MIGRATION_ROOT/a100"
export IMAGE=ditserve:a100-runtime
if "${DOCKER[@]}" buildx version >/dev/null 2>&1; then
  "${DOCKER[@]}" build --target runtime -t "$IMAGE" .
else
  DOCKER_BUILDKIT=0 "${DOCKER[@]}" build --target runtime-base -t "$IMAGE" .
fi
"${DOCKER[@]}" image inspect "$IMAGE" --format 'ID={{.Id}} bytes={{.Size}}'
"${DOCKER[@]}" history --no-trunc "$IMAGE"
"${DOCKER[@]}" run --rm --gpus all "$IMAGE" nvidia-smi -L
```

BuildKit 使用 `runtime`。原宿主旧 Docker 缺少 buildx；其兼容目标 `runtime-base` 与 `runtime` 内容相同，可避免旧构建器执行最终 `runtime` 之前的可选 LongLive 阶段。失败时不自动降级依赖版本。

基础镜像固定为带 digest 的 `python:3.10.20-slim-bookworm`。C++ 在构建阶段编译；运行层保留 `torch.compile` 所需 GCC/G++、Python 头文件和动态库。`configs/artifacts-runtime.json` 固定 76 个主环境安装包的 URL、版本、字节数及 SHA256，主解释器为 `/opt/venv/bin/python`。模型、历史结果、旧虚拟环境、Git 与缓存不进入镜像。首次构建需要访问镜像仓库、Debian 软件源及锁定 wheel 来源。

需要 LongLive 时再执行：

```bash
"${DOCKER[@]}" build --target longlive -t ditserve:a100-longlive .
"${DOCKER[@]}" image inspect ditserve:a100-longlive --format 'ID={{.Id}} bytes={{.Size}}'
"${DOCKER[@]}" run --rm ditserve:a100-longlive /opt/longlive-venv/bin/python -m pip check
```

该目标增加独立 `/opt/longlive-venv` 和 `/opt/longlive-source`，固定 torch `2.8.0+cu126`、diffusers `0.31.0`、flash-attn `2.8.3` 及 63 个锁定安装包。源码 commit 为 `c6a27e771e803b1c12ffa28b82ba69b42472020e`。主环境继续使用 `/opt/venv`，禁止混用两个环境的 `site-packages`。

运行 `image inspect` 和 `history` 后记录本机镜像与各层实测大小；不能用 wheel 下载量或旧虚拟环境目录大小代替最终镜像大小。

下载较慢时，可先用并发下载器准备精确 wheel 缓存，再经本机回环地址供构建读取。此缓存放在构建上下文之外；`--reuse-dir` 可重复指定已有 wheel 目录，只复用大小和 SHA256 匹配的文件。需要可选环境时，在清单参数后追加 `configs/artifacts-longlive.json`。下面子 shell 退出时仅停止自身 HTTP 服务；端口占用时选择另一个空闲端口。

```bash
export WHEELHOUSE_DIR="$MIGRATION_ROOT/wheelhouse"
python3 scripts/prepare_wheelhouse.py configs/artifacts-runtime.json \
  --output "$WHEELHOUSE_DIR" --workers 32
(
  python3 -m http.server 18765 --bind 127.0.0.1 --directory "$WHEELHOUSE_DIR" &
  WHEELHOUSE_SERVER_PID=$!
  trap 'kill "$WHEELHOUSE_SERVER_PID" 2>/dev/null || true; wait "$WHEELHOUSE_SERVER_PID" 2>/dev/null || true' EXIT
  "${DOCKER[@]}" build --network host --target runtime-base \
    --build-arg DITSERVE_WHEELHOUSE_URL=http://127.0.0.1:18765 -t "$IMAGE" .
)
```

安装仍强制官方 SHA256，缓存缺文件或字节不同会失败。`--direct` 仅在明确需要绕过失效代理时用于下载器；不改变依赖版本、来源或验证规则。

## 4. 挂载模型与数据卷，下载固定模型

```bash
export MIGRATION_VOLUME_ROOT="$(realpath -m "${MIGRATION_VOLUME_ROOT:-$MIGRATION_ROOT/volumes}")"
export MODEL_DIR="$MIGRATION_VOLUME_ROOT/models"
export DATA_DIR="$MIGRATION_VOLUME_ROOT/data"
mkdir -p "$MODEL_DIR" "$DATA_DIR/cache" "$DATA_DIR/results" "$DATA_DIR/configs"
CONTAINER=(--rm --user "$(id -u):$(id -g)" --shm-size=8g \
  --mount "type=bind,src=$DATA_DIR,dst=/data")
MODEL_MOUNTS=(--mount "type=bind,src=$MODEL_DIR,dst=/models")
TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then TOKEN_ARGS=(-e HF_TOKEN); fi
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" "${TOKEN_ARGS[@]}" \
  "$IMAGE" ditserve migrate models download --root /models --groups core
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none \
  "$IMAGE" ditserve migrate models verify --root /models --groups core
```

两条命令均应返回 `status: verified`。下载入口显式设置 `HF_HUB_OFFLINE=0` 和 `TRANSFORMERS_OFFLINE=0`；校验与模型运行保持离线。每次核对已有文件的大小、SHA256 与分片索引，合格文件复用，缺失或损坏文件恢复到锁定版本。运行 manifest 写入 `/models/.provisioning/`。若模型服务要求授权，从宿主环境提供 `HF_TOKEN`，不要写入 Dockerfile、build arg、配置或报告。

文件清单以 `configs/models.lock.json` 为准。下表不包含缓存、目录开销与临时空间：

| 组 / 本地目录 | 来源与固定 revision | 文件字节数 |
| --- | --- | ---: |
| core / `/models/wan` | Wan-AI/Wan2.1-T2V-1.3B-Diffusers，`0fad780a534b6463e45facd96134c9f345acfa5b` | 28,928,887,859 |
| core / `/models/sana` | Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers，`aa76e7f4f4928f378716b6716a2130fba3caf5b1` | 7,703,534,462 |
| core / `/models/torch/hub/checkpoints` | LPIPS 的 AlexNet `alexnet-owt-7be5be79.pth` | 244,408,911 |
| sessions / `/models/mistral` | mistralai/Mistral-7B-Instruct-v0.2，`63a8b081895390a26e140280378bc85ec8bce07a` | 14,485,820,536 |
| longlive / `/models/longlive/weights` | LongLive `cda9138d93872ef109d9a74c270184eb3a5acc6e` 与原生 Wan 组件 `37ec512624d61f7aa208f7ea8140a131f93afc9a` | 20,367,375,526 |

core 合计 **36,876,831,232 bytes**；全部组共 **71,730,027,294 bytes**。会话与 LongLive 均依赖 core。按需执行下面对应块，未选择的组不会影响默认验收：

```bash
# 可选：Mistral 会话模型。
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" "${TOKEN_ARGS[@]}" \
  "$IMAGE" ditserve migrate models download --root /models --groups core,sessions

# 可选：LongLive，使用已构建的独立环境镜像。
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" "${TOKEN_ARGS[@]}" \
  ditserve:a100-longlive ditserve migrate models download --root /models --groups core,longlive
```

LongLive provisioning 还用独立解释器检查依赖隔离、源码身份与原生加载结构；失败时不认可运行环境。该检查不执行 GPU 推理。

### 已有快照复用

目标机已有完整目录时，先将 `EXISTING_WAN`、`EXISTING_SANA`、`EXISTING_TORCH_CHECKPOINTS` 设为对应宿主绝对路径，再使用下面替代挂载。父卷可写以生成 `.provisioning`，模型子目录只读。直接绑定的目录须包含实际文件，不能让内部符号链接指向挂载外的 Hugging Face `blobs`；此类 cache 应按后面的离线打包方式解引用到新卷，原 cache 保持不变。

```bash
: "${EXISTING_WAN:?set the absolute Wan snapshot directory}"
: "${EXISTING_SANA:?set the absolute Sana snapshot directory}"
: "${EXISTING_TORCH_CHECKPOINTS:?set the absolute torch hub checkpoints directory}"
MODEL_MOUNTS=(--mount "type=bind,src=$MODEL_DIR,dst=/models" \
  --mount "type=bind,src=$(realpath -e "$EXISTING_WAN"),dst=/models/wan,readonly" \
  --mount "type=bind,src=$(realpath -e "$EXISTING_SANA"),dst=/models/sana,readonly" \
  --mount "type=bind,src=$(realpath -e "$EXISTING_TORCH_CHECKPOINTS"),dst=/models/torch/hub/checkpoints,readonly")
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none \
  "$IMAGE" ditserve migrate models download --root /models --groups core
```

应全部复用且 `downloaded_files: 0`。只读模型缺失或不匹配时失败，不修改原快照；改用新可写卷下载正确版本。

Wan/Sana/Mistral 的加载目录还会拒绝清单外的替代权重、分片索引及 tokenizer/config 覆盖文件，避免库优先加载未经校验的旧文件。出现 `unexpected_loader_file` 时使用新空卷按锁下载，或仅把清单列出的文件复制到新目录；不要直接删除共享快照内容。文档、下载缓存和验证元数据可以保留，LPIPS 的共享 checkpoint 目录允许其他模型文件。

### 模型离线转运

在已有合格模型的源机，用该机已定义的卷挂载导出：

```bash
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none "$IMAGE" \
  tar --dereference -czf /data/models-core.tar.gz -C /models wan sana torch
(cd "$DATA_DIR" && sha256sum models-core.tar.gz > models-core.tar.gz.sha256)
```

要包含选装组，在打包参数中追加实际已有的 `mistral`、`longlive` 目录，不复制 token 或旧 readiness 标记。若原 HF cache 有跨目录链接，可在宿主执行 `tar --dereference -czf wan-snapshot.tar.gz -C "$EXISTING_WAN" .`，在目标卷新建 `wan/` 再解包，Sana 同理；不能只复制断开的符号链接。

将模型包和校验文件送到目标机 `$DATA_DIR` 后，恢复默认父卷挂载并重建本机 manifest：

```bash
(cd "$DATA_DIR" && sha256sum -c models-core.tar.gz.sha256)
tar -xzf "$DATA_DIR/models-core.tar.gz" -C "$MODEL_DIR"
MODEL_MOUNTS=(--mount "type=bind,src=$MODEL_DIR,dst=/models")
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none \
  "$IMAGE" ditserve migrate models download --root /models --groups core
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none \
  "$IMAGE" ditserve migrate models verify --root /models --groups core
```

此处 `download` 虽将 HF 离线变量设为 0，但 Docker 禁止联网；完整文件只做本地重验，缺失则失败。选装组修改 `--groups` 并使用对应镜像。以上是模型离线转运；新镜像构建仍需第 3 节的软件源，完全隔离目标机可在联网同架构机器构建后另用 `docker save` / `docker load` 转运已验证镜像。

## 5. 绑定新环境与接收历史快照

选择第一张 A100 做单卡验收，八卡检查使用全部物理 UUID。`inspect` 只读盘点，不初始化 CUDA：

```bash
export ACCEPTANCE_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')"
export ALL_GPU_UUIDS="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --gpus all --pid=host --network none \
  "$IMAGE" ditserve migrate inspect
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none \
  "$IMAGE" ditserve migrate configure --models-root /models --data-root /data \
  --gpu-uuid "$ACCEPTANCE_GPU_UUID" --output /data/configs
if [ -n "${HISTORY_ARCHIVE:-}" ] && [ ! -d "$DATA_DIR/history" ]; then
  test -f "$HISTORY_ARCHIVE"
  tar -xzf "$HISTORY_ARCHIVE" -C "$DATA_DIR"
fi
```

`configure` 生成 `/data/configs/environment.json` 和 `scheduling-a100.json`。同内容可重复执行，已有绑定不同时要求新目录。它集中绑定模型、结果/临时卷、解释器和完整 UUID，不改写历史配置。环境配置声明 `environment_acceptance_only`、`research_qualification: not_inherited`；模型冒烟会在启动 GPU 前核对路径、revision 与解释器绑定。

容器按宿主 UID/GID 写卷。GPU 验收保留 `--pid=host`：`nvidia-smi` 返回宿主 PID，占用守卫据此核对进程组。`--gpus all` 保持八卡可见，模型子进程由配置 UUID 筛选；仅做单卡验收时，也可改成 `--gpus "device=$ACCEPTANCE_GPU_UUID"`。共享内存为 8 GiB，不需要 privileged、Docker socket 或全宿主目录挂载。

历史快照可选，Git 克隆后未设置 `HISTORY_ARCHIVE` 时跳过导入。另行接收历史包时，应先按其配套清单验证 SHA256，再设置为该包的绝对路径。2026-10-15 的旧快照展开约 262 MiB，布局为 `/data/history/results/...`，记录当时 20,320 项已转移文件和 11,081 项未转移内容。源机在该快照之后进行了结果精简和实验续跑，部分原始输入已删除，不能将旧快照描述当成当前源机状态；最新历史成果需另行接收其 Excel 与报告。完整旧 gzip、pruned 凭据和必要冻结源码作为兼容例外保留，重复文件在目标内硬链接去重。

历史快照用于查看证据，不能直接续跑或重新评测省略的原始输入。元数据里的旧绝对路径保持原样；预算、资格和授权不会因迁移被重置或变为新 A100 授权。导出没有删除或重写源结果。

`followup_resume`、`followup_combo_retry`、`followup_combo_retry_report` 的最新源码和测试已经纳入本仓库。这些是已有实验的继续/报告入口，依赖源机私有工作目录中的冻结源码、队列、父账本和绝对路径哈希。Git 克隆仅提供代码，不能自动恢复这些运行状态；新 A100 环境仍从独立配置、校准和 qualify 开始。

## 6. 顺序执行 CPU、八卡与模型验收

每次使用新输出目录。成功或失败都会保存 `results.xlsx` 和 `REPORT.md`，程序拒绝覆盖已有报告。检查 JSON `status`、退出码及工作簿里的失败/未验证项目，报告文件存在不代表验收成功。报告包含加载、编译、执行与清理的实际 GPU 秒数。

### CPU 与编译

```bash
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none \
  "$IMAGE" ditserve migrate check --suite cpu --timeout 600 \
  --output "/data/results/migration-cpu-$(date -u +%Y%m%dT%H%M%S%N)"
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --network none \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e DITSERVE_STAGE_COMPILER_CPU_SMOKE=1 \
  -e PYTHONPATH=/opt/ditserve/python:/opt/ditserve/tests/python \
  "$IMAGE" python -m unittest discover -s tests/python -v
```

第一条验证固定依赖、`pip check`、doctor、真实 CPU Inductor 和 C++ schema 协议，必须返回 `passed`；必需编译测试跳过会判失败。第二条运行保留主线的全套测试，skip 应逐项解释，不计作通过。Docker 构建阶段还需 CTest 通过。

第二条在尚未启动研究实验的新机上执行，显式开启额外的 decoder CPU 编译测试。已有实验正在计量 CPU 编码耗时时，不开启该测试；将其记为待验证，并在空闲窗口补测。

默认镜像不带旧在线实验的原始负载，因此历史只读负载测试可能报告 `historical fixed workload raw source unavailable`；这是明确未转移的历史输入检查，不能算作通过，也不需要为了消除 skip 把旧原始结果塞进镜像。新环境和新模型的必需检查仍应全部通过。

宿主独立 C++ 检查如下；vendored JSON 头文件支持断网构建，CTest 命令兼容 CMake 3.16：

```bash
sudo apt-get install -y cmake g++ libssl-dev
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build .cache/build --parallel 2
(cd .cache/build && ctest --output-on-failure)
```

### 八卡 CUDA 与 NCCL

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --gpus all --pid=host --network none \
  "$IMAGE" ditserve migrate check --suite gpu --devices "$ALL_GPU_UUIDS" \
  --expected-gpus 8 --timeout 600 \
  --output "/data/results/migration-eight-gpu-$(date -u +%Y%m%dT%H%M%S%N)"
```

要求 8 个不同 A100/SM80 UUID，逐卡执行 FP16、BF16、Flash SDPA 和 CUDA Inductor，再由内置同进程组监督器启动八个 NCCL rank，验证 all-reduce 与全部 rank 完成证据。超时只清理本次启动的进程组。GPU busy、PID 无法确认、卡数不足、重复 UUID 或异构设备均失败；等待占用方释放后重测，不结束其他作业。

### 单卡默认模型

```bash
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --gpus all --pid=host --network none \
  "$IMAGE" ditserve migrate smoke --config /data/configs/environment.json --groups core --timeout 900 \
  --output "/data/results/migration-models-$(date -u +%Y%m%dT%H%M%S%N)"
```

默认冒烟离线加载 Wan，生成 256×448、17 帧、4 步小样本并运行 LPIPS 自比较；Sana 使用原 BF16 的 1024px、两步配方。长期只保留工作簿和报告，不保存生成媒体或张量。LPIPS 自比较证明依赖与权重可用，不是生成质量资格。

### 按需：会话与 LongLive

```bash
# 已准备 core,sessions 后执行。
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --gpus all --pid=host --network none \
  "$IMAGE" ditserve migrate smoke --config /data/configs/environment.json --groups sessions --timeout 900 \
  --output "/data/results/migration-sessions-$(date -u +%Y%m%dT%H%M%S%N)"

# 已构建可选镜像并准备 core,longlive 后执行。
"${DOCKER[@]}" run "${CONTAINER[@]}" "${MODEL_MOUNTS[@]}" --gpus all --pid=host --network none \
  ditserve:a100-longlive ditserve migrate smoke --config /data/configs/environment.json --groups longlive --timeout 900 \
  --output "/data/results/migration-longlive-$(date -u +%Y%m%dT%H%M%S%N)"
```

会话显式采用 `a100_fp32`：Mistral FP16、Sana 全组件 FP32、MATH attention。检查有限 token、精确 KV CPU 保存恢复、恢复后 token 与原生参考一致，以及 Sana 暂停恢复与原生输出一致性。Mistral/Sana 分开加载以限制 40GB 卡的冒烟内存，未验证二者共驻研究容量。LongLive 使用独立解释器运行一个原生块，验证身份和有限输出。

通过状态只覆盖相应环境检查。八卡 NCCL 不是 SP8 推理；本交付不新增八卡任务调度器。单卡冒烟不能替代正式校准、质量评测或研究 qualify。

## 7. 新研究与历史协议边界

`scheduling-a100.json` 是新 A100 调度模板，后续新研究从新 `calibrate` 开始，使用新 run/ledger。模板保留协议已有的独立 12 GPU·小时预算定义，实际启动须按新任务明确预算和范围；环境配置阶段不执行 `prepare` 或 `run`。

`research-scheduling-next` 和 `research-sessions` 完整监督流程仍绑定历史 V100 UUID 与授权，不能仅替换 GPU 名字、路径或预算绕过门槛。A100 会话当前提供适配器验收；完整研究需要新协议、校准、qualify 和授权。优化及研究方向入口同样保留冻结配置、依赖版本、来源和预算守卫，应为新环境建立独立绑定，不能把旧结果当成已续跑。

旧 `configs/*.json` 为历史冻结引用保留，并非新机可直接执行的默认配置。开展研究前读对应 `docs/`；新结果按 Excel+报告收尾，盲评、续跑或调查输入在真实依赖结束前另行保留。

## 8. 故障处理与精确清理

| 现象 | 处理与复验 |
| --- | --- |
| wheel 哈希、下载或 apt 失败 | 核对联网、代理、空间及锁定 URL，保留失败记录，重跑相同构建；不取消哈希或自行升级依赖。旧 Docker 使用 `runtime-base`。 |
| 容器无 GPU / device driver 错误 | 宿主先确认 `nvidia-smi`，再检查 Toolkit 和 Docker runtime 配置；不在镜像内安装宿主驱动。 |
| 缺模型、大小/SHA256/shard 不符 | 在可写卷重跑相同组 `models download`，再离线 `verify`；只读复用源保持原样。401/403 时检查模型授权和宿主 `HF_TOKEN`。 |
| `unexpected_loader_file` | 存在清单外可改变加载结果的文件；改用新空模型卷或只复制锁定文件。下载器不会自动删除共享旧文件。 |
| Python/torch/扩展版本错误 | 检查镜像及 `/opt/venv`、`/opt/longlive-venv`，分别运行 `pip check`，去掉跨环境路径注入并重新构建锁定镜像。 |
| busy / ownership unknown | 核对 UUID、`--pid=host` 及宿主进程可见性，等待已有任务释放；不关闭守卫或结束其他任务。 |
| NCCL 超时或 rank 证据不全 | 查看工作簿，检查八卡可见性、`nvidia-smi topo -m`、共享内存与驱动，修复后用新目录重测；仍失败时八卡保持未验收。 |
| 配置旧路径或已有绑定不同 | 用 `configure` 写到新的 `/data/configs-*` 目录，并更新后续 `--config`；不修改历史快照通过来源校验。 |
| 报告目录已存在 | 使用新的 `--output`，保留已有通过/失败证据；未选装项目记录为未验证。 |
| 空间不足 | 分别核对 Docker data-root、模型卷、`/data/cache`、结果与源机 `.cache/results-work`，按具体归属与消费者处理；迁入 cache 不算释放空间。 |

迁移不自动删除原机实验、模型、旧镜像或缓存，也不运行全局 Docker prune。清理某个已封闭旧 run 前，核实账本/队列、实际评测/盲评/续跑依赖和来源引用，导出并回读 Excel 后用明确文件白名单或既有可靠入口清理。`--prune-raw` 只适用于逐项审核过的独立封闭 run，不能用于整个 `results`、模型目录或混合工作目录。旧 gzip/manifest/pruned/定位记录及仍被引用的冻结源码继续保留，直到迁移读者验证等价。

需要再次导出历史时使用 `migrate history-export`，明确提供源项目与新的独立目标目录。该入口只读检查 `results` 和 `.cache/results-work`，复制稳定轻量证据并列出全部未转移路径；它不删除源文件、不授予研究资格，也不把未审查的原始输入判为可删。
