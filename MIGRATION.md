# DitServe Kairos Cluster Migration & Rebuttal Restoration Guide

本手册用于在清空远端 `jjjkairos` 共享存储后（或在新机器/空白工作区上），**仅通过拉取 GitHub `ditserve` 仓库**，快速将代码构建、模型权重、提示词数据集、校准包、作业脚手架与运行镜像环境完整恢复到 `jjjkairos` 集群的 `DITSERVE` 项目对应位置，以支持论文 Rebuttal 核查与补充实验。

> **离线 Rebuttal 免复原提示**：若仅需核查已有论文数据、重算任意分位数、抽样分析或绘制微观调度甘特图，**无需恢复集群模型**——截至 `2026-11-17 08:46` 的全部章节汇总与 167 份逐请求压缩时间线（`.ledger.json.gz`）已完整归档在仓库内的 [`results/`](./results/README.md) 与 [`results/_archive/`](./results/_archive/README.md) 中。

---

## 0. 目录映射与硬约束

### 0.1 路径映射总表

| 资产类别 | GitHub 仓库路径 | 远端 SSH 写入路径 (`jjjkairos`) | 作业容器内挂载路径 |
| :--- | :--- | :--- | :--- |
| 代码构建 | `deploy/kairos/tools/deploy_build.sh` | `/data/share/jjjkairos/code/DITSERVE/builds/<commit>/` | `/shared/code/DITSERVE/builds/<commit>/` |
| 作业脚本 | `deploy/kairos/scripts/` | `/data/share/jjjkairos/data/DITSERVE/scripts/` | `/shared/data/DITSERVE/scripts/` |
| 提示词数据集 | `deploy/kairos/datasets/` | `/data/share/jjjkairos/data/DITSERVE/datasets/` | `/shared/data/DITSERVE/datasets/` |
| 模型清单 | `deploy/kairos/manifests/formal-manifests/` | `/data/share/jjjkairos/data/DITSERVE/models/formal-manifests/` | `/shared/data/DITSERVE/models/formal-manifests/` |
| 模型权重 | 按清单从 HF / ModelScope 下载 | `/data/share/jjjkairos/data/DITSERVE/models/<model_dir>/` | `/shared/data/DITSERVE/models/<model_dir>/` |
| 校准包 | `results/_archive/1_local_formal_v2/calibration-bundles/` | `/data/share/jjjkairos/output/DITSERVE/acceptance/<bundle_id>/` | `/shared/output/DITSERVE/acceptance/<bundle_id>/` |
| 实验输出 | —（新实验生成） | `/data/share/jjjkairos/output/DITSERVE/` | `/shared/output/DITSERVE/` |

### 0.2 必须遵守的安全与平台硬约束

1. **禁止本机 WSL 或 `jjjkairos` 集群构建镜像**：严禁在本地 WSL / Docker Desktop 或集群节点执行 `docker build` / `buildx` / `podman` / `kaniko` 等任何容器构建操作。
2. **远端 `/data` 共享盘停止线**：每次写入、传输或提交作业前必须运行 `ssh jjjkairos 'df -h /data'`；若已用空间 **> 810G** 或使用率 **≥ 82%**，必须立即停止操作。
3. **大文件（>5G）本地暂存目录**：下载模型或镜像 `.tar` 时，只能存放在项目根目录或 `/mnt/d/Transfer/DITSERVE/`（先用 `df -h` 确认空间充足），严禁写入 WSL 原生盘（`~`、`/tmp`、`/var`）。远端逐文件核验 SHA256 通过后再删除本地暂存副本。
4. **镜像上传唯一通道**：仅通过 Kairos Web 控制台（`http://127.0.0.1:30800`）上传已有 `.tar` 文件；禁止 `docker push` 或 `rsync/scp` 镜像。
5. **凭据与 SSH 前置**：
   - SSH 别名 `jjjkairos`（密钥认证，已含 `30800/30300` 端口转发）；
   - Kairos 控制台登录凭据保存在本地 `~/key/kairos_login.json`（仅在内存读取，严禁打印或提交入库）；
   - 保持控制台隧道命令（使用 `127.0.0.1:30800`，不要用 `localhost`）：
     ```bash
     ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes jjjkairos 'sleep infinity'
     ```
6. **`DITSERVE_BUILD` 强制要求**：所有作业必须显式设置环境变量 `DITSERVE_BUILD=<7位commit>`，并通过日志开头 `build=... core=...` 及 `BUILD_INFO.json` 核对版本。

---

## 1. Tier 1：最小可跑 Rebuttal 复原流程（远端约占 `~39 GB`）

Tier 1 恢复后即可直接复现主实验全部 8×A100 / 16×A100 评测（`SD3.5-Medium` + `Wan2.2-TI2V-5B` 混合服务、消融阶梯、Soak、121f 弹性、多随机种子等）。

### Step 1. 拉取 GitHub 仓库并核验远端空间

```bash
git clone git@github.com:zydzydQAQ/ditserve.git
cd ditserve
git checkout main   # 或 8a100（与 origin/main 同步）

# 确认远端共享存储未触及 810G / 82% 停止线（Tier 1 需预留至少 ~45G）
ssh jjjkairos 'df -h /data'
```

### Step 2. 核验 Kairos 内网运行镜像（优先直接复用）

当前唯一已验证运行镜像存放在 Kairos 内网 Registry（**不占 `/data` 共享盘空间**，清理 `/data` 时已保留）：

- **镜像地址**：`192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3`
- **Manifest Digest**：`sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`

启动 SSH 隧道后，用脱敏客户端核验镜像仍在库且 digest 一致：

```bash
python3 deploy/kairos/tools/kairos-api.py get /api/images
```

> 若 `/api/images` 中该 tag 与 digest 均存在，则**镜像环境已就绪，无需上传或重建**。若因平台重置导致镜像丢失，请跳转到 **§2.4 镜像兜底重建流程**。

### Step 3. 恢复远端目录骨架、作业脚本、数据集、清单与校准包（`~5 MB`）

```bash
# 1. 创建远端标准目录结构
ssh jjjkairos 'mkdir -p \
  /data/share/jjjkairos/code/DITSERVE/builds \
  /data/share/jjjkairos/data/DITSERVE/{scripts,datasets,models/formal-manifests} \
  /data/share/jjjkairos/output/DITSERVE/{results,acceptance}'

# 2. 同步作业脚本（-L 解析相对符号链接，使 benchmarks/ 脚本一并落盘）
rsync -avL deploy/kairos/scripts/ jjjkairos:/data/share/jjjkairos/data/DITSERVE/scripts/

# 3. 同步提示词数据集（diffusiondb.json, vbench.json, vidprom.json，共 3.5 MB）
rsync -avP deploy/kairos/datasets/ jjjkairos:/data/share/jjjkairos/data/DITSERVE/datasets/

# 4. 同步模型官方校验清单
rsync -avP deploy/kairos/manifests/formal-manifests/ \
  jjjkairos:/data/share/jjjkairos/data/DITSERVE/models/formal-manifests/
rsync -avP deploy/kairos/manifests/sd3.5-medium.manifest.json \
  jjjkairos:/data/share/jjjkairos/data/DITSERVE/models/sd3.5-medium.manifest.json

# 5. 同步主实验 8xA100 校准包（b1/b2 与 121f cal1）
rsync -avP results/_archive/1_local_formal_v2/calibration-bundles/ \
  jjjkairos:/data/share/jjjkairos/output/DITSERVE/acceptance/
rsync -avP results/_archive/3_remote_kairos_outputs/results/kairos-wan22-native-calibration/ \
  jjjkairos:/data/share/jjjkairos/output/DITSERVE/results/kairos-wan22-native-calibration/
```

### Step 4. 暂存并发布代码构建（`builds/<commit>/`）

`deploy_build.sh` 默认采用 `--mode cluster`：仅把干净源码与已校验的 `nlohmann/json.hpp`（内置在 `deploy/kairos/deps/`）暂存到集群 `builds/.src-<commit>/`，首次提交带 `DITSERVE_BUILD=<commit>` 的容器作业时会自动在容器内编译 C++17、跑 `ctest` + `doctor` 并原子发布 `builds/<commit>/`（不消耗本地 WSL 内存）。

```bash
# 按需发布论文基线 commit 或最新 HEAD（例如主矩阵 720da72、5 级消融阶梯 3878f01、或当前 Rebuttal 分支 HEAD）
bash deploy/kairos/tools/deploy_build.sh 720da72
bash deploy/kairos/tools/deploy_build.sh 3878f01
# 若 Rebuttal 有新提交：
# bash deploy/kairos/tools/deploy_build.sh HEAD
```

### Step 5. 下载、校验并上传 Tier 1 必选模型（共 `~38 GB`）

| 模型标识 | 远端目标目录 (`/data/share/jjjkairos/data/DITSERVE/models/...`) | 上游源与精确 Revision | 体积 | 校验清单（仓库内路径） |
| :--- | :--- | :--- | :---: | :--- |
| **`sd35`** | `sd3.5-medium` | HF: `stabilityai/stable-diffusion-3.5-medium` @ `b940f670f0eda2d07fbb75229e779da1ad11eb80`<br>ModelScope 镜像: `AI-ModelScope/stable-diffusion-3.5-medium` (`2a0e12bc`) | ~16 GB | `deploy/kairos/manifests/formal-manifests/20261103-inputs-v1/sd35.manifest.json` |
| **`wan22`** | `wan2.2-ti2v-5b-official-b8fff731` | HF: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff7315c768468a5333511427288870b2e9635` | ~22 GB | `deploy/kairos/manifests/formal-manifests/20261103-inputs-v1/wan22-official-b8fff731.manifest.json` |

> **下载与上传步骤（逐模型执行，每步核验空间并及时清理本地暂存）**：
>
> 1. **检查本地 D 盘与远端 `/data` 空间**：
>    ```bash
>    df -h /mnt/d
>    ssh jjjkairos 'df -h /data'
>    mkdir -p /mnt/d/Transfer/DITSERVE/models /mnt/d/Transfer/DITSERVE/hf-cache
>    ```
> 2. **下载 `sd3.5-medium` 到 `/mnt/d/Transfer/DITSERVE/models/sd3.5-medium`**（优先 HuggingFace；国内网络可用 `HF_ENDPOINT=https://hf-mirror.com` 或 `modelscope download`）：
>    ```bash
>    HF_HOME=/mnt/d/Transfer/DITSERVE/hf-cache \
>    huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
>      --revision b940f670f0eda2d07fbb75229e779da1ad11eb80 \
>      --local-dir /mnt/d/Transfer/DITSERVE/models/sd3.5-medium
>    # 校验本地逐文件大小与 SHA256
>    python3 deploy/kairos/tools/verify_manifest.py \
>      --manifest deploy/kairos/manifests/formal-manifests/20261103-inputs-v1/sd35.manifest.json \
>      --root /mnt/d/Transfer/DITSERVE/models/sd3.5-medium
>    ```
> 3. **上传 `sd3.5-medium` 到 `jjjkairos` 并在远端复核 SHA256，通过后清理本地暂存**：
>    ```bash
>    ssh jjjkairos 'df -h /data'
>    rsync -avP /mnt/d/Transfer/DITSERVE/models/sd3.5-medium/ \
>      jjjkairos:/data/share/jjjkairos/data/DITSERVE/models/sd3.5-medium/
>    # 远端逐文件 SHA256 复核（把 verify_manifest.py 传 stdin 执行）
>    ssh jjjkairos 'python3 - --manifest /data/share/jjjkairos/data/DITSERVE/models/formal-manifests/20261103-inputs-v1/sd35.manifest.json --root /data/share/jjjkairos/data/DITSERVE/models/sd3.5-medium' < deploy/kairos/tools/verify_manifest.py
>    # 核验 errors=0 后再删除本地副本释放 D 盘空间：
>    rm -rf /mnt/d/Transfer/DITSERVE/models/sd3.5-medium /mnt/d/Transfer/DITSERVE/hf-cache
>    ```
> 4. **按相同流程下载、校验、上传 `wan2.2-ti2v-5b-official-b8fff731`**（仅下载清单中 21 个运行时必需文件即可，也可全量拉取后核验清单）：
>    ```bash
>    HF_HOME=/mnt/d/Transfer/DITSERVE/hf-cache \
>    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
>      --revision b8fff7315c768468a5333511427288870b2e9635 \
>      --local-dir /mnt/d/Transfer/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731
>    python3 deploy/kairos/tools/verify_manifest.py \
>      --manifest deploy/kairos/manifests/formal-manifests/20261103-inputs-v1/wan22-official-b8fff731.manifest.json \
>      --root /mnt/d/Transfer/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731
>
>    ssh jjjkairos 'df -h /data'
>    rsync -avP /mnt/d/Transfer/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731/ \
>      jjjkairos:/data/share/jjjkairos/data/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731/
>    ssh jjjkairos 'python3 - --manifest /data/share/jjjkairos/data/DITSERVE/models/formal-manifests/20261103-inputs-v1/wan22-official-b8fff731.manifest.json --root /data/share/jjjkairos/data/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731' < deploy/kairos/tools/verify_manifest.py
>    rm -rf /mnt/d/Transfer/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731 /mnt/d/Transfer/DITSERVE/hf-cache
>    ```
>    *(注：若旧脚本引用 `/shared/data/DITSERVE/models/wan2.2-ti2v-5b`，可在远端建立软链接 `ln -sfn wan2.2-ti2v-5b-official-b8fff731 /data/share/jjjkairos/data/DITSERVE/models/wan2.2-ti2v-5b`，零额外空间占用。)*

### Step 6. 提交 1×1 GPU 冒烟验收（触发首次容器编译并验证闭环）

先检查 `/api/jobs` 无冲突与 `/data` 空间正常，再提交一个 1×1 GPU 冒烟任务（以 `3878f01` 为例）：

```bash
ssh jjjkairos 'df -h /data'
python3 deploy/kairos/tools/kairos-api.py get /api/jobs

# 提交 1x1 冒烟（自动触发 builds/.src-3878f01 -> builds/3878f01 容器内编译、ctest 与 doctor）
python3 deploy/kairos/tools/kairos-api.py submit '{
  "name": "ditserve-rebuttal-smoke-1gpu-r1",
  "image": "192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3",
  "gpuCount": 1,
  "nodeCount": 1,
  "mountSharedStorage": true,
  "env": {"DITSERVE_BUILD": "3878f01"},
  "command": "bash -lc \"source /shared/data/DITSERVE/scripts/kairos_common.sh && ditserve_init_build && ditserve doctor --python /opt/venv/bin/python && ls -lh /shared/data/DITSERVE/models/sd3.5-medium/model_index.json /shared/data/DITSERVE/models/wan2.2-ti2v-5b-official-b8fff731/model_index.json\""
}'
```

通过 `python3 deploy/kairos/tools/kairos-api.py logs user-jjjkairos ditserve-rebuttal-smoke-1gpu-r1` 核对：
- 输出包含 `build=3878f01 core=<sha256>`；
- `ditserve doctor` 通过；
- `/shared/code/DITSERVE/builds/3878f01/BUILD_INFO.json` 已生成；
- 两个核心模型 `model_index.json` 均可读。

---

## 2. Tier 2：按需扩展组件（仅在 Rebuttal 需要对应实验时恢复）

### 2.1 `Wan2.1-T2V-1.3B`（Suite 2 / 对比实验，`~27 GB`）

- **上游源**：`Wan-AI/Wan2.1-T2V-1.3B-Diffusers` @ `0fad780a534b6463e45facd96134c9f345acfa5b`
- **远端目标**：`/data/share/jjjkairos/data/DITSERVE/models/wan`
- **校验清单**：`deploy/kairos/manifests/formal-manifests/20261103-inputs-v1/wan21.manifest.json`
- 按 Step 5 相同流程经 `/mnt/d/Transfer/DITSERVE/models/wan` 下载、校验、`rsync`、远端复核后删除本地副本。

### 2.2 LPIPS AlexNet 权重（可选感知质量评测，`234 MB`）

- **来源与 SHA256**：见 `deploy/kairos/manifests/torch-hub-alexnet.json`
  - URL：`https://download.pytorch.org/models/alexnet-owt-7be5be79.pth`
  - SHA256：`7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`
- **远端目标**：`/data/share/jjjkairos/data/DITSERVE/models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`

### 2.3 外部对比系统源码快照（`xDiT` / `TetriServe` / `GF-DiT` / `group-free-collectives`，`~43 MB`）

精确上游仓库与 commit 已保存在 `deploy/kairos/manifests/external-systems-20261103-source-v1/*.source.json`：

| 项目 | 上游仓库 | 锁定 Commit | 远端恢复路径 (`/data/share/jjjkairos/data/DITSERVE/external-systems/20261103-source-v1/...`) |
| :--- | :--- | :--- | :--- |
| `xDiT` | `https://github.com/xdit-project/xDiT` | `16c34e7f070de65d9d2c67a6d74e0007dd8b426a` | `xDiT/` |
| `TetriServe` | `https://github.com/DiT-Serving/TetriServe` | `414ec2956c09466ccd7842fda9851511462bd42a` | `TetriServe/` |
| `GF-DiT` | `https://github.com/SJTU-Liquid/GF-DiT` | `442531e499891d4ffa654e87c076f03326a37b98` | `GF-DiT/` |
| `group-free-collectives` | `https://github.com/SJTU-Liquid/group-free-collectives` | `9aadcdc6d4abc30af13feac908fbb4be98313398` | `group-free-collectives/` |

### 2.4 镜像兜底重建流程（仅当内网 Registry `r3` 镜像丢失时使用）

> **严禁在本机 WSL 或 `jjjkairos` 集群执行构建！** 必须先获得用户批准并在**独立的外部合规 Linux/GPU 构建机**上完成构建导出：

1. 在外部构建机克隆本仓库，依据 [`docker/Dockerfile`](./docker/Dockerfile)、[`docker/artifacts-runtime.json`](./docker/artifacts-runtime.json) 与 [`docker/install_dependencies.py`](./docker/install_dependencies.py) 构建 `linux/amd64` 镜像（CUDA 12.6 + SM80 / A100 + PyTorch + FlashAttention + vLLM + xDiT 运行时）：
   ```bash
   # 仅在外部合规构建机执行，切勿在本地 WSL 或 jjjkairos 运行：
   docker build --platform linux/amd64 \
     -f docker/Dockerfile \
     -t ditserve:external-serving-vllm-cu126-sm80-rebuttal .
   docker save -o ditserve-rebuttal.tar ditserve:external-serving-vllm-cu126-sm80-rebuttal
   sha256sum ditserve-rebuttal.tar
   ```
2. 将 `.tar` 归档传输到本地 `/mnt/d/Transfer/DITSERVE/images/`（先 `df -h /mnt/d` 核验空间充足）。
3. 经 SSH 隧道打开 Kairos Web 控制台（`http://127.0.0.1:30800`），按控制台“使用文档”直接上传 `.tar` 文件（严禁 `docker push` 或 `rsync`）。
4. 通过 `python3 deploy/kairos/tools/kairos-api.py get /api/images` 核对新入库镜像的 repository/tag 与 manifest digest；核验通过后再删除本地 `/mnt/d/Transfer/DITSERVE/images/*.tar`。
5. 若镜像 tag 发生变化，更新 `deploy/kairos/tools/deploy_build.sh` 中的 `RUNTIME=...` 并执行 Step 6 的 1×1 GPU 冒烟验证。

---

## 3. 复原验收核对清单

- [ ] `ssh jjjkairos 'df -h /data'` 确认低于 810G / 82% 停止线
- [ ] `/api/images` 确认 `external-serving-vllm-cu126-sm80-20261104-r3`（digest `3d8bcd2e...`）可用
- [ ] `sd3.5-medium` 与 `wan2.2-ti2v-5b-official-b8fff731` 在远端通过 `verify_manifest.py` 零错误 SHA256 核验
- [ ] `datasets/`、`scripts/`、`formal-manifests/`、`calibration-bundles/` 已落盘远端
- [ ] `deploy_build.sh` 已暂存目标 commit，且 1×1 GPU 冒烟作业输出正确 `build=<commit> core=<sha256>` 并通过 `doctor`
- [ ] `/mnt/d/Transfer/DITSERVE/` 下已核验上传的本地大文件暂存副本已清理干净
