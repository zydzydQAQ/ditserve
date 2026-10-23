# 本机双 RTX 4090

本机环境使用 `configs/env/rtx4090-2x24g.json`。模型和运行数据放在数据盘：

```text
/mnt/ssd0/zyd/model/wan2.1-t2v-1.3b/
/mnt/ssd0/zyd/model/sana-sprint-0.6b/
/mnt/ssd0/zyd/ditserve-data/
/mnt/ssd0/zyd/images/
```

源项目为 `zyd@172.31.225.239:/home/zyd/code/dit-rewrite`，目标双 4090 主机为 `172.31.225.185`。两个模型共 36,632,428,614 字节，已复制为本地实际文件；项目清单中的 33 个必需文件均通过 SHA256 校验。镜像归档 `/mnt/ssd0/zyd/images/ditserve-rewrite-runtime.tar.gz` 共 3,757,147,317 字节，SHA256 为 `e98bb0e65f9d32923e457faee05c79e05a3ba72b851548cd081b0bae7580ff2a`，已校验。

归档中的标签为 `ditserve:rewrite-runtime`，镜像 ID 为 `sha256:ecb1055e440e4cc32c2cca5733852a6c83544c8032fec2a899089dcf6b08d7bc`，revision `bc8029da61e8b34a24ec5ee166b4d3ffbe5c02a5` 与两端项目一致。无需为切换 4090 环境重建镜像。来源和校验记录保存在 `/mnt/ssd0/zyd/ditserve-data/checks/migration-receipt.json`。

传输使用已获源端授权的专用身份文件 `~/.ssh/ditserve_transfer_ed25519`，可用下面的方式连接：

```bash
ssh -i ~/.ssh/ditserve_transfer_ed25519 -o IdentitiesOnly=yes zyd@172.31.225.239
```

`docker/run-4090.sh` 默认连接专用 socket `unix:///run/ditserve-docker/docker.sock`；可通过 `DITSERVE_DOCKER_HOST` 或 `DOCKER_HOST` 覆盖，前者优先。管理员安装步骤见 [专用 Docker 服务说明](../docker/4090-daemon/README.md)，其 Docker 和 containerd 持久数据均放在 `/mnt/ssd2/zyd/docker-data`。模型、镜像归档和用户缓存仍在 SSD0。

脚本将模型只读挂到 `/models`，结果、账本、缓存和临时文件挂到 `/data`。它使用当前用户的 UID/GID、宿主 network/PID/IPC 命名空间和无限制 memlock，并显式传入 `NCCL_P2P_DISABLE=1`，直接运行 `torchrun` 时也会生效。仓库 `configs/` 可写挂到镜像的配置目录，便于保存本机成本画像。

从仓库根目录运行；不带参数时执行 `ditserve doctor`，不初始化 CUDA。其他参数作为完整容器命令原样传递：

```bash
./docker/run-4090.sh
./docker/run-4090.sh ditserve doctor --allow-gpu
./docker/run-4090.sh ditserve run \
  --config /opt/ditserve/share/ditserve/configs/studies/acceptance.json \
  --allow-gpu --run-id run01
```

默认镜像为 `ditserve:rewrite-runtime`。可用以下变量覆盖镜像或宿主目录：

```bash
DITSERVE_IMAGE=ditserve:rewrite-runtime \
DITSERVE_MODELS_DIR=/mnt/ssd0/zyd/model \
DITSERVE_DATA_DIR=/mnt/ssd0/zyd/ditserve-data \
  ./docker/run-4090.sh ditserve doctor
```

系统 `/usr/bin/nvidia-smi` 实际是静态输出脚本，无法提供项目需要的真实 CSV 和 GPU 进程信息。真实工具已从 NVIDIA 官方、匹配 `550.163.01` 版本的软件包提取并校验 SHA256，放在 `/mnt/ssd0/zyd/ditserve-data/tools/nvidia-smi`，来源记录在旁边的 `nvidia-smi.source.json`。启动脚本在该文件存在且可执行时将其只读挂到容器 `/usr/local/bin/nvidia-smi`，并让 `/usr/local/bin` 优先于 `/usr/bin`。这样使用镜像中已有、普通用户可访问的目录，同时保留 NVIDIA 容器运行时的 `/usr/bin` 挂载。可用 `DITSERVE_NVIDIA_SMI=/其他路径/nvidia-smi` 覆盖；显式指定的文件不可执行时，脚本会报错退出。

截至 2026-10-19，管理员已安装并启动专用 Docker/containerd，镜像已导入，镜像 ID 和 revision 与迁移归档一致。实际 DockerRootDir 为 `/mnt/ssd2/zyd/docker-data`，存储驱动为 `overlay2`。服务未设置开机自启；重新登录后 `zyd` 获得专用组权限。

容器 `doctor --allow-gpu` 已确认两张 RTX 4090、锁定依赖、模型目录、宿主资源、CUDA 初始化、GPU 架构和 NCCL；唯一失败项是现有 `MIN_DRIVER=560` 检查，本机实际驱动为 `550.163.01`。该要求仍保留，且仅用于 `doctor` 诊断；正式 `run` 另行执行 GPU 独占准入。

实际小张量检查已全部通过：两张卡各自的 BF16 矩阵乘法、Flash SDPA、两次 Inductor/Triton 执行，以及 `NCCL_P2P_DISABLE=1` 下的双 rank all-reduce 与 all-to-all。版本为 torch `2.10.0+cu126`、Triton `3.6.0`、NCCL `2.27.5`。结果在 `checks/doctor-container.json`、`checks/cuda-kernels.json`、`checks/nccl.json`。

550 驱动满足 NVIDIA 的 CUDA 12.x minor compatibility 最低版本 `525.60.13`，但较新驱动 API 和 PTX JIT 等路径仍有限制；上述通过结果只覆盖本次已测路径，不能替代完整模型验收或据此全局放宽驱动要求。[CUDA 12.6 版本表](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)、[minor compatibility 限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html#application-considerations-for-minor-version-compatibility)

`docker/check-4090-runtime.py` 提供不加载模型的小张量诊断：逐卡检查 BF16 矩阵乘法、强制 Flash SDPA 和两次 Inductor/Triton 执行；分布式模式检查两卡 NCCL all-reduce 与 all-to-all。它输出 JSON 并在失败时返回非零，不能替代模型验收或改变项目的驱动要求。镜像导入后可执行：

```bash
install -m 0644 docker/check-4090-runtime.py /mnt/ssd0/zyd/ditserve-data/tools/
./docker/run-4090.sh timeout 300 python /data/tools/check-4090-runtime.py
./docker/run-4090.sh timeout 120 torchrun --standalone --nproc_per_node=2 \
  /data/tools/check-4090-runtime.py --distributed
```

分布式进程组超时为 30 秒，`timeout` 在容器内限制诊断命令。脚本已通过语法、`--help` 和本机实际 GPU 验证。

已在数据盘完成 C++ 构建，`ctest` 的 5 项测试全部通过。宿主 Python 未安装模型依赖，实际模型运行使用已迁移的镜像。`acceptance`、`cost_profile`、`block_exact`、`sp_equivalence` 的配置解析均通过；接入本机画像后，`serving` 也已通过解析。检查输出保存在 `/mnt/ssd0/zyd/ditserve-data/checks/`。启动脚本已通过 `bash -n`、假 Docker 参数测试和本机实际运行。

镜像内 CPU 契约测试共 81 项，结果为 `OK (skipped=4)`。通过启动脚本运行这些测试时，需要移除默认 GPU 环境变量，否则 CPU 测试夹具会意外绑定双卡配置：

```bash
./docker/run-4090.sh env -u DITSERVE_ENV -u DITSERVE_MODELS_ROOT \
  CUDA_VISIBLE_DEVICES= timeout 180 python -m unittest discover \
  -s /opt/ditserve/share/ditserve/tests/python
```

GPU 1 的单卡冒烟及完整 `acceptance` 已通过。前者检查 Wan 官方参考循环的输出/latent 逐位一致，以及两次 Sana 1024×1024 输出的有限值和形状；后者三阶段 `numeric`、`block`、`mixed` 均通过配置的数值门槛，混合负载完成两个视频和两张图像，无失败。记录分别在 `results/rtx4090-gpu1-smoke-rtx4090-model-smoke/migration01/` 和 `results/rtx4090-gpu1-smoke-acceptance/migration01/`，路径相对于数据目录。

释放桌面占用后，默认 `rtx4090-2x24g` 环境的完整双卡 `acceptance` 也已通过：`numeric` 和 `block` 在两张卡上的数值门槛全部通过，`mixed` 完成两个视频和两张图像，无失败，资源清理已验证。结果和校验记录为：

```text
/mnt/ssd0/zyd/ditserve-data/results/rtx4090-2x24g-acceptance/migration-dual01/REPORT.md
/mnt/ssd0/zyd/ditserve-data/results/rtx4090-2x24g-acceptance/migration-dual01/results.xlsx
/mnt/ssd0/zyd/ditserve-data/checks/acceptance-dual.verify.json
```

双卡验收期间，经用户授权暂停了宿主 `gdm.service`，以及 `aiseon` 用户的 `gnome-remote-desktop.service`、`snap.snapd-desktop-integration.snapd-desktop-integration.service`，服务未被禁用或卸载。验收完成后，已按用户最新要求恢复这三个服务，均为 `active/running`，RDP 已监听 3389 端口。后续恢复桌面可使用以下管理员命令：

```bash
sudo systemctl start gdm.service
sudo -u aiseon env XDG_RUNTIME_DIR=/run/user/1000 \
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus systemctl --user start \
  gnome-remote-desktop.service snap.snapd-desktop-integration.snapd-desktop-integration.service
```

本机已实测当前服务配置中的 17/33 帧 480p Wan 和 Sana 1024，画像为 `configs/profiles/rtx4090-sp1-local.json`，默认双卡环境已引用它。SP1 模型成本来自物理 GPU 1；双卡独占验收结束后，另行测量两卡 all-to-all 链路，画像 `provenance` 为 `measured`。来源包含运行记录 SHA256、模型测量卡 UUID、驱动、镜像 ID 和链路记录 SHA256。81 帧视频未纳入当前服务类别及本次成本测量。

Wan 17/33 帧的实测单步中位耗时约 584/1344 ms，Sana 热运行单次约 240 ms。Wan 最大 reserved 峰值为 12.443 GiB，Sana 为 8.547 GiB，合计约 20.99 GiB；保留 Wan `memory_fraction=0.54`、Sana `0.40` 和 expandable segments。该成本文件用于本机调度估算，长时间负载/SLO 和实验性 SP2 尚未验收。

测量用配置及摘要在数据盘 `checks/cost-profile-4090-serving.study.json`、`checks/cost-profile-gpu1.summary.json`，结果在 `results/rtx4090-gpu1-smoke-cost-profile-4090-serving/migration01/`。链路原始记录为 `/mnt/ssd0/zyd/ditserve-data/collectives-rtx4090-2x24g.json`。后续重测应选用新的输出文件名：

```bash
./docker/run-4090.sh timeout 120 torchrun --standalone --nproc_per_node 2 \
  -m ditserve.evaluation.collectives \
  --output /data/collectives-rtx4090-2x24g-recheck.json
```

真实 serving 当前每卡独立运行 SP1；SP2 数值验证仍是单独的实验。两个模型的固定版本和 SHA256 校验方式见 [迁移手册](../MIGRATION.md#2-模型)。
