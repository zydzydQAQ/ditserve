# 容器镜像与代码发布

jjjkairos 当前使用既有镜像 `192.168.1.114:30500/jjjkairos/ditserve:xdit-dev-016e1ca-slim`，manifest digest 为 `sha256:2009ea26f63bcf0360ce7b4d71ed508c91c0702c97dd97e372cecfb672a818a1`。镜像提供 torch 2.10.0+cu126、diffusers 0.36.0、transformers 4.57.6、cache-dit 1.5.0 和固定 xDiT/xFuser 运行环境；模型权重从共享数据目录读取，不装入镜像。

禁止在本机 WSL、Docker Desktop WSL 后端或任何 Kairos 节点构建镜像，也禁止通过 registry 推送、rsync/scp 或作业容器绕过。确需新依赖或新镜像时，应停止发布并由用户提供明确批准的新镜像来源；已有 `.tar` 只能按控制台“使用文档”上传并核验。

## 日常代码发布

日常修改不重建镜像。代码在本地 `8a100` 分支提交后，使用项目既有 `.local/ditserve-kairos-rewrite-runtime/tools/deploy_build.sh [COMMIT]`：本地编译并运行 CTest，再发布到 `/data/share/jjjkairos/code/DITSERVE/builds/<commit>/`。同一提交不覆盖，脚本不拉取或构建镜像。

每个作业必须显式设置 `DITSERVE_BUILD=<commit>`，挂载共享盘，并将产物写到 `/shared/output/DITSERVE/`。作业开头的 `build=... core=...` 和运行目录 `BUILD_INFO.json` 用来核对实际代码；未设置该变量会错误回退到镜像内置代码。

Kairos 登录节点只用于文件传输与平台访问，不直接运行 GPU 计算。首次验证使用 1×1 GPU；多机实验保留平台注入的 `$RANK`、`$MASTER_ADDR` 和 `$MASTER_PORT`。提交前检查重复作业、当前镜像和 `df -h /data`，遵守项目共享存储停止线。

## 运行边界

作业容器是临时的，只有 `/shared` 挂载内容保留。模型、数据、代码和输出分别使用项目约定的 DITSERVE 子目录，不触碰其他项目。worker 使用完整 GPU UUID 绑定设备，文本编码和模型加载保持离线。多卡 NCCL 需要镜像和作业提供的共享内存及 memlock 配置，实际拓扑以作业内 `nvidia-smi topo -m` 为准。

Kairos 作业不依赖 `--pid=host`；平台冒烟已验证 NVML 可见当前容器 worker PID。其他运行环境不能套用这一结论，需单独验证 GPU 隔离检查。
