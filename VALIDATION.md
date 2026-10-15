# 迁移交付验证记录

## GitHub 源码发布版（2026-10-16）

发布目录以原 A100 迁移交付为基线，补入当前工作区的 `followup_resume.py`、`followup_combo_retry.py`、`followup_combo_retry_report.py` 及其 3 个测试文件，并更新发布用 `.gitignore`、GitHub 克隆步骤和可选历史包说明。原机工作目录、活动队列和 `.migration/a100` 的已验证旧交付保持不变。

`SOURCE_SNAPSHOT.json` 保留 2026-10-15 初始隔离时的历史来源清单，不代表当前发布树；当前发布内容由重新生成的 `DELIVERY_MANIFEST.json` 和 Git commit 共同标识。最新续跑工具仍依赖旧工作状态、冻结源码和账本，不意味着 Git 克隆即可续跑旧实验。

| 本次执行项目 | 结果 |
| --- | --- |
| 最新 3 个测试模块 | 56/56 通过；未加载 torch/diffusers/transformers |
| 完整发布源码 CPU 回归 | 1584 项：1582 通过、2 跳过、0 失败、0 错误；293.633 秒 |
| C++ Release 构建与 CTest | 通过；core-contracts 1/1 |
| MIGRATION / README 的 Bash 语法 | 18 / 2 个代码块通过 `bash -n` |
| 发布源码审查 | 无模型、虚拟环境、原始实验数据或压缩包；凭据模式扫描未发现实际凭据 |

完整回归使用已锁定的 `ditserve:a100-runtime` 镜像（`e9e0af86828b5`），以非 root UID 1002、断网、无 GPU、只读根文件系统和只读发布源码挂载运行。它验证本次发布源码在该依赖环境中的行为；镜像本体仍是此前交付，需在目标机按 Dockerfile 构建当前 commit。此次未运行新的 GPU 实验。

两项跳过：旧实验固定负载未随源码交付；decoder 的真实 CPU 编译切换测试需要显式开启且不干扰正在计量 CPU 编码的原机实验。跳过不计为通过。单卡模型、CUDA 编译、A100 会话/LongLive 有限执行和八卡 NCCL 仍需目标机验收。源机结果后来经过清理和续跑，下文历史快照统计只描述旧交付时刻。

首次将既有源码纳入独立 Git 仓库时，`git diff --cached --check` 提示 5 处原有空白格式问题：`configs/synthetic.json`、`workloads/preemption.json`、`optimizations/cache_executor.py`、`optimizations/wan.py` 的文件末尾空行，以及 `sessions/reporting.py` 的行尾空格。发布保持这些已验证源文件的原始字节；该格式检查未计为通过，不影响上述运行测试结果。

## 2026-10-15 原交付的历史验证基线

本次在 `.migration/a100` 隔离副本实施。原项目工作目录、已有 Python 环境、运行队列、模型和原始结果未修改；未运行 GPU 模型实验。源快照身份见 `SOURCE_SNAPSHOT.json`，最终交付文件见 `DELIVERY_MANIFEST.json`。

交付前重新对照原目录：快照记录的 **427 个源码/配置文件全部保持原始大小和 SHA256**。

## 已执行的源码验证

| 项目 | 实际结果 |
| --- | --- |
| C++ Release 构建与 CTest | 通过；`core-contracts` 1/1，使用随包 JSON 头文件 |
| 保留主线全套 CPU 回归 | 1523 项，1522 通过、1 跳过，184.341 秒；隐藏 CUDA，Gloo 使用本地回环网络 |
| 最终迁移 CLI / 模型 / 监督专项检查 | 52/52 通过，4.915 秒；包含 LongLive 环境证据门槛与额外模型加载文件拒绝 |
| 真实 CPU Inductor | Sana 与 Wan 两项编译/复用检查通过；未用 mock 替代编译 |
| 缺失、损坏、错误模型与分片索引 | 原 16 项包含于源码回归；新增替代权重/tokenizer 覆盖保护后，20/20 模型迁移测试通过 |
| 真实当前主机八卡门槛 | 按预期拒绝：`expected 8 GPUs, found 2`；未启动 GPU 进程 |
| 文档命令语法 | MIGRATION 的 17 个、README 的 2 个 Bash 块通过 `bash -n` |
| CLI 注册表 | 7 个注册命令的 `--help` 全部成功 |

跳过项为 `test_optimizations_stages.DecoderTests.test_real_cpu_inductor_cache_transition_and_backend_switch`。该测试要求显式设置 `DITSERVE_STAGE_COMPILER_CPU_SMOKE=1`，且已有 GPU 实验不在计量 CPU 编码耗时。原机有活动研究队列，未开启此项；新机空闲时按 MIGRATION 的全套测试命令补验。跳过不计为通过。

原 `.venv-previs` 上的完整迁移 CPU 门槛按预期拒绝 bootstrap 差异：pip 为 23.0.1 而新锁为 25.3，setuptools 为 79.0.1 而新锁为 80.9.0，未安装锁定的 wheel 0.45.1。主研究依赖版本匹配。未为通过检查修改活动环境；新镜像必须独立通过此门槛。

## 模型和历史交付

固定模型清单的 56 个文件已从原机完整读取并核对 SHA256，总计 **71,730,027,294 bytes**；未复制或修改模型。默认 Wan/Sana/LPIPS 为 **36,876,831,232 bytes**；Mistral 增加 **14,485,820,536 bytes**；LongLive 增加 **20,367,375,526 bytes**。此验证证明文件身份，不代表新镜像或 A100 推理已通过。

历史导出回读验证通过，复制 20,320 个文件，逻辑大小 601,179,938 bytes；仅在交付目录内硬链接去重，实际文件数据 254,065,764 bytes，另有总工作簿/报告。`ditserve-a100-history.tar.gz` 为 **71,006,203 bytes**，包含 17,149 个内部硬链接，未发现越界归档路径。对压缩包进行了独立回读，全部 20,320 个文件与工作簿中的大小/SHA256 相符。未转移的 11,081 个文件、9,406,516,766 bytes 及受保护的活动组逐项记录在历史根工作簿。原始数据不在此次清理范围。

130 项早期入口、独占测试和文档保存在 `legacy/`，逐项大小/哈希见其清单；默认镜像排除此目录。历史资格、预算和授权未迁移为新运行许可。

## 镜像与目标机验收

实际镜像 ID、字节数及各层大小由 `scripts/inspect_images.py` 从 Docker 读取，随交付保存在 [docker/build-manifest.json](docker/build-manifest.json)。

| 镜像 | 实测大小 | 最终 ID 前缀 |
| --- | ---: | --- |
| `ditserve:a100-runtime` | 7,179,097,889 bytes（6.69 GiB） | `e9e0af86828b5` |
| `ditserve:a100-longlive` | 14,699,706,801 bytes（13.69 GiB） | `f36134cc78a99` |

镜像运行层保留编译器及 Python 头文件，模型、历史、旧环境、缓存和完整 CUDA SDK 不进入默认镜像。默认镜像的主虚拟环境层为 6,792,223,305 bytes，占绝大部分；该版本保留锁定研究栈与真实编译能力。旧共享研究镜像包含其他项目资产，不用于声称同功能镜像的精确压缩比例。

最终默认镜像在非 root UID、断网、无 GPU 的条件下通过迁移 CPU 门槛；原始轻量证据见 [verification/runtime-cpu/REPORT.md](verification/runtime-cpu/REPORT.md) 和其工作簿。构建、全套容器回归及可选环境检查明细见 [docker/validation.json](docker/validation.json)；文件或通过记录缺失时，对应项目保持未验证。

最终镜像按文档使用非 root `1002:1002` 执行全套主线回归，**1,528 项：1,526 通过、2 跳过、0 失败、0 错误**，耗时 287.782 秒。除上述需显式开启的 decoder 编译测试外，另一跳过项为 `FrozenProtocolTests.test_real_readonly_quick_traces_have_one_seed_eight_replays_192_requests`：默认镜像不包含该历史实验的原始负载。两个跳过项均不计为通过。LongLive 在断网、非 root 条件下通过两套环境全部 76/63 个依赖版本、独立 `sys.path`、扩展导入、固定源码 commit 和干净 checkout 检查；未初始化 CUDA。

最终镜像同步了迁移文档中的变量初始化、清单外模型加载文件处理说明，并将 `/opt/ditserve` 的源码/配置权限规范为所有用户可读、目录可遍历。部分历史配置原为 `0600`，需要此修复才能按文档使用任意宿主 UID。运行代码、测试和依赖内容未改变；各次测试所用镜像 ID 与最终交付镜像 ID 分别记录。

基础镜像固定 digest，Python 安装包固定全部版本与 SHA256。Debian apt 使用发行版软件源，不是固定时间快照，因此不承诺每次构建的镜像 ID 逐字节相同；实际系统包版本与构建来源由验证记录保留。

以下项目必须在目标机完成，当前不能标为通过：单卡 Wan/Sana 与 LPIPS、CUDA Inductor、Mistral KV 保存恢复、LongLive 有限执行、八张同规格 A100 的逐卡 FP16/BF16/Flash 以及八进程 NCCL。完整复现与失败处理命令见 [MIGRATION.md](MIGRATION.md)。环境验收不构成新的研究质量资格、SP8 推理或八卡任务调度实现。
