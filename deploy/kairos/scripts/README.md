# DITSERVE Kairos 实验启动脚本与运维工具 (`scripts/` & `tools/`)

固定运行镜像：`192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3`（manifest digest：`sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`）。禁止在本机 WSL 或集群节点构建镜像。每个作业必须挂载共享盘，设置唯一 `DITSERVE_RUN_ID` 和已发布/暂存的 `DITSERVE_BUILD`，失败重试使用新 ID，不回退到镜像内置源码。

## 1. 单一事实源（符号链接收敛）

与 `ditserve/benchmarks/` 共享的基准入口统一以 `ditserve/benchmarks/` 为唯一事实源，本目录下保留相对符号链接以兼容既有脚本与单测路径：

- `scripts/kairos_formal_benchmark.sh` -> `../../../ditserve/benchmarks/kairos_formal_benchmark.sh`
- `scripts/cluster_config.py` -> `../../../ditserve/benchmarks/cluster_config.py`
- `scripts/cross_modal_memory_study.py` -> `../../../ditserve/benchmarks/cross_modal_memory_study.py`
- `tools/p1_isolated_job.py` -> `../../../ditserve/benchmarks/p1_isolated_job.py`

通过 `tools/prepare_formal_release.py` 打包不可变 operator 快照时，会自动解引用上述符号链接并将实体文件连同 SHA256 写入 `input-manifest.json`。

## 2. 集群作业启动脚本 (`scripts/`)

| 脚本 | 资源 | 用途 |
| --- | --- | --- |
| `kairos_common.sh` | 公共库 | 环境变量与路径初始化、共享盘惰性编译发布（`.src-<commit>` -> `builds/<commit>`）、日志重定向及同分配原生校准函数 `kairos_calibrate_native_profile` |
| `kairos_formal_benchmark.sh` | 1×2..8 | 正式协议入口（链接自 `ditserve/benchmarks/`）：支持 `calibration`、`p0`、`p1`、`p1_isolated`、`serve`（含 `DITSERVE_SERVE_BATCH` 批量子实验与同构拓扑画像复用） |
| `kairos_targeted_tests.sh` | 1×1 | 分配 GPU 运行时算术探针 + 固定环境 Python 单元与契约测试套件，完成后由 `kairos_contract_receipt.py` 签发 `CONTRACT` 阶段凭据 |
| `kairos_contract_receipt.py` | 无 GPU | 校验 `runtime-smoke.json`、`contract.log` 与 digest 锁定作业规格，生成不可覆盖的 `contract-receipt.json` |
| `kairos_tools_validation.sh` | 1×1 | 调用 `tools/kairos_tools_validation.py` 对冻结 core/worker 与 operator 快照执行工具链自检 |
| `kairos_runtime_audit.py` | 1×1..8 | 采集实际 GPU 拓扑、依赖版本与三模型配置，并提供有界显存/利用率采样（`--memory-summary`，仅存极值与均值，不落盘高频 CSV） |
| `kairos_native_serving.sh` | 1×8 | 从拟合画像与公开 prompt pool 生成并执行冻结 serving trace；支持 `DITSERVE_CALIBRATE=1` 同分配校准、`DITSERVE_SUITE`（`main`/`ablation`/`main+ablation`/`preempt`/`sweep`）与 `DITSERVE_QUALITY_ARM` |
| `kairos_native_approx.sh` | 1×8 | 同分配校准后在固定起始 `cfg2sp3` 下执行 Exact、`fixed-1..7` 与曲率折叠对照，计算分层配对质量指标（PSNR/SSIM/LPIPS）并运行混合负载对照 |
| `kairos_gfc_serving.sh` | 1×8 | 同一冻结 trace 顺序执行 `nccl` 与 `gfc` 集合通信后端，对 `static-cfg2sp3,edf-cfg2` 做输出 digest 一致性校验并保留各自性能指标 |
| `kairos_val_i6_shm.sh` / `run_val_i6_shm_pool.py` | 1×2..8 | Item 6 验证：真实 SD3.5 worker 跨进程 `/dev/shm` 编码器缓存池并发竞态与引用计数验证 |
| `kairos_val_i7_ulysses.sh` / `run_val_i7_interleave.py` | 1×2..8 | Item 7 验证：Wan2.1-T2V-1.3B SP2 交错 Ulysses `all-to-all` 路由（GFC vs NCCL）与位级精确性验证 |
| `cluster_config.py` / `cross_modal_memory_study.py` | 辅助 | 集群配置生成与跨模态显存分析入口（链接自 `ditserve/benchmarks/`） |
| `ditserve_output_retention.py` | 无 GPU | 远端 `/shared/output/DITSERVE` 的 `manifest`、`plan`、`apply`、`verify` 四步留存精简工具，保护未闭合依赖，禁止笼统 `rm -rf` |

## 3. 本地发布、规格生成与队列守卫工具 (`tools/`)

| 工具 | 用途 |
| --- | --- |
| `deploy_build.sh` | 将本地 `ditserve` 已提交版本打包上传至 `/data/share/jjjkairos/code/DITSERVE/builds/.src-<commit>`（默认 `cluster` 惰性编译模式）或本地编译发布，写一次永不覆盖 |
| `prepare_formal_release.py` | 从指定 git commit 导出 `tests`/`benchmarks`/`configs` 与当前 `scripts`/`tools`，生成带全文件 SHA256 的本地 `formal-releases/<short>/input-manifest.json` |
| `publish_formal_release.py` | 经共享存储门禁校验后，将本地准备好的 `formal-releases/<short>` 原子上传并远端复核 SHA256 至 `/data/share/jjjkairos/data/DITSERVE/{tests,scripts/releases}/<short>` |
| `generate_formal_batch.py` | 从已提交的 `configs/benchmarks/formal-mixed-v1.json` 生成 digest 锁定的不可变 `contract`、`p0`、`calibration` 作业规格 |
| `generate_p1_batch.py` | 基于 `formal-mixed-v1.json` 与 `p1-qualification-v1.json` 生成 P0 门控的 P1 pair/role 诊断规格 |
| `generate_batch_20261102.py` | 生成分阶段（`p0`、`main`、`quality`、`gfc`、`sweep`）验证作业规格批次 |
| `p1_isolated_job.py` | P1 同分配隔离校准与 ABBA 回放的 preflight/prepare/spec 辅助模块（链接自 `ditserve/benchmarks/p1_isolated_job.py`） |
| `kairos_queue.py` | 核心队列提交与准入控制：提供共享盘门禁（`storage_gate`）、实时节点与配额核算（`live_capacity` / `capacity_fits`）、阶段凭据门禁（`stage_gate`）及串行/回填提交循环 |
| `kairos_resources.py` | 复用 `kairos_queue` 的只读资源与共享盘观察器（支持 `--watch --interval 30`），不修改任何作业状态 |
| `kairos_storage_guard.py` | 复用 `kairos_queue` 的显式白名单作业存储守卫：当 `/data` 达到停止线时仅终止白名单内作业 |
| `submit_and_monitor_formal_v2.py` | 复用 `kairos_queue` 存储门禁与 API 客户端，提交 formal-v2 基准批次并监控至 `Running` 状态后抓取初始日志 |
| `kairos_tools_validation.py` | 容器内执行的工具链自检 runner：校验 `input-manifest.json`、构建产物哈希与 CUDA 探针，并在子进程运行隔离的 Python 契约测试 |

## 4. 存储门禁与输出留存约定

- **共享存储停止线**：所有上传、队列提交与容器内执行均先检查 `/data`（容器内 `/shared`）；工具层门禁在已用空间 `> 790G` 或使用率 `>= 80%` 时立即拒绝继续操作（严格早于项目硬上限 `810G / 82%`）。
- **输出留存**：正式输出默认只保留一份日期事项报告（`YYYYMMDD-事项.md`）与 `summary.json`。运行中的记录与含未闭合依赖（如待做离线质量评测的 NPY）的工作目录受 `ditserve_output_retention.py` 保护，仅在消费者完成且核验通过后按精确清单清理。
