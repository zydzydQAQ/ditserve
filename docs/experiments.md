# 实验配置

正式入口为 `.cache/build/ditserve run --config <study.json> --run-id <id>`，配置 schema 为 `ditserve.study.v1`。`configs/studies/mock.json` 是 CPU 样例；`configs/studies/{profile,online,state,sp}.json` 是通用模板。原生入口自行设置 worker 包路径，直接使用 Python API 时才需要 `PYTHONPATH=workers`。

jjjkairos 功能 preflight 不再使用静态历史配置。`benchmarks/cluster_config.py` 在作业内读取 `nvidia-smi` 的 GPU UUID，并生成 Wan2.1/Wan2.2 + SD3.5 的当前模型路径、worker、warmup 和小负载：

```bash
python3 /opt/ditserve/share/ditserve/benchmarks/cluster_config.py \
  --model wan21 --output /shared/output/DITSERVE/acceptance/<run>/preflight.json
ditserve run --config <preflight.json> --run-id <run-id> --allow-gpu
```

该配置仅验证功能，`profile.provenance=synthetic`，计划和 SLO 数字不是性能测量。正式实验必须由当前运行校准成本画像，并保留配置、BUILD_INFO、镜像 digest、模型 revision、拓扑和逐请求执行记录。

## 执行器与矩阵

| kind | 用途 |
| --- | --- |
| `profile` | 顺序或配对测量 |
| `online` | 单/多 worker 在线回放、预编码和异步解码 |
| `state` | 驻留、快照与迁移 |
| `sp` | 显式多 rank 动作 |
| `session` | 通用多阶段 replay 和请求依赖 |
| `cluster` | 原生多 GPU 混合图像/视频执行，正式 serving 入口 |

矩阵键是相对 phase 的 JSON pointer，生成笛卡尔积。`worker_config` 可修改模型配置，但不能改变 worker 设备或身份。每个 cell 独立创建模型进程，配对观测有独立 observation ID；失败观测保留在分母中。

worker 以完整 GPU UUID 绑定设备，模型使用 `cuda:0`。显式 `actions` 可执行 prepare/advance/finish/release/evaluate/distributed 等操作，state ID 必须稳定。C++ 控制预算、矩阵、进程监督和播放时钟，Python 保留模型、张量、RNG、质量和报告。

## 当前模型边界

- Wan2.1-T2V-1.3B 与 Wan2.2-TI2V-5B：`wan` 后端，xDiT/xFuser 固定 revision，DiT FP16、UMT5 CPU FP32，支持 step/block、CFG/USP、continuation 和显式 CFG fold。VAE dtype 与 attention 必须写入证据；BF16 VAE 不是 FP32 逐位一致。
- SD3.5-Medium：`sd35` 后端，BF16、Flash、固定离线 batch 和单 CPU Generator；是原子图像请求，不支持步骤迁移或连续批。
- Sana、FLUX、Mistral、LongLive 专用后端已删除。当前论文不使用其历史数据。

固定运行环境为 torch 2.10.0+cu126、diffusers 0.36.0、transformers 4.57.6、cache-dit 1.5.0 和 xDiT `16c34e7f070de65d9d2c67a6d74e0007dd8b426a`。CPU/Gloo 契约不等于 CUDA/NCCL 资格。

## 工作负载与输出

可使用冻结 `workload` 或确定性 `generator`（offline_all、steady、Poisson、bursty、Gamma）。`evaluation/workloads.py` 提供公开 prompt pool 的冻结混合 trace，manifest 记录来源和哈希；它不是生产日志。

`output_mode` 为 `file`、`digest` 或 `check`。无下游质量评测的 serving 实验使用 `check`，质量子集显式保留输出并登记依赖。`block_<form>` 与对应 `static_<form>` 的受控对照使用 `digest`，由 `benchmarks/native_exact.py` 检查全部视频哈希、形态和真实停止记录；这不是跨 SP 或 FP32 的等价声明。warmup 在到达时钟前完成，模型加载和首次 kernel 不计入首批请求；所有注册 GPU 从阶段开始到清理确认均计费。

## 原生 cluster 策略

策略行为的唯一来源是 `src/ditserve/control/policy.cpp` 的注册表（`PolicyTraits`）；规划器、执行器能力检查和模拟器读同一行。外部方法在同一执行器、同一 worker 和同一画像上重实现：只有 arXiv 预印本的方法按**方法简称**命名，已发表系统（TetriServe, ASPLOS'26）和真实引擎（xDiT）用系统名并注明重实现范围。`benchmarks/native_serving.py --suite` 生成论文各表的策略列表。

**主对比（`--suite main`）**

| 策略 | 论文名 | 来源 | 行为 | 未复现部分 |
| --- | --- | --- | --- | --- |
| `xdit`（=`xdit-cfg2`） | xDiT | xDiT 引擎常规部署 | 两个模型各占固定 GPU（图像份额 `partition_image_gpus` 按 trace 的 GPU 时间占比预先算出）；视频固定 cfg2、FCFS、开始后不抢占 | — |
| `tetriserve` | TetriServe (re-impl.) | ASPLOS'26 | step 级弹性 SP，每 `round_steps` 步一轮；每轮对可动视频做 0/1 group knapsack：价值=保持按时的请求数，代价=最小可行 GPU 数；剩余 GPU 先给救不回的请求再按 EDF 加宽 | 原文 batching、多节点；round 用 step 数近似固定时长 |
| `sbp-esp` | SBP-ESP | GenServe 的方法（arXiv:2604.04335） | step 边界让出视频 GPU 给图像，每步用同一 deadline knapsack 重规划，图像动态 batch（需画像含 `batch_ms`），VAE 单卡 | 原文效用函数；当前画像无 `batch_ms` 时图像 batch 不生效 |
| `edf-bestfit` | EDF-BF | GF-DiT 的方法（arXiv:2606.13501）`EdfBestFit` | EDF 次序取能按时完成的最小 size，否则取剩余可放的最大 size；再按 EDF 逐级加宽剩余 GPU | 任意 rank 组与 group-free collectives 需 `collective_backend=gfc`（阶段 B 才放开非对齐组） |
| `edf-cfg2` | EDF-CFG2 | 启发式 | 每个视频固定一对 CFG GPU，EDF，图像在视频 step 边界取卡 | — |
| `cfg2-first` | CFG2-First | 启发式 | step 边界弹性，但剪除被支配形态、视频不窄于一对 CFG GPU、只在每卡效率 ≥0.9 时加宽；无子步抢占 | — |
| `ditserve` | DitServe | 本文 | `cfg2-first` 的形态规则 + block 级 exact 子步抢占（doorbell）+ pair-fit 图像放置 | — |

**消融（`--suite ablation`，以 `ditserve` 为满配逐项拿掉）**：`abl-no-block`（整步量子）、`abl-no-prune`（不剪支配形态）、`abl-no-mindeg`（允许单卡 admit）、`abl-first-fit`（图像放最低号空卡）、`block-cfg2`（只保留 block 抢占、不弹性）、`edf-cfg2`（全部拿掉）、`ditserve-v1`（旧手调 knapsack 规划器）。

**受控与诊断**：`static-<form>`（校准与对照）、`block-<form>`（同形态 + 子步抢占，与 `edf-cfg2` 做 digest 配对，`--suite preempt`）、`ditserve-approx`（DitServe + CFG 折叠，Approx 单独成表）、`diag-*`（特征缓存、v3、tiered、2D-USP，不进论文表）。

**旧名**：旧记录与旧配置中的名字经别名表解析，运行结果写 canonical ID 并记录 `policy_alias_of`；历史 REPORT/XLSX 不改写。

| 旧名 | 新 ID |
| --- | --- |
| `heterodit_exact`、`knapsack_dp` | `ditserve-v1` |
| `heterodit_v5` | `ditserve-v1-approx` |
| `step_elastic` / `block_elastic` / `round_elastic` | `abl-step-elastic` / `abl-block-elastic` / `abl-round-elastic` |
| `static_<form>`、`block_<form>` | `static-<form>`、`block-<form>` |
| `heterodit_approx`、`heterodit_v3`、`ablation_tiered`、`ablation_2d_usp`、`static_partition` | `diag-feature-cache`、`diag-v3`、`diag-tiered`、`diag-2d-usp`、`diag-elastic-partition` |
| `vanilla_xdit`、`tridentserve_stage`、`vllm_omni`、`gf_dit`、`genserve_40gb` | 已删除：借用外部系统名的粗略近似，读取时报错 |

**规划器选项**：`prune_dominated`（某 class 的形态若被另一张卡数不多、步时不长且至少一项更优的形态支配则剪除）、`min_video_degree`（视频最小组大小）、`partition_image_gpus`（只作用于分区策略）、`round_steps`、`widen_efficiency`、`image_placement`、`order`、`demote_late`；省略时取注册表默认值。

**集合通信后端**：`options.collective_backend=nccl|gfc`。`gfc` 使用 vendored 的 group-free collectives（`workers/gfc`，MIT，来自 GF-DiT），承载 Ulysses all-to-all、SP 输出 all-gather、CFG 预测交换和 doorbell 归约；初始化失败即报错，不回退 NCCL。上线前先跑 `benchmarks/gfc_microbench.py`（建组开销、时延、与 NCCL 逐位一致），再用 `benchmarks/compare_digests.py` 比较同 trace 的 nccl/gfc 两次运行。

`ditserve-approx` 不默认叠加 Taylor、DBCache 或 INT8 Ring。Exact 与 Approx 结果必须分表。固定步数折叠与曲率门控都由 worker 的 `options.cfg_fold` 决定；只有 worker 在完整 CFG 步后返回 `cfg_fold_eligible=true`，规划器才可折叠。`max_folded_steps` 是每请求的独立计算预算，必须进入 continuation 的配置校验；曲率臂要与固定折叠臂报告实际折叠步数，不得把不同计算量直接称为算法收益。折叠状态随 continuation 导出/导入，最终 exact anchor 由配置决定。

并行形态支持 Wan 合法头数对应的 1/2/3/4/6/8 卡 SP 度及 CFG 组合，例如 `sp3`、`cfg2sp3`；`cfg2u3r1` 规范化为 `cfg2sp3`。组按拓扑顺序对齐，8 卡可实现 6+2→3+5。每次 assignment 和 worker `applied` 必须一致，否则运行失败。

`options.sp_transport`：`pinned` 限真实 SP1；`resume` 仅供 mock；`export` 在 2/4/8 卡进程组间通过 host-staged、带哈希的 step-boundary 状态换组。SP/CFG 的 mid-step packet 不跨组，SP1 可在 block 边界导出。文本预编码 ticket 避免换组时重复编码。

`options.async_rank0_decode=true` 使用 `decode_start` 启动后台 VAE，再轮询 `decode_poll` 并结算。VAE 固定使用 MATH；decode 期间该 GPU 不接受新视频，避免 DiT Flash 与 VAE MATH 的全局设置竞争，但独立图像 worker 仍可重叠执行。它不是同步 decode 的折扣模型，也不再同时做 chunked decode。

其他选项包括 `image_batch`、锁步 `pairing`、`reserve_image_gpus`、`quantum_blocks/quantum_ms`、feasible-first EDF 和 work-neutral widening。重实现的外部方法必须在论文中写明未复现部分。

## 成本画像与模拟

`profile.provenance` 为 measured、extrapolated、synthetic 或 mixed。视频类包含 steps、blocks_per_step、prepare/encode/finish、SP1 compute、通信量、collectives 和 CFG exchange；links 按度数和放置校准 bandwidth/latency/contention/overlap。模拟器与真实执行共用 `cluster_plan`，但模拟输出只代表成本模型。

正式 serving 可选择在同一作业内先校准；画像 `source.gpu_uuids` 来自校准预算身份。`--require-profile-match` 要求它与本次 `nvidia-smi` 分配逐一相同，缺失或跨分配复用会在模型加载前失败。三臂 Approx 共享一次校准，Exact/固定折叠/曲率折叠仍各自生成独立 run。What-If 的 CFG fold 起点必须显式写 `options.cfg_fold_eligible_from_step`，`cfg_fold_step_scale` 和 `async_decode_scale` 必须显式给出来源明确的标定值；缺失或越界会失败，不再使用隐藏的固定折扣。校准集与验证集必须分离，不允许从评测行反写带宽或 overlap 后宣称零重拟合。

## 数值口径与证据

“Exact”只相对同一优化 runtime 的不中断执行，绑定模型、dtype、attention、sampler、seed 和 decoder。CFG parallel 的逐位一致需单独验证；token-sharded SP 不默认逐位一致。BF16 VAE、INT8 通信、特征缓存和 CFG fold 均属单独数值模式。

最终交付为 `results/<experiment>/<run>/results.xlsx` 与 `REPORT.md`。`completed` 仅表示技术执行完成，性能、数值和语义资格分别判定。预算未结算、worker 未清理或依赖未闭合时不得归档为完成。重试使用新 run ID，保留失败现场。

## worker 协议

stdin/stdout 使用带版本的 JSONL，错误为 `ok:false`。模型和张量留在 Python，C++ 只接收状态 ID、进度和有来源的文件元数据。stdout 专用于协议，日志进入 stderr；异常、卡死和取消由所属进程组监督并清理。
