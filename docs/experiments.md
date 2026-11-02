# 实验配置

正式入口为 `.cache/build/ditserve run --config <study.json> --run-id <id>`，配置 schema 为 `ditserve.study.v1`。`configs/studies/mock.json` 是 CPU 样例；`configs/studies/{profile,online,state,sp}.json` 是通用模板。原生入口自行设置 worker 包路径，直接使用 Python API 时才需要 `PYTHONPATH=workers`。

jjjkairos 功能 preflight 不再使用静态历史配置。`benchmarks/cluster_config.py` 在作业内读取 `nvidia-smi` 的 GPU UUID，并生成 Wan2.1/Wan2.2 + SD3.5 的当前模型路径、worker、warmup 和小负载：

```bash
python3 /opt/ditserve/share/ditserve/benchmarks/cluster_config.py \
  --model wan21 --output /shared/output/DITSERVE/acceptance/<run>/preflight.json
ditserve run --config <preflight.json> --run-id <run-id> --allow-gpu
```

该配置仅验证功能，`profile.provenance=synthetic`，计划和 SLO 数字不是性能测量。正式实验须校准并核对当前放置的画像，将配置、BUILD_INFO、镜像 digest、模型 revision、拓扑和逐请求执行记录保留到分析/评测闭合，再发布关键摘要。单机原生配置支持 2/4/6/8 张 GPU；更大规模模拟不代表已完成多节点 GPU 测试。

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
- SD3.5-Medium：`sd35` 后端，BF16、Flash，具有原子生成、micro-batch、step-wise continuous batch 与 token packing 路径；后两者仍需独立 GPU 正确性/性能资格验证，不能因接口存在就算作已验证贡献。单请求多图与多请求批处理的 seed 协议不同，须记录实际执行方式。

目标运行环境固定为 torch 2.10.0+cu126、diffusers 0.36.0、transformers 4.57.6、cache-dit 1.5.0 和 xDiT `16c34e7f070de65d9d2c67a6d74e0007dd8b426a`，实际版本须进入输出身份。CPU/mock/Gloo 契约不等于 CUDA/NCCL 资格或 GPU 加速结果。

## 工作负载与输出

可使用冻结 `workload` 或确定性 `generator`（offline_all、steady、Poisson、bursty、Gamma）。`evaluation/workloads.py` 提供公开 prompt pool 的冻结混合 trace，manifest 记录来源和哈希；它不是生产日志。

`output_mode` 为 `file`、`digest` 或 `check`。无下游质量评测的 serving 使用 `check`，质量子集保留文件并登记依赖。`--suite preempt` 固定 `edf-cfg2/block-cfg2`、CFG2 和 digest；`native_exact.py` 核对完整输入、runtime、输出形状/dtype/有限性及实际停止。`--no-preemption-diagnostic` 用低负载全视频对照测常驻开销，必须由 `native_exact.py --no-preemption` 确认零停止。哈希相同不替代权重完整性验证，也不证明跨 SP 或 FP32 一致。

各策略共享相同 trace、图像批次上限和输出设置。`--image-batch` 默认 1；大于 1 时必须有从 2 到上限每个批次的实测 `sd35-preflight.batch_ms`，缺失即拒绝配置。连续批 `options.continuous_batch` 与原子 `image_batch>1` 互斥，原生 serving 生成器不自动开启连续批。批处理成员、实际 batch_size 和批耗时必须在执行记录中可核对。

CLI 默认按 trace seed 可复现地随机排列策略，可用 `--policy-order/--order-seed` 控制；多种顺序和重复运行用于识别顺序漂移。warmup 在到达时钟前完成，模型加载/预热不计入请求延迟，全部登记 GPU 从阶段开始到清理确认仍计费。

## 原生 cluster 策略

策略行为的唯一来源是 `src/ditserve/control/policy.cpp` 的注册表（`PolicyTraits`）；规划器、执行器能力检查和模拟器读同一行。外部方法在同一执行器、同一 worker 和同一画像上重实现：只有 arXiv 预印本的方法按**方法简称**命名，已发表系统（TetriServe, ASPLOS'26）和真实引擎（xDiT）用系统名并注明重实现范围。`benchmarks/native_serving.py --suite` 生成论文各表的策略列表。

**主对比（`--suite main`）**

| 策略 | 论文名 | 来源 | 行为 | 未复现部分 |
| --- | --- | --- | --- | --- |
| `xdit`（=`xdit-cfg2`） | xDiT 固定分区 | 一个明确的 xDiT 部署配置 | 两模型固定分区，图像份额按 trace 的 GPU 时间占比预先算出；视频 cfg2、FCFS、不抢占 | 不是自动选优后的全部 xDiT 配置 |
| `tetriserve` | TetriServe (re-impl.) | ASPLOS'26 | step 级弹性 SP，每 `round_steps` 步一轮；每轮对可动视频做 0/1 group knapsack：价值=保持按时的请求数，代价=最小可行 GPU 数；剩余 GPU 先给救不回的请求再按 EDF 加宽 | 原文 batching、多节点；round 用 step 数近似固定时长 |
| `sbp-esp` | SBP-ESP | GenServe 的方法（arXiv:2604.04335） | step 抢占、逐步 deadline knapsack、弹性 SP、单卡 VAE；图像 batch 服从统一显式上限和实测画像 | 原文效用函数与完整系统未复现 |
| `edf-bestfit` | EDF-BF | GF-DiT 的方法（arXiv:2606.13501）`EdfBestFit` | EDF 次序取能按时完成的最小 size，否则取剩余可放的最大 size；再按 EDF 逐级加宽剩余 GPU | GFC 后端单独验证；当前组选择仍受拓扑对齐约束，未复现完整任意组调度 |
| `edf-cfg2` | EDF-CFG2 | 启发式 | 每个视频固定一对 CFG GPU，EDF，图像在视频 step 边界取卡 | — |
| `cfg2-first` | CFG2-First | 启发式 | step 边界弹性，但剪除被支配形态、视频不窄于一对 CFG GPU、只在每卡效率 ≥0.9 时加宽；无子步抢占 | — |
| `ditserve` | DitServe | 本文 | `cfg2-first` 规则 + block 子步抢占（doorbell）+ pair-fit 图像放置 | 输出保持需同形态配对验收；跨 SP 不承诺逐位一致 |

**消融（`--suite ablation`，以 `ditserve` 为满配逐项拿掉）**：`abl-no-block`（整步量子）、`abl-no-prune`（不剪支配形态）、`abl-no-mindeg`（允许单卡 admit）、`abl-first-fit`（图像放最低号空卡）、`block-cfg2`（只保留 block 抢占、不弹性）、`edf-cfg2`（全部拿掉）、`ditserve-v1`（旧手调 knapsack 规划器）。

**受控与诊断**：`static-<form>`（校准与对照）、`block-<form>`（同形态 + 子步抢占，与 `edf-cfg2` 做 digest 配对，`--suite preempt`）、`ditserve-approx`（DitServe + CFG 折叠，Approx 单独成表）、`diag-*`（特征缓存、v3、tiered、2D-USP，不进论文表）。

重实现只支持所列机制范围，结果不能直接宣称击败原系统。旧名由 `policy.cpp` 别名表处理，结果写 canonical ID 和 `policy_alias_of`；历史报告不改写。

**规划器选项**：`prune_dominated` 仅对放置不变、无并发 SP 的画像做全局支配剪枝；放置或竞争可能翻转排序时不剪。`min_video_degree`、`partition_image_gpus`、`round_steps`、`widen_efficiency`、`image_placement`、`order`、`demote_late` 的默认值见注册表。门铃按等待图像需求及停止/恢复代价选择视频组，已响铃组释放的容量计入需求，避免对同一请求反复扩大抢占范围；这些实现仍需真实负载消融。

**集合通信后端**：`options.collective_backend=nccl|gfc`。`gfc` 使用 vendored 的 group-free collectives（`workers/gfc`，MIT，来自 GF-DiT），承载 Ulysses all-to-all、SP 输出 all-gather、CFG 预测交换和 doorbell 归约；初始化失败即报错，不回退 NCCL。上线前先跑 `benchmarks/gfc_microbench.py`（建组开销、时延、与 NCCL 逐位一致），再用 `benchmarks/compare_digests.py` 比较同 trace 的 nccl/gfc 两次运行。

`ditserve-approx` 不默认叠加 Taylor、DBCache 或 INT8 Ring。Exact 与 Approx 结果必须分表。固定步数折叠与曲率门控都由 worker 的 `options.cfg_fold` 决定；只有 worker 在完整 CFG 步后返回 `cfg_fold_eligible=true`，规划器才可折叠。`max_folded_steps` 是每请求的独立计算预算，必须进入 continuation 的配置校验；曲率臂要与固定折叠臂报告实际折叠步数，不得把不同计算量直接称为算法收益。折叠状态随 continuation 导出/导入，最终 exact anchor 由配置决定。

并行形态受模型头数和分配卡数约束，可包含 `sp3`、`cfg2sp3` 等；Wan2.1 不使用非法的 SP8。组按拓扑顺序对齐。每次 assignment 和 worker `applied` 必须一致，否则运行失败。

`options.sp_transport`：`pinned` 限真实 SP1；`resume` 仅供 mock；`export` 在合法进程组间通过 host-staged、带哈希的 step-boundary 状态换组。SP/CFG 的 mid-step packet 不跨组，SP1 可在 block 边界导出。文本预编码 ticket 避免换组时重复编码。

`options.async_rank0_decode=true` 使用 `decode_start` 启动后台 VAE，再轮询 `decode_poll` 并结算。VAE 固定使用 MATH；decode 期间该 GPU 不接受新视频，避免 DiT Flash 与 VAE MATH 的全局设置竞争，但独立图像 worker 仍可重叠执行。它不是同步 decode 的折扣模型，也不再同时做 chunked decode。

其他选项包括 `image_batch`、锁步 `pairing`、`reserve_image_gpus`、`quantum_blocks/quantum_ms`、feasible-first EDF 和 work-neutral widening。重实现的外部方法必须在论文中写明未复现部分。

## 成本画像与模拟

`profile.provenance` 为 measured、extrapolated、synthetic 或 mixed。视频类包含步数、block数、编码/解码与compute/通信项。实测 `forms.<form>.placements` 按拓扑有序 rank 范围绑定放置；不匹配或并发 SP 时走分析回退，不能把另一组的测量当作本组实测。`fit_native_profile.py` 拒绝失败校准、混杂shape及不完整batch身份；held-out form误差只验证同校准硬件/shape上的形态，不能宣称已做跨模型、放置或负载验证。

模拟器与真实执行共用 `cluster_plan`。`scalability_study.py` 的 Rate/SLO/CV/GPU 扫描是模拟及规划器 CPU 计时；超过实测规模的点为外推，不是集群扩展性或多节点 GPU 性能证据。

正式 serving 应开启 `--require-profile-match`：画像 GPU UUID 与所用设备的有序前缀逐一匹配，缺失或跨分配复用即拒绝。该身份检查不代替拓扑及竞争状态测量。Exact/固定折叠/曲率折叠共享校准和固定形态但保留独立运行；`--quality-arm` 生成受控配置。What-If 折叠起点和计算/解码折扣必须显式给出有来源的值。校准与验证数据必须分离，不得按评测行反写参数。

## 数值口径与证据

“Exact”只相对同一优化 runtime 的不中断执行，绑定模型、dtype、attention、sampler、seed 和 decoder。CFG parallel 的逐位一致需单独验证；token-sharded SP 不默认逐位一致。BF16 VAE、INT8 通信、特征缓存和 CFG fold 均属单独数值模式。

默认长期交付为日期事项报告与 `summary.json`，详见[结果处理](results.md)。执行原始记录保留至分析/评测依赖闭合；摘要不支持原始重评。`completed` 不表示性能、数值或语义资格通过，未完成评测不得把依赖标为已闭合。主结果须有多seed、足量请求、失败分母和误差区间；有限trace的按时goodput不等于稳定容量。

## worker 协议

stdin/stdout 使用带版本的 JSONL，错误为 `ok:false`。模型和张量留在 Python，C++ 只接收状态 ID、进度和有来源的文件元数据。stdout 专用于协议，日志进入 stderr；异常、卡死和取消由所属进程组监督并清理。
