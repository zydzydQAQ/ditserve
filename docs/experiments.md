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

**规划器选项**：`prune_dominated` 仅对放置不变、无并发 SP 的画像做全局支配剪枝；放置或竞争可能翻转排序时不剪。`min_video_degree`、`partition_image_gpus`、`round_steps`、`widen_efficiency`、`image_placement`、`order`、`demote_late` 的默认值见注册表。加宽按全部剩余步数、量子开销和迁移成本评估，且不得把当前可按时的视频变成迟到；`widen_remaining_work=false` 保留旧单步规则作为负收益消融。lazy 门铃只在子步停止能把图像从迟到变为按时时触发，再按停止/恢复代价选择最便宜的视频组；已响铃组释放的容量计入需求。`doorbell.deadline_skip_events` 记录被过滤的无效抢占决策，这些实现仍需真实负载消融。

**集合通信后端**：`options.collective_backend=nccl|gfc`。`gfc` 使用 vendored 的 group-free collectives（`workers/gfc`，MIT，来自 GF-DiT），承载 Ulysses all-to-all、SP 输出 all-gather、CFG 预测交换和 doorbell 归约；初始化失败即报错，不回退 NCCL。上线前先跑 `benchmarks/gfc_microbench.py`（建组开销、时延、与 NCCL 逐位一致），再用 `benchmarks/compare_digests.py` 比较同 trace 的 nccl/gfc 两次运行。

`ditserve-approx` 不默认叠加 Taylor、DBCache 或 INT8 Ring。Exact 与 Approx 结果必须分表。固定步数折叠与曲率门控都由 worker 的 `options.cfg_fold` 决定；只有 worker 在完整 CFG 步后返回 `cfg_fold_eligible=true`，规划器才可折叠。`max_folded_steps` 是每请求的独立计算预算，必须进入 continuation 的配置校验；曲率臂要与固定折叠臂报告实际折叠步数，不得把不同计算量直接称为算法收益。折叠状态随 continuation 导出/导入，最终 exact anchor 由配置决定。主 Approx 默认 `fission_on_demand=true`：仅有其他等待请求可利用释放容量时才换组，无需求时保持原 CFG 组 fold-only；CFG2 可收缩为 SP1，不再受完整 CFG 的最小双卡约束。显式 `approx-<form>` 对照默认仍强制 fission。

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

## 冻结的正式混合分辨率协议（2026-11-02）

`configs/benchmarks/formal-mixed-v1.json` 是本项目正式 serving 协议，`benchmarks/formal_benchmark.py` 生成可执行配置。它使用公开 prompt 数据集，不是官方 serving benchmark 或生产到达日志；官方 VBench 是独立的质量协议。协议整体 canonical SHA256 写入每份配置，修改矩阵须产生新协议身份，不能事后改写已测 trace。

| 模型 | 开发/P0 | 正式空间分辨率（高×宽） | 正式帧数/步数 |
| --- | --- | --- | --- |
| Wan2.1-1.3B | 480×832，33 帧，20 步 | 480×832、720×1280，各占视频一半 | 81 / 50 |
| Wan2.2-5B | 512×896，49 帧，20 步 | 512×896、704×1280，各占视频一半 | 121 / 50 |
| SD3.5-Medium | 512×512，28 步 | 512×512、1024×1024，各占图像一半 | 1 / 28 |

Wan2.1 的 480p 是推荐分辨率，720p 为额外压力配置；Wan2.2 的 704×1280 对齐官方示例，较小配置用于空间异质性。两者均必须经过独立 GPU 资格验证，不能为了避免 OOM 或负结果静默删除某档。各档视频保持相同帧数/步数以隔离分辨率变量。设置依据：[Wan2.1 模型卡](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)、[Wan2.2 模型卡](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)、[SD3.5 模型卡](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)。这不是对其官方质量配置的全部复现：本运行采用的 dtype、VAE 和 attention 仍需单独标注与验证。

图像 prompt 使用冻结 DiffusionDB revision `fb620fbe49fa4420e0734bd9c0df11f51176b61f`，视频使用 VidProM revision `0c0900e6a037b058b1903bc1de9bcb6037d9fc24`，原元数据 SHA256 也必须匹配。读取既有 sealed pool，不下载视频/图像。以规范化文本的带盐 SHA256 桶分成 development 20%、calibration 20%、test 60%，相同文本或仅空白差异不会跨集合；实际生成保留原 prompt 文本。校准不使用正式测试 prompt，阈值调优也不能使用 test 分区。哈希抽样数按桶分布确定，不承诺恰好 20/20/60 个数比。

正式每 cell 为 200 请求、seed 301/302/303；开发为 40 请求、seed 201/202/203。视频/图像请求比例为 20/80 与 50/50；模态内部均分 shape。到达为固定请求数下的 Poisson 条件到达、归一化 Gamma（CV 参数 2）或 2 次突发（活跃比例 0.25），负载 0.6/0.9/1.2。Gamma 的有限样本实际 CV 不保证恰等于参数 2；突发负载是活跃段负载，长期平均由 manifest 单独记录，三类不能混为同一平均到达率。SLO 为每档校准 SP1 端到端服务时间的 video×1.5、image×2.5；所有策略使用同一冻结输入与 deadline。95% 每模态按时目标是稳定容量研究目标，有限 trace 不授予容量资格。

默认图像批上限 1，策略顺序按 seed 随机化。增加批次或到达/SLO 扫描应显式形成新协议版本并逐 shape 实测，不采用线性批耗时或复用开发画像。不同 shape 的分析 links 保留在各自 `profile.classes.<shape>.links`；全局 resize/quantum 开销取各校准的保守最大值，并明确标为聚合，不能称为一次全局实测。

三个入口按顺序执行（需 `PYTHONPATH=workers`，例子中的 UUID、路径应替换为实际值）：

```bash
python3 benchmarks/formal_benchmark.py calibrate --model wan21 \
  --video-shape wan21-480p --image-shape sd35-512 \
  --datasets /shared/data/DITSERVE/datasets --model-manifests manifests.json \
  --gpu-uuids GPU-0,GPU-1 --output calibration-480.json
# 另一独立校准为 wan21-720p + sd35-1024；运行并取得两份 run.json 后：
python3 benchmarks/formal_benchmark.py assemble --model wan21 \
  --runs calibration-480/run.json calibration-720/run.json \
  --runtime-facts runtime-facts.json --output shape-profiles.json
python3 benchmarks/formal_benchmark.py serve --model wan21 \
  --bundle shape-profiles.json --datasets /shared/data/DITSERVE/datasets \
  --model-manifests manifests.json --gpu-uuids GPU-0,GPU-1 \
  --split test --seed 301 --load 0.6 --video-fraction 0.2 --pattern poisson \
  --suite main --output serving.json
```

`manifests.json` 格式为 `{ "wan21": { "path": "/shared/data/DITSERVE/models/.../manifest.json", "sha256": "..." }, "sd35": { "path": "...", "sha256": "..." } }`。生成器只绑定声明，worker 必须实核文件清单才能授予权重资格。`assemble` 拒绝旧版无协议记录、混杂 shape、跨 GPU 分配、跨核心版本及重复 shape；每个组合分别组装，不能拿 Wan2.1 的画像代替 Wan2.2。运行时若换到其他 GPU 分配，必须重新校准。

P0 用 `wan21-dev/sd35-dev` 或 `wan22-dev/sd35-dev` 校准，再用 `serve --split development --seed 201 --diagnostic contention` 生成固定 CFG2 digest 对照；`--diagnostic no_preemption` 生成低负载全视频对照。无图像的 no-preemption 对照仍需检查实际停止数为零；contention 对照必须实际发生抢占，不能靠名称授予资格。开发 P0 通过仅放行正式 shape 的资格/校准，不自动证明正式性能。

质量方面，32 prompts×3 seeds 是配对误差研究的起点；正式 VBench 须由既有 `evaluation/vbench.py` 保持完整维度和 prompt 清单、每 prompt 5 seeds、temporal flickering 25 seeds 以及固定依赖，不得拿 VidProM 子集输出代称 VBench 分数。实际执行前还需通过阶段门禁，当前协议和配置生成测试本身不代表 GPU 或质量验收已通过。

新的随机分区不能消除既往探索污染。协议另外绑定 `known-exploration-prompts-v1.json`：从当前可读的 13 份历史原始记录与固定预热文本提取 162 个规范化 prompt 哈希，并记录每份来源哈希；这些 prompt 只允许进入 development。源 pool 已标 `split=dev` 或 `service_split=dev` 的全部文本也被排除出 calibration/test，包括其空白变体。若 pool 声明有旧 dev 集而样本角色丢失，生成器失败。该排除仅覆盖已知本地历史与 pool 角色，不能据此宣称未知远端、私有或未留存探索从未见过测试 prompt；发现新历史须更新排除清单和协议身份，再冻结实验。

P0 的 `p0_diagnostics` 现在独立冻结：contention 为 40 请求、50/50、bursty 活跃段负载 1.2、图像 SLO×1.1、视频×1.5，且关闭 lazy deadline 筛选（`lazy_preemption=false`），用于在有图像等待时检验真实安全点停止/恢复路径；no-preemption 为 40 个纯视频、Poisson、负载 0.1，保持 lazy 开启。每种场景内部的 EDF-CFG2 与 Block-CFG2 共享同一 trace/SLO，两个场景之间允许不同到达和构成。所有覆盖值写入配置 `benchmark.p0_diagnostic_controls`；原 CLI 请求值单独保留，不伪装为正式矩阵设置。

contention 的压力设置不能保证真实 GPU 必然停止，`native_exact` 仍必须观测到实际 stop 才通过；no-preemption 必须为零。关闭 lazy 只验证停止路径，不证明正式 deadline 门铃策略收益。P1 必须另测 `lazy_preemption=true` 下的实际停止、无抢占开销与端到端收益；正式主表保持原图像 SLO×2.5 和 lazy 默认开启。若诊断零停止，应调查执行证据，不自动调 SLO 重试直到得到有利结果。

安装布局为 `share/ditserve/benchmarks/formal_benchmark.py` 与 `share/ditserve/configs/benchmarks/`；CMake 已整体安装 `configs/`，协议和历史排除清单随同提交部署。生成器按自身脚本位置解析排除文件，缺失或哈希不符即失败，不依赖本地 `.local/evidence` 路径；清单中的历史路径只作来源说明。

## 分阶段执行与当前边界（2026-11-03）

这一轮本地实现了正式协议生成、每档独立通信画像、模型完整清单实核、P0 证据汇集和资源队列。它们属于实验基础与正确性修复，不产生新的 GPU 加速或质量结论。P0 尚未获得新 GPU 验收，P1–P5 尚未实施完成；`stage_gates.py` 对没有专门评估器和证据的阶段始终返回 pending，作业 Succeeded 不能提升阶段。

| 阶段 | 按序完成的工作 | 放行条件与主要产物 |
| --- | --- | --- |
| CONTRACT | 固定提交、现有镜像 digest，先 1×1 CUDA 冒烟，再完整契约与日期回归 | 两套非空测试无失败/跳过；核心与实际 Python 文件哈希绑定；只允许校准及 P0 诊断 |
| P0 | Wan2.1/Wan2.2 各做 contention 和 no-preemption，验证完整权重清单及逐请求 digest | 四个指定 case 全通过，实际停止/零停止、输入及 deadline 对齐、延迟对照齐全；只放行 P1 |
| P1 | 正式各 shape 的 OOM/数值资格；lazy 开启的 stop/step 响应与常驻开销；区分编排、停止、恢复、迁移成本 | 重复测量及区间，明确无法抢占的阶段；正式 shape 不可借用开发档资格，负收益如实保留 |
| P2 | 固定 CFG2、静态/弹性并行、Exact 策略及机制消融；完善按剩余工作定价的扩缩容和 deadline 门铃 | 同 trace、同算力、同 SLO、同质量的配对收益；按模态约束，不能仅靠总体均值掩盖视频退化 |
| P3 | 完整混合分辨率矩阵、多 seed、持续到达与稳定容量实验；补成本预测误差和队列稳定性 | 每模型 54 个 workload 配置，再乘策略数；报告失败分母、尾延迟、每模态 SLO 和容量区间；200 请求有限 trace 另列 |
| P4 | Approx 与 Exact 分表；配对数值/语义质量、相同实际折叠计算量对照；完善按需求释放 CFG 资源 | 至少 32 prompt×3 seed 的配对研究；证明释放卡确被其他请求使用；官方 VBench 独立满足完整协议 |
| P5 | 对齐外部系统的可支持功能与预算；增加不同模型家族和硬件/规模 | 官方/固定版本对照、适用边界与消融；现有两个 Wan 版本不能代替跨家族泛化，现有集群不能代替不同互连硬件 |

先用 development/calibration 数据定位 P1/P2 问题；正式 test 分区用于冻结后的评测。正式矩阵为 2 个视频模型×3 seed×3 负载×2 模态比例×3 到达模式，共 108 个 workload 配置，策略数另计。先测成本并估算 GPU 小时，再展开完整矩阵；失败项保留在分母中，不因 OOM、无收益或质量下降而换用更简单的未声明配置。

操作脚本位于项目根 `.local/ditserve-kairos-rewrite-runtime/`：`tools/generate_formal_batch.py` 只从已提交协议生成不可覆盖的规格，`scripts/kairos_formal_benchmark.sh` 在同一分配内校准并执行。正式模型清单必须来自已经冻结的预期模型 revision/文件哈希；不得把当前任意目录扫描所得哈希自行当成上游真实性证明。worker 首次加载前逐文件复核，此次初始化 I/O 不计为稳态吞吐收益。

`formal_benchmark.py preflight` 在 GPU replay 前检查来源、历史排除和剩余文本数量。开发 P0 至少需要 40 个独立视频 prompt、20 个图像 prompt；正式 test 至少需要 100 个视频、160 个图像，另需独立 calibration 文本。当前真实远端 pool 的剩余容量尚未复查，预检不足时应先扩充同源池并冻结新的输入身份，不能循环复用文本冒充独立请求。

先生成并通过 contract 规格，再给 P0 队列传 `--stage-gate <contract-receipt.json> --required-stage CONTRACT`。两个视频模型的 P0 作业可在资源允许时并行，但 CONTRACT 与 P0 不能放在同一个无门槛批次。`tools/kairos_resources.py --watch --interval 30` 只读观察；`tools/kairos_queue.py --backfill` 可让同阶段独立作业使用零散卡。必须以实时账号配额、节点可分配且未分配卡数为上限，GPU 利用率低不等于卡空闲，不能占用其他作业的分配。单个原子实验的 GPU 宽度与校准画像不可在运行中改写。

测试包必须从同一提交导出 `tests/`、`benchmarks/`、`configs/`；日期套件读取的 operator `scripts/` 和 `tools/` 也须同步核验，包含新增正式生成器与资源监控入口。测试环境内的路径是执行输入，不把另一个工作区的测试通过当成该发布版通过。镜像 digest 的证据当前来自提交规格绑定，不声称独立检查过运行 pod 的 imageID。

四份 P0 原始记录完成后，用明确路径汇集，缺项、重复、不同协议/模型/核心、失效来源均不能放行：

```bash
PYTHONPATH=workers python3 benchmarks/collect_stage_evidence.py \
  --contract-receipt /path/to/contract-receipt.json \
  --runs /path/to/wan21-contention/run.json /path/to/wan21-no_preemption/run.json \
         /path/to/wan22-contention/run.json /path/to/wan22-no_preemption/run.json \
  --output-dir /path/to/new-p0-collection
```

采集器不按时间戳猜测“最新”结果，失败也写明 blocked 原因。队列使用 receipt 时重算原始依赖；因此正式 run.json、contract 日志与证据清单须继续保护，不能因为已经存在 summary 就删除。最终归档迁移、依赖闭合及清理需另行核验。

容器 receipt 的来源路径通常是 `/shared/...`，下载到 WSL 后不能直接套用。`benchmarks/relocate_stage_evidence.py` 接受原 receipt 的本地副本、`--original-location` 与显式 `--map /shared/output/DITSERVE=<本地output镜像根>`（按引用补 data/code 映射），在逐项原始 SHA256 和重算门禁结果一致后，才向全新目录写 `stage-evidence.json/stage-receipt.json`。队列使用新的本地 receipt；原始测量文件不改写，映射缺失、来源损坏或状态漂移均失败，重定位不是省略依赖的归档。

2026-11-03 恢复执行：用户恢复网络访问并指示继续，控制台与 SSH 已可访问，重新核对了当前 web 使用文档和 OpenAPI。共享盘当次为 757G、74%；账号配额 16、其他项目占用 12，本项目可用余量 4。先前的 API 权限阻塞已解除。本地检查确认没有目标运行镜像，采用已提出并随继续指示放行的集群 1×1 替代冒烟；候选构建保持 `pending_cluster`，直到完整 CONTRACT 通过。之后计划在余量仍允许时，两模型各用 2 卡独立校准和 P0，对该分配记录资格；不能把 2 卡结果当成 8 卡验证。正式 operator 脚本和工具随构建复制到 `scripts/releases/<commit>/` 的独立目录，避免运行期间共享脚本变化。没有构建、推送或传输镜像；状态与容量仍以每次实时检查为准。

恢复访问后的实际输入核验：两个 sealed pool 各 4096 条，image 排除后 development/calibration/test 为 947/769/2380，video 为 865/795/2436，满足输入容量门槛。Wan2.1 与 SD3.5 的既有清单对应固定上游 revision；Wan2.2 旧目录明确复用 Wan2.1 的文本编码器，属于组合快照，不能当成完整官方 Wan2.2 模型。正式 Wan2.2 将使用新路径 `wan2.2-ti2v-5b-official-b8fff731`，补齐固定 revision 的 3 个官方 encoder 权重分片及 2 个配置文件（共 11,361,874,539 字节），保留原目录。`model_manifests` 映射可显式指定新 `model_path`，实际 worker 仍需全清单验证。输入就绪前先运行 Wan2.1；P0 的 contention/no-preemption 可生成两个独立 2 卡作业，各自在自身分配校准，不把不同分配的画像混用。

本次冻结前另修复三处控制器边界：lazy 门铃按同一候选组核对每个图像的 deadline，并预留已经响铃的可用容量；remaining-work 扩组禁止单步 pairing 收益绕过整体成本保护；CFG fission 保留 class 最小并行度的安全约束。该门铃是逐图像 EDF 可行性启发式，不是全局最优求解。fission 按需求触发仍待 P4 证明释放容量的实际复用与布局往返收益。

### P1 固定形态诊断的准备入口（2026-11-03）

`configs/benchmarks/p1-qualification-v1.json` 单独绑定原正式协议，保留全部六档正式 shape、开发 prompt 分区、三个开发 seed。四个 video/image pair 分别运行 contention 与 no-preemption，共 24 个源记录、1920 个策略请求。两臂固定 CFG2、digest，lazy 均开启，SLO 使用正式 video×1.5/image×2.5；不沿用 P0 关闭 lazy 的压力条件。

`benchmarks/p1_benchmark.py --pair wan21-0 --role contention --seed 201 --bundle <same-allocation-bundle> --datasets <sealed-pools> --model-manifests <frozen-mapping> --output <new-config>` 生成单个 case。稀疏到达的时间预算按冻结规则使用实际 trace horizon、校准服务时间及启动余量计算，避免正式高分辨率 trace 超过开发预算；这只是执行预算估算，不是实测性能。

操作工具 `tools/generate_p1_batch.py <commit> --gpus 2 --model-manifests <remote-mapping>` 只导出不可覆盖的规格。每作业独立校准一个正式 pair，再运行三个 seed；提交仍要求 `--required-stage P0`。不得为跳过这项门槛把 P1 标为 calibration 或直接使用 CONTRACT receipt。

`benchmarks/p1_qualification.py --manifest <evidence.json> --output <new-review.json>` 接收 `ditserve.p1-evidence.v1`：protocol、plan、p0_receipt 均为路径与 SHA256 引用，runs 明确列出 pair_id/role/seed/path/sha256。它重验原始 P0、实际 GPU 分配、完整运行时与模型身份、实际 trace/SLO/shape，以及两模态有限 digest。缺失、执行失败和零停止不从计划分母中删除；误差区间按独立 seed 聚合。

该入口目前仅评审固定 CFG2 子资格，不生成 P1 晋级 receipt。完整 P1 还缺所有允许形态的资格和独立恢复/导入/编排计时；语义质量归属独立评测。三个 seed 的区间是小样本负载变化估计，不是同输入计时重复性证明。阈值为 null 时只报告效果区间，不能事后挑阈值宣布收益。

另有独立的 `p1_isolated_batch.py --inputs INPUTS.json --inputs-sha256 SHA256 --output CONFIG.json`，只编译有限离线批诊断：每个正式 shape pair、同两张 GPU，8 条开发视频同时到达，三个 seed 各执行 EDF→Block→Block→EDF，共 96 策略请求/shape pair、全计划 384 次。图像模型驻留与 warmup 保留，测量窗口无图像；两轮时序重复嵌套在 seed 内。输入清单绑定正式校准原始记录、模型映射、core/Python/runtime 身份，并要求重新拟合后与 bundle 一致，不能借用其他分配或四卡校准。添加 `--config` 与 `--run` 仅审查实际 observation/cell/attempt 的覆盖与失败分母，不授予数值、零停止、性能或 P1 资格。该诊断不能替代原 v1 的低负载 Poisson 唤醒问题，提交仍须 P0 门禁；原 v1 不变。

2026-11-03 的 `12b0918` P0 contention 在第一个 Block-CFG2 cell 暴露固定形态恢复漏洞：step 边界请求的 GPU 虽空闲，仍可能保留另一视频未完成的 solver step。最终互斥检查正确拒绝计划；30 请求完成、10 中断，另一策略 40 请求尚未执行，不能授予资格。修复在固定恢复分支补充已有 owner 筛选，保留 mid-step/合法 lockstep 及 pinned 路径；用真实到达/截止时间与所有 59 个可能的中途 block 位置做回归。新核心需要新的 CONTRACT/P0；旧失败数据和仍未闭合的原始依赖继续保留。

同日成本审查发现另一独立问题：profile 的 `blocks_per_step=60` 计完整 CFG 双分支，真实 Wan CFG2 每 rank 执行 30 个。原 planner 将 rank 的 `blocks_done` 当完整分支计数，600 ms 步执行一半时估余下 450 ms 而非 300 ms；时间量子也可能下发约两倍实际 block。修复显式区分 native rank 计数与 mock/simulator 原计数，按观测形态换算已完成比例，并统一剩余工作、边界等待、量子及配对计数。实际 folded 步的计数同样只有 30；边界上上一轮的 folded 状态不证明下一轮已折叠。这里只修正计数，仍沿用完整 CFG 步时估计，未授予 folded 成本准确性或性能上界。新 core 须独立 CONTRACT/P0 和同核心校准，旧 bundle 不直接继承资格。

`92b72ff` 的两组开发抢占诊断均为 80/80 请求完成、20 视频摘要配对一致，完整 P0 仍缺无抢占用例。其 Wan2.2 最大停止 370.45 ms 的 controller 时间线表明：ring 后约 26.63 ms 导入才就绪，约 1.74 ms 后开始 advance，rank advance 联合跨度约 341.85 ms，最终观察约 0.23 ms。该相邻 wall 分解不相加并行 rank RPC，也不代表完整恢复成本；advance 内部前导与 kernel 尚未细分。导入就绪后、advance 前的可取消边界是后续独立机制实验，不能把稳态 block 估计当作首量子的总停止保证。

### P2 配对比较与单机制消融（2026-11-03）

`benchmarks/p2_contrasts.py` 从至少三个 seed 的冻结配置准备 reference/candidate 两臂、计划和空证据模板。支持固定/弹性 block 对照、pruning、minimum-pair、placement、lazy、remaining-work 和 continuous doorbell；也可准备与指定系统策略的同框架比较。它审查实际 policy/options 默认值，拒绝被显式 options 抹成同臂的消融。doorbell 对照统一 block 检查间隔；仅 CFG2 的配置不能用于证明形态剪枝/扩组收益。xDiT 分区基线若没有 image 分区外的合法 CFG2 组（例如两卡）直接标为不可支持，不能将其不可执行算成 DitServe 的胜出。

示例：`PYTHONPATH=workers python benchmarks/p2_contrasts.py --protocol configs/benchmarks/formal-mixed-v1.json --workload-id <name> --config <seed301.json> --config <seed302.json> --config <seed303.json> --contrast block-elastic --contrast lazy --output-dir <new-dir>`。这些是可执行配置，工具本身不提交作业，也不绕过阶段资格。

`benchmarks/p2_benefit.py --manifest <evidence.json> --output <new-summary.json>` 明确读取每个 case/arm 的哈希绑定 raw run；selector 包含 phase、attempt、cell 和执行 order。按冻结配置枚举请求分母，保留失败、中断以及未开始的策略 cell；没有测得 cell 耗时时不补造零吞吐或速度比。模态分别统计并按独立 seed 聚合，不以总均值掩盖视频退化。它还检查源码/核心/运行时/模型/分配/实际输入一致性，摘要不替代跨形态数值或语义质量资格。

开发 P0/P1 来源的 split、role、SLO 和诊断 controls 都保留，不能因派生 P2 配置而变为正式测试主表。已用实际失败的 `12b0918` 原始记录验证审查器：Block 的 30 完成/10 中断与尚未启动 EDF 的 40 项分别保留；该结果仅用于验证分析工具，不能作为收益比较。P2 阶段仍需要真实受控测量及独立质量证据。

### 成本观测与后续阶段工具（2026-11-03）

原始 cluster request 增加 `controller_timing`，记录同一控制器回放时钟上的任务启动、prepare/export、并发 import 就绪、advance 完成与控制器观察时间，并保留异常前缀。独立调度计时锚保留原有抢占估计语义。停止和下一次恢复通过 event ID 关联，首次分发、换组、原地恢复及 tiered restore 分开；规划计时只包 `cluster_plan`。这些是嵌套区间，rank RPC 并行；stop 与原 advance 重叠，不能相加成迁移/恢复开销。新测量没有增加 worker RPC、CUDA 同步或 barrier，但真实观测开销仍须测量。

`benchmarks/p1_costs.py --manifest <cost-evidence.json> --output <new-summary.json>` 读取 `{schema: ditserve.p1-cost-evidence.v1, runs: [{path,sha256}]}`。只报告原始时序诊断；导出到所有新 target 就绪不代表所有旧副本回收，停止到下一 dispatch 包含排队与其他请求服务，下一次 advance 完成还包括有用模型计算。原始 `run.json` 是该分析的未闭合依赖，不能在分析前删除。

P3 的 `p3_stream.py` 生成固定 warmup/measurement/drain 观察窗内不归一化的 Poisson 到达，请求数自然随机，模态与分辨率按冻结概率取样；全计划 prompt 余量不足即拒绝。`p3_window_review.py` 复算到达、SLO 和输入规格，保留失败、未开始和右删失分母，分别统计 arrival cohort 与完成窗口，按 seed 给区间。当前 drain 是分析截止，未实现核心精确停止；有限窗口与相关 queue bins 不授予稳定容量资格。

P4 的 `p4_quality_plan.py prepare` 固定同一组每模型至少 32 prompt × 3 个 generation seed，并将提示词选择 seed 与生成 seed 分离。正式 50 步的 late-window、折叠预算与质量阈值需独立开发证据；固定 N=1..K、adaptive 与 exact 的控制保持实际 unconditional-forward 数量可比，N=0 复用 exact 且不记为算法收益。它检查官方模型映射、同分配每档校准与完整/逐配对临时存储预算，尚不生成可执行作业或质量 receipt。LPIPS 权重、完整 VBench 和依赖闭合仍待完成。PSNR 的无限值以 exact-match 关系报告，不做 ∞−∞ 或任意截顶。

`p4_stream_quality.py --plan ... --evidence ... --output ...` 现可离线读取已绑定该计划的成对 NPY：最多同时打开两份只读 memmap，以逐帧工作数组计算 MSE、PSNR、SSIM 和时序差分误差。文件与数组 SHA256、实际 matrix/worker_config/video_worker_options 合并顺序、模型、输入、GPU 放置与完整折叠历史都须核验；旧的未绑定输出不能直接改名进入此计划。所有缺失、失败项保留在计划分母，按 prompt 聚类重采样并保留 seed/shape 关联，另按 shape 与实际折叠 N 分层。无限 PSNR 以精确关系和有限 MSE 差分处理。该工具没有 GPU runner；v2 通过显式 `--lpips` 接入离线适配器，实际包、源码、权重与真实 CPU selftest 均通过后才逐帧评分，任一帧异常则该对 LPIPS 为 unknown，不只平均成功帧。VBench 仍为 unknown，不能授予 P4 或算法优越性资格；memmap 也不是硬 RSS 上限。`p4_quality_batches.py --plan ... --batches ... --output ...` 只归并原始封存 v2 批次，核对同一计划、核心、模型、运行时及指标身份，重新按 prompt 聚类计算统计量。完全相同且已核验的 exact 参考可跨批复用并只计一次；重复候选、冲突参考和不同 LPIPS 可用状态拒绝合并。完整分母保留，原始输出依赖继续开放，不执行删除或磁盘回收。

`p4_offline_lpips.py selftest` 是独立 CPU 自检入口，上述 stream 的可选 LPIPS 路径也执行同一真实自检。它固定当前镜像的包版本、关键源码和两份本地权重哈希，以 `pretrained=False, pnet_rand=True` 构造后严格装载全部特征层及五个标定头，防止 torchvision 隐式下载；评分前须通过真实权重的精确对照、扰动、对称性及独立公式复算。Python 网络/子进程拦截用于专用单线程 Python 进程，不是 OS 沙箱或全部 native 依赖认证。本地缺少 torch 时必须返回 unknown，mock 测试不能代替固定镜像中的实际自检。它不授予 P4 或 VBench 资格，也不闭合原始数据的清理依赖。

`native_metrics.py` 的释放容量与图像重叠改为按 GPU 合并时间区间，batch 成员不重复累计 GPU 时间；使用最近一次 full-CFG 布局，无法追溯的换组不获得释放额度。该指标仍是 placement/window proxy，`reuse_qualified=false`，尚缺 release ACK、epoch 生命周期、真实受益执行与 fold-only/forced/demand 因果对照。不同 attempt/order 的请求分母不得合并，缺身份的多臂旧记录拒绝推断。

P5 的 `p5_external_preflight.py` 对照固定官方源码声明和已测运行环境，仅检查选定依赖子集；未观测包保持 unknown，不等于未安装或可运行。`p5-external-v1.json` 固定 TetriServe、GF-DiT、xDiT 来源和接口，并列出模型、输入、算力、质量、计时与规模证据要求。当前已知 TetriServe 的 diffusers/NumPy 和 GF-DiT 的 cache-dit 与现有镜像冲突；不得安装替换或自行构建镜像来绕过项目限制。同框架策略仍标为重实现，SD3 不自动代表 SD3.5、两个 Wan 版本不代表跨家族、两个独立 8 卡作业不代表跨节点 16 卡。工具输出始终是预审，不能提升 P5。
