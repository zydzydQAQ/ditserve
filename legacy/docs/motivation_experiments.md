# 视频 DiT 多请求 Serving Motivation 实验设计

本文档固化论文 motivation 节的实验设计：从缓存、抢占、内存、SP、LP 五个角度，用本机 A100-PCIE-40GB（物理 0）与 V100-PCIE-32GB（物理 1）产出量化证据。本文档只做设计与预算规划；任何 GPU 执行均沿用既有账本纪律，先确认归属再启动。

## 定位与核心命题

对立面是应用透明的 GPU 时分复用（如 Nixie, OSDI'26）：它不感知 DiT 语义。本研究命题：视频 DiT 的领域结构——步进去噪、DBCache 特征缓存生命周期、CFG 双分支——使**应用感知调度严格优于透明复用**，且质量约束的精确状态恢复使其安全。统一抽象：**特征缓存是一阶可调度状态**，其生命周期证书同时驱动抢占动作选择、内存回收准入与代价模型；异构布局与 SP/LP 是同一状态放置问题的布局维度。

已有实测事实（本仓库既有结果）：

- F1：A100 上 Flash attention 使单请求 E2E 110.68s→37.79s（去噪 93.68s→18.57s）。
- F2：CPU 文本编码约 18.7s/请求，有界流水线带来 20-26% 在线 E2E 下降（`results/serving-optimization/`）。
- F3：batch=2 吞吐仅 +1.86~2.10%，峰值 allocated 显存 11.07→18.98 GiB——并发受显存容量约束，非算力。
- F4：DBCache（Fn=8/Bn=0/warmup=8/阈值/m1 上限）使单步成本双峰化，compute/reuse 交替且由策略状态决定。
- F5：keep/hot/cold 状态迁移与逐步位级恢复已在单卡 A100 验证（run03 冻结记录）；刷新证书制导的精确回收逐步对照通过。
- F6：SP4→SP2→SP4 跨进程弹性迁移已在 CPU/Gloo 逐位验证；GPU 数值验证需 4 张同型卡，本机不满足，维持后置。

## 实验矩阵

形状默认 480×832、17 帧、50 步、CFG=5、FP16 DiT、FP32 VAE；M2 用 32 步/混合帧长见各行。所有真实运行使用固定本地模型快照（revision `0fad780a`），HF 离线，每实验新输出目录。

### M1 — 步级成本双峰性 × 硬件（缓存）

- 命题：单步去噪成本双峰且可预测；compute:reuse 代价比在 A100/V100 上不同 → 代价模型需缓存+硬件双感知。
- 方法：`ditserve profile --model wan` 诊断运行记录逐 forward CUDA event 耗时并映射缓存证据；cache ∈ {off, cache-dit 0.08/m1}，2 个固定 prompt × seed 0，warmup 1 + 无观测 1 + 诊断 1。
- 产出：双卡步时阶梯图、compute/reuse 步成本比、缓存命中映射。

### M2 — 长尾服务时间下的 HoL 阻塞与 SLO 崩塌（抢占）

- 命题：混合帧长下 FCFS 的按时完成率在中等负载即崩塌；EDF/SRTF 受不可抢占与均值代价估计限制 → 引出步粒度抢占与缓存感知估计。
- 方法：`ditserve benchmark run --backend wan --attention flash --policies fcfs,edf,srtf`；缩小版 formal：480×832、17/49 帧混合、每 trace 24 请求、Poisson 参考负载 {0.5, 1.0}、1 个 trace seed。
- 产出：goodput-负载曲线 ×3 策略，标注优先级反转事件。样本量仅支持描述性统计，不声称稳定 p99。

### M3 — 容量约束下的精确回收准入（内存）

- 命题：并发度由显存而非算力决定；证书制导精确回收在无质量损失下提升准入数。
- 方法：复用已就位的 `research mechanism --kind state`（含两请求自然容量试验），叠加 batch=2 既有证据；不经本设计新增配置。
- 产出：keep vs 精确回收准入对照、LPIPS 质量门证据。

### M4 — 异构相位加速比矩阵（布局）

- 命题：encode(CPU)/compute 步/reuse 步/decode 各相位双卡加速比不同 → 同构假设的放置策略在异构集群失真。
- 方法：同一冻结输入双卡 profile（复用 M1 数据，补 VAE decode 计时）；实测比值灌入 trace replay 对比放置策略，明确标注 replay 非在线验证。
- 产出：2×4 相位×GPU 加速比矩阵、放置策略 replay 对比。

### M5 — 抢占/迁移成本实测（抢占）

- 命题：步边界抢占+恢复成本 << 剩余步成本；hot(CPU offload) 受 PCIe 带宽约束，双卡不同。
- 方法：WanStepWorker keep/hot/cold 切换延迟；A100 侧大部分已由 research state 阶段覆盖，补 V100 侧 worker 迁移计时。
- 产出：动作×GPU 迁移成本表，对照单步成本。

## 预算现状与归属（2026-10-13 实测）

| 账本 | 上限 | 已用 | 剩余 | 本设计归属 |
| --- | ---: | ---: | ---: | --- |
| benchmark（权威账本在旧仓库 `~/code/dit/results/video-benchmark/gpu-budget.json`） | 7,200 GPU·s | 5,810.7 | **1,389.3** | M1-A100、M2 |
| research（`results/research-pilot/gpu-budget.json`） | 28,800 GPU·s | 1,906.5 | 26,893.5（分桶：标定+机制 9,000、SP 9,000、复核 3,600 等，余量不互借） | M3（机制桶，已分配未用） |
| V100 profile/迁移计时（M1-V100、M5-V100） | — | — | — | 无既有归属，见决策点 |

粗估（执行前以 probe 实测修正）：M1-A100 约 13 次生成 × ~40s ≈ 600-1,200 GPU·s（可装进 benchmark 余量，但占满后 M2 无余额）；M2 约 2 负载 × 3 策略 × 24 请求 × 40-120s ≈ 5,800-14,000 GPU·s，**远超 benchmark 余量**；M1-V100 约 13 次 × 400-700s ≈ 5,200-9,100 V100 GPU·s，无既有账本。

**决策点（需用户裁决，不擅自扩账）**：
1. M2 及 M1/M5 的 V100 部分超出现有余量——追加 benchmark 预算、为 V100 测量设立归属，或裁剪实验规模（如 M2 减为 1 负载 × 3 策略、M1-V100 减为 1 prompt × 2 配置）。
2. 不允许新建账本规避上限、不改既有账本路径；超预算单元按 fail-closed 停止。

## 执行顺序

1. V100 线（物理 1，当前空闲）：重建 `.venv-v100`（钉死配方：uv + cp310 + torch 2.10.0+cu126 含 sm_70 wheel + `configs/requirements-v100.txt`，冻结 lock）→ preflight 验证 V100/sm_70/math SDPA → M1-V100 → M5-V100。
2. A100 线（物理 0，session-editing 占用解除后，占用守卫自动放行）：M1-A100 → M3（research state 机制）→ M2。
3. 双卡数据齐备后：M4 矩阵与 replay 汇总。

每阶段输出新目录；失败、OOM、缺测保留记录不外推。A100/V100 不混合执行 SP；SP 与 LP 的 GPU 验证维持原协议门槛。

## 命令骨架（执行时按冻结配置补全参数）

```bash
export PYTHONPATH=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# M1/M5 V100（预算决策点 1 后）
.venv-v100/bin/python -m ditserve profile --model wan --cache none      --attention cache-dit-native --height 480 --width 832 --frames 17 --steps 50 --seed 0 --warmup 1 --repeats 1 --output results/motivation/m1-v100-none.json  --artifact-dir results/motivation/m1-v100-none
.venv-v100/bin/python -m ditserve profile --model wan --cache cache-dit --attention cache-dit-native --height 480 --width 832 --frames 17 --steps 50 --seed 0 --warmup 1 --repeats 1 --output results/motivation/m1-v100-c008.json --artifact-dir results/motivation/m1-v100-c008

# M1 A100（装入 benchmark 余量则先行）
.venv-previs/bin/python -m ditserve profile --model wan --cache none      --attention flash --height 480 --width 832 --frames 17 --steps 50 --seed 0 --warmup 1 --repeats 1 --output results/motivation/m1-a100-none.json  --artifact-dir results/motivation/m1-a100-none
.venv-previs/bin/python -m ditserve profile --model wan --cache cache-dit --attention flash --height 480 --width 832 --frames 17 --steps 50 --seed 0 --warmup 1 --repeats 1 --output results/motivation/m1-a100-c008.json --artifact-dir results/motivation/m1-a100-c008

# M3（research 机制桶）
export DIT_RESEARCH_RUN=results/research-pilot/run04
.venv-previs/bin/python -m ditserve research mechanism --devices 0 --kind state --output "$DIT_RESEARCH_RUN"

# M2（预算决策点 1 后；profile/trace/run/evaluate/report 全流程，新目录）
.venv-previs/bin/python -m ditserve benchmark run --trace <frozen-trace>/trace.json --backend wan --device cuda:0 --attention flash --text-prefetch 2 --policies fcfs,edf,srtf --output results/motivation/m2-<load> --budget-path /home/zyd/code/dit/results/video-benchmark/gpu-budget.json
```

## 2026-10-13 M1/M5-V100 实测记录

### M1 完成（双卡 8 组，全部 exit 0；耗时记入 results/motivation/gpu-seconds.jsonl）

480×832、17 帧、50 步、CFG=5、seed 0、cache-dit-native attention、FP16 DiT、CPU FP32 文本编码、FP32 VAE；缓存臂 DBCache Fn=8/Bn=0/warmup=8/threshold=0.08/max_continuous=3。

| 运行 | E2E (s) | 编码 (s) | 峰值 allocated (GiB) | 单步 min/med/max (ms) |
| --- | ---: | ---: | ---: | --- |
| a100-none-object | 164.60 | 14.58 | 11.06 | 2957/2967/2995 |
| a100-c008-object | 127.86 | 14.65 | 11.16 | 790/2965/3015 |
| a100-none-water | 163.94 | 14.03 | 11.06 | 2955/2965/3021 |
| a100-c008-water | 124.88 | 16.12 | 11.16 | 789/2961/3009 |
| v100-none-object | 281.23 | 14.46 | 10.48 | 5200/5214/5221 |
| v100-c008-object | 219.02 | 16.91 | 10.58 | 1389/5212/5234 |
| v100-none-water | 283.62 | 16.19 | 10.48 | 5221/5226/5234 |
| v100-c008-water | 209.46 | 15.17 | 10.58 | 1390/5204/5231 |

观测：
- 单步成本双峰确认：compute 步 A100 2965ms / V100 5214ms；reuse 步 A100 790ms / V100 1389ms；双卡 compute:reuse 比均约 3.75:1。
- **缓存复用日程由 (prompt, 缓存配置) 确定且与硬件无关**：object 两卡缓存步同为 [11,14,…,44] 共 17 步，water 同为 [9,11,13,…,43] 共 19 步——支持"缓存感知的确定性代价模型"命题；跨卡相位比在此对上恒定 1.76×，代价模型跨卡迁移只需单一标量（对本硬件对的描述性结论，不外推其他 GPU 组合）。
- 缓存臂 E2E：A100 -22.3%/-23.8%，V100 -22.1%/-26.2%。CPU 编码 14-17s 为显著串行份额。
- 汇总：`results/motivation/m1-summary.json`；提取器 `results/motivation/m1_extract.py`。

### M5-V100（微基准，非 worker 实测）

V100 PCIe 拷贝带宽（固定 8 线程、钉页内存、5 次取中位）：D2H 12.2 GiB/s、H2D 11.4 GiB/s（`results/motivation/m5-v100-pcie.json`）。2 GiB 请求态换出约 170ms，相对 V100 单步 5.2s 占比 ~3%——步粒度抢占的传输成本在 V100 上可忽略。A100 侧真实 keep/hot/cold 切换成本待 run05 mechanism 证据；A100 PCIe 微基准在 run05 结束后补测（避免触发研究占用守卫）。

### 执行偏差记录

- `ditserve profile` 的 `--attention` 仅接受 `default/cache-dit-native`（flash 是 benchmark worker 参数）；M1 双卡统一 cache-dit-native，与本文档命令骨架中的写法不同，以实际执行为准。
- run04 被 probe 拒止：隔离观测含此前争用记录，协议要求保留污染证据并启用新运行目录。M3 改用 run05（preflight CPU 验证已过 → probe --use-reserve → calibrate → mechanism state → report），沿用同一累计账本；run04 保持冻结。
- M1 首次 V100 尝试因离线权重解析失败（4 组 exit 2，各约 5-6s），存根移入 `results/motivation/attempt-failed-modelpath/`；修正为 `--model-id` 直指固定快照后重跑成功。

## 2026-10-13 run05（M3/M5-A100）结论记录

run05 全链路 ALL_STAGES_OK（probe 16min / calibrate 19min / mechanism 63min / report），账本累计 10,004.1 / 28,800 GPU·s（本轮 +8,098，其中 calibrate 期间一次外部争用按协议熔断并保留证据）。报告：`results/research-pilot/run05/REPORT.md`。

**M3（内存/准入）**：`memory_admission = insufficient_evidence`——A100 40GB 上两个 Wan-1.3B 请求（峰值约 11 GiB）从未触发 keep 容量拒绝，准入机制无触发条件。诚实负向观察：**1.3B 小模型在 40GB 卡上不构成容量压力**，状态管理收益需要更大模型/更长视频/更高并发才能显现；这本身成为 motivation 的边界条件证据。

**M3 交叉（抢占动作收益）**：`state_crossover = not_observed`——t008-hot-t008 相对 keep 的反序配对收益仅 0.03%~0.30%，低于 5% 门槛。A100 上 hot 换出在容量富余时没有收益，与准入观察一致。

**M5-A100（真实迁移成本，mechanism 实测）**：第 16 步 hot snapshot 的状态盘点为 198.4 MB（特征 175.7 MB + 必要存储 22.8 MB + RNG 5 KB），切换耗时 **16.4ms**；对照单步 2,965ms，迁移成本约占 0.55%。相位耗时：denoise 31.56s / decode 2.74s / prepare 11.3ms。质量门 `quality_qualified=true`，入选候选 `t008-hot-t008`。

**M5 双卡带宽**：A100 D2H 12.3 / H2D 11.5 GiB/s，V100 D2H 12.2 / H2D 11.4 GiB/s（同 PCIe 3.0 x16 链路，`m5-a100-pcie.json` / `m5-v100-pcie.json`）。两卡迁移成本均带宽约束且基本一致；相对各自单步成本（A100 2.97s / V100 5.21s）均可忽略 → **步粒度抢占在双卡上都便宜；抢占的价值取决于容量压力与优先级反转，而非迁移开销**。

**SP**：`sp_path = insufficient_evidence`（维持需 4 张同型卡的原判）。

## 2026-10-13 M2-replay 与 M4 记录

### M2-replay（模拟，非在线证据）

C++ 模拟器灌入 M1 实测标定成本（`cost-{a100,v100}-measured.json`，provenance 按核心守卫记为 synthetic、标定血缘在 description）；另有 8× 状态 what-if 臂（`cost-a100-pressure.json`，显式标注为投影非测量）。64 请求、混合帧长 17/49/81 帧（>17 帧为线性帧数投影）、50 步、threshold 0.08、SLO 2.5×、batch=1、单卡。归档：`results/motivation/m2-replay/archive/`，汇总 `m2-replay-summary.json`。

实测状态臂（a100，SLO 达成率）：

| 负载 | batch_only | cache_only | naive | edf | path_aware | state_aware |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.844 | 0.984 | 0.984 | 1.000 | 1.000 | 1.000 |
| 0.75 | 0.734 | 0.938 | 0.938 | 1.000 | 1.000 | 1.000 |
| 1.00 | 0.594 | 0.859 | 0.859 | 0.969 | 0.969 | 0.922 |
| 1.25 | 0.469 | 0.781 | 0.781 | 0.938 | 0.938 | 0.891 |

- 混合帧长下非抢占策略（naive/cache_only）随负载崩塌（ρ=1.25 时 0.781），batch_only 最差（0.469）→ HoL 与长尾效应确认（模拟证据）。
- 抢占策略（edf/path_aware）以 11-19 次抢占维持 0.94-1.0；**state_aware 在过载时 392 次抢占 + 129 次换出反降至 0.891（低于 edf 的 0.938）——抢占本身有代价，无脑状态管理会发生抖动**，这正是"缓存状态感知的抢占动作选择"的动机空缺。V100 臂同构且 state_aware 过载退化更明显（0.859 vs edf 0.938）。

8× 状态压力臂（a100p，投影）：所有持状态策略 SLO 0.297（19 完成 / **45 准入拒绝**），batch_only 不持状态反而 0.594-0.844 但靠超时消化。state_aware 仅 1 次抢占、无法缓解拒绝 → **容量压力下瓶颈从排队转移到准入，现有策略无一能回收状态换取准入**——精确回收/容量感知准入的动机空缺。run05 已证明 1.3B@40GB 无此压力，本臂回答"压力从何而来"。

### M4 相位×GPU 表（实测 + 标注残差）

| 相位 | A100 | V100 | V100/A100 |
| --- | ---: | ---: | ---: |
| 文本编码（CPU FP32） | 14.0-16.9 s | 14.5-16.9 s | ≈1.0（同 CPU） |
| compute 步（CFG 对） | 2.965 s | 5.214 s | 1.76 |
| reuse 步（CFG 对） | 0.790 s | 1.389 s | 1.76 |
| 解码（e2e 残差估计，含未拆分开销） | ~1.7 s | ~6.0 s | ~3.5 |
| E2E（cache off） | 164.6 s | 281.2 s | 1.71 |

去噪相位跨卡比恒定 1.76×（compute 与 reuse 同比例）；缓存复用日程硬件无关（M1）。对本硬件对，代价模型跨卡迁移只需单一标量；该结论不外推其他 GPU 组合。放置策略 replay 不再单列：恒定相位比下异构放置退化为速度加权，无额外结构问题需要 solve。
