# Kairos 2/4 卡预实验汇总（2026-10-18 至 10-19）

## 实验环境

- 硬件：Kairos 集群（`jjjkairos`），A100-PCIE-40GB，无 NVLink，拓扑为 PHB：所有卡经 CPU host bridge 互连。驱动 570.133.20，低于清单要求的 570.181。
- 作业规模：1×1、1×2、1×4 容器作业。每张卡配 8 个 CPU、64 GiB 内存上限、32 GB `/dev/shm`；`/shared` 为 NFS 4.2。
- 镜像：在 `ditserve:rewrite-runtime`（`3a5e738`）上增量替换 `/opt/ditserve`，venv 不变。使用过的标签：
  - `rewrite-runtime`：K0–K2 首轮；
  - `8a100-1c6282a`：K0–K2 复测，以及 K3–K5；
  - `8a100-2edc0a8`：K6 中的 sp4x，以及 K8 的 2 卡；
  - `8a100-face7c1`：K6 中的 sp2r，以及 K7、K8 的 4 卡；
  - `8a100-224099f`：K11。
- 脚本：`.local/ditserve-kairos-rewrite-runtime/scripts/`。
- 结果：`/shared/output/DITSERVE/results/<experiment>/<run>/results.xlsx`。作业日志、生成的配置和预算账本在 `acceptance/<run>/`。

所有真机结果都是单 seed 的描述性数据。多 seed 的结论只来自校准后的模拟器。

## 结果一览

| 编号 | 问题 | 结论 | 证据 |
| --- | --- | --- | --- |
| K0 平台探测 | 容器内能否用 nvidia-smi 的 PID 判断 GPU 独占 | `pid_visible`：容器内 PID 与 nvidia-smi 一致，不需要 hostPID | `acceptance/probe-20261018t145616z`（rewrite-runtime）、`probe-20261018t162113z`（1c6282a） |
| K1 CPU 验收 | 模拟器、mock study、工作簿校验 | 24/24 通过，`verified`。结果归档以只读句柄对 NFS 上的账本加 `LOCK_EX`，报 EBADF；已修复（417a92a） | `acceptance/cpu-20261018t145219z`（EBADF）、`cpu-20261018t145616z`、`cpu-20261018t162113z` |
| K2 E0 | 每张卡的数值与混合回放验收 | 1、2、4 卡作业中每张卡都 PASSED：Wan 状态往返逐位一致，Wan/Sana 混合回放完成 | `results/target-acceptance/*` |
| K3 collectives | 对齐 pair/quad 的 all-to-all 每 rank 带宽 | pair 单独测 3.97 GB/s，两对并发各 2.32 GB/s；quad 2.42 GB/s | `acceptance/pre2-*`、`pre4-*` |
| K4 cluster（SP1 固定） | 多卡真实执行下 block 抢占是否成立 | 2 卡：static_sp1 31/50 按时，block_elastic 50/50，图像 p95 3.02 → 0.59 s。4 卡：92/100 对 100/100，1.41 → 0.47 s。两种策略下视频全部按时 | `kairos-cluster-sp1-{2,4}gpu/pre*` |
| K5 成本画像 | 目标机上的 E2 成本 | 与本机 A100 相差 0.7% 以内；UMT5 在 CPU 上编码 16–19 s，Sana 309 ms | `kairos-e2-cost-profile/e2-20261018t162809z` |
| K6 SP 等价性 | 整步 SP 以及 SP 组内 block 执行与 SP1 的偏差 | 480×832×17、8 步。整步 SP2/SP4 相对 SP1：latent max_abs 0.374/0.511，rmse 0.0166/0.0235，视频 PSNR 52.6/49.2 dB，多次运行结果相同。SP 组内 13-block quanta 与整步 SP 逐位一致（SP2、SP4）。每个 quantum 的额外开销为 +0.7/+3.7 ms，每步 +0.5%/+3.1%（不计首步预热） | `kairos-sp{2,4}-equivalence/sp2r-*`、`sp4x-*` |
| K7 SP 扩展曲线 | SP 的实际加速与代价模型拟合 | 见下表。在 SP 效率不超过 1 的约束下拟合规划器模型：η=1.0、gbps[2]=4.81、gbps[4]=2.72，留一形状交叉验证每步误差最大 4.7% | `kairos-sp-scaling-{2,4}gpu/scale*` |
| K8 cluster（真实 SP>1） | 真实后端上 SP 组内 block 执行与 step 边界换组 | 2 卡：block_elastic 50/50，static_sp1 32、static_sp2 20、step_elastic 30。4 卡：block_elastic 100/100（12 个视频用过 SP2、5 个用过 SP4，换组 29 次、1.0 GiB），static_sp1 94、static_sp2 61、static_sp4 32、step_elastic 68。每次换组约 16 MiB，导出 70–150 ms | `kairos-cluster-sp-2gpu/sp2x-*`、`kairos-cluster-sp-4gpu/sp4c-*` |
| K9 模拟器校准 | 用 K7 的参数复现 K8 | 按时数：4 卡 static_sp2 原 profile 36，校准后 57，真机 61；static_sp4 30/34/32；step_elastic 62/68/68；block_elastic 三者都是 100。剩下的偏差来自两个 SP2 组并发争用（实际约 3.8 GB/s）和每个 quantum 约 10 ms 的端到端开销 | 本地 `simulate`，配置来自 K8 |
| K10 负载扫描（模拟） | 弹性 SP 在负载下的价值 | 见"主要发现" 2、3 | 本地 `simulate`，3–10 个 seed |
| K11 迟到判断（真机） | 验证 K10 的预测 | 失败，没有结论：4 卡、2× 负载、seed 108，比较 block_elastic 的 widest、free 与 SP1 三种设置。r39 处报 `layout can change only at a healthy step boundary`，整个单元终止，200 个请求中 111 个失败。原因是规划器的漏洞：处在 step 边界的视频会直接回到原来的组，但原组的卡可能正被另一个视频占在 step 中途，而 export 下每个 worker 同一时刻只允许一个请求停在 step 中途。已修复，`cluster_plan` 末尾新增契约检查，模拟器和执行器都会执行它。待重跑 | `kairos-cluster-sp-4gpu-fit-custom-l2-s108-matrix/feas4-20261019t000743z` |

K7 每步耗时（整步，去掉首步后的中位数）：

| 形状 | token 数 | SP1 | SP2（加速） | SP4（加速） |
| --- | --- | --- | --- | --- |
| 480×832×17 | 7.8k | 0.58 s | 0.63 s（0.92×） | 0.63 s（0.93×） |
| 480×832×33 | 14k | 1.23 s | 1.22 s（1.01×） | 1.14 s（1.08×） |
| 480×832×81 | 32.8k | 4.15 s | 3.35 s（1.24×） | 2.84 s（1.46×） |
| 720×1280×17 | 18k | 1.72 s | 1.58 s（1.09×） | 1.42 s（1.21×） |
| 720×1280×33 | 32.4k | 4.14 s | 3.34 s（1.24×） | 2.81 s（1.47×） |
| 720×1280×81 | 75.6k | 16.8 s | 11.4 s（1.48×） | 8.05 s（2.09×） |

## 主要发现

1. **PCIe 上 Ulysses SP 的效率很低，而且随形状变化。** 计算量随 token 数平方增长，all-to-all 只线性增长。所以 1.3B 模型在 17 帧时 SP 反而更慢，到 720p×81 帧 SP4 也只有 2.09×（效率 0.52）。用"计算/d + 通信量/带宽"的模型，在 η=1 时就能把 6 个形状拟合到 5% 以内，说明计算部分基本线性扩展，损失几乎全在通信上。由此推断，更大的模型（5B、14B）hidden 更宽，计算与通信之比更高，效率应该更好，需要实测确认。
2. **固定 SP（static_spK）在这类机器上吃亏。** 每个视频占的卡多了 K 倍，速度只快 1.2–1.5 倍，并发减半，导致排队。K8 中 static_sp2/4 的按时数远低于 static_sp1。
3. **弹性 SP 的问题出在迟到判断，不在 SP 本身。**
   - feasible-first EDF 判断迟到时假设请求能拿到最宽的组。负载下很少有整组空闲，于是只有空闲整机才救得了的视频保留了 EDF 优先级，以 SP1 开跑后迟到，拖累了其后本可按时完成的请求。
   - 在校准后的模拟器中（4 卡，10 个 seed），2× 负载下默认的 block_elastic 视频按时率为 0.914（480p）和 0.818（720p），比只用 SP1 的 0.954/0.909 还低。10 个 seed 中分别有 4 个和 6 个更差。
   - `feasibility_degree="free"`（224099f）改用规划时刻空闲的最宽组判断迟到，按时率变为 0.957/0.912，不低于只用 SP1。视频 p95 也低于只用 SP1：480p 为 356 s 对 369 s，720p 为 1654 s 对 2092 s。
   - 低负载时弹性 SP 的收益主要在尾延迟：720p 1× 负载下 p95 为 620 s 对 718 s。
   - step_elastic 改用 free 后也同样改善，基线应当一并启用。
4. **block 抢占的开销很小。** worker 侧每个 quantum 只多 1–4 ms；端到端约 10 ms，按 300 ms 的 quantum 计约 3%。这不是弹性 SP 收益有限的原因。
5. **SP 组之间会争用 host bridge。** 两个 SP2 组并发时，有效带宽从 4.8 降到约 3.8 GB/s，与 K3 的探测结果一致。目前每个 SP 度只有一个带宽值，规划器和模拟器都还没有建模这种争用。

## 局限与待办

- 真机只跑了一个 seed（K11 的 seed 是按模拟预测挑的）；多 seed 结论只来自模拟器。
- K9、K10 是在修复 K11 漏洞之前模拟的，其中可能包含真实 SP worker 会拒绝的交错执行（新的契约测试在紧期限、2–3× 负载、free 下能复现），需要在修复后的版本上重跑。
- 720p 的解码时间按 token 数外推，没有在 serving 路径上实测。
- 模拟器没有计入 SP 组并发争用和每个 quantum 的固定开销。
- 尚未完成：图像批处理、slack-based victim 选择、TetriServe 式 round-based 基线；Wan2.2-5B 的下载与移植；8 卡验收与主实验。
