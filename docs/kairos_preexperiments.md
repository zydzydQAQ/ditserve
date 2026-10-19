# Kairos 2/4 卡预实验汇总（2026-10-18 至 10-19）

## 实验环境

- 硬件：Kairos 集群（`jjjkairos`），A100-PCIE-40GB，无 NVLink，拓扑为 PHB：所有卡经 CPU host bridge 互连。驱动 570.133.20，低于清单要求的 570.181。2026-10-19 用 1×8 作业整机确认（gpu03，`acceptance/nvlink-20261019t1105z/interconnect.txt`）：8 卡两两之间都是 PHB，`nvidia-smi nvlink --status` 对每张卡报告 "all links are inActive"；P2P 读写可用但走 PCIe；PCI 地址为同一总线上连续的 `00:06.0`–`00:0D.0`，是虚拟机直通布局。
- 作业规模：1×1、1×2、1×4 容器作业。每张卡配 8 个 CPU、64 GiB 内存上限、32 GB `/dev/shm`；`/shared` 为 NFS 4.2。
- 镜像：在 `ditserve:rewrite-runtime`（`3a5e738`）上增量替换 `/opt/ditserve`，venv 不变。使用过的标签：
  - `rewrite-runtime`：K0–K2 首轮；
  - `8a100-1c6282a`：K0–K2 复测，以及 K3–K5；
  - `8a100-2edc0a8`：K6 中的 sp4x，以及 K8 的 2 卡；
  - `8a100-face7c1`：K6 中的 sp2r，以及 K7、K8 的 4 卡；
  - `8a100-224099f`：K11。
  - 2026-10-19 起只保留 `8a100-224099f` 一个镜像（其余标签已删除，可按提交号用 `.local/ditserve-kairos-rewrite-runtime/tools/Dockerfile.kairos-update` 重建）。代码改为按提交发布到 `/shared/code/DITSERVE/builds/<commit>/`，作业用 `DITSERVE_BUILD=<commit>` 选择版本。K11′ 用的是 `f2e89d4`。
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
| K9b 模拟器校准（修复后） | 加入争用与固定开销后能否复现 K8 | 在 f2e89d4 上重放 K8。规划用 K8 当时的 probe 画像（`planner_profile`），实际耗时用拟合画像加 `concurrent_gbps {"2": 3.9}` 和 `quantum_overhead_ms` 12（两者都从 K8 工作簿实测得到），每张卡一个编码器。4 卡每步时间相对真机：SP1 4.17/4.25 s（−2.0%）、并发 SP2 3.65/3.69 s（−1.1%，不加争用时 −8.4%）、SP4 2.78/2.82 s（−1.4%）、step_elastic 混合 3.92/3.98 s（−1.5%）。按时数：static_sp1 92/94、static_sp2 59/61、static_sp4 32/32、step_elastic 68/68、block_elastic 100/100。2 卡 SP2 每步 3.39 s 与真机相同，加入争用后不变（2 卡上只可能有一个 SP2 组）。没对上的是 4 卡 static_sp2 的视频 p95（330 s 对 417 s） | `results/kairos-k9b-replay/f2e89d4`（本地） |
| K10 负载扫描（模拟） | 弹性 SP 在负载下的价值 | 见"主要发现" 2、3 | 本地 `simulate`，3–10 个 seed |
| K10b 负载扫描（修复后） | 修复漏洞并校准成本后 K10 的结论是否还成立 | 4 卡，10 个 seed × 1/1.5/2× 负载，f2e89d4。原设置（K10 的画像，1 个编码器）下 2× 负载视频按时率：480p widest 0.915、free 0.957、只用 SP1 0.954；720p 0.834/0.912/0.909，与修复前基本一致。校准设置（K9b 的画像，每卡一个编码器）下：480p 0.891/0.931/0.929，720p 0.776/0.881/0.878；step_elastic(free) 视频 0.947/0.912，但图像按时率只有 0.43/0.18，block_elastic 的图像始终为 1.00。720p 视频 p95：widest 2710 s、free 2458 s、只用 SP1 2688 s、step_elastic 2069 s | `results/kairos-k10-sweep/f2e89d4`（本地） |
| K11 迟到判断（真机） | 验证 K10 的预测 | 失败，没有结论：4 卡、2× 负载、seed 108，比较 block_elastic 的 widest、free 与 SP1 三种设置。r39 处报 `layout can change only at a healthy step boundary`，整个单元终止，200 个请求中 111 个失败。原因是规划器的漏洞：处在 step 边界的视频会直接回到原来的组，但原组的卡可能正被另一个视频占在 step 中途，而 export 下每个 worker 同一时刻只允许一个请求停在 step 中途。已修复，`cluster_plan` 末尾新增契约检查，模拟器和执行器都会执行它。见 K11′ | `kairos-cluster-sp-4gpu-fit-custom-l2-s108-matrix/feas4-20261019t000743z` |
| K11′ 迟到判断（真机，修复后） | 修复后在真机上验证 free，并检验 K9b 的成本模型（样本外） | 与 K11 相同的 trace（4 卡、2× 负载、seed 108，200 个请求），build f2e89d4，`CLUSTER_PROFILE=fit2`。两个 1×4 作业，800 个请求全部完成，没有失败，也没有契约错误。按时（总/视频/图像）：block_elastic widest 183/30/153，free 189/36/153，只用 SP1 189/35/154；step_elastic(free) 104/36/68。free 不低于只用 SP1，但只多 1 个视频（单 seed，在随机波动范围内）；widest 比两者少 5–6 个视频。事先记录的模拟预测为 190/36、193/39、192/38、94/39：排序一致，但每个单元都多估了约 3 个按时视频。每步时间（视频 action 时间 / 32 步，中位数）模拟比真机短 2.1–2.5%，单元用时短 1–4% | `kairos-cluster-sp-4gpu-fit2-custom-l2-s108-matrix/k11b-20261019t033804z`、`k11s-20261019t033804z`；预测见 `results/kairos-k11p-predictions/f2e89d4`（本地） |

### Wan2.2-TI2V-5B 快速预实验（2026-10-19，build e2de3e4）

模型目录 `wan2.2-ti2v-5b-te21`：5B 的 DiT、VAE 和 scheduler（revision `b8fff73`，每个文件都按 HF 官方 SHA256 校验），加 Wan2.1 的 UMT5 文本编码器和 tokenizer。当时 5B 的文本编码器还在传输；两者架构相同，tokenizer 的两个大文件 SHA256 也相同，只是 1.3B 的编码器以 FP32 存储。DiT 用 FP16，Flash attention，关闭 cache。

| 编号 | 问题 | 结论 | 证据 |
| --- | --- | --- | --- |
| K12 5B 数值验收 | 逐 token 时间步的步进路径能否在 GPU 上复现官方 pipeline | 1 张卡：64×64×5、3 步，经热/冷快照往返后，latent 和视频都与官方 `WanPipeline` 逐位一致；block worker 与 Sana 同卡混合回放 4/4 完成。5B 自己的文本编码器到位后，用完整的 `wan2.2-ti2v-5b`（22 个文件全部按官方 SHA256 校验，清单在 `models/.provisioning/wan22-ti2v-5b.json`）重做，同样通过 | `results/target-acceptance-5b-te21/q5b-20261019t0943z-gpu0`、`results/target-acceptance-5b/e05b-20261019t1038z-gpu0` |
| K13 5B SP4 等价性 | 逐 token 时间步按序列切分后，SP 是否成立 | 512×896×17、8 步：SP4 组内 13-block quanta 与整步 SP4 逐位一致。相对 SP1：latent max_abs 1.51、rmse 0.0508，视频 rmse 0.0105（约 39.6 dB），1.3B 的 K6 为 0.511/0.0235/49.2 dB。最初用 480×832×17 时失败，原因是 5B 每帧只有 390 个 token，不能被 4 整除 | `kairos-sp4-equivalence-5b-te21/q5b2-20261019t0952z` |
| K14 5B SP 扩展曲线 | 5B 的 SP 加速，以及成本模型能否迁移 | 见下表。704×1280×121（官方 720p、5 秒）SP1 每步 8.26 s，SP4 为 1.71×；同为 720p，1.3B 是 2.09×。直接用 K7 在 1.3B 上拟合的链路参数（η=1、4.81/2.72 GB/s），只换成 5B 的解析通信量，16 个形状×度数的每步时间误差都在 3.7% 以内 | `kairos-sp-scaling-4gpu-5b-te21/q5b2-20261019t0952z` |

| K15 8 卡探测与 5B SP8 | 整机 all-to-all 带宽与 SP8 扩展 | 1×8（gpu 节点，build e2de3e4）。collectives：pair 单独 3.80–3.96 GB/s，4 对并发 2.17–2.32；quad 与 octet 不论并发还是单独都是 2.26–2.43 GB/s——**host bridge 聚合约 19 GB/s 是上限**，≥4 卡同时通信时每卡只有约 2.4 GB/s。SP 扩展（5B，8 种形状，SP1/2/4/8）：704×1280×121 每步 8.29/6.09/4.72/2.70 s，即 SP2 1.36×、SP4 1.76×、**SP8 3.07×**；512×896×81 为 2.09×（SP8）。SP8 比 SP4 快只因每卡通信量更少（7/64 对 3/16），带宽相同。拟合 `links.8`：4.81/2.72/2.72 GB/s（SP2/4/8 等效带宽）、延迟 0.18/0.28/0.48 ms，所有度数在 6 个非 17 帧形状上的最大误差：SP2 3.3%、SP4 4.1%、SP8 2.6%；4 对并发时 `concurrent_gbps` 约 2.8 | `acceptance/p12-20261019t1210z/collectives.json`、`kairos-sp-scaling-8gpu-5b/p12-20261019t1210z`；本地 `results/kairos-8gpu-probe` |

| K16 CFG 并行（Stage A 微基准） | 把两个 CFG 分支分到两张卡上（`FixedLayout.cfg=2`），每步只交换一次预测，能否无损且接近 2× | 1×2（build bac8ea8，5B，Flash，cache off，block 执行器）。逐位一致：512×896×17、8 步，整步 CFG2 与 13-block quanta 的 CFG2，latent 和视频都与单卡 SP1 `exact`（两个 rank）。每步时间（4 步，去首步中位数）：704×1280×121 SP1 8.26 s、SP2 6.04 s（1.37×）、**CFG2 4.11 s（2.01×）**；512×896×81 SP1 1.97、SP2 1.74（1.13×）、CFG2 0.99（1.99×）。成本模型（compute/2 + 一次 latent 交换）预测 4.13/0.99 s，误差 ≤0.5%。1×4（gpu06）：704×1280×121 **CFG2×SP2 3.06 s（2.70×）**，预测 3.14（2.6%）；512×896×81 0.89 s（2.22×）。同一作业里 block 执行器上的 SP4 为 3.38 s（2.44×），而 K14/K15 用整步 worker 测的 SP4 是 4.72–4.82 s（gpu05、gpu06），SP2 两种 worker 一致（6.04 对 6.09）——差异待同节点 A/B 核实（下一轮 P5 已加 `sp2-step`/`sp4-step` 对照）。Gate A 的 CFG2 部分通过（门槛 ≥1.85×） | `kairos-cfg2gpu-5b/p5a-20261019t1250z`、`kairos-cfg4gpu-5b/p5b-20261019t1250z`；本地 `results/kairos-5b-cfg/` |

K15 每步耗时（5B，8 卡，整步，去掉首步后的中位数）：

| 形状 | token 数 | SP1 | SP2 | SP4 | SP8 |
| --- | --- | --- | --- | --- | --- |
| 512×896×33 | 4.0k | 0.77 s | 0.73 s（1.05×） | 0.65 s（1.18×） | 0.44 s（1.74×） |
| 512×896×81 | 9.4k | 1.97 s | 1.75 s（1.13×） | 1.53 s（1.29×） | 0.94 s（2.09×） |
| 512×896×121 | 13.9k | 3.27 s | 2.70 s（1.21×） | 2.31 s（1.42×） | 1.37 s（2.39×） |
| 704×1280×81 | 18.5k | 4.86 s | 3.77 s（1.29×） | 3.12 s（1.56×） | 1.83 s（2.66×） |
| 704×1280×121 | 27.3k | 8.29 s | 6.09 s（1.36×） | 4.72 s（1.76×） | 2.70 s（3.07×） |

K14 每步耗时（整步，去掉首步后的中位数）：

| 形状 | token 数 | SP1 | SP2（加速） | SP4（加速） |
| --- | --- | --- | --- | --- |
| 512×896×17 | 2.2k | 0.41 s | 0.43 s（0.95×） | 0.41 s（1.02×） |
| 512×896×33 | 4.0k | 0.77 s | 0.73 s（1.05×） | 0.67 s（1.14×） |
| 512×896×81 | 9.4k | 1.97 s | 1.76 s（1.12×） | 1.57 s（1.25×） |
| 512×896×121 | 13.9k | 3.26 s | 2.71 s（1.21×） | 2.36 s（1.38×） |
| 704×1280×17 | 4.4k | 0.83 s | 0.79 s（1.05×） | 0.73 s（1.14×） |
| 704×1280×33 | 7.9k | 1.60 s | 1.45 s（1.10×） | 1.32 s（1.21×） |
| 704×1280×81 | 18.5k | 4.85 s | 3.77 s（1.29×） | 3.19 s（1.52×） |
| 704×1280×121 | 27.3k | 8.26 s | 6.07 s（1.36×） | 4.82 s（1.71×） |

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

1. **PCIe 上 Ulysses SP 的效率很低，而且随形状变化。** 计算量随 token 数平方增长，all-to-all 只线性增长。所以 1.3B 模型在 17 帧时 SP 反而更慢，到 720p×81 帧 SP4 也只有 2.09×（效率 0.52）。用"计算/d + 通信量/带宽"的模型，在 η=1 时就能把 6 个形状拟合到 5% 以内，说明计算部分基本线性扩展，损失几乎全在通信上。原先推断更大的模型（5B、14B）hidden 更宽、计算与通信之比更高，SP 效率应该更好。K14 不支持这个推断：Wan2.2-5B 的 VAE 空间压缩是 16 倍，同一分辨率下 token 数只有 1.3B 的 1/4，注意力的计算量随之大减。所以 720p 时 5B 的 SP4 只有 1.71×，低于 1.3B 的 2.09×；按 token 数相近的形状比较（1.3B 480×832×81 为 32.8k，5B 704×1280×121 为 27.3k），5B 略好（1.71× 对 1.46×）。同一个链路模型无需重新拟合就能预测 5B 的每步时间（误差 ≤3.7%），说明损失确实全在通信。
2. **固定 SP（static_spK）在这类机器上吃亏。** 每个视频占的卡多了 K 倍，速度只快 1.2–1.5 倍，并发减半，导致排队。K8 中 static_sp2/4 的按时数远低于 static_sp1。
3. **弹性 SP 的问题出在迟到判断，不在 SP 本身。**
   - feasible-first EDF 判断迟到时假设请求能拿到最宽的组。负载下很少有整组空闲，于是只有空闲整机才救得了的视频保留了 EDF 优先级，以 SP1 开跑后迟到，拖累了其后本可按时完成的请求。
   - 在校准后的模拟器中（4 卡，10 个 seed），2× 负载下默认的 block_elastic 视频按时率为 0.914（480p）和 0.818（720p），比只用 SP1 的 0.954/0.909 还低。10 个 seed 中分别有 4 个和 6 个更差。
   - `feasibility_degree="free"`（224099f）改用规划时刻空闲的最宽组判断迟到，按时率变为 0.957/0.912，不低于只用 SP1。视频 p95 也低于只用 SP1：480p 为 356 s 对 369 s，720p 为 1654 s 对 2092 s。
   - 低负载时弹性 SP 的收益主要在尾延迟：720p 1× 负载下 p95 为 620 s 对 718 s。
   - step_elastic 改用 free 后也同样改善，基线应当一并启用。
   - 现已把 free 设为两种弹性策略的默认值，并且不再把留给图像的卡和被其他视频占着的卡算作空闲。以上数字是改动前用显式 free 得到的。在修复后的版本上重跑（K10b），原设置下结论不变。
   - 但成本校准后，block_elastic(free) 对只用 SP1 的视频优势缩小到 0.002–0.003，收益主要剩下 720p 的视频长尾（p95 2458 s 对 2688 s）。step_elastic(free) 的视频按时率反而最高，代价是图像大面积超时。在这类 PCIe 机器上，block 级抢占的价值主要在保护短的图像请求；弹性 SP 对 1.3B 视频的按时率帮助有限。5B 的计算与通信之比更高，需要用实测成本重新评估。
4. **block 抢占的开销很小。** worker 侧每个 quantum 只多 1–4 ms；端到端约 10 ms，按 300 ms 的 quantum 计约 3%。这不是弹性 SP 收益有限的原因。
5. **SP 组之间会争用 host bridge，而且整机有一个约 19 GB/s 的聚合上限。** 两个 SP2 组并发时，有效带宽从 4.8 降到约 3.8 GB/s，四个 SP2 组并发时约 2.8（K15）；quad 和 octet 不论是否并发都在每卡 2.4 GB/s 的上限上。f2e89d4 起成本画像可以用 `links.<d>.concurrent_gbps` 描述这种争用，用 `quantum_overhead_ms` 描述每个 quantum 的固定开销，规划器和模拟器都会计入（见 K9b）。

## 局限与待办

- 真机只跑了一个 seed（K11/K11′ 的 seed 108 是按修复前的模拟预测挑的）；多 seed 结论只来自模拟器。K11′ 中 free 与只用 SP1 只差 1 个视频，单 seed 不足以区分两者。
- 模拟器在 K11′ 的每个单元都多估了约 3 个按时视频（共 46 个），按时率的绝对值要按这个偏差来看待，排序可以参考。
- K9、K10 是在修复 K11 漏洞之前模拟的，已在修复后的版本上重跑（K9b、K10b）。
- 720p 的解码时间按 token 数外推，没有在 serving 路径上实测。
- 争用带宽 3.9 GB/s 和固定开销 12 ms 都是从 K8 的 4 卡数据拟合出来的，属于样本内参数；K11′ 是第一个样本外检验。模拟器按发出时刻的并发数给 SP quantum 计价，之后并发数变化时不重新计时；4 卡 static_sp2 的视频长尾仍被低估（p95 330 s 对 417 s）。
- Wan2.2-5B 的 Ulysses SP 要求 token 数能被 SP 度整除：480×832 每帧 390 个 token，不能用于 SP4/SP8。可用 512×896（448）或 704×1280（880），否则需要在切分时补齐 token。
- 5B 在 FP16 下 SP4 相对 SP1 的偏差（视频约 39.6 dB）明显大于 1.3B（49.2 dB）。官方通常用 BF16；SP 输出质量要用语义或盲评确认，不能只看数值。
- K13、K14 用的是 Wan2.1 的文本编码器，它只影响 CPU 上的提示词编码，不影响 DiT 的 SP 结果；K12 已用完整 5B 模型重做并通过。5B 的成本画像（解码、prepare）还要补测。
- 尚未完成：图像批处理、slack-based victim 选择、TetriServe 式 round-based 基线；用 5B 成本画像做集群回放与负载扫描；8 卡验收与主实验。
