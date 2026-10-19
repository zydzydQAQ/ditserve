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
| K16b 分支交错与第二轮形态计时 | 把 cond/uncond 两个分支逐 block 交错到两个 CUDA stream 上（`execution="block_interleaved"`），能否无损地把 all-to-all 藏在另一分支的计算后面；SP4 的执行器差异；形态计时的第二个样本 | build bcf569d；p5c 1×2（gpu03），p5d 1×4（gpu06，与 p5b 同节点但只有 1 张卡相同）。(1) CFG2 与 CFG2-block 再次与 SP1 逐位一致。**交错形态 sp2i 不一致**（两作业 latent rmse 4.7/5.3，视频 rmse 0.48，输出无效）；CPU/Gloo 契约测试是单 stream，抓不到。原因是两个 stream 共用 caching allocator：主 stream 分配、cond 分支过了第 0 块就丢掉的初始 hidden state 被主 stream 的下一次分配复用时，侧 stream 还在读它。已在 `block.py` 给侧 stream 读的张量加 `record_stream`，待下一轮核实。(2) 704×1280×121 每步（p5d）：SP1 8.24，CFG2 4.13（2.00×），**CFG2×SP2 3.21（2.56×，预测 3.13）**，SP2 5.31（1.55×），SP2i 4.31（1.91×），SP4 4.02（2.05×），SP4i 3.38（2.43×）；512×896×81：CFG2 0.99（1.97×），CFG2×SP2 0.96（2.03×），SP2 1.47，SP2i 1.12，SP4 1.30，SP4i 1.08。p5c（gpu03）：SP2 6.12（1.35×），SP2i 4.99（1.65×），CFG2 4.19（1.97×）。(3) 整步 worker 与 block 执行器在同一作业里一致（SP2 5.30/5.31，SP4 4.01/4.02），K16 里 3.38 对 4.72–4.82 的差异来自**作业拿到的 GPU 组合**而不是执行器：同在 gpu06，p5b 的四张卡 SP4 3.38 s，p5d 的四张卡 4.02 s，gpu05（K14）4.82 s；SP2 在 gpu06 这组为 5.31，gpu03 为 6.12，K14/K15 为 6.07/6.09。按 K15 的链路模型反推，等效每 rank 带宽 SP2 为 5.1（gpu03）至 8.7 GB/s（p5d），SP4 为 4.0（p5d）、6.0（p5b）、2.7 GB/s（K14/K15）。nvidia-smi 里 28 对都是 PHB，看不出这种差别。(4) 交错藏起来的通信比例（(SP − SPi)/min(计算, 通信)）：SP2 0.57（gpu03）/0.84（gpu06），SP4 0.33；通信接近计算时两条链互相等待，`max(计算, 通信)` 的全重叠模型高估，成本模型改为按度数给 `sp_overlap`。Gate A：CFG2 ≥1.85× 通过；交错 SP4 2.43× 刚过 2.4× 门槛但尚未无损，而且在同一硬件上不如无损的 CFG2×SP2（3.21 s） | `kairos-cfg2gpu-5b/p5c-20261019t1335z`、`kairos-cfg4gpu-5b/p5d-20261019t1335z`；本地 `results/kairos-5b-cfg/p5c-2gpu`、`p5d-4gpu` |
| K16c 分支交错第三轮（allocator 修复后） | 修复 `record_stream` 后交错是否无损；形态计时第三个样本 | build c27bbf7，p5e 1×2、p5f 1×4，都在 gpu06（又一组 GPU）。(1) CFG2/CFG2-block 仍与 SP1 逐位一致。sp2i 对 sp1 的偏差从 latent rmse 4.7–5.3 降到 **0.063**（max_abs 1.38，视频 rmse 0.0141，约 37 dB），且两个作业的数值完全相同——这不再是竞争（竞争的结果每次不同），而是确定性的数值差异，量级与 K13/K18 里 SP 本身相对 SP1 的差异相同（GPU 上按 token 分片后 GEMM 的 M 维不同，SP2 自身就不与 SP1 逐位一致）。所以 sp2i 的 atol=0 门不该对 sp1，而该对同一布局的非交错 sp2；脚本已改为 sp2/sp2i、sp4/sp4i 成对比较（第四轮核实）。(2) 704×1280×121 每步（p5f）：SP1 8.28，CFG2 4.12（2.01×），CFG2×SP2 3.06（2.70×），SP2 6.22（1.33×），SP2i 4.97（1.67×），SP4 3.39（2.44×），**SP4i 2.73（3.03×）**；512×896×81：CFG2 0.99，CFG2×SP2 0.89（2.22×），SP2 1.74，SP2i 1.29，SP4 1.08，SP4i 0.87（2.27×）。p5e（1×2）：SP2 5.95，SP2i 5.06，CFG2 4.11。这组卡的 SP4 带宽与 p5b 相同（3.38/3.39），交错后 SP4i 2.73 s 快于 CFG2×SP2 3.06 s；在 p5d 那组慢卡上则相反（3.38 对 3.21）。藏起来的通信比例：SP2 0.49–0.60，SP4 0.50（p5d 组为 0.84/0.33）。Gate A 的交错速度门（SP4i ≥2.4×）在两组卡上都过（2.43×、3.03×），无损性待第四轮按正确的参照核实 | `kairos-cfg2gpu-5b/p5e-20261019t1345z`、`kairos-cfg4gpu-5b/p5f-20261019t1345z`；本地 `results/kairos-5b-cfg/p5e-2gpu`、`p5f-4gpu` |
| K16e 分支交错第四轮：与同布局的非交错 SP 逐位一致 | 把 sp2i/sp4i 的 atol=0 门改对同布局的 sp2/sp4 后，交错是否无损 | build c694b60（worker 代码同 c27bbf7），p5g 1×2、p5h 1×4。512×896×17、8 步：**sp2i 与 sp2、sp4i 与 sp4 的 latent 和视频都 `exact`**（两个作业一致）；交错与非交错对 sp1 的偏差完全相同（SP2：latent max_abs 1.378、rmse 0.0630，视频 rmse 0.0141；SP4：0.961/0.0396/0.0089），即第三轮看到的差异就是 SP 自身相对 SP1 的 GEMM 分片差异，交错本身不改变任何结果。至此 Gate A 的交错部分通过：无损（相对同度数的 SP），SP4i 2.43–3.03×。计时：p5g（1×2）704×1280×121 SP2 5.34，SP2i 4.35（1.90×），CFG2 4.12（2.00×）；p5h（1×4，与 p5d 同类的慢组）SP2 6.11，SP2i 4.96（1.65×），SP4 4.01，SP4i 3.35（2.44×），CFG2×SP2 3.21（2.55×）；按本作业拟合链路后各形态误差在 ±5% 内（中形状的 sp2/cfg2sp2 −9%，该作业 t-sp2 的 block 样本 1.90 s 明显偏离 step 样本 1.74 s） | `kairos-cfg2gpu-5b/p5g-20261019t1415z`、`kairos-cfg4gpu-5b/p5h-20261019t1415z`；本地 `results/kairos-5b-cfg/p5{g,h}-*` |
| K19b 基线与组件消融（模拟） | TetriServe 式轮次换组、SRTF/FCFS 排序、以及在形态弹性之上逐个关闭组件 | 同 K19 的设置，1×/1.5×，10 seed，build 661f92d。视频按时率/图像按时率/视频 p95：`round_elastic`（每 5 步才可换组）1× 0.937/0.832/761，1.5× 0.828/0.776/1422——比 step_elastic 差（换组太迟，图像又等 step 边界）；`order=fcfs` 图像 0.379/0.175（短请求排在长视频后面）；`srtf` 与 edf 相当（1× p95 454 对 463）。在 block_wide 上单独关闭：eager 抢占 ≈ lazy（1.5× p95 1002 对 1013）；`feasibility_degree=widest` 1.5× 视频 0.787 对 0.874（重现 K10b 的迟到判断问题）；`late_short_ms=0` 无差别；整段解码 ≈ 分段解码（8 卡 20/80 下图像总能找到空闲卡，分段解码的价值要在 4 卡或突发轨迹上看）。表格见 `results/kairos-p8-sim/baselines/summary.json` | 本地 `results/kairos-p8-sim/baselines` |
| K20 5B 服务对的显存（P9，真机 OOM 诊断） | p6a/p6b 为何在第一个 704×1280×121 解码时 OOM | 1×1，build c694b60，`kairos_mem5b.sh`：Sana 两张图后 allocated 6.57 GB、**reserved 10.16 GB**（3.6 GB 缓存未用；expandable_segments 下 8.55）。Wan 5B 加载后 11.96 GB（DiT FP16 10 GB + VAE FP32 2.7 GB）；512×896×81 step 中途 packet +0.45 GB，去噪峰值 13.7 GB；**整段解码峰值 22.1 GB**（+10 GB 瞬态），之后 allocator 保留 27 GB 不还；704×1280×121 两步去噪不再增加，8 个 latent 帧一段的解码峰值 **31.3 GB、reserved 36 GB**。所以 OOM 不是泄漏，是 FP32 VAE 解码的瞬态：720p 一段（8 帧）要 19 GB，而 Sana 常驻 10 GB、Wan 基线 12 GB；p6 里 1 帧一段仍需约 12 GB 瞬态（OOM 时进程 25.4 GB）。解码后 Wan 进程保留 36 GB 缓存，下一张图像也会 OOM。处理：worker 在每次解码/出图后 `empty_cache`（把 reserved 收回到 live），模板给两种 worker 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；1 帧一段的瞬态见 p9c | `kairos-mem-5b/p9a-20261019t1445z`、`kairos-mem-5b-expandable-segments-true/p9b-20261019t1445z`；本地 `results/kairos-5b-mem/p9{a,b}` |
| K20b 归还缓存后的重跑仍 OOM | `release_cache` + expandable segments 能否让 FP32 的 720p 解码与 Sana 同卡 | p6c/p6d（build b55cdb0，同 p6a/p6b 的设置）：仍在第一个 704×1280×121 的解码处 OOM，但进程已只有 29.9 GB、卡上剩 2.5 GB，差的就是那个 2.6 GB 的块（p6a 时剩 0.2 GB）。1 帧一段的解码（p9c）首段 +3 GB，整个分段解码的进程峰值仍是 32.2 GB，所以 FP32 VAE 的 720p 解码（约 +20 GB）与 Sana（常驻 6.6 GB、出图峰值 8.2 GB）在 40 GB 上放不下，缓存归还只把缺口从 2.4 GB 缩到 0.1–3 GB。出路：VAE 用 BF16（P10 度量偏差与峰值）或长类别降到 512×896（解码峰值 22 GB，K20） | 本地 `results/kairos-5b-cluster/p6{c,d}` 对应 `kairos-cluster-sp-4gpu-fit2-5b-custom-l1p0-s101{-matrix,}/p6{c,d}-20261019t1515z` |
| K21 5B VAE 用 BF16（P10） | BF16 解码相对 FP32 官方配方的偏差与显存峰值 | 1×1，build 6a97029，`kairos_vae_dtype.sh`（两个 Wan worker 同卡，`config.vae_dtype`）。512×896×81、20 步：latent 逐位相同（DiT 不变），**视频 PSNR 57.0 dB、LPIPS 均值 0.00025（p95 0.00029）、max_abs 0.05**，远小于 SP 的偏差（K18）。显存：进程基线 10.63 GB（FP32 为 11.95），**704×1280×121 解码峰值 21.3 GB（FP32 32.2）**，给 Sana（峰值 8.2 GB）留 18 GB；解码时间 720p 17.0 s（FP32 33.2 s），中形状 6.8 s（10.5 s）。`release_cache` 生效：FP32 worker 解码后 reserved 从 36 GB 回到 12 GB。决定：5B 的主实验 VAE 用 BF16（论文中注明为可度量的偏差），成本画像按 BF16 重测（P3b） | `kairos-vae-dtype-5b/p10b-20261019t1545z`；本地 `results/kairos-5b-vae/p10b` |
| K17b 5B 成本画像（BF16 VAE，P3b） | 主实验配方（BF16 解码）的 SP1 成本 | 1×1，build 6a97029，`kairos_cost_profile.sh WAN_VAE_DTYPE=bfloat16`。每步与 K17 相同（704×1280×121 8.20 s，512×896×81 1.96 s）；**解码** 704×1280×121 17.3 s（FP32 32.9）、704×1280×81 11.6 s、512×896×81 6.3 s、512×896×17 1.7 s；编码 15.5–17.5 s；Sana 343 ms。模板 `cluster_5b.json` 由此重生成（`vae_dtype: bfloat16`，1× 负载 = 0.1498 rps，长/中类别 SP1 GPU 时间 427/104 s） | `kairos-e2-cost-profile-5b-vae16/p3b-20261019t1550z`；本地 `results/kairos-5b-cost/p3b` |
| K22 5B 四卡回放与模拟预测（P6，Stage C） | 真机上形态弹性（含 CFG2 形态与 widen）对比只用 SP 的 block_elastic 和 static_sp1；模拟器样本外预测 | **预测（真机跑之前记录，`results/kairos-5b-cluster/predictions-p6ef.json`）**：4 卡、1× 负载、seed 101、80 个请求（12 视频/68 图像），fit2 4 卡链路、overhead 12 ms、resize 200 ms：static_sp1 按时 72（视频 12/12、图像 60/68，视频 p95 520 s）；block_elastic 只用 SP 80（12/12、68/68，视频 p95 460 s）；block_elastic 形态+widen 80（12/12、68/68，**视频 p95 304 s**）。真机结果待 p6e（matrix：两组 options）/p6f（static_sp1），build 11fcba2 | 本地 `results/kairos-5b-cluster/predictions-p6ef.json` |
| K16d 按放置拟合链路（成本模型的自校准） | 用一个作业自己的 t-形态计时反推每 rank 等效带宽和交错藏起来的比例（serving 系统启动时可做的事），成本模型在该放置上对所有形态的误差 | `tools/wan22/fit_links.py`，延迟取 K15 值。等效带宽（GB/s，SP2/SP4）：p5d 8.6/3.9，p5f 4.7/5.8，p5c 5.1/–，p5e 6.2/–，而 K15 的 collectives 探测只有 3.8–4.0/2.4：模型里 cache-dit 把 Q/K/V 三个 all-to-all 异步同时发出，有效带宽高于逐个测的探测值，所以**校准要用模型的通信模式（每种形态跑一步），不能用裸 collectives 探测**。拟合后同一作业内各形态的误差：sp2/sp2i/sp4/sp4i/cfg2 都在 ±6% 内（p5e 的 512×896×81 为 +7–9%，2 卡作业只有 1 个 SP 度可拟合）；例外是 p5d 的 cfg2sp2（−17%/−26%）：它的两个 SP2 子组是 {0,1} 和 {2,3}，而 t-sp2 只测了 {0,1} 这对快链路（8.6 GB/s），{2,3} 显然慢得多——同一放置里不同 pair 的带宽也不同，跨子组的形态要按最慢的子组计价。p5f 的 SP4（5.8）反而快于其 SP2 对 {0,1}（4.7），也是同一现象。藏起来的比例：SP2 0.57–0.87，SP4 0.33–0.52。结论：链路表应按对齐组（4 个 pair、2 个 quad、1 个 octet）分别校准，形态按其子组中最慢者计价；`ClusterCost` 目前只按度数建链路，是待办 | 本地 `results/kairos-5b-cfg/p5{c,d,e,f}-*`，`fit_links.py` 输出 |
| K17 5B 成本画像（P3） | serving 路径上 5B 的 SP1 成本：prepare、编码、整段解码、挂起状态 | 1×1，build bcf569d，`kairos_cost_profile.sh WAN_MODEL=wan2.2-ti2v-5b`（7 个 32 对齐形状，6 步，E2 协议）。704×1280×121：每步 8.15 s、**整段解码 32.9 s**、CPU 编码 18.9 s、prepare 36 ms、step 边界挂起状态 92 MB（offload 64 ms、restore 8 ms）；704×1280×81：4.76 s/步、解码 21.9 s；512×896×81：1.93 s/步、解码 10.3 s、挂起 37 MB；512×896×17：0.40 s/步、解码 2.2 s。Sana 1024² 323 ms。解码占长类别 SP1 服务时间的 7%，分段解码（31 个 latent 帧，每段约 1.1 s）因此对图像很重要。由此生成 `configs/studies/target-8xa100/cluster_5b.json`（长/中两类 50 步、Sana，按请求数 20/80，1× 负载 = 8 卡 SP1 的 GPU 时间容量 0.1455 rps，视频期限 1.5×、图像 2.5×） | `kairos-e2-cost-profile-5b/p3-20261019t1325z`；本地 `results/kairos-5b-cost/p3` |
| K18 5B SP 输出质量（P4） | 50 步、serving 形状下 FP16 Ulysses SP2/SP4 相对 SP1 的偏差（LPIPS 阈值 0.05） | 1×4，build bcf569d，`kairos_sp_quality.sh`。512×896×81：SP2 视频 PSNR 25.8 dB、LPIPS 均值 0.041（p95 0.097）；**SP4 21.7 dB、LPIPS 0.107（p95 0.148），超过阈值**。704×1280×121：SP2 与 SP4 都是 60.3 dB、LPIPS 0.0002（两者输出相同，偏差只来自 token 分片后 GEMM 的 M 维不同）。偏差随形状/seed 混沌放大：中形状 50 步后样本明显不同（同一场景、细节不同），长形状几乎一致。结论：SP 形态的输出是"近似"的（与 GenServe/xDiT 一样接受这种数值差异），CFG2/CFG2×SP2 逐位一致；主实验中 SP 形态只作基线，不再加 BF16 选项。附带计时（整步 worker，50 步，这次作业的 GPU 组合）：704×1280×121 SP1 413.7 s、SP2 303.1 s（1.36×）、SP4 171.8 s（2.41×）；512×896×81 99.1/87.9/54.6 s | `kairos-sp-quality-5b/p4-20261019t1325z`；本地 `results/kairos-5b-quality/p4` |
| K19 形态弹性的模拟扫描（P8，Gate B） | 规划器按成本模型在 SP1/CFG2/SP2/CFG2×SP2/SP4/CFG2×SP4/SP8 之间选形态，相比只用 SP 的弹性能否降低视频长尾、提高按时率 | 本地模拟器，build c27bbf7 + widen（`tools/wan22/sim_sweep_5b.py`）：8 卡，`cluster_5b.json` 的类别与成本（长/中 5B + Sana，20/80），`links` 用 K15 的 fit2（4.81/2.72/2.72 GB/s，两组并发 3.9），`sp_overlap` 0（交错未核实），quantum 300 ms，负载 1×/1.5×/2×，10 个 seed。视频按时率 / 图像按时率 / 视频 p95（s）的均值，1×：block_elastic（只用 SP）0.991/1.000/571，**block_forms 0.997/1.000/467**，block_wide（`widen_efficiency` 0.95）1.000/1.000/463，static_cfg2 0.997/0.822/459，static_sp1 0.981/0.881/607，step_elastic 0.997/0.694/532；1.5×：0.857/1.000/1110，0.875/1.000/1010，0.874/1.000/1013，static_cfg2 0.904/0.818/1021，static_sp1 0.843/0.767/1235，step_elastic 0.887/0.679/1026；2×：0.757/1.000/1559，0.767/1.000/1521，0.771/1.000/1509，static_cfg2 0.787/0.827/1525。逐 seed：1× 下 block_forms 把视频 p95 降 ≥15% 的有 8/10（均值 −18%），1.5× 只有 2/10（−9%）；按时率 +0.03 的 1.5× 只有 1/10（+0.018）。**Gate B 的 p95 条件通过、按时率条件不通过**：1.5× 以上系统过载，CFG2 不增加吞吐（2 张卡 4.1 s ≈ 1 张卡 8.2 s），按时率由吞吐决定，所有策略都在 0.86–0.90；static_cfg2 的视频最高（0.904）是以图像 0.82 为代价的。形态弹性的价值是在不伤图像的前提下压低视频 p50/p95（1× 下 p50 −13%，p95 −18%）。static_cfg2sp2 最差（0.838/0.685）：四卡形态效率 0.64，固定用它白白损失吞吐 | 本地 `results/kairos-p8-sim/{c27bbf7,widen}`（每单元 .json/.out.json 与 summary.json） |

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
6. **同一节点上不同的 GPU 组合，SP 带宽相差近一倍（K16b）。** 同在 gpu06，两次 1×4 作业分到的四张卡，SP4 每步 3.38 s 对 4.02 s；SP2 的等效带宽在 5.1–8.7 GB/s 之间。nvidia-smi 的拓扑（全部 PHB）看不出差别，是虚拟机直通后 PCIe 位置不可见。成本模型的链路参数因此是"每次分配"的量，serving 系统启动时要探测（K3 的 collectives 探测约 1 分钟），不能只查静态表。
7. **CFG 并行是 PCIe 节点上最划算的两卡/四卡形态（K16、K16b）。** 704×1280×121 上 CFG2 2.00×、CFG2×SP2 2.56×，都与 SP1 逐位一致；同样的卡数，SP2 只有 1.35–1.55×，SP4 1.71–2.44×。原因是 CFG2 每步只交换一个 10.5 MB 的 latent，而 SP2 每 rank 要搬 10 GB。分支交错能把 SP2 的通信藏起来 57–84%、SP4 的 33%，但在 GPU 上还没做到无损（见 K16b）。

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
- 分支交错（`block_interleaved`）相对同度数的非交错 SP 逐位一致（K16e，SP2 已核实、SP4 待 p5h），但 SP 本身相对 SP1 不是逐位一致的（K13/K18）；论文里"无损"指不改变 SP 的输出。交错形态的计时在修复 allocator 竞争前后一致（K16b 对 K16c/K16e）。
- 形态计时都只有一个 seed，且链路带宽随作业分到的 GPU 组合变化（K16b），不同作业之间的加速比不能直接比，只能比同一作业内的形态。
- 成本模型的链路按 SP 度建表；同一放置里不同 pair 的带宽可差近一倍（K16d），跨子组的形态（CFG2×SP2）会被高估。待办：链路表按对齐组校准（启动时每种形态各跑一步），`ClusterCost.step_ms` 按组取最慢子组的速率。
