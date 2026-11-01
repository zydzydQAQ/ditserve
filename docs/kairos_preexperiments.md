# Kairos 预实验汇总（2026-10-18 至 10-31）

> **2026-10-31 论文主线与文档精简说明（详见 [ditserve_innovation_review.md](file:///mnt/e/000_code/zyd-2026/reports/ditserve_innovation_review.md) 与 [DiT serving 权衡曲线与论文定位.md](file:///mnt/e/000_code/zyd-2026/reports/DiT%20serving%20%E6%9D%83%E8%A1%A1%E6%9B%B2%E7%BA%BF%E4%B8%8E%E8%AE%BA%E6%96%87%E5%AE%9A%E4%BD%8D.md)）**：
> - **主模型组合**：**`Wan2.1-T2V-1.3B` / `Wan2.2-TI2V-5B` + `SD3.5-Medium`**（带 CFG 的双模态生产模型组合；`SD3.5-Medium` 统一开启 `SDPA_FLASH` 公平基线）。
> - **双层架构合流**：第一层 **`DITSERVE-Exact`**（默认无损核心：Block 边界位一致子步抢占、位一致 CFG2 + 双流交错、PCIe 解析成本模型、UMT5 有效 token 截断 + 负向缓存、Rank-0 异步 BF16 分块 3D-VAE）；第二层 **`DITSERVE-Approx`**（§6 可选 `quality_tier="relaxed"` 松弛度门控降级：单算子已验证的 Taylor-DBCache + 固定后段 CFG 折叠，正交叠加于 `DITSERVE-Exact` 之上）。
> - **模型与数据精简原则**：已清理与 `Wan2.1/Wan2.2 + SD3.5-Medium` 无关的 `Sana` / `FLUX` 实验、代码与配置；第四节仅保留纯 **`Wan2.1-T2V-1.3B` / `Wan2.2-TI2V-5B`** 单模型校准与跨模型泛化必要数据（PCIe 成本模型跨模型迁移、位一致 CFG2、双流交错、隔离态确定性与视频合批否定结论；所有保留的已验证 `results.xlsx` 原件完整保留）。

## 1. 实验环境

- 硬件：Kairos 集群（`jjjkairos`），A100-PCIE-40GB，无 NVLink，拓扑为 PHB：所有卡经 CPU host bridge 互连。驱动 570.133.20，低于清单要求的 570.181。2026-10-19 用 1×8 作业整机确认（gpu03，`acceptance/nvlink-20261019t1105z/interconnect.txt`）：8 卡两两之间都是 PHB，`nvidia-smi nvlink --status` 对每张卡报告 "all links are inActive"；P2P 读写可用但走 PCIe；PCI 地址为同一总线上连续的 `00:06.0`–`00:0D.0`，是虚拟机直通布局。
- 作业规模：1×1、1×2、1×4、1×8 容器作业。每张卡配 8 个 CPU、64 GiB 内存上限、32 GB `/dev/shm`；`/shared` 为 NFS 4.2。
- 镜像：历史实验使用过 `rewrite-runtime`、`8a100-1c6282a`、`8a100-2edc0a8`、`8a100-face7c1`、`8a100-224099f` 与 `xdit-016e1ca`；2026-10-29 入库候选镜像为 `192.168.1.114:30500/jjjkairos/ditserve:xdit-dev-016e1ca-slim`（digest `sha256:2009ea26f63bcf0360ce7b4d71ed508c91c0702c97dd97e372cecfb672a818a1`）。代码按提交发布到 `/shared/code/DITSERVE/builds/<commit>/`，作业必须显式设置 `DITSERVE_BUILD=<commit>` 选择版本。
- 脚本：`.local/ditserve-kairos-rewrite-runtime/scripts/`。
- 结果：`/shared/output/DITSERVE/results/<experiment>/<run>/results.xlsx`。作业日志、生成的配置和预算账本在 `acceptance/<run>/`。

---

## 2. Wan2.1 + SD3.5 主线基线（2026-10-28，W0–W6）

主线配置位于 [cluster_wan21_sd35.json](file:///mnt/e/000_code/zyd-2026/ditserve/configs/studies/target-8xa100/cluster_wan21_sd35.json)，第一阶段基线 build 为 `1917afe`。主形状为 `Wan2.1-T2V-1.3B` `480x832x121`（50 步）与 `SD3.5-medium` `1024x1024`（28 步）。2026-10-28 基线采集时 `Wan` 为 cache off、UMT5 CPU FP32，`SD3.5` 为默认 `SDPA_MATH`（2026-10-31 已在 [cluster_wan21_sd35.json](file:///mnt/e/000_code/zyd-2026/ditserve/configs/studies/target-8xa100/cluster_wan21_sd35.json) 中切换为公平基线 `"attention": "flash"`、NVML 后台探测、负向提示词缓存及 12 头兼容的 CFG 优先形态 `["cfg2", "cfg2sp2", "cfg2sp4"]`）。

### W0 当前后端的完整共置

两次作业分别完整执行 Wan 50 步、SD3.5 28 步和 31 个 latent 帧的分段解码，包含首次解码与 SD3.5 出图并发。整卡采样间隔 100 ms；时间单位为 ms，显存为 MiB。

| Wan VAE | 完整动作 | 整卡最大 used / 最小 free | Wan 每步中位 | 分段解码合计 | 并发 SD3.5 完整调用（MATH） |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 | 完成，无 OOM | 33847 / 6596 | 7543.1 | 8564.0 | 50652.4 |
| FP32 | 完成，无 OOM | 38781 / 1662 | 7581.8 | 13130.9 | 51919.1 |

证据：`results/kairos-wan-sd35-mem/wan21-sd35-mainline-20261028t070646z`、`.../wan21-sd35-mainline-fp32-20261028t072056z`，以及 `reports/wan21-sd35-mainline-20261028t070646z/{baseline-bf16-summary,baseline-fp32-summary}.json`。两次工作簿均通过 `results verify`。FP32 虽完成，但最小 free 只有 1662 MiB，不能写成稳定余量。

### W1 SD3.5 原生一致性与成本（MATH 基线 vs FLASH 修正）

`sd35-mainline-native-20261028t071047z` 在完整 28 步、1024²、cache off 下，原生 Diffusers 与 xDiT 输出逐位一致，最终 `results verify` 通过。三次未插桩调用（`SDPA_MATH`）分别为 48.358、48.404、48.511 s，中位 **48.404 s**，整卡最大 used 24667 MiB。`sd35-mainline-stage-20261028t082607z` 插桩分解显示：Transformer CUDA 占阶段合计的 **99.30%**，每步中位 **1.713 s**；VAE decode 约 **194 ms**。证据：`results/kairos-sd35/sd35-mainline-native-20261028t071047z`、`results/kairos-sd35-cost/sd35-mainline-cost-20261028t072056z`、`reports/wan21-sd35-mainline-20261028t070646z/sd35-stage-summary.json`。

> **关键配置修正（2026-10-31）**：W1 中 `SD3.5` 耗时 `48.404 s` 是因为未显式指定 `"attention": "flash"`，触发了 `workers/ditserve/models/sd35.py` 默认的 `"math"` 慢路径。启用 `"attention": "flash"` 后（见第 3 节 HeteroDiT v2/v3 实测），`SD3.5` 单图耗时降至 `0.77 s`（512²）–`2.5 s`（1024²），恢复"图像服务时间（~0.8–2.5 s）远小于 `Wan2.1` 单个去噪步（7.54 s）"的物理事实，使子步 Block 抢占在 `Wan2.1 + SD3.5` 主线上真正绑定。

### W2 Wan VAE 精度的质量边界

两组 `wan21-sd35-vae-q1-20261028t072546z` 与 `wan21-sd35-vae-q2-20261028t074634z` 在不同 prompt/seed、完整 50 步后，FP32 与 BF16 的 latent 均**逐位一致**。BF16 相对 FP32 解码视频：第一组 **PSNR 52.368 dB、平均 LPIPS 0.000493、p95 LPIPS 0.000548**；第二组 **PSNR 59.225 dB、平均 LPIPS 0.000431、p95 LPIPS 0.000563**。指标保存在已验证工作簿与 `vae-quality-q1-summary.json`、`vae-quality-q2-summary.json` 中。

### W3 固定负向提示词缓存

`wan21-prompt-cache-20261028t074207z` 顺序比较缓存关闭/开启，正提示词每次重新编码，固定空负向提示词。三次 CPU 编码的中位数从 **14.567 s** 降为 **7.041 s**（开启组首次冷 miss 14.951 s，随后三次均命中，缓存项 8 MiB）。证据见 `acceptance/wan21-prompt-cache-20261028t074207z/prompt-cache-summary.json`。

### W4 Wan 当前后端的数值与状态恢复

独立的 numeric 单阶段作业 `target-acceptance-vae-bfloat16/wan21-xdit-numeric-r3-20261028t075204z-gpu0` 已完成且数值门通过：官方循环、逐步执行和热/冷恢复保持逐位一致，最终工作簿通过 `results verify`。

### W5 第一轮调度 pilot：有效单元与 watchdog 失败

`wan21-sd35-pilot-r2-20261028t073930z` 使用 2 视频 + 8 图像、NVML 后台探测、300 ms block quantum，正反序执行。前两个单元完成（因 W6 发现的 RNG 重置缺陷导致图像与视频到达未重叠，且当时 `SD3.5` 仍为 48.4 s 的 `MATH` 慢路径，故 static step 与 block 的图像按时均为 2/8、视频均为 1/2）。视频的 `action_ms - model_ms` 在 step 下约 0.35 s/请求（81 quanta），block 下约 2.30–2.36 s/请求（1531 quanta），即约 **1.5 ms/quantum**。第三阶段因 `phase watchdog cannot admit the next complete work unit` 中断，工作簿 SHA256 `1666e64c4b061f1afc83ebe5c20102f80fb822665c9a559b0af480b757a1e9aa` 通过 `results verify`。

### W6 新 VAE 直接分段解码与在线到达修复

- **VAE 分段解码修复**：`wan-vae-inspect-r2-20261028t085924z` 证实 `Wan2.1` VAE 新加载后调用官方 `vae.clear_cache()` 即可初始化 `_conv_num=33`，FP32 分段与整段输出逐位一致。
- **在线到达生成器修复**：修复了每个样本重新构造同种子 RNG 导致同模态到达时刻重合的缺陷。修正后保留 seed 201 Poisson 轨迹（SHA256 `e8ecc3107777ce52e7af3085ce1a35207b55a8d53c9968573853459dbc73d1a3`）与 2 视频 + 8 图像同到压力轨迹（SHA256 `fe1f7151cf52e220cb9e31dc61836008b3e1379f0f302919c3d44e3cd256e851`）。
- **CPU 测试门禁**：完整固定环境 CPU 测试 **323 项通过、1 项跳过**；mock study 工作簿 SHA256 `e3a8d3fb7cea7c8317f65091168c5717b3eddb7d6567fb895446966d2e322754` 通过 `results verify`。

---

## 3. HeteroDiT v1–v5 阶段优化与结果来源分级（2026-10-29 至 10-31）

为保障论文数据严谨性，对 2026-10-29 至 10-31 期间 `heterodit-*` 目录下的全部结果按证据来源分为四级：**`measured_single_op`**（真机单算子/单步实测，可进入论文）、**`mismatched_baseline`**（基线注意力/精度不对称，须重测公平基线后使用）、**`choreographed_script`**（脱离 C++ 调度器的硬编码脚本编排，且缺 `BUILD_INFO.json`，隔离）、**`analytical_scaled`**（含常数乘子或写死基线，废弃）。

### 3.1 可保留的真机单算子/微基准实测（`measured_single_op`）

| 模块 / 实验 | 真机实测数据（A100-PCIE-40GB） | 论文角色与归属层级 | 结果路径 |
| --- | --- | --- | --- |
| **UMT5 文本编码有效 token 截断 + GPU 驻留** | CPU FP32 512-token 全长编码 `14.57–15.43 s`（开启负向缓存 `7.04 s`）；裁掉 496 个 pad token 后 CPU 仅需 **`434.2 ms`**（35× 提速），GPU BF16 有效长度仅 **`20.36 ms`**，与全长编码**输出逐位一致**（`cosine_sim = 1.000000`, `max_abs_diff = 0.0`） | **`DITSERVE-Exact`** 工程支撑（消除緊 SLO 下编码阻塞） | `results/heterodit-v2-opt-20261029-r1/V2_OPTIMIZATION_REPORT.md` |
| **SD3.5-Medium `SDPA_FLASH` 与微批处理** | 512²、20 步下：`SDPA_MATH` `2327.6 ms` → **`SDPA_FLASH` `797.2 ms`**（2.92× 无损提速）；Flash 下 micro-batch `bs=1/2/4` 吞吐为 **`1.30 / 1.54 / 1.66 img/s`**（单图有效耗时 `769.1 / 649.4 / 603.7 ms`） | **`DITSERVE-Exact`** 公平基线必选项（已写入 `cluster_wan21_sd35.json`） | `results/heterodit-v2-opt-20261029-r1/V2_OPTIMIZATION_REPORT.md`、`results/heterodit-v3-opt-20261030-r7/V3_OPTIMIZATION_REPORT.md` |
| **3D-VAE BF16 分块与 Rank-0 异步重叠** | 480×832×121 整段 FP32 解码 `13.13 s`（峰值 38.8 GB）→ BF16 分块解码 **`8.56 s`**（峰值 33.8 GB，PSNR `52.4–59.2 dB`，LPIPS `0.00043`）；480×832×33 帧下 BF16 + CUDA Event 异步流重叠使主循环暴露解码时间从 `860.9 ms` 降至 **`0.13 ms`** | **`DITSERVE-Exact`** 40 GB 显存共驻与尾阶段优化 | `results/kairos-wan-sd35-mem/wan21-sd35-mainline-20261028t070646z`、`results/heterodit-v2-opt-20261029-r1/V2_OPTIMIZATION_REPORT.md` |
| **Wan2.1-1.3B（12 头）8 卡 2D-USP×CFG 实测步时** | `480×832×33` 单步真机耗时：1 卡 `sp1` `1824.2 ms`；2 卡 `u2r1` `981.6 ms`（1.86×）、`cfg2` ~`912 ms`（2.00×）；4 卡 `cfg2u2r1` `539.7 ms`（**3.38×**）优于 `u4r1` `562.2 ms`（3.24×）与 `u2r2` `613.3 ms`（2.97×）；8 卡 **`cfg2u4r1` `530.4 ms`（3.44×，全场最快）** 优于 `u4r2` `547.7 ms`（3.33×）与 `cfg2u2r2` `557.2 ms`（3.27×）；纯 `u8r1` 因 $12 \bmod 8 \neq 0$ 不可用 | **`DITSERVE-Exact`** C1/C2 支撑：证明 12 头模型在 8 卡上必须依赖 `CFG2 × Ulysses4`（`cfg2sp4`） | `results/heterodit-nccl-2d-usp-20261029-r3/NCCL_2D_USP_REPORT.md`（仅取 `Raw Step` 实测列） |
| **单卡近似档质量-延迟 Pareto（Wan2.1）** | 1 卡 20 步 `Wan2.1` 真机：无损基线 `35.42 s`；`DBCache-Taylor (thr=0.08)` `26.82 s`（**1.32×**，PSNR `30.89 dB`，SSIM `0.9426`，LPIPS `0.0367`）；`CFG-Fold (split=0.70)` `27.71 s`（**1.28×**，PSNR `37.73 dB`，SSIM `0.9794`，LPIPS `0.0110`）；组合 `Hybrid (thr=0.12, split=0.65)` `21.38 s`（**1.66×**，PSNR `34.43 dB`，SSIM `0.9613`，LPIPS `0.0235`） | **`DITSERVE-Approx`**（§6 可选 `quality_tier="relaxed"` 离线校准表） | `results/heterodit-quality-replay-20261029-r4/QUALITY_REPORT.md` |

### 3.2 隔离与废弃的记录（禁止直接写入论文正文）

1. **基线不对称（`mismatched_baseline`）**：`benchmarks/run_8gpu_heterodit_v5_suite.py`（第 404、428 行）与 `benchmarks/run_8gpu_baselines_and_ablation.py`（第 461–489、982 行）将所有对比基线硬编码为 `attention="math"` + `FP32-VAE`（SD3.5 单图 `2327–2332 ms`），而本系统开启 `attention="flash"` + `BF16-VAE`（`766–797 ms`）；同为 `SDPA_FLASH` 时，`MM-DiT-DBCache` 从 `797.2 ms` 变为 `796.1 ms`（`PSNR=99.99, SSIM=1.0000`，实际跳过 0 步）。
2. **合成乘子与写死基线（`analytical_scaled`）**：`benchmarks/run_nccl_2d_usp_benchmark.py`（第 62、69、156 行）的 `Cache35%` 列系用 `raw_step_ms * (1.0 - 0.35)` 直接乘出；`benchmarks/run_8gpu_baselines_and_ablation.py`（第 243、1282 行）的 `23360.05 ms` 为写死常数。
3. **未触发的启发式（`choreographed_script`）**：
   - `Slack-Aware Micro-Batch Coalescing`（`benchmarks/run_real_verification_suite.py` 第 871–877 行）依赖读取未来到达时刻 `img_arrivals[j] > cur_t`，且实测合批次数为 `0`；
   - `Curvature-Guided CFG Folding` 在 `HETERODIT_8GPU_V5_SUITE.json`（第 50–119 行）中 `curvature_triggered_folds: 0`（实际折叠来自硬编码 `folded_steps: [11..18]`）；在 `C2_C4_C3_REAL.json` 中强行把阈值调至 `0.35/0.95` 触发后，PSNR 跌至 `25.10–28.55 dB`。
4. **编排脚本、消融回退与溯源缺失（`choreographed_script`）**：
   - `run_8gpu_heterodit_v5_suite.py` 只跑 1 个视频 + 3 波 `time.sleep` 图像，未接入 C++ `control/cluster.cpp`，混合服务全程未测视频/图像质量；
   - 1×8 消融（`HETERODIT_8GPU_BASELINES_AND_ABLATION.json`）中，在 `A5 (8卡 u4r2)` 上加 `A6 (6+2 分区)` 使图像平均延迟从 `811.4 ms` 恶化到 `1035.2 ms`、视频从 `7889.1 ms` 恶化到 `8251.3 ms`；2×4 跨节点（`heterodit-v5-8gpu-2x4-20261031-r3`）中 `Policy 6 (Morphing)` 总按时率跌至 `23.53%`，远低于 `Static-Partition-4v4i` 的 `94.12%`；
   - 全部 8 个 `results/heterodit-v5-*` 目录均缺失 `BUILD_INFO.json`、`results.xlsx` 和 `REPORT.md`。

---

## 4. Wan2.1 / Wan2.2 单模型校准与跨模型扩展必要数据汇总

本节保留纯 `Wan2.1-T2V-1.3B`（12 注意力头、8× VAE 空间压缩）与 `Wan2.2-TI2V-5B`（24 注意力头、16× VAE 空间压缩、`704×1280×121` 8.20 s/步、`512×896×81` 1.96 s/步）的单模型校准与跨模型扩展证据（与 `Sana`/`FLUX` 相关的实验已全部清理；`Wan2.1/Wan2.2 + SD3.5-Medium` 8 卡混合服务实验见第 2、3 节及 [heterodit_v5_8gpu_experiments.md](file:///mnt/e/000_code/zyd-2026/ditserve/docs/heterodit_v5_8gpu_experiments.md)）。所有保留实验均通过 `results verify`，对应工作簿完整保留在 `ditserve/results/` 与 `/shared/output/DITSERVE/results/`。

### 4.1 必要数据一：PCIe SP/CFG 扩展与解析成本模型跨模型迁移（K7 / K14 / K15 / K16d）

- **跨模型零重拟合迁移（K7 → K14）**：直接复用在 `Wan2.1-1.3B`（6 个形状，K7）上拟合的 PCIe 链路参数（$\eta=1.0$，$B_2=4.81\text{ GB/s}$，$B_4=2.72\text{ GB/s}$），仅代入 `Wan2.2-5B` 的解析 all-to-all 通信量，在 `Wan2.2-5B` 的 **16 个形状×度数**组合上每步时间预测误差全部 **$\le 3.7\%$**（K15 扩至 8 卡 `SP2/4/8` 最大误差分别为 `3.3% / 4.1% / 2.6%`）。
- **主机桥聚合上限与放置差异（K15 / K16b / K16d）**：8 卡 PHB 节点上 host bridge 聚合上限约 `19 GB/s`（$\ge 4$ 卡并发 all-to-all 时每 rank 约 `2.4 GB/s`）。同一节点（`gpu06`）不同四卡组合的 `SP4` 步时相差 **19%**（`3.38 s` vs `4.02 s`，`nvidia-smi topo` 均显示 `PHB`），证明启动时按对齐子组执行 1 步模型通信自校准（`links.groups`）的必要性。

| 模型与形状 | Token 数 | SP1 步时 | SP2（加速） | CFG2（位一致加速） | SP4（加速） | CFG2×SP2（加速） | SP8（加速） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Wan2.1-1.3B` `720×1280×81` (K7) | 75.6k | 16.80 s | 11.40 s (1.48×) | — | 8.05 s (2.09×) | — | 不可用 (12 头) |
| `Wan2.2-5B` `512×896×81` (K15/K16) | 9.4k | 1.97 s | 1.74 s (1.13×) | **0.99 s (1.99×)** | 1.53 s (1.29×) | **0.89 s (2.22×)** | 0.94 s (2.09×) |
| `Wan2.2-5B` `704×1280×121` (K15/K16) | 27.3k | 8.26 s | 6.04 s (1.37×) | **4.11 s (2.01×)** | 4.02–4.72 s (1.76–2.05×) | **3.06–3.21 s (2.56–2.70×)** | 2.70 s (3.07×) |

### 4.2 必要数据二：位一致 CFG2、双流交错隐藏与隔离态可预测性（K16 / K16e / K35 / K36 / K37）

1. **位一致验证（K16 / K16e）**：`CFG2`（整步与 13-block 量子）相对单卡 `SP1` 的 latent 与解码视频在双 rank 上均**逐位一致（`exact`）**；双流交错 `sp2i / sp4i`（修复 caching allocator `record_stream` 后）相对同度数非交错 `sp2 / sp4` **逐位一致（`exact`）**。而纯 `SP2 / SP4` 相对 `SP1` 存在 FP16 GEMM 切分偏差（`512×896×81` 50 步后 LPIPS `0.041 / 0.107`）。
2. **双流交错通信隐藏（K16b–d / K37 P13b/d）**：在 40 步统计（去首步 39 步，K37）下，将 `cond/uncond` 分支交错到两条 CUDA stream 上可隐藏 `33%–87%` 的 Ulysses all-to-all 通信：`704×1280×121` 上 `SP2i` 比 `SP2` 快 **22.5%**（`4.96 s` vs `6.08 s`），`SP4i` 快 **16.2%**（`4.08 s` vs `4.74 s`），`SP8i` 快 **15.3%**（`2.33 s` vs `2.69 s`，达 **3.58×**）。
3. **隔离态延迟确定性（K35 P13a）与视频合批否定结果（K36 P14b/c）**：
   - `Wan2.2-5B` 300 步隔离态测量：`704×1280×121` 单步 `CV = 0.46%`（`p99/median = 1.006`），`512×896×81` 单步 `CV = 0.38%`，跨 prompt/seed 均值极差 `< 1%`；带通信的 `SP2/4/8` 单步 `CV <= 0.25%`（K37）。
   - 视频动态 batching 在单卡上吞吐为 `0.99–1.00×`（显存从 `15.7 GiB` 暴涨至 `31.1 GiB`），单卡双请求锁步交错多出 `+3.3%` 流切换开销；而 `SD3.5-Medium` 图像 batch 2–4 有显著吞吐收益（W4，`batch=4` 达 `21.74 s/张` vs 单张 `51.83 s` MATH）。这从物理上证成了"仅对图像做 micro-batching、对视频做子步抢占与 CFG/SP 并行"。

### 4.3 保留的单模型校准与扩展工作簿索引

- **Wan2.1 / Wan2.2 成本、扩展与交错校准**：`results/kairos-sp-scaling-{2,4}gpu`（K7）、`results/kairos-sp{2,4}-equivalence`、`results/kairos-5b-quick`（K14）、`results/kairos-8gpu-probe`（K15）、`results/kairos-5b-cfg/p5{a..h}-*`（K16–K16e）、`results/kairos-5b-vae`（K21）、`results/kairos-5b-quality`、`results/kairos-5b-pair`、`results/kairos-pair1`、`results/kairos-wan-batch/p14{,b}`（K36）、`results/kairos-calib-steps/p13{b,d}`（K37）、`results/kairos-links-calibrate-8gpu`。
- **Wan2.1 / Wan2.2 + SD3.5 主线基线（W0–W6）**：`results/kairos-sd35`、`results/kairos-sd35-cost`、`results/kairos-wan-sd35-mem`、`results/kairos-wan-sd35-mem-5b`、`results/kairos-wan21-prompt-cache`、`results/kairos-wan21-sd35-vae-quality`、`results/kairos-wan21-sd35-mainline-s201-l1p0`、`results/target-acceptance-vae-bfloat16`。
