# HeteroDiT-v5 (DitServe) 1-GPU / 4-GPU / 8-GPU 实验总报告与数据索引

> **最后更新**：2026-10-31  
> **硬件环境**：`jjjkairos` 集群（`NVIDIA A100-PCIE-40GB`，16 张 GPU 配额，支持单机 `1×8` 与跨节点 `2×4` 拓扑，PyTorch `2.10.0+cu126`，NCCL `2.27.5`）  
> **代码分支**：`/mnt/e/000_code/zyd-2026/ditserve` (`8a100`)

---

## 一、所有实验结果 JSON 与代码路径索引（供其他 Agent 直接读取）

### 1. 核心原始实验 JSON 文件（位于 `/mnt/e/000_code/zyd-2026/ditserve/results/`）

| 实验维度 | 拓扑配置 | 本地绝对路径 |
| --- | --- | --- |
| **8 卡全量 11 系统对比 (`B1..B9, Ours-1, Ours-2`) + 8 级累加消融 (`A0..A7`) + 5 项正交消融 (`C1..C5`)** | **单机 8×A100 (`1×8`)** | `/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-v5-8gpu-ablation-1x8-20261031-r2/HETERODIT_8GPU_BASELINES_AND_ABLATION.json` |
| **8 卡全量 11 系统对比 (`B1..B9, Ours-1, Ours-2`) + 8 级累加消融 (`A0..A7`) + 5 项正交消融 (`C1..C5`)** | **跨节点 8×A100 (`2×4`)** | `/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-v5-8gpu-ablation-2x4-20261031-r2/HETERODIT_8GPU_BASELINES_AND_ABLATION.json` |
| **8 卡五项优化验证 + 10 种并行矩阵 (`SP1..cfg2u4r1`) + 6 策略评测** | **单机 8×A100 (`1×8`)** | `/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-v5-8gpu-1x8-20261031-r9/HETERODIT_8GPU_V5_SUITE.json` |
| **8 卡五项优化验证 + 10 种并行矩阵 (`SP1..cfg2u4r1`) + 6 策略评测** | **跨节点 8×A100 (`2×4`)** | `/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-v5-8gpu-2x4-20261031-r3/HETERODIT_8GPU_V5_SUITE.json` |
| **4 卡五项优化验证（`3+1` 子集群 + Rank0 异步 VAE + 曲率 CFG + `xfuser` INT8 Ring）** | **单机 4×A100 (`1×4`)** | `/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-v5-4gpu-1x4-20261031-r5/RING_INT8_AND_4GPU_REAL.json` |
| **4 卡五项优化验证（`3+1` 子集群 + Rank0 异步 VAE + 曲率 CFG + `xfuser` INT8 Ring）** | **跨节点 4×A100 (`2×2`)** | `/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-v5-4gpu-2x2-20261031-r5/RING_INT8_AND_4GPU_REAL.json` |
| **单卡全套验证（精简后回归测试）** | **单卡 1×A100 (`1×1`)** | `/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-slim-1gpu-20261031-r3/REAL_VERIFICATION_SUITE.json` |
| **图表文件（PNG / PDF / LaTeX 表）** | 汇总图表 | `/mnt/e/000_code/zyd-2026/ditserve/results/figures_and_tables/` |

### 2. 核心实现与评测脚本路径

- **五项联合优化核心模块**：`/mnt/e/000_code/zyd-2026/ditserve/workers/ditserve/models/hetero.py`
  - `AsymmetricSubClusterController`（Opt #1：`3+1`、`6+2`、`6+2 -> 3+5` 动态 CFG-Morphing 非对称子集群控制器）
  - `ConcurrentVAEDecodeStreamer`（Opt #2：`submit_rank0_disaggregated` & `collect_rank0_disaggregated` 多卡 Rank-0 异步 3D-VAE 解码卸载）
  - `AsymmetricCFGStepper`（Opt #4：`observe_cfg_pair` 基于二阶轨迹曲率的动态 CFG 折叠）
  - `AsymmetricRingQuantizer`（Opt #5：`install_xfuser_hook` 嵌入 `xfuser` USP Ring Attention P2P 内核钩子）
- **CPU 文本编码器优化池**：`/mnt/e/000_code/zyd-2026/ditserve/workers/ditserve/models/encoder_pool.py`
  - `ProactiveEncoderPool`（Opt #3：`UMT5-XXL` Active-Token 序列动态裁剪 + BF16 / `torch.ao` INT8 动态权重量化 + Pinned Memory LRU 缓存）
- **`xfuser` USP Ring P2P 钩子集成点**：
  - `/mnt/e/000_code/zyd-2026/ditserve/workers/xfuser/core/long_ctx_attention/hybrid/usp.py`
  - `/mnt/e/000_code/zyd-2026/ditserve/workers/ditserve/models/xdit/fp8_comms.py`
- **评测脚本**：
  - 8 卡 11 系统 Baseline 与消融实验：`/mnt/e/000_code/zyd-2026/ditserve/benchmarks/run_8gpu_baselines_and_ablation.py`
  - 8 卡五项优化与 10 种并行拓扑评测：`/mnt/e/000_code/zyd-2026/ditserve/benchmarks/run_8gpu_heterodit_v5_suite.py`
  - 4 卡五项优化评测：`/mnt/e/000_code/zyd-2026/ditserve/benchmarks/run_ring_int8_crossnode.py`

---

## 二、Part A：8×A100 全量 11 系统对比（9 个 Baselines + 2 个 Ours）

**统一 8 卡混合负载 Trace（共 17 个请求）**：
- `t = 0 ms`：1 个 `Wan2.1-T2V-1.3B` 视频生成请求（`480×832×17`，20 步；严格 8 卡 SLO $\le 9,000\text{ ms}$，跨节点宽松 SLO $\le 12,000\text{ ms}$）
- `t = 700 ms`（**Wave 1**）：4 张 `SD3.5-Medium` 图像突发（SLO $\le 1,500\text{ ms}$）
- `t = 1,800 ms`（**Wave 2**）：4 张 `SD3.5-Medium` 图像突发（SLO $\le 1,500\text{ ms}$）
- `t = 3,000 ms`（**Wave 3**）：8 张 `SD3.5-Medium` 高并发图像脉冲（SLO $\le 1,500\text{ ms}$）

### 1. 单机 8 卡（`1×8` Intra-Node PCIe）11 系统实测对比

| 编号与系统策略 | 架构机制 | Wave 1 延迟 (4 图) | Wave 2 延迟 (4 图) | Wave 3 延迟 (8 图) | 图像平均延迟 | 图像 P99 延迟 | 视频端到端延迟 | 图像 SLO ($\le 1.5\text{s}$) | 视频 SLO ($\le 9\text{s}$) | **总 SLO 达成率** |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **B1: Vanilla-xDiT-8GPU** | 8 卡 `u4r2` 静态 FCFS 不可抢占 + FP32 VAE | 14,551.2 ms | 13,451.2 ms | 14,621.5 ms | 14,311.3 ms | 14,621.5 ms | 12,879.6 ms | 0.0% | 0.0% | **0.0%** |
| **B2: Static-SP1-EDF** | 8 路独立 SP1 Worker 非抢占 EDF | 2,314.3 ms | 3,531.0 ms | 6,953.5 ms | 4,938.1 ms | 6,953.5 ms | 12,346.4 ms | 0.0% | 0.0% | **0.0%** |
| **B3: Static-Partition-4v4i** | 静态对半切分：4 卡视频 `u4r1` + 4 卡专职图像（含块缓存与 BF16 VAE） | 651.8 ms | 621.0 ms | 1,006.1 ms | 821.3 ms | 1,006.1 ms | 10,947.3 ms | **100.0%** | 0.0% | **94.12%** |
| **B4: Static-Partition-6v2i** | 静态非对称切分：6 卡视频 `u3r2` + 2 卡图像（无借卡） | 1,419.2 ms | 1,729.2 ms | 3,252.4 ms | 2,413.3 ms | 3,252.4 ms | 14,327.3 ms | 25.0% | 0.0% | **23.53%** |
| **B5: TridentServe-Stage** | 8 卡 `u4r2` 阶段边界抢占（仅在扩散与 VAE 间让出） | 12,916.8 ms | 14,135.5 ms | 15,286.8 ms | 14,406.5 ms | 15,286.8 ms | 19,720.6 ms | 0.0% | 0.0% | **0.0%** |
| **B6: TetriServe-Round** | 8 卡 `u4r2` 粗粒度 5 步 Round 边界抢占 | 2,917.0 ms | 2,597.8 ms | 2,179.5 ms | 2,468.5 ms | 2,917.0 ms | 15,038.3 ms | 0.0% | 0.0% | **0.0%** |
| **B7 / A0: GenServe-40GB** | 8 卡 `u4r2` 单步抢占 + 真实 PCIe 冷权重换入换出 (`4,500.3 ms`) | 2,915.3 ms | 5,920.4 ms | 8,742.0 ms | 6,579.9 ms | 8,742.0 ms | 24,295.3 ms | 0.0% | 0.0% | **0.0%** |
| **B8 / A1: vLLM-Omni / DitServe-GangStep** | 8 卡 `u4r2` 单步 Gang 抢占 + Tier-0 常驻 + 精确全步 ($\tau=0$) | 1,211.2 ms | 871.6 ms | 1,014.4 ms | 1,027.9 ms | 1,211.2 ms | 14,991.3 ms | **100.0%** | 0.0% | **94.12%** |
| **B9: HeteroDiT-Joint-v4** | 8 卡 `u4r2` Gang 抢占 + 静态 AsymCFG + Taylor + 同步 BF16 VAE | 1,038.5 ms | 1,072.0 ms | 1,044.7 ms | 1,050.0 ms | 1,072.0 ms | 9,431.8 ms | **100.0%** | 0.0% | **94.12%** |
| **Ours-1 / A6: HeteroDiT-Elastic-v5** | **Opt #1-#5：`6+2` 弹性非对称子集群 (`u3r2`) + INT8-Ring + 曲率 CFG + Rank0 异步 VAE** | 1,006.7 ms | 1,003.0 ms | 1,065.6 ms | 1,035.2 ms | 1,065.6 ms | **8,251.3 ms** | **100.0%** | **100.0%** | **100.0%** |
| **Ours-2 / A7: HeteroDiT-Morphing-v5** | **Opt #1-#5：`6+2 -> 3+5` 动态 CFG-Morphing (`cfg2u3r1 -> u3r1`) + 曲率 CFG + Rank0 异步 VAE** | 1,006.6 ms | 918.2 ms | **597.5 ms** | **780.0 ms** | **1,006.6 ms** | **7,358.8 ms** | **100.0%** | **100.0%** | **100.0%** |

### 2. 跨节点 8 卡（`2×4` Cross-Node Inter-Node）11 系统实测对比

| 编号与系统策略 | Wave 1 延迟 | Wave 2 延迟 | Wave 3 延迟 | 图像平均延迟 | 视频端到端延迟 | 图像 SLO ($\le 1.5\text{s}$) | 视频 SLO ($\le 9\text{s}$ / $\le 12\text{s}$) | **总 SLO ($\le 9\text{s}$ / $\le 12\text{s}$)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **B1: Vanilla-xDiT-8GPU** | 23,550.3 ms | 22,450.3 ms | 23,604.2 ms | 23,302.3 ms | 21,886.9 ms | 0.0% | 0.0% / 0.0% | **0.0% / 0.0%** |
| **B2: Static-SP1-EDF** | 2,323.0 ms | 3,544.7 ms | 6,970.9 ms | 4,952.4 ms | 12,476.3 ms | 0.0% | 0.0% / 0.0% | **0.0% / 0.0%** |
| **B3: Static-Partition-4v4i** | 646.9 ms | 621.8 ms | 1,009.0 ms | 822.1 ms | 10,879.0 ms | **100.0%** | 0.0% / 100.0% | **94.12% / 100.0%** |
| **B4: Static-Partition-6v2i** | 1,413.5 ms | 1717.2 ms | 3,216.4 ms | 2,390.9 ms | 27,962.2 ms | 25.0% | 0.0% / 0.0% | **23.53% / 23.53%** |
| **B5: TridentServe-Stage** | 22,567.8 ms | 23,793.5 ms | 24,924.9 ms | 24,052.8 ms | 29,367.8 ms | 0.0% | 0.0% / 0.0% | **0.0% / 0.0%** |
| **B6: TetriServe-Round** | 5,489.1 ms | 5,172.6 ms | 4,788.0 ms | 5,059.4 ms | 25,317.5 ms | 0.0% | 0.0% / 0.0% | **0.0% / 0.0%** |
| **B7 / A0: GenServe-40GB** | 3,066.2 ms | 5,783.5 ms | 8,336.9 ms | 6,380.9 ms | 33,743.3 ms | 0.0% | 0.0% / 0.0% | **0.0% / 0.0%** |
| **B8 / A1: vLLM-Omni / DitServe-GangStep** | 1,181.2 ms | 876.4 ms | 1,473.8 ms | 1,251.3 ms | 24,761.7 ms | **100.0%** | 0.0% / 0.0% | **94.12% / 94.12%** |
| **B9: HeteroDiT-Joint-v4** | 1,164.6 ms | 1,210.4 ms | 1,134.9 ms | 1,161.2 ms | 17,184.4 ms | **100.0%** | 0.0% / 0.0% | **94.12% / 94.12%** |
| **Ours-1 / A6: HeteroDiT-Elastic-v5** | 1,004.7 ms | 1,015.3 ms | 1,284.2 ms | 1,147.1 ms | 14,807.3 ms | **100.0%** | 0.0% / 0.0% | **94.12% / 94.12%** |
| **Ours-2 / A7: HeteroDiT-Morphing-v5** | 1,004.0 ms | 936.9 ms | **710.1 ms** | **840.3 ms** | **9,429.8 ms** | **100.0%** | 0.0% / **100.0%** | **94.12% / 100.0%** |

---

## 三、Part B：8 级逐项累加消融实验（Cumulative One-Factor Ablation `A0 -> A7`）

| 消融阶段 | 累加引入的核心技术 | 单机 `1×8` 图像均值 | 单机 `1×8` 视频延迟 | 单机 `1×8` 视频边际收益 ($\Delta$) | 单机 `1×8` 总 SLO ($\le 9\text{s}$) | 跨节点 `2×4` 图像均值 | 跨节点 `2×4` 视频延迟 | 跨节点 `2×4` 视频边际收益 ($\Delta$) | 跨节点 `2×4` 总 SLO ($\le 12\text{s}$) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A0** | 基线：8 卡 `u4r2` 单步 Gang 抢占 + 冷 PCIe 权重换入换出 + FP32 同步 VAE | 6,579.9 ms | 24,295.3 ms | — | 0.0% | 6,380.9 ms | 33,743.3 ms | — | 0.0% |
| **A1** | **+ Tier-0 永久 HBM 双模型共驻 & Opt #3 CPU 编码池 + Pinned LRU 缓存** | 1,027.9 ms | 14,991.3 ms | **-9,304.0 ms** | 94.12% | 1,251.3 ms | 24,761.7 ms | **-8,981.6 ms** | 94.12% |
| **A2** | **+ Opt #5 `xfuser` USP 内嵌 `AsymmetricRingQuantizer` INT8 Ring P2P 钩子** | 1,181.4 ms | 12,655.9 ms | **-2,335.4 ms** | 94.12% | 1,051.6 ms | 18,069.6 ms | **-6,692.1 ms** | 94.12% |
| **A3** | **+ 二阶 Taylor 补偿 DBCache ($\tau=0.18$) & SD3.5 Hooked MM-DiT 缓存** | 835.5 ms | 10,387.9 ms | **-2,268.0 ms** | 94.12% | 910.6 ms | 14,266.2 ms | **-3,803.4 ms** | 94.12% |
| **A4** | **+ Opt #4 基于轨迹曲率的动态非对称 CFG 折叠 (`AsymmetricCFGStepper`)** | 826.4 ms | **8,349.1 ms** | **-2,038.8 ms** | **100.0%** | 854.8 ms | **10,901.0 ms** | **-3,365.2 ms** | **100.0%** |
| **A5** | **+ Opt #2 Rank-0 解耦异步 BF16 3D-VAE 解码卸载 (`ConcurrentVAEDecodeStreamer`)** | 811.4 ms | **7,889.1 ms** | **-460.0 ms** | **100.0%** | 877.5 ms | **10,518.0 ms** | **-383.0 ms** | **100.0%** |
| **A6** | **+ Opt #1a 非对称 `6+2` 弹性子集群切分 (`HeteroDiT-Elastic-v5`, `u3r2`)** | 1,035.2 ms | **8,251.3 ms** | 零全局阻塞 | **100.0%** | 1,147.1 ms | 14,807.3 ms | 跨机 `u3r2` Ring 瓶颈 | 94.12% |
| **A7** | **+ Opt #1b+#4 动态 CFG-Morphing `6+2 -> 3+5` 子集群 (`HeteroDiT-Morphing-v5`)** | **780.0 ms** | **7,358.8 ms** | **-892.5 ms** | **100.0%** | **840.3 ms** | **9,429.8 ms** | **-5,377.5 ms** | **100.0%** |

---

## 四、Part C：5 项优化组件级正交消融实验（Component-Level Orthogonal Ablations `C1..C5`）

- **`C1`（Opt #1 子集群拓扑正交消融）**：验证 `8+0 Gang`、`4+4 Static`、`6+2 Static`、`6+2 Elastic`、`6+2 -> 3+5 CFG-Morphing` 五种拓扑；`6+2 -> 3+5` 在单机下达到 `Wave 3 = 597.5 ms, Video = 7,358.8 ms`，跨节点达到 `Wave 3 = 710.1 ms, Video = 9,429.8 ms`。
- **`C2`（Opt #2 多卡 3D-VAE 解码消融）**：
  - `FP32 同步 8 卡屏障`：`1,535.96 ms`（Ranks 1..7 空转浪费 `10,751.7 GPU·ms`，显存 `484.23 MB`）
  - `BF16 同步 8 卡屏障`：`1,087.70 ms`（Ranks 1..7 空转浪费 `7,613.9 GPU·ms`，显存 `242.12 MB`）
  - `BF16 Rank-0 解耦异步流解码`：**`1.244 ms`**（Ranks 1..7 仅需 `8.71 GPU·ms` 即可释放，**消除 `1,234.4×` 多卡屏障气泡**）
- **`C3`（Opt #3 CPU 侧 `UMT5-XXL` 5.68B 编码器消融）**：
  - `FP32 Full-512`：`21.163 GB`，`23,360.05 ms`（$1.0\times$）
  - `FP32 + Active-Token 序列裁剪`：`21.163 GB`，`1,137.14 ms`（**$20.54\times$ 提速**，余弦相似度 `1.000000` 数学等价无损）
  - `BF16 + Active-Token 序列裁剪`：`10.582 GB`（节省 50% 内存），`3,882.09 ms`（$6.02\times$，余弦相似度 `0.999632`）
  - `torch.ao INT8 动态权重量化 + 序列裁剪`：**`8.226 GB`（节省 61.1% 内存）**，**`474.77 ms`（$49.20\times$ 提速）**
  - `ProactiveEncoderPool Pinned LRU 缓存命中`：**`1.1885 ms`（$19,654.7\times$ 提速，余弦相似度 `1.000000`）**
- **`C4`（Opt #4 基于轨迹曲率的动态 CFG 阈值扫描）**：
  - `Exact 20-step`：`10,927.79 ms`
  - `Profile A (min_full=11, eps_cfg=0.35)`：`7,471.33 ms`（$1.463\times$，`PSNR = 28.55 dB, SSIM = 0.9041`）
  - `Profile B (min_full=8, eps_cfg=0.35)`：`7,160.98 ms`（$1.526\times$，`PSNR = 26.72 dB, SSIM = 0.8687`）
  - `Profile C (min_full=6, eps_cfg=0.95)`：`6,849.05 ms`（$1.596\times$，`PSNR = 25.10 dB, SSIM = 0.8377`）
- **`C5`（Opt #5 `xfuser` USP 内嵌 INT8 Ring P2P 钩子消融）**：
  - 每次 20 步生成拦截 240 个 Ring KV P2P 包，将通信流量从 `1,437.70 MB` 压缩至 `730.08 MB`（节省 `49.22%` 带宽），单机 `u4r2` 提速 `1.28×`（`560.85 -> 440.37 ms/step`），跨节点 `u4r2` 提速 `1.43× ~ 2.02×`（`1,025.08 -> 716.66 ms/step`，孤立微基准 `1,321.06 -> 652.86 ms/step`），潜变量余弦相似度 `0.999998 ~ 1.000000`。

---

## 7. `Wan2.2-TI2V-5B` + `SD3.5-Medium` 8 卡实机验证与对比（2026-11-01）

对应原始 JSON 数据：
- 单机 8 卡 (`1×8` PCIe)：`/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-wan22-8gpu-1x8-20261101-r2/WAN22_8GPU_VERIFICATION.json`
- 双机 8 卡 (`2×4` Cross-Node)：`/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-wan22-8gpu-2x4-20261101-r2/WAN22_8GPU_VERIFICATION.json`
- 曲率门控 CFG 折叠 Pareto 扫描：`/mnt/e/000_code/zyd-2026/ditserve/results/heterodit-wan22-8gpu-1x8-20261101-r2/WAN22_C4_CURVATURE_SWEEP.json`

### 7.1 架构与 40GB A100 Tier-0 HBM 常驻核验
- **`Wan2.2-TI2V-5B` 模型结构**：30 层、`num_attention_heads = 24`、`head_dim = 128`（`hidden_dim = 3072`）、`ffn_dim = 14336`、48 通道高压缩 `16×16×4` 3D-VAE（`spatial_multiple = 32`）以及逐 Token 时间步（`expand_timesteps = True`）。
- **40GB A100 显存占用实测**：
  - `Wan2.2-TI2V-5B` DiT（`4,999.79M` 参数，FP16）：**`9,536.84 MB`**
  - `Wan2.2-TI2V-5B` 3D-VAE（`704.69M` 参数，BF16）：**`1,344.09 MB`**（若用默认 FP32 则占 `2,688.17 MB`，解码峰值 HBM 达 **`27,404.58 MB`** vs BF16 **`21,931.59 MB`**）
  - `SD3.5-Medium` MM-DiT（`4,710.51 MB`）+ 2D-VAE（`159.87 MB`）
  - **Tier-0 总常驻 HBM**：**`17,380.09 MB`（`16.97 GiB`）**，在 40GB A100 上留出 **`22,141.56 MB`（`21.62 GiB`）** 激活裕量；而若不通过 `ProactiveEncoderPool` 卸载 `UMT5-XXL`（`21,670.95 MB`）与 `T5-XXL/CLIP`（`9,120 MB`），静态权重总计 **`48.2 GB` > `40.4 GB`**，在 40GB A100 上直接触发 OOM。

### 7.2 8 卡并行扩展矩阵（`Wan2.2` 的 24 头同时解锁 `u3r1`、`u4r1`、`u6r1` 与 `u8r1`）
| 并行配置 (`480×832×17`, 20 steps) | 单机 `1×8` Step (ms) | 单机 20-Step (ms) | 双机 `2×4` Step (ms) | 双机 20-Step (ms) | 关键结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `1-GPU (SP1: u1r1)` | `342.65` | `6,853.0` | `342.89` | `6,857.8` | `16×16×4` 高压缩 VAE 将 Token 数从 `7,800` 降至 `1,950` |
| `3-GPU (u3r1)` | `365.16` | `7,303.2` | `375.92` | `7,518.4` | 支撑 `3+5` 形态吸收器与双视频并发流水 |
| `4-GPU (u4r1)` | `343.86` | `6,877.2` | `346.92` | `6,938.3` | 节点内 4 卡 Ulysses |
| `6-GPU (u3r2 - FP16 Ring)` | `389.60` | `7,792.0` | `740.54` | `14,810.9` | 跨节点 FP16 Ring 受限于以太网带宽 |
| `6-GPU (u3r2 + Opt #5 INT8 Ring)` | **`310.11`** | **`6,202.1`** | **`593.12`** | **`11,862.5`** | **Opt #5 INT8 Ring 节省 49.22% 流量，单机/跨节点提速 `1.25×`** |
| `6-GPU (cfg2u3r1)` | **`203.61`** | **`4,072.2`** | **`415.61`** | **`8,312.1`** | **6 卡最优拓扑：较 `u3r2` 快 `1.91×`（单机）/ `1.78×`（跨节点）** |
| `8-GPU (u8r1 - 24 头新解锁!)` | `266.75` | `5,334.9` | `427.61` | `8,552.3` | `Wan2.1` (`12%8!=0`) 无法运行，`Wan2.2` (`24%8==0`) 原生支持 |
| `8-GPU (u4r2 - FP16 Ring)` | `343.41` | `6,868.2` | `567.63` | `11,352.6` | 8 卡标准 2D-USP 基线 |
| `8-GPU (u4r2 + Opt #5 INT8 Ring)` | **`282.16`** | **`5,643.3`** | **`373.59`** | **`7,471.7`** | **Opt #5 在跨节点 `2×4` 上将 `u4r2` 提速 `1.52×`，CosSim = `1.0`** |
| `8-GPU (cfg2u4r1)` | **`177.06`** | **`3,541.2`** | **`179.55`** | **`3,591.0`** | 8 卡纯视频最快拓扑（`1.94×` vs SP1） |

### 7.3 五项优化在 `Wan2.2-TI2V-5B` 上的关键放大效应
1. **Opt #2（Rank-0 解耦异步 BF16 3D-VAE）**：`Wan2.2` 的 48 通道 `704.69M` 参数 3D-VAE 在 FP32 下同步阻塞 8 卡长达 **`1,982.30 ms`**（浪费 Ranks 1..7 共 **`13,876.1 GPU·ms`**，且峰值显存高达 `27.40 GB`）；Opt #2 将 Ranks 1..7 释放延迟降至 **`0.685 ms`**（削减气泡 **`2,893.2×`**），并在后台流以 BF16 完成解码（`1,182.77 ms`，节省 `5.47 GB` 峰值显存）。
2. **Opt #4（曲率门控动态 CFG 折叠）**：`Wan2.2` 的 48 通道高压缩潜空间在步 1..10 具有显著更高的轨迹曲率（步 3 达 `2.55`，步 7 与步 10 仍有 `1.11 ~ 1.12` 的曲率尖峰）。若盲目套用 `Wan2.1` 的静态步 6 折叠规则，SSIM 会跌至 `0.8248`（PSNR `19.69 dB`）；而通过 **Opt #4 曲率门控触发器**自动避开前 10 步高曲率区，在步 11..13 曲率平稳后触发折叠，实测达到 **PSNR `29.60 dB`、SSIM `0.9720`**（Profile A）与 **PSNR `27.04 dB`、SSIM `0.9521`**（Profile B，提速 `1.19×`）。

### 7.4 8 卡混合负载 10 系统对比与双视频并发争用实测（`Wan2.2-TI2V-5B + SD3.5-Medium`）
- **单视频 + 3 波 16 图混合负载（单机 `1×8` / 双机 `2×4`）**：
  - `B1 (Vanilla-xDiT-8GPU)`：单机视频 `8,839.0 ms`、图像均值 `10,346.4 ms`（SLO `5.88%`）；双机视频 `13,814.2 ms`、图像均值 `15,296.2 ms`（SLO `0.00%`）。
  - `B7 / A0 (GenServe-40GB 冷切权重)`：`Wan2.2` 的 `9.31 GiB` DiT 单次 PCIe 往返置换高达 **`11,684.8 ms`**（单机）/ **`9,819.1 ms`**（双机），导致视频延迟劣化至 **`44,606.7 ms`**，总 SLO `0.00%`。
  - `B8 / A1 (DitServe-GangStep)`：双机视频延迟 **`15,766.4 ms`**（违反 13.5s 严格 SLO），单机视频延迟 `11,227.5 ms`。
  - `B3-Vanilla (原生无缓存 Static-4v4i)`：图像均值 `1,182.4 ms`（Wave 3 达 `1,482.2 ms` 逼近违约线），视频延迟 `8,788.9 ms`。
  - `B3-Opt (搭载 HeteroDiT 缓存的 Static-4v4i)`：单视频下图像均值 `825.8 ms`（Wave 3 `1,014.9 ms`），视频延迟 `7,405.3 ms`。
  - **`Ours-2: HeteroDiT-Morphing-v5 (6+2 -> 3+5)`**：
    - **单机 `1×8`**：图像均值 **`690.4 ms`**（Wave 3 仅 **`585.2 ms`**，比 `B3-Opt` 快 **`1.73×`**），视频延迟 **`5,525.5 ms`**（比 `B3-Opt` 快 **`1.34×`**，比 `B8 GangStep` 快 **`2.03×`**），**100% 总 SLO**！
    - **双机 `2×4`**：图像均值 **`843.6 ms`**（Wave 3 仅 **`580.7 ms`**），视频延迟 **`6,826.5 ms`**（比 `B8 GangStep` 快 **`2.31×`**，比 `B9 Joint-v4` 快 **`1.51×`**），**100% 总 SLO**！
- **双视频并发争用实测（2 个 `Wan2.2-TI2V-5B` 视频在 `t=0ms` 与 `t=500ms` 到达）**：
  - `B3-Opt (Static-Partition-4v4i)`：由于视频被死锁在 4 张卡上，第 2 个视频 `v1` 必须串行排队等待 `v0` 完成全部 20 步 + 同步 VAE，导致 `v1` 延迟飙升至 **`14,310.6 ms`**（平均视频延迟 `10,858.0 ms`），违反 13.5s SLO！
  - **`Ours-2 (HeteroDiT-Morphing-v5)`**：当 `v0` 在第 6 步从 6 卡 `cfg2u3r1` 蜕变为 3 卡 `u3r1`（Ranks 0..2）时，释放出的 Ranks 3..5 立即组成第二个独立的 3 卡 `u3r1` 子集群并行执行 `v1`！实测 `v0 = 5,219.2 ms`、**`v1 = 6,241.5 ms`**（平均视频延迟 **`5,730.3 ms`**），**第 2 个视频比 `Static-4v4i` 快 `2.29×`，平均视频延迟快 `1.89×`**！
