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
