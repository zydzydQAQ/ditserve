# HeteroDiT-v5 (DitServe) 实验结果目录索引 (`results/`)

本目录存放 `HeteroDiT-v5` 在 `jjjkairos` 集群（`A100-PCIE-40GB`）上的全部实测 JSON 数据与图表文件。完整实验解读与汇总表格见 [`../docs/heterodit_v5_8gpu_experiments.md`](../docs/heterodit_v5_8gpu_experiments.md)。

## 最新核心结果文件（2026-10-31）

1. **8 卡单机 (`1×8` Intra-Node) 11 系统 Baseline 对比 + `A0..A7` 累加消融 + `C1..C5` 正交消融**：
   - `heterodit-v5-8gpu-ablation-1x8-20261031-r2/HETERODIT_8GPU_BASELINES_AND_ABLATION.json`
2. **8 卡跨节点 (`2×4` Cross-Node) 11 系统 Baseline 对比 + `A0..A7` 累加消融 + `C1..C5` 正交消融**：
   - `heterodit-v5-8gpu-ablation-2x4-20261031-r2/HETERODIT_8GPU_BASELINES_AND_ABLATION.json`
3. **8 卡单机 (`1×8`) 五项优化验证与 10 种并行矩阵 (`SP1..cfg2u4r1`)**：
   - `heterodit-v5-8gpu-1x8-20261031-r9/HETERODIT_8GPU_V5_SUITE.json`
4. **8 卡跨节点 (`2×4`) 五项优化验证与 10 种并行矩阵 (`SP1..cfg2u4r1`)**：
   - `heterodit-v5-8gpu-2x4-20261031-r3/HETERODIT_8GPU_V5_SUITE.json`
5. **4 卡单机 (`1×4`) 与跨节点 (`2×2`) 五项优化验证（`3+1` 非对称子集群 + INT8 Ring + Rank0 Async VAE）**：
   - `heterodit-v5-4gpu-1x4-20261031-r5/RING_INT8_AND_4GPU_REAL.json`
   - `heterodit-v5-4gpu-2x2-20261031-r5/RING_INT8_AND_4GPU_REAL.json`
6. **单卡 (`1×1`) 精简后完整回归验证套件**：
   - `heterodit-slim-1gpu-20261031-r3/REAL_VERIFICATION_SUITE.json`
7. **8 卡 `Wan2.2-TI2V-5B + SD3.5-Medium` 实机验证与 10 系统 Baseline + 双视频并发争用对比（单机 `1×8` 与跨节点 `2×4`）**：
   - `heterodit-wan22-8gpu-1x8-20261101-r2/WAN22_8GPU_VERIFICATION.json`
   - `heterodit-wan22-8gpu-1x8-20261101-r2/WAN22_C4_CURVATURE_SWEEP.json`
   - `heterodit-wan22-8gpu-2x4-20261101-r2/WAN22_8GPU_VERIFICATION.json`
8. **论文图表与 LaTeX 表格**：
   - `figures_and_tables/`
