# 20261116 — DitServe 16×A100 全量 42 单元正式评测矩阵刷新归档（`720da72`）

- **提交与核心哈希**：`720da72` (`720da72e883a6995de79a56df971d0cdf15c538f`, `core_sha256 = 5fca839ebbb3e2df6c966a767dad7bfb638730ed01a9626dae27d63d4e3a9d44`)
- **容器镜像**：`192.168.1.114:30500/jjjkairos/ditserve@sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`
- **Kairos 16×A100 作业清单（全部 `Succeeded`, `restarts: 0`）**：
  1. `d20261116-formal-v2-scale-n0-8gpu-720da72-r1`（`gpu06`, 8× A100）
  2. `d20261116-formal-v2-scale-n1-8gpu-720da72-r1`（`gpu05`, 8× A100）
  3. `d20261116-formal-v2-cb-ref-p-8gpu-720da72-r1`（`gpu06`, 8× A100）
  4. `d20261116-formal-v2-cb-ref-g2-8gpu-720da72-r1`（`gpu05`, 8× A100）

## 关键指标前后汇总（`8c50c7a` → `720da72`）

- **§5.2 Modality Mix Scale**：Poisson 均值 `91.4% -> 91.2%`（$v=0.1$ 从 `97.0%` 升至 **`99.0%`** 反超 `xDiT` `98.0%`），Gamma 均值 `89.8% -> 90.4%`（`+0.6 pp`），锚点图像 P95 降至 **`14.59 / 15.07 s`**。
- **§5.3 Rate Scale**：Poisson 均值 `87.25% -> 87.50%`（`+0.25 pp`），Gamma 均值 `87.25% -> 87.12%`，全 16 点图像 SAR 达 **`96.0%–100.0%`**，图像 P95 锁在 **`14.50–16.60 s`**。
- **§5.4 SLO Scale**：Poisson 均值 `89.6% -> 90.6%`（`+1.0 pp`），Gamma 均值 `86.4% -> 87.6%`（`+1.2 pp`）；严苛 $\kappa=1.0\times$ 从 `66.0% / 56.0%` 跃升至 **`69.0% / 60.0%`**（`+3.0 / +4.0 pp`）。
- **§5.5 Scalability**：Poisson 374 请求合并 SAR 从 `93.85%` 升至 **`94.65%`**（`2/4/8/16` 卡分别 `95.83% / 94.0% / 96.0% / 94.0%`）；Gamma 合并 SAR **`93.05%`**。
- **§5.6 Ablation Study**：Poisson 主策略 **`94.0%`**（留一法 `abl-no-block 84% / abl-first-fit 82% / abl-no-prune 82%`），Gamma 主策略 **`92.0%`**（留一法 `85% / 75% / 86%`）。
