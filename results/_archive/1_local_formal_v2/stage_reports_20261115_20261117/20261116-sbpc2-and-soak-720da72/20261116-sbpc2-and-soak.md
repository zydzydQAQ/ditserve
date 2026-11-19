# 20261116 — SBP+c2 全扫描矩阵（32 单元）、abl-no-cb 同步消融（2 单元）与 500 请求五阶段长稳实验归档（`720da72`）

- **提交与核心哈希**：`720da72` (`720da72e883a6995de79a56df971d0cdf15c538f`, `core_sha256 = 5fca839ebbb3e2df6c966a767dad7bfb638730ed01a9626dae27d63d4e3a9d44`)
- **容器镜像**：`192.168.1.114:30500/jjjkairos/ditserve@sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`
- **校准包（Calibration Bundle）**：`d20261106-formal-v2-wan22-b1-21a1ca3-r1` (`sha256: 3d838143675930045036f31ef7b9441751cb0cd2c00c0cdd1bfce3105f23e569`)
- **Kairos 16×A100 作业清单**：
  1. `d20261116-formal-v2-sbpc2-n0-8gpu-720da72-r1`（15 个正式评测切片 + `ditserve`, `sbp-esp-cfg2` 2×500 请求五阶段浸泡）
  2. `d20261116-formal-v2-sbpc2-n1-8gpu-720da72-r1`（19 个正式评测切片 + `tetriserve` 1×500 请求五阶段浸泡）

## 1. 跨 §5.2–§5.4 强化基线 `SBP+c2` (`sbp-esp-cfg2`) vs `DitServe` 汇总

- **§5.2 Modality Mix Scale**：`SBP+c2` 5 点平均 SAR 为 **`86.60%` (Poisson) / `79.80%` (Gamma)**，`DitServe` 达 **`91.20% / 90.40%`**。
- **§5.3 Rate Scale**：`SBP+c2` 8 点平均 SAR 为 **`81.75%` (Poisson) / `70.75%` (Gamma)**，`DitServe` 达 **`87.50% / 87.12%`**。
- **§5.4 SLO Scale**：`SBP+c2` 5 点平均 SAR 为 **`72.60%` (Poisson) / `65.00%` (Gamma)**，`DitServe` 达 **`90.60% / 87.60%`**。

## 2. §5.6 `abl-no-cb` 同步 `720da72` 复测结果

- **Poisson (`anchor-abl-nocb`)**：Total SAR **`94.0%`**（Image `100.0%` / Video `88.0%`，Image P95 `15.43 s`），对比 `DitServe` **`94.0%`**。
- **Gamma (`gamma2-abl-nocb`)**：Total SAR **`93.0%`**（Image `100.0%` / Video `86.0%`，Image P95 `15.63 s`），对比 `DitServe` **`92.0%`**。

## 3. §5.8 500-Request 五阶段非平稳浸泡与零漂移遥测

- **`TetriServe`**：500-Req Total SAR **`66.60%`**（Img `71.59%` / Vid `61.02%`，Img P95 `1385.16 s`）；`P3` 恢复 SAR `74.0%`、`P5` `68.0%`；漂移斜率 HBM `+835.46 MB/hr`、SHM `-5.08 MB/hr`、RSS `+3424.48 MB/hr`。
- **`SBP+c2`**：500-Req Total SAR **`71.60%`**（Img `79.92%` / Vid `62.29%`，Img P95 `20.98 s`）；`P3` 恢复 SAR `79.0%`、`P5` `76.0%`；漂移斜率 HBM `+1080.94 MB/hr`、SHM `-3.99 MB/hr`、RSS `+4816.02 MB/hr`。
- **`DitServe`**：500-Req Total SAR **`85.40%`**（Img `98.11%` / Vid `71.19%`，Img P95 `18.61 s`）；`P3` 恢复 SAR `97.0%`、`P5` `92.0%`；漂移斜率 HBM `+1166.24 MB/hr`、SHM `-3.49 MB/hr`、RSS `+3584.16 MB/hr`。
