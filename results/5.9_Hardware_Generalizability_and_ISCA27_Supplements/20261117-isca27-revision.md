# 20261117-isca27-revision

本报告归档 `DitServe` 针对 ISCA 2027 评审意见补充的全部 8×A100 真机评测（`d20261117-rev-n0-8gpu-720da72-r1` 与 `d20261117-rev-n1-8gpu-720da72-r1`，共 7 个真机评测单元，全部通过 `numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu`）以及本地校准 C++ 模拟器三维硬件泛化与控制器超参数敏感性分析，配套紧凑摘要见 [summary.json](summary.json)。

## 1. 版本与环境溯源

- **代码提交**：`720da72e883a6995de79a56df971d0cdf15c538f`（`8a100` 分支）
- **运行镜像**：`192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3`（`sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`）
- **校准包复用**：`d20261106-formal-v2-wan22-b1-21a1ca3-r1/bundle.json`（`sha256:94c3238aee42925b57eb10dc1ad086c656db280cfba5c079b0a53d195aa62ff6`）及 121 帧 `Wan2.2-720p` 校准包（`d20261103-formal-cal8-wan22-51740c0-r1-cal1`）

## 2. 平衡锚点（$\lambda=12.0$ rpm, $v=0.5, \kappa=1.50$）静态池 CFG2 与 EDF 基线真机对比

| 分布 (`seed 301`) | 策略 | 总 SAR | 图像 SAR | 视频 SAR | 图像 P95 (s) | 视频 P95 (s) | 资格状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Poisson ($\mathrm{CV}=1.0$) | **DitServe** | **96.0%** | **100.0%** | **92.0%** | **14.59** | **573.92** | `verified_real_gpu` |
| Poisson ($\mathrm{CV}=1.0$) | `Static-Part` (`partition`, EDF+CFG2) | 84.0% | 86.0% | 82.0% | 16.12 | 638.77 | `verified_real_gpu` |
| Poisson ($\mathrm{CV}=1.0$) | `xDiT-c2` (`xdit-cfg2`, FCFS+CFG2) | 54.0% | 84.0% | 24.0% | 15.42 | 558.79 | `verified_real_gpu` |
| Gamma ($\mathrm{CV}=2.0$) | **DitServe** | **94.0%** | **100.0%** | **88.0%** | **15.07** | **623.15** | `verified_real_gpu` |
| Gamma ($\mathrm{CV}=2.0$) | `Static-Part` (`partition`, EDF+CFG2) | 71.0% | 56.0% | 86.0% | 24.24 | 908.80 | `verified_real_gpu` |
| Gamma ($\mathrm{CV}=2.0$) | `xDiT-c2` (`xdit-cfg2`, FCFS+CFG2) | 36.0% | 54.0% | 18.0% | 33.26 | 573.73 | `verified_real_gpu` |

## 3. 多随机种子（`seeds 301, 302, 303`，每策略 300 个真机请求）统计显著性验证

| 策略 | Poisson `301 / 302 / 303` | Poisson `Mean ± Std` | Gamma `301 / 302 / 303` | Gamma `Mean ± Std` |
| --- | :---: | :---: | :---: | :---: |
| **DitServe** | `96.0% / 88.0% / 94.0%` | **92.7 ± 4.2%** | `94.0% / 87.0% / 94.0%` | **91.7 ± 4.0%** |
| `SBP+c2` (GenServe + CFG2×SP) | `90.0% / 78.0% / 81.0%` | 83.0 ± 6.2% | `82.0% / 80.0% / 79.0%` | 80.3 ± 1.5% |
| `TetriServe` | `79.0% / 77.0% / 72.0%` | 76.0 ± 3.6% | `74.0% / 68.0% / 70.0%` | 70.7 ± 3.1% |
| `xDiT-c2` | `54.0% / 56.0% / 63.0%` | 57.7 ± 4.7% | `36.0% / 47.0% / 46.0%` | 43.0 ± 6.1% |

在全部 6 组独立流量轨迹中，`DitServe` 均位列第一（Poisson 图像 SAR 三组全为 `100.0%`，Gamma 图像 SAR 达 `99.3 ± 1.2%`）。

## 4. 超过 $L^*$ 序列长度（121 帧 `Wan2.2-720p`，`27.3K > 25.0K` tokens）在线 `CFG2×SP` 弹性验证

- **真机实测（`d20261117-rev-n0-8gpu-720da72-r1-121f-elastic`）**：121 帧 720p 单步耗时从 `SP1` 的 `7.85 s` 降至 `CFG2` 的 `3.90 s`、`C2S2` 的 `3.55–3.77 s` 以及 `C2S4` 的 `2.40 s`（`3.27×` 加速）。在线混合服务下，`DitServe` 将 `4/6`（`66.7%`）视频动态扩缩容至 `C2S2`，最快视频端到端延迟降至 `213.4 s`（突破固定双卡 `Blk+c2` 的单任务物理下界 `219.9 s`），同时保持 `100.0%` 图像 SAR（`8.03 s` P95；总 SAR `50.0%` vs. `SBP+c2` 的 `16.7%` 总 SAR / `33.3%` 图像 SAR / `199.27 s` 图像 P95）。
- **紧视频截止期（`0.49× T0 = 200.9 s, 0.8 rpm`）校准评估**：固定双卡 `Blk+c2` 因单视频耗时 $\ge 219.9\text{ s}$ 视频 SAR 为 `0.0%`（总 SAR `50.0%`），`SBP+c2` 因 4 步轮次锁卡阻塞图像仅达 `58.3%`，而 `DitServe` 通过在线 `CFG2 -> C2S3/C2S4` 扩宽结合单块抢占达 **91.7%** 总 SAR（图像 `100.0%`、视频 `83.3%`）。

## 5. 三维耦合硬件泛化（§6.9 / Fig. 17）与控制器超参数敏感性（§5）

- **耦合三维扩展定律**：当 Tensor Core 算力增速快于 PCIe 带宽（`A100 P4` $\to$ `H100 P5` $\to$ `B200 P6`），`C2->C2S2` 交叉点 $L^*$ 从 `25.0K` 上移至 `26.5K` 与 `29.8K` tokens（比一维仅缩放带宽外推高 `6.5×`）；仅 NVSwitch（`H200 NV4` / `B200 NV5`）将 $L^*$ 压低至 `4.6K` / `5.7K` tokens，且 `DitServe` 在 `H200 NV4` 与 `B200 NV5` 上仍保持 `+21.0%–31.7%` 同代 Goodput 增益（归一化 Goodput 达 `1.32×, 2.88×, 3.61×, 6.57×`）。
- **超参数鲁棒性**：在 `late_short_ms` (`0–4 s`)、`widen_efficiency` (`0.70–1.00`)、`max_step_dilation` (`2.5–8.0`)、`quantum_blocks` (`2–8`) 与 `late_deadband_ms` (`0–5 s`) 全范围扫描下，平衡锚点 SAR 波动均 $\le 2.0\text{ pp}$。
