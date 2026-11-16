# §5.4 SLO Scale

- **对应论文小节**：`body/05-evaluation.tex` — `\subsection{SLO Scale}`
- **对应论文图表**：`Figure 11` (`fig_eval_slo_scale.pdf`)
- **固定参数**：8× A100-PCIE-40GB，$R = 12\text{ req/min}$，$v = 0.5$，扫描截止期倍数 $\kappa \in \{1.0\times, 1.25\times, 1.5\times, 1.75\times, 2.0\times\}$（每点 $N=100$ 请求）
- **对比策略（6 策略）**：`xDiT`、`TetriServe`、`EDF-BestFit`、`SBP-ESP`、`SBP+c2` (`sbp-esp-cfg2`)、**`DitServe`**
- **构建版本**：`720da72`

---

## 1. 核心实验结论

1. **全 10 个截止期区间全面大幅领先（含强化基线 `SBP+c2`，Fig. 11）**：
   - **Poisson ($CV=1.0$)**：`DitServe` 在 $\kappa \in \{1.0\times..2.0\times\}$ 取得 **69.0%, 92.0%, 96.0%, 97.0%, 99.0% SAR**（5 点均值 **90.6%**），显著超越 `SBP+c2` (72.6%)、`SBP-ESP` (66.2%)、`EDF-BestFit` (65.2%)、`TetriServe` (63.2%) 与 `xDiT` (45.4%)。
   - **Gamma ($CV=2.0$)**：`DitServe` 取得 **60.0%, 91.0%, 94.0%, 95.0%, 98.0% SAR**（5 点均值 **87.6%**），超越 `SBP+c2` (65.0%)、`TetriServe` (64.6%)、`SBP-ESP` (62.8%)、`EDF-BestFit` (60.0%) 与 `xDiT` (31.4%)。
2. **极限零松弛 $\kappa = 1.0\times$ 与紧截止期 $\kappa = 1.25\times$ 显著增益**：
   - 在 $\kappa = 1.0\times$ 下，纯步级 SP 基线崩溃至 **1.0%–5.0% SAR**（图像 **0.0%**），即便解锁 CFG2 的 `SBP+c2` 受限于整步轮次等待也仅达 **16.0% / 10.0% SAR**（图像 **0.0% / 0.0%**）；`DitServe` 结合 `42.2–63.8 ms` 门铃、`shrink_decode`、FlashPS 三流异步解码与 `flash_attn_varlen` 连续批处理，将图像 SAR 拉升至 **54.0% / 44.0%**，总 SAR 跃升至 **69.0% / 60.0%**，每按时请求 GPU 成本降至 **111.44 / 128.67 GPU·s**。

---

## 2. SLO Scale SAR (%) 汇总表（对应 `Fig. 11`）

### 2.1 Poisson Arrivals ($CV = 1.0$)

| 策略 (Policy) | $\kappa=1.0\times$ | $\kappa=1.25\times$ | $\kappa=1.5\times$ | $\kappa=1.75\times$ | $\kappa=2.0\times$ | **5-Scale Mean SAR (%)** | $\kappa=1.0\times$ GPU-s/Ontime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `xDiT` | 3.0 | 51.0 | 53.0 | 56.0 | 64.0 | 45.4 | 3,188.43 |
| `TetriServe` | 5.0 | 59.0 | 78.0 | 83.0 | 91.0 | 63.2 | 1,564.64 |
| `EDF-BestFit` | 2.0 | 60.0 | 80.0 | 88.0 | 96.0 | 65.2 | 3,874.94 |
| `SBP-ESP` | 4.0 | 64.0 | 79.0 | 87.0 | 97.0 | 66.2 | 1,995.66 |
| `SBP+c2` | 16.0 | 67.0 | 90.0 | 95.0 | 95.0 | 72.6 | 461.80 |
| **`DitServe`** | **69.0** | **92.0** | **96.0** | **97.0** | **99.0** | **90.6** | **111.44** |

### 2.2 Gamma Bursty Arrivals ($CV = 2.0$)

| 策略 (Policy) | $\kappa=1.0\times$ | $\kappa=1.25\times$ | $\kappa=1.5\times$ | $\kappa=1.75\times$ | $\kappa=2.0\times$ | **5-Scale Mean SAR (%)** | $\kappa=1.0\times$ GPU-s/Ontime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `xDiT` | 1.0 | 34.0 | 37.0 | 40.0 | 45.0 | 31.4 | 9,263.40 |
| `TetriServe` | 4.0 | 60.0 | 77.0 | 89.0 | 93.0 | 64.6 | 1,916.85 |
| `EDF-BestFit` | 1.0 | 45.0 | 73.0 | 85.0 | 96.0 | 60.0 | 7,650.47 |
| `SBP-ESP` | 5.0 | 59.0 | 74.0 | 85.0 | 91.0 | 62.8 | 1,502.95 |
| `SBP+c2` | 10.0 | 51.0 | 82.0 | 87.0 | 95.0 | 65.0 | 716.65 |
| **`DitServe`** | **60.0** | **91.0** | **94.0** | **95.0** | **98.0** | **87.6** | **128.67** |

---

## 3. 目录文件说明

- `summary.json`：5 档截止期倍数 × 2 种分布 × 6 策略的完整紧凑指标与分位数汇总
- `slo_scale_metrics.csv`：60 行完整明细表（`2 分布 × 5 档 SLO × 6 策略`）
