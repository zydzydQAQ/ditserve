# DitServe 论文实验结果归档总览（按论文当前小章节命名组织）

本目录汇总整理了 `DitServe` 论文全部实验与硬件特征化测量结果（对齐论文源码 `ditserve-writing/body/03-motivation.tex` 与 `body/05-evaluation.tex` 及最新提交 `720da72` / `3878f01`）。所有子目录严格按照论文小章节（`§3.1–§3.3` 与 `§5.1–§5.9`）命名，每个子目录均包含结构化报告 `README.md`（或日期报告）、紧凑指标摘要 `summary.json` 以及可直接用于核验与绘图的 `.csv` 数据表。

---

## 1. 测试平台与版本溯源

- **硬件集群**：`jjjkairos` 集群，单节点 8× `NVIDIA A100-PCIE-40GB`（双路 PCIe 4.0 Host Bridge `PHB`，无 NVLink/NVSwitch；§5.5 扩展至 2×8 = 16 卡）
- **验证镜像**：`192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3`（`sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`）
- **代码版本**：`ditserve` (`8a100` 分支) 最新提交 **`720da72`**（含 42 组主策略轨迹、32 组 `SBP+c2` 跨全扫描矩阵、`abl-no-cb` 同步消融、3×500 请求五阶段长稳浸泡实验及 36 组跨并行度 Step/Block `CV` 特征化矩阵；无损复用的非 CB 基线提交 `1b54f37`, `43f1ca3`, `7a77c45`, `5d9ee3f`, `8c50c7a`）
- **校准包（Calibration Bundle）**：`d20261106-formal-v2-wan22-b1-21a1ca3-r1`（`sha256: 3d838143675930045036f31ef7b9441751cb0cd2c00c0cdd1bfce3105f23e569`）

---

## 2. 论文小章节与子目录映射总表

| 论文小节 | 论文英文标题 (`\subsection{...}`) | 对应图表 | 子目录链接 | 核心实验结论摘要 |
| :---: | :--- | :--- | :--- | :--- |
| **§3.1** | `Characterization I: Modality-Asymmetric SP and CFG Scaling on PCIe` | `Fig. 3`, `Fig. 4`, `Table 1–2 CV` | [`3.1_Characterization_I_Modality-Asymmetric_SP_and_CFG_Scaling_on_PCIe/`](./3.1_Characterization_I_Modality-Asymmetric_SP_and_CFG_Scaling_on_PCIe/) | 纯 SP 在图像上 `0.42×–0.77×` 负加速，CFG2 在视频达 `1.88×–1.91×` E2E；全矩阵 36 组配置 Step `CV` **`0.05%–3.50%`**（`P99/P50` $\le 1.09$） |
| **§3.2** | `Characterization II: Step vs. Block Timelines and Modality-Asymmetric Preemption` | `Fig. 5`, `Block CV` | [`3.2_Characterization_II_Step_vs_Block_Timelines_and_Modality-Asymmetric_Preemption/`](./3.2_Characterization_II_Step_vs_Block_Timelines_and_Modality-Asymmetric_Preemption/) | 视频 30 层同构 Block 空间 `CV` **`0.08%–3.85%`**、时间 `CV` **`0.33%–4.52%`**；紧 SLO 原地暂停等待缩短 **`26.77×`**（`0%` $\to$ **`100%` SAR**） |
| **§3.3** | `Characterization III: Cross-Modal HBM Wall, Prompt Padding, and Batch Waste` | `Table 2` | [`3.3_Characterization_III_Cross-Modal_HBM_Wall_Prompt_Padding_and_Batch_Waste/`](./3.3_Characterization_III_Cross-Modal_HBM_Wall_Prompt_Padding_and_Batch_Waste/) | S4 解耦省 **`19.45 GiB/GPU`**（`0 OOM`），T5 正负分离缓存 **`3.96×` 冷 / `80,927×` 热**，单核 `flash_attn_varlen` **`1.71×–2.02×`** + `4.5×` 跨度门禁 |
| **§5.1** | `Experimental Setup` | `Table 1`, 42-Trace Matrix | [`5.1_Experimental_Setup/`](./5.1_Experimental_Setup/) | 3 大 DiT 模型参数、校准包 6 档分辨率实测耗时与 42 组评测工作负载清单（`720da72`） |
| **§5.2** | `Modality Mix Scale` | `Fig. 7`, `Fig. 8` | [`5.2_Modality_Mix_Scale/`](./5.2_Modality_Mix_Scale/) | 6 策略对比：$v \in \{0.1..0.9\}$ 平均 SAR **`91.2% / 90.4%`**（`SBP+c2` `86.6% / 79.8%`）；图像 P95 **`14.59 / 15.07 s`** |
| **§5.3** | `Rate Scale` | `Fig. 9`, `Fig. 10` | [`5.3_Rate_Scale/`](./5.3_Rate_Scale/) | 6 策略对比：$R \in \{8..36\}$ rpm 平均 SAR **`87.5% / 87.1%`**（`SBP+c2` `81.8% / 70.8%`）；图像 P95 锁在 **`14.50–16.60 s`** |
| **§5.4** | `SLO Scale` | `Fig. 11` | [`5.4_SLO_Scale/`](./5.4_SLO_Scale/) | 6 策略对比：$\kappa \in \{1.0\times..2.0\times\}$ 平均 SAR **`90.6% / 87.6%`**（`SBP+c2` `72.6% / 65.0%`）；$\kappa=1.0\times$ 达 **`69.0% / 60.0%`** |
| **§5.5** | `Scalability` | `Fig. 12` | [`5.5_Scalability/`](./5.5_Scalability/) | $2 \to 16$ 卡弱扩展合并 SAR **`94.7% / 93.1%`**（Poisson 全规模 100% 图像 SAR），Goodput 扩展 **`7.49×` / `8.47×`** |
| **§5.6** | `Ablation Study` | `Fig. 15` | [`5.6_Ablation_Study/`](./5.6_Ablation_Study/) | 5 级累加阶梯 Poisson **`25%→51%→58%→63%→77%`**（Goodput `3.44×`），Gamma **`22%→32%→46%→58%→69%`**（Goodput `3.96×`） |
| **§5.7** | `Hardware and Runtime Subsystem Microbenchmarks` | `Table 3` | [`5.7_Hardware_and_Runtime_Subsystem_Microbenchmarks/`](./5.7_Hardware_and_Runtime_Subsystem_Microbenchmarks/) | GreenPool **`56.1×`**、GFC **`>1.4M×`**、FlashPS 三流 **`5.12×`**、分块 VAE **`22.7×`**、PinnedSlab **`5.01×`** |
| **§5.8** | `Sustained Multi-Phase Stability` | `Fig. 14` (Soak) | [`5.8_Sustained_Multi-Phase_Stability/`](./5.8_Sustained_Multi-Phase_Stability/) | 500 请求五阶段潮汐浸泡：**`85.4%` SAR** vs `SBP+c2` `71.6%` / `TetriServe` `66.6%`；`0 OOM`，HBM 斜率 **`+1166.24 MB/hr`** |
| **§5.9** | `Hardware Generalizability and ISCA27 Supplements` | `Fig. 17`, Multi-Seed, `121f` Elastic | [`5.9_Hardware_Generalizability_and_ISCA27_Supplements/`](./5.9_Hardware_Generalizability_and_ISCA27_Supplements/) | 3 种子 (`301/302/303`) 均第一（**`92.7±4.2%` / `91.7±4.0%`**）、`xDiT-c2` / `Static-Part` 对比、121 帧 720p (`27.3K > L^*`) 在线 `C2S2` 弹性与三维硬件外推 |

---

## 3. 文件组织规范

每个小章节子目录下统一包含：
1. `README.md`：本小节实验设置、完整对比表格、关键机制归因与结论摘要；
2. `summary.json`：紧凑结构化 JSON 摘要（包含标量指标、分位数、曲线网格及 `run_id` / `config_sha256` 溯源哈希，遵循项目规范不包含数 MB 逐请求原始序列）；
3. `*.csv`：可直接导入 Python / OriginPro / Excel 复核与绘图的干净表格数据。

---

## 4. 全量离线补充归档（[`_archive/`](./_archive/)）

为支持无 GPU 平台离线写作、后续改稿与审稿回复（Rebuttal），[`_archive/`](./_archive/) 目录已将本地 `.local/` 与远端 `jjjkairos` 集群上截至 `2026-11-17 08:46` 的全部底层实验凭据、补充套件（含 `Wan2.1-T2V-1.3B` Suite 2、xDiT/TetriServe 对齐、质量门禁、`720da72` 全矩阵/`SBP+c2`/Soak、`3878f01` 5 级阶梯与 ISCA'27 补充评测）以及 **167 个**正式实验切片的**逐请求调度与抢占时间线明细**（`.ledger.json.gz`）汇总为自包含归档（共 `3,240` 个文件，`210.04 MB`）。详见 [`_archive/README.md`](./_archive/README.md) 与 [`_archive/ARCHIVE_MANIFEST.json`](./_archive/ARCHIVE_MANIFEST.json)；集群环境复原请见仓库根目录 [`MIGRATION.md`](../MIGRATION.md)。
