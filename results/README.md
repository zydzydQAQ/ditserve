# DitServe 论文实验结果归档总览（按论文当前小章节命名组织）

本目录汇总整理了 `DitServe` 论文当前全部实验与硬件特征化测量结果（100% 对齐论文最新源码 `ditserve-writing/body/03-motivation.tex` 与 `body/05-evaluation.tex` 及最新修复提交 `8c50c7a`）。所有子目录严格按照论文当前小章节（`§3.1–§3.3` 与 `§5.1–§5.7`）命名，每个子目录均包含结构化报告 `README.md`、紧凑指标摘要 `summary.json` 以及可直接用于核验与绘图的 `.csv` 数据表。

---

## 1. 测试平台与版本溯源

- **硬件集群**：`jjjkairos` 集群，单节点 8× `NVIDIA A100-PCIE-40GB`（双路 PCIe 4.0 Host Bridge `PHB`，无 NVLink/NVSwitch；§5.5 扩展至 2×8 = 16 卡）
- **验证镜像**：`192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3`（`sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`）
- **代码版本**：`ditserve` (`8a100` 分支) 最新提交 **`8c50c7a`**（弱扩展提交 `ae4ba3d`；无损复用的非 CB 基线提交 `1b54f37`, `43f1ca3`, `7a77c45`, `5d9ee3f`）
- **校准包（Calibration Bundle）**：`d20261106-formal-v2-wan22-b1-21a1ca3-r1`（`sha256: 3d838143675930045036f31ef7b9441751cb0cd2c00c0cdd1bfce3105f23e569`）

---

## 2. 论文小章节与子目录映射总表

| 论文小节 | 论文英文标题 (`\subsection{...}`) | 对应图表 | 子目录链接 | 核心实验结论摘要 |
| :---: | :--- | :--- | :--- | :--- |
| **§3.1** | `Characterization I: Modality-Asymmetric SP and CFG Scaling on PCIe` | `Fig. 3`, `Fig. 4` | [`3.1_Characterization_I_Modality-Asymmetric_SP_and_CFG_Scaling_on_PCIe/`](./3.1_Characterization_I_Modality-Asymmetric_SP_and_CFG_Scaling_on_PCIe/) | 纯 SP 在图像上 `0.42×–0.77×` 负加速，视频 `42%–72%` 通信开销；CFG2 在视频达 `1.88×–1.91×` E2E (`<0.7%` 通信)，相变点 $L^* \approx 25\text{K}–38.1\text{K}$ |
| **§3.2** | `Characterization II: Step vs. Block Timelines and Modality-Asymmetric Preemption` | `Fig. 5` | [`3.2_Characterization_II_Step_vs_Block_Timelines_and_Modality-Asymmetric_Preemption/`](./3.2_Characterization_II_Step_vs_Block_Timelines_and_Modality-Asymmetric_Preemption/) | 视频单 Block (`45–94 ms`) 原地暂停 `63.75 ms`，紧 SLO 下等待缩短 **`26.77×`**（`0%` $\to$ **`100%` SAR**，`624/624` SHA256 一致） |
| **§3.3** | `Characterization III: Cross-Modal HBM Wall, Prompt Padding, and Batch Waste` | `Table 2` | [`3.3_Characterization_III_Cross-Modal_HBM_Wall_Prompt_Padding_and_Batch_Waste/`](./3.3_Characterization_III_Cross-Modal_HBM_Wall_Prompt_Padding_and_Batch_Waste/) | S4 解耦每卡省 **`19.45 GiB`**（`0 OOM`），UMT5 截断提速 `5.87×–15.58×`，`cu_seqlens` 打包提速 **`1.68×–2.02×`** |
| **§5.1** | `Experimental Setup` | `Table 1`, 42-Trace Matrix | [`5.1_Experimental_Setup/`](./5.1_Experimental_Setup/) | 3 大 DiT 模型参数、校准包 6 档分辨率实测耗时与 42 组评测工作负载清单 |
| **§5.2** | `Modality Mix Scale` | `Fig. 7`, `Fig. 8` | [`5.2_Modality_Mix_Scale/`](./5.2_Modality_Mix_Scale/) | $v \in \{0.1..0.9\}$ 平均 SAR **`91.4%` (Poisson) / `89.8%` (Gamma)**；均衡锚点归一化 P95 **`0.95× / 1.16×`** |
| **§5.3** | `Rate Scale` | `Fig. 9`, `Fig. 10` | [`5.3_Rate_Scale/`](./5.3_Rate_Scale/) | $R \in \{8..36\}$ rpm 平均 SAR **`87.2% / 87.3%`**；重过载下图像 P95 锁在 **`14.77–16.23 s`**（基线高达 `364.89 s`） |
| **§5.4** | `SLO Scale` | `Fig. 11` | [`5.4_SLO_Scale/`](./5.4_SLO_Scale/) | $\kappa \in \{1.0\times..2.0\times\}$ 平均 SAR **`89.6% / 86.4%`**；$\kappa=1.0\times$ 达 **`66.0% / 56.0%`**（基线 `1%–5%`） |
| **§5.5** | `Scalability` | `Fig. 12` | [`5.5_Scalability/`](./5.5_Scalability/) | $2 \to 16$ 卡弱扩展合并 SAR **`93.9% / 93.6%`**，16 卡 Goodput 达 2 卡的 **`8.08×` / `8.84×`** |
| **§5.6** | `Ablation Study` | `Fig. 13` | [`5.6_Ablation_Study/`](./5.6_Ablation_Study/) | 阶梯 `74%→87%→92%→94%` (Poisson) 与 `64%→78%→93%→94%` (Gamma)；L1 3 参数模型 **`0.0%` 遗憾** |
| **§5.7** | `Hardware and Runtime Subsystem Microbenchmarks` | `Table 3` | [`5.7_Hardware_and_Runtime_Subsystem_Microbenchmarks/`](./5.7_Hardware_and_Runtime_Subsystem_Microbenchmarks/) | GreenPool 降抖动 **`56.1×`**、GFC 建组 **`>1.4M×`**、分块 VAE **`22.7×`**、PinnedSlab **`5.01×`**、Approx **`2.3–5.4×`** |

---

## 3. 文件组织规范

每个小章节子目录下统一包含：
1. `README.md`：本小节实验设置、完整对比表格、关键机制归因与结论摘要；
2. `summary.json`：紧凑结构化 JSON 摘要（包含标量指标、分位数、曲线网格及 `run_id` / `config_sha256` 溯源哈希，遵循项目规范不包含数 MB 逐请求原始序列）；
3. `*.csv`：可直接导入 Python / OriginPro / Excel 复核与绘图的干净表格数据。

---

## 4. 全量离线补充归档（[`_archive/`](./_archive/)）

为支持无 GPU 平台离线写作、后续改稿与审稿回复（Rebuttal），[`_archive/`](./_archive/) 目录已将本地 `.local/` 与远端 `jjjkairos` 集群上的全部底层实验凭据、补充套件（含 `Wan2.1-T2V-1.3B` Suite 2、xDiT/TetriServe 对齐与质量门禁）以及 70 个正式实验切片的**逐请求调度与抢占时间线明细**（`.ledger.json.gz`）汇总为自包含归档（共 `2,619` 个文件，`157.01 MB`）。详见 [`_archive/README.md`](./_archive/README.md) 与 [`_archive/ARCHIVE_MANIFEST.json`](./_archive/ARCHIVE_MANIFEST.json)。
