# §5.1 Experimental Setup

- **对应论文小节**：`body/05-evaluation.tex` — `\subsection{Experimental Setup}`（引用 `Table 1: tab:bg_models`）
- **核心构建版本**：`8c50c7a`（最新连续批处理修复提交；弱扩展节点运行于 `ae4ba3d`；基线复用提交 `1b54f37, 5d9ee3f, 7a77c45, 43f1ca3`）
- **校准包（Calibration Bundle）**：`d20261106-formal-v2-wan22-b1-21a1ca3-r1`（`sha256: 3d838143675930045036f31ef7b9441751cb0cd2c00c0cdd1bfce3105f23e569`）

---

## 1. 硬件测试床与软件环境

| 项目 | 实测配置规格 |
| :--- | :--- |
| **GPU 硬件** | 8× NVIDIA A100-PCIE-40GB / 节点（40 GiB HBM2，可用 39.50 GiB，108 SMs/GPU；§5.5 扩展至 2×8 = 16 GPUs） |
| **互连拓扑** | 双路 CPU PCIe 4.0 Host Bridge (`PHB`)，无 NVLink / NVSwitch |
| **验证容器镜像** | `192.168.1.114:30500/jjjkairos/ditserve:external-serving-vllm-cu126-sm80-20261104-r3` |
| **镜像 Digest** | `sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05` |
| **深度学习栈** | CUDA `12.6`, Driver `570.133.20`, PyTorch `2.10.0+cu126`, Diffusers `0.36.0`, FlashAttention-2, xDiT `16c34e7f` |
| **提示词数据集** | VidProM（视频）与 DiffusionDB（图像），校准集与测试集提示词严格哈希分桶隔离 |

---

## 2. 模型参数与单步基准表（对应论文 `Table 1: tab:bg_models`）

| 属性 (Property) | `SD3.5-Medium` (T2I) | `Wan2.1-T2V-1.3B` (T2V) | `Wan2.2-TI2V-5B` (TI2V) |
| :--- | :---: | :---: | :---: |
| Backbone $(K, D, H)$ | `24, 1536, 24` | `30, 1536, 12` | `30, 3072, 24` |
| Text Encoder (BF16) | `10.39 GiB (68.5%)` | `10.58 GiB (78.5%)` | `10.58 GiB (49.9%)` |
| DiT + VAE (BF16) | `4.60 + 0.16 GiB` | `2.65 + 0.24 GiB` | `9.32 + 1.32 GiB` |
| Tokens $L$ / Branch | `1.0K / 4.1K` | `14.0K / 32.8K / 75.6K` | `5.8K / 13.9K / 27.3K–43.2K` |
| Steps $S$ (Blocks) | `28 (1-GPU 2B)` | `20 / 50 (60 blk)` | `20 / 50 (60 blk)` |
| SP Volume / Step | `0 MB (1-GPU)` | `10.4 / 24.2 / 55.7 GB` | `8.6 / 20.5 / 40.2 GB` |
| CFG2 Exchange | `Fused 2B` | `1.8 / 4.2 / 9.7 MB` | `2.2 / 5.3 / 10.5 MB` |
| SP1 Step (ms) | `59.2 / 221.4` | `1150 / 3965 / 16213` | `1061 / 3069 / 7813` |
| CFG2 Step (ms) | `1.72 / 6.51 s E2E` | `579 / 1998 / 8154` | `538 / 1545 / 3902` |
| Rank-0 VAE (s) | `0.16 s (2D)` | `1.83 / 6.91 / 10.8` | `3.75 / 9.54 / 19.3` |

---

## 3. 对比基线与 42 组负载矩阵概览

- **Track 1 主对比策略（5 策略，§5.2–§5.4）**：`xDiT`（静态分区）、`TetriServe`（ASPLOS'26 轮次弹性 SP 0/1 背包）、`EDF-BestFit`（GF-DiT 严格 EDF 最佳适配）、`SBP-ESP`（GenServe 步边界截止期背包 + 迟到降级）、**`DitServe`**（本文 L1+L2+L3 全开）。
- **Track 2 弱扩展与三层消融策略（§5.5–§5.6）**：`edf-cfg2`、`block-cfg2`、`abl-no-block`、`abl-first-fit`、`abl-no-prune`、`abl-no-cb`、**`DitServe`**。
- **42 组评测工作负载（Poisson $CV=1.0$ & Gamma $CV=2.0$）**：
  - **§5.2 Modality Mix Scale**：$v \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ @ $R=12\text{ rpm}, \kappa=1.5\times$（10 组，共享 $v=0.5$ 锚点）
  - **§5.3 Rate Scale**：$R \in \{8, 12, 16, 20, 24, 28, 32, 36\}\text{ rpm}$ @ $v=0.5, \kappa=1.5\times$（16 组，$1.06\times–4.77\times R_0$）
  - **§5.4 SLO Scale**：$\kappa \in \{1.0\times, 1.25\times, 1.5\times, 1.75\times, 2.0\times\}$ @ $R=12\text{ rpm}, v=0.5$（10 组）
  - **§5.5 Scalability**：$N_{\text{GPU}} \in \{2, 4, 8, 16\}$ @ $1.5\text{ rpm/GPU}$ ($R \in \{3, 6, 12, 24\}\text{ rpm}$, $N \in \{24, 50, 100, 200\}$)（8 组）
  - **§5.6 Ablation Study**：7 策略 @ $13.2\text{ rpm}$ ($\lambda=1.75 R_0, v=0.5, \kappa=1.5\times$) 及紧截止期 $\kappa=1.0\times$ CB 对照（4 组）

---

## 4. 目录文件说明

- `summary.json`：测试床、模型参数、校准包 UUID/SHA256 与策略配置总表
- `model_workload_parameters.csv`：对应 `Table 1` 的三大 DiT 模型规格表
- `calibration_classes_wan22.csv`：`Wan2.2 + SD3.5` 校准包全部 6 档分辨率实测耗时参数
- `workload_matrix_42traces.csv`：42 组正式评测 Trace 清单、运行 ID 与配置 SHA256
