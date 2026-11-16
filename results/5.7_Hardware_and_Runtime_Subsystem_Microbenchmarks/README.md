# §5.7 Hardware and Runtime Subsystem Microbenchmarks

- **对应论文小节**：`body/05-evaluation.tex` — `\subsection{Hardware and Runtime Subsystem Microbenchmarks}`
- **对应论文图表**：`Table 3` (`tab:eval_microbench`)
- **测试平台**：2× / 8× NVIDIA A100-PCIE-40GB GPUs（108 SMs/GPU，PCIe 4.0 PHB）
- **实测溯源**：`d20261104-mot-sys-2gpu-r1`、`d20261104-mot-alg-2gpu-r1`、`d20261104-gfc8-r3img-8ada0a3-r2`、`d20261115-val-img80-8gpu-01bee01-r1`

---

## 1. 硬件与运行时子系统微基准总表（对应论文 `Table 3: tab:eval_microbench`）

| # | 子系统机制 (Mechanism) | 基线表现 $\to$ `DitServe` 实测表现 | 加速收益 (Gain) | 数值/质量门禁 (Gate) | 实测作业 ID |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | **GreenPool SM Partitioning** | 抖动 $\text{P99}-\text{P50}$：`235.93 ms` $\to$ **`4.21 ms`** (`88:20` / `72:36` SM 硬件隔离) | **`56.06×`** (`-89.2%` P99) | `diff = 0` | `d20261104-mot-sys-2gpu-r1` |
| **2** | **GFC IPC Ring Collectives** | `ncclCommInitRank`：`3.64 s` (2G) / `28.80 s` (8G) $\to$ **`2.0–2.5 µs`** 掩码切换 | **`> 1.4×10^6×`** | `1,112/1,112` exact | `d20261104-gfc8-r3img-8ada0a3-r2` |
| **3** | **Varlen `cu_seqlens` + FlashPS 3-Stream** | 异构 Padding (`444.9 ms`) / 串行三阶段 (`19.86 s`) $\to$ **`260.1 ms` 单核融合 + `3.88 s` 三流流水 + `4.5×` 膨胀门禁** | **`1.71×–2.02×`** 步 / **`5.12×`** 流水 (`-172.4 ms` 气泡) | `cos = 0.999994`, 拦截 `512+1440` (`8.85×`) | `d20261115-val-img80-8gpu-01bee01-r1` |
| **4** | **Decode Shrink & Chunked VAE** | Rank-0 3D-VAE 阻塞等待：`6,912.0 ms` $\to$ **`304.1 ms`** 分帧抢占 + 伴随卡释放 | **`22.73×`** 等待缩短 | `diff = 0` | `d20261104-mot-sys-2gpu-r1` |
| **5** | **Lossless UMT5 Token Crop** | `189 ms` GPU / `5.1 s` CPU (512 pad) $\to$ **`28 ms` GPU / `0.33 s` CPU** | **`6.75×` GPU / `15.58×` CPU** | `max|Δ| = 0.0` | `d20261104-mot-sys-2gpu-r1` |
| **6** | **S4 Disaggregated Pool + Split T5** | 跨模态峰值 `40.1–54.0 GiB` (OOM) / T5 冷编 `54.79 s` $\to$ **`17.1–27.6 GiB` + 正负分离 `13.84 s` 冷 / `0.68 ms` 热** | **`-19.45 GiB/GPU`**, **`3.96×` 冷 / `80,927×` 热** | `0 OOM`, `max|Δ| = 0.0` | `d20261115-val-img80-8gpu-01bee01-r1` |
| **7** | **PinnedSlabPool & Cert. Omit** | 426.9 MB 卸载 P99：`209.07 ms` $\to$ **`41.69 ms`**（`certified_omit` `29.2 ms`） | **`5.01×`** (`7.31×` omit) | `diff = 0` | `d20261104-mot-sys-2gpu-r1` |
| **8** | **DitServe-Approx (CFG Fold)** | 首图延迟：`22.71 s` $\to$ **`4.18 s`** (`CFG2->SP1` 按需裂变，单步省 `55.2%–71.5%` GPU-ms) | **`2.27×–5.44×`** | `24.4–27.0 dB` PSNR (`0.96–0.97` SSIM) | `d20261104-mot-alg-2gpu-r1` |

---

## 2. 目录文件说明

- `summary.json`：8 项主表微基准、SD3.5 连续批处理/FlashPS 三流/分离式 T5 缓存实测及正交扩展指标汇总
- `subsystem_microbenchmarks.csv`：对应论文 `Table 3` 的结构化 CSV 数据表
