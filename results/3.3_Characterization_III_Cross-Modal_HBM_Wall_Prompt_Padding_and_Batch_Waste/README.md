# §3.3 Characterization III: Cross-Modal HBM Wall, Prompt Padding, and Batch Waste

- **对应论文小节**：`body/03-motivation.tex` — `\subsection{Characterization III: Cross-Modal HBM Wall, Prompt Padding, and Batch Waste}`
- **对应论文图表**：`Table 2` (`tab:mot_memory`)
- **实测溯源**：`d20261104-mot-sys-2gpu-r1`、`cross_modal_memory_study.json`、`continuous_batch_study.json`

---

## 1. 核心实测结论（Insight 4）

1. **跨模态同驻 HBM 显存墙（Table 2）**：在 40 GB A100（可用 `39.50 GiB`）上同时驻留 `UMT5-XXL` (`10.58 GiB`) 与 `T5-XXL` (`8.87 GiB`) 会占去 **19.45 GiB/GPU**，叠加激活与缓存悬垂 (`0.58–2.53 GiB`) 后峰值达 **40.1–54.0 GiB（100% OOM）**；反应式权重换入换出 (`S1`) 每次停顿 **2.74–3.91 s**；朴素 512-token CPU 编码 (`S2/S3`) 在 17–24 token 提示词上浪费 **95.9%** 自注意力算力，每次阻塞 **14.8–15.2 s**。
2. **S4 编码器解耦 + UMT5 无损有效 Token 截断**：每卡节省 **19.45 GiB**（8 卡共省 **155.60 GiB**，**0 OOM**），CPU 编码提速 **5.87×–8.08×**（缓存池命中提速 **15.58×**）、GPU 提速 **5.52×–6.75×**，H2D DMA 载荷缩减 **15.5×**（仅 **0.10–0.40 ms**），暂停点 `empty_cache()` 在 **0.60–1.62 ms** 内回收 `0.58–2.30 GiB` 悬垂显存。
3. **图像静态 2D Padding 浪费 vs. `cu_seqlens` 连续批处理**：异构 $512^2/1024^2$ 静态 2D 补零浪费 **37.5%–50.0% pad tokens** 并暴露 **180.0 ms** 编解码气泡；步级连续批处理配合 FlashAttention-2 `cu_seqlens` 打包实现 **1.682×–2.020×** 单步加速与 **2.20×** 在线 Goodput（最大误差 `4.77e-7`）。

---

## 2. 跨模态内存策略对比表（对应论文 `Table 2: tab:mot_memory`）

| 策略 (Strategy) | 静态权重 (Static) | 峰值显存 (Peak HBM) | 权重换页 (Swap) | 编码/DMA 耗时 | 运行状态 (Status) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `S0: Naive GPU` | `28.6 – 36.4 GiB` | `40.1 – 54.0 GiB` | `0 ms` | `28 – 80 ms` | **OOM** (`> 39.5 GiB`) |
| `S1: Wt Swap` | `13.5 – 21.2 GiB` | `17.9 – 33.7 GiB` | `2.7 – 3.9 s` | `28 – 80 ms` | **0% SAR** (换页阻塞) |
| `S2/3: CPU Enc` | `9.2 – 25.8 GiB` | `17.1 – 42.5 GiB` | `0 ms` | `3.7 – 15.2 s` | **Stall** (95.9% Pad 浪费) |
| **`S4: DitServe`** | **`9.2 – 16.9 GiB`** | **`17.1 – 27.6 GiB`** | **`0 ms`** | **`0.1 – 0.8 ms`** | **0 OOM (`-19.45 GiB/GPU`)** |

---

## 3. `SD3.5-Medium` 变分辨率 `cu_seqlens` 打包实测表

| 异构批次构成 | Padded vs. Packed Tokens | 消除 Padding | Padded 耗时 (ms) | Packed 耗时 (ms) | **Packed vs. Padded 加速比** |
| :--- | :---: | :---: | ---: | ---: | :---: |
| `B=2` (`1×512² + 1×1024²`) | `8,192` vs. `5,120` | **37.5%** | `444.88` /步 | **`264.44` /步** | **`1.682×`** |
| `B=3` (`2×512² + 1×1024²`) | `12,288` vs. `6,144` | **50.0%** | `621.88` /步 (`6013.9` /图) | **`311.00` /步 (`2977.7` /图)** | **`2.000×` – `2.020×`** |
| `B=4` (`2×512² + 2×1024²`) | `16,384` vs. `10,240` | **37.5%** | `867.99` /步 | **`507.96` /步** | **`1.709×`** |
| `B=8` (`4×512² + 4×1024²`) | `32,768` vs. `20,480` | **37.5%** | `1,719.87` /步 | **`1,003.32` /步** | **`1.714×`** |

---

## 4. 目录文件说明

- `summary.json`：S0–S4 显存账本、UMT5 截断与 `cu_seqlens` 打包结构化数据
- `memory_strategies_s0_s4.csv`：论文 `Table 2` 跨模态内存策略对比表
- `sd35_cu_seqlens_batching.csv`：变分辨率无填充序列打包实测表
