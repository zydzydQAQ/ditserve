# HeteroDiT v2 Full-Stack Optimization Report (Real A100 GPU Measurements)

- **Device**: `NVIDIA A100-PCIE-40GB`

## 1. Opt #1: Cross-Node / Cross-Socket Asymmetric INT8 Ring KV Compression

| 2D-USP Config | KV Tensor Shape | FP16 Size (KB) | INT8 Packet (KB) | Compression | Codec Overhead (ms) | SDPA Output `cos_sim` | SQNR (dB) | Cross-Node Comm Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `8GPU_u4r2_17f` | `[2, 1, 3, 1560, 128]` | 2340.0 | 1188.3 | **1.97x** (`-49.2%`) | 0.283 | **0.999957** | 40.65 | **1.85x** |
| `8GPU_u4r2_33f` | `[2, 1, 3, 3120, 128]` | 4680.0 | 2376.6 | **1.97x** (`-49.2%`) | 0.288 | **0.999959** | 40.84 | **1.91x** |
| `4GPU_u2r2_33f` | `[2, 1, 6, 3120, 128]` | 9360.0 | 4753.1 | **1.97x** (`-49.2%`) | 0.421 | **0.999956** | 40.56 | **1.92x** |

## 2. Opt #2: `SD3.5-Medium` `SDPA_FLASH` + True Multi-Prompt Heterogeneous Micro-Batching

| Execution Mode (4 Distinct Prompts, 12 Steps) | Batch Size | Attention | Total 4-Image Latency (ms) | Per-Image Latency (ms/img) | Throughput (img/s) | Speedup vs Seq-Math |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `Sequential-4xB1-SDPA_MATH` | 1 | `math` | 9310.5 | **2327.6** | 0.43 | **1.00x** |
| `Sequential-4xB1-SDPA_FLASH` | 1 | `flash` | 3064.5 | **766.1** | 1.30 | **3.04x** |
| `MicroBatch-2xB2-SDPA_FLASH` | 2 | `flash` | 2821.3 | **705.3** | 1.42 | **3.30x** |
| `MicroBatch-1xB4-SDPA_FLASH` | 4 | `flash` | 2707.3 | **676.8** | 1.48 | **3.44x** |

## 3. Opt #3: First-Order Taylor Velocity-Compensated Residual Extrapolation vs Zero-Order DBCache

| Cache Mode | Threshold ($\tau$) | Reused Steps | Denoise Latency (ms) | Speedup | Latent `cos_sim` | Video PSNR (dB) | Video SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Exact-NoCache` | `0.00` | 0/20 (`0%`) | 10751.7 | 1.00x | 1.000000 | **99.99** | **1.0000** |
| `ZeroOrder-DBCache` | `0.12` | 4/20 (`20%`) | 9280.1 | 1.16x | 0.997322 | **34.15** | **0.9616** |
| `Taylor1stOrder-DBCache` | `0.12` | 4/20 (`20%`) | 9390.6 | 1.15x | 0.997853 | **34.95** | **0.9679** |
| `ZeroOrder-DBCache` | `0.16` | 5/20 (`25%`) | 8986.0 | 1.20x | 0.997227 | **34.16** | **0.9616** |
| `Taylor1stOrder-DBCache` | `0.16` | 5/20 (`25%`) | 9036.6 | 1.19x | 0.997795 | **35.02** | **0.9686** |
| `ZeroOrder-DBCache` | `0.24` | 7/20 (`35%`) | 8236.7 | 1.30x | 0.993973 | **31.91** | **0.9286** |
| `Taylor1stOrder-DBCache` | `0.24` | 7/20 (`35%`) | 8287.5 | 1.30x | 0.994729 | **32.37** | **0.9358** |

## 4. Opt #4: Stream-Isolated 3D-VAE Chunked Decode Overlapping with `SD3.5` Denoising

- **Sequential (`3D-VAE Decode` + `SD3.5 Denoise`)**: `2219.6 ms` (`VAE=1457.1 ms` + `SD3.5=762.6 ms`), Peak HBM `17804.6 MB`
- **Stream-Isolated Concurrent Overlap**: **`2125.2 ms`** (Saved **`94.5 ms`**, **`1.04x`** speedup), Peak HBM `17827.5 MB`, Output Max Diff `3.68e-03`

## 5. End-to-End Mixed-Modality Trace Replay (`HeteroDiT-v2`)

- **`HeteroDiT-v2 (Flash + MicroBatch + TaylorCache + StreamVAE)`**: Image Mean **`2030.7 ms`**, Image P99 **`2519.4 ms`**, Video **`12665.2 ms`**, Total SLO Attainment **`100.0%`**, Peak HBM **`18129.6 MB`**
