# HeteroDiT v3 Full-Stack Optimization Report (Opts #9-#12 on Real A100 GPUs)

- **Device**: `NVIDIA A100-PCIE-40GB`

## 1. Opt #9: Asymmetric Late-Step CFG Reuse + First-Order Taylor DBCache Co-Execution (`Wan2.1-T2V-1.3B`)

| Execution Mode | Threshold ($\tau$) | Reused Cache Steps | Folded CFG Steps | Effective Branch Passes | Denoise Latency (ms) | Speedup | Latent `cos_sim` | Video PSNR (dB) | Video SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Exact-FullCFG (Baseline)` | `0.00` | 0/20 | 0/20 | 40/40 | **11015.4** | **1.00x** | 1.000000 | **99.99** | **1.0000** |
| `AsymmetricCFG-Only (Fold Steps 14..18)` | `0.00` | 0/20 | 5/20 | 35/40 | **9702.3** | **1.14x** | 0.997919 | **35.86** | **0.9710** |
| `Taylor1stOrder + AsymmetricCFG (Fold 17..18)` | `0.12` | 4/20 | 2/20 | 30/40 | **9067.9** | **1.22x** | 0.996577 | **33.38** | **0.9534** |
| `Taylor1stOrder + AsymmetricCFG (Fold 17..18)` | `0.16` | 4/20 | 2/20 | 30/40 | **9021.4** | **1.22x** | 0.996577 | **33.38** | **0.9534** |
| `Taylor1stOrder + AsymmetricCFG (Fold 14..18)` | `0.16` | 5/20 | 5/20 | 25/40 | **8271.2** | **1.33x** | 0.995956 | **32.73** | **0.9437** |
| `Taylor1stOrder + AsymmetricCFG (Fold 16..18)` | `0.24` | 5/20 | 3/20 | 27/40 | **8415.3** | **1.31x** | 0.994532 | **32.07** | **0.9348** |

## 2. Opt #10: `Wan2.1` `BF16` Tensor-Core Chunked 3D-VAE Decode with FP32 Latent Denormalization

| 3D-VAE Mode (5 Chunks, 17 Frames @ 832x480) | Decode Latency (ms) | Speedup vs FP32 | Saved Latency (ms) | Peak HBM (MB) | PSNR vs FP32 (dB) | SSIM vs FP32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FP32 Chunked 3D-VAE (Baseline)` | 1485.2 | 1.00x | 0.0 | 26892.1 | 99.99 | 1.00000 |
| **`BF16 Tensor-Core Chunked 3D-VAE (Opt #10)`** | **967.1** | **1.54x** | **518.0** | **22891.3** | **54.39** | **0.99889** |

- **BF16 3D-VAE + Stream-Isolated Overlap with `SD3.5` Denoising Wall Time**: **`1719.1 ms`** (Peak HBM `13893.3 MB`)

## 3. Opt #11: `SD3.5-Medium` `SDPA_FLASH` xDiT Engine + `channels_last` 2D-VAE + MM-DiT Block Cache

| Execution Mode (12 Steps, 512x512) | Batch Size | Cache | Total Latency (ms) | Per-Image Latency (ms/img) | Throughput (img/s) | Speedup vs Baseline | PSNR (dB) | SSIM |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SDPA_MATH + Contiguous-VAE (Baseline BS=1)` | 1 | `False` | 2332.7 | **2332.7** | **0.43** | **1.00x** | 99.99 | 1.0000 |
| `SDPA_FLASH + ChannelsLast-2DVAE (BS=1)` | 1 | `False` | 797.2 | **797.2** | **1.25** | **2.93x** | 47.05 | 0.9955 |
| `SDPA_FLASH + ChannelsLast-2DVAE + MM-DiT-DBCache (BS=1, tau=0.22)` | 1 | `True` | 796.1 | **796.1** | **1.26** | **2.93x** | 99.99 | 1.0000 |
| `MicroBatch-BS4 + SDPA_FLASH + ChannelsLast-2DVAE (4 Prompts)` | 4 | `False` | 2846.0 | **711.5** | **1.41** | **3.28x** | 99.99 | 1.0000 |
| `MicroBatch-BS4 + SDPA_FLASH + ChannelsLast-2DVAE + MM-DiT-DBCache (HeteroDiT-v3)` | 4 | `True` | 2845.3 | **711.3** | **1.41** | **3.28x** | 99.99 | 1.0000 |

## 4. End-to-End Mixed-Modality Trace Replay (`HeteroDiT-v3` on 1xA100)

- **`HeteroDiT-v3 (All 12 Optimizations: Taylor+CFGReuse + BF16-3DVAE + Flash-Cache-MicroBatch)`**: Image Mean **`2153.4 ms`**, Image P99 **`2698.2 ms`**, Video **`11493.8 ms`**, Total SLO Attainment **`100.0%`**, Peak HBM **`14139.8 MB`**

## 5. Opt #12: 8xA100 11-Policy & 6-Stage Orthogonal Ablation (`heterodit_v3` C++ Control Plane)

| Ablation Stage (Extreme HoL Contention: Gamma CV=3.0, Load=1.25) | Total SLO (%) | Image SLO (%) | Video SLO (%) | Image P99 (ms) | Video P99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A0: Baseline (Block-Preempt + 1D-SP)` | **77.78%** | 87.04% | 50.00% | 1888.3 | 35504.5 |
| `A1: + Tiered Memory (EncoderPool + Pinned Prefetch)` | **83.33%** | 91.98% | 57.41% | 1814.3 | 33909.8 |
| `A2: + 2D Hybrid USP (u2r2 / u4r2 / cfg2u2r2)` | **83.33%** | 90.12% | 62.96% | 1800.4 | 34246.8 |
| `A3: + 3-Stage Knapsack DP & Micro-Batch (HeteroDiT-Exact)` | **85.19%** | 93.83% | 59.26% | 1684.4 | 32140.3 |
| `A4: + Laxity-Gated Dynamic Cache (HeteroDiT-Approx)` | **89.81%** | 93.83% | 77.78% | 1666.3 | 19076.7 |
| `A5: + INT8-Ring + Taylor-CFG + BF16-VAE + Flash-Batch (HeteroDiT-v3)` | **99.07%** | 100.00% | 96.30% | 1333.1 | 17107.5 |
