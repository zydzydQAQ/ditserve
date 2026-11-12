# HeteroDiT Real-GPU Quality-Latency Pareto & Mixed-Workload Replay Report

- **GPU**: `NVIDIA A100-PCIE-40GB`

## 1. `Wan2.1-T2V-1.3B` (20 Steps, `480x832x17`) Real Quality-Latency Pareto Curve

| Mode / Threshold ($\tau$) | 20-Step Denoise (ms) | Speedup | Skipped CFG Passes | Latent Cosine Sim | Latent RMSE | Decoded Video PSNR (dB) | Decoded Video SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exact (Lossless)` | **10999.8** | **1.00×** | `0/40` (`0.0%`) | `1.000000` | `0.00000` | **99.99 dB** | **1.0000** |
| `DBCache (tau=0.06)` | **11041.3** | **1.00×** | `0/40` (`0.0%`) | `1.000003` | `0.00000` | **99.99 dB** | **1.0000** |
| `DBCache (tau=0.08)` | **11037.5** | **1.00×** | `0/40` (`0.0%`) | `1.000003` | `0.00000` | **99.99 dB** | **1.0000** |
| `DBCache (tau=0.12)` | **9475.9** | **1.16×** | `8/40` (`20.0%`) | `0.997873` | `0.06147` | **33.07 dB** | **0.9673** |
| `DBCache (tau=0.16)` | **9100.6** | **1.21×** | `10/40` (`25.0%`) | `0.997357` | `0.06864` | **32.47 dB** | **0.9601** |
| `DBCache (tau=0.24)` | **8319.5** | **1.32×** | `14/40` (`35.0%`) | `0.995504` | `0.08957` | **30.55 dB** | **0.9357** |

## 2. Real A100 End-to-End Mixed-Modality Trace Replay (`Wan2.1` + Bursty `SD3.5`)

| System | Image Mean Latency (ms) | Image P99 Latency (ms) | Video Latency (ms) | Makespan (ms) | Image SLO% | Video SLO% | Total SLO% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Vanilla-xDiT (Non-Preemptive)` | 28353.3 | **29724.7** | **24449.6** | 31524.7 | 0.0% | 0.0% | **0.0%** |
| `GenServe-40GB (Step-Preempt + Cold Swap)` | 22544.1 | **25096.1** | **31355.7** | 31355.7 | 0.0% | 0.0% | **0.0%** |
| `HeteroDiT-Exact (Ours, Lossless)` | 19042.7 | **20374.6** | **27105.8** | 27105.8 | 0.0% | 0.0% | **0.0%** |
| `HeteroDiT-Approx (Ours, Laxity Cache tau=0.12)` | 19900.7 | **21280.5** | **28039.9** | 28039.9 | 0.0% | 0.0% | **0.0%** |
