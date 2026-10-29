# HeteroDiT Real-GPU Quality-Latency Pareto & Mixed-Workload Replay Report

- **GPU**: `NVIDIA A100-PCIE-40GB`

## 1. `Wan2.1-T2V-1.3B` (20 Steps, `480x832x17`) Real Quality-Latency Pareto Curve

| Mode / Threshold ($\tau$) | 20-Step Denoise (ms) | Speedup | Skipped CFG Passes | Latent Cosine Sim | Latent RMSE | Decoded Video PSNR (dB) | Decoded Video SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exact (Lossless)` | **10756.2** | **1.00×** | `0/40` (`0.0%`) | `1.000000` | `0.00000` | **99.99 dB** | **1.0000** |
| `DBCache (tau=0.06)` | **10823.9** | **0.99×** | `0/40` (`0.0%`) | `1.000003` | `0.00000` | **99.99 dB** | **1.0000** |
| `DBCache (tau=0.08)` | **10906.5** | **0.99×** | `0/40` (`0.0%`) | `1.000003` | `0.00000` | **99.99 dB** | **1.0000** |
| `DBCache (tau=0.12)` | **9347.5** | **1.15×** | `8/40` (`20.0%`) | `0.997873` | `0.06147` | **33.07 dB** | **0.9673** |
| `DBCache (tau=0.16)` | **9022.7** | **1.19×** | `10/40` (`25.0%`) | `0.997357` | `0.06864` | **32.47 dB** | **0.9601** |
| `DBCache (tau=0.24)` | **8214.9** | **1.31×** | `14/40` (`35.0%`) | `0.995504` | `0.08957` | **30.55 dB** | **0.9357** |

## 2. Real A100 End-to-End Mixed-Modality Trace Replay (`Wan2.1` + Bursty `SD3.5`, `Image SLO = 4,000 ms`, `Video SLO = 14,500 ms`)

| System | Image Mean Latency (ms) | Image P99 Latency (ms) | Video Latency (ms) | Makespan (ms) | Image SLO% | Video SLO% | Total SLO% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Vanilla-xDiT (Non-Preemptive)` | 14564.8 | **15454.6** | **10822.4** | 17854.6 | 0.0% | 100.0% | **25.0%** |
| `GenServe-40GB (Step-Preempt + Cold Swap)` | 4706.7 | **6436.9** | **18180.2** | 18180.2 | 33.3% | 0.0% | **25.0%** |
| `HeteroDiT-Exact (Ours, Lossless)` | 2762.7 | **3765.3** | **15951.4** | 15951.4 | **100.0%** | 0.0% | **75.0%** |
| `HeteroDiT-Approx (Ours, Laxity Cache tau=0.16)` | 2949.0 | **3458.1** | **13761.7** | 13761.7 | **100.0%** | **100.0%** | **100.0%** |
