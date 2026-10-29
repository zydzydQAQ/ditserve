# HeteroDiT 4×A100-PCIE-40GB NCCL & 2D Hybrid USP Benchmark Report

- **GPU Model**: `NVIDIA A100-PCIE-40GB` × `4`
- **Zero-Rebuild Dynamic Group Switching Overhead (`handle.adopt + xdit.activate`)**: `93.3 us`

## 1. End-to-End Real `Wan2.1-T2V-1.3B` (`480x832x17`) Multi-GPU Denoising Latency & Accuracy

| Parallel Form | Ranks | Ulysses | Ring | Status | Step Latency (ms) | Speedup vs SP1 | Cosine Sim vs SP1 | RMSE vs SP1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SP1 (1-GPU Baseline)` | `[0]` | 1 | 1 | `PASSED` | **547.31** | **1.00×** | 1.000005 | 0.00000 |
| `1D-Ulysses SP2 (u2r1)` | `[0, 1]` | 2 | 1 | `PASSED` | **630.76** | **0.87×** | 0.999975 | 0.00553 |
| `1D-Ulysses SP4 (u4r1)` | `[0, 1, 2, 3]` | 4 | 1 | `PASSED` | **409.76** | **1.34×** | 0.999981 | 0.00496 |
| `2D-Hybrid-USP 4-GPU (u2r2: Ulysses=2 x Ring=2)` | `[0, 1, 2, 3]` | 2 | 2 | `PASSED` | **590.40** | **0.93×** | 0.999980 | 0.00509 |

## 2. NCCL Collective & Overlapped Attention Microbenchmarks Across `Wan2.1` Shapes (60 Blocks/Step)

| Video Shape | Tokens | Parallel Paradigm | GPUs | Ulysses | Ring | Comm/Step (MB) | Step Comm+Attn (ms) | Effective BW (GB/s) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `480x832x17` | 7020 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 1234.0 | **317.60** | 3.79 |
| `480x832x17` | 7020 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 925.5 | **223.65** | 4.04 |
| `480x832x17` | 7020 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 1234.0 | **299.22** | 4.03 |
| `480x832x17` | 7020 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 617.0 | **161.95** | 3.72 |
| `480x832x17` | 7020 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 1234.0 | **194.66** | 4.02 |
| `480x832x33` | 14040 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 2468.0 | **624.71** | 3.86 |
| `480x832x33` | 14040 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 1851.0 | **425.08** | 4.25 |
| `480x832x33` | 14040 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 2468.0 | **583.66** | 4.13 |
| `480x832x33` | 14040 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 1234.0 | **318.41** | 3.78 |
| `480x832x33` | 14040 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 2468.0 | **379.30** | 4.13 |
| `720x1280x17` | 18000 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 3164.1 | **796.02** | 3.88 |
| `720x1280x17` | 18000 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 2373.1 | **540.82** | 4.29 |
| `720x1280x17` | 18000 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 3164.1 | **745.28** | 4.15 |
| `720x1280x17` | 18000 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 1582.0 | **406.28** | 3.80 |
| `720x1280x17` | 18000 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 3164.1 | **484.88** | 4.14 |
| `480x832x81` | 32760 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 5758.6 | **1448.24** | 3.88 |
| `480x832x81` | 32760 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 4318.9 | **958.21** | 4.40 |
| `480x832x81` | 32760 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 5758.6 | **1347.87** | 4.17 |
| `480x832x81` | 32760 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 2879.3 | **735.96** | 3.82 |
| `480x832x81` | 32760 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 5758.6 | **876.70** | 4.17 |
