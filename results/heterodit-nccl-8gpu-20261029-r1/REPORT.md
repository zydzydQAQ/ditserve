# HeteroDiT 8×A100-PCIE-40GB NCCL & 2D Hybrid USP Benchmark Report

- **GPU Model**: `NVIDIA A100-PCIE-40GB` × `8`
- **Zero-Rebuild Dynamic Group Switching Overhead (`handle.adopt + xdit.activate`)**: `82.42 us`

## 1. End-to-End Real `Wan2.1-T2V-1.3B` (`480x832x17`) Multi-GPU Denoising Latency & Accuracy

| Parallel Form | Ranks | Ulysses | Ring | Status | Step Latency (ms) | Speedup vs SP1 | Cosine Sim vs SP1 | RMSE vs SP1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SP1 (1-GPU Baseline)` | `[0]` | 1 | 1 | `PASSED` | **555.63** | **1.00×** | 1.000005 | 0.00000 |
| `1D-Ulysses SP2 (u2r1)` | `[0, 1]` | 2 | 1 | `PASSED` | **515.00** | **1.08×** | 0.999975 | 0.00553 |
| `1D-Ulysses SP4 (u4r1)` | `[0, 1, 2, 3]` | 4 | 1 | `PASSED` | **511.25** | **1.09×** | 0.999981 | 0.00496 |
| `2D-Hybrid-USP 4-GPU (u2r2: Ulysses=2 x Ring=2)` | `[0, 1, 2, 3]` | 2 | 2 | `PASSED` | **703.60** | **0.79×** | 0.999980 | 0.00509 |
| `2D-Hybrid-USP 8-GPU (u4r2: Ulysses=4 x Ring=2)` | `[0, 1, 2, 3, 4, 5, 6, 7]` | 4 | 2 | `PASSED` | **1347.48** | **0.41×** | 0.999968 | 0.00625 |
| `1D-Ulysses SP8 (u8r1 - Expected Head Divisibility Error)` | `[0, 1, 2, 3, 4, 5, 6, 7]` | 8 | 1 | `EXPECTED_HEAD_DIVISIBILITY_FAILURE` | N/A (`AssertionError: h must be divisible by world_siz`) | N/A | N/A | N/A |

## 2. NCCL Collective & Overlapped Attention Microbenchmarks Across `Wan2.1` Shapes (60 Blocks/Step)

| Video Shape | Tokens | Parallel Paradigm | GPUs | Ulysses | Ring | Comm/Step (MB) | Step Comm+Attn (ms) | Effective BW (GB/s) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `480x832x17` | 7020 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 1234.0 | **199.04** | 6.05 |
| `480x832x17` | 7020 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 925.5 | **305.56** | 2.96 |
| `480x832x17` | 7020 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 1234.0 | **417.49** | 2.89 |
| `480x832x17` | 7020 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 617.0 | **209.47** | 2.88 |
| `480x832x17` | 7020 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 1234.0 | **271.22** | 2.89 |
| `480x832x17` | 7020 | `1D-Ulysses-SP8 (u8r1, Padded H=16)` | 8 | 8 | 1 | 720.2 | **830.62** | 0.85 |
| `480x832x17` | 7020 | `2D-Hybrid-USP-8GPU (u4r2, Native H=12)` | 8 | 4 | 2 | 771.7 | **760.03** | 0.99 |
| `480x832x17` | 7020 | `2D-Hybrid-CFG-8GPU (cfg2u2r2, Native H=12)` | 8 | 2 | 2 | 617.0 | **211.36** | 2.85 |
| `480x832x17` | 7020 | `HeteroDiT-Approx-8GPU (u4r2+Cache35%)` | 8 | 4 | 2 | 771.7 | **488.17** | 1.00 |
| `480x832x33` | 14040 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 2468.0 | **389.69** | 6.18 |
| `480x832x33` | 14040 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 1851.0 | **594.53** | 3.04 |
| `480x832x33` | 14040 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 2468.0 | **828.34** | 2.91 |
| `480x832x33` | 14040 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 1234.0 | **415.81** | 2.90 |
| `480x832x33` | 14040 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 2468.0 | **538.33** | 2.91 |
| `480x832x33` | 14040 | `1D-Ulysses-SP8 (u8r1, Padded H=16)` | 8 | 8 | 1 | 1439.7 | **1728.85** | 0.81 |
| `480x832x33` | 14040 | `2D-Hybrid-USP-8GPU (u4r2, Native H=12)` | 8 | 4 | 2 | 1542.5 | **1547.88** | 0.97 |
| `480x832x33` | 14040 | `2D-Hybrid-CFG-8GPU (cfg2u2r2, Native H=12)` | 8 | 2 | 2 | 1234.0 | **418.08** | 2.88 |
| `480x832x33` | 14040 | `HeteroDiT-Approx-8GPU (u4r2+Cache35%)` | 8 | 4 | 2 | 1542.5 | **979.43** | 1.00 |
| `720x1280x17` | 18000 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 3164.1 | **495.47** | 6.24 |
| `720x1280x17` | 18000 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 2373.1 | **759.29** | 3.05 |
| `720x1280x17` | 18000 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 3164.1 | **1060.83** | 2.91 |
| `720x1280x17` | 18000 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 1582.0 | **531.05** | 2.91 |
| `720x1280x17` | 18000 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 3164.1 | **690.05** | 2.91 |
| `720x1280x17` | 18000 | `1D-Ulysses-SP8 (u8r1, Padded H=16)` | 8 | 8 | 1 | 1845.7 | **2302.94** | 0.78 |
| `720x1280x17` | 18000 | `2D-Hybrid-USP-8GPU (u4r2, Native H=12)` | 8 | 4 | 2 | 1977.5 | **1934.56** | 1.00 |
| `720x1280x17` | 18000 | `2D-Hybrid-CFG-8GPU (cfg2u2r2, Native H=12)` | 8 | 2 | 2 | 1582.0 | **534.72** | 2.89 |
| `720x1280x17` | 18000 | `HeteroDiT-Approx-8GPU (u4r2+Cache35%)` | 8 | 4 | 2 | 1977.5 | **1198.73** | 1.05 |
| `480x832x81` | 32760 | `1D-Ulysses-SP2 (u2r1)` | 2 | 2 | 1 | 5758.6 | **894.06** | 6.29 |
| `480x832x81` | 32760 | `1D-Ulysses-SP4 (u4r1)` | 4 | 4 | 1 | 4318.9 | **1358.43** | 3.10 |
| `480x832x81` | 32760 | `2D-Hybrid-USP (u2r2)` | 4 | 2 | 2 | 5758.6 | **1925.18** | 2.92 |
| `480x832x81` | 32760 | `2D-Hybrid-CFG (cfg2u2)` | 4 | 2 | 1 | 2879.3 | **962.76** | 2.92 |
| `480x832x81` | 32760 | `HeteroDiT-Approx (u2r2+Cache35%)` | 4 | 2 | 2 | 5758.6 | **1252.21** | 2.92 |
| `480x832x81` | 32760 | `1D-Ulysses-SP8 (u8r1, Padded H=16)` | 8 | 8 | 1 | 3359.2 | **3808.28** | 0.86 |
| `480x832x81` | 32760 | `2D-Hybrid-USP-8GPU (u4r2, Native H=12)` | 8 | 4 | 2 | 3599.1 | **3502.61** | 1.00 |
| `480x832x81` | 32760 | `2D-Hybrid-CFG-8GPU (cfg2u2r2, Native H=12)` | 8 | 2 | 2 | 2879.3 | **971.76** | 2.89 |
| `480x832x81` | 32760 | `HeteroDiT-Approx-8GPU (u4r2+Cache35%)` | 8 | 4 | 2 | 3599.1 | **2206.33** | 1.04 |
