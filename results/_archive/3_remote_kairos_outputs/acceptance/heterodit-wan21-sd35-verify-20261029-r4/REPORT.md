# HeteroDiT (`ditserve-hetero` `cd5b390`) Verification Report on `jjjkairos`

## 1. Real Hardware Co-Location & Runtime Verification (`Wan2.1-T2V-1.3B + SD3.5-Medium`)
- **Container Image**: `192.168.1.114:30500/jjjkairos/ditserve:xdit-dev-016e1ca-slim`
- **Build Commit**: `cd5b390`
- **GPU Device**: `NVIDIA A100-PCIE-40GB` (`40442.38 MB`)
- **Dual-Model HBM Residency (Before Encoder Offload)**: `18512.84 MB`
- **Dual-Model HBM Residency (After T5-XXL Encoder Pool Offload)**: `9437.58 MB` (**Reclaimed `9075.26 MB` HBM**)
- **Peak HBM Allocated During Concurrent Preemption + Chunked 3D-VAE Decode**: `18588.09 MB` (< 40 GB ceiling: `True`)
- **ProactiveEncoderPool Stats**: `{"entries": 1, "bytes": 5464064, "max_entries": 32, "max_bytes": 536870912, "hits": 1, "misses": 0, "staged_pushes": 1}`
- **Tiered VideoState Offload (HBM -> Host Pinned DRAM)**: `17.221 ms` (`21561376 B`)
- **Tiered VideoState Lookahead Prefetch (Host Pinned DRAM -> HBM)**: `6.306 ms` (`21561376 B`)

## 2. Calibrated 8×A100-PCIE-40GB Multi-Seed Scheduler Evaluation
```text

=== Bursty Gamma (CV=2.5, Load=0.95, Dual-Tier Video SLO) (gamma_cv2.5_load0.95) ===
Policy                                   | Total SLO% |  Img SLO% |  Vid SLO% | Img P99(ms) | Vid P99(ms) | Batches | ApproxQ
------------------------------------------------------------------------------------------------------------------------
Static-SP1 (FCFS/EDF No-Preempt)         |     95.31% |   100.00% |    81.25% |      1163.7 |     23508.7 |     0.0 |     0.0
Static-CFG2 (2-GPU Fixed)                |     98.96% |   100.00% |    95.83% |      1068.8 |     13769.2 |     0.0 |     0.0
TetriServe-Round (5-Step Round)          |     95.31% |   100.00% |    81.25% |      1105.8 |     25679.4 |     0.0 |     0.0
GenServe-Step (Step-Boundary SP)         |     99.48% |   100.00% |    97.92% |      1125.0 |     11911.7 |     0.0 |     0.0
DitServe-Baseline (Block-Elastic)        |     99.48% |   100.00% |    97.92% |      1096.7 |     11487.1 |     0.0 |     0.0
HeteroDiT-Exact (Knapsack+2D-USP+Tiered) |    100.00% |   100.00% |   100.00% |      1573.9 |     11046.7 |     4.0 |     0.0
HeteroDiT-Approx (Laxity Cache+2D-USP)   |    100.00% |   100.00% |   100.00% |      1520.2 |      9160.6 |     3.3 |   139.0

=== Heavy Burst (Bursty Pattern, Load=1.10, Tight SLO) (bursty_load1.10) ===
Policy                                   | Total SLO% |  Img SLO% |  Vid SLO% | Img P99(ms) | Vid P99(ms) | Batches | ApproxQ
------------------------------------------------------------------------------------------------------------------------
Static-SP1 (FCFS/EDF No-Preempt)         |     90.62% |   100.00% |    62.50% |       964.0 |     17311.0 |     0.0 |     0.0
Static-CFG2 (2-GPU Fixed)                |    100.00% |   100.00% |   100.00% |       900.0 |     11483.0 |     0.0 |     0.0
TetriServe-Round (5-Step Round)          |     94.27% |   100.00% |    77.08% |       939.4 |     19735.7 |     0.0 |     0.0
GenServe-Step (Step-Boundary SP)         |     99.48% |   100.00% |    97.92% |       972.6 |      8309.7 |     0.0 |     0.0
DitServe-Baseline (Block-Elastic)        |     99.48% |   100.00% |    97.92% |       962.2 |      8294.5 |     0.0 |     0.0
HeteroDiT-Exact (Knapsack+2D-USP+Tiered) |     99.48% |   100.00% |    97.92% |      1210.8 |      8526.7 |     0.7 |     0.0
HeteroDiT-Approx (Laxity Cache+2D-USP)   |    100.00% |   100.00% |   100.00% |       960.6 |      7978.9 |     0.0 |    42.3

=== Extreme HoL Contention (Gamma CV=3.0, Load=1.25, Tight Dual-Tier SLO) (gamma_cv3.0_load1.25_hol) ===
Policy                                   | Total SLO% |  Img SLO% |  Vid SLO% | Img P99(ms) | Vid P99(ms) | Batches | ApproxQ
------------------------------------------------------------------------------------------------------------------------
Static-SP1 (FCFS/EDF No-Preempt)         |     82.81% |    96.21% |    53.33% |      1841.7 |     26072.9 |     0.0 |     0.0
Static-CFG2 (2-GPU Fixed)                |     92.71% |    97.73% |    81.67% |      1740.7 |     22722.6 |     0.0 |     0.0
TetriServe-Round (5-Step Round)          |     85.94% |    97.73% |    60.00% |      1754.3 |     27098.0 |     0.0 |     0.0
GenServe-Step (Step-Boundary SP)         |     88.54% |    97.73% |    68.33% |      1745.8 |     15905.5 |     0.0 |     0.0
DitServe-Baseline (Block-Elastic)        |     88.54% |    97.73% |    68.33% |      1746.4 |     16232.8 |     0.0 |     0.0
HeteroDiT-Exact (Knapsack+2D-USP+Tiered) |     90.10% |   100.00% |    68.33% |      1648.9 |     15025.9 |     8.7 |     0.0
HeteroDiT-Approx (Laxity Cache+2D-USP)   |     94.79% |   100.00% |    83.33% |      1636.9 |      9495.9 |     8.0 |   209.7

```
