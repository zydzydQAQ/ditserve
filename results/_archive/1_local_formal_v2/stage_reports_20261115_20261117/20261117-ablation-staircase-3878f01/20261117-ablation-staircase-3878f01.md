# 20261117 — 5-Rung Cumulative Design Staircase Ablation (`3878f01`)

- **Build**: `3878f01` (`3878f018d3da99c0fd7526c2435d8d16fd927e8e`)
- **Image**: `192.168.1.114:30500/jjjkairos/ditserve@sha256:3d8bcd2edf64c889977042569555177503c10834023b58f07f14f68cfe8bca05`
- **Jobs**: `d20261117-formal-v2-staircase-p-8gpu-3878f01-r1`, `d20261117-formal-v2-staircase-g2-8gpu-3878f01-r1`
- **Operating Point**: `λ = 76.0 req/min`, `v = 0.3` (70 images + 30 videos), `κ = 1.25×`, 8× NVIDIA A100-PCIE-40GB
- **Shared Storage Check**: `790G / 1024G (78%)` (< 810G / 82% hard stop)

| Rung | Policy ID | Module Added | Poisson SAR (%) | Poisson (Img / Vid %) | Gamma CV=2.0 SAR (%) | Gamma (Img / Vid %) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `stair-base` (`Base`) | Base (Pure SP Step-Elastic) | **25.0%** | 18.6% / 40.0% | **22.0%** | 15.7% / 36.7% |
| 2 | `stair-prune` (`+Prune`) | +§4.2 Offline Profiler (CFG2 + L* Frontier) | **51.0%** | 51.4% / 50.0% | **32.0%** | 21.4% / 56.7% |
| 3 | `abl-first-fit-nocb` (`+Block`) | +§4.4.1–4.4.2 Block Preemptor & Companion Release (First-Fit) | **58.0%** | 64.3% / 43.3% | **46.0%** | 42.9% / 53.3% |
| 4 | `abl-no-cb` (`+PairFit`) | +§4.3 Online Scheduler (Tiered Sizing + Pair-Fit) | **63.0%** | 74.3% / 36.7% | **58.0%** | 70.0% / 30.0% |
| 5 | `ditserve` (`+CB`) | +§4.4.3 Continuous Batcher (DitServe) | **77.0%** | 90.0% / 46.7% | **69.0%** | 81.4% / 40.0% |
