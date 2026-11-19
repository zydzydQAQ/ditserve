# Motivation Step & Block Predictability (CV%) Report — Shard `s2` (`d20261116-mot-cv-s2-8gpu-720da72-r1`)

- **Timestamp (UTC)**: `2026-11-16T10:46:03+00:00`
- **Host / Hardware**: `d20261116-mot-cv-s2-8gpu-720da72-r1-master-0` — `8x NVIDIA A100-PCIE-40GB`
- **Build**: `720da72`
- **Completed Configs**: `3`

| Modality | Model | Resolution | Form | GPUs | Step Mean (ms) | Step P99/P50 | Step Temporal CV (%) | Cross-Prompt Step CV (%) | Block Mean (ms) | Block Temporal CV Mean / P95 / Max (%) | Block Spatial CV All / Homog (%) | Peak HBM (GiB) |
| :--- | :--- | :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: |
| image | `SD3.5-Medium` | `1440x1440` | **sp4** | 4 | 738.27 | 1.0016 | **0.08%** | **0.08%** | 30.77 | 0.41% / 0.52% / 0.56% | 29.46% / **0.05%** | 7.37 |
| image | `SD3.5-Medium` | `1440x1440` | **sp8** | 8 | 483.73 | 1.0032 | **0.15%** | **0.15%** | 20.19 | 0.73% / 0.95% / 0.95% | 28.40% / **0.09%** | 7.28 |
| image | `SD3.5-Medium` | `1440x1440` | **cfg2** | 2 | 260.36 | 1.0068 | **0.31%** | **0.40%** | 10.79 | 0.53% / 0.55% / 0.56% | 23.46% / **0.13%** | 7.08 |
