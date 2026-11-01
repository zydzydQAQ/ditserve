# Vendored: group-free collectives (gfc)

- Upstream: https://github.com/SJTU-Liquid/group-free-collectives
- Commit: `9aadcdc6d4abc30af13feac908fbb4be98313398`
- License: MIT (see `LICENSE` in this directory). Copyright (c) 2026 GFC contributors.
- Paper: GF-DiT: Scheduling Parallelism for Diffusion Transformer Serving, arXiv:2606.13501.

The `gfc/` package sources are copied without semantic changes. DITSERVE only imports this
package when a study sets `options.collective_backend = "gfc"`; the adapter lives in
`workers/ditserve/models/xdit/collectives.py`. Requirements (torch>=2.10, triton>=3.6) are
already satisfied by the fixed runtime image (torch 2.10.0+cu126, triton 3.6.0), so no image
change is needed. Upstream tests and benchmarks are not vendored; the platform check is
`benchmarks/gfc_microbench.py`.
