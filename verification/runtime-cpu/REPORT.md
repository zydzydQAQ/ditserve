# Migration acceptance

UTC: 2026-10-15T08:42:41.026828+00:00

Status: **passed**. Full evidence: [results.xlsx](results.xlsx).

Each workbook row stores a JSON pointer, chunk index, and exact JSON scalar. Concatenate chunks before decoding a long scalar. Failures, missing dependencies and skipped/unverified checks remain explicit. No generated media or tensors are retained.

GPU seconds include subprocess loading, compilation, execution and cleanup. These are environment checks, not scientific qualification, a performance benchmark, or evidence of SP8 inference. Unselected suites remain unverified.
