# 20261108-d20261108-formal-v2-cb-ablation-8gpu-1b54f37-r1-anchor-abl

实验 `formal-wan22-test-poisson-s301-l1p75-v0p5-slo1p5-ablation`，运行 `d20261108-formal-v2-cb-ablation-8gpu-1b54f37-r1-anchor-abl`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`37d406406d604ee102f14e084acc5d33af8d418e70d1c2000c00f17ac81356d7`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / edf-cfg2 | 全部 | completed | 100 | 0.74 | 370752 | 0.0782262 | 0.0782262 |
| cluster / edf-cfg2 | image | completed | 50 | 0.58 | 44586.4 | 0.0306543 | 0.0306543 |
| cluster / edf-cfg2 | video | completed | 50 | 0.9 | 686045 | 0.0475671 | 0.0475671 |
| cluster / ditserve | 全部 | completed | 100 | 0.93 | 501561 | 0.0976939 | 0.0976939 |
| cluster / ditserve | image | completed | 50 | 1 | 14901.5 | 0.0525194 | 0.0525194 |
| cluster / ditserve | video | completed | 50 | 0.86 | 645174 | 0.0451667 | 0.0451667 |
| cluster / block-cfg2 | 全部 | completed | 100 | 0.87 | 637572 | 0.0905822 | 0.0905822 |
| cluster / block-cfg2 | image | completed | 50 | 0.86 | 18987.7 | 0.0447677 | 0.0447677 |
| cluster / block-cfg2 | video | completed | 50 | 0.88 | 732852 | 0.0458089 | 0.0458089 |
| cluster / abl-no-block | 全部 | completed | 100 | 0.8 | 457748 | 0.0830565 | 0.0830565 |
| cluster / abl-no-block | image | completed | 50 | 0.76 | 30303.7 | 0.0394489 | 0.0394489 |
| cluster / abl-no-block | video | completed | 50 | 0.84 | 541746 | 0.0436014 | 0.0436014 |
| cluster / abl-no-cb | 全部 | completed | 100 | 0.92 | 499874 | 0.096596 | 0.096596 |
| cluster / abl-no-cb | image | completed | 50 | 0.98 | 15595.9 | 0.0514444 | 0.0514444 |
| cluster / abl-no-cb | video | completed | 50 | 0.86 | 645326 | 0.0451451 | 0.0451451 |
| cluster / abl-first-fit | 全部 | completed | 100 | 0.82 | 362798 | 0.0821165 | 0.0821165 |
| cluster / abl-first-fit | image | completed | 50 | 0.72 | 67274 | 0.0360485 | 0.0360485 |
| cluster / abl-first-fit | video | completed | 50 | 0.92 | 716515 | 0.046062 | 0.046062 |
| cluster / abl-no-prune | 全部 | completed | 100 | 0.84 | 852309 | 0.0650257 | 0.0650257 |
| cluster / abl-no-prune | image | completed | 50 | 0.96 | 15104.3 | 0.0371553 | 0.0371553 |
| cluster / abl-no-prune | video | completed | 50 | 0.72 | 889626 | 0.0278665 | 0.0278665 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
