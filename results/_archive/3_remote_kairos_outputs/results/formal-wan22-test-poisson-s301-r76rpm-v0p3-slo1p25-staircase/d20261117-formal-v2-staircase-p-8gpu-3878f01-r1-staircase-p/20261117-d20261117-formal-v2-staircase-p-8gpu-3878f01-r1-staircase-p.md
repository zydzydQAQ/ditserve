# 20261117-d20261117-formal-v2-staircase-p-8gpu-3878f01-r1-staircase-p

实验 `formal-wan22-test-poisson-s301-r76rpm-v0p3-slo1p25-staircase`，运行 `d20261117-formal-v2-staircase-p-8gpu-3878f01-r1-staircase-p`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`46666614b137bf0332cc50a47fefb6c6daf20e0436db4e7a9b00af2162e1d21e`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / stair-base | 全部 | completed | 100 | 0.25 | 527982 | 0.0347674 | 0.0347674 |
| cluster / stair-base | image | completed | 70 | 0.185714 | 237562 | 0.0180783 | 0.0180783 |
| cluster / stair-base | video | completed | 30 | 0.4 | 603093 | 0.0166876 | 0.0166876 |
| cluster / stair-prune | 全部 | completed | 100 | 0.51 | 441982 | 0.0806223 | 0.0806223 |
| cluster / stair-prune | image | completed | 70 | 0.514286 | 50237.4 | 0.0569066 | 0.0569066 |
| cluster / stair-prune | video | completed | 30 | 0.5 | 525781 | 0.0237111 | 0.0237111 |
| cluster / stair-pairfit | 全部 | completed | 100 | 0.51 | 449356 | 0.0800484 | 0.0800484 |
| cluster / stair-pairfit | image | completed | 70 | 0.542857 | 19220.7 | 0.0596404 | 0.0596404 |
| cluster / stair-pairfit | video | completed | 30 | 0.433333 | 529437 | 0.0204033 | 0.0204033 |
| cluster / ditserve | 全部 | completed | 100 | 0.77 | 456252 | 0.119607 | 0.119607 |
| cluster / ditserve | image | completed | 70 | 0.9 | 18267 | 0.0978515 | 0.0978515 |
| cluster / ditserve | video | completed | 30 | 0.466667 | 520105 | 0.0217448 | 0.0217448 |
| cluster / abl-no-cb | 全部 | completed | 100 | 0.63 | 453420 | 0.100012 | 0.100012 |
| cluster / abl-no-cb | image | completed | 70 | 0.742857 | 17812.9 | 0.0825446 | 0.0825446 |
| cluster / abl-no-cb | video | completed | 30 | 0.366667 | 525419 | 0.0174614 | 0.0174614 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
