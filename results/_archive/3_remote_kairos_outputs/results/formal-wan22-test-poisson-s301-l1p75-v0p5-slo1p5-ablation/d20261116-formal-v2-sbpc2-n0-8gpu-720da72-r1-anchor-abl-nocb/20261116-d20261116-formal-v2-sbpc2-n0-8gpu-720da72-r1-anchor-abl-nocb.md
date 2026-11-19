# 20261116-d20261116-formal-v2-sbpc2-n0-8gpu-720da72-r1-anchor-abl-nocb

实验 `formal-wan22-test-poisson-s301-l1p75-v0p5-slo1p5-ablation`，运行 `d20261116-formal-v2-sbpc2-n0-8gpu-720da72-r1-anchor-abl-nocb`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`ef4b658ed306ddaa1c9b6de760cb631cce6e596ec6cb70f2cd0c73b79e0c84a8`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / abl-no-cb | 全部 | completed | 100 | 0.94 | 497361 | 0.0991688 | 0.0991688 |
| cluster / abl-no-cb | image | completed | 50 | 1 | 15432.5 | 0.052746 | 0.052746 |
| cluster / abl-no-cb | video | completed | 50 | 0.88 | 644141 | 0.0464165 | 0.0464165 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
