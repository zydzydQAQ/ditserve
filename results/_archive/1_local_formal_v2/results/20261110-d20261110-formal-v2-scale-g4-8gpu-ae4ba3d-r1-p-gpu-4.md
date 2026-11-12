# 20261110-d20261110-formal-v2-scale-g4-8gpu-ae4ba3d-r1-p-gpu-4

实验 `formal-wan22-test-poisson-s301-r6rpm-v0p5-slo1p5-g4gpu-scalability`，运行 `d20261110-formal-v2-scale-g4-8gpu-ae4ba3d-r1-p-gpu-4`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`27a15a2f441d68d7c0586afd60cc51b74494c19417909e10e4e960ff3d1ae09e`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / ditserve | 全部 | completed | 50 | 0.92 | 537208 | 0.0467086 | 0.0467086 |
| cluster / ditserve | image | completed | 25 | 1 | 14907.3 | 0.0253843 | 0.0253843 |
| cluster / ditserve | video | completed | 25 | 0.84 | 559112 | 0.0213228 | 0.0213228 |
| cluster / block-cfg2 | 全部 | completed | 50 | 0.9 | 535826 | 0.0458897 | 0.0458897 |
| cluster / block-cfg2 | image | completed | 25 | 0.96 | 15665.2 | 0.0244739 | 0.0244739 |
| cluster / block-cfg2 | video | completed | 25 | 0.84 | 604180 | 0.0214147 | 0.0214147 |
| cluster / edf-cfg2 | 全部 | completed | 50 | 0.76 | 561113 | 0.0396393 | 0.0396393 |
| cluster / edf-cfg2 | image | completed | 25 | 0.72 | 20615.8 | 0.018776 | 0.018776 |
| cluster / edf-cfg2 | video | completed | 25 | 0.8 | 584537 | 0.0208622 | 0.0208622 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
