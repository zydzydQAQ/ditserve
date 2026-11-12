# 20261108-d20261108-formal-v2-cb-refresh-8gpu-1b54f37-r1-p-rate-32

实验 `formal-wan22-test-poisson-s301-r32rpm-v0p5-slo1p5-ditserve`，运行 `d20261108-formal-v2-cb-refresh-8gpu-1b54f37-r1-p-rate-32`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`ecab2e8a1a01ff27ef266a8c6e34777dbcbf7bfe45d448db43ba6a9edb7d16a4`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / ditserve | 全部 | completed | 100 | 0.79 | 666090 | 0.0831439 | 0.0831439 |
| cluster / ditserve | image | completed | 50 | 0.98 | 14954.6 | 0.0515661 | 0.0515661 |
| cluster / ditserve | video | completed | 50 | 0.6 | 705286 | 0.0315711 | 0.0315711 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
