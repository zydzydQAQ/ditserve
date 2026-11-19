# 20261116-d20261116-formal-v2-sbpc2-n1-8gpu-720da72-r1-soak-tetriserve

实验 `formal-wan22-test-soak5phase-s301-n500-slo1p5-tetriserve`，运行 `d20261116-formal-v2-sbpc2-n1-8gpu-720da72-r1-soak-tetriserve`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`6ad7aa2b62f3894f43e3f1552bfbb10ef2df97763664bc188c90983b9534ee38`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / tetriserve | 全部 | completed | 500 | 0.666 | 2.24612e+06 | 0.0802932 | 0.0802932 |
| cluster / tetriserve | image | completed | 264 | 0.715909 | 1.38516e+06 | 0.0455689 | 0.0455689 |
| cluster / tetriserve | video | completed | 236 | 0.610169 | 2.40546e+06 | 0.0347191 | 0.0347191 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
