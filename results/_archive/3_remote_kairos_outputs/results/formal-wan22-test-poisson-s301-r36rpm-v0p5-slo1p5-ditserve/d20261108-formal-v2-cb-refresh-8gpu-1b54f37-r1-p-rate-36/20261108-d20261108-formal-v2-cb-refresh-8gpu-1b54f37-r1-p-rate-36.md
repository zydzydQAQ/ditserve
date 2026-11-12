# 20261108-d20261108-formal-v2-cb-refresh-8gpu-1b54f37-r1-p-rate-36

实验 `formal-wan22-test-poisson-s301-r36rpm-v0p5-slo1p5-ditserve`，运行 `d20261108-formal-v2-cb-refresh-8gpu-1b54f37-r1-p-rate-36`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`9d0bc157a159aa025deb52ca069f87f6dcc5b636cc9df6f44af7133a96891e41`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / ditserve | 全部 | completed | 100 | 0.81 | 644065 | 0.0870087 | 0.0870087 |
| cluster / ditserve | image | completed | 50 | 1 | 14825.8 | 0.0537047 | 0.0537047 |
| cluster / ditserve | video | completed | 50 | 0.62 | 705813 | 0.0332969 | 0.0332969 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
