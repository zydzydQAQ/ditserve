# 20261110-d20261110-formal-v2-scale-n0-8gpu-ae4ba3d-r3-p-gpu-16

实验 `formal-wan22-test-poisson-s301-r24rpm-v0p5-slo1p5-g16gpu-scalability`，运行 `d20261110-formal-v2-scale-n0-8gpu-ae4ba3d-r3-p-gpu-16`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`1882f708722d0c67aefe4c089df2210702d128a76a3524272db17e9c325020c2`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / ditserve | 全部 | completed | 200 | 0.935 | 687114 | 0.182729 | 0.182729 |
| cluster / ditserve | image | completed | 100 | 1 | 14977.9 | 0.097702 | 0.097702 |
| cluster / ditserve | video | completed | 100 | 0.87 | 723806 | 0.0850007 | 0.0850007 |
| cluster / block-cfg2 | 全部 | completed | 200 | 0.93 | 552468 | 0.17712 | 0.17712 |
| cluster / block-cfg2 | image | completed | 100 | 0.97 | 15845.1 | 0.0923585 | 0.0923585 |
| cluster / block-cfg2 | video | completed | 100 | 0.89 | 666544 | 0.0847413 | 0.0847413 |
| cluster / edf-cfg2 | 全部 | completed | 200 | 0.82 | 669743 | 0.16561 | 0.16561 |
| cluster / edf-cfg2 | image | completed | 100 | 0.81 | 20214.3 | 0.081785 | 0.081785 |
| cluster / edf-cfg2 | video | completed | 100 | 0.83 | 721194 | 0.0838044 | 0.0838044 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
