# 20261110-d20261110-formal-v2-scale-n0-8gpu-ae4ba3d-r3-g2-gpu-2

实验 `formal-wan22-test-gamma-s301-r3rpm-v0p5-slo1p5-cv2p0-g2gpu-scalability`，运行 `d20261110-formal-v2-scale-n0-8gpu-ae4ba3d-r3-g2-gpu-2`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`e9ccbf22718161db76f7001d862b1bb1e996d7e2e163e8df8f52cb38e78aa63c`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / ditserve | 全部 | completed | 24 | 0.833333 | 425486 | 0.0219587 | 0.0219587 |
| cluster / ditserve | image | completed | 12 | 0.833333 | 22093.7 | 0.0109792 | 0.0109792 |
| cluster / ditserve | video | completed | 12 | 0.833333 | 593853 | 0.0109792 | 0.0109792 |
| cluster / block-cfg2 | 全部 | completed | 24 | 0.833333 | 422221 | 0.0220446 | 0.0220446 |
| cluster / block-cfg2 | image | completed | 12 | 0.833333 | 23478.5 | 0.0110221 | 0.0110221 |
| cluster / block-cfg2 | video | completed | 12 | 0.833333 | 590290 | 0.0110221 | 0.0110221 |
| cluster / edf-cfg2 | 全部 | completed | 24 | 0.666667 | 645976 | 0.0177633 | 0.0177633 |
| cluster / edf-cfg2 | image | completed | 12 | 0.583333 | 62169.8 | 0.00777133 | 0.00777133 |
| cluster / edf-cfg2 | video | completed | 12 | 0.75 | 717411 | 0.00999172 | 0.00999172 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
