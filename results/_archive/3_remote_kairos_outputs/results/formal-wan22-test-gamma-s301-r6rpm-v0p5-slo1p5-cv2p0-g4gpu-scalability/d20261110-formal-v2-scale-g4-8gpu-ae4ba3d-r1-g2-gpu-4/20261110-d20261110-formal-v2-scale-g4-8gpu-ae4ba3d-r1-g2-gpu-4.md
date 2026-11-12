# 20261110-d20261110-formal-v2-scale-g4-8gpu-ae4ba3d-r1-g2-gpu-4

实验 `formal-wan22-test-gamma-s301-r6rpm-v0p5-slo1p5-cv2p0-g4gpu-scalability`，运行 `d20261110-formal-v2-scale-g4-8gpu-ae4ba3d-r1-g2-gpu-4`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`31b35f34239a64ffd0094c4aa47f533ab6efe111c8fc05c616c393f772c207f4`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / ditserve | 全部 | completed | 50 | 0.94 | 340243 | 0.0496597 | 0.0496597 |
| cluster / ditserve | image | completed | 25 | 0.96 | 20677.8 | 0.0253573 | 0.0253573 |
| cluster / ditserve | video | completed | 25 | 0.92 | 500088 | 0.0243007 | 0.0243007 |
| cluster / block-cfg2 | 全部 | completed | 50 | 0.9 | 341663 | 0.0501499 | 0.0501499 |
| cluster / block-cfg2 | image | completed | 25 | 0.92 | 19887.4 | 0.0256315 | 0.0256315 |
| cluster / block-cfg2 | video | completed | 25 | 0.88 | 495665 | 0.0245171 | 0.0245171 |
| cluster / edf-cfg2 | 全部 | completed | 50 | 0.7 | 343423 | 0.0389857 | 0.0389857 |
| cluster / edf-cfg2 | image | completed | 25 | 0.52 | 29473 | 0.01448 | 0.01448 |
| cluster / edf-cfg2 | video | completed | 25 | 0.88 | 511034 | 0.0245046 | 0.0245046 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
