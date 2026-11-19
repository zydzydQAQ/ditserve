# 20261117-d20261117-formal-v2-staircase-g2-8gpu-3878f01-r1-staircase-g2

实验 `formal-wan22-test-gamma-s301-r76rpm-v0p3-slo1p25-cv2p0-staircase`，运行 `d20261117-formal-v2-staircase-g2-8gpu-3878f01-r1-staircase-g2`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`ffcf643c27a027f4ad68257ea5fdd7a2f5ca16e1f4dbbe6e43300886e7333530`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / stair-base | 全部 | completed | 100 | 0.22 | 562664 | 0.0276499 | 0.0276499 |
| cluster / stair-base | image | completed | 70 | 0.157143 | 280138 | 0.0138242 | 0.0138242 |
| cluster / stair-base | video | completed | 30 | 0.366667 | 580410 | 0.0138242 | 0.0138242 |
| cluster / stair-prune | 全部 | completed | 100 | 0.32 | 419083 | 0.0540775 | 0.0540775 |
| cluster / stair-prune | image | completed | 70 | 0.214286 | 176638 | 0.0253471 | 0.0253471 |
| cluster / stair-prune | video | completed | 30 | 0.566667 | 520722 | 0.0287267 | 0.0287267 |
| cluster / stair-pairfit | 全部 | completed | 100 | 0.45 | 453667 | 0.0715092 | 0.0715092 |
| cluster / stair-pairfit | image | completed | 70 | 0.485714 | 22970.5 | 0.0540254 | 0.0540254 |
| cluster / stair-pairfit | video | completed | 30 | 0.366667 | 534972 | 0.0174788 | 0.0174788 |
| cluster / ditserve | 全部 | completed | 100 | 0.69 | 458688 | 0.109366 | 0.109366 |
| cluster / ditserve | image | completed | 70 | 0.814286 | 28592 | 0.0903382 | 0.0903382 |
| cluster / ditserve | video | completed | 30 | 0.4 | 530897 | 0.0190186 | 0.0190186 |
| cluster / abl-no-cb | 全部 | completed | 100 | 0.58 | 458611 | 0.0916434 | 0.0916434 |
| cluster / abl-no-cb | image | completed | 70 | 0.7 | 18995.7 | 0.0774174 | 0.0774174 |
| cluster / abl-no-cb | video | completed | 30 | 0.3 | 535448 | 0.0142195 | 0.0142195 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
