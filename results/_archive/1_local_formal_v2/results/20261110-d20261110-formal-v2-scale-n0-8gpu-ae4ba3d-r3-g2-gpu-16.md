# 20261110-d20261110-formal-v2-scale-n0-8gpu-ae4ba3d-r3-g2-gpu-16

实验 `formal-wan22-test-gamma-s301-r24rpm-v0p5-slo1p5-cv2p0-g16gpu-scalability`，运行 `d20261110-formal-v2-scale-n0-8gpu-ae4ba3d-r3-g2-gpu-16`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`980c2cdeec2e062bdbc5bbe4fa38c41ea574c41b476ccc2eec9cf2bd85a497d0`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / ditserve | 全部 | completed | 200 | 0.945 | 542822 | 0.194035 | 0.194035 |
| cluster / ditserve | image | completed | 100 | 1 | 15004.7 | 0.102648 | 0.102648 |
| cluster / ditserve | video | completed | 100 | 0.89 | 702788 | 0.0913568 | 0.0913568 |
| cluster / block-cfg2 | 全部 | completed | 200 | 0.925 | 520514 | 0.190361 | 0.190361 |
| cluster / block-cfg2 | image | completed | 100 | 0.97 | 16006.3 | 0.0997991 | 0.0997991 |
| cluster / block-cfg2 | video | completed | 100 | 0.88 | 627924 | 0.0905394 | 0.0905394 |
| cluster / edf-cfg2 | 全部 | completed | 200 | 0.79 | 559650 | 0.161182 | 0.161182 |
| cluster / edf-cfg2 | image | completed | 100 | 0.72 | 26676.4 | 0.073441 | 0.073441 |
| cluster / edf-cfg2 | video | completed | 100 | 0.86 | 675818 | 0.0877212 | 0.0877212 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
