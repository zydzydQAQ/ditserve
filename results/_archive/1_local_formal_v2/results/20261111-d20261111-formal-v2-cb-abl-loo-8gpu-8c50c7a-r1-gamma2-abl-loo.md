# 20261111-d20261111-formal-v2-cb-abl-loo-8gpu-8c50c7a-r1-gamma2-abl-loo

实验 `formal-wan22-test-gamma-s301-l1p75-v0p5-slo1p5-cv2p0-ablation`，运行 `d20261111-formal-v2-cb-abl-loo-8gpu-8c50c7a-r1-gamma2-abl-loo`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`c84af0890ced99c57c36494deec4a62bf9455ffe59d1b0e8a84fe3a231ed313f`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / abl-no-block | 全部 | completed | 100 | 0.8 | 382389 | 0.0854078 | 0.0854078 |
| cluster / abl-no-block | image | completed | 50 | 0.7 | 22391.5 | 0.0373631 | 0.0373631 |
| cluster / abl-no-block | video | completed | 50 | 0.9 | 753246 | 0.0480382 | 0.0480382 |
| cluster / abl-no-prune | 全部 | completed | 100 | 0.86 | 875299 | 0.0693538 | 0.0693538 |
| cluster / abl-no-prune | image | completed | 50 | 1 | 15213.4 | 0.0403195 | 0.0403195 |
| cluster / abl-no-prune | video | completed | 50 | 0.72 | 907518 | 0.0290301 | 0.0290301 |
| cluster / abl-first-fit | 全部 | completed | 100 | 0.76 | 639929 | 0.0850451 | 0.0850451 |
| cluster / abl-first-fit | image | completed | 50 | 0.66 | 142888 | 0.0369232 | 0.0369232 |
| cluster / abl-first-fit | video | completed | 50 | 0.86 | 717053 | 0.048112 | 0.048112 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
