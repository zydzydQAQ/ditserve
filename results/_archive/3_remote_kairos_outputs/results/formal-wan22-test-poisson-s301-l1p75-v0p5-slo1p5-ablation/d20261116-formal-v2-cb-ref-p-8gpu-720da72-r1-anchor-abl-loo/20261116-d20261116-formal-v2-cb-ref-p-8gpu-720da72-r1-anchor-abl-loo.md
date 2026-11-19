# 20261116-d20261116-formal-v2-cb-ref-p-8gpu-720da72-r1-anchor-abl-loo

实验 `formal-wan22-test-poisson-s301-l1p75-v0p5-slo1p5-ablation`，运行 `d20261116-formal-v2-cb-ref-p-8gpu-720da72-r1-anchor-abl-loo`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`307192b43e9baf77f9fc1110c32b31be7fb172c1826bc58f90897856ebe8d0d8`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / abl-no-block | 全部 | completed | 100 | 0.84 | 360059 | 0.089997 | 0.089997 |
| cluster / abl-no-block | image | completed | 50 | 0.76 | 27326.5 | 0.0407098 | 0.0407098 |
| cluster / abl-no-block | video | completed | 50 | 0.92 | 614855 | 0.0492803 | 0.0492803 |
| cluster / abl-no-prune | 全部 | completed | 100 | 0.82 | 912370 | 0.0594311 | 0.0594311 |
| cluster / abl-no-prune | image | completed | 50 | 0.94 | 17715.1 | 0.0340622 | 0.0340622 |
| cluster / abl-no-prune | video | completed | 50 | 0.7 | 930672 | 0.0253655 | 0.0253655 |
| cluster / abl-first-fit | 全部 | completed | 100 | 0.82 | 349583 | 0.090106 | 0.090106 |
| cluster / abl-first-fit | image | completed | 50 | 0.72 | 104148 | 0.0395554 | 0.0395554 |
| cluster / abl-first-fit | video | completed | 50 | 0.92 | 574709 | 0.0505431 | 0.0505431 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
