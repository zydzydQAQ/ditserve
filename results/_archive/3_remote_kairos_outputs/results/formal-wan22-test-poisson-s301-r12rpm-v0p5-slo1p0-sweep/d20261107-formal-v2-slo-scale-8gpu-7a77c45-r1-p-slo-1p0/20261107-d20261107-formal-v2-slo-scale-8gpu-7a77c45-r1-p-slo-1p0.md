# 20261107-d20261107-formal-v2-slo-scale-8gpu-7a77c45-r1-p-slo-1p0

实验 `formal-wan22-test-poisson-s301-r12rpm-v0p5-slo1p0-sweep`，运行 `d20261107-formal-v2-slo-scale-8gpu-7a77c45-r1-p-slo-1p0`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`8412883d68b35bcc1edeeb00e84dc26f092431f1a0a1a378d362eaf8b7f2a6c3`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.03 | 509472 | 0.00250917 | 0.00250917 |
| cluster / xdit | image | completed | 50 | 0 | 15596.5 | 0 | 0 |
| cluster / xdit | video | completed | 50 | 0.06 | 592573 | 0.00250907 | 0.00250907 |
| cluster / tetriserve | 全部 | completed | 100 | 0.05 | 473449 | 0.00511327 | 0.00511327 |
| cluster / tetriserve | image | completed | 50 | 0 | 34238.8 | 0 | 0 |
| cluster / tetriserve | video | completed | 50 | 0.1 | 485048 | 0.00511299 | 0.00511299 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.04 | 462139 | 0.00400894 | 0.00400894 |
| cluster / sbp-esp | image | completed | 50 | 0 | 25572.2 | 0 | 0 |
| cluster / sbp-esp | video | completed | 50 | 0.08 | 473821 | 0.0040087 | 0.0040087 |
| cluster / ditserve | 全部 | completed | 100 | 0.41 | 506078 | 0.0434233 | 0.0434233 |
| cluster / ditserve | image | completed | 50 | 0 | 15542.7 | 0 | 0 |
| cluster / ditserve | video | completed | 50 | 0.82 | 650063 | 0.0434204 | 0.0434204 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.02 | 450987 | 0.00206467 | 0.00206467 |
| cluster / edf-bestfit | image | completed | 50 | 0 | 17727 | 0 | 0 |
| cluster / edf-bestfit | video | completed | 50 | 0.04 | 463991 | 0.00206455 | 0.00206455 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
