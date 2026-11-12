# 20261107-d20261107-formal-v2-slo-scale-8gpu-7a77c45-r1-p-slo-2p0

实验 `formal-wan22-test-poisson-s301-r12rpm-v0p5-slo2p0-sweep`，运行 `d20261107-formal-v2-slo-scale-8gpu-7a77c45-r1-p-slo-2p0`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`0dd1b8a8cd01831e1c34fd2cd5d5c5e77323f7f197f6d5b467d0189bb757b977`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.64 | 508532 | 0.0534315 | 0.0534315 |
| cluster / xdit | image | completed | 50 | 0.92 | 15549.5 | 0.0384023 | 0.0384023 |
| cluster / xdit | video | completed | 50 | 0.36 | 591213 | 0.015027 | 0.015027 |
| cluster / tetriserve | 全部 | completed | 100 | 0.91 | 463034 | 0.0892713 | 0.0892713 |
| cluster / tetriserve | image | completed | 50 | 0.9 | 20921.4 | 0.0441424 | 0.0441424 |
| cluster / tetriserve | video | completed | 50 | 0.92 | 520398 | 0.0451233 | 0.0451233 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.97 | 453931 | 0.0954915 | 0.0954915 |
| cluster / sbp-esp | image | completed | 50 | 0.98 | 17446.9 | 0.0482352 | 0.0482352 |
| cluster / sbp-esp | video | completed | 50 | 0.96 | 472929 | 0.0472508 | 0.0472508 |
| cluster / ditserve | 全部 | completed | 100 | 0.99 | 418439 | 0.0980054 | 0.0980054 |
| cluster / ditserve | image | completed | 50 | 1 | 15630.4 | 0.0494947 | 0.0494947 |
| cluster / ditserve | video | completed | 50 | 0.98 | 444362 | 0.0485048 | 0.0485048 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.96 | 478525 | 0.096142 | 0.096142 |
| cluster / edf-bestfit | image | completed | 50 | 1 | 15802.3 | 0.0500703 | 0.0500703 |
| cluster / edf-bestfit | video | completed | 50 | 0.92 | 493901 | 0.0460646 | 0.0460646 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
