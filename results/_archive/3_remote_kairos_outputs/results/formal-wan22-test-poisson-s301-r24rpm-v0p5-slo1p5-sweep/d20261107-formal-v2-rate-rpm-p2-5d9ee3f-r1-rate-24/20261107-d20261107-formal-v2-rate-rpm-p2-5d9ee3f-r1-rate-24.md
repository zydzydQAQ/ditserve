# 20261107-d20261107-formal-v2-rate-rpm-p2-5d9ee3f-r1-rate-24

实验 `formal-wan22-test-poisson-s301-r24rpm-v0p5-slo1p5-sweep`，运行 `d20261107-formal-v2-rate-rpm-p2-5d9ee3f-r1-rate-24`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`e59e963a9a85f968aeab7d57441090d3028795e8a9ce512dce873b7632e56d30`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.35 | 713024 | 0.0296414 | 0.0296414 |
| cluster / xdit | image | completed | 50 | 0.5 | 29601.8 | 0.0211715 | 0.0211715 |
| cluster / xdit | video | completed | 50 | 0.2 | 794277 | 0.00846859 | 0.00846859 |
| cluster / tetriserve | 全部 | completed | 100 | 0.71 | 738102 | 0.076536 | 0.076536 |
| cluster / tetriserve | image | completed | 50 | 0.82 | 20152.7 | 0.044194 | 0.044194 |
| cluster / tetriserve | video | completed | 50 | 0.6 | 748596 | 0.032337 | 0.032337 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.71 | 765509 | 0.076398 | 0.076398 |
| cluster / sbp-esp | image | completed | 50 | 0.82 | 107106 | 0.0441144 | 0.0441144 |
| cluster / sbp-esp | video | completed | 50 | 0.6 | 777831 | 0.0322788 | 0.0322788 |
| cluster / ditserve | 全部 | completed | 100 | 0.84 | 647904 | 0.0881542 | 0.0881542 |
| cluster / ditserve | image | completed | 50 | 1 | 15534.3 | 0.0524695 | 0.0524695 |
| cluster / ditserve | video | completed | 50 | 0.68 | 683795 | 0.0356792 | 0.0356792 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.68 | 611012 | 0.0729977 | 0.0729977 |
| cluster / edf-bestfit | image | completed | 50 | 0.96 | 16079 | 0.0515242 | 0.0515242 |
| cluster / edf-bestfit | video | completed | 50 | 0.4 | 667542 | 0.0214684 | 0.0214684 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
