# 20261112-d20261112-formal-v2-abl-cfg2-p-8gpu-8c50c7a-r1-anchor-abl-cfg2

实验 `formal-wan22-test-poisson-s301-l1p75-v0p5-slo1p5-ablation`，运行 `d20261112-formal-v2-abl-cfg2-p-8gpu-8c50c7a-r1-anchor-abl-cfg2`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`94d3ce4ac93caec9c3036854083ee7b97e8b87673ef9004c29807131f6c40448`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / tetriserve-cfg2 | 全部 | completed | 100 | 0.83 | 468909 | 0.0899461 | 0.0899461 |
| cluster / tetriserve-cfg2 | image | completed | 50 | 0.86 | 17787.4 | 0.0465952 | 0.0465952 |
| cluster / tetriserve-cfg2 | video | completed | 50 | 0.8 | 650118 | 0.0433444 | 0.0433444 |
| cluster / edf-bestfit-cfg2 | 全部 | completed | 100 | 0.79 | 461942 | 0.0844334 | 0.0844334 |
| cluster / edf-bestfit-cfg2 | image | completed | 50 | 0.96 | 16729.2 | 0.0512978 | 0.0512978 |
| cluster / edf-bestfit-cfg2 | video | completed | 50 | 0.62 | 490677 | 0.0331298 | 0.0331298 |
| cluster / sbp-esp-cfg2 | 全部 | completed | 100 | 0.85 | 503708 | 0.0915616 | 0.0915616 |
| cluster / sbp-esp-cfg2 | image | completed | 50 | 0.86 | 17504.5 | 0.046316 | 0.046316 |
| cluster / sbp-esp-cfg2 | video | completed | 50 | 0.84 | 637081 | 0.0452389 | 0.0452389 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
