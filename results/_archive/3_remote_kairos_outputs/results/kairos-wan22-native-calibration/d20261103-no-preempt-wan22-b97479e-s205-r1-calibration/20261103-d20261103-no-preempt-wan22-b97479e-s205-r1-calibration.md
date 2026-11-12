# 20261103-d20261103-no-preempt-wan22-b97479e-s205-r1-calibration

实验 `kairos-wan22-native-calibration`，运行 `d20261103-no-preempt-wan22-b97479e-s205-r1-calibration`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`9c451c976eb5add80f0642aa633163bcfc00539c1c608fc5412312a67f3cad84`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 72810.5 | 0.0262158 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6701.79 | 0.0131078 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 76289.9 | 0.0131078 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 64572.2 | 0.0295768 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6652.38 | 0.0147883 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 67620.6 | 0.0147883 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 73263.6 | 0.0260521 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6658.49 | 0.013026 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 76769.1 | 0.013026 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
