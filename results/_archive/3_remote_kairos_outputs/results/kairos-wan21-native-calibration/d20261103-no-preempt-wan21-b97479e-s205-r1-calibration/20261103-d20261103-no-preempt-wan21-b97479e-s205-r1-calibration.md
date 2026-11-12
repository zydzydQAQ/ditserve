# 20261103-d20261103-no-preempt-wan21-b97479e-s205-r1-calibration

实验 `kairos-wan21-native-calibration`，运行 `d20261103-no-preempt-wan21-b97479e-s205-r1-calibration`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`4684d71a2e6cfe78c31635a346d170f21e57e481c6b06ea9217e8723ae631f45`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 73469.7 | 0.0259793 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6688.91 | 0.0129895 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 76984.5 | 0.0129895 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 65799.9 | 0.0290236 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6720.02 | 0.0145117 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 68909.4 | 0.0145117 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 71293 | 0.0267757 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6662.52 | 0.0133878 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 74694.6 | 0.0133878 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
