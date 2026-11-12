# 20261103-d20261103-formal-calibration-wan21-b97479e-r1-cal1

实验 `kairos-wan21-native-calibration`，运行 `d20261103-formal-calibration-wan21-b97479e-r1-cal1`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`50e04f93a01199140c3b1d5b8e1acb6e805c8d6bf054e001ac4b808eb0dcca0e`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 857173 | 0.00221747 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6783.43 | 0.00110873 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 901931 | 0.00110873 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 456280 | 0.00416723 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6828.94 | 0.00208361 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 479936 | 0.00208361 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 591768 | 0.00321257 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6834.14 | 0.00160628 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 622554 | 0.00160628 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 365836 | 0.00519846 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 6862.79 | 0.00259922 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 384729 | 0.00259922 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 483151 | 0.00393528 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 6793.83 | 0.00196764 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 508223 | 0.00196764 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 437372 | 0.00434753 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 6843.92 | 0.00217376 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 460031 | 0.00217376 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
