# 20261103-d20261103-formal-calibration-wan22-b97479e-r1-cal0

实验 `kairos-wan22-native-calibration`，运行 `d20261103-formal-calibration-wan22-b97479e-r1-cal0`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`246f942f773650c213872b72c20f544ff34d98a9c648c4bbe7e13de7120d8f15`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 204533 | 0.00929381 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 1915.95 | 0.00464689 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 215197 | 0.00464689 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 131680 | 0.014439 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 1837.7 | 0.00721946 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 138514 | 0.00721946 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 184462 | 0.0103056 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 1907.78 | 0.00515275 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 194070 | 0.00515275 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 133138 | 0.0142812 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 1925.84 | 0.00714056 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 140044 | 0.00714056 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 169964 | 0.011185 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 1882.14 | 0.00559249 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 178810 | 0.00559249 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 167945 | 0.0113196 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 1898.58 | 0.00565978 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 176684 | 0.00565978 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
