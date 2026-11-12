# 20261103-d20261103-formal-calibration-wan21-b97479e-r1-cal0

实验 `kairos-wan21-native-calibration`，运行 `d20261103-formal-calibration-wan21-b97479e-r1-cal0`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`74748c4cddafffeb77164122863e98c515f7268b5ceab83344bbd85bf2c072cb`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 248821 | 0.007639 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 1950.53 | 0.00381949 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 261814 | 0.00381949 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 150880 | 0.0126006 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 1872.27 | 0.00630026 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 158723 | 0.00630026 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 213645 | 0.00889727 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 1926.79 | 0.00444862 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 224788 | 0.00444862 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 158050 | 0.0120289 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 1922.72 | 0.00601438 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 166267 | 0.00601438 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 192253 | 0.00988758 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 1847.25 | 0.00494377 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 202274 | 0.00494377 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 188956 | 0.0100604 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 1936.74 | 0.00503017 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 198799 | 0.00503017 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
