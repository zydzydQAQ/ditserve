# 20261103-d20261103-serve-wan21-4f0e5ff-s301-diag-calibration

实验 `kairos-wan21-native-calibration`，运行 `d20261103-serve-wan21-4f0e5ff-s301-diag-calibration`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`ea1e62587cd2e2974a2d1af78db0e2c3603f6c09361cbca5373eb42f21667626`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 38708.1 | 0.0495053 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6568.03 | 0.0247522 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 40399.7 | 0.0247522 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 28746.7 | 0.0668649 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6623.28 | 0.0334321 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 29911 | 0.0334321 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 40291.9 | 0.0475496 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6671.07 | 0.0237745 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 42061.4 | 0.0237745 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 30189.1 | 0.0636351 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 6626.86 | 0.0318171 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 31429.2 | 0.0318171 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 34907.7 | 0.0549501 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 6617.53 | 0.0274747 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 36396.7 | 0.0274747 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 32827.6 | 0.0584686 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 6631.24 | 0.0292339 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 34206.4 | 0.0292339 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
