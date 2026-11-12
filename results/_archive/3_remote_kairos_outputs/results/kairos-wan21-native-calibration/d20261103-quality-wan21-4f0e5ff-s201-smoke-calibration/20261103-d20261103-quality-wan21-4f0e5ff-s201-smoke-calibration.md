# 20261103-d20261103-quality-wan21-4f0e5ff-s201-smoke-calibration

实验 `kairos-wan21-native-calibration`，运行 `d20261103-quality-wan21-4f0e5ff-s201-smoke-calibration`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`e4814906512accdcd30c583fdab7933f2be8785402cf1f0fb5ab63f4e907b215`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 37907.9 | 0.0505608 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6588.02 | 0.0252801 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 39556.3 | 0.0252801 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 28257.9 | 0.0680273 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6559.18 | 0.0340129 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 29400 | 0.0340129 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 39963 | 0.0479401 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6604.73 | 0.0239697 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 41718.7 | 0.0239697 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 30221.8 | 0.0635637 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 6609.54 | 0.0317812 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 31464.5 | 0.0317812 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 34852.9 | 0.0550383 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 6629.04 | 0.0275187 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 36338.3 | 0.0275187 | 未验证 |
| cluster / static_cfg2sp3 | 全部 | completed | 2 | 1 | 32565.8 | 0.058944 | 未验证 |
| cluster / static_cfg2sp3 | image | completed | 1 | 1 | 6637.44 | 0.0294713 | 未验证 |
| cluster / static_cfg2sp3 | video | completed | 1 | 1 | 33930.5 | 0.0294713 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 31206.7 | 0.0615337 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 6587.43 | 0.0307665 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 32502.5 | 0.0307665 | 未验证 |
| cluster / static_sp6 | 全部 | completed | 2 | 1 | 35803.3 | 0.0535591 | 未验证 |
| cluster / static_sp6 | image | completed | 1 | 1 | 6568.74 | 0.0267788 | 未验证 |
| cluster / static_sp6 | video | completed | 1 | 1 | 37341.9 | 0.0267788 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
