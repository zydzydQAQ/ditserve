# 20261102-d20261102-no-preempt-wan21-c1c9293-s205-r1-calibration

实验 `kairos-wan21-native-calibration`，运行 `d20261102-no-preempt-wan21-c1c9293-s205-r1-calibration`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`fccd5ff66d94b30fdfbe89ee23390af27417cc60d47e8a579543327b5ee260e9`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 38350.7 | 0.049982 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6739.81 | 0.0249909 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 40014.4 | 0.0249909 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 28375.2 | 0.067754 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6650.84 | 0.0338768 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 29518.6 | 0.0338768 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 40593.2 | 0.0471926 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6653.06 | 0.0235962 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 42379.5 | 0.0235962 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 29696.2 | 0.0647125 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 6711.81 | 0.032356 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 30905.9 | 0.032356 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 34221.2 | 0.0560719 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 6724.01 | 0.0280359 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 35668.5 | 0.0280359 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 32145.3 | 0.0597336 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 6747.05 | 0.0298667 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 33482 | 0.0298667 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
