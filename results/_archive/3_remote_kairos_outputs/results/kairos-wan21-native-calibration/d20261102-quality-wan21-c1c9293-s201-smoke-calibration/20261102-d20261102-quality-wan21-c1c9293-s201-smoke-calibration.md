# 20261102-d20261102-quality-wan21-c1c9293-s201-smoke-calibration

实验 `kairos-wan21-native-calibration`，运行 `d20261102-quality-wan21-c1c9293-s201-smoke-calibration`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`1c89fcc0f1d6403d3800317a64160b6115f4ad83d9a2fa393802562ba6c06fea`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 38750.6 | 0.0494603 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6719.3 | 0.02473 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 40436.5 | 0.02473 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 28129.5 | 0.0683621 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6726.8 | 0.0341809 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 29256 | 0.0341809 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 40102.1 | 0.0477784 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6704.3 | 0.0238891 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 41859.9 | 0.0238891 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 29487.6 | 0.065178 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 6732.92 | 0.0325889 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 30685.2 | 0.0325889 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 34661.2 | 0.0553543 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 6737.47 | 0.027677 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 36130.9 | 0.027677 | 未验证 |
| cluster / static_cfg2sp3 | 全部 | completed | 2 | 1 | 33436.3 | 0.0573963 | 未验证 |
| cluster / static_cfg2sp3 | image | completed | 1 | 1 | 6662.07 | 0.028698 | 未验证 |
| cluster / static_cfg2sp3 | video | completed | 1 | 1 | 34845.4 | 0.028698 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 32160 | 0.0597045 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 6732.46 | 0.0298521 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 33498.3 | 0.0298521 | 未验证 |
| cluster / static_sp6 | 全部 | completed | 2 | 1 | 34686.2 | 0.0553147 | 未验证 |
| cluster / static_sp6 | image | completed | 1 | 1 | 6746.06 | 0.0276572 | 未验证 |
| cluster / static_sp6 | video | completed | 1 | 1 | 36156.8 | 0.0276572 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
