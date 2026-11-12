# 20261102-d20261103-formal-p0-wan21-stop-92b72ff-r1-contention

实验 `formal-wan21-development-bursty-s201-l1p2-v0p5-contention`，运行 `d20261103-formal-p0-wan21-stop-92b72ff-r1-contention`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`a77f0b7fd1bddeff8aeaa852157a31b95f9decd1123663b3839fc7c307905950`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / block-cfg2 | 全部 | completed | 40 | 0.725 | 84994.7 | 0.0448514 | 未验证 |
| cluster / block-cfg2 | image | completed | 20 | 0.75 | 1945.25 | 0.0231986 | 未验证 |
| cluster / block-cfg2 | video | completed | 20 | 0.7 | 157741 | 0.021652 | 未验证 |
| cluster / edf-cfg2 | 全部 | completed | 40 | 0.5 | 102730 | 0.0311223 | 未验证 |
| cluster / edf-cfg2 | image | completed | 20 | 0.25 | 2369.08 | 0.00778046 | 未验证 |
| cluster / edf-cfg2 | video | completed | 20 | 0.75 | 139326 | 0.0233414 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
