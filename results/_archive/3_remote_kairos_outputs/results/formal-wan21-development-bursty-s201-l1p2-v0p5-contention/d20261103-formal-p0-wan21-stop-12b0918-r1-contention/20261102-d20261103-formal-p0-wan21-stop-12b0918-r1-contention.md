# 20261102-d20261103-formal-p0-wan21-stop-12b0918-r1-contention

实验 `formal-wan21-development-bursty-s201-l1p2-v0p5-contention`，运行 `d20261103-formal-p0-wan21-stop-12b0918-r1-contention`：执行 `failed`，归档 `deferred`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`0079964fa890a8c2e081c8fca7b98f0be3cc973fa3224ce743d61f249a9354bb`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / 0 | 全部 | failed | 0 | 未记录 | 未记录 | 未记录 | 未验证 |
| 运行合计 | 全部 | failed | 40 | 0.425 | 69235.4 | 0.0322132 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

失败与未通过门槛（重复错误合并，文本最多 400 字，完整错误以 SHA256 绑定）：

- 阶段 cluster，尝试 1（1 次）：cluster plan advances w0-v0012 on GPU 0 while w0-v0013 is inside a solver step there

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。

未闭合依赖：["resume_or_failure_investigation"]
