# 20261103-d20261103-quality-wan21-4f0e5ff-s201-smoke-fixed7

实验 `kairos-wan21-native-serving-s201-l1p0`，运行 `d20261103-quality-wan21-4f0e5ff-s201-smoke-fixed7`：执行 `completed`，归档 `deferred`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`b98d4d17b51881db4208685e2e264399d74cda605709ae9440ff321ab542e130`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / fold-cfg2sp3 | 全部 | completed | 4 | 0.5 | 74177.9 | 0.0215353 | 未验证 |
| cluster / fold-cfg2sp3 | image | completed | 0 | 未记录 | 未记录 | 0 | 未验证 |
| cluster / fold-cfg2sp3 | video | completed | 4 | 0.5 | 74177.9 | 0.0215347 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 4 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。

保留原因：
- `/shared/output/DITSERVE/.cache/results-work/kairos-wan21-native-serving-s201-l1p0/d20261103-quality-wan21-4f0e5ff-s201-smoke-fixed7/attempts/cluster/attempt-1/cell-0/artifacts/wan0/output-31abb19b212c69d9-18239.npy`：dependencies remain open: [{"closed":false,"consumer":"native_quality"}]
- `/shared/output/DITSERVE/.cache/results-work/kairos-wan21-native-serving-s201-l1p0/d20261103-quality-wan21-4f0e5ff-s201-smoke-fixed7/attempts/cluster/attempt-1/cell-0/artifacts/wan0/output-d4c84c0f5578d6d1-18262.npy`：dependencies remain open: [{"closed":false,"consumer":"native_quality"}]
- `/shared/output/DITSERVE/.cache/results-work/kairos-wan21-native-serving-s201-l1p0/d20261103-quality-wan21-4f0e5ff-s201-smoke-fixed7/attempts/cluster/attempt-1/cell-0/artifacts/wan0/output-bbba37ac5bfc3a4e-18285.npy`：dependencies remain open: [{"closed":false,"consumer":"native_quality"}]
- `/shared/output/DITSERVE/.cache/results-work/kairos-wan21-native-serving-s201-l1p0/d20261103-quality-wan21-4f0e5ff-s201-smoke-fixed7/attempts/cluster/attempt-1/cell-0/artifacts/wan0/output-b6ffaa0cbdf9faa6-18308.npy`：dependencies remain open: [{"closed":false,"consumer":"native_quality"}]
