# kairos-wan21-native-calibration / ditserve-calib-wan21-70ff5c6-r1

实验状态：`failed`。归档状态：`deferred`。

[工作簿](results.xlsx) 保存完整配置、精确指标、请求记录、失败、预算、来源与清理凭据。
缺失不是零；归档校验不授予质量、性能或语义资格。

completed 仅表示执行完成。以下资格只对应已配置门槛；数值一致性通过不能替代语义/盲评或性能验证。

| 阶段 | 执行状态 | 尝试 | 数值 numeric | 语义 semantic | 性能 performance |
| --- | --- | --- | --- | --- | --- |
| cluster | failed | 1 | unverified | unverified | unverified |

指标概览（19 项中展示前 19 项；完整值及单位见工作簿）：

| 指标 | 已记录值 |
| --- | --- |
| run/communication_p50_ms | null |
| run/communication_p99_ms | null |
| run/completed | 19 |
| run/control_ms | 0.0 |
| run/deadline_attainment | 0.95 |
| run/denominator | "all frozen requests, including failed and blocked" |
| run/failed | 1 |
| run/incomplete | 0 |
| run/latency_p50_ms | 6659.003181383014 |
| run/latency_p95_ms | 41194.645470101386 |
| run/model_ms | 258357.70259052515 |
| run/observation_count | 10 |
| run/observation_ms | 356325.80842543393 |
| run/ontime | 19 |
| run/qualification_status | "unverified" |
| run/qualification_verified_requests | 0 |
| run/qualified_goodput_rps | null |
| run/qualified_ontime | 0 |
| run/requests | 20 |

错误与未通过/未验证门槛（3 项中展示前 3 项，每项最多 400 字；完整证据见工作簿）：

- 阶段 cluster，尝试 1：worker AssertionError: h must be divisible by world_size, got 12 and 8
- 阶段 unknown，尝试 unknown：worker AssertionError: h must be divisible by world_size, got 12 and 8
- 请求 video-calibration：worker AssertionError: h must be divisible by world_size, got 12 and 8

保存 20 条请求、2 条失败记录。

已确认不存在的原始文件：0；保留白名单文件：0；已回收字节：0。

原始数组、视频和完整事件日志不嵌入工作簿。已删除输入仅保留历史指标，无法恢复原始数据或重新评测。
工作目录的 run.json、收尾日志、账本及冻结来源是恢复/来源控制文件，不自动删除。

尚未闭合的评测/续跑依赖：["resume_or_failure_investigation"]
