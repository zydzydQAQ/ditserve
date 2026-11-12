# kairos-wan22-native-calibration / ditserve-serve-wan22-4d63638-s204-bursty-r1-calibration

实验状态：`completed`。归档状态：`completed`。

[工作簿](results.xlsx) 保存完整配置、精确指标、请求记录、失败、预算、来源与清理凭据。
缺失不是零；归档校验不授予质量、性能或语义资格。

completed 仅表示执行完成。以下资格只对应已配置门槛；数值一致性通过不能替代语义/盲评或性能验证。

| 阶段 | 执行状态 | 尝试 | 数值 numeric | 语义 semantic | 性能 performance |
| --- | --- | --- | --- | --- | --- |
| cluster | completed | 1 | unverified | unverified | unverified |

指标概览（19 项中展示前 19 项；完整值及单位见工作簿）：

| 指标 | 已记录值 |
| --- | --- |
| run/communication_p50_ms | null |
| run/communication_p99_ms | null |
| run/completed | 20 |
| run/control_ms | 0.0 |
| run/deadline_attainment | 1.0 |
| run/denominator | "all frozen requests, including failed and blocked" |
| run/failed | 0 |
| run/incomplete | 0 |
| run/latency_p50_ms | 18940.054564271122 |
| run/latency_p95_ms | 40169.71086729318 |
| run/model_ms | 280871.5613251552 |
| run/observation_count | 10 |
| run/observation_ms | 372713.7452401221 |
| run/ontime | 20 |
| run/qualification_status | "unverified" |
| run/qualification_verified_requests | 0 |
| run/qualified_goodput_rps | null |
| run/qualified_ontime | 0 |
| run/requests | 20 |

保存 20 条请求、0 条失败记录。

已确认不存在的原始文件：0；保留白名单文件：0；已回收字节：0。

原始数组、视频和完整事件日志不嵌入工作簿。已删除输入仅保留历史指标，无法恢复原始数据或重新评测。
工作目录的 run.json、收尾日志、账本及冻结来源是恢复/来源控制文件，不自动删除。
