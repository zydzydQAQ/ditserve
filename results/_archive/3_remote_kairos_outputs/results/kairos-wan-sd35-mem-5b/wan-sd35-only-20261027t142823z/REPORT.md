# kairos-wan-sd35-mem-5b / wan-sd35-only-20261027t142823z

实验状态：`failed`。归档状态：`deferred`。

[工作簿](results.xlsx) 保存完整配置、精确指标、请求记录、失败、预算、来源与清理凭据。
缺失不是零；归档校验不授予质量、性能或语义资格。

completed 仅表示执行完成。以下资格只对应已配置门槛；数值一致性通过不能替代语义/盲评或性能验证。

| 阶段 | 执行状态 | 尝试 | 数值 numeric | 语义 semantic | 性能 performance |
| --- | --- | --- | --- | --- | --- |
| colocate | failed | 1 | unverified | unverified | unverified |

指标概览（19 项中展示前 19 项；完整值及单位见工作簿）：

| 指标 | 已记录值 |
| --- | --- |
| run/communication_p50_ms | null |
| run/communication_p99_ms | null |
| run/completed | 0 |
| run/control_ms | 0.0 |
| run/deadline_attainment | null |
| run/denominator | "all frozen requests, including failed and blocked" |
| run/failed | 0 |
| run/incomplete | 0 |
| run/latency_p50_ms | null |
| run/latency_p95_ms | null |
| run/model_ms | 0.0 |
| run/observation_count | 0 |
| run/observation_ms | 0.0 |
| run/ontime | 0 |
| run/qualification_status | "unverified" |
| run/qualification_verified_requests | 0 |
| run/qualified_goodput_rps | null |
| run/qualified_ontime | 0 |
| run/requests | 0 |

错误与未通过/未验证门槛（1 项中展示前 1 项，每项最多 400 字；完整证据见工作簿）：

- 阶段 colocate，尝试 1：worker OutOfMemoryError: CUDA out of memory. Tried to allocate 3.51 GiB. GPU 0 has a total capacity of 39.49 GiB of which 2.06 GiB is free. Process 105 has 17.76 GiB memory in use. Including non-PyTorch memory, this process has 19.66 GiB memory in use. Of the allocated memory 19.04 GiB is allocated by PyTorch, and 125.81 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memor…

保存 0 条请求、1 条失败记录。

已确认不存在的原始文件：0；保留白名单文件：0；已回收字节：0。

原始数组、视频和完整事件日志不嵌入工作簿。已删除输入仅保留历史指标，无法恢复原始数据或重新评测。
工作目录的 run.json、收尾日志、账本及冻结来源是恢复/来源控制文件，不自动删除。

尚未闭合的评测/续跑依赖：["resume_or_failure_investigation"]
