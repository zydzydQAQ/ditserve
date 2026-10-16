# 结果保存与收尾

继续使用原位置的 [save-experiment-results skill](/home/zyd/.codex/skills/save-experiment-results/SKILL.md)，以及其[项目兼容约束](/home/zyd/.codex/skills/save-experiment-results/references/dit-rewrite.md)。技能规则保持不变，项目参考中的命令已同步原生入口；未批量迁移、清理现有结果。

实现集中在 `workers/ditserve/results/`：包入口负责新实验收尾，`compact.py` 负责旧目录的显式导出和清理凭据，`legacy.py` 负责原归档格式。正式命令是 `.cache/build/ditserve report` 和 `results`；Python API 使用 `ditserve.results.finalize/load/verify`。原顶层 `compact_results`、`result_storage`、`archive` 转发模块已移除，调用方分别改用 `ditserve.results.compact`（包含 `export_report`）和 `ditserve.results.legacy`；历史归档格式与哈希语义不变。质量和参考比较位于 `workers/ditserve/evaluation/`。

`configs/result_storage.json` 只控制旧 `export_report` API 的 `auto_export_reports` 和 `export_directory`。依赖闭合、活动保护、验证后清理及两文件交付是实现约束，不由该配置开关决定；原生 `report` 使用新 finalizer。

新实验长期交付位于 `results/<experiment>/<run>/`，只有 `results.xlsx` 和 `REPORT.md`。工作簿可独立回读完整配置、指标、请求、失败、预算快照、来源和清理凭据；类型、长整数、ID 和长字符串精确保留，超限分片/分表或报错。原始张量、视频、权重和完整事件日志不嵌入工作簿。Excel 所需 `openpyxl` 按需导入，可使用 `reports` 可选依赖或项目已有 `.cache/tools`。

`completed` 表示技术执行完成。报告分别展示数值 `numeric`、语义 `semantic`、性能 `performance` 的门槛状态；缺少对应证据时为 `unverified`。归档校验、数值一致性和语义/盲评是不同结论，失败与未评测状态原样保留。

## 工作目录与累计预算

默认工作目录为 `.cache/results-work/<experiment>/<run>/`。原始输入只保留到实际使用者结束：质量评测、盲评、续跑或故障调查需要哪些文件，就在 `run.json.artifacts[].dependencies` 中逐文件记录；全局依赖记录在 `retention.pending_dependencies`。不能仅凭账本空闲、执行成功或报告存在就清空依赖。

累计账本默认是工作目录外的 `.state/<experiment>/budget.json`，由 C++ 管理，`run.json.budget_ledger` 记录其位置。重试不能换账本绕过已用预算。工作簿保存该次运行的预算快照，不替代后续实验继续使用的累计账本。

C++ 先停止 worker、检查 GPU 占用、结算并释放预算锁，再调用 Python finalizer。运行/阶段仍为 `running`、`admitted`、`stopping`、`unsettled`，账本仍有活动会话或相关锁被占用时，收尾受保护或延期，不能通过修改账本强行归档。

## 导出、重试与精确清理

下面示例在项目根目录执行，`example/run01` 需换为实际实验和运行名。先保留原始数据导出，再在依赖确实结束后重试收尾；这些命令不启动模型或 GPU。

```bash
.cache/build/ditserve report .cache/results-work/example/run01 --output results/example/run01 --keep-raw
.cache/build/ditserve results verify results/example/run01
.cache/build/ditserve results inspect results/example/run01

# 依赖结束后，默认尝试清理符合条件的显式白名单文件。
.cache/build/ditserve report .cache/results-work/example/run01 --output results/example/run01
```

`--keep-raw` 只发布结果；其 `completed` 不表示原始数据已删除。省略该选项时，也只清理路径位于本次工作目录内、大小和 SHA256 匹配、依赖全部闭合的显式白名单文件。数组/媒体可直接登记；事件日志还需绑定源哈希、字节数、总记录数、聚合方法、指标和省略字段。禁止路径逃逸及符号链接；模型、账本、配置、冻结来源不自动删除。

依赖闭合使用空列表，或带实际原因的记录，例如 `[{"closed": true, "reason": "独立盲评已完成，结果已入工作簿"}]`。原始数据已有删除凭据后，只允许记录依赖闭合，不能改测量值、来源绑定或历史身份来通过校验。尚未发生的评测不能标为已完成。

finalizer 先写临时 Excel/报告并回读验证，再发布；删除前写入持久意图，删除后记录事实。导出、删除或依赖检查失败返回 `deferred`，保留工作目录并运行同一条 `report` 命令重试。活动会话返回 `protected`；锁冲突也可能返回 `deferred`，应先解决真实活动或占用。已完成归档只校验，损坏时不自动覆盖修复。

原生 `report` 未给 `--output` 时优先沿用收尾日志保存的目标，包括先前 `--results-root` 指定的位置；目标及工作目录身份仍由 finalizer 校验。仅执行命令不需要 `PYTHONPATH`；使用另一个已有解释器时传 `--python /absolute/path/to/python`。

`run.json`、`.finalization.json`、锁和冻结源码属于恢复/来源控制文件，当前实现保留在工作目录；必要凭据同时进入工作簿。报告及返回值记录保留文件、具体依赖、回收字节和工作目录占用。迁出 `results` 不等于释放空间，不应把仍有原始数据的工作目录宣称为已清空。删掉原始输入后只能读取历史指标，无法恢复或重新评测。

## 历史结果只读兼容

| 形式 | 支持与边界 |
| --- | --- |
| 新 `ditserve.results.v1` 两文件 | 完整类型回读、元数据哈希及报告一致性校验；旧报告模板按保存版本验证。 |
| `ditserve.compact-results.v1` | 读取 `manifest.json`、`data.json.gz`、工作簿；验证压缩证据及哈希。`pruned.json` 与定位文件用于核实已删除原始数据。 |
| `ditserve.archive.v1` | 读取单个 step 的压缩证据、工作簿及报告；归档根目录的 `archive.json`/`INDEX.md` 支持 `results verify`。`results inspect` 使用具体 step。 |
| `ditserve.sessions.report.v1` 两文件 | 回读旧类型化工作表及证据，禁止公式；没有独立历史哈希时，不能据此证明原始报告内容完整或重新授予科学资格。 |
| `ditserve.wan_breakdown.v1` 两文件 | 回读 path-array 类型化工作表，按原插入顺序重算 payload 哈希；不改用新格式的键排序哈希。 |
| 普通旧 JSON/report 目录 | 不猜测 schema、不重跑旧 runner；需要时显式 compact，保留原目录。 |

```bash
.cache/build/ditserve results verify /path/to/historical/archive
.cache/build/ditserve results inspect /path/to/historical/step
.cache/build/ditserve results compact /path/to/closed/legacy/run
```

`compact-results` 是 `results compact` 的兼容名称。历史 compact 的 `--prune-raw` 会选择该目录内全部可清理原始数组，因此只用于已逐项确认没有后续使用者的独立目录；活动账本检查不能代替依赖审查。已有清理凭据的归档不得覆盖。旧 gzip 可能是完整小文本或溢出指标的唯一副本，账本、来源快照、定位文件和凭据也可能仍被引用，均不为凑齐两文件而删除。新流程不要求旧实验原地续跑，也不自动执行历史迁移或清理。
