# 结果保存与收尾

本项目以本页及 [`ditserve.results`](../workers/ditserve/results/__init__.py) 的实现和测试作为结果保存规范，不依赖本机额外安装的技能文件。

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

## Workflow 实验的关键证据保留

[`workflow_retention`](../workers/ditserve/evaluation/workflow_retention.py) 为 `workflow_report`、`workflow_serving_report --final` 和 serving 驱动的最终归档提供默认精简。预览只生成预览文件，不触发精简；`--keep-raw` 显式保留冗余原始文件。

长期保留正常实验的指标、JSON、日志、配置、冻结源码、原始 `.nsys-rep` / `.ncu-rep`、latent、embedding 和诊断证据。视频默认保留哈希、形状和数值比较结果；需要原始视频开展人工质量评估时，须显式开启下述视频保存选项。自动精简仍不能把已显式保留、待评测的视频当作冗余数据删除。

仅以下已核实的冗余数据可以进入逐文件白名单：同时有同名前缀原始 capture 和有效解析 JSON 的 SQLite 导出；profiling 的重复输出数组；E4 输出传输测速数组；与保留副本 SHA256 完全相同的旧编码 embedding。删除前记录路径、字节数、SHA256、原因及保留的替代证据，再经 Excel/报告回读验证、活动检查和精确匹配后删除。不会按后缀扫描删除全部数组，也不会修改冻结源码和原始测量 JSON。

历史工作目录已有规范 `run.json` 时，可以先查看清理计划，再执行精简；以下命令只使用 CPU，不重跑实验：

```bash
PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_retention \
  --root /mnt/ssd0/zyd/ditserve-data/workflow/run01 --dry-run
PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_retention \
  --root /mnt/ssd0/zyd/ditserve-data/workflow/run01
```

执行环境需安装 `openpyxl`，也可使用现有运行镜像的 `/opt/venv/bin/python`。`run02` 使用相同入口，只需替换根目录。输出中的回收字节来自实际删除凭据，不能将候选大小当成已释放空间。活动 study、预算会话或驱动/后处理锁被占用时保持保护；未闭合依赖继续保留。serving 驱动在释放 `driver.lock` 后才进入精简，避免归档重新获取同一把锁。

## 默认不保留原始视频

2026-10-22 根据用户“删掉，并修改代码中保存视频的部分”的明确要求，实验默认不再长期保存完整 RGB 视频数组。配置文件使用 `save_videos: false`；需要完整视频进行后续质量评估时显式开启。CLI 编码/回放工具使用 `--save-videos`；E4 输出传输实验默认 `--output-mode check`，仅显式 `--output-mode npy` 测量落盘输出。

SP 诊断在每个 prefix 的内存作用域内保留参考视频，仍计算完整逐数组误差与 exact 门槛；elastic 跨进程对照使用受生命周期管理的临时参考文件，全部子进程回收后清理。并发及 SP2 性能实验继续记录视频哈希，latent/embedding 和非视频诊断数组照常保留。编码实验需要重建参考视频时，先用保留的 latent 额外解码并严格匹配原始视频哈希、形状和 dtype，再执行原有数值比较；重建独立记入非性能阶段，不能当作新性能收益。

原生 `acceptance`、`block_exact`、`sp_equivalence` 的数值阶段通过 `video_outputs` 显式声明临时视频，并默认 `save_videos: false`。原有完整数组 `evaluate` 和跨 rank 共享参考继续执行；视频在阶段结束后清理，异常时先回收 worker。视频哈希及清理回执保留，但临时文件不注册为永久 artifact，latent 仍保存到原目录。普通模型生成输出不受此实验配置影响。历史冻结副本保持原样，后续实验应冻结包含本次修改的新版本。

本次历史视频删除单独记录在 `/mnt/ssd0/zyd/ditserve-data/maintenance/video-removal-20261022/`：`plan.json` 记录逐文件 SHA-256、形状、大小及用户授权，`deleted.jsonl` 记录实际删除，`receipt.json` 记录回收量与保留文件检查。原始测量 JSON、冻结源码和工作簿不重写，不把未完成的语义质量评估标记为通过；历史路径可能指向本次已明确删除的视频。已归档的耗时、哈希和比较结论仍可阅读，但这些视频不能再用于画面复核或原始视频指标重算。

本次实际删除 **374 份**独立视频数组，回收 **41,642,721,280 字节（38.782806 GiB）**：run01 为 20 份，run02 为 36 份，run03 为 318 份（含 16 份小尺寸验收视频）。[删除凭据](/mnt/ssd0/zyd/ditserve-data/maintenance/video-removal-20261022/receipt.json) 的 SHA-256 为 `ccaa744c6afe2af1c0274c7363eab59b2c0efb4218d473df006886f75d4160ec`；16 项控制文件及归档绑定未改变，复查未发现剩余独立视频数组。21,735 项保留文件的元数据检查发现一项并发编辑：`run03/maintenance/final-integrity-occupancy-r015.py`，已单独记录且未覆盖；首次删除脚本在此处停止验收，后续独立核对确认全部删除已完成、其余保留文件与所有控制/归档哈希不变，原失败日志继续保留。

验证覆盖默认不落盘、显式保留、完整视频误差、参考哈希不符拒绝比较、临时文件正常/异常清理、共享参考和归档无悬空 artifact。相关 Python 测试通过，C++ 构建及 5/5 CTest 通过；配置矩阵的过时预期已与当前 RTX4090 成本配置对齐，4 项配置测试通过。本次验证使用 CPU 与 mock，不重复运行 GPU 性能实验；不能将本次存储策略修改称为新增 GPU 加速结果。

已有删除凭据的运行保留原始规范 `run.json` 和 `ANALYSIS.html`，再次运行正式报告入口会复用并校验归档，不能重建时间戳、指标或来源哈希。`--keep-raw` 不能恢复已删除文件；进一步分析请使用保留证据，新的测量使用新的运行目录。

## Latent、embedding 与诊断数组的保留

最终 latent 按独立参考保留来源映射，包含模型、配置、种子、步数、参考方法和原始数值资格。内容完全相同的参考可以共享物理文件；失败方案产生的不同结果仍单独保留，不能与通过方案合并。1 步、4 步 prefix 的末端仅标为阶段检查点，不能称作完整 20 步终态。提取 NPZ 内的 `latent_after.npy` 时核对成员唯一性、形状、dtype、有限值、数组内容 SHA-256 和序列化文件 SHA-256。

embedding 保留基线及各编码方案的原路径，只对完整文件 SHA-256 相同且逐字节比较一致的副本使用硬链接去重。CPU 线程数、精度或编码方案不同，不能仅凭同一 prompt/seed 判定相同。历史数据只读使用；后续实验使用新目录，避免原地覆盖共享 inode。

逐步 latent、各层激活和 Q/K/V/FFN 仅在定位结论已明确、没有后续重放依赖时进入逐文件清理清单。2026-10-22 的专项审计发现 `sp-linear-replay`、`sp-linear-candidates` 仍绑定 96 个数组，因此这些文件及其原路径、哈希必须保留。已拒绝的 tile/selective 候选保留失败指标与端点，不能将清理解释为修复通过；弹性迁移的 `.pt` 状态包也不在本次清理范围。

本次用户授权的专项操作及逐文件凭据位于 `/mnt/ssd0/zyd/ditserve-data/maintenance/tensor-retention-20261022/`。脚本先等实验、预算和归档锁释放，再在锁内重读最新输入绑定；归档与规范输入匹配后才提取端点、去重和删除。删除前持久保存计划与端点映射，完成后重新校验保留数组、历史 JSON 和归档。该专项清理不改变现有自动归档的白名单规则，不会把所有中间数组默认视为可删。

run01、run02 已完成精确去重：保留 88 个原有数组路径，36 个路径改为引用相同内容的硬链接，实际回收 **380,387,328 字节（362.765625 MiB）**；798 项历史 JSON/归档哈希不变。[批次回执](/mnt/ssd0/zyd/ditserve-data/maintenance/tensor-retention-20261022/run01-run02/receipt.json)。run03 由同目录的 `finish.py` 独立后台执行：等待 occupancy 实验和归档完成，清理后重新持锁复核所有存活数组及最新输入绑定，通过后生成总 `receipt.json`。`status.json` 保存当前阶段，`background.log` 保存日志；只有完成回执中的实际回收量才能计为释放空间，不能使用候选大小替代。

2026-10-22 16:08（北京时间），上述后台任务已全部完成并通过独立复核。三个 run 合计删除 **4,568 个**无重放依赖的中间文件，106 个保留路径使用精确内容硬链接，实际回收 **68,658,552,832 字节（63.943260 GiB）**。246 个原有数组路径保留；run03 的 240 条端点来源映射保留为 36 份不同内容的文件，明确区分完整 20 步终态和部分 prefix 端点；96 个重放输入逐路径验证通过。两批共 4,259 项历史 JSON/归档检查通过。[总回执](/mnt/ssd0/zyd/ditserve-data/maintenance/tensor-retention-20261022/receipt.json) SHA-256 为 `91680e2cdeed86052c876f2b4d64093a3d7817328ef8cf9969241c5592c55bec`；[独立复核](/mnt/ssd0/zyd/ditserve-data/maintenance/tensor-retention-20261022/independent-final-verification.json)同时检查了删除路径不存在、全部保留数组内容、36 份端点 payload 和锁内最新输入绑定。加上此前视频删除，本次两项专项操作累计回收 **102.726067 GiB**；这是存储回收量，不是 GPU 性能收益。

## 工具、状态包与冻结副本的第二轮精简

2026-10-22 22:01（北京时间），在用户继续精简的授权下，完成第二轮专项清理，实际回收文件分配空间 **3,014,688,768 字节（2.807648 GiB）**：删除 10 份 Nsight 安装包/下载分片残留，回收 878,112,768 字节；删除 52 份已完成验证的迁移状态包，回收 809,709,568 字节；删除已封存的 r002–r017 构建目录内 5,184 个普通文件和 16 个可执行文件别名，回收 696,401,920 字节；7,654 个冻结源码/检查点重复路径改用完整内容一致的硬链接，回收 630,464,512 字节。按唯一 inode 计算，四项互不重复，目录块与新增审计凭据不计入上述文件回收量。[工具回执](/mnt/ssd0/zyd/ditserve-data/maintenance/second-compaction-20261022/tools/receipt.json)与[数据及构建回执](/mnt/ssd0/zyd/ditserve-data/maintenance/second-compaction-20261022/execution/receipt.json)记录实际操作。

状态包保留 12 份代表样本，覆盖现有序列化 schema、阶段、数学策略及迁移边界；被删除的具体历史包无法再次导入，但原指标、来源与 SHA 记录保留。原 acceptance 的失败启动仍保留失败状态；其中 3 份退役包的历史 `closed:false` 标记未改写，独立[闭合附录](/mnt/ssd0/zyd/ditserve-data/maintenance/state-packet-retention-20261022/acceptance-block-closure-appendix.json)逐份绑定实际导出/导入 SHA，以及恢复后视频和 latent 的零容差通过证据。此判断仅用于这些包的数值验证和代表样本保留，不授予语义质量或在线调度资格。

执行全程持有三个 run 的驱动、预算和归档等锁，并重读当前依赖；清理后 4,469 项历史 JSON/归档、452 个当前输入绑定及 282 个保留数组/端点路径校验通过。96 个历史重放数组仍须保留原路径，下一轮 r018 最终审计会再次读取历史绑定。所有检查点、冻结可执行文件及构建日志保留；r001 未封存构建和新版本目录不在删除范围。硬链接保持原路径和内容，但 inode 与时间戳可能统一，冻结目录须继续只读使用。已安装的 Nsight 提取目录保留，清理前后版本检查通过；模型与 GPU 实验均未在本轮修改或执行。

22:02:59 的[实际磁盘统计](/mnt/ssd0/zyd/ditserve-data/maintenance/second-compaction-20261022/storage-after.json)显示，实验/结果/工具数据为 **10.474770 GiB**，仓库、两套模型和数据合计 **44.637001 GiB**，加上镜像归档及专用 Docker 数据目录合计 **54.588245 GiB**。统计使用实际分配块并去重硬链接；新增审计凭据也占空间，因此目录净下降与删除文件回执总量不同。

22:04:51 的[独立验收](/mnt/ssd0/zyd/ditserve-data/maintenance/second-compaction-20261022/independent-verification.json)通过，未发现错误；另行验证了 16 份源码 seal、53 份 native manifest、80 个检查点绑定文件与全部删除路径。两部分操作及磁盘快照由[本轮总回执](/mnt/ssd0/zyd/ditserve-data/maintenance/second-compaction-20261022/receipt.json)关联，旧回执与历史归档未覆盖。

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
