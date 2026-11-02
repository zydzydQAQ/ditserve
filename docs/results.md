# 结果保存与收尾

结果实现位于 `workers/ditserve/results/`。正式命令为 `.cache/build/ditserve report` 和 `results`；Python API 为 `ditserve.results.finalize/load/verify`。

新实验默认只发布 `results/<experiment>/<run>/YYYYMMDD-<run_id>.md` 与 `summary.json`。run_id 已有日期前缀时直接使用，不重复加日期；否则取最后一次预算会话结束的 UTC 日期，无会话时取首次归档日期。JSON schema 为 `ditserve.results-summary.v1`，保留配置/来源哈希、去重模型配置、GPU 身份、各策略与模态的标量指标、资格、失败摘要和清理凭据；不复制完整配置、逐请求、事件、源码清单或采样序列。

`load/verify` 自动识别新旧格式。校验只证明保存内容和报告一致；`completed` 只表示执行完成，数值、语义、性能资格缺证据时仍为 `unverified`。摘要明确不能用于续跑或原始重评。确需完整 Excel 元数据时，可显式调用 `finalize(work, destination, format="full")`；默认路径不依赖 openpyxl，已有归档不转换格式。

## 工作目录与依赖

默认工作目录为 `.cache/results-work/<experiment>/<run>/`。累计账本由 C++ 管理，重试不能换账本绕过已用预算。worker 停止、GPU 占用核验和预算结算先于 Python finalizer；活动会话、锁冲突、未闭合依赖或源文件变化会返回 `protected`/`deferred`，不能修改账本或测量值强行归档。

质量评测、续跑、重评和故障调查依赖必须逐文件登记，原始输入保留到实际使用者结束；报告存在不等于依赖闭合。`--keep-raw` 只发布结果。默认清理只处理工作目录内、大小和 SHA256 匹配、依赖已闭合的显式白名单文件，并记录删除意图、事实和回收字节。日志删除还必须具有绑定原哈希的完整聚合。完整 `run.json`、账本和控制文件不会因发布摘要而自动删除。

```bash
.cache/build/ditserve report .cache/results-work/example/run01 --output results/example/run01 --keep-raw
.cache/build/ditserve results verify results/example/run01
.cache/build/ditserve results inspect results/example/run01
```

收尾失败可沿用同一命令恢复。依赖闭合不允许改变测量记录；已完成报告不可通过改写输入修订。删掉原始输入后只能读取保留指标，无法恢复或重新评测。

## Kairos output 留存

共享盘清理由项目留存流程按范围单独执行：先回读验证报告，生成摘要和清单，再删除并复核。活动作业、待续跑/重评/调查及校验失败的目录受保护。新格式以日期报告和摘要为长期交付；历史 Excel/REPORT 仍按原格式校验，不能因格式升级批量覆盖或删除。模型、镜像及其他项目不属于结果归档的删除目标。

## 旧格式只读兼容

`compact.py` 与 `legacy.py` 保留旧 compact/archive/session/wan-breakdown 的校验与读取；显式旧格式导出仍保留兼容入口。历史硬件采集/重绘脚本已移除，读取器和固定 fixtures 保留。旧结果不授予当前实现性能或质量资格，也不支持重建已删除原始输入。

```bash
.cache/build/ditserve results verify /path/to/historical/archive
.cache/build/ditserve results inspect /path/to/historical/step
.cache/build/ditserve results compact /path/to/closed/legacy/run
```
