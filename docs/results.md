# 结果保存与收尾

结果实现位于 `workers/ditserve/results/`：包入口负责新实验 finalizer，`compact.py` 负责显式导出和删除凭据，`legacy.py` 负责只读旧格式。正式命令为 `.cache/build/ditserve report` 和 `results`；Python API 使用 `ditserve.results.finalize/load/verify`。质量与参考比较位于 `workers/ditserve/evaluation/`。

新实验长期交付为 `results/<experiment>/<run>/results.xlsx` 和 `REPORT.md`。工作簿包含完整配置、指标、请求、失败、预算快照、来源与清理凭据；原始张量、视频、权重和完整事件日志不嵌入工作簿。`completed` 仅表示技术执行完成，数值、语义和性能资格分别判定，缺少证据时为 `unverified`。

## 工作目录与依赖

默认工作目录为 `.cache/results-work/<experiment>/<run>/`。累计账本由 C++ 管理，重试不能换账本绕过已用预算。worker 停止、GPU 占用核验和预算结算先于 Python finalizer；活动会话、锁冲突、未闭合依赖或源文件变化会返回 `protected`/`deferred`，不能修改账本或测量值强行归档。

原始输入只保留到实际使用者结束。质量评测、续跑、重评和故障调查依赖必须逐文件登记；报告存在不等于依赖闭合。`--keep-raw` 只发布结果，不删除原始数据。默认清理只处理工作目录内、大小和 SHA256 匹配、依赖已闭合的显式白名单文件，并记录删除意图、事实和回收字节。

```bash
.cache/build/ditserve report .cache/results-work/example/run01 --output results/example/run01 --keep-raw
.cache/build/ditserve results verify results/example/run01
.cache/build/ditserve results inspect results/example/run01
```

失败时沿用同一命令重试，不覆盖已核验的历史报告。删掉原始输入后只能读取保留指标，无法恢复或重新评测。

## Kairos output 留存

jjjkairos 收尾遵循项目 `ditserve` skill：只保留已核验的两文件报告、关键配置/指标/错误摘要、BUILD_INFO、job spec 和删除凭据。清理必须先回读报告、生成 manifest 和 plan，再按清单 apply/verify；不得用笼统 `rm -rf` 代替。活动作业、待续跑、待重评、待调查或校验失败的目录受保护。

本次用户已授权清理本地及远端旧 output，并明确旧原始数据不能续跑或重评；模型、镜像和历史 builds 不在该授权范围内。新实验未完成前，不删除其运行目录或原始依赖。

## 旧格式只读兼容

`compact.py` 与 `legacy.py` 继续读取旧 compact/archive/session/wan-breakdown 格式并验证原哈希语义。旧结果只读兼容不代表旧数据仍是当前模型的有效证据，也不提供旧 runner 原地续跑。固定 fixtures 用于验证读者，不承担性能或质量结论。

```bash
.cache/build/ditserve results verify /path/to/historical/archive
.cache/build/ditserve results inspect /path/to/historical/step
.cache/build/ditserve results compact /path/to/closed/legacy/run
```
