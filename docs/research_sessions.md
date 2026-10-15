# 真实 LLM → 图像双轮会话实验

本实验研究真实自回归 LLM 与 Sana 图像生成串联时，阶段调度是否改善整个会话。首批固定 trace seed `301`，两个外生负载参数 `0.5`、`0.9`，每条 trace 有 12 个会话，每个会话两轮，共比较三种策略、六次重放。不按性能门槛自动挑选胜者，不将单 seed 描述作为稳定收益结论。

迁移到 A100 后，环境验收显式使用 `MistralRuntime(model_path, device="cuda:0", profile="a100_fp32")` 与 `SanaRuntime(model_path, device="cuda:0", profile="a100_fp32")`。该配方要求 SM80，分别保持 Mistral FP16、Sana 全组件 FP32 和 MATH attention，状态所有权、KV 保存恢复及两步 SCM 语义不变。`SchedulingSana` 也接受 `a100_fp32`；其已有 `a100_bf16` 配方继续服务原 BF16 研究路径。

两个会话适配器的默认配方仍为 `v100_fp32`。本文后面的 `research-sessions` 研究监督入口保留旧 V100 UUID、资格和授权约束；A100 适配器验收不授予研究资格，新研究必须重新绑定协议、校准、qualify 与 GPU 预算。目标机环境验收步骤见项目根目录 `MIGRATION.md`。

## 输入与因果依赖

冻结的外生输入包含每个会话的首次到达、两轮用户输入、固定思考时间和生成 seed 字段。LLM 使用确定性 greedy 解码，不用记录的 `llm_seed` 做随机采样；trace seed 固定外生到达，`image_seed` 固定 Sana 随机过程。每轮真实执行 LLM prepare/prefill/decode，再以真实输出作为图像提示，执行图像 encode/denoise/decode。第二轮 LLM 使用本会话真实对话历史。

第二轮释放时刻为本会话第一轮图像实际完成时刻加 `think_ms`。在第一轮未完成时不建立就绪第二轮；失败则保留依赖阻塞。不能将所有第二轮预先放入队列，也不能用别的策略第一轮完成时刻决定其到达。不同策略第二轮到达不同是闭环会话的真实结果，共享的 trace hash 标识外生输入。

外生会话首次到达采用指数间隔，平均间隔为两轮独立服务估计总和除以负载参数；第一个会话从零开始。此参数不是实测 GPU 利用率。思考时间不算额外 GPU 工作。

## 冻结成本和截止时间

独立校准各完整阶段的服务成本，估计采用 `max(p95, 1.1 × mean)`，在所有策略之间冻结。六个阶段为 `llm_prepare`、`llm_prefill`、`llm_decode`、`image_encode`、`image_denoise`、`image_decode`。decode 阶段成本是完整 token decode 合计，denoise 阶段成本是完整图像步骤合计；单 token 与单图像步成本另用于执行器更新剩余时间。

所有轮使用相同相对截止窗口 `2.5 × sum(stage_ms)`，从本轮实际释放开始计时。成本预测仅用于排序，不保证期限。阶段 EDF 将同一窗口按独立阶段成本比例划分，阶段截止为释放时刻加累计比例窗口。

## 三种策略

| 策略 | 选择规则 |
| --- | --- |
| `session_serial` | 一轮从 LLM 到图像完成期间保持运行；思考间隔不占用调度器，可运行其他会话 |
| `stage_edf` | 就绪动作按其固定阶段截止时间排序 |
| `session_slack` | 就绪动作按 `本轮截止 − 当前时刻 − 当前轮全部剩余成本` 升序排序 |

三种策略使用相同合法动作边界和暂停粒度，不把更细抢占边界作为某一策略的隐含优势。平局依次按本轮截止、释放时刻、请求 ID 排序。未来到达、尚未解除依赖、未就绪及已结束请求不可参与选择。已超时请求继续执行或留下明确失败记录。

纯 CPU 接口：`make_trace(config, cost_model, load, seed)` 返回外生 trace；`release_turn(session, round_index, trace, previous=None)` 的 round index 为 0/1，产生的 row.round 为 1/2；`stage_deadline(row, stage, trace)` 返回冻结阶段截止；`select(ready, now, arm, active_id=None)` 返回 request ID 或 `None`。`active_id` 是当前轮而非跨思考时间的整个会话。

配置键为 `session_count`、`think_ms`、`deadline_multiplier` 和 `session_inputs`（每项恰有两条用户输入，可循环分配或逐会话提供）。执行器更新 `ready`、`status`、`stage_deadline_ms` 和 `remaining_total_ms`。

## 统计和资格

逐轮成功要求该轮图像完成且不超过本轮 deadline；整会话成功要求两轮分别成功，只有最后一轮按时不够。第二轮 deadline 是最终完成参照，不能抹去第一轮违约。会话延迟从外生首次到达到第二轮图像完成，包含思考时间。

报告包含逐轮与会话 goodput、吞吐、TTFT、token 间隔 p95、图像等待、轮与会话端到端 p95，以及质量和容量原始检查。TTFT 默认从轮释放至首 token，另记录从 LLM 实际开始至首 token。图像等待从 LLM 输出可供图像阶段使用到图像阶段实际开始。goodput 分母为实际 trace 墙钟时间，包括空闲、思考和最终排空，不扣减日志开销；失败、缺失、依赖阻塞不被隐藏。p95 采用线性插值，并给出样本数。

只有完整的 12 会话 × 2 轮记录、三个策略共享外生 trace/设备/成本身份、输出参考检查通过且监督完整的组可作配对比较。记录保持每轮生成 token IDs、图像输出哈希及原始质量检查；数值或 token 一致不等于独立语义质量评审。全部失败和未开始状态保留。

`qualify` 保存真实 LLM/Sana 执行、共驻容量及暂停恢复参考检查；`pilot` 保存首批六次重放；`state` 保存真实 KV 与图像中间状态的容量压力和恢复检查。如果真实请求规模未形成状态压力，报告 `not_applicable`，不通过填充无关张量制造压力。

## 输出和收尾

trace 内只保存当前 trace 的独立 partial，累计 records 仅在 trace 外保存。大数组和逐事件日志是有明确依赖期限的临时输入，不能在计时 GPU 路径里做重型工作簿导出或全量哈希。

`build_report(root, ledger=None)` 只读；`report(root, ledger=None, retention=None)` 更新根目录 `results.xlsx` 和 `REPORT.md`。工作簿完整保存逐轮/逐会话标量、配置、原始质量/容量字段、预算、来源哈希与清理事实，字符串按字面量写入，精确整数、浮点十进制表示及超长字符串显式保留类型/分片；发布前逐单元格回读验证。活动账本期间延期重型 Excel 导出。

事件日志流式汇总为数量、种类、状态、时间范围，标记仅聚合保留；原始事件顺序和载荷清理后不能重建。工作簿不保存张量、视频或二进制。报告不执行删除；监督器只在阶段技术完成、预算已结算、实际评测/恢复依赖关闭且归档验证后，按文件路径、大小、SHA256 的明确白名单清理本轮原始文件。清理收据经再次回读并入工作簿。

控制与源码依赖也结束后，监督器可显式退休这些输入，使 run 内最终只保留 `results.xlsx`、`REPORT.md`，独立累计账本保留在 run 外。`archived(root)` 与 `load_archive(root)` 可从完整带类型工作簿恢复冻结配置、阶段指标与清理事实，归档查看不会重算或启动实验。控制文件清理后用 `report(..., retention=receipt, archive_only=True)` 只更新清理表和报告，即使部分删除失败也不因输入缺失覆盖既有指标。仍有真实消费者的来源或控制文件必须保留并说明例外。

## 入口、自动顺序与状态判读

配置为 `configs/research_sessions.json`，默认 run 为 `results/research-sessions/run01`，独立累计账本为 `results/research-sessions/gpu-budget.json`。在仓库根目录执行以下入口；`preflight` 是只读检查，`prepare` 绑定独立协议与账本，`continue` 是获授权后的自动执行入口。

```bash
PYTHONPATH=python .venv-v100/bin/python -m ditserve research-sessions preflight --phase qualify
PYTHONPATH=python .venv-v100/bin/python -m ditserve research-sessions prepare
PYTHONPATH=python .venv-v100/bin/python -m ditserve research-sessions continue
PYTHONPATH=python .venv-v100/bin/python -m ditserve research-sessions report
```

入口支持 `--config`、`--output`、`--ledger` 指定独立位置；单阶段入口为 `run --phase qualify|pilot|state`，不会绕过依赖、隔离、冻结源码或 CPU 验证。配置及源码必须通过最终 CPU 验证绑定，源码变化后需要重新测试并记录验证。

自动顺序为：等待旧 `research-scheduling-next/run01/completion` 在独占监督下完成并释放 V100，确认本轮 CPU 验证与设备条件，然后依次执行 `qualify → pilot → state → 报告与依赖核验 → 显式归档清理`。旧实验后续已取消的阶段不会在此入口恢复。新实验仅使用配置绑定的 V100，不因其他卡空闲自动换卡。

累计 GPU 预算上限为 `null`，表示已获授权的无限累计额度；实际占用仍完整计费，包含加载、预热、失败与清理。每阶段默认 48 小时 watchdog 用于防止进程挂起，是单次运行保护，不是恢复旧的累计预算上限。

| 状态 | 含义与后续行为 |
| --- | --- |
| `waiting` | 旧 completion、设备或最终 CPU 验证尚未满足；继续等待，不提前加载模型 |
| `passed` / `technical_complete=true` | 该阶段技术证据完整；不等于调度策略已获得稳定性能收益 |
| `state: not_applicable` | 已测真实并发状态未出现容量压力，不构造假压力证明收益 |
| `state: stopped` 且 `technical_complete=true` | 容量诊断完整但无可行救援，或清理后 keep 已能恢复；保留负结果，监督结算且依赖关闭后仍可归档 |
| `failed` / `incomplete` | 正确性、隔离、容量或执行异常导致未完成；保留调查输入与原始状态，不自动改精度、模型、输入或期限 |
| `archive_only=true` | 已完成的轻量归档；`report` 和 `continue` 只读原指标与清理事实，不重跑、不重新授予资格 |

只有完整依赖关闭才进行最后两文件收尾；若清理失败，保留可重试凭据和未删文件，并明确报告，不把删除意图当作已删除。
