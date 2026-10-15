# DiT serving 三方向：运行与证据说明

本入口按 **1 步内抢占 → 2 缓存状态与显存压力 → 3 交互更新与播放** 顺序，使用现有机器完成有限验证。协议为 [research_directions.json](/home/zyd/code/dit-rewrite/configs/research_directions.json)，命令实现为 [runner.py](/home/zyd/code/dit-rewrite/python/ditserve/directions/runner.py)。本文件说明当前实现与恢复方式，不维护实时性能排行榜。

截至 2026-10-14，run01 的 GPU 阶段均已结算，账本 `active=null`。已完成结果如下；具体指标与资格以 [REPORT.md](/home/zyd/code/dit-rewrite/results/research-directions/run01/REPORT.md)、原始阶段报告和 supervisor 记录为准。

- **D1**：`profile`、`correctness`、`preemption` 通过；`online` 因方向预算耗尽，以 `incomplete`、`eligible=false` 停止。完成 57/60 次 replay，其中 11 组包含同 trace 的全部五种策略，可用于配对描述；其余两次完整 replay 属于未齐组，不进入完整组比较。在线矩阵未全部验证，不能提升为方向通过或无收益。
- **D2**：选定 81 帧，`screen` 6/6、`qualify` 20/20 通过，资格覆盖 8 个开发输入的正反序及 4 个留出输入。真实压力 K=1/2/4/8/16 中 keep 均能接纳 urgent，因此按冻结规则停止，没有执行容量边界后的四动作矩阵。另已完成 `state_cost` 六项成本补测及输出检查；它只提供单输入、每臂两次的成本证据，不授予容量、状态策略或 SP 资格。
- **D3**：四组官方参考正确性比较通过；包含稳态去噪、上下文处理、解码和传输的单会话路径为 **5.61 FPS**，低于固定 16 FPS 门槛，`profile` 合法停止。未运行 `update`、双会话或语义盲评；本机固定规格的实时性未成立，serving 策略收益仍未验证。

最终 CPU 回归 **1186 项全部通过**。新增 GPU 用量 **11.0193/24 小时**，D1/D2/D3 分别为 7.9884/2.9270/0.1039 小时，reserve 未使用；未转借方向余额。首次收尾回归因工具目录中的 NumPy 与 Python 版本不兼容而失败，修正导入路径后完整重跑，原失败日志保留。见 [回归记录](/home/zyd/code/dit-rewrite/results/research-directions/run01/validation/cpu-regression-completion.json)与[账本及源码快照审计](/home/zyd/code/dit-rewrite/results/research-directions/run01/validation/gpu-completion-audit.json)。

D3 稳态样本来自单条轨迹的 6 个 chunk、共 72 帧：完整 chunk 耗时均值/p95 为 **2.1393/2.1430 秒**；去噪与上下文为 1.4973/1.4991 秒，解码及传输为 0.6420/0.6442 秒。均为 n=6 的描述性统计。只算去噪与上下文的派生吞吐为 8.01 FPS，仍不能替代完整路径门槛。本轮尚未启动实际播放器会话，卡顿、条件到播放延迟及语义新鲜度没有实测结论。详见 [D3 最终审计](/home/zyd/code/dit-rewrite/results/research-directions/run01/validation/d3-final-audit.json)。

## 本机入口与分阶段继续

以下命令使用本机现有环境与绝对路径。在同一个 shell 中定义入口；这一步不运行实验：

```bash
directions() {
  PYTHONPATH=/home/zyd/code/dit-rewrite/python \
    /home/zyd/code/dit-rewrite/.venv-previs/bin/python \
    -m ditserve research-directions "$@" \
    --config /home/zyd/code/dit-rewrite/configs/research_directions.json \
    --output /home/zyd/code/dit-rewrite/results/research-directions/run01 \
    --ledger /home/zyd/code/dit-rewrite/results/research-directions/gpu-budget.json
}
```

查看与准备命令：

```bash
directions report
directions preflight --direction 1
directions prepare
```

`report` 只读实验记录并更新汇总，不启动模型、不写回 GPU 账本。`preflight` 查询设备占用、模型文件及验证清单，不运行模型。`prepare` 冻结配置并初始化或校验新增预算账本，只在没有活动会话时执行；现有 run01 不需要反复 prepare。D1/D2 使用配置中的主 Python，D3 工作进程使用 LongLive 专用 Python。

`preflight --direction 1/2/3` 只用相应方向所需模型决定准入。LongLive 的权重下载完成不足以显示可用：其 provisioning manifest 还必须有通过的 `environment_validation`，指向匹配 Python 路径、绝对记录路径及 SHA256 的 CPU 二进制/后端验证记录。D1/D2 不因未完成的 LongLive 环境而被阻挡。

阶段顺序如下。**已有 run 从下一项未执行且前置证据合格的阶段继续，不重跑表中的已完成项。**

| 方向 | 依次执行的 phase | 前置要求 |
| --- | --- | --- |
| 1 | `profile` → `correctness` → `preemption` → `online` | 每个后续阶段要求前一阶段的 supervisor 通过且未撤销资格 |
| 2 | `screen` → `qualify` → `state_cost` → `pressure` → `actions` | 先有 D1 的 `terminal.json`，方向内再逐阶段检查 |
| 3 | `correctness` → `profile` → `update` → `multisession` | 先有 D2 的 `terminal.json`，方向内再逐阶段检查 |

表中是新 run 的标准顺序。run01 的 D2 `state_cost` 是在 `pressure` 完成后、D3 开始前补齐的独立测量，保留了原压力停止结论；没有重跑已完成压力阶段，也没有改变原科学参数或资格门槛。其成本协议与实际执行源码单独冻结，不能将这一补测顺序当作新 run 跳过前置检查的方式。

单阶段命令示例，按当时真实前置状态选择一条执行：

```bash
directions run --direction 1 --phase online
directions run --direction 2 --phase screen
directions run --direction 2 --phase qualify
directions run --direction 2 --phase state_cost
directions run --direction 2 --phase pressure
directions run --direction 2 --phase actions
directions run --direction 3 --phase correctness
directions run --direction 3 --phase profile
directions run --direction 3 --phase update
directions run --direction 3 --phase multisession
```

这些是逐条执行的入口列表，不是无条件串行脚本。`run` 结束后会更新报告；正常的 CLI 退出码本身不等于实验通过，应读取该阶段的 `status`、`eligible`、`reason` 及 supervisor 证据。下一方向所需的 terminal 既可能来自完成，也可能来自明确停止；不能删掉失败报告来绕过方向内门槛。

`--wait-idle-seconds 60` 可在准入前等待设备空闲或所需资产就绪，这段等待不计 GPU 预算。`--timeout-seconds` 是本次运行的时间上限，不扩展方向额度。预算充足但显式 timeout 小于最低运行单元时，准入返回可重试的 `blocked`，不生成本方向 terminal；方向剩余预算无法容纳最低单元和清理时，返回免费的 `stopped` 并生成 terminal，含 `not_measured_budget_limited`，不把未做实验写成无收益。

`run-all` **只用于全新 output**。它从 D1 的第一个阶段开始，不会跳过已有阶段；已有 worker 报告或冻结源码的阶段不能原位重跑。新 output 仍必须复用同一新增账本，不会获得另一份 24 小时。例中的目录需确实是尚未使用的新目录：

```bash
PYTHONPATH=/home/zyd/code/dit-rewrite/python \
  /home/zyd/code/dit-rewrite/.venv-previs/bin/python -m ditserve research-directions run-all \
  --config /home/zyd/code/dit-rewrite/configs/research_directions.json \
  --output /home/zyd/code/dit-rewrite/results/research-directions/new-run-name \
  --ledger /home/zyd/code/dit-rewrite/results/research-directions/gpu-budget.json
```

## 预算、隔离和失败恢复

新增预算由 [gpu-budget.json](/home/zyd/code/dit-rewrite/results/research-directions/gpu-budget.json) 独立累计，分配固定如下：

| 用途 | 上限 |
| --- | --- |
| D1 | 8 GPU 小时 |
| D2 | 6 GPU 小时 |
| D3 | 8 GPU 小时 |
| 修复重试 reserve | 2 GPU 小时 |
| 合计 | 24 GPU 小时 |

历史 [research-pilot/gpu-budget.json](/home/zyd/code/dit-rewrite/results/research-pilot/gpu-budget.json) 保持原样。新增账本记录其路径、摘要和历史累计值；不清零旧支出，也不因更换 output 重置新增支出。普通阶段同时受总额和本方向余额约束，未使用额度不会自动转给其他方向。

计费以 supervisor 准入后 `begin` 到确认子进程组清理完毕后的 `finish` 为边界，使用单调时钟乘 GPU 数。模型加载、持卡期间 CPU 编码与质量评价、空闲、失败、写出和清理都在其中；这不是仅统计 CUDA kernel 的计费方式。纯离线报告/媒体转换另行执行时不计入原实验延迟。报告将已结算支出与 active 会话的临时估算分开，临时估算不回写账本。

选定物理 A100 由 GPU UUID 绑定，在工作进程中映射为 `cuda:0`。工作进程导入模型前校验活动账本、会话 ID、supervisor 父进程、进程组、配置、output 和设备可见性。运行中及最终清理后都检查隔离；发现外部 GPU 任务或冻结源码变化会撤销该阶段资格，即使 worker 曾写入通过。清理针对整个自有进程组；无法确认清理完成时保留未结算 active，后续 GPU 准入继续阻止。没有自动清空 active 或免费重跑的恢复命令。

修复已计费失败阶段时，`run --reserve-for /绝对路径/supervisor-阶段.json` 只接受原报告为 `failed` 或 `incomplete`、有 charge，且 direction、phase、配置摘要均一致的情况。重试使用新 output 和同一账本；后续 phase/方向仍需有效前置证据，CLI 不自动搬运这些证据，也不能把 `--reserve-for` 当作跳过前置检查的开关。科学配置变化不属于同配置修复重试。保留原失败 output、源码快照、日志与 charge，不覆盖它们。

## 三个方向的测量范围

**D1：缓存关闭的 Wan 步内暂停与 Sana 图像请求。** GPU 共驻留范围为 Wan DiT/VAE 与 Sana 组件；Wan 文本编码器沿用既有 Flash 基线的 CPU FP32 路径，不能称整个 Wan pipeline 都在 GPU。实际组件位置写在 `execution.residency`。正确性比较逐步 latent、solver 历史和最终输出，并在 Wan 暂停时实际运行另一模型。

`preemption` 使用预编码图像和有限视频前缀，只用于抢占初筛。`online` 则用独立到达线程，在视频请求真实到达后提交单个 CPU 编码队列；编码完成后才可首次 GPU dispatch，不提前读取未来请求的条件。图像请求走包含其编码的真实生成路径。所有策略使用同一冻结到达 trace 和共享 EDF 规则。在线 wait 从到达到首次 GPU dispatch，包含视频 CPU 预处理等待；latency 到完成；goodput 用完整 replay 的墙钟时间，包含空闲和 drain。报告中的配对描述性指标仅使用完整、同 trace 的策略组；未完成 `active_trace` 不混入最终配对指标。有限样本不支持稳定 p99、SP 或调度方法新颖性的结论。

**D2：同 A100、SP=1 的缓存状态机制。** 固定 480×832、49/81 帧、32 步、CFG=5，比较无缓存参考与 `paired_008`。筛选不能继承旧 CFG 或迁移实验资格；新规格需本轮 8 个开发输入的正反序比较及 4 个留出输入。run05 的 17/33 帧热态搬运与保留对照的负面观察只作为历史证据保留；CFG run03 的资格限于 256×448×17、32 步、CFG=5、各卡 SP=1，均不能包装为本轮长规格的通过或失败。

速度口径为固定 CPU 预编码输入下的状态准备、去噪和解码。每个 action trial 在背景 prefix 与 arrival 之前显式 warm urgent prompt，冷、暖 bank 的差别都留在已计费 setup；它不构成完整在线端到端编码加速。压力使用 1/2/4/8/16 个真实暂停请求，不用填充张量制造 OOM。

`state_cost` 独立测量已通过资格的 81 帧规格，不以 urgent OOM 为前提；当前补测协议针对 run01，若新 run 仅 49 帧通过资格，此阶段会 `blocked`，不能直接外推。它固定一个既有开发输入与第 16 步边界，对 keep、完整 CPU 换出并恢复、清空 feature cache 后强制重算执行正反两轮；记录同步搬运时间、实际 allocated 变化、下一步及完整后缀成本，并检查最终输出。新增的 [成本微测量协议](/home/zyd/code/dit-rewrite/configs/research_direction2_state_cost.json) 随该阶段源码冻结，原科学配置、历史资格与预算不变。每臂 n=2 只能描述成本，不能授予容量收益或控制策略资格。run01 六项已完成，完整搬运输出与同轮 keep 精确一致，两轮 keep 也一致，flush 通过既有最终输出质量门槛；这些检查不替代新策略完整开发/留出资格。即使 keep 在压力范围内始终成功，独立成本仍需测量；只有容量边界后的四动作比较可以跳过。

四个动作是 `keep`、`full_cpu`、`flush`、`wait_refresh`。`full_cpu` 执行真实 tensor CPU 搬出与恢复，包含必要状态及缓存，不用 pickle 序列化估算代替搬运；恢复后完成背景后缀。`flush` 清缓存并强制重算；`wait_refresh` 等待真实轨迹上的刷新资格。到 urgent 完成及全部请求完成的计时包含相应动作、等待、搬运、恢复与后缀成本。OOM 按 setup/prefix/action/urgent/resume/finish 留存，只有两次 keep 都在 urgent 阶段 OOM，且候选两次完成并通过相应质量检查，才列为观察到的准入动作。动作矩阵完成不自动授予完整的新动作质量资格。

当前可验证的是同卡显存换出/恢复。A100/V100 的旧 SP=1 异构迁移诊断不等于 SP resize；缺少同型多卡硬件的重分片问题仍为 blocked，本轮结果最多属于方向 2 的部分验证。

**D3：LongLive 流式条件更新和播放。** 固定 16 FPS、480×832、每块 3 latent 帧。因因果缓存逐渐填满，chunk 0/1/2/3 的上下文仍处于增长或首次满窗阶段；从 0 编号的 chunk 4 才首次发生滚动写入。配置仍保留 `profile_blocks=10`、`warmup_blocks=2`，实现取 `effective_warmup=max(configured_warmup,ceil(local_attn_size/latent_frames_per_chunk))=4`，使用索引 4–9 的六个稳态块，并要求至少三个稳态样本。报告分别记录配置预热、实际预热、测量索引与源码审计依据。单会话仍需达到 16 FPS，两会话另需单卡服务曲线至少 32 FPS；低于门槛就停止，不降低播放 FPS。

播放缓冲 B=0/1/2 表示**当前播放或启动块之外，最多 B 个预取块**。按真实 chunk 身份计槽，生成前预留整个新块；当前帧在其显示周期内仍占槽。首块 9 像素帧、后续块 12 像素帧不会被统一的帧数阈值混淆。`peak_queued_frames` 保留实际排队帧峰值，同时报告 `peak_resident_chunks`、`peak_ahead_chunks` 和缓冲语义。

latest-only 两臂在下一合法 GPU 边界独立于生成缓冲门槛处理更新；condition 返回后重新收集已经到达的新条件，合并后必要时再 recache，才生成下一个块。FIFO 保持一项更新后生成一个块。已编码/recache 却未产生块的版本仍保留成本与被覆盖记录；superseded、pending、窗口截断和未完成事件不从分母消失。

播放器为独立线程的 **16 FPS headless CPU 帧消费端**，时间戳是真实单调时钟采样，不能代表浏览器、网络、显示合成或显示器延迟。记录条件应用、首次条件匹配解码、首次条件匹配播放三种技术延迟，以及 stall、丢帧和版本来源。条件版本匹配不证明画面语义已经改变；部分块的废弃成本是按帧比例分摊联合块耗时的估算，完整块耗时也只是同步去噪/上下文墙钟时间，不是独立 kernel 计时。

## 证据位置与阅读顺序

默认根目录为 `/home/zyd/code/dit-rewrite/results/research-directions/run01`，以下相对项均位于该根目录：

| 产物 | 用途 |
| --- | --- |
| `protocol.json` | 冻结的科学配置与摘要 |
| `sources/directionN-phase/manifest.json` | 本次实际执行源码、配置、历史输入和资产 manifest 的摘要；同目录是执行快照 |
| `directionN/preflight-phase.json`、`directionN/environment-phase.json` | 准入时设备/模型观察和工作进程环境 |
| `directionN/phase.log`、`directionN/phase-isolation.jsonl` | 原始日志及逐次隔离观察 |
| `directionN/worker-phase.json` | worker 的本地结果，尚不能单独授权后续阶段 |
| `directionN/supervisor-phase.json` | 包含最终隔离、退出码、charge、源码/配置身份的阶段终审 |
| `directionN/admission-phase.json`、`directionN/terminal.json` | 准入阻止/预算停止，以及方向推进所需终止记录 |
| `direction1/direction1-phase.json` | 逐样本、逐 replay 和活动 trace |
| `direction2/screen.json` 等、`direction2/partial_units/` | 配对、资格、压力、动作及中断单元 |
| `direction2/state_cost.json`、`direction2/supervisor-state_cost.json` | 独立成本补测、逐臂输出检查与监督结算；不覆盖压力阶段的停止结论 |
| `direction3/phase/report.json`、`direction3/phase/partial.json` | 逐块、条件更新、播放、废弃工作与部分证据 |
| `report.json`、`REPORT.md` | 受监督状态与描述性指标的汇总入口 |
| `validation/` | CPU 测试日志、实现审计、验证摘要；不替代 GPU 实验结果 |

每个 phase 从自己的冻结源码目录运行。修复工作区源码不会改变已经启动的快照；后续 phase 使用新快照，并在结果中保留各自身份。大模型权重不复制进源码快照，资产清单及摘要随阶段归档。大数组按实际报告引用定位：D2 的配置资产根位于 `/mnt/data/zyd-preempt-phase16-01a1296d/dit-directions-assets/artifacts`，不能假定每个 output 内都有全部原数组。

先看 supervisor/admission 状态，再看资格与具体 gate，最后查看原始单元和数组。`passed` 可能只表示有限动作矩阵完成；`eligible=false` 仍会停止该方向。汇总区分“值得继续”“简单策略已足够”“当前条件无收益”和“资源不足或未验证”。预算中断、工程失败、隔离失败和缺测不能变成科学上的无收益；缺失指标不填零。active 预算只是未结算估计。

`report` 在账本无活动会话且可选导出依赖可用时还会尝试轻量 Excel 导出；导出被推迟不会改变原始报告或实验结论。不要为了导出表格而删除尚在使用的原始证据。

## 普通预览与语义盲评

run01 已导出 24 条普通预览，每方向 8 条，代表比较如下。预览统一以 16 FPS 播放，不能用成片播放速度推断生成帧率，也不是语义盲评结果。

| 方向 / 样本 | 候选视频 | 对照视频 |
| --- | --- | --- |
| D1，81 帧恢复正确性 | [拆分恢复输出](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction1/correctness-d1-correctness-1-0-output-a96f8d2e88ba.mp4) | [连续执行参考](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction1/correctness-d1-correctness-1-0-reference_output-68b43ea1eeb8.mp4) |
| D2，81 帧缓存资格 | [paired_008](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction2/qualify-paired_008-output-b05010088a3d.mp4) | [无缓存参考](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction2/qualify-off-output-a0b489271c7f.mp4) |
| D2，完整状态换出恢复 | [full_cpu](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction2/state_cost-full_cpu-output-d94ab0ab1fa2.mp4) | [同轮 keep](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction2/state_cost-keep-output-89244a202cf1.mp4) |
| D3，属性更新正确性 | [逐块执行](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction3/correctness-p0-s101-output-e66adca8a316.mp4) | [官方参考](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/direction3/correctness-p0-s101-reference-output-6746dfab479d.mp4) |

全部预览及原件保留范围见 [媒体 manifest](/home/zyd/code/dit-rewrite/results/research-directions/run01/media/manifest.json)。D3 正确性只说明执行与参考一致；例如该属性样本预览的初始球呈黄色，不能据输出一致宣称模型准确遵循了初始“红球”提示。

建议在 GPU phase 结束后进行 CPU 媒体导出，避免压缩争用持卡期间的 CPU 资源：

```bash
directions export-media --limit 8
directions report
```

run01 可导出普通预览；因未运行 D3 `update`，没有可供该阶段语义盲评导出的更新轨迹。不要将正确性样本或 profile 片段当作已完成交互盲评。

`export-media` 每方向最多 8 条视频，生成 `media/manifest.json`、16 FPS MP4 和六帧联系图。选择完整比较组，优先失败样本，并轮转阶段和报告规格；同级较大规格优先，配对不因剩余额度被拆开，chunk 片段最后选择。manifest 写明来源与媒体 SHA256、帧数、未选数量、缺失原件和有界预览范围。只有报告实际引用且仍存在的数组才可导出；部分成功留出/动作只留指标与 hash，不能宣称全部视频齐全。原数组保持不变，普通预览不是盲评或质量通过。

其他已生成更新轨迹的 run 可使用 `directions export-blind`。该命令要求已有 D3 `direction3/update/report.json`，按记录中的实际播放时间线导出匿名事件窗口，包括 startup/stall 的黑屏或持帧、窗口截断和素材失败。它保存完整事件分母，不把被覆盖或未播放的更新消失掉。产物位于：

- `direction3/update/blind-windows/public/`：给评审者的匿名视频、联系图、manifest 和空白 `ratings.csv`。
- `direction3/update/blind-windows/private-key.json`：私有映射、策略身份、源报告与素材摘要，不能一并交给盲评者。

已存在的盲评目录不会被覆盖。导出成功或空评分表不表示完成盲评；当前技术指标、普通预览及匿名窗口本身均不自动授予语义质量/新鲜度资格。实际评审结果需单独形成可审计证据。

相关实现：[protocol.py](/home/zyd/code/dit-rewrite/python/ditserve/directions/protocol.py)、[budget.py](/home/zyd/code/dit-rewrite/python/ditserve/directions/budget.py)、[reporting.py](/home/zyd/code/dit-rewrite/python/ditserve/directions/reporting.py)。本轮实现验证见 [validation](/home/zyd/code/dit-rewrite/results/research-directions/run01/validation)；历史协议另见 [research_cfg.md](/home/zyd/code/dit-rewrite/docs/research_cfg.md) 与 [research_migration_diagnostic.md](/home/zyd/code/dit-rewrite/docs/research_migration_diagnostic.md)，其资格范围不自动转移。
