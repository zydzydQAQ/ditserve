# DiT 混合负载：独立 8 GPU 小时优化初筛

本协议用于已有 Wan/Sana 混合 serving 路径的有限配对初筛。文档描述可执行协议与验收规则，不预填性能结果。实测结论以各阶段 `terminal.json`、对应 attempt 的原始证据及独立预算账本为准。局部加速不能替代在线资格；小样本的描述性 p95 不支持稳定 p99 或生产容量结论。

## 固定输入与执行顺序

配置入口为 `configs/serving_optimizations.json`，冻结配置与实际执行源码位于 `results/optimization-screen/run01/protocol.json` 和 `sources/<phase>-<attempt>/`。配置校验拒绝静默改变科学参数；修复尝试单独冻结源码、保留旧尝试与费用。

依次执行 `sana_compile`、`sana_batch`、`wan_compile`、`text_encoder`、`vae`、`cache_k4`。默认使用已有 A100 40GB；V100 编码回退只在前置 A100 编码容量失败及监督器准入后实施，双卡持有期间按两张 GPU 计费。不得在监督器外启动模型或绕过设备占用检查。

在线负载直接读取历史 `direction1-online.json` 中两条完整 K4 轨迹：seed=102、load=0.5、burst；seed=101、load=0.5、ordinary。每条有 24 个请求，保留原始到达时间、deadline、服务估计与生成 seed；不按更快的候选重新标定负载。视频规格为 480×832、81 帧、32 步，默认 cache-off、CFG=5；图像沿用冻结 Sana 配方。默认视频调度边界为 K4。

## CLI 与证据

在仓库根目录使用已准备好的独立环境，GPU 实验仅从以下监督器入口启动：

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve optimize-serving inspect
PYTHONPATH=python .venv-previs/bin/python -m ditserve optimize-serving run --wait-idle-seconds 7200
PYTHONPATH=python .venv-previs/bin/python -m ditserve optimize-serving report
```

`run --phase sana_compile` 可选择单个阶段；后续阶段仍要求之前阶段已有终局记录。`--retry-from /absolute/path/to/failed/supervisor.json` 只用于同配置、同阶段、已结算失败的修复尝试；优先计入该阶段尚未使用的额度，该额度不足以启动工作进程时才使用独立备用额度。失败用量和旧尝试不重置。普通 `report` 不加载模型；报告模块保留来源文件、SHA256、JSON 指针及逐样本数。

Wan 的仅在线复测使用独立入口，不能与普通重试参数合用：

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve optimize-serving run --phase wan_compile --online-replay-from /absolute/path/to/audited/supervisor.json
```

该入口只接受同配置、已经结算且有明确在线测量审计的来源。准入重新核对 8 个恢复样本和 16 个性能样本的完整性、质量结果与证据哈希，并要求数学实现源码一致；不能用来跳过未完成或失败的局部资格检查。新 attempt 引用原局部证据，保留原账单和停止审计；新进程仍重新执行容量检查、预热及两条轨迹的完整 AB/BA 八次在线运行。原在线结果不授予新资格，必须由新测量和监督器重新判定。

复测报告中的 24 行局部证据逐项标记复用、未重测，新增局部执行为 0；与历史记录是同一组样本，不能合并增加样本量。复制的 `compilation` 统计属于原始尝试，不能解释为新进程的冷编译耗时。新进程加载、容量检查、编译预热和在线执行都计入新的持卡账单，但没有独立结构化计时的组件明确标为未单独测量。日志可导出的“累计保存结束至下一轨迹准备开始”只是混合 host 准备间隔，包含可能的容量检查、后端准备、预热、主机与日志工作，不含首次累计保存前的模型加载，也不是精确 compile-only 或 GPU 耗时；它已包含在持卡成本中，不重复加账。

各 attempt 保留配置、环境、工作日志、GPU 占用观测、局部质量与性能证据、必要的在线原始事件。监督器失败、非零退出、预算中断和清理不确定不会被局部成功字段覆盖。只有监督器有效结算、局部和在线资格都通过的后端可成为后续默认值。

## 验收与统计口径

局部配对保留 AB/BA 顺序及逐输入质量，不用总体均值掩盖单样本质量失败。Sana 图像参考差异门槛为 LPIPS≤0.01；Wan 视频为 LPIPS≤0.05。同后端连续执行/恢复检查使用冻结 `rtol=1e-5`、`atol=1e-6`。LPIPS 是相对参考差异，不能解释为绝对画质或语义正确性。缓存三输入筛查不能代替完整开发集、留出集质量资格。

缓存正确性还须记录真实 32 步提交路径，并核对原子/K4、连续/K4 的路径序列。三个固定敏感输入的恢复轨迹合计必须实际覆盖 full 和 reuse；两边均无复用不能证明 reuse 已验证。固定 Sana 插入仍在第 0/8/16 步的前缀内，提交后记录该步最终走 full 或 reuse，不能称为在复用后缀内插入。Sana 插入与两条路径的交叉覆盖只作描述，未命中的组合注明未单独验证，不额外扩大用户规定的验收门槛。

缓存阶段的完整资格按固定顺序补测：8 个开发输入的 AB、BA，随后 4 个留出输入，共 20 个 off/paired 配对。后续实现按每个完整配对的预算准入，不要求开始前一次容纳全部 40 次生成；下一配对预算不足时不启动任一实验臂，保留已通过的三输入筛查及已完成配对，报告已完成数、固定分母 20、剩余输入/顺序与所需/剩余秒。部分资格为 `partial` 或 `not_measured_budget_limited`，不能进入缓存在线实验；全部 20 对质量及双顺序性能门槛通过才授完整缓存资格。配对执行中的实际中断仍保留失败/未完成证据并走原硬停止路径。该预算准入修正仅用于后续冻结的缓存阶段，不改当前或历史冻结实现、输入、顺序、质量阈值及计算算术；共享模块源码哈希会变化，不能据此再次复用要求源码一致的旧 Wan 在线重测入口。

系统验收需要两条固定轨迹均完成 AB 和 BA，总计八次完整运行，且请求身份完全一致。总体 goodput 至少提升 5%，或图像 E2E p95 至少下降 10%；每条轨迹同时要求 goodput 损失不超过 5%、图像 E2E p95 增幅不超过 5%、视频按时完成比例下降不超过 0.05。失败或未完成请求留在到达分母中；完成请求的 p95 单独标注样本数。不足以完成矩阵时仅记录未验证或完整的归因基线，不授予系统资格。

图像 batch=2 只合并所有已到达、未完成请求的 EDF 队列最前面的两张兼容图像，且两张都须就绪；不能越过尚在编码的视频，也不等未来请求。两张图像各自保留 deadline，完成时间均取整个批次输出完成时刻。视频 CPU 编码由独立真实到达线程提交；A100 GPU 编码由统一 EDF 调度器串行执行。编码后、prepare 后和视频量子后重新选择就绪请求。

事件记录编码入队/开始/就绪、prepare、视频量子、视频 finish 内 decode/convert/sync、图像批服务。host 调用区间可能嵌套或包含异步提交，不可将区间求和当作 GPU 利用率。只有观测到编码等待造成的空闲占轨迹时间至少 5%，或编码服务占视频 E2E 合计至少 20%，才进入编码优化。最慢 5% 图像等待时间中，视频 finish 区间覆盖至少 20% 时才进入 VAE 优化。冷编译、预热和质量检查均计入持卡费用，但应独立于稳态性能样本记录。

采用本修复后新源码的在线尝试（包括阶段三仅在线复测及后续阶段）额外包含 `first_a100_stream_marker` 和视频的 `first_dit_stream_marker`，保留原有 host dispatch 字段。计时事件池在到达时钟开始前预创建并 record 初始化，随后记录并同步一次 CUDA 原点；此准备计入持卡费用。到达后只在首次 A100 工作及首次视频量子提交前记录当前 stream 的事件，轨迹内不新增同步。正常结束利用 backend 已有同步后非阻塞 query 读取设备 elapsed；异常未完成事件保留 pending。图像批共享服务标记，不虚构 Sana 内部 DiT 时间。

这些数值表示 GPU stream 实际到达首次提交前标记的时刻，不能称为首个模型 kernel 的精确开始时间。GPU 原点的 host record/synchronize 时间区间单独保留；设备 elapsed 和 host arrival 属于不同时间原点，禁止直接相减计算请求延迟。`kernel_timestamp_status` 仍为 unverified。已经冻结的旧阶段一、阶段三 attempt02 及更早尝试不包含这项新证据，不回填或推断。

同一观测修复将高频保存改为仅写当前轨迹的 `partial.json`，不重复序列化此前已经完成的轨迹；`online.json`（缓存四臂实验为 `cache-online.json`）仅在轨迹外的初始、整轨迹完成和终止边界更新累计结果。`active_trace_path` 供进度报告读取独立 partial，以轨迹身份、顺序和实验臂去重；未合并的 partial 即使标 completed 也不授系统资格。每条轨迹前先写 preparing 身份记录，避免显示上一条轨迹的残留进度。

轨迹内保存频率与调度策略保持不变。`partial_snapshot_build` 和 `partial_flush` 分别记录构造快照、序列化及原子写入回调的 host 起止；本次写入完成后才把事件加入内存，由下一份快照或最终累计行携带，不递归重写本次日志。最终/异常快照自身的保存耗时，以及轨迹外累计文件保存耗时，另存小型 `online-flushes.json` / `cache-online-flushes.json`；该计时文件自身写入不递归计时。终局保存发生在 trace elapsed 截止之后，但仍计入 GPU 持卡账本。旧 attempt02 的未覆盖主机间隙与日志规模相关，只能作为限制；不能事后把间隙全扣成日志开销，也不能据此修正 GPU 性能或补授资格。

## 独立预算

本轮账本为 `results/optimization-screen/gpu-budget.json`，总上限 28,800 GPU 秒（8 GPU 小时）。历史 24 GPU 小时账本只读绑定，不重置、不抵消历史费用。

| 阶段 | 初始额度 |
| --- | ---: |
| Sana 编译 | 1 GPU 小时 |
| Sana 两图批处理 | 1.5 GPU 小时 |
| Wan 编译 | 2 GPU 小时 |
| 文本编码 | 0.5 GPU 小时 |
| VAE | 0.5 GPU 小时 |
| cache + K4 | 1.5 GPU 小时 |
| 修复备用 | 1 GPU 小时 |

阶段未使用额度仅可向后结转；普通阶段不得提前借用未来或修复备用额度。模型加载、预热、编译、质量评估持卡、失败尝试、等待正在执行的 CPU 编码以及进程清理均在会话计费范围内。清理只针对本监督器建立的子进程组；清理无法确认时保持活动账本，不能假定费用已结算。

## 报告与轻量归档

`ditserve.optimizations.reporting.write_report(output, ledger)` 生成 `report.json` 与 `REPORT.md`，逐阶段区分监督器资格、局部描述性证据、质量失败和在线样本。报告列出局部 service 的 mean/p95/n、在线 goodput、图像 E2E p95 与其完成样本数、视频按时数与到达分母。原始事件只汇总计数和区间分布，不把逐事件原文塞进 Excel。

重试不会隐藏旧失败尝试。`previous_attempts` 保留旧监督器状态、已结算 GPU 秒、来源 hash 和 `baseline-audit-stop.json` 等停止审计；`previous_samples` 单列旧局部行、未完成单元、在线请求与事件摘要。旧样本不混入最新尝试的均值或资格，旧费用已包含在权威总账中，不再次相加。重试仍在执行时，报告按活动账本选择当前 attempt，并将上一份终局记录保留在历史区。

`ditserve.optimizations.reporting.archive(output, ledger)` 只在全部阶段终止、账本无活动会话并取得锁后开放。它构建 `archive-metadata/` 小型中间目录，复用 `compact_results.export/verify` 生成 `excel/results.xlsx`，保存来源 hash、逐样本指标和保留清单。最多为每阶段四个原始输出生成 PNG 关键帧接触表，失败样本优先。视频阶段另选一组实际引用的同输入、同顺序参考/候选，最多导出两个完整 81 帧、16 fps 的 MP4，并回读确认帧数与帧率；这两项优先占用接触表配额。没有完整对应组时明确记录 missing，不臆造参考。接触表与导出视频均不构成语义盲评，导出帧率也不等于实测 serving 吞吐。编译缓存不遍历、不嵌入、不清理。

此入口不删除任何原始文件。跨输出目录的原数组、仍待质量资格/修复/续跑依赖的数组、hash 不匹配的证据均明确标注不可删。`raw-retention-plan.json` 仅供审阅；轻量中间目录的 Excel 不能作为删除原数组或事件的授权依据。若后续要清理，必须另建绑定原 source root、原 hash、预览与完整汇总的显式清单，完成归档验证后再使用独立清理流程。中断或重复归档不覆盖已有成品。

独立收尾接口 `ditserve.optimizations.retention.finalize` 仅接受本轮普通 `.npy` 的明确白名单。每项须提供原始证据 JSON 的 SHA256、绑定数组路径和 SHA256 的 JSON 指针，以及评测、归因和续跑依赖已结束的具体理由；阶段停止本身不代表依赖关闭。接口还要求主 Excel 已通过回读、关键 PNG/完整配对视频已校验、六阶段均有终局记录、设备清理完成且预算账本无活动会话。它针对真实数组叶目录重新导出和回读，随后仅删除清单所列文件，保存独立清单、Excel、定位文件和删除凭据。未完成资格检查的候选及清单外数组、事件、编译缓存、来源快照和历史结果继续保留。该接口须在独立单线程进程中运行。
