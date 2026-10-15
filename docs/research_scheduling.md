# D1 调度与抢占：独立 12 GPU 小时协议

本轮比较在相同无缓存执行器上，派发规则与暂停粒度是否改善混合图像／视频服务。入口为 `research-scheduling`，配置为 `configs/research_scheduling.json`，默认输出为 `results/research-scheduling/run01`。本文件描述实现与运行规则，**不表示任何新 GPU 阶段已经通过**；实际结论以当前运行的 `supervisor.json`、完整配对记录和质量检查为准。

历史 D1 的 57/60 次 replay、其他方向及旧缓存资格保持原样。历史部分结果不自动取得本轮质量、设备、成本或调度资格；CPU 测试通过也不等于 GPU 验证。

## 预算与设备

新增累计上限为 **43,200 GPU·秒（12 GPU 小时）**，独立账本为 `results/research-scheduling/gpu-budget.json`。旧 `results/research-directions/gpu-budget.json` 只记录为历史来源，不重置、不继承其余额。单 GPU 会话从准入至确认自有进程组清理结束计费，包括加载、编码准备、评价、失败、持卡空闲及清理。更换 output 不获得新额度；续跑继续引用同一账本。

用户已将**本轮固定为 V100**；配置的 `device_preference` 仅包含 V100，即使 A100 空闲也不切换过去。执行路径为 Wan FP16／EFFICIENT，**Sana 全 FP32／MATH**，没有 Sana FP16 自动降精度路径。这是明确的 V100 执行变体，不继承旧 A100 结果；必须重新完成当前设备的共驻留完整生成、两条 prompt／seed 301 的暂停恢复正确性和成本校准。V100 资格失败即停止，不自动切换设备或配方。校准后锁定 GPU UUID、解释器、后端和 Sana 路径。

模型只使用配置中的本地固定资产，不下载替代权重。只有受监督的独占 GPU 窗口可授予性能资格。**本轮暂缓 `legacy` 旧 A100 矩阵收尾**；不能把 V100 结果拼接到旧 A100 配对，旧缺测保持未验证。

## 阶段与首批六次 replay

阶段依次为 `calibrate → pilot → screen → validate`。后续阶段必须有前一阶段当前配置、当前设备的受监督通过证据。

| 阶段 | 范围 | 资格边界 |
| --- | --- | --- |
| `calibrate` | 当前设备的执行正确性、共驻留能力、独立服务与阶段成本 | 不授予在线调度收益 |
| `pilot` | seed 301，两场景各三臂，共六次完整 replay | 只用于初筛与成本、触发机会描述，不是完整开发集或留出结论 |
| `screen` | seed 301，完整四场景、预先声明的候选及对照；压力场景附无视频保护消融 | 完整开发矩阵通过后才冻结候选 |
| `validate` | seed 401、402，四场景，冻结候选与两个强对照 | 留出失败即停止，不利用留出重选候选 |

首批 `pilot` 固定为：

1. `original_burst`：`edf_b1 → edf_b4 → guarded_b4`。
2. `tight_deadline`：`guarded_b4 → edf_b4 → edf_b1`。

每次均为 24 个真实请求，所有比较臂共享冻结的到达、prompt、seed、规格和 deadline。完整配对组按实测成本乘安全系数进行准入，并预留清理开销；不足时保留缺测，不能删除较慢或失败的单元来缩小矩阵。

## 工作负载与策略

所有视频使用配置中的固定规格、32 步、CFG=5、缓存关闭。图像 deadline 为独立完整服务时间的 2.5 倍。视频与图像分别按各自请求序号交替使用 prompt。

| 场景 | 到达模板 | 视频请求 | 视频 deadline |
| --- | --- | --- | --- |
| `original_burst` | 旧生成器 burst，基准 load 0.9 | `r000`、`r012` | 2.5 × 独立服务时间 |
| `tight_deadline` | ordinary，基准 load 0.9 | `r000`、`r012` | 1.25 × 独立服务时间 |
| `original_ordinary` | ordinary，基准 load 0.5 | `r000`、`r012` | 2.5 × 独立服务时间 |
| `video_contention` | 完全复用同 seed 的 `original_burst` 到达 | 增加 `r006` 为视频 | 1.5 × 独立服务时间 |

`video_contention` 改变请求类型后不重新放慢到达；0.9 仅是基础 trace 的标签，不表示转换后的实际负载仍为 0.9。报告应保留实际到达间隔、突发与视频重叠情况，不能只凭场景名认定存在更强竞争。

`edf_b1`、`edf_b4` 使用同一纯 EDF，分别在一个或四个 block 后重新派发。`guarded_b4` 使用预计可完成 EDF、等待保护和视频恢复保护，固定四 block。`guarded_adaptive` 使用相同派发规则，在一、四、八 block 间选择；`feasible_b4` 是去掉视频恢复保护的消融。`edf_step` 为整步 EDF 对照，不允许在半步 continuation 上调用原生整步执行。

候选仅查看已经到达且 GPU 就绪的请求。成本预测来自冻结校准，不读取未来完成时间。剩余成本包含准备、剩余去噪和解码；本轮 continuation 常驻 GPU，不执行 CPU／GPU 换出恢复，状态切换和检查开销按完整动作墙钟计入，不虚构独立的 restore 项。非负 slack 只表示“预计可完成”，不是硬期限保证。

连续 GPU 就绪等待达到独立服务时间的 2.5 倍时触发等待保护；多个请求按 `ready_since_ms + 2.5 × service_ms` 的触发时间排序，执行一个动作后重置该请求的等待起点。没有等待保护时，先在预计可完成请求中按 EDF 选择；若均预计超期，仍按 EDF 完成任务。视频恢复保护只可插入一个量子，并要求原候选及其他预计可完成请求不因该量子而超期。任何选择发生约束冲突时，包括等待保护，也回退到 EDF 选择请求并执行最短合法动作，同时计数；这不是期限或持续在线公平性保证。有限 trace 中所有请求仍须 drain，不丢弃超期请求。

自适应选择满足时间上界的最大量子，上界同时考虑图像独立 deadline 间隔的一半与其他预计可完成请求的最小 slack。准备、Sana 完整生成和视频解码仍是不可抢占段；将动作标为量子一不会缩短这些阶段。

## 执行入口

以下函数仅定义命令；单独定义它不启动 GPU。所有命令继续使用同一配置、output 与 ledger：

```bash
scheduling() {
  PYTHONPATH=/home/zyd/code/dit-rewrite/python \
    /home/zyd/code/dit-rewrite/.venv-previs/bin/python \
    -m ditserve.scheduling.runner "$@" \
    --config /home/zyd/code/dit-rewrite/configs/research_scheduling.json \
    --output /home/zyd/code/dit-rewrite/results/research-scheduling/run01 \
    --ledger /home/zyd/code/dit-rewrite/results/research-scheduling/gpu-budget.json
}
```

只读预检、冻结协议和查看报告：

```bash
scheduling preflight
scheduling prepare
scheduling report
```

`preflight` 查询设备和本地资产，不加载模型；`prepare` 创建或核对协议和独立账本，不运行 GPU。`report` 整理已有证据和轻量结果，不产生新的实验资格。

在各自前置证据通过后，按阶段执行：

```bash
scheduling run --phase calibrate
scheduling run --phase pilot
scheduling report
scheduling run --phase screen
scheduling run --phase validate
scheduling report
```

这些是分阶段入口，不是无条件执行列表。`--wait-idle-seconds 60` 只在准入前等待可用设备，不增加预算；`--timeout-seconds` 限制本次执行时间，也不扩展额度。已尝试阶段的源码、日志与结果不可原位覆盖；失败或预算停止须保留证据，不能通过更换目录清零账本。若账本仍有未结算会话，必须先核实自有进程清理状态。

校准已经由监督进程启动时，可用独立 CPU 控制器自动接续，避免重复启动校准：

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve.scheduling.study --after-calibration
```

控制器复用同一配置、目录和账本，等待已有校准的监督结论；每阶段结束先生成报告，再检查是否允许下一阶段。`controller.json` 保存进展，重复启动不会重跑已有尝试。只有完整筛选得到否定结论或完整留出验证结束后，才重新导出、回读验证并按清单清理原始 `.npy`；实际删除以 `excel/pruned.json` 和 `cleanup-summary.json` 为准。失败、中断及预算不足保留原始证据。

## 评价与留存

每条完整 replay 包含真实到达、编码、排队、GPU 执行、解码及 drain。goodput 分母为完整墙钟时长；失败、超期和未完成请求不从分母消失。阶段记录区分 CPU 编码提交／开始／结束、GPU 就绪等待、首次派发、恢复等待、准备、量子与解码，并统计保护触发、冲突和预测误差。日志写出及调度器开销计入实际时间，不从性能结果扣除。

动作明细按 replay 缓冲追加至临时 `execution-events.jsonl`，周期性阶段快照只保存请求统计，不重复写入累积的动作历史。完成记录包含事件文件路径、数量和 SHA256，并核对事件数与请求动作计数。输出数组仍同步写出，保持原计时边界；报告单列输出写出与 checkpoint 开销，不将其事后扣除。事件写入或关闭失败时仍须清理编码线程和模型状态，失败运行不能获得资格。

开发候选须相对 `edf_b1`、`edf_b4` 两个对照均满足：压力场景合计 goodput 至少改善 5%，原场景各自 goodput 不下降超过 2%，每场景视频按时完成数不低于对照。完整开发后按总体 goodput 冻结候选，两候选相差小于 1% 时选择 `guarded_b4`；留出还要求每个 seed 的压力场景收益为正。上述均为工程继续门槛，有限 seed 的 p95、CPU 回放及首批六次 replay 不支持稳定 p99、统计显著性或一般最优性结论。

每阶段在 `runroot/<phase>/result.json` 保留工作结果，`worker.json` 与 `supervisor.json` 记录执行及终审，`sources/` 保留冻结源码与清单。质量、隔离、完整矩阵或预算任一条件不满足时，报告区分失败与证据不足，不填零、不外推。长期结果使用轻量 Excel 和可核验摘要；活动实验、待评价视频及续跑所需证据不得提前清理。没有观测到策略收益或保护触发机会，也是合法的停止结论。
