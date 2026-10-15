# V100 调度后续实验：补齐竞争场景与分离粒度、输出写盘影响

本轮根据用户 2026-10-15 的新授权继续使用同一张 V100，累计 GPU 预算不设上限，仍记录实际用时并限制每阶段最长运行时间。旧 `research-scheduling/run01` 的 12 小时协议、8.9943 小时已结算账本和未完成结论保持原样。新入口为 `ditserve.scheduling.next_runner`，配置为 `configs/research_scheduling_next.json`，输出与账本分别位于 `results/research-scheduling-next/run01` 和 `results/research-scheduling-next/gpu-budget.json`。

本文件中的后续策略均是**待测候选**。异步输出保存尚无 GPU 收益结论；固定两 block 也不预设优于一个 block。CPU 测试只验证接口、排队和失败处理，不能替代当前设备的完整生成质量和配对性能证据。

## 已有证据与优先级

下表只统计旧 `screen/result.json` 中 14 条唯一 replay；该文件已经包含首批六条 pilot，不能再次把 pilot 加入总数。明细来源是该文件引用的 `execution-events.jsonl`，14 个事件文件均核对了记录中的 SHA256。总 replay 墙钟为 27,631.910 秒，包括到达空闲和最终 drain，不包括模型加载等会话外围时间。

| 计时项 | 合计秒 | 占完整 replay 墙钟 |
| --- | ---: | ---: |
| 调度选择器 | 81.027 | 0.293% |
| checkpoint | 246.250 | 0.891% |
| 同步输出保存 | 80.743 | 0.292% |
| 视频去噪动作 | 15,407.567 | 55.760% |
| 视频解码动作 | 780.634 | 2.825% |
| 图像完整生成动作 | 342.334 | 1.239% |

选择器计时包含没有就绪任务时的循环检查。普通到达场景每次 replay 在此花费约 12.2–15.2 秒，但主要受固定到达跨度约束，同场景各臂总时长差异很小。因此，单独优化 CPU 选择器不是目前最可能产生可观 serving 收益的方向。表中未列出的时间也不能全部算成 CPU 开销：它包含尚无到达、视频 CPU 编码未完成等正常等待。每请求的等待可能相互重叠，不能直接相加当作互斥的整机时间分解。

一个 block 的去噪动作约 0.287 秒，四 block 约 1.146 秒；旧 `edf_step` 的整步约 17.213 秒。一个 block 相对四 block 的总去噪时间差仅约 0.06%–0.61%，目前不足以证明更粗粒度带来有意义的吞吐优势。两 block 的约 0.574 秒仅是根据已有量子作出的估计，必须实测并核对完整轨迹。

旧 14 条 replay 的 28 个视频全部按时完成，视频恢复保护没有出现收益机会。`original_burst` 中 `edf_b1`、`edf_b4`、`guarded_b4`、`guarded_adaptive` 分别为 18、16、16、15 个按时请求；`tight_deadline` 的 `edf_b1` 和 `edf_b4` 均为 24/24，旧保护候选为 23/24。旧完整开发矩阵因预算准入停止，`video_contention` 未测，留出未测。应保留“已测部分保护候选未显示优势”和“完整矩阵尚未完成”两个不同结论，不能将旧结果改写成正式通过或正式完整失败。旧报告见 `results/research-scheduling/run01/REPORT.md`。

## 解码后的空档值得独立验证

视频解码本身平均为 27.880 秒、最大 28.248 秒，是明显的不可抢占段。但两条具体超期记录说明，不能把所有视频附近的图像等待都归因于解码：

| 旧 replay / 请求 | 前一视频 decode 结束 ms | 图像到达且 GPU 就绪 ms | 图像开始 ms | 到达后无 GPU action 的等待 ms | 超期 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `original_ordinary/edf_b4/r016` | 1,913,548.142 | 1,913,842.658 | 1,916,378.800 | 2,536.142 | 883.979 |
| `tight_deadline/guarded_b4/r020` | 1,335,426.102 | 1,335,981.875 | 1,338,184.637 | 2,202.762 | 554.753 |

这两个请求都在视频解码结束后才到达，`blocked_by_ms` 为空。旧执行器在记录视频完成后同步 `np.save`，然后释放状态、写 checkpoint 并进入下一次派发。对应 replay 的输出保存合计分别为 5.689 秒和 5.636 秒。七条超期图像记录与视频解码后空档的重叠合计为 19.030 秒。

原事件日志没有每个输出文件保存的独立起止，因此上述空档严格表示“解码后的持久化、checkpoint 和控制器间隙”；不能声称全部由 `np.save` 导致。独立保持 EDF 与一个 block 不变、只更换输出保存方式，才能辨别这部分影响。普通到达场景必须保留，因为该场景已出现这一尾延迟线索。

排除整步对照后，286 个图像请求的到达至首次派发等待合计 224.658 秒：与视频去噪重叠约 97.564 秒、其他图像生成重叠约 88.978 秒、视频解码重叠约 13.626 秒。只有五个图像实际与视频 decode 重叠。这支持先做低改动的保存方式对照，再决定是否开发 VAE 分段抢占。VAE 分段必须维护解码器状态与时间上下文，并另做完整输出等价性验证；当前实现没有提供此能力。

## 本轮执行矩阵

执行配方继续固定为 Wan FP16/EFFICIENT 与 Sana 全 FP32/MATH，缓存关闭，设备锁定为 `GPU-a058066e-bde9-c534-efb1-d2a68c117681`。不切换 A100，不补写旧 A100 矩阵。阶段开始时校验旧校准、冻结输入、模型资产、设备与数值执行源码的来源，后续在线动作只读取已经到达且 GPU 就绪的视图。

| 阶段 | 新测内容 | 在线 replay 数 |
| --- | --- | ---: |
| `completion` | seed 301 的 `video_contention`，旧六臂 `edf_b1/edf_b4/guarded_b4/guarded_adaptive/feasible_b4/edf_step` | 6 |
| `granularity` | seed 301，四个原定场景，四臂 `edf_b1/edf_b4/edf_b2/edf_b1_async` | 16 |
| `replicate` | seed 401、402，四场景，`edf_b1/edf_b4/edf_step` 加开发阶段至多一个冻结候选 | 24 或 32 |

合计新增 46 次（没有候选）或 54 次（选出一个候选）在线 replay。`granularity` 之前另对两条 prompt 验证两 block：每条分别运行完整参考、原生逐步和暂停恢复轨迹，共六段完整视频计算，并检查中间 latent、sampler 状态、解码输出及 Sana 插入后的数值一致性。它们不计入在线 replay 数。

首组 `completion` 预计约 3–4 小时，属于规划估计而非完成承诺。各阶段按固定次序执行，并交替反转场景内臂顺序；单次阶段 watchdog 为 48 小时。预算无限不表示无限自动重试：技术失败、质量失败、外部占卡、来源不一致或输出保存失败均需停止并保留证据。

`completion` 将新六条记录与明确标注来源的旧 14 条记录组合成一份新的完整性审计，旧文件和旧资格不改写。`granularity` 的四臂全部重新运行，不能拿旧同步测量直接充当新异步臂的配对基线。

`edf_b2` 复用旧纯 EDF 的请求选择规则，只改变 block 量子为 2。`edf_b1_async` 复用旧 `edf_b1` 的派发规则，只改变已生成 CPU 数组的保存方式；执行时刻改变后，就绪集合和实际派发顺序也可能改变。两个候选分开报告，不能把异步保存的收益称为抢占策略收益。原始 `policy.ARMS` 保持不变，由独立 `next_policy` 适配新臂。

候选仍须相对两个强对照同时满足原开发门槛：压力场景合计 goodput 至少提高 5%，原场景各自不下降超过 2%，每场景视频按时数不降低。完整 16 条开发矩阵结束后才允许选择。两候选均通过时按总体 goodput 选一个，差异小于 1% 优先 `edf_b2`。未选出候选时，留出阶段仍复现三个基础粒度对照；不能利用留出重新选择候选。有限 seed 结果不支持稳定 p99 或一般最优性结论。

## 有界异步输出与计时口径

`next_persistence.AsyncArtifacts` 是单独的 CPU NumPy 保存器。构造不创建 replay 目录；执行器创建目录后调用 `save(request_id, output, latent)`，得到与同步路径相同的绝对文件路径。返回只表示提交成功，文件必须在 `drain()` 完成后才可读取。只接受 NumPy 数组，不接收 CUDA tensor，不复制或修改数组；调用者在 drain 前不得原地修改这些数组。

一个后台线程串行保存，运行中加排队最多两个请求批次，因而最多两个在途视频输出。提交窗口已满时，前台等待并累计 `queue_wait_ms`。每次 replay 结束必须 drain，异常和正常退出都必须 close；close 即使遇到保存失败，也先确保后台线程结束再抛出异常。后台失败在下次 save、drain 或 close 传播，未写完的排队批次取消，失败运行不得取得质量或性能资格。

图像／视频请求的 `finish_ms` 仍表示完整输出已经计算完成，与同步基线保持相同边界；文件保存属于实验产物记录。完整 replay 的 `elapsed_ms` 和 goodput 分母包含最终 drain。异步臂额外记录 `serving_complete_ms`，仅辅助区分服务计算结束与产物排空结束，不能替换主要分母。

`output_persistence_ms` 在同步臂记录同步保存墙钟，在异步臂记录提交和容量等待；`artifact_writer.write_ms` 是可能与 GPU 计算重叠的后台工作量。`drain_wait_ms` 记录最终前台等待，`submitted/completed/files/bytes/max_in_flight` 核对产物规模与队列上界。不能把后台写盘时间再次加进前台开销，不能从实际时间中事后扣除写盘、checkpoint 或调度成本。checkpoint 和事件日志方式在新四臂之间保持一致。

新候选筛选与复现阶段还逐请求记录保存／提交、状态释放的起止时间和耗时，辅助定位解码后的派发间隙。补齐旧矩阵阶段沿用原计时路径。旧记录的 checkpoint 每 replay 从约 2.177 秒增长至 35.401 秒，与整个阶段累计快照变大有关；报告保留实测 checkpoint 开销，不能把顺序相关的记录成本误认为策略收益。

本模块的 CPU 测试覆盖两批次容量与第三次提交等待、单后台线程、只读非连续数组回读、关闭期间拒绝等待提交者、异步失败传播和线程退出；`next_policy` 测试覆盖新臂的 EDF 一致性与合法两 block 成本。通过这些检查仍不表示异步臂有 serving 收益。

## 入口与留存

以下入口使用新目录和新账本；已有阶段由监督器和控制器接续，不重复提交同一阶段。

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve.scheduling.next_runner preflight
PYTHONPATH=python .venv-previs/bin/python -m ditserve.scheduling.next_runner prepare
PYTHONPATH=python .venv-previs/bin/python -m ditserve.scheduling.next_runner run --phase completion
PYTHONPATH=python .venv-previs/bin/python -m ditserve.scheduling.next_study
PYTHONPATH=python .venv-previs/bin/python -m ditserve.scheduling.next_runner report
```

这些是各入口示例；已有 `completion` 运行时不再执行第三行，控制器会等待已有监督结果再接续。父监督进程根据配置使用 V100 解释器，源码在各阶段启动时冻结。每阶段先保存报告再继续；`controller.json`、阶段 `result.json`、`supervisor.json` 与 ledger 保留实际状态。

长期记录使用经过回读验证的轻量 Excel、统计摘要和文件 SHA256。活动实验、失败或未验证产物继续保留。仅在新运行的全部声明阶段技术完成、账本结算且导出验证成功后，按显式清单清理已验证新运行的原始数组和已经摘要覆盖的事件日志；旧来源目录始终只读。删除结果单独记录，不能以清理动作覆盖旧实验结论。
