# 双 4090 DiT serving 补充实验：run02

本轮在当前双 RTX 4090 上验证在线混合负载、编码器放置、CPU 预取、同卡 DiT/VAE 重叠与 SP 数值一致性。它延续 [第一轮工作流实验](workflow-4090.md)，使用独立目录和冻结源码，不改生产 worker 默认行为，也不重新执行第一轮完整 Nsight Compute 矩阵。

本轮还没有 A100 实测。4090 上的吞吐、显存余量、通信成本和双卡扩展比例不能直接换算成 8×A100 40GB PCIe 的结果；SP 仍保留 `NCCL_P2P_DISABLE=1`。

## 环境与输出位置

从 `/home/zyd/code/ditserve` 执行下文命令。使用已导入的 `ditserve:rewrite-runtime` 镜像和专用 Docker socket `unix:///run/ditserve-docker/docker.sock`；启动器通过 `ditserve-docker` 组访问，无需重复安装镜像或下载模型。

| 内容 | 主机位置 | 容器位置 |
| --- | --- | --- |
| Wan 模型，只读 | `/mnt/ssd0/zyd/model/wan2.1-t2v-1.3b` | `/models/wan2.1-t2v-1.3b` |
| 实验及缓存数据盘 | `/mnt/ssd0/zyd/ditserve-data` | `/data` |
| 本轮输出 | `/mnt/ssd0/zyd/ditserve-data/workflow/run02` | `/data/workflow/run02` |
| 本轮冻结源码 | `run02/sources/frozen` | `/workspace`，只读 |

共同模型配方为 Wan2.1-T2V-1.3B、480×832、17/33 帧、20 步、CFG=5；DiT FP16/Flash、VAE FP32/MATH。FP32 matmul TF32 关闭，cuDNN TF32 保持开启。在线比较的两个 worker 各绑定八个物理核：GPU 0 对应 `0–7`，GPU 1 对应 `60–67`；控制器使用 CPU 59。CPU 核数实验使用明确记录的掩码，不能与在线比较的固定 CPU 预算混淆。

## 准备与运行

准备只执行一次；它解析镜像为不可变 digest，复制 `workers/`、`configs/`、`docker/`、`tests/python/`，记录文件哈希并生成初始任务。准备不会启动 GPU 工作，`docs/` 不属于冻结源码。

```bash
cd /home/zyd/code/ditserve
python3 docker/run-serving-study.py --run-id run02 --prepare
```

目标目录必须尚不存在。已准备的 `run02` 直接运行或续跑：

```bash
python3 docker/run-serving-study.py --run-id run02 --run
```

最终归档默认精简经过验证的冗余数据；需要完整原始副本时加 `--keep-raw`。旧冻结控制器保持原样，采用新归档逻辑时按下文创建分析修订，或使用当前源码的 CPU 精简入口处理已结算目录。

入口默认自动 `exec` 到 `run02/sources/frozen/docker/run-serving-study.py`。若存在已绑定的 `analysis-source.json`，则使用其中记录的控制器/报告修订版，并校验完整文件摘要；所有 GPU 任务仍由 `study.source` 指向的原始测量快照执行。驱动依次执行任务，GPU 实验始终串行；一个双卡任务内部可以同时使用两张卡。

也可以显式指定已准备的根目录：

```bash
study_root=/mnt/ssd0/zyd/ditserve-data/workflow/run02
python3 docker/run-serving-study.py --root "$study_root" --run
```

每次恢复都会验证冻结源码实际字节、输入配置、镜像/任务指纹。已完成任务还会检查其 `measurements/<job>/run.json` 的 SHA256，验证成功才跳过。源码、配置或完成结果被改动时会拒绝继续；修改实验条件应准备新的运行名。模型清单对小文件记录 SHA256，对大权重记录大小及修改时间，模型挂载保持只读。

## 本轮矩阵与顺序

校准完成后才生成在线到达轨迹、负载率、SLO 和路由成本；这些值一经生成便被冻结。每条混合轨迹有 16 个请求，17/33 帧各半，使用指数到达间隔；同一比较使用相同请求、到达时间、种子和负载率。三条独立轨迹的种子为 `20261021–20261023`。

| 阶段 | 实际内容 | 重复与门槛 |
| --- | --- | --- |
| 校准 | 双卡 CPU 编码 worker 的三组 16 请求饱和队列；17/33 帧分别隔离测量。 | 每种帧数三个隔离样本；饱和吞吐中位数定义 `λ0`，各帧数隔离服务时间中位数的两倍定义实验 SLO。校准不完整则停止后续在线比较。 |
| 编码数值补测 | 中文、截断提示词 × 两个种子 × 17/33 帧；CPU FP32 普通执行、CPU embedding 桥接、GPU FP32/BF16 embedding 回放。 | 先验证 CPU 桥接 exact，再报告各精度的 embedding、latent、video 差异；回放只作数值证据，不作为常驻部署性能。 |
| 放置比较 | A：两张卡分别运行 CPU FP32 编码 + DiT/VAE；B：GPU 0 专门运行 FP32 编码器，经 CPU embedding 中转给 GPU 1 的 DiT/VAE。 | `0.5λ0`、`0.8λ0`，每档三条配对轨迹；两种配置使用相同全局 FIFO 和到达轨迹。整卡余量必须满足门槛。 |
| 在线路由 | 双 CPU 编码 worker 的 `shared_fifo` 与 `least_work_fifo`。 | `0.5λ0`、`0.8λ0`、`1.1λ0`，每档三条轨迹；其中低负载共享 FIFO 基线与放置比较复用。 |
| CPU 与预取 | 2/4/8/16 核的短英文、截断编码微基准；随后比较八核和一个由微基准选择的对照核数，预取深度 0/1/2。 | 编码微基准预热一次、重复三次；预取每次四个混合请求、三轮。对照核数选择规则冻结为与八核编码中位耗时差异最大的候选，不能预称为最优核数。 |
| P4 DiT/VAE | A 去噪与 B 解码：17/17、17/33、33/17、33/33 四种帧数组合，串行与两个 CUDA stream。 | 全四组合先普通 runtime→串行→双 stream exact 门槛；通过后每组合三次配对。单 host 提交线程，独立请求状态。 |
| P5 SP 诊断 | SP1 自重放、SP2 `cache_dit`、`baseline`、`qkv_pipeline`、`heads_pipeline`。 | 固定完整 20 步 scheduler，依次运行前缀 1、4、20；前级未过不进入后级。每步 latent 与前缀视频最大绝对误差门槛为 0.05，不放宽。失败后定位首个分歧 block 及其 Q/K/V。 |
| 可选确认 | 只对满足预先规定筛选条件的一个放置或路由方案生成新轨迹。 | 六条新种子轨迹，每条 32 请求；仍受剩余预算和完整 block 准入控制，不保证全部执行。 |

`shared_fifo` 将已到达请求保留在全局 FCFS 队列中，先给 GPU 空闲 worker，再使用可用预取 slot，并列按 rank。`least_work_fifo` 使用相同 slot 限制，在领取全局队头时按冻结的预计剩余服务量选择 worker；领取后保持该 worker FIFO。预测只扣实际正在执行阶段的 elapsed，报告预测误差，不声称最优调度。

在线等待队列上限为 64，预取深度为 1。深度包括正在编码和编码已 ready 的请求，不包括当前 GPU 消费者。请求到达前不进行编码；超额等待请求记录为拒绝，不从 SLO 分母中删除。轨迹有明确截止点，截止时未完成和未到达的请求也保留在结果中。

## 计时、容量与正确性

正常在线计时从预定到达时刻到 D2H、finite 和 hash 检查完成，模型加载和预热在观测窗口外。报告同时保留队列等待、编码、GPU 服务、响应时间、吞吐、SLO 达成率和预测误差。短轨迹 p95 只是描述性结果；同一条轨迹中的请求不能当作独立重复样本。

P4 主指标 `wall_ms` 包含两个预编码请求的 GPU prepare、B 的 20 步 DiT、B VAE 与 A DiT 的串行/重叠窗口、A VAE，以及两者 D2H/finite/hash。`stage_pair_wall_ms` 单独描述 A DiT + B VAE 窗口，不能替代完整两请求指标。CUDA 完成事件和 NVML busy 不是实际 kernel 重叠的证明；没有对应时间线时只报告计时观察。

单独的 P4 Nsight Systems 诊断限定为 17/17 帧、一轮串行/双 stream 配对。解析器按同一 GPU 上 `dit_step/role=A` 与 `vae_decode/role=B` 的实际 kernel 区间求交集，输出 `dit_vae_kernel_overlap_ms`，排除 B 的前置去噪、外层 NVTX 范围和拷贝；缺少任一角色的 kernel 时为不可用，不填零。该 profiler 任务只补充时间线证据，不参与正常计时加速比。

P4 和在线放置比较要求观测到的空闲显存至少为 `max(1 GiB, 5% 整卡容量)`。NVML 缺测不能当作容量通过；采样峰值也可能遗漏短暂尖峰。NVML GPU busy、memory-I/O busy 分别不等于 SM Active、DRAM 带宽或 achieved occupancy。本轮默认不运行完整 NCU，也不把 profiler replay 时间并入正常性能。

P5 保存每步 latent/CFG prediction 统计与首个非零、首个越阈值步骤附近的张量。数值失败后才执行逐 block 诊断：将 SP1 对应序列分片与 SP2 本地输出比较，再保存选定 block 的输入、Q/K/V、norm-Q/norm-K、输出投影与输出。诊断重放、CPU 拷贝和文件写入均不是性能证据；内部层差异不改变原有 0.05 请求门槛。

文本精度的数值接近不等于语义或感知质量通过。未完成盲评时，语义质量保持 `unverified`，SLO 达成率不能称为已验收质量的 goodput。

## 八小时累计预算与桌面保护

`budget.json` 的总额度固定为 28,800 秒，重启驱动、续跑和重试都不重置。记账包括任务启动、模型加载、预热、测量、失败和清理；准备与报告生成不属于任务热路径。若进程异常退出留下 active attempt，恢复时会保守计入从开始到恢复的耗时，并标记 `recovered_uncertain`，不会将这段时间免费抹去。

| Family | 初始额度 |
| --- | ---: |
| calibration_quality | 60 分钟 |
| placement | 100 分钟 |
| online | 90 分钟 |
| cpu | 60 分钟 |
| overlap | 45 分钟 |
| sp | 60 分钟 |
| extension | 65 分钟 |

前面 family 的剩余额度可顺延到后面的 family；这些额度不等于保证执行时间。驱动会按整个 block 估计是否有足够余额，另留 90 秒清理余量；不够时记录 `complete_block_budget_deferred` 或 `family_budget_deferred`，保留未执行任务。任务自身还受独立超时约束。八小时是累计任务额度，并不承诺八小时内完成全部矩阵；停止后可能得到 `partial` 报告。

桌面和远程服务保持开启。启动前及运行中约每两秒检查所选 GPU 的外部 compute/graphics 进程；若桌面或其他工作负载占用目标卡，任务会拒绝开始或停止本次实验容器，驱动停止继续提交。监控不可用也停止，不推定 GPU 独占。看门狗只清理本实验的唯一容器，不结束桌面或其他用户进程。

## 续跑与完整 block 重试

普通续跑使用相同 `--run` 命令。已完成任务通过哈希校验后跳过，已有未完成 attempt 不会直接覆盖。

遗留 `budget.active` 必须先核实原启动器和唯一命名实验容器已停止。驱动在持有任务锁时自动调用恢复助手，预算锁覆盖身份检查、清理与保守结算；无法验证时保留 active 并拒绝提交。也可单独运行 `python3 docker/recover-serving-study.py --root "$study_root"` 查看恢复审计，结果保存在 `recoveries/`。

仅修正报告或控制逻辑时，可在旧控制器完全退出、当前任务完成并结算后运行 `--freeze-analysis`，再按普通命令续跑。修订保存于 `sources/analysis-NN/`，文件摘要和原测量来源写入独立绑定文件与 `analysis-source.json`。运行中的控制器固定该修订，指针或文件变化会被拒绝；不要用这个机制更换实际测量代码或重置预算。

若一个比较 block 被中断或需重做，先等待原任务及看门狗完成清理、预算结算，再为整个 block 创建新 attempt：

```bash
study_root=/mnt/ssd0/zyd/ditserve-data/workflow/run02
python3 docker/run-serving-study.py --root "$study_root" --run \
  --retry-block load-0.5-r0
```

`BLOCK` 必须是 `study.json` 中当前未被 supersede 的实际 block 名。`load-0.5-r0` 是一个例子，不能代替其它失败 block。该操作将 block 内所有 arm 一起重试，包括其中已经完成的 arm；新的 job ID、输入路径和输出目录带 `-attemptN`，历史结果继续保留，旧任务标记为 superseded。恢复的预算仍累计原 attempt 成本。再次重试时使用当前 block 名，例如 `load-0.5-r0-attempt2`。

可用以下只读命令查看当前 block 和 job：

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path('/mnt/ssd0/zyd/ditserve-data/workflow/run02')
study = json.loads((root / 'study.json').read_text())
for job in study['jobs']:
    if not job.get('superseded'):
        print(job['block'], job['id'])
PY
```

本轮不要直接用旧 suite 的 `--retry-failed` 绕过严格恢复：失败输出、数值诊断文件和 NCCL rendezvous 都应获得新的 attempt 路径。

## 查看结果与状态

驱动每个任务结束后刷新预览；也可只读数据并刷新报告，不启动 GPU：

```bash
python3 docker/run-serving-study.py --root "$study_root" --preview
```

| 文件 | 用途 |
| --- | --- |
| `study.json`、`inputs/`、`calibration.json` | 冻结的计划、请求轨迹、配置、校准值及跳过/重试决策。 |
| `budget.json` | 累计额度、每次 attempt 成本、active/恢复状态。 |
| `analysis-source.json`、`sources/analysis-*.binding.json` | 独立控制器/报告修订摘要；测量来源仍为原始快照。 |
| `launches/<job>/launch.json`、`stdout.log`、`stderr.log` | 实际命令、哈希、容器、GPU 独占检查、退出及超时证据。 |
| `measurements/<job>/run.json` | 观测请求、计时、资源、正确性和实验自身状态；诊断张量保留在同目录下。 |
| `ANALYSIS-LIVE.html`、`analysis-preview.json` | 进行中或中断后的预览。 |
| `ANALYSIS.html`、根目录 `run.json` | 驱动结算后的汇总；结算不代表所有计划任务都成功。 |

启动账本 `completed` 只表示该命令正常退出。实验的 `qualification_failed`、`correctness_failed`、`capacity_failed`、`invalid_memory_margin` 等仍需要单独阅读，不能按退出码将其计入有效加速比。P5 数值失败是负面实验结果，不是“SP 已通过但速度慢”；后续 4/20 步可能因此明确未运行。

汇总 `partial` 表示存在未执行、不完整或失败的计划证据；应查看 family/block 的预算与失败原因。`study.status=settled` 表示驱动完成本次结算，不是宣称完整矩阵成功。失败和重试记录保留；第一轮 `run01` 继续独立保存。

驱动完成结算后先释放 `driver.lock`，再归档及精简。质量视频、latent、embedding、图集、SP/重叠诊断、原始 profiler capture、JSON、日志和冻结源码保留；只有符合 [关键证据保留规则](results.md#workflow-实验的关键证据保留) 的冗余副本可经大小/SHA256 匹配及报告回读验证后删除。活动 study、预算会话、锁冲突和未闭合依赖仍受保护，不能因为报告生成成功而忽略。

已结算历史目录可单独查看和执行精简，无需重启驱动或 GPU：

```bash
PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_retention \
  --root "$study_root" --dry-run
PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_retention \
  --root "$study_root"
```

解释器需有 `openpyxl`。使用报告入口时，`workflow_serving_report --root "$study_root" --final` 默认归档并精简，可用 `--archive <目录>` 指定交付目录、`--keep-raw` 保留原始文件。不带 `--final` 的预览和 `--quality-gallery` 均不清理。已有删除凭据后再次生成正式报告会复用原始 `run.json` / `ANALYSIS.html` 并校验两文件归档，避免已删除输入导致历史指标或时间戳变化。
