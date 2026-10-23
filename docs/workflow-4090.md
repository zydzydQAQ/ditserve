# 双 4090 视频 DiT 工作流实验

这套实验测量 Wan 的 CPU 文本编码、GPU 去噪、VAE 解码和数据搬运，并比较请求间流水、双卡并行及通信重叠。实验入口位于 `workers/ditserve/evaluation/workflow_*.py`，不修改生产 worker 的默认部署。环境和模型来源见 [本机双 RTX 4090](rtx4090-local.md)，归档规则见 [结果保存与收尾](results.md)。

## 条件与实验范围

默认使用本地 Wan2.1-T2V-1.3B，480×832、17/33 帧、20 步、CFG=5，DiT FP16、VAE FP32。DiT 使用 Flash attention，VAE 使用 MATH attention；保持运行镜像的 cuDNN TF32 默认配置，编码实验关闭 FP32 matmul TF32，并记录两项开关。Sana 使用本地 0.6B 模型和现有 1024×1024 配方。

正常阶段测量预热一次、重复五次；队列对照默认四个请求、三轮，支持的对照按轮次交替顺序。E6 各方案按固定顺序独立预热、重复测量。冷加载单独记录，不计入热请求延迟。T0–T5 使用固定英文提示词及 seed=20261019；另对短英文、中文、截断边界和非空负提示词做一次编码数值检查。E1–E6 使用另一组四条提示词及 seed=1234+i。

**两组实验的提示词、种子不同，不能直接进行跨组输出哈希比较，也不能把 T5 与 E2 的结果称为同负载配对加速比。** 各组内部保持输入一致；跨组比较需先统一请求规格再重测。

| 实验 | 实际比较内容 |
| --- | --- |
| E1 | 同一队列的串行 CPU 编码与后台预取；包含编码、推理和输出检查。 |
| E2 | GPU 1 单卡 `single`、GPU 0 单卡 `single_other`、双卡分担请求 `dual` 三种方案，分别测预编码输入和包含 CPU 编码的完整流水线，三轮轮换执行位置。三种方案的每个 worker 都使用 `prefetch` 调度：完整流水线在后台预取 CPU 编码并与 GPU 工作流水执行；预编码方案提前缓存 embedding。双卡分别与两张单卡基线配对比较；模型加载、预热在共同开始时间之前。 |
| E3 | 同步 H2D 与 pinned buffer、独立 copy stream 的异步 H2D；两组均传 FP32、在 GPU 转 FP16。CPU 编码预先完成，输入准备和 pinning 另报耗时。 |
| E4 | 同步输出、后台 CPU 输出、后台输出加异步 D2H；最多保留两个待完成输出。默认写 `.npy`，可另用 `--output-mode check` 对照。完成时间包含检查和所选写文件操作。 |
| E5 | Wan 单独、Sana 单独、同 stream 交替、同进程双 CUDA stream 并发；分别在 DiT/VAE 阶段加入十个 Sana 请求。VAE 交替以 causal latent-frame 为边界。当前实现是双 stream 执行器，不等于已完成 MPS 测试。 |
| E6 | 两卡 SP2：现有 cache-dit、自定义串行交换基线、Q/K/V 投影与通信流水、三组 head 通信与 attention 流水。每个 head 仍访问完整序列；NCCL P2P 禁用。 |

| 编码方案 | 实际部署与计时范围 |
| --- | --- |
| T0 | CPU FP32 编码，GPU 1 运行 DiT/VAE；完整视频基线。 |
| T1 | GPU 1 FP32 编码器单独运行，不加载 DiT/VAE。 |
| T2 | GPU 1 BF16 编码器单独运行，不加载 DiT/VAE。 |
| T3 | BF16 编码器与 DiT/VAE 同卡常驻，不加载 Sana；测完整视频和峰值显存。 |
| T4 | BF16 编码器每请求从 CPU 换入 GPU、编码后搬回 CPU 并释放缓存；换入、换出全部计入编码与整请求耗时。 |
| T5 | GPU 0 BF16 编码，embedding 经 CPU 中转到 GPU 1 去噪、解码；除阶段测量，还测串行与后台编码预取队列。 |

T1/T2 的阶段任务只测编码及 embedding 数值；下游完整视频由单独的 embedding 回放任务检查，不能把回放计时并入编码器常驻性能。T3/T4/T5 的完整视频直接与 T0 比较。模型是否装得下以正常运行结果为准；OOM、权限不足和 profiler 额外开销造成的失败需分别记录。

每个 worker 默认 `--cpu-affinity auto`，通过 NVML CPU affinity 和 Linux core/package 信息选择推理 GPU 本地的八个物理核，并记录实际掩码。单进程 T5 按 DiT 所在 GPU 绑定；两个 GPU worker 的 E2/E6 分别绑定。`--cpu-affinity none` 仅用于诊断，不能与默认绑定结果混作同一条件。线程数默认八，环境中关闭 tokenizer 自行扩展线程。

## 如何解释测量

| 指标 | 含义与边界 |
| --- | --- |
| CPU 平均核数 | 阶段进程 CPU 时间除以对应时间窗口；不是占整台 240 逻辑 CPU 服务器的百分比（本机 120 个物理核，每核两路 SMT）。默认八核绑定每个物理核只选一个逻辑 CPU。并发阶段的进程时间不能直接归因给某一个线程。 |
| 阶段耗时占比 | 同一请求的阶段窗口耗时之和除以该请求实测总耗时，再对配对比例取中位数。视频分母为 `inference_wall_ms`，单独编码分母为 `encode_ms`；这不是 CPU/GPU 利用率。 |
| SM Active | Nsight Systems 采样窗口内 SM 活跃程度；可按阶段时间加权，不能等同于算力利用率。 |
| Elapsed warp slots | 相对于 elapsed cycles 的活跃 warp 槽比例，分母不同于 NCU achieved occupancy。 |
| Achieved occupancy | Nsight Compute 针对采样 kernel 的活跃 warp 占可驻留上限比例；选定范围内前 128 个 kernel 的结果不能称为整个阶段的 occupancy，也不是随机抽样。 |
| 分配器显存 | 起点 allocated 是该请求阶段首个窗口开始时的活跃张量分配；allocated 峰值含常驻模型，reserved 峰值还含分配器缓存。阶段增量相对各窗口开始计算，聚合阶段取最大值；不能用总峰值减首个窗口基线替代，也不能跨阶段相加。 |
| 整卡显存 | 按设备/UUID及阶段窗口匹配 NVML 样本，报告每请求窗口内采样最大值，再取中位数。CUDA 阶段起止边界快照单独列出；缺少窗口内样本时不能将边界值称为完整阶段峰值。allocated/CUDA 可用容量与 NVML 已用/总量各有自己的分母。 |
| 显存带宽 | Nsight 硬件计数器测得的 DRAM 活动/吞吐；与显存容量是两个指标。NVML memory-I/O busy 不能替代 DRAM 带宽。 |
| 通信重叠 | CUDA 计算区间与 H2D、D2H、NCCL 区间的交集，分别除以对应通信区间并集。CPU API 很快返回并不能证明设备实际重叠。 |
| 计算/通信耗时占比 | 各类 GPU 区间并集除以该实验臂的执行包络；包络覆盖 host NVTX 及其关联 GPU 操作，包含异步 GPU 尾部。旧解析结果回退到明确标注的 host 包络。计算和通信可重叠，占比不必相加为 100%。 |

CPU 编码阶段自身没有 GPU kernel，但 GPU 上仍有常驻权重。硬件计数器反映整张卡；并发任务存在时不能仅凭阶段窗口把全部 SM 活动归因给该阶段。NCCL kernel 还可能包含设备等待，不能把其完整持续时间解释为线缆传输时间。

CPU 编码时 allocated 较小，不代表整卡显存只占这么多：上一轮 VAE 释放的张量可能仍留在 PyTorch reserved 缓存中，驱动也有额外占用。NVML 默认等待间隔为 50 ms，查询本身另有开销，报告展示阶段窗口内的实测采样周期；短阶段可能没有样本，短峰值也可能漏采。不能将采样周期写成恒定 50 ms，或把 sampled peak 当成连续监控的精确峰值。

正常运行提供延迟、吞吐和显存证据。阶段 runner 为归因而同步设备；E1–E6 的时间线 recorder 不主动插入设备同步。Nsight Systems 单独运行，提供 CUDA 时间线和默认 1 kHz 硬件采样。Nsight Compute 通过 kernel replay 获取所选算子的 occupancy/带宽等，改变执行、缓存和并发；其耗时不用于加速比。

E3 的 `wall_ms` 从 CPU 输入已准备完毕后开始，不含 latent 创建和 pinning；`input_setup_ms` 单列准备成本，`including_input_setup_ms` 将其加回。报告同时保留主计时加速比和含准备成本的配对加速比，两者均不含已预先完成的文本编码。

纯调度、搬运、输出异步化要求 latent/输出精确一致。E6 先用目标分辨率和帧数做四步 SP1/SP2 检查，latent/video 最大绝对误差均不超过 0.05 后才做完整性能测量；这不代表已经证明二十步数值误差也满足同一门槛。BF16 报告 finite、shape、max error、RMSE、cosine，数值差异不等于视觉质量结论。没有盲评时语义质量保持 `unverified`。

## 准备与运行

从项目根目录执行。模型位于 `/mnt/ssd0/zyd/model/`；数据、缓存和全部实验输出位于 `/mnt/ssd0/zyd/ditserve-data/`，容器内分别为 `/models` 和 `/data`。模型只读挂载，使用已导入的 `ditserve:rewrite-runtime` 镜像和专用 Docker socket。

选择尚不存在的运行名，例如 `repro01`。准备只冻结 `workers/`、`configs/` 和文件哈希，生成任务清单，不启动 GPU：

```bash
python3 docker/prepare-workflow-study.py --run-id repro01
workflow_root=/mnt/ssd0/zyd/ditserve-data/workflow/repro01

# 可先运行明确选择的子集；完整任务去掉 --only。
python3 docker/run-workflow-suite.py --jobs "$workflow_root/jobs.json" \
  --only T0-encoding,E1-f17
python3 docker/run-workflow-suite.py --jobs "$workflow_root/jobs.json"
```

清单顺序为 T0 基线、E1–E6 的 17/33 帧对照、T1–T5。运行器串行启动任务，保存命令、来源快照、日志、超时及 GPU 监控结果。`reserved_gpu_seconds` 是任务占用窗口乘 GPU 数量，包含加载和 CPU 等待，不是 kernel 活跃时间。

新清单的 CPU 基线任务 ID 是 `T0-encoding`；现有 `run01` 中手工安排的历史任务叫 `T0-baseline`。`--only` 应使用相应 `jobs*.json` 中的实际 ID，不能混用这两个名字。

桌面服务保持开启。启动前及运行中每两秒检查所选 GPU 的 compute/graphics 进程，发现外部占用会拒绝开始或中止本次实验容器，并停止当前批次；不会结束其他用户的进程。显卡监控或基础设施不可用也会保留失败状态并停止批次，不把无法验证独占的运行当作有效样本。

完成任务再次调用会跳过；失败任务须显式 `--retry-failed`，旧启动账本另存为 `attempt-N.json`，日志追加保留。**该开关不会自动为模型结果、trace 和 rendezvous 换输出路径。** 已经产生输出的失败任务应复制为新的 job ID，并把全部 `--out`/`--output`、nsys `--output` 改为新的目录；新条件或源码变更直接使用新 run-id。不要删除旧证据来让续跑通过。

## 分析工具与补测

时间线任务使用兼容 CUDA 12.6 的 Nsight Systems，默认路径由 `prepare-workflow-study.py --nsys` 指定。安装包可用 `docker/fetch-workflow-profiler.py --directory /mnt/ssd0/zyd/ditserve-data/tools/nsight-workflow` 下载并校验；该脚本只下载，不自动解包或安装。解包后的工具放数据盘即可，无需替换系统安装。

```bash
python3 docker/run-workflow-suite.py --jobs "$workflow_root/jobs-trace.json" \
  --only trace-T0,trace-T2,trace-E1-f17
```

分析任务默认一次预热、一次有效重复，仍执行实际二十步。标记为 `counters: true` 的任务通过独立容器的 root 和 `CAP_SYS_ADMIN` 获取计数器权限，不永久修改驱动权限。T5 默认时间线还包含四请求、一次配对的专卡流水队列；T0–T4 不增加队列测量。不要与正常性能测量同时运行 profiler。

准备脚本还生成 `jobs-ncu.json`：T1/T2 文本编码、T0 第二步 DiT 和 VAE 解码（后两项各含 17/33 帧），共六项诊断。每项预热一次、有效重复一次，仅采样指定 NVTX 范围的前 128 个 kernel；工作负载输出各自放在 `profiling/ncu/<id>/encoder`，报告放在 `ncu/`，不复用正常 T0 基线目录。

```bash
python3 docker/run-workflow-suite.py --jobs "$workflow_root/jobs-ncu.json"
```

NCU 任务同样经过 GPU 占用检查。限定到一个已测量阶段和少量 kernel，避免对整个二十步流程使用 `--set full`。例如 NVTX 过滤条件：

```text
GPU 编码：regex:ditserve[.]workflow.text_encoder.*"purpose":"encoder_measurement".*"repeat":0.*/
DiT 第2步：regex:ditserve[.]workflow.dit_step.*"purpose":"video_measurement".*"repeat":0.*"step":1[^0-9].*/
VAE 解码：regex:ditserve[.]workflow.vae_decode.*"purpose":"video_measurement".*"repeat":0.*/
```

配合 `--nvtx --nvtx-include '<过滤条件>' --range-filter ::1`，用时间线中的主要 GEMM/卷积 kernel 名称进一步筛选，再设较小的 `--launch-count`。末尾 `/` 表示 NVTX push/pop range；过滤条件中的通配字符避免把标签里的 `/` 误作嵌套范围分隔符。T0 的 CPU 文本编码不适用 GPU kernel occupancy。

Profiler 任务结束后，优先使用宿主机后处理入口导出并解析已有 capture：

```bash
python3 docker/postprocess-workflow-study.py --root "$workflow_root" --preview

# parser 更新后重新解析；旧解析结果和 CSV 归档为 .bak，原始 capture 保留。
python3 docker/postprocess-workflow-study.py --root "$workflow_root" --reparse --preview

# 全部 NCU 采集结束后，仅重新解析 NCU；保留已有时间线解析和来源绑定。
python3 docker/postprocess-workflow-study.py --root "$workflow_root" --kind ncu --reparse --preview
```

它只处理 `traces/*.nsys-rep` 和 `ncu/*.ncu-rep`，跳过 probe 和符号链接，不启动 GPU 工作。每个 capture 保存 SHA256、导出/导入命令及 stdout/stderr 日志，结果写为同名前缀的 `.parsed.json`，汇总账本为 `postprocess.json`。已有 SQLite 经完整性校验后复用；`--reparse` 更新解析结果，不重新导出已有 SQLite。处理过程中 capture 发生变化会失败并保留原因。缺少某类 capture 只是缺测，不生成零值。

NCU 2024.1 的 CSV 可能在 NVTX 标签中输出未转义的 JSON 引号。导入器只对可完整解码的 workflow 元数据修复 CSV 引用，再检查列数、kernel 身份、指标单位及重命名导出的关联；原始 capture 和 CSV 保留。解析为空表示缺测或解析失败，不能据此报告零 occupancy 或零吞吐。采集进程运行期间使用下面的 watcher，只处理已完成任务；`--kind` 本身不检查任务是否结束。

`--preview` 另外更新 `ANALYSIS-LIVE.html`，不写规范 `run.json`、正式 `ANALYSIS.html` 或最终归档。仅刷新报告也可运行 `PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_report --root "$workflow_root" --preview`；此路径不需要 `openpyxl`。持续运行时，匹配到尚未完成或被 guard 中止的 launch 的记录不进入有效性能/计数器统计。

运行期间可以并行启动 CPU 后处理，让已完成 capture 的导出、解析和预览与后续 GPU 测量同时进行：

```bash
python3 docker/watch-workflow-postprocess.py --root "$workflow_root"
```

当前 `run01` 默认读取 resume、replay、trace、NCU 四份任务清单；其他运行可用 `--jobs` 显式列出清单。watcher 默认绑定测量进程未使用的 CPU 10、nice=19，每 45 秒检查一次。它要求匹配的启动记录成功退出、GPU 监控无冲突、capture 时间落在该次启动窗口内，并连续两次观察到文件不变，才执行 CPU 导入。它不会启动模型或采集，也不会写最终归档。状态和日志为 `watch-postprocess.json`、`watch-postprocess.log`；创建 `watch-postprocess.stop` 可要求退出，重新启动前需移走该停止文件。没有活跃实验进程持续 90 秒后自动退出，最长运行 48 小时。普通后处理命令与 watcher 共用写锁，默认最多等待 600 秒，watcher 会在处理完当前 capture 后让出锁。

文本编码、阶段显存和数值回放还可单独生成紧凑的 CPU 汇总：

```bash
taskset -c 9 nice -n 19 python3 docker/summarize-workflow-encoding.py --root "$workflow_root"
```

结果写入 `reviews/parallel-analysis/encoding-summary.json`，复用正式报告的阶段聚合和启动有效性检查。数组只读取格式头；数值比较来自原始实验记录，不代表重新计算了数组内容，也不代表已完成视觉质量验收。

需要手工处理单个 capture 时，可用以下命令；输出名与后处理入口保持一致，已存在的解析输出须先保留旧版本。当前 parser 可从请求 ID 恢复 T5 的帧数及 serial/prefetch 分组，并包含异步 GPU 尾部，不需要重跑已捕获的模型工作：

```bash
workflow_nsys=/mnt/ssd0/zyd/ditserve-data/tools/nsight-workflow/extracted/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys
"$workflow_nsys" export --type sqlite \
  --output "$workflow_root/traces/T0.sqlite" "$workflow_root/traces/T0.nsys-rep"

PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_trace \
  --sqlite "$workflow_root/traces/T0.sqlite" \
  --out "$workflow_root/traces/T0.parsed.json"

PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_ncu \
  --report "$workflow_root/ncu/ncu-T2-text_encoder.ncu-rep" \
  --out "$workflow_root/ncu/ncu-T2-text_encoder.parsed.json"
```

这两个解析命令不启动模型或 CUDA 工作。缺少采样、kernel 表、权限或匹配范围时保留缺测及错误，不能把空值补成零。

T1/T2 编码进程结束、显存释放后，再回放保存的英文 embedding。回放先要求 T0 的 latent/video 与正常基线精确一致，随后比较 T1/T2；不再次执行文本编码，也不产生性能结论。当前源码已提供 `workflow_embedding_replay`，新运行名的快照会包含它；早期不含此模块的冻结快照应另建补测快照并保留其哈希，不能原地修改。

例如将以下内容保存为 `$workflow_root/jobs-replay.json`，然后运行 `python3 docker/run-workflow-suite.py --jobs "$workflow_root/jobs-replay.json"`：

```json
{
  "root": "/mnt/ssd0/zyd/ditserve-data/workflow/repro01",
  "source": "/mnt/ssd0/zyd/ditserve-data/workflow/repro01/sources/frozen",
  "jobs": [{
    "id": "embedding-replay", "devices": [1], "gpu_count": 1,
    "timeout_seconds": 3600,
    "command": ["/opt/venv/bin/python", "-m", "ditserve.evaluation.workflow_embedding_replay",
      "--encoder-root", "/data/workflow/repro01/encoder",
      "--out", "/data/workflow/repro01/embedding-replay",
      "--variants", "T1,T2", "--frames", "17,33", "--steps", "20",
      "--device", "cuda:1", "--model-path", "/models/wan2.1-t2v-1.3b"]
  }]
}
```

该入口的 `--validate-only` 只核对元数据、数组格式、精度及基线哈希，不导入 Torch、不初始化 CUDA；预检也会写 `run.json`，应使用独立输出目录。它不能代替实际 T0 桥接及 T1/T2 下游数值检查。缺失或未完成的编码方案标为 unavailable。

## 证据目录与归档

工作目录 `/mnt/ssd0/zyd/ditserve-data/workflow/<run-id>/` 保留：`sources/` 冻结源码、`jobs*.json` 任务清单、`launches/` 启动账本和日志、`encoder/` 阶段记录/数组、`overlap/` 对照结果、`profiling/` profiler 工作负载、`traces/` 时间线和解析结果、`ncu/` kernel 报告。早期 `run01` 可能含 `encoder-v2/`；报告时显式选择有效编码基线，旧版本只作历史上下文。

最终使用运行镜像内含 `openpyxl` 的解释器生成报告；当前会话需拥有专用 Docker 组权限：

```bash
./docker/run-workflow-profile.sh /opt/venv/bin/python \
  -m ditserve.evaluation.workflow_report \
  --root /data/workflow/repro01 \
  --encoder-root /data/workflow/repro01/encoder
```

报告命令不加载模型，输出工作目录内 `ANALYSIS.html`、规范 `run.json`，并按 [结果保存与收尾](results.md) 写入数据盘 `results/workflow/<run-id>/REPORT.md` 和 `results.xlsx`。默认保留关键实验证据并精简冗余数据：原始 `.nsys-rep` / `.ncu-rep`、解析 JSON、指标、日志、冻结源码及正常编码/embedding 回放的质量数组保留；具备 capture 和有效解析的 SQLite、profiling 重复数组、E4 输出测速数组及哈希完全相同的旧编码 embedding，经精确登记和归档验证后删除。需要完整原始副本时加 `--keep-raw`；`--preview` 不清理数据。已有删除凭据后重复运行只复用并校验原归档，不重新生成规范数据或修改指标哈希。

历史运行可直接使用 `PYTHONPATH=workers python3 -m ditserve.evaluation.workflow_retention --root /mnt/ssd0/zyd/ditserve-data/workflow/run01 --dry-run` 查看清理计划；去掉 `--dry-run` 才执行，解释器需有 `openpyxl`。活动任务、未闭合依赖或占用中的驱动/后处理锁会阻止相关清理；模型和冻结来源不在清理范围。完整命令及删除凭据说明见 [Workflow 关键证据保留](results.md#workflow-实验的关键证据保留)。

报告分别标注技术执行、数值门槛、语义质量和性能证据；尚未运行或未完成的实验保持缺测/未验证状态，不因生成报告而变成已通过。

报告将正常阶段/队列计时、Nsight 时间线与硬件计数器、NCU 采样 kernel 诊断、embedding 下游数值回放分开。E2 标明每个方案的设备和两个单卡基线；T5 只将自身同帧数、同轮次的 serial/prefetch 配对比较，latent 与 video 哈希均需一致。源记录按输出路径和时间匹配启动账本：超时、外部 GPU 占用或监控失败会否决该轮证据，即使子程序写下 `completed`；无匹配账本及早期无监控记录明确标注。Profiler 账本匹配原始 capture 前缀及时间，不使用较晚的 JSON 导入时间。

完成度以根目录的 `jobs*.json` 为计划依据，按任务 ID 去重，排除 probe 和执行前撤换的历史任务，并保留替代任务说明。正常阶段须覆盖所有计划帧数任务；trace/NCU 另查原始 capture 与有效解析，replay 另查计划中的变体/帧数组合。仅完成一个帧数或正常任务全部完成，不代表整套研究完成。OOM 单独标为容量不足的负面结果，不算性能通过。没有计划文件时标注为仅已发现证据；显式 `--include` 标注为选定证据范围，两者均不宣称整套任务已完成。
