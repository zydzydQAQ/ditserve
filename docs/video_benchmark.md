# 视频 DiT 在线服务 Benchmark

本套件把公开提示词、冻结请求流、执行后端和质量评测分开。首个真实后端为已有的固定 Wan2.1-T2V-1.3B worker：单 GPU、batch=1、CPU 文本编码、FP16 DiT、FP32 VAE、UniPC 与 MATH attention。它是可重复的调度实验环境，不代表经过内核优化的生产服务性能。

## 准备与运行

从项目根目录运行。CPU 示例依赖 NumPy；安装 `pip install '.[reports]'`，或者使用本仓库已有 `PYTHONPATH=benchmarks:.cache/tools`。真实 Wan 使用已有 `.venv-previs/bin/python`；不修改历史环境或下载新的模型。

```bash
export PYTHONPATH=benchmarks:.cache/tools
python3 -m ditserve benchmark prepare --dataset vbench --download --output .cache/benchmark-data/vbench.json
python3 -m ditserve benchmark profile --config smoke --dataset .cache/benchmark-data/vbench.json --backend cpu-test --output results/video-benchmark/example-profile
python3 -m ditserve benchmark trace --config smoke --dataset .cache/benchmark-data/vbench.json --profile results/video-benchmark/example-profile/profile.json --pattern poisson --load 0.7 --output results/video-benchmark/example-trace
python3 -m ditserve benchmark run --trace results/video-benchmark/example-trace/trace.json --backend cpu-test --policies fcfs,edf,srtf --output results/video-benchmark/example-run
python3 -m ditserve benchmark evaluate --run results/video-benchmark/example-run
python3 -m ditserve benchmark report --run results/video-benchmark/example-run
```

输出目录必须使用新名称。已生成的数据和 trace 不覆盖；重跑创建新的 attempt，不能拼接两次在线时间线。`run --dry-run` 只校验冻结输入，不加载 GPU。

对真实 GPU，把 `profile` 和 `run` 的 Python 换成 `.venv-previs/bin/python`，后端换成 `--backend wan --device cuda:0`，并为 profile、trace、run 使用独立目录。真实后端必须使用真实 profile；CPU fixture 的 profile 不能用于真实实验。`--model-path` 可指定同一固定模型的本地 snapshot。

默认 GPU 预算账本为 `results/video-benchmark/gpu-budget.json`，总上限 7200 GPU 秒，含加载、预热、失败和持卡工作。多进程不能同时使用该账本；中断会保留并保守收费。历史 PreViS 预算不受影响。GPU 或依赖不可用时记录 unverified，不将 CPU 数据替换进去。

## 数据与请求流

VBench 默认固定在官方 v0.1.5 commit `1039fe66d7aa2f4b214003ca2d35f5011c48bfee`，保留全部 946 条官方 metadata。`split=evaluation` 保留质量评测用途，`service_split` 只用于本套件的开发与服务测试划分。若用其中开发样本调过算法，不能再把完整 VBench 称为全部未见过的测试数据。

VidProM 导入仅获取 `VidProM_unique.csv`。先解析官方 revision，再下载不可变版本；遍历 CSV 并用固定哈希优先级抽取最多 4096 条，默认其中 32 条作开发集，余下作服务测试。`--sample-count`、`--dev-count` 可调整，输出记录实际扫描与筛选数量、版本和文件哈希。下载和缓存命中均核对官方文件大小与内容哈希，网络截断不能成为成功快照。支持 `--source` 导入本地官方 JSON/CSV；不带 `--download` 时不会联网，使用者负责确认其来源。默认保留 prompt 原文和空白。

```bash
python3 -m ditserve benchmark prepare --dataset vidprom --download --output .cache/benchmark-data/vidprom.json
```

每个请求保存内容来源、prompt、模型 revision、分辨率、帧数、步数、生成 seed、固定缓存配置、计划到达和绝对 deadline。到达时间与其余随机流独立；改变速率不改变内容。trace 同时导出 `requests.jsonl` 和兼容旧 C++ v2 的 `core-trace.json`。

| 模式 | 到达语义 |
| --- | --- |
| poisson | 指数间隔；首个间隔也采样 |
| gamma | shape=1/CV²，scale=1000×CV²/rate_rps，单位毫秒 |
| foreground_burst | 周期4/rate秒，开始时一个长请求，周期1/4处三个短请求；末组按请求数截断 |
| offline_all | 全部在0时刻到达，仅作诊断 |
| replay | 稳定排序并回放来源时间戳，可显式缩放时间 |

VidProM 的原生时间戳没有时区，回放记录所用时区假设；这是经过过滤和去重的社区数据回放，不是完整生产请求日志。重放文件也可直接给出 `arrival_ms` 和 `sample_id`。

默认 SLO 为对应规格独立服务时间的2.5倍。相对负载按请求混合的参考服务时间换算，不称为实测饱和容量。成本、SLO 与全部请求在比较策略之前冻结，调度器只能看到已到达请求。

`smoke` 使用256×256、9/17帧、32步、每trace6请求；以参考负载0.7分别测试Poisson、Gamma(CV=2)、长短突发。`formal` 使用480×832、17/49/81帧、50步、300请求、5个seed；Poisson参考负载扫描0.25/0.5/0.75/1/1.25，Gamma在固定参考负载1下扫描CV=2/4。配置中将负载扫描与CV扫描分开，不自动形成昂贵的全组合。两档主比较默认关闭近似缓存。

## 后端与比较边界

- `cpu-test`：确定性小数组与虚拟时钟，用于验证调度和接口，不是 Wan 模拟精度或 GPU 性能证据。
- `core`：既有 C++ 合成参考执行器。使用 `profile --backend core` 和 `run --backend core --policies core:naive,core:edf,core:state_aware`；不把 core 策略别名当成真实 FCFS/SRTF。其参考状态 parity 不能替代视频质量。
- `wan`：真实逐步生成，接入 FCFS、EDF、SRTF；同优先级保持当前请求。仅在安全边界切换，优先保留 GPU 状态，空间不足时完整 hot 卸载；主比较不执行 cold 丢缓存或改阈值。

PreViS 自适应实验依赖其专有校准与策略语义。通用接口对尚未适配的自适应策略明确返回 unsupported，不默默运行 EDF；历史实验仍由原入口执行。

文本编码、等待、调度、去噪、传输、解码计入端到端时间。真实到达由独立线程注入，保留实际入队时间与发送延迟。模型加载/预热在服务计时之外，但仍计入 GPU 预算。输出保存采用有界队列，当前请求结束点为解码完成；保存排队可能延迟后续请求，这些影响保留在实测结果中。

## 质量、统计与证据

每条请求建立同输入、seed、配置的独立执行参考。恢复检查采用 latent/output `rtol=1e-5, atol=1e-6`，并比较执行分支历史。参考一致性不是绝对画质评分。近似缓存另用显式无缓存参考检查 LPIPS，默认均值门槛0.05；缺少权重或参考时质量为unknown。

正式 VBench 使用独立生成清单，每prompt通常5个seed，Temporal Flickering为25个。可用 `evaluate --vbench-info <官方metadata> --output <manifest.json>` 导出清单；仅支持已固定版本的官方评测 checkout，不能将小规模服务trace冒充完整VBench评分。官方依赖未安装或样本不完整时保留not_run/incomplete，不生成替代分数。

以下为正式质量实验的接口示例，首轮验收不执行该全量生成。先准备与 `formal` 对应的真实 profile，再运行：

```bash
python3 -m ditserve benchmark evaluate --vbench-info /path/to/VBench_full_info.json --config formal --shape long --output /path/to/vbench-generation.json
python3 -m ditserve benchmark trace --config formal --dataset /path/to/vbench.json --profile /path/to/formal-profile/profile.json --vbench-manifest /path/to/vbench-generation.json --output /path/to/quality-trace
.venv-previs/bin/python -m ditserve benchmark run --trace /path/to/quality-trace/trace.json --backend wan --policies fcfs --output /path/to/quality-run
python3 -m ditserve benchmark evaluate --export-videos --run /path/to/quality-run --vbench-manifest /path/to/vbench-generation.json --videos-dir /path/to/videos
.venv-previs/bin/python -m ditserve benchmark evaluate --vbench-manifest /path/to/vbench-generation.json --videos-dir /path/to/videos --repo /path/to/pinned-VBench --output /path/to/vbench-score
```

质量清单默认冻结81帧、480×832、50步和16 FPS。相同prompt/seed跨维度只生成一次，导出时复用；完整清单、源文件哈希与采样协议均需匹配。独立质量trace全部在0时刻到达，允许包含服务开发集提示词，超时从首次执行计算，其时间只作诊断、不进入在线性能提升比较。模拟后端输出不能导入官方评分。导出需要本地ffmpeg，评测需要固定版本官方checkout与已安装的依赖、权重；不隐式下载评测权重。

固定版官方标准加载器每prompt读取5段；Temporal Flickering的25段用同版官方`custom_input`入口逐项评分。结果明确标记此调用差异，仅报告官方维度结果，不拼出声称可对比排行榜的总分。GPU评分所有维度共享剩余预算的总截止时间。

报告区分完成吞吐、按时完成goodput、参考质量通过且按时完成goodput；失败、拒绝和超时保留在已到达分母，尚未到达请求单独列出。延迟分位数只针对完成请求并报告样本数。少量请求的P99只作描述性统计。不完整策略配对不进入提升表。

每次运行保留 sealed manifest、trace/profile哈希、源码哈希、环境、逐请求状态、输出哈希和质量overlay。`report` 重建JSON、CSV、Excel、中文说明与SVG图；不删除原始数据。GPU占用墙钟与各阶段耗时单列，不把它们声称为精确kernel利用率。

真实 profile 绑定物理 GPU UUID、驱动与精度/attention配置，不能把A100标定用于V100。普通中断会保存终态；强制杀进程无法保证最后一个阶段落盘，账本会将未关闭会话保守计费。重新执行必须使用新attempt。报告包含按哈希复制的数组证据，原路径失效后仍可从`report/evidence/run.json`重新评价、重建报告。

实现接口位于 `benchmarks/ditserve/benchmark`：`datasets`负责来源与划分，`workload`负责请求流，`protocols`负责官方质量清单，`backends`负责安全阶段执行，`runner`负责调度，`evaluation`和`reporting`负责独立评价及导出。扩展后端需明确是否为实测，并实现独立状态、阶段切换、完整卸载恢复与执行身份记录。

官方来源：[VBench固定版代码](https://github.com/Vchitect/VBench/tree/1039fe66d7aa2f4b214003ca2d35f5011c48bfee)、[VidProM提示词数据](https://huggingface.co/datasets/WenhaoWang/VidProM)、[Wan2.1固定模型](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/0fad780a534b6463e45facd96134c9f345acfa5b)。

## CPU 回归

```bash
PYTHONPATH=benchmarks:.cache/tools python3 -m unittest discover -s tests/python -p 'test_benchmark_*.py' -v
```

正式 GPU 验收顺序是独立生成与暂停恢复一致性、开发集profile、三种请求流下的三策略对照、离线质量与报告。任一正确性关口失败或预算不足，保留该attempt和失败原因；不调整输入后覆写原结果。
