# 单张 V100 的 DiT／VAE 阶段冲突实验

此实验测量两个真实 Wan 请求的阶段共驻与执行干扰，不实现在线队列，也不报告 goodput。模型、权重 revision、精度和 attention 沿用 `benchmarks/configs/v100.json`：物理 GPU 1 的固定 UUID，进程内 `cuda:0`，FP16 DiT、FP32 VAE、CPU FP32 文本编码器、native attention 和 SDPA MATH。

## 执行与预算

模型快照仍保存在本地缓存。实际执行需要另行准备 GPU Python 环境；下列 `.venv-v100` 指运行该协议的环境，当前目录整理不会安装 GPU 依赖。在项目根目录执行：

```bash
.venv-v100/bin/python scripts/run_phase_conflict.py --dry-run
.venv-v100/bin/python scripts/run_phase_conflict.py --output .cache/runs/phase-conflict/UNIQUE-RUN-ID
```

如需优先进行下文的归因补测，首次启动加 `--defer-cache`。它完成全部无缓存核心实验后退出，再按共享剩余预算运行诊断。默认入口保留条件触发固定缓存复核的行为。

默认上限为 6 GPU 小时。采用保守记账：每次子进程从启动到退出的完整墙钟均计入，包含 CPU 模型加载、预热、失败与输出保存。父进程启动前估算成组开销；运行中也受剩余总预算和单 worker 超时限制。GPU 身份、空闲显存及使用率在每个 worker 启动前重新检查，不使用其他 GPU。

实验执行和补测期间保留失败记录与原始输出。修复实现错误并审查记录后，可使用相同输出目录恢复：

```bash
.venv-v100/bin/python scripts/run_phase_conflict.py --output .cache/runs/phase-conflict/UNIQUE-RUN-ID --resume --retry-failed
```

恢复不会重置预算或覆盖旧尝试。已有运行中记录、缺失预算账目、改变模型／协议或不匹配的原始输出将阻止恢复。OOM 作为配置结果保留，不自动改小该策略输入。超时或异常会结束独立 worker 后再返回；后台 GPU 工作无法通过 `Future.cancel()` 撤销。

独立归因账本中一旦存在运行中或已耗时尝试，原入口的 `--resume` 会拒绝继续，避免遗漏该部分费用；诊断入口会同时核算两份账本。不要重置账本以恢复实验。

## 工作负载与四个策略

B 使用固定 `water` 提示词，先真实生成最终 latent，单独记录准备成本及输出参考。A 使用 `object` 提示词。每个案例从同一 B-ready 状态开始，主计时包含 A 的 CPU 编码，排除 B 的历史生成与重新载入成本，因此称为“剩余完成时间”。

| 策略 | B 解码顺序 |
| --- | --- |
| immediate | 在 A 开始前完成 |
| deferred | 等 A 完成并清理已结束的去噪状态后执行 |
| blocking | 在 A 的指定步后解码，A 等待 B 完成再继续 |
| overlap | 在 A 的指定步后启动解码，A 继续去噪 |

四策略使用同样的预创建 VAE 工作线程和 CUDA stream。B 仅调用 VAE；A 是唯一 DiT 请求。B stream 等待输入与指定步的就绪事件；A 自己进入 VAE 前等待 B 完成。每个策略只持一份模型，不使用人工显存上限或填充张量。已结束的 scheduler／cache 状态及时清理，活动 A 的状态不清理。

先用 256×256、17 帧、12 步验证正确性。随后在 480×832 下用 4 个去噪步和完整解码筛查 17、33、49 帧；33 帧 OOM 时补25帧并停止扩大。4步输出仅用于内存探测，不能用于质量结论。选择最大能独立完成50步的规格后，先测seed0四策略及逆序关键配对，再按预算测seed1、提前插入和混合帧数。正式 CFG=5、batch=1，主要插入点为完成第25步后，补充点为第10步后。

固定缓存只在无缓存主实验出现可重复冲突／干扰信号且仍有预算时复核，参数不按结果调整。

## 结果解释

每个案例保存 job、源码指纹与副本、环境、原始日志、输出数组／视频、CUDA事件和allocator峰值。`manifest.json` 记录全部尝试与预算；汇总入口为：

```bash
.venv-v100/bin/python scripts/summarize_phase_conflict.py .cache/runs/phase-conflict/UNIQUE-RUN-ID
```

追加 `--export-media` 可校验保存的数组并导出视频。默认汇总不重新读取所有大数组；运行记录中的哈希核对与对实际数组重新计算哈希是不同的检查。

## 独立归因补测

核心协议结束后，可从已有案例复刻内核时间线；以下命令中的目录和案例编号应替换为实际记录：

```bash
.venv-v100/bin/python scripts/trace_phase_conflict.py \
  --suite .cache/runs/phase-conflict/UNIQUE-RUN-ID \
  --case-id same-f33-b33-s0-overlap-k25-r0-none \
  --output .cache/runs/phase-conflict/UNIQUE-RUN-ID/review/kernel-trace
```

该补测保持已有模型、输入、预热和双 stream 机制，在正式请求上启用 Kineto。它通过 CPU 请求标记与 CUDA launch correlation 关联实际 GPU kernel。CUDA事件区间和 CPU launch 区间不能替代内核活动区间。OOM 案例必须具有相同输入的成功串行参考及 B provenance；失败后的部分时间线只能提供正向并发证据，不能用缺少见证证明没有并发。诊断耗时不参与核心策略的性能比较。

独立测试分配器敏感性时，使用另一个新输出目录，并增加：

```bash
--no-profiler --allocator-config 'backend:native,expandable_segments:True'
```

此模式只修改新子进程的分配器配置，关闭 Kineto，保留原有 CUDA事件记录。末尾的只读 allocator 快照用于检查是否实际出现 expandable segment；配置请求、实际段使用、段增长和 OOM 成因不能视为同一件事。即使成功，也应检查输出与区间交叠，不能直接宣布获得调度收益。

两个入口共用同一 GPU 锁。追加补测使用 `trace-attempts.json` 独立记账，六小时上限扣除核心与所有补测的累计耗时，失败也计入。`--dry-run` 只读检查来源与剩余预算；核心仍运行、来源不匹配或估计费用超过剩余预算时，不启动 GPU 子进程。2 GiB 磁盘余量只是启动最低检查，不是时间线大小上界。

## 一致性与研究边界

输出哈希比较针对同缓存模式、同输入的执行顺序改变。它验证系统执行一致性，不证明缓存与无缓存的感知质量等价。无观测和诊断运行若不同，应先调查观察器，而非接受不一致作为加速代价。

交叠运行的 allocator 峰值是联合值，不能通过两个独立峰值相加或相减推导缓存大小。CUDA事件区间交叠不单独证明内核同时运行。回调调用栈可能保留本步临时张量，必须和必要的 latent、scheduler、缓存状态分开归因。

继续研究需要可重复的自然压力／执行干扰，以及不能由同一个简单固定顺序处理的工作负载差异。没有自然冲突、单请求自身OOM、所有规格都由简单串行规则充分处理，均不支持直接开发复杂调度器。V100 MATH attention 上的现象需在现代 GPU／attention 后端复核后再推广。

## 最终归档

历史结果按协议步骤整理在 [阶段冲突结果](../../results/INDEX.md)。原始数组、B-ready latent 和完整 trace 已在预览与归档校验通过后删除，记录中的像素哈希和源码指纹仅表示历史证据。新实验在 `.cache/runs` 内完成所有补测后再归档；归档报告可以重建，但不能恢复原始数组或作为 `--resume`／归因补测的输入。失败、OOM、预算、源码快照和内核审计摘要保留在各步骤的 `data.json.gz` 中。

全部补测结束后，运行 `python3 scripts/archive_results.py finalize .cache/runs/phase-conflict/UNIQUE-RUN-ID --series phase-conflict --step UNIQUE-RUN-ID --remove-raw` 完成验证后清理。需要继续重试或补测时省略 `--remove-raw`；归档失败不会清理原文件。
