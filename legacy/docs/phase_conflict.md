# 单张 V100 的 DiT／VAE 阶段冲突实验

此实验测量两个真实 Wan 请求的阶段共驻与执行干扰，不实现在线队列，也不报告 goodput。模型、权重 revision、精度和 attention 沿用 `configs/v100.json`：物理 GPU 1 的固定 UUID，进程内 `cuda:0`，FP16 DiT、FP32 VAE、CPU FP32 文本编码器、native attention 和 SDPA MATH。

## 执行与预算

模型快照仍保存在本地缓存。实际执行需要另行准备 GPU Python 环境；下列 `.venv-v100` 指运行该协议的环境，当前目录整理不会安装 GPU 依赖。在项目根目录先设置 `export PYTHONPATH=python`，再执行：

```bash
.venv-v100/bin/python -m ditserve.experiments.run_phase_conflict --dry-run
.venv-v100/bin/python -m ditserve.experiments.run_phase_conflict --output .cache/runs/phase-conflict/UNIQUE-RUN-ID
```

`--defer-cache` 在完成无缓存核心实验后退出；默认入口保留条件触发固定缓存复核的行为。当前版本未迁移历史独立归因补测入口。

默认上限为 6 GPU 小时。采用保守记账：每次子进程从启动到退出的完整墙钟均计入，包含 CPU 模型加载、预热、失败与输出保存。父进程启动前估算成组开销；运行中也受剩余总预算和单 worker 超时限制。GPU 身份、空闲显存及使用率在每个 worker 启动前重新检查，不使用其他 GPU。

实验执行和补测期间保留失败记录与原始输出。修复实现错误并审查记录后，可使用相同输出目录恢复：

```bash
.venv-v100/bin/python -m ditserve.experiments.run_phase_conflict --output .cache/runs/phase-conflict/UNIQUE-RUN-ID --resume --retry-failed
```

恢复不会重置预算或覆盖旧尝试。已有运行中记录、缺失预算账目、改变模型／协议或不匹配的原始输出将阻止恢复。OOM 作为配置结果保留，不自动改小该策略输入。超时或异常会结束独立 worker 后再返回；后台 GPU 工作无法通过 `Future.cancel()` 撤销。

如果迁入的 suite 包含已经收费的历史归因账本，`--resume` 会拒绝继续，避免遗漏费用；不要重置账本以恢复实验。

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
.venv-v100/bin/python -m ditserve.experiments.summarize_phase_conflict .cache/runs/phase-conflict/UNIQUE-RUN-ID
```

追加 `--export-media` 可校验保存的数组并导出视频。默认汇总不重新读取所有大数组；运行记录中的哈希核对与对实际数组重新计算哈希是不同的检查。

## 历史归因证据

旧仓库曾用独立 Kineto 时间线与 allocator 补测定位干扰；当前重写没有迁移该诊断入口，不提供执行命令。历史摘要仍可用于了解已测现象，但不能充当当前源码的内核并发验证。CUDA 事件区间或 CPU launch 区间也不能单独替代内核活动区间。

## 一致性与研究边界

输出哈希比较针对同缓存模式、同输入的执行顺序改变。它验证系统执行一致性，不证明缓存与无缓存的感知质量等价。无观测和诊断运行若不同，应先调查观察器，而非接受不一致作为加速代价。

交叠运行的 allocator 峰值是联合值，不能通过两个独立峰值相加或相减推导缓存大小。CUDA事件区间交叠不单独证明内核同时运行。回调调用栈可能保留本步临时张量，必须和必要的 latent、scheduler、缓存状态分开归因。

继续研究需要可重复的自然压力／执行干扰，以及不能由同一个简单固定顺序处理的工作负载差异。没有自然冲突、单请求自身OOM、所有规格都由简单串行规则充分处理，均不支持直接开发复杂调度器。V100 MATH attention 上的现象需在现代 GPU／attention 后端复核后再推广。

## 最终归档

历史结果按协议步骤整理在 [阶段冲突结果](/home/zyd/code/dit/results/INDEX.md)。原始数组、B-ready latent 和完整 trace 已在预览与归档校验通过后删除，记录中的像素哈希和源码指纹仅表示历史证据。新实验在 `.cache/runs` 内完成所有补测后再归档；归档报告可以重建，但不能恢复原始数组或作为 `--resume`／归因补测的输入。失败、OOM、预算、源码快照和内核审计摘要保留在各步骤的 `data.json.gz` 中。

当前编排器和汇总模块会写入当前仓库的 `results/phase-conflict` 归档并保留原始文件。使用 `python3 -m ditserve archive verify --root results` 校验当前归档，或用 `python3 -m ditserve archive rebuild <当前归档步骤目录>` 重建报告；当前命令行不提供历史 `finalize --remove-raw` 清理接口，不要在旧仓库执行重建。
