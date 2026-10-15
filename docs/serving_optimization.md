# 视频 serving 优化与配对验收

本轮先完善迁移启动检查，再接入高效单卡 attention、CPU 编码流水线和配对在线验证。真实动态 batch、异步状态搬运与多卡执行尚未实现，只有实测瓶颈支持时才投入下一阶段。

## 执行配置

`WanStepWorker` 的 `attention` 支持 `math`、`flash`、`efficient`，默认 `math`。选择只作用于 DiT，VAE 保持 FP32/MATH。仅启用指定 SDPA 内核，不静默回退；独立官方循环、逐步执行和 hot 恢复使用同一配置。profile 绑定物理 GPU、版本、精度及 attention，不能跨后端复用成本。

`benchmark run --text-prefetch 2` 启用一个 CPU 编码线程，最多保留两个运行中、排队或完成待消费的编码请求。只有已到达请求可进入队列，payload 绑定请求、worker 且只消费一次。编码完成后 GPU 安装、去噪、解码仍由单个调度线程执行；不复用重复提示词的 embeddings。SRTF 对已编码请求扣除已完成的 CPU 编码成本。

E2E 从计划到达到解码完成，编码排队和 GPU 等待都计入。`text_encode` 为 CPU 阶段，可与 GPU 阶段重叠，不能将各阶段耗时相加当作 E2E。取消时等待正在执行的 CPU 编码结束后才释放模型；清理计入 GPU 账本，超出在线时间窗的事件另标记。

## 校验与小规模实测

GPU 环境使用 `PYTHONPATH=python`，不要混入为系统 Python 构建的 `.cache/tools`。固定线程数、离线模型和本地 LPIPS 权重路径，并记录环境。新实验输出使用新目录。

```bash
export PYTHONPATH=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
# 本机历史权重只读复用；在其他机器上替换为已有的离线目录。
export TORCH_HOME=/home/zyd/code/dit/.cache/torch

.venv-previs/bin/python -m ditserve benchmark profile --backend wan --device cuda:0 --attention math --config optimization --dataset /path/to/frozen-dataset.json --output results/serving-optimization/math-profile --budget-path /path/to/authoritative-gpu-budget.json
.venv-previs/bin/python -m ditserve benchmark profile --backend wan --device cuda:0 --attention flash --config optimization --dataset /path/to/frozen-dataset.json --output results/serving-optimization/flash-profile --budget-path /path/to/authoritative-gpu-budget.json

python3 -m ditserve benchmark optimize-plan --dataset /path/to/frozen-dataset.json --math-profile results/serving-optimization/math-profile/profile.json --flash-profile results/serving-optimization/flash-profile/profile.json --output results/serving-optimization/pilot-plan
```

默认 pilot 是 480×832、17帧、32步、第一条 seed、near 场景、三个请求。完整计划 `--stage expanded` 另列三 seeds × under/near/over/burst，生成计划不会启动 GPU。Flash 串行的独立 profile 同时用于流水线变体，计划显式记录别名；每组请求、生成 seed、到达和 deadline 完全一致。负载和 SLO 由冻结 Flash 参考标定，调度剩余时间与容量则读取当前变体的执行 profile。

依次运行计划中的 `math_serial`、`flash_serial`、`flash_prefetch` trace，参数必须匹配冻结变体。三组分别使用 `--attention math --text-prefetch 0`、`--attention flash --text-prefetch 0` 和 `--attention flash --text-prefetch 2`。Flash 两组增加 `--quality-reference /path/to/math-run`；保存独立参考后，先和 MATH 的同输入原始输出计算整段平均 LPIPS，默认门槛 0.05，任一样本失败则不开始在线执行。该门槛仅约束参考差异，不代表绝对视频画质。

每组执行完成后运行 `benchmark evaluate --run /path/to/run` 验证本后端逐请求独立参考一致性，再用 `benchmark report` 导出逐请求报告。三组完成后：

```bash
python3 -m ditserve benchmark optimize-report --runs /path/to/math-run /path/to/flash-run /path/to/prefetch-run --output /path/to/ablation-summary.json
```

汇总核对冻结输入、执行参数、设备身份、完整请求覆盖、原始数组哈希、参考一致性与跨 attention 质量。失败、超时、未测与不完整配对保留；不能将缺测填零或归为性能提升。

## 预算与下一阶段门槛

沿用唯一的 benchmark 累计账本，上限仍为 7200 GPU 秒，加载、预热、独立参考、失败、持卡质量评估与清理均计费。`--budget-path` 指向原权威文件；不复制原账本来获得新的余额。同级旧仓库有历史账本时默认使用它；若当前和旧目录同时存在独立账本，必须明确选择并核对累计费用。该权威账本及配套锁、原子写入临时文件是旧仓库历史只读规则的唯一写入例外，历史代码和归档仍保持只读。

`optimize-report` 记录真实 CPU 工作、step/decode、抢占、换出与最大 GPU 就绪队列。多个请求就绪且去噪占主导时，仅放行 batch=2 可行性测量，还需要实际内存、吞吐和逐请求质量证据。状态调度先要求真实换出或容量限制，随后测切换成本；当前不自动开启 cold 或改变缓存阈值。小 pilot 不建立稳定 P99 或三 seeds 的可重复收益。

## 2026-10-12 小规模在线验收

设备为单张 A100 PCIe 40GB，Wan 1.3B，480×832、17 帧、32 步，关闭缓存，FCFS。两个场景各三个请求，复用同三个测试提示词及生成种子；仅一个 trace seed。所有变体共享请求、到达时间和约 94.47 秒的相对 deadline，该 deadline 由 Flash 独立 profile 标定。

| 场景 | MATH 串行平均 E2E | Flash 串行平均 E2E | Flash + CPU 流水线平均 E2E | 按时完成数（依次） |
| --- | ---: | ---: | ---: | --- |
| 普通到达（near） | 221.16 秒 | 61.88 秒 | 45.97 秒 | 0/3、3/3、3/3 |
| 突发到达（burst） | 235.86 秒 | 75.94 秒 | 60.60 秒 | 0/3、2/3、3/3 |

流水线相对 Flash 串行的平均 E2E 分别减少 25.71% 和 20.20%。普通场景中实际观测到约 37.14 秒的 CPU/GPU 重叠；CPU 编码总工作量仍为 56.30 秒，不能把重叠解释为编码免费。独立单请求 profile 从 MATH 的 110.68 秒降到 Flash 的 37.79 秒，其中去噪从 93.68 秒降到 18.57 秒。

18 个在线输出和 latent 均与各自独立参考完全一致。三个独特视频的 Flash/MATH 平均 LPIPS 为 0.01057、0.00115、0.01407，均通过预设 0.05 门槛；这是参考差异验收，不是完整 VBench 或绝对画质结论。其余十个 seed/场景组合未执行，三请求样本不支持稳定 P99 或生产容量结论。

原始结果位于 `results/serving-optimization/`，配对计划为 `paired-plan.json`，汇总为 `ablation-summary.json`。`source-freeze-final.json` 与 `sources-final/` 保存被测源码；MATH profile 后的非计时逻辑修复由 `source-continuity.json` 记录。第一次 Flash profile 与本地 CPU 测试重叠，已在 `flash-profile-timing-exclusion.json` 中排除并完整计费；所有计划均使用重新测量的 `flash-profile-clean/profile.json`。

固定 batch=2 的两轮反序实验也已完成：两个单请求的 GPU 阶段合计约 40.1 秒，batch=2 约 39.3 秒，吞吐仅提高 1.86%–2.10%；计入重新编码后，两轮总服务时间均略长。峰值 allocated 显存从约 11.07 GiB 增至 18.98 GiB。两轮共八项视频质量检查全部通过，分别直接对照同轮 Flash 单请求和原 MATH 参考；每个请求的初始 latent、编码和随机数状态一致。详情见 `batch2/batch2-result.json`，这是两个请求的离线可行性证据，不是动态 batching 实现或在线收益。

据此保留 Flash 与有界编码流水线，暂缓动态 batch 和复杂状态策略。优先测文本编码加速的时间、显存与质量代价，再扩展长视频、混合长度和更多种子的在线验证。本轮无状态换出或抢占，不能外推其他负载的状态调度需求。

GPU 实验结束后只额外修复了编码线程清理被 `KeyboardInterrupt` 中断的边界：必须等实际 Future 完成后才传播中断。`source-after-validation.json` 记录这一变化；正常编码/调度方法未改，原性能快照保留，未重新运行 GPU。最终 459 项 Python 测试、C++ 合约检查及全部阶段的 CPU preflight 通过。
