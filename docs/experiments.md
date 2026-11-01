# 实验配置

正式入口为 `.cache/build/ditserve run --config <study.json> --run-id <id>`，配置 schema 为 `ditserve.study.v1`。`configs/studies/mock.json` 是 CPU 样例；`configs/studies/{profile,online,state,sp}.json` 是通用模板。原生入口自行设置 worker 包路径，直接使用 Python API 时才需要 `PYTHONPATH=workers`。

jjjkairos 功能 preflight 不再使用静态历史配置。`benchmarks/cluster_config.py` 在作业内读取 `nvidia-smi` 的 GPU UUID，并生成 Wan2.1/Wan2.2 + SD3.5 的当前模型路径、worker、warmup 和小负载：

```bash
python3 /opt/ditserve/share/ditserve/benchmarks/cluster_config.py \
  --model wan21 --output /shared/output/DITSERVE/acceptance/<run>/preflight.json
ditserve run --config <preflight.json> --run-id <run-id> --allow-gpu
```

该配置仅验证功能，`profile.provenance=synthetic`，计划和 SLO 数字不是性能测量。正式实验必须由当前运行校准成本画像，并保留配置、BUILD_INFO、镜像 digest、模型 revision、拓扑和逐请求执行记录。

## 执行器与矩阵

| kind | 用途 |
| --- | --- |
| `profile` | 顺序或配对测量 |
| `online` | 单/多 worker 在线回放、预编码和异步解码 |
| `state` | 驻留、快照与迁移 |
| `sp` | 显式多 rank 动作 |
| `session` | 通用多阶段 replay 和请求依赖 |
| `cluster` | 原生多 GPU 混合图像/视频执行，正式 serving 入口 |

矩阵键是相对 phase 的 JSON pointer，生成笛卡尔积。`worker_config` 可修改模型配置，但不能改变 worker 设备或身份。每个 cell 独立创建模型进程，配对观测有独立 observation ID；失败观测保留在分母中。

worker 以完整 GPU UUID 绑定设备，模型使用 `cuda:0`。显式 `actions` 可执行 prepare/advance/finish/release/evaluate/distributed 等操作，state ID 必须稳定。C++ 控制预算、矩阵、进程监督和播放时钟，Python 保留模型、张量、RNG、质量和报告。

## 当前模型边界

- Wan2.1-T2V-1.3B 与 Wan2.2-TI2V-5B：`wan` 后端，xDiT/xFuser 固定 revision，DiT FP16、UMT5 CPU FP32，支持 step/block、CFG/USP、continuation 和显式 CFG fold。VAE dtype 与 attention 必须写入证据；BF16 VAE 不是 FP32 逐位一致。
- SD3.5-Medium：`sd35` 后端，BF16、Flash、固定离线 batch 和单 CPU Generator；是原子图像请求，不支持步骤迁移或连续批。
- Sana、FLUX、Mistral、LongLive 专用后端已删除。当前论文不使用其历史数据。

固定运行环境为 torch 2.10.0+cu126、diffusers 0.36.0、transformers 4.57.6、cache-dit 1.5.0 和 xDiT `16c34e7f070de65d9d2c67a6d74e0007dd8b426a`。CPU/Gloo 契约不等于 CUDA/NCCL 资格。

## 工作负载与输出

可使用冻结 `workload` 或确定性 `generator`（offline_all、steady、Poisson、bursty、Gamma）。`evaluation/workloads.py` 提供公开 prompt pool 的冻结混合 trace，manifest 记录来源和哈希；它不是生产日志。

`output_mode` 为 `file`、`digest` 或 `check`。无下游质量评测的 serving 实验使用 `check`，质量子集显式保留输出并登记依赖。warmup 在到达时钟前完成，模型加载和首次 kernel 不计入首批请求；所有注册 GPU 从阶段开始到清理确认均计费。

## 原生 cluster 策略

| 策略 | 实际行为 |
| --- | --- |
| `static_sp1/2/3/4/6/8` | 固定 SP，整步执行，开始后留在原组 |
| `static_cfg2`、`static_cfg2sp2/3/4` | 固定 CFG/SP 形态 |
| `step_elastic` | step 边界弹性放置，GenServe-like，不是完整 GenServe 复现 |
| `round_elastic` | 固定 round 边界弹性放置，TetriServe-like |
| `block_elastic` | block quantum、lazy preemption 和 step 边界换组 |
| `heterodit_exact` | block、knapsack、可选 tiered state，不使用特征缓存或 CFG fold |
| `heterodit_approx` / `heterodit_v3` | 独立的 whole-step 特征缓存研究路径，真实后端限 SP1 pinned |
| `heterodit_v5` | 独立 Approx：block、worker 驱动 CFG fold、形态收缩和真实异步 decode |

`heterodit_v5` 不默认叠加 Taylor、DBCache 或 INT8 Ring。Exact 与 Approx 结果必须分表。固定步数折叠与曲率门控都由 worker 的 `options.cfg_fold` 决定；只有 worker 在完整 CFG 步后返回 `cfg_fold_eligible=true`，规划器才可折叠。`max_folded_steps` 是每请求的独立计算预算，必须进入 continuation 的配置校验；曲率臂要与固定折叠臂报告实际折叠步数，不得把不同计算量直接称为算法收益。折叠状态随 continuation 导出/导入，最终 exact anchor 由配置决定。

并行形态支持 Wan 合法头数对应的 1/2/3/4/6/8 卡 SP 度及 CFG 组合，例如 `sp3`、`cfg2sp3`；`cfg2u3r1` 规范化为 `cfg2sp3`。组按拓扑顺序对齐，8 卡可实现 6+2→3+5。每次 assignment 和 worker `applied` 必须一致，否则运行失败。

`options.sp_transport`：`pinned` 限真实 SP1；`resume` 仅供 mock；`export` 在 2/4/8 卡进程组间通过 host-staged、带哈希的 step-boundary 状态换组。SP/CFG 的 mid-step packet 不跨组，SP1 可在 block 边界导出。文本预编码 ticket 避免换组时重复编码。

`options.async_rank0_decode=true` 使用 `decode_start` 启动后台 VAE，再轮询 `decode_poll` 并结算。VAE 固定使用 MATH；decode 期间该 GPU 不接受新视频，避免 DiT Flash 与 VAE MATH 的全局设置竞争，但独立图像 worker 仍可重叠执行。它不是同步 decode 的折扣模型，也不再同时做 chunked decode。

其他选项包括 `image_batch`、锁步 `pairing`、`reserve_image_gpus`、`quantum_blocks/quantum_ms`、`order`、feasible-first EDF 和 work-neutral widening。基线名称只描述实际实现，不能把 step/round 基线写成完整外部系统。

## 成本画像与模拟

`profile.provenance` 为 measured、extrapolated、synthetic 或 mixed。视频类包含 steps、blocks_per_step、prepare/encode/finish、SP1 compute、通信量、collectives 和 CFG exchange；links 按度数和放置校准 bandwidth/latency/contention/overlap。模拟器与真实执行共用 `cluster_plan`，但模拟输出只代表成本模型。

What-If 的 CFG fold 起点必须显式写 `options.cfg_fold_eligible_from_step`，`cfg_fold_step_scale` 和 `async_decode_scale` 必须显式给出来源明确的标定值；缺失或越界会失败，不再使用隐藏的固定折扣。校准集与验证集必须分离，不允许从评测行反写带宽或 overlap 后宣称零重拟合。

## 数值口径与证据

“Exact”只相对同一优化 runtime 的不中断执行，绑定模型、dtype、attention、sampler、seed 和 decoder。CFG parallel 的逐位一致需单独验证；token-sharded SP 不默认逐位一致。BF16 VAE、INT8 通信、特征缓存和 CFG fold 均属单独数值模式。

最终交付为 `results/<experiment>/<run>/results.xlsx` 与 `REPORT.md`。`completed` 仅表示技术执行完成，性能、数值和语义资格分别判定。预算未结算、worker 未清理或依赖未闭合时不得归档为完成。重试使用新 run ID，保留失败现场。

## worker 协议

stdin/stdout 使用带版本的 JSONL，错误为 `ok:false`。模型和张量留在 Python，C++ 只接收状态 ID、进度和有来源的文件元数据。stdout 专用于协议，日志进入 stderr；异常、卡死和取消由所属进程组监督并清理。
