# 实验配置

`configs/studies/mock.json` 是可直接运行的 CPU 样例；六个通用 GPU 模板需要填写真实环境和 GPU UUID。`wan_profile_corrected.json` 属于已有 Wan 补采流程，使用前先读其专项说明。所有新实验都使用 `ditserve.study.v1`，不再新增 next/followup/专项 runner。正式入口为 `.cache/build/ditserve run --config configs/studies/mock.json`，无需设置 `PYTHONPATH`；`--python` 可指定已有解释器，模型自身的 `workers.<name>.python` 仍优先使用配置值。

生产控制实现位于 `src/`，模型、评测和报告适配位于 `workers/ditserve/`，Python 包名仍为 `ditserve`。原生入口自动设置 worker 导入路径；直接使用 Python API 时设置 `PYTHONPATH=workers`。

```json
{
  "schema": "ditserve.study.v1",
  "experiment": "example",
  "workers": {"model": {"backend": "mock"}},
  "budget": {"limit_gpu_seconds": 600, "watchdog_seconds": 300,
             "repair_reserve_gpu_seconds": 60, "carry": "forward",
             "stage_limits": {"screen": 240, "confirm": 300}},
  "phases": [
    {"id": "screen", "kind": "online", "policy": "edf",
     "generator": {"count": 4, "seed": 7, "pattern": "steady",
                   "rate_rps": 2, "spec": {"steps": 3}},
     "matrix": {"/policy": ["fcfs", "edf", "srtf"]}, "paired": true},
    {"id": "confirm", "kind": "state", "requires": ["screen"],
     "generator": {"count": 2, "spec": {"steps": 3}},
     "transitions": [{"at_step": 1, "mode": "hot"}]}
  ]
}
```

## 执行器和矩阵

| kind | 主要字段和用途 | 模板 |
| --- | --- | --- |
| `profile` | 顺序/配对测量，`paired:true` 按正序、逆序执行相同冻结矩阵 | `profile.json` |
| `online` | 到达回放、策略、预编码、异步解码、优化筛选 | `online.json` |
| `state` | `resident_limit`、hot/cold 驻留、状态快照和迁移 | `state.json` |
| `sp` | 有限 `actions`，各 rank 同时执行相同操作 | `sp.json` |
| `streaming` | prompt 更新、FIFO/latest、0/1/2 chunk 缓冲、独立帧消费 | `streaming.json` |
| `session` | LLM–DiT stages、session ID、KV 继承、think time | `session.json` |
| `cluster` | 多 GPU 并发回放：`gpus` 按拓扑顺序列出每卡 `{video, image}` worker，`profile` 为成本画像，策略见下文 | `local-a100/` |

矩阵键是相对于 phase 的 JSON pointer，按键排序生成笛卡尔积，最多 10000 个 cell。例如 `/policy`、`/quantum`、`/worker_config/options/execution`。`worker_config` 合并到该 cell 指定 `worker`（默认 `model`）的配置；不能改变 worker 设备或身份。每个观测的模型进程持续复用，cell 间重新创建以隔离配置。配对时记录 cell、正/逆次序和独立 observation ID，失败观测保留在分母中。

`workers.<name>` 包含 `backend`、`python`、`gpu_uuid`、`config` 和可选环境变量；单个进程只绑定自己的完整 GPU UUID，模型使用 `cuda:0`。不能用环境变量覆盖 GPU 绑定、冻结源码或离线加载。SP 实现使用 2/4/8 个独立 rank 的对齐组（rank 按 PCIe/socket 拓扑顺序编号），Flash attention；四卡布局与原固定组相同。block worker 可加入 SP 组：step 在首个 block 前开启，停在 step 内部的每个 quantum 是组内共识安全点，求解器提交时关闭；组只在 step 边界按控制器下发的 `group`/`epoch` 切换。8 卡与 SP 内 block 执行未经 GPU 验证；缓存特征的跨 SP 迁移仍不支持。集体操作必须列出不同 worker，`actions[].per_rank` 可提供每个 rank 的补丁。

显式 `actions` 可用于任意 kind，动作包括 `prepare`、`advance`、`snapshot`、`restore`、`reference`、`finish`、`release`、`evaluate`、`distributed`。动作必须自己声明稳定 state ID，后续通过该 ID 引用；常规回放自动处理状态 ID。这样可以表达校准、独立基线、数值检查和 SP 流程，不增加 Python 编排层。

## 保留的模型限制

| 模型路径 | 现有限制 |
| --- | --- |
| Wan step/block/CFG | 固定 torch 2.10.0+cu126、Diffusers 0.36.0、cache-dit 1.5.0；DiT FP16、VAE FP32、UMT5 CPU FP32，整步求解器状态保持原布局。 |
| Sana step 与 Mistral | 原生步骤实现沿用 sm70 V100 门槛。Sana 为 1024²、FP32、SCM 2 步、CFG 4.5；Mistral 为 FP16、greedy KV、无量化/滑动窗口。 |
| Sana pipeline | Ampere 及更新 GPU、BF16，eager/transformer_compile/vae_compile/transformer_graph，固定 batch 1/2；原子调用，不提供步骤快照。 |
| FLUX | 官方本地管线，默认 BF16，设备须支持；可选固定 cache-dit。调用为完整管线。 |
| LongLive | A100 BF16、固定来源修订、本地完整权重、独立现有环境；480×832、16 fps、每 chunk 3 个 latent 帧、最多两驻留会话。 |

这些限制来自现有实现；本次 CPU 测试未扩展或重新授予 GPU 支持资格。

## 工作负载和调度边界

使用冻结 `workload` 数组，或确定性 `generator`。生成器支持 `offline_all`、`steady`、`poisson`、`bursty`、`gamma`，输入包括 seed、count、rate_rps、cv、prompts、spec/specs、service_ms、slo_ms；`classes` 给出按权重混合的请求类别（class、kind、worker、spec、service_ms、slo_ms 或 slo_scale），同一 seed 下不含 classes 的轨迹与之前一致。模型参数在 `spec`；到达、期限、成本估计是 C++ 调度元数据。

请求可以指定 `arrival_ms`、`deadline_ms`、`session_id`、`after`、`think_ms`、`stages`。`after` 必须引用负载中更早的一项；前一轮完成后才释放下一轮。stage 包含 worker、spec、估计 cost_ms 和可选 `prompt_from_previous:true`。Mistral 自动保存该 session 的对话与 KV，不共享其他 session 状态。

策略包括 `fcfs`、`edf`、`srtf`、`edf_b1`、`edf_b1_async`、`edf_b2`、`edf_b4`、`edf_step`、`guarded_b4`、`feasible_b4`、`guarded_adaptive`、`session_serial`、`stage_edf`、`session_slack`。Wan 可采用 step/block/paired_block 边界，CFG 和 cache 配方由 worker 验证；hot/cold 与跨设备快照只允许后端声明的合法边界。FLUX 保留完整管线调用边界，没有新增 token/block 抢占能力。FLUX和 `wan_pipeline` 可用 `spec.batch_size` 表达原有固定离线batch：重复同一prompt，整批使用一个CPU Generator(seed)。`wan_pipeline` 不提供步骤抢占、SP或迁移；启用cache后同一worker的steps/CFG配方固定。

`output_mode` 控制 finish 的输出处理：`file`（默认，写入工作目录供后续评测）、`digest`（只计算哈希）、`check`（只校验形状与 finite）。没有下游评测的 serving 实验应使用 `check`，避免把数十 MB 至数百 MB 的输出写盘计入 GPU 关键路径。`warmup` 在回放时钟开始前对列出的 worker 执行一次完整的 prepare/advance/finish/release（模型加载与首次 kernel 不计入首批到达）。`text_prefetch` 限定同时预编码的已到达请求数，`text_prefetch_workers` 限定只对这些 worker 的请求预编码（混合轨迹中只有 Wan 提供异步编码），`decode_mode:"async"` 启用后端支持的异步解码。Wan 异步 VAE 保持 MATH attention 的原限制。模型、加载、控制时间分别记录；通信采样排除进程首次启动、模型加载和 worker 操作耗时，仍包含 JSON 编解码。采样保留每请求前 128 次常驻动作及总计数，不声称是纯管道延迟。

流式 `updates` 按 arrival_ms 排序，版本递增。`prompt_policy` 为 `fifo`、`latest_only`、`latest_only_flush`；`buffer_chunks` 为 0/1/2。播放测量是 C++ 线程的墙钟帧分发，不是浏览器或物理显示时延。drain 超时保留失败及未完成请求，语义盲评保持独立。

FLUX 可设置 `workers.<name>.config.options.profile:true`，记录 text/DiT/scheduler/VAE/postprocess 的 CUDA event spans、逐步 finite 检查和 allocator 峰值。该测量有插桩开销，不能当作无插桩延迟或 kernel-only 时间。用 `/worker_config/options/profile:[false,true]` 表达对照；Wan 分解测量使用 `wan_profile_corrected.json` 的显式 actions。

## 多 GPU 调度与模拟

`cluster` 执行器与 `ditserve simulate` 共用 `control/cluster.cpp` 的 `cluster_plan`。GPU 按拓扑顺序编号，SP 组为对齐的 2 的幂（同 switch pair、同 socket quad、整机）。每个 quantum 结束后请求不占 GPU，重新规划：

| 策略 | 行为 |
| --- | --- |
| `static_sp1/2/4/8` | 固定 SP 度，整步 quantum，已开始的视频留在原组（近似 xDiT 固定并行） |
| `step_elastic` | EDF；按期限选最小可行 SP 度，无其他等待请求时占用最大空闲组；只在 step 边界换组（近似 GENSERVE/TetriServe 的 step 级弹性 SP） |
| `block_elastic` | 同上，但视频以 `options.quantum_blocks`（默认 4）个 block 推进，或按 `options.quantum_ms` 由当前类别与 SP 度换算 block 数；图像只有在等不到该视频 step 边界时才在 block 边界抢占（`lazy_preemption`，默认开启） |
| `static_cfg2`、`static_cfg2sp2`、`static_cfg2sp4` | 固定并行形态：组的两半各跑一个 CFG 分支，每步交换一次预测（与 SP1 逐位一致），半组内再做 SP；其余同 static_spK |

视频组的**并行形态**（`Form`）由 cfg ∈ {1, 2} 和 SP 度 sp 组成，占 cfg × sp 张卡，名字为 `sp1`、`sp2`、`sp4`、`sp8`、`cfg2`、`cfg2sp2`、`cfg2sp4`。弹性策略在 `options.forms`（默认为 `options.degrees` 对应的纯 SP 形态，即 `{1, 2, 4, 8}` → `sp1/sp2/sp4/sp8`，两者不能同时给出）中选择：组的大小按原来的最小可行度数规则决定，同一大小的形态取成本模型预测每步时间最小的。规划结果的每条 assignment 带 `form`，状态里停在 step 中途的请求带 `form`（缺省为 `sp<组大小>`），step 内不换形态。模拟器与执行器按形态计价并记录每个请求用过的 `forms`；`sp_groups_running` 只计 sp ≥ 2 的组（CFG 对每步只交换一个 latent，不争用 host bridge）。

`options.reserve_image_gpus` 给出静态分区基线：最后 R 张卡只服务图像。视频类别的 `decode_chunks` 把 VAE 解码拆成按 latent 帧推进的可抢占 chunk（`options.chunked_decode`，按 `quantum_ms` 换算每次推进的帧数）；真实 Wan worker 用 `finish` 的 `decode_frames` 执行，结果与整段解码逐位一致（CPU 契约测试）。Wan 的 `block_export`/`block_import` 可在 step 内任意 block 边界导出续算状态（未变动的求解器状态 + 运行中的 packet）并在另一进程恢复。

所有策略默认采用 feasible-first EDF（`options.demote_late`，默认开启）：即使占用最大组也赶不上期限的请求排在仍可按时完成的请求之后，避免过载时迟到的长视频饿死新到的短请求；剩余工作不超过 `options.late_short_ms`（默认 2000 ms）的迟到请求保留 EDF 位置，避免小的超时被放大成长尾。

迟到判断（feasible-first 降级）假设的 SP 度由 `options.feasibility_degree` 决定：
- `free`：弹性策略（`step_elastic`、`block_elastic`）的默认值。用规划时刻真正可用的最宽对齐组：留给图像的卡（`reserve_image_gpus`），以及被别的视频停在 step 中途或正在分段解码占着的卡，都不算空闲。
- `widest`：用策略的最大度，在负载下偏乐观。整组空闲很少见，只有空闲整机才能救的请求仍保留 EDF 优先级，以窄组运行后迟到，拖累其后本可按时完成的请求。`configs/studies/local-a100/e4*.json` 显式使用 `widest`，以复现原来的 E4 结果。
- static_spK 始终按固定度数（再提升到该类别的 `min_degree`）判断，忽略这个选项，因此覆盖所有策略的 `/options` 矩阵仍然可用。

成本画像 `profile` 的 `provenance` 必须为 measured、extrapolated、synthetic 或 mixed。视频类别给出 steps、blocks_per_step、可选 min_degree（单卡放不下时的最小 SP 度，固定 SP 策略也会提升到该值）、prepare/encode/finish、SP1 每步计算时间、每步 Ulysses all-to-all 全序列字节数与集合通信次数；`links` 按 SP 度给出每 rank 带宽和延迟；`sp_efficiency` 与 `resize_ms` 描述并行效率与 step 边界换组成本。还有几个可选字段，不填时结果与以前完全相同：
- `links.<d>.concurrent_gbps`（如 `{"2": 3.9}`）：n 个 SP 组（度数 ≥ 2）同时推进时的每 rank 带宽，取不大于 n 的最大键。用来描述多个组争用 host bridge 的情况。
- `quantum_overhead_ms`：每发出一个 quantum 的固定端到端开销。
- 类别的 `cfg_exchange_mb`：CFG 并行时两半每步交换的预测大小（一个 latent，`cost_profile.py` 按架构算出；5B 720p×121 约 10.5 MB），按 `links.2` 的带宽和延迟计价。
- `sp_overlap` ∈ [0, 1]（可在 `links.<d>.sp_overlap` 按度数覆盖）：worker 以 `block_interleaved` 交错两个 CFG 分支时，SP 形态（cfg = 1）每步被另一分支的计算藏起来的通信比例，取计算项与通信项中较小者乘以该比例；cfg = 2 的形态每个 rank 只有一个分支，不受影响。

每步时间：`compute/(cfg·sp·η) + (comm/cfg)·(sp−1)/sp²/gbps(sp, n) − 隐藏量 + (collectives/cfg)·latency(sp) + [cfg=2] exchange/gbps(2) + latency(2)`。

规划器在迟到判断、度数选择、quantum 大小和图像 slack 中都会计入这两项。执行器和模拟器把正在运行的 SP 推进数作为 `sp_groups_running` 传给规划器。模拟器按发出时刻的并发组数给每个 SP quantum 计价，之后并发数变化时不重新计时。`simulate` 配置可另给 `planner_profile`：规划器按它估算，实际耗时仍按 `profile`。这样可以复现"用一套画像规划、按真实耗时执行"的真机运行。模拟结果每个请求记录 `action_ms`（按 quantum 类型分）和 `quanta`，与真机工作簿对应。模拟结果只描述成本模型，不是模型执行。

`options.sp_transport` 决定请求如何到达新组的 rank：`pinned`（真实后端默认）固定 SP1 且不换组（`real_backends_pinned_sp1`）；`resume`（仅 mock）按 step/block 计数在新 rank 重新 prepare；`export` 让各卡视频 worker 组成一个进程组（`options.sp_port`，默认 29600），由一个 rank prepare，进入组的 rank 在任何 rank 推进前导入其 step 边界状态的 host-staged 副本（`block_export`/`block_import`，目录 `options.transfer_dir`，默认工作目录下 `transfer/`）。`export` 需要 2/4/8 张卡；请求记录 `transfers`、`transfer_bytes`、`transfer_export_ms`。mock 后端可按 `options.costs` 和 `sp_degree` 模拟耗时以测试并发与换组。`simulate` 配置包括 `profile`、`generator`/`workload`、`gpus`、`policies`、`options`、`matrix`；未给出时使用内置 synthetic 示例（`configs/simulation.json` 为完整样例）。

## 预算、阶段和资格

`limit_gpu_seconds:null` 表示无限累计额度；有限值包括修复预留，普通运行不能用预留。`stage_limits` 可限制各阶段，`carry:"forward"` 将之前阶段的剩余额度向后结转；`none` 不结转。实际计费可能因清理超过额度，仍须如实记入。CPU mock 为零 GPU 费用，但 wall-clock watchdog 仍有效。

`requires:["parent"]` 只要求技术完成；`requires:[{"phase":"parent","requires":"qualified"}]` 要求该阶段配置的 gates 全部通过。gates 用 metric JSON pointer、op（eq/ge/le/gt/lt）、value 和 scope（numeric/semantic/performance）描述。没有证据或未配置的 scope 显示 `unverified`，数值一致性不能自动授予语义或性能资格。

阶段失败后 `--resume` 创建新 attempt；成功阶段跳过，失败 cell 不原地伪续。修复预算须同时提供 `--repair --retry-of <failed-session-id>`，绑定同一配置与失败阶段。累计账本独立于 run，换 run 不重置实验账本。清理未确认时账本保持 active/unsettled，并拒绝继续使用；不能靠手动清空 active 绕过调查。

`--phase ID` 可只执行一个已满足依赖的阶段。新格式恢复要求配置、运行 ID、账本和冻结源码一致。历史实验只读，通过旧结果适配器访问；历史脚本的逐阶段续跑不在新接口兼容范围内。

## worker 协议

标准输入/输出为带版本 JSONL，每条消息有单调动作编号：

```json
{"version":1,"id":1,"op":"prepare","spec":{"request_id":"r0","steps":3}}
{"version":1,"id":1,"ok":true,"result":{"state_id":"r0","progress":{"step":0,"steps":3,"done":false}}}
```

错误响应为 `ok:false,error:{type,message}`。模型、张量、RNG、KV 和缓存留在 Python；C++ 只接收状态 ID、进度和有来源的文件元数据。stdout 专用于协议，日志进入 stderr。C++ 用非阻塞管道、进程组和期限监督；模型异常、错误消息、取消、卡死会终止所属进程组，清理无法证实时不结账。无需 RPC 服务或 LibTorch。
