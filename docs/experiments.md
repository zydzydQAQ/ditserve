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

矩阵键是相对于 phase 的 JSON pointer，按键排序生成笛卡尔积，最多 10000 个 cell。例如 `/policy`、`/quantum`、`/worker_config/options/execution`。`worker_config` 合并到该 cell 指定 `worker`（默认 `model`）的配置；不能改变 worker 设备或身份。每个观测的模型进程持续复用，cell 间重新创建以隔离配置。配对时记录 cell、正/逆次序和独立 observation ID，失败观测保留在分母中。

`workers.<name>` 包含 `backend`、`python`、`gpu_uuid`、`config` 和可选环境变量；单个进程只绑定自己的完整 GPU UUID，模型使用 `cuda:0`。不能用环境变量覆盖 GPU 绑定、冻结源码或离线加载。SP 的固定实现需要四个独立 rank，Flash attention，支持 degree 1/2/4；缓存特征的跨 SP 迁移仍不支持。集体操作必须列出不同 worker，`actions[].per_rank` 可提供每个 rank 的补丁。

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

使用冻结 `workload` 数组，或确定性 `generator`。生成器支持 `offline_all`、`steady`、`poisson`、`bursty`、`gamma`，输入包括 seed、count、rate_rps、cv、prompts、spec/specs、service_ms、slo_ms。模型参数在 `spec`；到达、期限、成本估计是 C++ 调度元数据。

请求可以指定 `arrival_ms`、`deadline_ms`、`session_id`、`after`、`think_ms`、`stages`。`after` 必须引用负载中更早的一项；前一轮完成后才释放下一轮。stage 包含 worker、spec、估计 cost_ms 和可选 `prompt_from_previous:true`。Mistral 自动保存该 session 的对话与 KV，不共享其他 session 状态。

策略包括 `fcfs`、`edf`、`srtf`、`edf_b1`、`edf_b1_async`、`edf_b2`、`edf_b4`、`edf_step`、`guarded_b4`、`feasible_b4`、`guarded_adaptive`、`session_serial`、`stage_edf`、`session_slack`。Wan 可采用 step/block/paired_block 边界，CFG 和 cache 配方由 worker 验证；hot/cold 与跨设备快照只允许后端声明的合法边界。FLUX 保留完整管线调用边界，没有新增 token/block 抢占能力。FLUX和 `wan_pipeline` 可用 `spec.batch_size` 表达原有固定离线batch：重复同一prompt，整批使用一个CPU Generator(seed)。`wan_pipeline` 不提供步骤抢占、SP或迁移；启用cache后同一worker的steps/CFG配方固定。

`text_prefetch` 限定同时预编码的已到达请求数，`decode_mode:"async"` 启用后端支持的异步解码。Wan 异步 VAE 保持 MATH attention 的原限制。模型、加载、控制时间分别记录；通信采样排除进程首次启动、模型加载和 worker 操作耗时，仍包含 JSON 编解码。采样保留每请求前 128 次常驻动作及总计数，不声称是纯管道延迟。

流式 `updates` 按 arrival_ms 排序，版本递增。`prompt_policy` 为 `fifo`、`latest_only`、`latest_only_flush`；`buffer_chunks` 为 0/1/2。播放测量是 C++ 线程的墙钟帧分发，不是浏览器或物理显示时延。drain 超时保留失败及未完成请求，语义盲评保持独立。

FLUX 可设置 `workers.<name>.config.options.profile:true`，记录 text/DiT/scheduler/VAE/postprocess 的 CUDA event spans、逐步 finite 检查和 allocator 峰值。该测量有插桩开销，不能当作无插桩延迟或 kernel-only 时间。用 `/worker_config/options/profile:[false,true]` 表达对照；Wan 分解测量使用 `wan_profile_corrected.json` 的显式 actions。

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
