# 实验配置

实验配置（`configs/studies/*.json`，schema `ditserve.study.v1`）与机器无关。`ditserve run|resolve --env <名字>` 用 `configs/env/<名字>.json` 把它们绑定到本机，再交给控制器执行。`ditserve resolve` 只打印绑定结果并做与 `run` 相同的校验，不启动任何 worker。

## 环境文件（`ditserve.env.v1`）

| 字段 | 含义 |
| --- | --- |
| `gpus` | `count`；可选 `uuids`（固定顺序，多卡按 PCIe 拓扑排列）、`forbid_uuids`、`name_contains`、`min_memory_mib`。没有 `uuids` 时，取编号最小的符合条件的卡 |
| `models_root`、`models` | 逻辑模型名对应路径（相对路径以 `models_root` 为根）、`model_id` 和 `revision`。环境变量 `DITSERVE_MODELS_ROOT` 可覆盖 `models_root` |
| `worker_environment` | 所有 worker 的默认环境变量，例如线程数和 4090 上的 `NCCL_P2P_DISABLE=1`。不能覆盖 GPU 绑定、源码路径或离线加载 |
| `backends.<backend>` | 该后端 worker 的默认 config，例如 `attention`、`memory_fraction` |
| `cost_profile`、`links` | 实测成本画像（`configs/profiles/` 下的名字），以及覆盖画像链路的假设场景名（`configs/profiles/link-scenarios.json`） |
| `serving_classes` | 这台机器能服务的请求类别和权重 |
| `host` | doctor 检查用的内存、`/dev/shm` 和文本编码器内存需求 |

实验配置可以使用以下写法：

- **按卡展开的 worker：** `per_gpu: true` 展开成 `name.0`…`name.{k-1}`，每张选中的卡一个。
- **绑定单张卡：** `gpu: N` 绑定第 N 张选中的卡；机器卡数不够时，该 worker 被跳过并记录。
- **逻辑模型名：** `model: <逻辑名>` 解析为 `config.model_path`。
- **cluster slot 模板：** cluster 阶段的 `gpus: {video, image}` 按卡数（或 `gpu_count`）展开成 slot；`warmup` 与 operations 阶段的 `workers` 展开为对应实例。
- **卡数门槛：** `min_gpus` 超过选中卡数的阶段，以及依赖它的阶段，会被跳过并记录。
- **引用环境字段：** 任意位置的字符串 `"@env.KEY"` 替换为环境文件的对应字段，例如 `"@env.cost_profile"`、`"@env.serving_classes"`。

绑定后，experiment 名加上环境前缀，结果写到 `results/<环境>-<实验>/`。冻结配置的 `environment` 记录 GPU 清单、选中的 UUID、每个 worker 的绑定、模型路径和跳过的阶段。换了机器或环境文件再续跑会被拒绝。

## 执行器

| kind | 用途 |
| --- | --- |
| `operations` | 按顺序执行 `actions`，每个动作同时发给阶段的所有 worker（rank），可用 `only_ranks` 只发给其中几个。相对路径在每个 rank 的私有目录下，`@cell/` 开头的路径在该 cell 所有 rank 共享的目录下。用于数值验收、成本测量、跨进程/跨卡恢复和 SP |
| `cluster` | 多卡并发回放。每张卡上有一个视频 worker 和一个图像 worker，每个 quantum 之后由 `cluster_plan` 重新决定放置、SP 度和抢占 |

动作包括 `prepare`、`advance`（`quantum:"step"` 或 `blocks:N`）、`finish`（`decode_frames` 为分块解码）、`release`、`reference`（diffusers 官方循环）、`block_export`/`block_import`、`encode_start`/`encode_poll`、`distributed`、`evaluate`（数值对比，`atol`/`rtol`）、`memory`（分配器峰值，`reset_peak`）、`hello`。

矩阵键是相对阶段的 JSON pointer，按键名排序取笛卡尔积，最多 10000 个 cell。`paired: true` 会再按逆序执行一遍。

## 调度（`cluster` 与 `ditserve simulate` 共用）

| 策略 | 行为 |
| --- | --- |
| `static_sp1/2/4/8` | 固定 SP 度，整步推进，已开始的视频留在原组（近似 xDiT） |
| `step_elastic` | EDF；按期限选最小可行 SP 度，没有其他等待请求时占用最大空闲组；只在 step 边界换组（近似 GENSERVE、TetriServe） |
| `block_elastic` | 同上，但视频按 `quantum_ms` 换算出的 block 数推进；图像在下一个 block 边界抢占 |

选项：

- `quantum_ms`：`block_elastic` 必须给出。时间目标让抢占延迟在不同 GPU 上可比。
- `chunked_decode`：默认开启。VAE 按 latent 帧推进，每个 quantum 解码的帧数也由 `quantum_ms` 换算。
- `reserve_image_gpus`：最后 R 张卡只服务图像。
- `demote_late`、`late_short_ms`：可行优先 EDF。赶不上期限的请求排在能赶上的之后，但剩余工作不超过 2s 的迟到请求保留 EDF 位置。
- `degrees`、`pin_started`：真实后端会被强制设为 SP1、不换组。

未知选项会被拒绝。

生成器字段：

- `pattern`：`offline_all`、`steady`、`poisson`、`bursty`、`gamma`。
- 到达率：`rate_rps`，或 `load`（实测容量的占用率；容量 = 卡数 / 按权重平均的 GPU 服务时间）。
- 请求数：`count`，或 `count_per_gpu`。
- `classes`：类别名、kind、权重、spec、`slo_scale`。未给出 `service_ms` 时由成本画像计算：编码 + 准备 + 步数×每步时间 + 解码。

扩散步数取自 spec。cluster 阶段加上 `simulate: true` 时，模拟器会用同一画像复算该 cell 的同一条 trace，结果记在 `simulated`。

成本画像的 `provenance` 必须是 measured、extrapolated、synthetic 或 mixed。视频类别给出 steps、blocks_per_step、decode_chunks、可选的 min_degree、prepare/encode/finish 耗时、SP1 每步计算时间、每步 Ulysses all-to-all 字节数和集合通信次数。`links` 按 SP 度给出每个 rank 的带宽和延迟。画像由 `python -m ditserve.evaluation.cost_profile` 从 `cost_profile` 实验的 run.json 生成。

模拟配置（`configs/simulation.json`）含 `profile`、`generator`、`gpus`、`policies`、`options`、`matrix` 和 `variants`。`variants` 是逐个合并进每个 cell 的补丁，用来表达"这个选项只配这些策略"。

## 模型路径的固定限制

- **Wan：** DiT FP16，VAE FP32，UMT5 在 CPU 上以 FP32 运行；不缓存特征，每步完整计算。形状按 Wan2.1-1.3B 写死（移植 5B 见 MIGRATION.md 第 9 节）。SP 需要 Flash attention 和 NCCL，至今未在 GPU 上验证。
- **Sana：** 官方 eager 管线，BF16，1024²，SCM 2 步，CFG 4.5，需要 Ampere 或更新的卡。一次调用跑完整条管线，不能在中途暂停。

## 预算、阶段和资格

- **GPU 秒预算：** `limit_gpu_seconds` 是该实验所有运行累计的 GPU 秒上限，`null` 表示不限。账本在 `.state/<实验>/budget.json`，换 run 不会重置。另有 `stage_limits`、`carry`（向后结转剩余额度）、`repair_reserve_gpu_seconds`。即使不限额，wall-clock watchdog 也照样生效。
- **阶段依赖与门槛：**
  - `requires:[{"phase":"p","requires":"qualified"}]` 要求前一阶段的 gates 全部通过。
  - gate 是 `metric` JSON pointer 加 op（eq/ge/le/gt/lt）、value 和 scope（numeric/semantic/performance）。
  - pointer 中的 `*` 段匹配每个元素：所有匹配项都通过才算通过，一个匹配都没有则为 unverified。
- **续跑：** 失败后 `--resume` 新建一个 attempt，失败的 cell 不会就地续跑。修复预算要同时给 `--repair --retry-of <失败 session>`。
- **GPU 隔离：** 运行前等所选 GPU 空闲（`admission_wait_seconds`），运行中每 2 秒检查一次，发现外来进程就中止。

## worker 协议

stdin/stdout 上传带版本号的 JSONL，每条消息有单调递增的编号。错误以 `ok:false,error:{type,message}` 返回。模型、张量、RNG 留在 Python 里，C++ 只接收状态 ID、进度和带来源的文件元数据。stdout 专用于协议，日志写到 stderr。模型异常、取消或卡死时，C++ 终止所属进程组；清理无法证实时不结账。
