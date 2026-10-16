# 能力覆盖与精简记录

重写将实验决策集中到 C++17：工作负载生成、调度、阶段依赖、矩阵/配对顺序、门槛、累计预算、重试、进程监督和播放时钟。Python 保留模型及张量状态、质量评测、Excel/报告和可选 API；正式 CLI 由 C++ 解析，Python 启动器只 exec 原生入口；C++ 与常驻 worker 通过 JSONL 交换不透明状态 ID 与结果元数据，不引入 LibTorch 或常驻 RPC 服务。

新的统一流程保留研究能力，但不逐个复刻旧专项命令、授权账本或 next/followup runner。旧证据保留原字段与哈希，只读访问；旧实验原地续跑不属于新接口兼容范围。新配置与恢复规则见[实验配置](experiments.md)，归档规则见[结果处理](results.md)。

## 能力覆盖

表中的 CPU 测试验证协议、数值小模型或本地 fixtures；不能推导真实 GPU 性能、容量、图像质量或语义资格。本次没有启动 GPU 实验、下载模型或重新安装已有模型环境。

| 旧能力 | 新模块 / 配置与边界 | CPU 验收位置 / GPU 状态 |
| --- | --- | --- |
| Python 参考模拟、成本模型、策略与指标 | `benchmarks/src/{simulator,workload,cost,policy,metrics}.cpp`；公共随机数在 `src/ditserve/runtime/random.cpp`，配置为 `configs/simulation.json`。合成成本仍明确标为非实测。 | `benchmarks/tests/test_simulator.cpp`、同目录 `fixtures/python-core-baseline.json`、`test_core_bridge.py`；不产生 GPU 结论。 |
| 多套预算、监督、phase/next/followup 控制 | `control/budget.cpp`、`runtime/process.cpp`、`control/study.cpp`；独立累计账本、阶段限额、向后结转、修复预留、wall-clock watchdog、失败费用及未结算保护。 | `test_supervision.cpp`、`test_study.py`：锁、超时、进程组、恢复身份、失败重试；GPU 占用与故障清理尚未真实验收。 |
| 配对 profile、筛选与确认、优化矩阵 | `control/study.cpp` / `control/dispatch.cpp`；`profile.json`、`online.json`；统一矩阵、正反顺序、独立 observation ID、配置门槛。 | `test_control.cpp`、`control-baseline.json`、`test_study.py`；没有新性能提升声明。 |
| 在线到达、微批策略、步边界调度与准入 | C++ `study_workload`、`select_action`、replay；steady/Poisson/bursty/gamma、FCFS/EDF/SRTF及 guarded/feasible 策略。模型合法步边界仍由 worker 校验。 | 冻结旧 Python oracle、决策边界与 mock 回放；真实吞吐/时延未验证。 |
| Wan 整步、block continuation、paired cache | `models/wan/{runtime,block,compile,cache}.py`；C++ 控制推进量与切换时机。半步 packet 不可作为可迁移整步状态。 | `test_model_blocks.py`：tiny Wan 30 层的 block/整步数值、求解器提交、取消与释放；真实编译/GPU路径未验证。 |
| CFG 各分支、oracle/native、split-prefix 与空上下文 | `models/wan/cfg*.py`、`models/wan/research.py`；策略和实际 spec 绑定，保留诊断隔离与失败回滚。 | `test_model_cfg.py`：四种门控、冷锚点、RNG及别名；GPU性能与质量资格未验证。 |
| hot/cold、缓存状态库存与跨 worker 迁移 | `state.json`、C++ 状态调度；`models/wan/{state,storage,migration}.py` 保存求解器/RNG/CFG/已知存储别名与两阶段所有权。 | `test_model_state.py`、`test_study.py`：往返、校验失败、ACK/释放/提交；跨 worker 当前仅 SP1 整步。`certified_omit` 必须持有效刷新证明；本地 `exact_refresh` 仅 validation 模式。 |
| 文本预编码与异步 VAE 解码 | C++ `text_prefetch`、`decode_mode`；Wan 编码线程和 `models/wan/async_decode.py`。异步 VAE 保留 MATH attention、SP1 限制。 | mock 回放及 CUDA stream 替身测试，检查请求身份与同步恢复；未验证真实并发收益。 |
| SP/Ulysses 固定进程组与重分片 | `sp.json`、C++ rank 并发 actions；`models/wan/{parallel,sp}.py`。固定四进程 NCCL world，组内 degree 1/2/4；只支持既有 cache-off all4↔group01 迁移，不支持缓存特征跨 SP 组。 | CPU 协议和状态重分片契约；多 GPU 等价性、通信、效率均未验证。 |
| Sana 与 Mistral 多轮 LLM–DiT session | `session.json`；C++ stage/think-time/依赖与策略，`models/session/runtime.py` 保存 KV、采样器、RNG及精确前缀所有权。 | `test_model_sessions.py`：实际 tiny Mistral greedy/KV与 tiny Sana SCM2步；交错隔离与状态重建。真实后端沿用原硬件限制，GPU未验证。 |
| Sana compiled/graph 候选与 FLUX 管线 | `models/sana/{runtime,compile,graph,hardware}.py`、`models/flux.py`、`profiling.py`。保留候选与失败门；FLUX及 `wan_pipeline` 保留固定离线 batch（重复 prompt、整批单CPU Generator），以完整管线为调用边界；不引入连续batch。 | JSON协议、惰性导入、FLUX局部hook数值与失败恢复；GPU compiled/graph容量、质量、在线资格和FLUX路径未验证。 |
| LongLive 在线 prompt 与播放缓冲 | `streaming.json`；C++ FIFO/latest-only/flush、0/1/2 chunk缓冲及独立帧消费；`models/longlive.py` 保留固定来源。 | `test_study.py`：更新、缓冲、drain超时、未完成请求分母；不是浏览器显示时延，LongLive GPU与语义盲评未验证。 |
| VBench/VidProM 数据冻结及独立质量评价 | `evaluation/`；完整VBench元数据、VidProM有界流式采样、分组划分、输出证据冻结、latent/video/cache配对、LPIPS、官方VBench。 | `test_evaluation.py`：本地元数据/数组、FFmpeg小视频、官方协议及超时替身；未下载数据，未运行官方权重评价。子集不伪造总分。 |
| Wan 分解 profile 与硬件采样 | `models/wan/profile.py`、`evaluation/cuda_profile.py`、`evaluation/nsys_metrics.py` 和 `evaluation/wan_profile/`；区分 GPU activity、CPU 注释与外部 Nsight 采样。 | 本地 trace/SQLite 与采集流程 fixtures；缺指标为 null。未接入实际调用链的 NCU 权限探针及专用测试已删除，Nsight 采样与数据读取保留。真实 GPU 采集未验证。 |
| 多套报告、归档与清理 | `results/` 统一两文件交付和可恢复 finalizer；保留旧 compact/archive/session readers及skill入口。 | `test_results.py`、`test_legacy_results.py`：类型/精度/溢出、只读旧证据、哈希、账本保护、依赖闭合、删除失败/崩溃恢复；所有删除仅临时 fixtures。 |

原 `benchmark`、`directions`、`experiments`、`gpu`、`optimizations`、`research`、`scheduling`、`sessions` 八个 Python 包已删除。当前源码和测试没有对这些包的 import 依赖；保留在数据里的旧 schema 名称不等于保留旧控制器。模型加载、张量与缓存的底层算法留在 `models/`，不是把旧 Python runner 换个入口继续调用。

## 结果兼容层为何保留

[save-experiment-results skill](/home/zyd/.codex/skills/save-experiment-results/SKILL.md) 及其规则保留；项目参考文件仅同步当前源码位置、启动路径，并标明旧接口属于历史冻结源码。结果兼容实现集中于 `results/compact.py` 和 `results/legacy.py`；顶层三个转发模块已移除，调用方直接使用所属功能模块。

- `results.compact.export/verify/prune`、显式白名单、持久删除意图/凭据及 `verify_compacted_reference` 仍是技能的旧格式接口。不能只保留“能打开 Excel”的读者而丢掉 gzip、缺失原始数据的哈希证明和活动保护。
- `results.compact.export_report` 为报告导出接口；新 run 转发统一 finalizer且不自动删除，旧 run 使用可重用导出，活动或失败状态明确返回。`runtime` 提供统一包路径和 JSON 解析。
- `results/legacy.py` 保留旧格式读取、逐格验证和报告一致性规则；无人调用的旧写入/重建入口已删除。兼容测试使用固定旧格式样例，不能通过运行当前写入器生成“预期值”掩盖读取回归。旧归档内容及其哈希语义不改写。

旧结果、旧累计账本、来源快照及删除凭据不随包删除而清理；没有批量迁移历史归档。新读者支持范围与无法证明的内容见[历史兼容说明](results.md#历史结果只读兼容)。

## 可恢复基线与统计

改写前快照为 `.cache/rewrite-baseline/20261016T064045Z/source.tar.gz`，逐文件字节数和 SHA256 在同目录 `manifest.json`；压缩包 SHA256 已重新计算并与清单一致：

```text
c2bafafcd408dddebb8f5928ac010ed6a5ac5a7ca3776b32a9fec77d5a63e721
```

该快照覆盖改写前项目源文件，包括当时工作区改动；不包含模型环境或实验结果。需要比较或恢复时，应先校验，再解包到独立目录，不覆盖当前工作区。

统计与 baseline manifest 使用相同口径：Python 为 `python/**/*.py`；C++ 为 `src/`、`include/`、`cmd/` 的 `.cpp/.hpp`；测试为 `tests/` 的 `.py/.cpp`。行数统计物理换行，包括注释与空行；不包含缓存、依赖、结果、文档、JSON fixtures和模型。专项实验脚本原本位于 Python 包内，因此计入 Python 行数。

这里的 `python/` 是历史基线目录；首次重组统计使用当时的 `src/ditserve/**/*.py`，当前包位于 `workers/ditserve/`。目录迁移不改写基线清单及其来源路径，C++ 按 `.cpp/.hpp` 后缀单独计数；本轮布局和独立统计见文末。

| 类别 | 改写前文件数 / 行数 | 首次重写后文件数 / 行数 |
| --- | ---: | ---: |
| Python 源码 | 199 / 61698 | 56 / 12767 |
| C++ 源码 | 19 / 1181 | 27 / 3039 |
| 测试源码 | 178 / 44986 | 23 / 4938 |

以上为首次重写验收快照；下方另记功能重组后的统计。首次重写生产源码合计从62879行降到15806行，减少74.86%；测试源码减少89.02%。代码行数减少不代表模型更快，也不代替覆盖范围核对。

## 验收记录

验收命令见[README](../README.md#构建和验证)。C++ 契约分为 core、control、supervision；Python 包含独立进程协议、mock统一流程、tiny模型数值和结果归档。模型相关CPU测试使用已有固定环境，并设置 `CUDA_VISIBLE_DEVICES=''` 与离线标志；基础环境缺少可选库时的跳过不计为模型验收通过。

报告缓存库只通过 `DITSERVE_REPORT_TOOLS` 按需附加到导入路径末尾；不能把系统Python的 `.cache/tools` 全局置于模型环境前面。真实模型依赖保持现有环境的版本与ABI。

首次重写验收：CTest **3/3** 通过（1.46秒）；固定模型环境的 Python **181项** 通过、**0跳过**（52.355秒），包括六类执行器的常驻 mock 端到端、最终 profiling/batch 接口和历史兼容测试。

CPU 实跑包含48条请求、6次观测，发布的[工作簿](../results/rewrite-validation/cpu-20261016/results.xlsx)和[报告](../results/rewrite-validation/cpu-20261016/REPORT.md)已通过两文件校验。并对既有 `research-cfg/excel` 和 `research-sessions/run01` 做了只读校验，原工作簿哈希不变。48份原始fixture输出共3696字节已按凭据清理；约2.54MB工作目录仍保留恢复控制文件与冻结源码。

同次CPU实跑记录常驻动作通信残差：p50 **0.157338ms**、p99 **0.316865ms**。计时排除进程启动、模型加载和worker操作时间，采样每请求前128次动作；仍包括JSON编解码和管道/调度等待。这是CPU fixture观测，不是纯管道延迟，也不能用于声称GPU服务收益。


## 功能重组与职责收敛

该阶段将 C++ 头文件与实现按 `simulation`、`control`、`runtime` 对齐，Python 模型按 Wan、Sana、会话分组，数值评测归 `evaluation`，结果实现归 `results`。以下记录保留阶段性统计，最新目录布局见 [README](../README.md)。

- 7 个内置策略类合为一个配置型实现，保留 `Policy/Registry` 扩展能力；合成后端统一走 `Backend/State`，具体状态和数值实现留在模块内部。
- 实验主流程保留准入、预算开始/结算、门槛与持久化；`execute_cells` 独立管理矩阵执行、worker 监督和清理证明。
- Mistral/Sana 共用状态迁出、恢复、释放与关闭方法；简单后端复用基础释放实现。LongLive 两文件合并，严格证书哈希共用一个实现。
- worker 分离协议校验、模型调用和无状态结果/评测服务，移除两个服务后端类；报告错误汇总独立于展示，重复序列化与文件哈希复用既有工具。

本轮恢复快照为 `.cache/refactor-baseline/20261016T082215Z/source.tar.gz`，同目录保存逐文件哈希和统计；实际结果、累计账本及冻结源码未改动。

| 统计口径 | 重组前 | 重组后 |
| --- | ---: | ---: |
| Python 类定义 | 59 | 57 |
| Python 函数/方法定义（含内部函数） | 651 | 644 |
| Python 源文件 / 行数 | 56 / 12767 | 60 / 12754 |
| C++ 源文件及头文件 / 行数 | 27 / 3039 | 25 / 2994 |

Python 文件数增加来自包入口和显式兼容导入；没有复制模型实现或删减数值/清理校验来压缩统计。重组后 CTest **3/3** 通过，Python CPU 测试 **184 项、0 跳过**通过（51.963 秒）；包含常驻 mock 端到端、数值状态、预算/故障、旧哈希和结果恢复。离线构建的 wheel 在临时目录安装后，统一入口、独立结果入口、10 个 worker 的惰性握手及旧结果 API 均通过。真实 GPU 路径未执行。


## 原生命令入口

正式程序为 `.cache/build/ditserve`，`interface/command.cpp` 统一处理 `doctor`、`simulate`、`run`、`report` 和 `results`。原 `cli.py` 的参数解析已移至 C++，原 `api.py` 与调用桥合并为 `runtime/client.py`；`python -m ditserve` 只提供兼容启动。低层 JSON 调用明确置于 `rpc generate|simulate|control|run|simulate-study`。帮助、版本和无模型 RPC 不依赖 Python；环境检查、评测和结果处理按需调用 Python worker。

原生入口自行传递包路径，用户不必设置 `PYTHONPATH`；默认采用已有 `.venv-previs/bin/python`，也可传 `--python`。`tests/python/test_native_cli.py` 用临时目录覆盖 CLI 参数、RPC、模拟交付、mock 运行/恢复、结果读取/收尾和失败退出状态。旧硬件采集包装器根据冻结清单选择原 Python CLI 或新原生入口，不修改旧来源、账本和 run。

本轮入口调整前快照为 `.cache/refactor-baseline/native-entry-20261016T085820Z/source.tar.gz`，用于独立恢复和比较；不覆盖历史结果或来源快照。

| 本轮生产源码 | 调整前文件数 / 行数 | 调整后文件数 / 行数 |
| --- | ---: | ---: |
| Python | 60 / 12754 | 58 / 12678 |
| C++ 源码及公开头 | 25 / 2994 | 27 / 3398 |

最终完整 CPU 套件 **194 项、0 跳过**通过（69.496 秒）；另外 **2 项**定向契约通过（1.278 秒），覆盖超过 4 MiB 的完整结果传输和取消后 worker/子进程组回收，共覆盖 **196 项**。CTest **4/4** 通过（1.39 秒）。安装至临时 prefix 后，原生入口、mock 运行、两文件校验、28 个 native 来源文件冻结均通过；离线 wheel 和 PATH 下的 Python 兼容启动器也已验收。本轮基线压缩包内所有文件的 SHA256 均与清单一致。此轮未执行真实 GPU 实验。

## 零散 Python 文件归并

Python 包顶层从 6 个文件收敛为标准 `__init__.py` 和 `__main__.py`。路径与严格 JSON 解析合入 `runtime`，移除未使用的 JSON 读取函数；三个结果转发文件删除，调用方直接使用 `results.compact`（包含 `export_report`）和 `results.legacy`。结果 skill 的项目参考同步这些路径，规则与历史格式保持不变。

Wan 专项采集、图表和发布集中在 `evaluation/wan_profile/{capture,figure,publish}.py`；移除旧 `scripts/`，使用包内导入和模块调用，不再注入 `sys.path`。C++ 仍是正式实验入口。

按 `src/**/*.py` 加专项 `scripts/**/*.py` 的相同口径，Python 源码由 **60 文件 / 13156 行**变为 **57 文件 / 13133 行**。恢复快照为 `.cache/refactor-baseline/python-modules-20261016T091923Z/source.tar.gz`；其中 137 个文件的 SHA256 均已核对。未改写实际结果、账本或冻结来源。

本轮完整 CPU 测试 **197 项、0 跳过**通过（73.572 秒），CTest **4/4** 通过（1.50 秒）。临时 prefix 安装、离线 wheel、安装后 mock 实验以及冻结 Python 包导入通过；包内已包含新的 Wan 分析模块，没有旧顶层转发文件。真实 GPU 未执行。

## 生产模块边界与引用清理

最新布局以 [README](../README.md) 为准。`src/` 仅保留 7 个 C++ 生产实现文件；公共头在 `include/`，正式入口在 `cmd/`。合成模拟的 6 个实现、6 个头及 golden 测试独立放在 `benchmarks/`；Python 模型、协议、评测和报告服务放在 `workers/ditserve/`。共享随机数留在核心，真实工作负载不依赖 benchmark；CMake 应用层同时链接核心和 benchmark，库依赖无环。安装和来源冻结同步新布局，旧 `sources/python`、`sources/native` 和 `sources/ditserve-core` 的读取格式不变。

引用审查包含 CMake 显式源文件和测试目标、Python imports、worker 后端/评测注册表、模块命令、unittest discovery、配置模板、文档及实际补采依赖。没有把测试“无人 import”当作删除依据。

| 删除或归并项 | 依据及替代覆盖 |
| --- | --- |
| `configs/synthetic.json` | 无消费者；数值重复成本模型默认值，当前示例统一用 `configs/simulation.json`。 |
| `evaluation/hardware_counters.py` 及其 8 项测试 | NCU 权限探针未接入 worker、命令或采集链，唯一使用者是自身测试；保留实际使用的 Nsight 采集与 SQLite 读取。 |
| Wan `verify_policy`、`state_summary`、`_context_record` | 无直接或动态调用；不改变模型推进、缓存或恢复路径。 |
| 旧归档写入、重建、索引生成闭包 | 仅测试造数据调用；读者改由固定旧格式样例验证，保留完整哈希、类型、报告一致性及只读检查。 |
| 重复哈希实现、无用导入/成员和未调用策略名称方法 | 文件哈希统一到 `runtime`；删除无状态作用的重复成员和 hook。 |
| 配置空声明与重复测试步骤 | 旧报告配置只保留实际消费的两个选项；非法 JSON 用例参数化，重复历史读取测试合并，保留失败和篡改行为覆盖。 |

所有有效模型后端和评测适配器保留。Wan 补采/图表仍有未闭合依赖，相关工具、配置和测试保留；论文 `.tex` 与所引用的 `.bib` 也保留。模型环境、实际结果、账本、迁移快照和历史冻结来源未清理。

| 源码统计（文件 / 物理行） | 本轮前 | 本轮后 |
| --- | ---: | ---: |
| Python 生产源码 | 57 / 13133 | 56 / 12781 |
| C++ 生产源码及头文件（合计） | 27 / 3398 | 27 / 3394 |
| 其中核心 / 应用 | — | 15 / 2444 |
| 其中 benchmark | — | 12 / 950 |
| 测试源码（含 benchmark 测试） | 25 / 5448 | 24 / 5326 |

目录迁移不算代码删除；合计生产代码减少 356 行，测试代码减少 122 行。另新增约 12 KB 的固定历史格式 fixture，来源及逐文件 SHA256 见 `tests/fixtures/legacy-archive/README.md`，不包含真实实验数据。快照 `.cache/refactor-baseline/production-layout-20261016T104354Z/source.tar.gz` 中 134 个文件的 SHA256 均已核验；两个控制/模拟 golden 文件内容未变。

最终构建、CTest **4/4**（1.39 秒）、完整 Python CPU 测试 **187 项、0 跳过**（69.673 秒）通过。临时 prefix 安装、安装后 mock 实验、30 个 native 来源文件冻结、冻结 Python 包导入、离线 wheel 和兼容启动器均通过。没有执行真实 GPU 验收，不新增性能或数值质量资格声明。
