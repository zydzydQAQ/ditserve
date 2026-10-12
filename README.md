# DiT Serving Lab(轻量重写版)

C++17 调度控制面 + Python 前端 API,用于检验固定特征缓存规则下的批处理、抢占和状态恢复。本仓库是 `~/code/dit` 的轻量重写:控制面语义与旧核心**逐位一致**(44 组同 seed 用例逐字段比对通过,golden fingerprint 不变),Python 侧从 4.7 万行收敛到约 1200 行。

合成成本结果不代表 GPU 实测性能;参考一致性不等于感知质量。

## 架构

```
┌────────────── Python 前端 API(零硬依赖)──────────────┐
│ cli.py     doctor / simulate / sweep / archive        │
│ api.py     simulate() / sweep() / doctor() 编排       │
│ core.py    CoreBridge:子进程 JSON 协议、二进制发现    │
│ workload.py  负载构造与 trace 加载                    │
│ archive.py   归档 verify / rebuild(xlsx+MD+gz)       │
│ report.py    Excel / CSV / Markdown 渲染              │
└──────────────┬───────────────────────────────────────┘
               │ stdin: schema v2 JSON ─ stdout: JSON
┌──────────────▼────────── C++17 控制面 ────────────────┐
│ Simulator(事件循环:到达/完成/抢占/卸载/恢复)         │
│ Policy 抽象 + PolicyRegistry ←— 6 内置策略(策略模式) │
│ Backend 抽象 ←— ReferenceBackend(合成参考执行器)     │
│ CostModel(值对象)            metrics(汇总函数)      │
└───────────────────────────────────────────────────────┘
```

| 目录 | 职责 |
| --- | --- |
| `src/` + `include/` | C++ 核心,9 个同名 .cpp/.hpp 模块对(CMake 强制一一对应) |
| `cmd/main.cpp` | `ditserve-core` 二进制:stdin 读 JSON,stdout 写 JSON,诊断走 stderr,错误返回 2 |
| `python/ditserve/` | Python 包(src 布局);`reports` extras 仅含 openpyxl |
| `tests/core` | CTest 契约测试 + 固定基线 fixture |
| `tests/python` | unittest:桥协议、归档生命周期、API 编排(注入假桥)、CLI 端到端 |
| `configs/` `workloads/` | 合成成本模型与抢占负载样例 |

## 构建和测试

需要 C++17 编译器、CMake 3.16+、OpenSSL 开发库和 Python 3.8+。nlohmann/json 固定 3.11.3,首次构建下载到 `.cache/deps` 并做 SHA256 校验。报告依赖 openpyxl 准备于 `.cache/tools`。

```bash
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/build -j4
CTEST_OUTPUT_ON_FAILURE=1 cmake --build .cache/build --target test
PYTHONPATH=python:.cache/tools python3 -m unittest discover -s tests/python -v
```

## 运行

```bash
export PYTHONPATH=python:.cache/tools   # 或 pip install '.[reports]'
python3 -m ditserve doctor
python3 -m ditserve simulate --policies all --count 32 --output .cache/runs/demo
python3 -m ditserve simulate --trace workloads/preemption.json --policies all --max-batch 1 --memory-mb 72
python3 -m ditserve sweep --count 16 --rates 2,12 --memories 128,256 --gpu-counts 1,4
python3 -m ditserve archive verify --root results
python3 -m ditserve archive rebuild results/synthetic/<step>
```

`--output` 是临时工作目录;归档验证通过后自动清理,`--keep-raw` 保留。最终 Excel 与报告写入 `results/synthetic/<run-id>/`,可用 `--results-root` 改归档根。`DITSERVE_CORE` 环境变量指定核心二进制。

六种内置策略共用同一事件循环:`cache_only`(固定缓存,FIFO,batch=1)、`batch_only`(禁缓存,FIFO 连续批处理)、`naive`(固定缓存,FIFO,混合路径按较贵分支计费)、`edf`(deadline 优先,步边界抢占)、`path_aware`(EDF + 仅同路径组批)、`state_aware`(按历史成本、剩余时间和恢复代价排序,分路径组批,按价值驱逐)。

## 扩展

**新增调度策略**(策略模式,不动调度器):

1. 在 `include/ditserve/policies.hpp` / `src/ditserve/policies.cpp` 写一个 `Policy` 子类,实现 `name()`、`priority()`,按需覆写 `uses_cache/preemptive/group_paths/same_path_only/state_aware/batch_limit` 钩子;
2. 在 `register_builtin_policies` 加一行注册,或在进程内调 `PolicyRegistry::instance().add(name, factory)`;
3. 用 `--policies <name>` 直接运行。

**替换执行后端**(依赖注入):实现 `Backend`/`State` 接口(prepare / peek_path / step_batch / restore / fingerprint),以 `Simulator(config, cost, backend)` 注入;调度器、内存记账与归档不需要任何改动。

## 接口与归档

Python 按整次实验调用 `ditserve-core generate|simulate`。输入是 `schema_version: 2` 的 JSON 对象,输出是一个 JSON 值。核心不依赖 Python 或 CUDA。v2 使用 `mt19937_64` 和固定采样算法;检查点覆盖 RNG、latent、求解器历史、缓存与路径历史,完整 fingerprint 用于孤立执行、批处理和恢复的一致性校验。

每个归档步骤含 `results.xlsx`、`REPORT.md`、`data.json.gz`(全部小文件原文 + SHA256,gzip mtime=0 可复现)。`verify` 逐字节核验证据、逐单元格比对重建工作簿、检查报告链接;`rebuild` 仅凭 data.json.gz 重建产物,无需 GPU 或模型。

## 与旧仓库的关系

- 语义兼容:同一 schema v2 协议、同一 RNG/采样算法、同一事件循环;同输入下 summary/records/events 逐字段一致(浮点容差 1e-9)。
- 未移植:PreViS v1/v2 GPU 实验系列、V100/phase-conflict 脚本对、视频 benchmark 子包、一次性迁移/下载工具、83 个镜像式测试。这些仍在 `~/code/dit` 及其 `results/` 归档中可用。
- 代码量:旧仓库约 48,300 行 → 本仓库约 2,600 行(C++ 1,181 + C++ 测试 226 + Python 包 730 + Python 测试 465)。
