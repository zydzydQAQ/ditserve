# DiT Serving Lab(轻量重写版)

C++17 调度控制面 + Python 前端 API + 真实 GPU 实验层,用于研究固定特征缓存规则下的批处理、抢占和状态恢复。本仓库取代已弃用的 `~/code/dit`:C++ 控制面语义与旧核心**逐位一致**(44 组同 seed 用例逐字段比对通过,golden fingerprint 不变),GPU 实验线为保真移植(同一钉死版本栈与协议守卫)。

合成成本结果不代表 GPU 实测性能;参考一致性不等于感知质量。

## 架构

```
┌────────────── Python 前端 API(零硬依赖)──────────────┐
│ cli.py      doctor / simulate / sweep / archive       │
│             / profile / benchmark(后两个转发到 GPU 层)│
│ api.py      编排;core.py  CoreBridge(子进程 JSON)    │
│ workload.py / archive.py / report.py / util.py        │
└──────────────┬───────────────────────────────────────┘
               │ stdin: schema v2 JSON ─ stdout: JSON
┌──────────────▼────────── C++17 控制面 ────────────────┐
│ Simulator 事件循环;Policy 策略 + Registry(6 内置);   │
│ Backend 抽象 ← ReferenceBackend(合成参考执行器)      │
└───────────────────────────────────────────────────────┘
┌────────────── 真实 GPU 实验层(惰性加载,钉死版本)───┐
│ gpu/           profiling(FLUX/Wan 测量)、worker       │
│                (WanStepWorker:keep/hot/cold 状态迁移)、│
│                phase_conflict 机制、quality(LPIPS)   │
│ benchmark/     视频在线服务 benchmark(VBench/VidProM、│
│                冻结 trace、FCFS/EDF/SRTF、质量评估)   │
│ experiments/   V100 配对、phase-conflict、PreViS2 协议 │
└───────────────────────────────────────────────────────┘
```

| 目录 | 职责 |
| --- | --- |
| `src/` + `include/` | C++ 核心,9 对同名模块(CMake 强制一一对应) |
| `cmd/main.cpp` | `ditserve-core` 二进制协议入口 |
| `python/ditserve/` | Python 包;`gpu/` 真实后端与测量底座,`benchmark/` 在线 benchmark,`experiments/` 协议编排与汇总 |
| `configs/` | 成本模型、v100.json、previs2.json、benchmark/{smoke,formal}.json、requirements-*.txt |
| `docs/` | 协议文档:video_benchmark / gpu_experiments / phase_conflict / previs2 |
| `tests/` | CTest 契约 + unittest(410 个用例) |

## 构建和测试

需要 C++17 编译器、CMake 3.16+、OpenSSL 和 Python 3.8+。nlohmann/json 固定 3.11.3(SHA256 校验下载到 `.cache/deps`)。报告依赖(numpy/openpyxl)在 `.cache/tools`。

```bash
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/build -j4
CTEST_OUTPUT_ON_FAILURE=1 cmake --build .cache/build --target test
PYTHONPATH=python:.cache/tools python3 -m unittest discover -s tests/python -v
```

## 运行(合成模拟,纯 CPU)

```bash
export PYTHONPATH=python:.cache/tools
python3 -m ditserve doctor
python3 -m ditserve simulate --policies all --count 32 --output .cache/runs/demo
python3 -m ditserve simulate --trace workloads/preemption.json --policies all --max-batch 1 --memory-mb 72
python3 -m ditserve sweep --count 16 --rates 2,12 --memories 128,256 --gpu-counts 1,4
python3 -m ditserve archive verify --root results
```

六种内置策略:`cache_only` / `batch_only` / `naive` / `edf` / `path_aware` / `state_aware`,共用同一事件循环。

## 运行(真实 GPU 实验)

GPU 栈在 `.venv-previs`(钉死:torch 2.10.0+cu126 / diffusers 0.36.0 / cache-dit 1.5.0 / transformers 4.57.6;版本不符 worker 直接拒绝——这是有意的协议守卫)。Wan 权重固定在 `.cache/huggingface`(符号链接到 SSD 缓存)。全程 HF 离线。

```bash
export PYTHONPATH=python
.venv-previs/bin/python -m ditserve profile --preflight          # 环境与内核冒烟(秒级)
python3 -m ditserve.experiments.run_v100 --dry-run               # V100 配对计划(不启动 GPU)
python3 -m ditserve.experiments.run_phase_conflict --dry-run     # 阶段冲突计划
python3 -m ditserve.experiments.run_previs2 --dry-run            # PreViS2 计划
python3 -m ditserve benchmark --help                             # 在线服务 benchmark
```

三套预算账本各自独立、fail-closed:benchmark 2 GPU·h、phase-conflict 6 GPU·h、PreViS2 12 GPU·h。账单规则不同且有真实资源后果,刻意不合并。先读 `docs/` 下对应协议再启动任何 GPU 作业。

## 扩展

- **新增调度策略**:在 `include/ditserve/policies.hpp` + `src/ditserve/policies.cpp` 写一个 `Policy` 子类并在 `register_builtin_policies` 注册一行;或进程内 `PolicyRegistry::instance().add(...)`。调度器零改动。
- **替换执行后端**:实现 `Backend`/`State` 接口,经 `Simulator(config, cost, backend)` 注入。

## 与旧仓库的关系(已弃用,保留为只读历史)

- C++ 控制面:同一 schema v2 协议、同一 RNG、同一事件循环;同输入输出逐字段一致。
- GPU 层:模块从旧仓库保真移植并适配新布局(`artifacts`→`util`、`previs_worker`→`gpu.worker` 等);PreViS v1 由 `experiments/previs2_base.py` 内联基类取代,不再迁移;previs_parallel、trace_phase_conflict、figures、一次性脚本与旧 `results/` 归档留在旧仓库。
- 归档格式同一 schema(`ditserve.archive.v1`);二进制产物(.npy/.mp4 等)登记 SHA256 但不嵌入归档。
- 代码量:旧仓库约 48,300 行 → 本仓库约 21,700 行(C++ 1,407、Python 包 12,981、Python 测试 7,278),GPU 能力完整但无 v1/parallel/一次性脚本。
