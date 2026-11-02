# DiT Serving Lab

C++17 管理实验，Python 保留模型、张量、评测和报告。统一路径为：

```text
C++ CLI / 控制器 → 常驻 Python workers
命令/调度/预算      模型与不透明状态 ID
阶段/监督          评测、摘要/报告
```

源码按运行职责和构建依赖组织，Python 包名仍为 `ditserve`。

```text
src/ditserve/          C++ 生产代码：control/、interface/、runtime/ 和基础类型
include/ditserve/      C++ 生产接口，与 src 模块对应
cmd/                  C++ main
benchmarks/           原生实验配置、评测及独立合成模拟
workers/ditserve/     Python 服务包
  runtime/            协议、环境检查、可选调用客户端
  models/             模型、张量与缓存状态；models/xdit/ 为 xDiT 适配层
  evaluation/         数值质量、数据集及 CUDA/Nsight 指标解析
  results/            关键摘要、日期报告、旧格式读取及依赖清理
workers/xfuser/       固定 xDiT 的 Python 执行包（xFuser），保留上游许可证
tests/                核心单元测试、worker/CLI 集成测试和兼容 fixtures
configs/              当前配置与固定环境依赖
docs/                 使用说明、能力边界及研究文档
```

`cmd/main.cpp` 启动 `src/ditserve/interface/command.cpp` 中的正式命令入口。C++ 基础核心提供类型、预算、调度和进程监督；benchmark 库依赖核心；应用层组合实验控制器、命令和 benchmark，依赖无环。Python 包只承担 worker 服务和可选客户端，`python -m ditserve` 仅转交原生程序。`results/` 是实际实验产物目录，源码位于 `workers/ditserve/results/`。

当前模型为 Wan2.1-T2V-1.3B、Wan2.2-TI2V-5B 和 SD3.5-Medium，xDiT/xFuser 固定于 `16c34e7f070de65d9d2c67a6d74e0007dd8b426a`。Wan 由 DITSERVE 管理 step/block 暂停、scheduler/RNG 与状态迁移；exact 只针对同一 runtime、同一并行形态的不中断对照，不保证跨 SP 逐位一致。SD3.5 提供原子生成和批处理路径；连续批与 token packing 仍需独立 GPU 资格验证。原生 serving 默认每批一个图像请求，增大批次必须显式提供相应实测画像，各策略共享相同上限。

本地编译、CPU/mock/Gloo 测试验证实现契约，不能证明 GPU 加速、CUDA/NCCL 正确性或模型质量；这些结论需要绑定当前代码、模型与硬件的 Kairos 实测。历史结果只读兼容，不自动继承到新实现。

## 构建和验证

Linux、C++17、CMake 3.16+、OpenSSL、Python 3.10+。nlohmann/json 固定 3.11.3 并校验 SHA256；保留 `-ffp-contract=off`。模型依赖按 `configs/requirements-*.txt` 和 `docker/artifacts-runtime.json` 锁定；jjjkairos 使用既有运行镜像，禁止在本机 WSL 或集群节点构建镜像。默认摘要仅依赖 Python 标准库；历史 Excel 读取及显式 full 导出需要 openpyxl 3.1.5。

```bash
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/build -j4
(cd .cache/build && ctest --output-on-failure)
PYTHONPATH=workers CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-previs/bin/python -m unittest discover -s tests/python -v
```

无模型依赖时可以使用 `python3` 跑同一套测试，涉及 torch/transformers/diffusers 的数值用例会跳过，须记录通过数和跳过数。固定模型环境的 CPU 测试仍不替代 GPU 验收。可以用 `DITSERVE_CORE` 指定二进制。

## 统一入口

```bash
.cache/build/ditserve doctor
.cache/build/ditserve simulate --config configs/simulation.json --run-id demo-sim
.cache/build/ditserve run --config configs/studies/mock.json --run-id demo-cpu
.cache/build/ditserve results verify results/cpu-smoke/demo-cpu
.cache/build/ditserve results inspect results/cpu-smoke/demo-cpu
.cache/build/ditserve report .cache/results-work/cpu-smoke/demo-cpu
```

运行原生入口无需设置 `PYTHONPATH`。默认使用项目已有 `.venv-previs/bin/python`，可用 `--python /absolute/path/to/python` 或 `DITSERVE_PYTHON` 指定解释器；`--root` 指定项目/结果根目录，`--package-root` 指定包含 `ditserve/` 的源码目录。`--json` 可放在子命令前后。已安装 Python 包的 `python -m ditserve` 是兼容启动器，不再解析实验参数。

`doctor` 只读环境元数据和 GPU 列表，不加载模型或初始化 CUDA。帮助、版本和 `rpc generate|simulate|control` 完全由 C++ 处理，Python 不可用时仍可运行。低层 stdin/stdout JSON 协议使用 `ditserve rpc <operation>`。`run` 的矩阵、阶段、重试和预算由 C++ 执行；选择阶段使用 `--phase ID`，同一新格式运行继续执行使用 `--resume --run-id ID`。重试新建 attempt，保留失败证据及历史费用。配置、运行身份、账本路径和执行源码发生变化时须创建新运行。

真实 GPU 运行需要先填写模板中的模型路径、已有环境 Python 和完整 GPU UUID，再显式给 `--allow-gpu`。所有登记 GPU 从阶段开始至清理确认均计费，包括模型加载、预热、失败和收尾；独立 wall-clock watchdog 对无限额度同样有效。不要把模板的占位配置直接用于正式实验。

```bash
.cache/build/ditserve run --config /absolute/path/to/study.json \
  --run-id trial-01 --allow-gpu
```

默认交付为 `results/<experiment>/<run>/YYYYMMDD-<run_id>.md` 和 `summary.json`，仅保留关键指标、资格、失败摘要与来源/清理凭据。摘要不能续跑或重新评测。累计账本在 `.state/<experiment>/budget.json`；运行控制记录及未闭合评测依赖保留在 `.cache/results-work/`。新增测试用 `test_YYYYMMDD_<事项>.py` 命名；历史报告与兼容测试不因命名要求重写。

[实验配置](docs/experiments.md) · [结果处理](docs/results.md) · [容器镜像](docs/container.md) · [论文设计与评估协议](docs/dit_serving_sections.tex)
