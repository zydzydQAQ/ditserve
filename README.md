# DiT Serving Lab

C++17 管理实验，Python 保留模型、张量、评测和报告。统一路径为：

```text
C++ CLI / 控制器 → 常驻 Python workers
命令/调度/预算      模型与不透明状态 ID
阶段/监督          评测、Excel/报告
```

源码按运行职责和构建依赖组织，Python 包名仍为 `ditserve`。

```text
src/ditserve/          C++ 生产代码：control/、interface/、runtime/ 和基础类型
include/ditserve/      C++ 生产接口，与 src 模块对应
cmd/                  C++ main
benchmarks/           合成模拟；独立 src/、include/、tests/ 和 golden 数据
workers/ditserve/     Python 服务包
  runtime/            协议、环境检查、可选调用客户端
  models/             模型、张量与缓存状态；models/xdit/ 为 xDiT 适配层
  evaluation/         数值质量、数据集及 wan_profile/ 分析工具
  results/            Excel、报告、旧格式读取及依赖清理
workers/xfuser/       固定 xDiT 的 Python 执行包（xFuser），保留上游许可证
tests/                核心单元测试、worker/CLI 集成测试和兼容 fixtures
configs/              当前配置与固定环境依赖
docs/                 使用说明、能力边界及研究文档
```

`cmd/main.cpp` 启动 `src/ditserve/interface/command.cpp` 中的正式命令入口。C++ 基础核心提供类型、预算、调度和进程监督；benchmark 库依赖核心；应用层组合实验控制器、命令和 benchmark，依赖无环。Python 包只承担 worker 服务和可选客户端，`python -m ditserve` 仅转交原生程序。`results/` 是实际实验产物目录，源码位于 `workers/ditserve/results/`。

当前支持 FLUX、SD3.5、Wan 和 Sana，DiT 执行固定使用 xDiT `16c34e7f070de65d9d2c67a6d74e0007dd8b426a` 的 xFuser。Wan/Sana 的逐步、连续批、scheduler/RNG、迁移与 C++ 调度仍由 DITSERVE 管理；FLUX、SD3.5 与原子 Wan 使用 xDiT pipeline wrapper。SD3.5-medium 使用独立 `sd35` 后端，保留负向 prompt、固定 batch 和单 CPU Generator。Mistral、LongLive 专用后端已删除，通用多阶段 replay 与历史结果只读兼容保留。合成模拟和 CPU mock 结果不能作为 GPU 性能或模型质量结论；真实 GPU 路径以 Kairos 验收记录为准。

## 构建和验证

Linux、C++17、CMake 3.16+、OpenSSL、Python 3.10+。nlohmann/json 固定 3.11.3 并校验 SHA256；保留 `-ffp-contract=off`。模型依赖按 `configs/requirements-*.txt` 和 `docker/artifacts-runtime.json` 锁定；运行镜像由仓库 Dockerfile 重建，报告使用 openpyxl 3.1.5，可复用 `.cache/tools`。

```bash
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release
cmake --build .cache/build -j4
(cd .cache/build && ctest --output-on-failure)
PYTHONPATH=workers CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-previs/bin/python -m unittest discover -s tests/python -v
```

无模型依赖时可以使用 `python3` 跑同一套测试，涉及 torch/transformers/diffusers 的数值用例会跳过；最终验收以固定模型环境的 CPU 测试为准。可以用 `DITSERVE_CORE` 指定二进制。

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

最终交付为 `results/<experiment>/<run>/results.xlsx` 和 `REPORT.md`。累计账本在 `.state/<experiment>/budget.json`；可恢复控制文件和有实际下游依赖的输入在 `.cache/results-work/`。`compact-results` 保留为结果 skill 兼容入口。

[实验配置](docs/experiments.md) · [结果处理](docs/results.md) · [容器镜像](docs/container.md) · [能力覆盖与精简统计](docs/rewrite.md)
