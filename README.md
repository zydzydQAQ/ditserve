# DiT Serving Lab

C++17 调度控制面与 Python GPU 执行层，用于 Wan/Sana 缓存、抢占、状态恢复和会话调度研究。当前交付面向单机 8 张 A100；环境构建和验收从 [MIGRATION.md](MIGRATION.md) 开始，实际已执行与未验证项目见 [VALIDATION.md](VALIDATION.md)。

本仓库包含可直接构建的迁移源码、Dockerfile 和固定依赖/模型清单。克隆 `https://github.com/zydzydQAQ/ditserve.git` 后按迁移文档执行；模型通过下载器准备，历史结果和源机运行状态另行交付，不是构建环境的前提。

默认镜像保留当前研究主线，模型与结果放在宿主挂载卷。主环境固定 Python 3.10、PyTorch 2.10.0+cu126、diffusers 0.36.0、transformers 4.57.6、cache-dit 1.5.0；完整依赖 wheel 和模型文件均有 SHA256 清单。Mistral 按需下载，LongLive 使用可选镜像内的独立解释器。

| 目录 | 职责 |
| --- | --- |
| `src/`、`include/`、`cmd/` | C++ 参考控制面，schema v2 JSON 子进程接口 |
| `python/ditserve/gpu/` | Wan 执行、缓存状态、profiling 和质量工具 |
| `python/ditserve/optimizations/`、`directions/` | 当前优化与三项研究方向，保留来源和预算守卫 |
| `python/ditserve/scheduling/`、`sessions/` | 调度与真实 LLM→图像会话；历史监督入口保留原资格约束 |
| `python/ditserve/research/` | 共享状态、摘要、预算、占用守卫和结果读取辅助代码 |
| `python/ditserve/migration/` | 环境盘点、模型准备、便携配置、CPU/单卡/八卡验收和历史导出 |
| `configs/`、`tests/` | 锁定清单、协议配置、C++ 与 CPU 合同测试 |
| `legacy/` | 旧 V100/PreViS、benchmark、异构迁移实验原件及哈希清单，默认镜像排除 |

CLI 使用统一惰性注册表；查看帮助不加载模型或初始化 CUDA：

```bash
PYTHONPATH=python python3 -m ditserve --help
PYTHONPATH=python python3 -m ditserve migrate --help
```

当前入口包括 `migrate`、`optimize-serving`、`research-directions`、`research-scheduling`、`research-scheduling-next`、`research-sessions`，以及 `doctor`、`simulate`、`sweep`、`profile`、`archive`、`compact-results`。后两项研究继续入口保留历史 V100 绑定；A100 环境验收使用 `migrate`，不继承旧资格或 GPU 预算。历史执行命令的恢复方式见 [legacy/README.md](legacy/README.md)。

源码同时保留最新 `ditserve.optimizations.followup_resume`、`followup_combo_retry` 和 `followup_combo_retry_report`。这些入口服务于已有优化实验的续跑和报告，严格依赖原 `.cache/results-work` 中的冻结源码、队列、账本和授权；不能在全新环境直接用于恢复旧实验。对应 3 个单元测试模块包含在当前交付中。

构建 C++ 控制面需要 CMake 3.16+、C++17 编译器和 OpenSSL 开发头文件；随源码交付的 nlohmann/json 3.11.3 头文件经固定哈希校验，可离线构建：

```bash
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build .cache/build --parallel 2
(cd .cache/build && ctest --output-on-failure)
PYTHONPATH=python python3 -m ditserve doctor
```

完整 CPU 测试使用 Docker 内的锁定环境，命令见迁移文档。研究协议说明见 [优化](docs/serving_optimizations_v2.md)、[研究方向](docs/research_directions.md)、[调度](docs/research_scheduling.md)、[会话](docs/research_sessions.md)。新实验长期交付 Excel 工作簿和报告；原始输入在评测、盲评、续跑或调查依赖关闭后才按明确文件清单清理，约束见 [结果保存说明](docs/result_storage.md)。

合成参考结果、环境冒烟和八卡 NCCL 通过分别只说明对应检查通过，不构成 GPU 性能结论、科学资格或 SP8 推理实现。
