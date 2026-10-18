# 设计与精简记录

## 保留的设计

| 设计 | 位置 | 证据 |
| --- | --- | --- |
| block 级（step 内）抢占 | `models/wan/block.py`、`control/cluster.cpp` 的 `block_elastic` | E3 单卡按时率：static_sp1 0.73/0.67/0.50，block 0.93/0.80/0.83（负载 0.4/0.6/0.8） |
| block 边界状态导出/恢复 | `block.py` 的 `export_block_state`/`import_block_state`；`parallel.py` 的请求 wire codec | E5：6 种形状 × 两个 CFG 分支，在新进程中恢复后逐位一致 |
| 可抢占的分块 VAE 解码 | `models/pipeline.py` 的 `ChunkedDecode`；`finish` 的 `decode_frames` | E7 突发 trace 按时数：block 7→10、static 5→8；逐位一致 |
| 可行优先 EDF + `late_short_ms` | `cluster_plan` | E9：图像 p95 从约 90s 降到 1.75s，block 按时数 10→11/24 |
| 时间目标 quantum（`quantum_ms`） | `cluster_plan` | 不同 GPU 的 block 耗时不同，用毫秒目标保持抢占延迟可比 |
| CPU 文本编码与预取 | `wan/runtime.py` 的 `encode_prompt`；cluster 的编码亲和 | 端到端延迟 −20%~−26%（serving-optimization） |
| 对齐的弹性 SP 组（2/4/8） | `wan/parallel.py`、`wan/sp.py`、`cluster_plan` | 只有模拟（E4）；真实执行器固定 SP1，`sp_equivalence` 是第一次 GPU 验证 |
| 共用调度的模拟器 | `benchmarks/src/cluster_sim.cpp` | E3 校准：平均误差 6.7 个百分点；精简前后 E4 重算的 144 个 cell 全部一致 |
| 环境绑定 | `control/environment.cpp`、`configs/env/` | 同一配置可在 1×A100、2×4090、8×A100 上解析（`tests/python/test_configs.py`） |

## 删除的设计（2026-10-18）

精简前最后一个完整版本是 `1c6282a`，在 GitHub 的 `8a100` 分支历史中，需要时可以从那里取回。

| 删除项 | 原因 |
| --- | --- |
| DBCache 请求级特征缓存、CFG 协同缓存 | 在 K4 抢占之上再加缓存，goodput 只变化 −0.2%（optimization-cache-online）；属于有损近似，不进主结论 |
| step 边界 hot/cold offload、`state` 执行器、`resident_limit` | K=16 时仍没有准入收益，也未观察到 state crossover；E2/E8 测得 offload 只需毫秒级 |
| 异构迁移（V100→A100 两阶段所有权） | V100 已不再使用；迁移收益实验从未开展 |
| 异步 VAE 解码 | 主配置使用 Flash，它跑不起来（只支持 MATH）；功能被分块解码取代 |
| 单卡 `online` replay 与 14 个 dispatch 策略 | 单 slot 的 `cluster` 已经覆盖；lazy 抢占在 E3 中没有收益 |
| Sana compile/CUDA graph/batch 候选、Wan torch.compile | 本地加速快 14%，但在线 goodput −1.7%；Sana 侧优化不在研究主线上 |
| Mistral 会话与共置 | 按用户选择删除；E8 结论保留在历史报告里 |
| LongLive 流式、FLUX、离线 Wan batch | 不在主线上；LongLive 还需要单独的 torch 2.8 环境 |
| VBench/数据集/LPIPS 评测 | 主线只依赖逐位或容差内的数值一致；镜像随之去掉 lpips、scipy 和 matplotlib |
| Wan profile / Nsight 工具链 | 对应研究已完成，结果在 `results/wan-profile-corrected/` |
| 单机绑定的 A100 配置（`local-a100/`、`target-8xa100/`） | 冻结配置都保存在各自的 `results.xlsx` 里；可移植的实验见 `configs/studies/` |

旧格式结果的读取器（`results/legacy.py`、`results/compact.py`）只保留在仓库中，供本机历史结果和 save-experiment-results 技能使用；镜像不包含它们。

## 规模

| | 精简前（`1c6282a`） | 精简后 |
| --- | --- | --- |
| Python worker | 12,317 行 | 4,828 行（不计旧格式读取器为 3,860 行） |
| C++（src/include/cmd） | 3,184 行 | 3,137 行（其中新增环境绑定约 230 行） |
| Python 测试 | 5,059 行 / 215 个 | 2,347 行 / 95 个 |
| JSON 配置 | 21,420 行 | 4,793 行 |
| 镜像依赖 | 73 个 wheel | 63 个 |
