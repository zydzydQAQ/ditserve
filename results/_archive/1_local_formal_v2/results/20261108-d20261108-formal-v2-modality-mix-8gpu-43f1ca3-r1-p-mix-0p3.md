# 20261108-d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p3

实验 `formal-wan22-test-poisson-s301-r12rpm-v0p3-slo1p5-sweep`，运行 `d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p3`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`c45ec6d7de3b31bdaf123dec0ca6364067b4058a7c5bc2cc6cb15e919d90a52c`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.74 | 289968 | 0.0839068 | 0.0839068 |
| cluster / xdit | image | completed | 70 | 0.785714 | 22264.7 | 0.0623611 | 0.0623611 |
| cluster / xdit | video | completed | 30 | 0.633333 | 310056 | 0.0215429 | 0.0215429 |
| cluster / tetriserve | 全部 | completed | 100 | 0.8 | 292322 | 0.0962907 | 0.0962907 |
| cluster / tetriserve | image | completed | 70 | 0.771429 | 55741.5 | 0.0649935 | 0.0649935 |
| cluster / tetriserve | video | completed | 30 | 0.866667 | 358565 | 0.0312932 | 0.0312932 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.92 | 301172 | 0.118022 | 0.118022 |
| cluster / sbp-esp | image | completed | 70 | 0.928571 | 31394.8 | 0.0833812 | 0.0833812 |
| cluster / sbp-esp | video | completed | 30 | 0.9 | 334501 | 0.0346353 | 0.0346353 |
| cluster / ditserve | 全部 | completed | 100 | 0.99 | 245106 | 0.124609 | 0.124609 |
| cluster / ditserve | image | completed | 70 | 1 | 15545.9 | 0.0881036 | 0.0881036 |
| cluster / ditserve | video | completed | 30 | 0.966667 | 317294 | 0.0365001 | 0.0365001 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.96 | 317900 | 0.120673 | 0.120673 |
| cluster / edf-bestfit | image | completed | 70 | 0.971429 | 16093.6 | 0.0854719 | 0.0854719 |
| cluster / edf-bestfit | video | completed | 30 | 0.933333 | 364273 | 0.0351943 | 0.0351943 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
