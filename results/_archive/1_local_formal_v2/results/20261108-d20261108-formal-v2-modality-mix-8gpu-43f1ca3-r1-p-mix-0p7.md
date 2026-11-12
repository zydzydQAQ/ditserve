# 20261108-d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p7

实验 `formal-wan22-test-poisson-s301-r12rpm-v0p7-slo1p5-sweep`，运行 `d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p7`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`323986542c4789d373ab16670f96db4ac5d83bd77b5dc31973a066ee6fd277d2`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.46 | 882304 | 0.0295557 | 0.0295557 |
| cluster / xdit | image | completed | 30 | 0.966667 | 15505.9 | 0.0186321 | 0.0186321 |
| cluster / xdit | video | completed | 70 | 0.242857 | 981769 | 0.0109223 | 0.0109223 |
| cluster / tetriserve | 全部 | completed | 100 | 0.82 | 929668 | 0.0633682 | 0.0633682 |
| cluster / tetriserve | image | completed | 30 | 0.933333 | 16884.6 | 0.0216364 | 0.0216364 |
| cluster / tetriserve | video | completed | 70 | 0.771429 | 944408 | 0.0417273 | 0.0417273 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.8 | 889727 | 0.0616486 | 0.0616486 |
| cluster / sbp-esp | image | completed | 30 | 0.966667 | 17901 | 0.0223462 | 0.0223462 |
| cluster / sbp-esp | video | completed | 70 | 0.728571 | 914587 | 0.0392984 | 0.0392984 |
| cluster / ditserve | 全部 | completed | 100 | 0.87 | 911768 | 0.0647262 | 0.0647262 |
| cluster / ditserve | image | completed | 30 | 1 | 15469.8 | 0.022318 | 0.022318 |
| cluster / ditserve | video | completed | 70 | 0.814286 | 941770 | 0.0424043 | 0.0424043 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.68 | 753740 | 0.0524666 | 0.0524666 |
| cluster / edf-bestfit | image | completed | 30 | 1 | 16166.9 | 0.0231454 | 0.0231454 |
| cluster / edf-bestfit | video | completed | 70 | 0.542857 | 766815 | 0.0293175 | 0.0293175 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
