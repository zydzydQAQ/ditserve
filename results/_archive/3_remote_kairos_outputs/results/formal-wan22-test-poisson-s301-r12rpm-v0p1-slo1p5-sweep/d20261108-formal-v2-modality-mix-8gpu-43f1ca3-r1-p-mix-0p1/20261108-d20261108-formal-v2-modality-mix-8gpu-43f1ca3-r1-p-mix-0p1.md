# 20261108-d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p1

实验 `formal-wan22-test-poisson-s301-r12rpm-v0p1-slo1p5-sweep`，运行 `d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p1`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`c70ca0d252f2381b86dd9729948aa8d7fed0032d72f666b69aed352d7b20029d`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.98 | 102843 | 0.13277 | 0.13277 |
| cluster / xdit | image | completed | 90 | 0.977778 | 15397.8 | 0.11922 | 0.11922 |
| cluster / xdit | video | completed | 10 | 1 | 247780 | 0.0135477 | 0.0135477 |
| cluster / tetriserve | 全部 | completed | 100 | 0.94 | 104437 | 0.146077 | 0.146077 |
| cluster / tetriserve | image | completed | 90 | 0.933333 | 15500.9 | 0.130533 | 0.130533 |
| cluster / tetriserve | video | completed | 10 | 1 | 222202 | 0.0155396 | 0.0155396 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.89 | 102633 | 0.138765 | 0.138765 |
| cluster / sbp-esp | image | completed | 90 | 0.877778 | 21169.9 | 0.123169 | 0.123169 |
| cluster / sbp-esp | video | completed | 10 | 1 | 213193 | 0.015591 | 0.015591 |
| cluster / ditserve | 全部 | completed | 100 | 1 | 57478.4 | 0.159461 | 0.159461 |
| cluster / ditserve | image | completed | 90 | 1 | 15395.8 | 0.143512 | 0.143512 |
| cluster / ditserve | video | completed | 10 | 1 | 146852 | 0.0159458 | 0.0159458 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.94 | 91678.3 | 0.146315 | 0.146315 |
| cluster / edf-bestfit | image | completed | 90 | 0.933333 | 16549.4 | 0.130746 | 0.130746 |
| cluster / edf-bestfit | video | completed | 10 | 1 | 218654 | 0.015565 | 0.015565 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
