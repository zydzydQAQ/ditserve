# 20261108-d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p9

实验 `formal-wan22-test-poisson-s301-r12rpm-v0p9-slo1p5-sweep`，运行 `d20261108-formal-v2-modality-mix-8gpu-43f1ca3-r1-p-mix-0p9`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`07a068acc169871362694a5096aa47ff3da91f4e8c260326804fc1db4a4c39bd`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.23 | 1.2806e+06 | 0.0115179 | 0.0115179 |
| cluster / xdit | image | completed | 10 | 1 | 15393.9 | 0.00500757 | 0.00500757 |
| cluster / xdit | video | completed | 90 | 0.144444 | 1.31139e+06 | 0.00650984 | 0.00650984 |
| cluster / tetriserve | 全部 | completed | 100 | 0.69 | 1.07543e+06 | 0.0441672 | 0.0441672 |
| cluster / tetriserve | image | completed | 10 | 0.9 | 16222.1 | 0.00576054 | 0.00576054 |
| cluster / tetriserve | video | completed | 90 | 0.666667 | 1.08924e+06 | 0.0384036 | 0.0384036 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.72 | 1.09916e+06 | 0.0465369 | 0.0465369 |
| cluster / sbp-esp | image | completed | 10 | 1 | 16211.8 | 0.00646307 | 0.00646307 |
| cluster / sbp-esp | video | completed | 90 | 0.688889 | 1.0999e+06 | 0.040071 | 0.040071 |
| cluster / ditserve | 全部 | completed | 100 | 0.75 | 1.04932e+06 | 0.0457051 | 0.0457051 |
| cluster / ditserve | image | completed | 10 | 1 | 15478.5 | 0.00609359 | 0.00609359 |
| cluster / ditserve | video | completed | 90 | 0.722222 | 1.05056e+06 | 0.0396083 | 0.0396083 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.5 | 1.02343e+06 | 0.0325946 | 0.0325946 |
| cluster / edf-bestfit | image | completed | 10 | 1 | 16406.8 | 0.00651838 | 0.00651838 |
| cluster / edf-bestfit | video | completed | 90 | 0.444444 | 1.02719e+06 | 0.0260735 | 0.0260735 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
