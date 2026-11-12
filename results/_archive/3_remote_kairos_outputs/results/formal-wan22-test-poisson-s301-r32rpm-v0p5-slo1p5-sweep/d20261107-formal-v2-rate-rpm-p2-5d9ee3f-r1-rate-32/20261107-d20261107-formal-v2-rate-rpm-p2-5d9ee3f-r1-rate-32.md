# 20261107-d20261107-formal-v2-rate-rpm-p2-5d9ee3f-r1-rate-32

实验 `formal-wan22-test-poisson-s301-r32rpm-v0p5-slo1p5-sweep`，运行 `d20261107-formal-v2-rate-rpm-p2-5d9ee3f-r1-rate-32`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`39127f1f30e84c1e1a2ef488c8d04ee32835223b1ad8923aea08d1148527f50f`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.21 | 760259 | 0.0181034 | 0.0181034 |
| cluster / xdit | image | completed | 50 | 0.22 | 48938.3 | 0.00948237 | 0.00948237 |
| cluster / xdit | video | completed | 50 | 0.2 | 848219 | 0.00862034 | 0.00862034 |
| cluster / tetriserve | 全部 | completed | 100 | 0.71 | 721682 | 0.0758112 | 0.0758112 |
| cluster / tetriserve | image | completed | 50 | 0.86 | 18827.7 | 0.0459103 | 0.0459103 |
| cluster / tetriserve | video | completed | 50 | 0.56 | 740944 | 0.0298951 | 0.0298951 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.76 | 708010 | 0.0790711 | 0.0790711 |
| cluster / sbp-esp | image | completed | 50 | 0.92 | 224458 | 0.0478557 | 0.0478557 |
| cluster / sbp-esp | video | completed | 50 | 0.6 | 741318 | 0.0312103 | 0.0312103 |
| cluster / ditserve | 全部 | completed | 100 | 0.8 | 647959 | 0.0851202 | 0.0851202 |
| cluster / ditserve | image | completed | 50 | 0.98 | 15585.2 | 0.0521325 | 0.0521325 |
| cluster / ditserve | video | completed | 50 | 0.62 | 695817 | 0.0329818 | 0.0329818 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.69 | 706028 | 0.0713557 | 0.0713557 |
| cluster / edf-bestfit | image | completed | 50 | 1 | 15826.1 | 0.0517032 | 0.0517032 |
| cluster / edf-bestfit | video | completed | 50 | 0.38 | 729070 | 0.0196472 | 0.0196472 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
