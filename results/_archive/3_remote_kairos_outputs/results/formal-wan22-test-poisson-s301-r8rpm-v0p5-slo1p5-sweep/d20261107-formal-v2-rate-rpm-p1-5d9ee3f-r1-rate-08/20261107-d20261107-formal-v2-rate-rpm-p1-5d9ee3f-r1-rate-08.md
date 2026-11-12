# 20261107-d20261107-formal-v2-rate-rpm-p1-5d9ee3f-r1-rate-08

实验 `formal-wan22-test-poisson-s301-r8rpm-v0p5-slo1p5-sweep`，运行 `d20261107-formal-v2-rate-rpm-p1-5d9ee3f-r1-rate-08`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`c2ec6c274def46723dcff6c61b49fcea38f78540b2b4ea60c9f29b022649aa28`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.7 | 329403 | 0.0565943 | 0.0565943 |
| cluster / xdit | image | completed | 50 | 0.92 | 15501.6 | 0.0371891 | 0.0371891 |
| cluster / xdit | video | completed | 50 | 0.48 | 386988 | 0.019403 | 0.019403 |
| cluster / tetriserve | 全部 | completed | 100 | 0.88 | 298033 | 0.0823871 | 0.0823871 |
| cluster / tetriserve | image | completed | 50 | 0.82 | 19745.1 | 0.0383829 | 0.0383829 |
| cluster / tetriserve | video | completed | 50 | 0.94 | 308228 | 0.0439999 | 0.0439999 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.89 | 283788 | 0.0848154 | 0.0848154 |
| cluster / sbp-esp | image | completed | 50 | 0.86 | 19667.5 | 0.0409758 | 0.0409758 |
| cluster / sbp-esp | video | completed | 50 | 0.92 | 297771 | 0.0438346 | 0.0438346 |
| cluster / ditserve | 全部 | completed | 100 | 1 | 254745 | 0.0963539 | 0.0963539 |
| cluster / ditserve | image | completed | 50 | 1 | 15511.4 | 0.0481743 | 0.0481743 |
| cluster / ditserve | video | completed | 50 | 1 | 275521 | 0.0481743 | 0.0481743 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.96 | 315129 | 0.0903829 | 0.0903829 |
| cluster / edf-bestfit | image | completed | 50 | 0.94 | 16580.4 | 0.0442475 | 0.0442475 |
| cluster / edf-bestfit | video | completed | 50 | 0.98 | 319831 | 0.0461304 | 0.0461304 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
