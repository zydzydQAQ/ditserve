# 20261107-d20261107-formal-v2-rate-rpm-p1-5d9ee3f-r1-rate-12

实验 `formal-wan22-test-poisson-s301-r12rpm-v0p5-slo1p5-sweep`，运行 `d20261107-formal-v2-rate-rpm-p1-5d9ee3f-r1-rate-12`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`5276921cb8873e7fe3ce257e19a81b7ad3ae2235a69b34c5b5b3aad4ff840b91`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.53 | 509138 | 0.0443101 | 0.0443101 |
| cluster / xdit | image | completed | 50 | 0.84 | 15552.2 | 0.0351123 | 0.0351123 |
| cluster / xdit | video | completed | 50 | 0.22 | 591482 | 0.00919607 | 0.00919607 |
| cluster / tetriserve | 全部 | completed | 100 | 0.78 | 515721 | 0.0779956 | 0.0779956 |
| cluster / tetriserve | image | completed | 50 | 0.78 | 18435.4 | 0.0389956 | 0.0389956 |
| cluster / tetriserve | video | completed | 50 | 0.78 | 662592 | 0.0389956 | 0.0389956 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.79 | 496715 | 0.0795183 | 0.0795183 |
| cluster / sbp-esp | image | completed | 50 | 0.78 | 119017 | 0.0392535 | 0.0392535 |
| cluster / sbp-esp | video | completed | 50 | 0.8 | 660617 | 0.04026 | 0.04026 |
| cluster / ditserve | 全部 | completed | 100 | 0.94 | 458522 | 0.0945885 | 0.0945885 |
| cluster / ditserve | image | completed | 50 | 1 | 15593.2 | 0.05031 | 0.05031 |
| cluster / ditserve | video | completed | 50 | 0.88 | 594874 | 0.0442728 | 0.0442728 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.8 | 444533 | 0.0834964 | 0.0834964 |
| cluster / edf-bestfit | image | completed | 50 | 0.96 | 17206.4 | 0.0500946 | 0.0500946 |
| cluster / edf-bestfit | video | completed | 50 | 0.64 | 460004 | 0.0333964 | 0.0333964 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
