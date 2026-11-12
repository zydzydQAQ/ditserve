# 20261107-d20261107-formal-v2-rate-rpm-p1-5d9ee3f-r1-rate-16

实验 `formal-wan22-test-poisson-s301-r16rpm-v0p5-slo1p5-sweep`，运行 `d20261107-formal-v2-rate-rpm-p1-5d9ee3f-r1-rate-16`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`e01b6cda916b63e237b4b1941ff40217a6bce2f96f532eac7292a11ba7709b48`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / xdit | 全部 | completed | 100 | 0.45 | 614259 | 0.0377055 | 0.0377055 |
| cluster / xdit | image | completed | 50 | 0.72 | 18157.3 | 0.0301631 | 0.0301631 |
| cluster / xdit | video | completed | 50 | 0.18 | 682710 | 0.00754078 | 0.00754078 |
| cluster / tetriserve | 全部 | completed | 100 | 0.77 | 678174 | 0.0808457 | 0.0808457 |
| cluster / tetriserve | image | completed | 50 | 0.8 | 20370.1 | 0.0419947 | 0.0419947 |
| cluster / tetriserve | video | completed | 50 | 0.74 | 751986 | 0.0388451 | 0.0388451 |
| cluster / sbp-esp | 全部 | completed | 100 | 0.8 | 666596 | 0.0855865 | 0.0855865 |
| cluster / sbp-esp | image | completed | 50 | 0.8 | 44326.5 | 0.0427907 | 0.0427907 |
| cluster / sbp-esp | video | completed | 50 | 0.8 | 758416 | 0.0427907 | 0.0427907 |
| cluster / ditserve | 全部 | completed | 100 | 0.91 | 606067 | 0.0931896 | 0.0931896 |
| cluster / ditserve | image | completed | 50 | 1 | 15572.8 | 0.0511998 | 0.0511998 |
| cluster / ditserve | video | completed | 50 | 0.82 | 701539 | 0.0419838 | 0.0419838 |
| cluster / edf-bestfit | 全部 | completed | 100 | 0.76 | 553674 | 0.0775101 | 0.0775101 |
| cluster / edf-bestfit | image | completed | 50 | 1 | 15833.8 | 0.0509896 | 0.0509896 |
| cluster / edf-bestfit | video | completed | 50 | 0.52 | 589197 | 0.0265146 | 0.0265146 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
