# 20261112-d20261112-formal-v2-abl-cfg2-g2-8gpu-8c50c7a-r1-gamma2-abl-cfg2

实验 `formal-wan22-test-gamma-s301-l1p75-v0p5-slo1p5-cv2p0-ablation`，运行 `d20261112-formal-v2-abl-cfg2-g2-8gpu-8c50c7a-r1-gamma2-abl-cfg2`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`4c81434c3fd47ceadedd253c768aff5044d5a32eb2a1159fe683ed9426b9e598`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / tetriserve-cfg2 | 全部 | completed | 100 | 0.72 | 489726 | 0.0801049 | 0.0801049 |
| cluster / tetriserve-cfg2 | image | completed | 50 | 0.74 | 112304 | 0.0411625 | 0.0411625 |
| cluster / tetriserve-cfg2 | video | completed | 50 | 0.7 | 719430 | 0.0389375 | 0.0389375 |
| cluster / edf-bestfit-cfg2 | 全部 | completed | 100 | 0.65 | 464317 | 0.0714644 | 0.0714644 |
| cluster / edf-bestfit-cfg2 | image | completed | 50 | 0.88 | 16464.7 | 0.0483725 | 0.0483725 |
| cluster / edf-bestfit-cfg2 | video | completed | 50 | 0.42 | 472175 | 0.0230869 | 0.0230869 |
| cluster / sbp-esp-cfg2 | 全部 | completed | 100 | 0.76 | 517781 | 0.0837123 | 0.0837123 |
| cluster / sbp-esp-cfg2 | image | completed | 50 | 0.74 | 20207.7 | 0.0407522 | 0.0407522 |
| cluster / sbp-esp-cfg2 | video | completed | 50 | 0.78 | 767712 | 0.042955 | 0.042955 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
