# 20261110-d20261110-formal-v2-cb-g2-ablation-8gpu-1b54f37-r1-gamma2-abl

实验 `formal-wan22-test-gamma-s301-l1p75-v0p5-slo1p5-cv2p0-ablation`，运行 `d20261110-formal-v2-cb-g2-ablation-8gpu-1b54f37-r1-gamma2-abl`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`80302d5522c42f1a869a7c5fafee8d1bfeacf4ba35b82db2d5615924a492b5d4`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / edf-cfg2 | 全部 | completed | 100 | 0.64 | 427438 | 0.068621 | 0.068621 |
| cluster / edf-cfg2 | image | completed | 50 | 0.46 | 34325.6 | 0.0246592 | 0.0246592 |
| cluster / edf-cfg2 | video | completed | 50 | 0.82 | 778779 | 0.0439577 | 0.0439577 |
| cluster / ditserve | 全部 | completed | 100 | 0.91 | 738349 | 0.0947984 | 0.0947984 |
| cluster / ditserve | image | completed | 50 | 0.96 | 15087.2 | 0.05 | 0.05 |
| cluster / ditserve | video | completed | 50 | 0.86 | 776288 | 0.0447916 | 0.0447916 |
| cluster / block-cfg2 | 全部 | completed | 100 | 0.78 | 605782 | 0.0848539 | 0.0848539 |
| cluster / block-cfg2 | image | completed | 50 | 0.68 | 21546.8 | 0.0369851 | 0.0369851 |
| cluster / block-cfg2 | video | completed | 50 | 0.88 | 762561 | 0.0478631 | 0.0478631 |
| cluster / abl-no-block | 全部 | completed | 100 | 0.78 | 397521 | 0.0846439 | 0.0846439 |
| cluster / abl-no-block | image | completed | 50 | 0.68 | 21790.4 | 0.0368932 | 0.0368932 |
| cluster / abl-no-block | video | completed | 50 | 0.88 | 749885 | 0.0477441 | 0.0477441 |
| cluster / abl-no-cb | 全部 | completed | 100 | 0.93 | 745358 | 0.0976004 | 0.0976004 |
| cluster / abl-no-cb | image | completed | 50 | 1 | 15565.3 | 0.0524701 | 0.0524701 |
| cluster / abl-no-cb | video | completed | 50 | 0.86 | 778864 | 0.0451242 | 0.0451242 |
| cluster / abl-first-fit | 全部 | completed | 100 | 0.71 | 399996 | 0.0748239 | 0.0748239 |
| cluster / abl-first-fit | image | completed | 50 | 0.54 | 187976 | 0.028452 | 0.028452 |
| cluster / abl-first-fit | video | completed | 50 | 0.88 | 732157 | 0.0463662 | 0.0463662 |
| cluster / abl-no-prune | 全部 | completed | 100 | 0.87 | 853712 | 0.0717473 | 0.0717473 |
| cluster / abl-no-prune | image | completed | 50 | 0.98 | 15289.8 | 0.0404066 | 0.0404066 |
| cluster / abl-no-prune | video | completed | 50 | 0.76 | 919816 | 0.0313358 | 0.0313358 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=verified_lossless_exact, semantic=verified_bitwise_invariant, performance=verified_real_gpu。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
