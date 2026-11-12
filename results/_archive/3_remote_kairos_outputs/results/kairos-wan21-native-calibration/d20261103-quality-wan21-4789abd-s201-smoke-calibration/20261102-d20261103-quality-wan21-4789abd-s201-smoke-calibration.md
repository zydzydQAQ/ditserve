# 20261102-d20261103-quality-wan21-4789abd-s201-smoke-calibration

实验 `kairos-wan21-native-calibration`，运行 `d20261103-quality-wan21-4789abd-s201-smoke-calibration`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`d2bc849b58cba56182c86770779b51b7b9d3917f8aa429446ff070bde44aee72`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 38431.6 | 0.0498653 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6578.68 | 0.0249323 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 40108.1 | 0.0249323 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 27831.6 | 0.0690853 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6588.04 | 0.0345422 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 28949.7 | 0.0345422 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 39506.2 | 0.048498 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6586.64 | 0.0242488 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 41238.8 | 0.0242488 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 33856.5 | 0.0566672 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 6547.26 | 0.0283331 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 35293.8 | 0.0283331 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 37944.1 | 0.0505138 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 6613.22 | 0.0252567 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 39593.1 | 0.0252567 | 未验证 |
| cluster / static_cfg2sp3 | 全部 | completed | 2 | 1 | 32345 | 0.0593489 | 未验证 |
| cluster / static_cfg2sp3 | image | completed | 1 | 1 | 6617.44 | 0.0296739 | 未验证 |
| cluster / static_cfg2sp3 | video | completed | 1 | 1 | 33699 | 0.0296739 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 39324.1 | 0.0487277 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 6638.22 | 0.0243637 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 41044.4 | 0.0243637 | 未验证 |
| cluster / static_sp6 | 全部 | completed | 2 | 1 | 34154.8 | 0.056172 | 未验证 |
| cluster / static_sp6 | image | completed | 1 | 1 | 6601.23 | 0.0280855 | 未验证 |
| cluster / static_sp6 | video | completed | 1 | 1 | 35605 | 0.0280855 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
