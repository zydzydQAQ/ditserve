# 20261103-d20261103-formal-calibration-wan22-b97479e-r1-cal1

实验 `kairos-wan22-native-calibration`，运行 `d20261103-formal-calibration-wan22-b97479e-r1-cal1`：执行 `completed`，归档 `completed`。

[关键摘要](summary.json) 保存来源/配置哈希、各策略指标、资格、失败摘要与清理凭据。
completed 仅表示执行完成；缺失不是零，未验证不是通过。摘要不能用于续跑或原始重评。

证据范围：supervised GPU experiment。

配置 SHA256：`ff2e3395eb94e8cca5a11512c6a4039b9a8d71f447f3936c9456232a83a6080a`。

| 阶段 / 策略 | 模态 | 状态 | 请求数 | SLO 达成率 | p95 延迟 ms | 按时 goodput req/s | 资格通过 goodput req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cluster / static_sp1 | 全部 | completed | 2 | 1 | 445772 | 0.00426552 | 未验证 |
| cluster / static_sp1 | image | completed | 1 | 1 | 6795.75 | 0.00213276 | 未验证 |
| cluster / static_sp1 | video | completed | 1 | 1 | 468876 | 0.00213276 | 未验证 |
| cluster / static_cfg2 | 全部 | completed | 2 | 1 | 256495 | 0.00741733 | 未验证 |
| cluster / static_cfg2 | image | completed | 1 | 1 | 6749.84 | 0.00370865 | 未验证 |
| cluster / static_cfg2 | video | completed | 1 | 1 | 269639 | 0.00370865 | 未验证 |
| cluster / static_sp2 | 全部 | completed | 2 | 1 | 351879 | 0.00540475 | 未验证 |
| cluster / static_sp2 | image | completed | 1 | 1 | 6733.41 | 0.00270237 | 未验证 |
| cluster / static_sp2 | video | completed | 1 | 1 | 370045 | 0.00270237 | 未验证 |
| cluster / static_cfg2sp2 | 全部 | completed | 2 | 1 | 237269 | 0.00801922 | 未验证 |
| cluster / static_cfg2sp2 | image | completed | 1 | 1 | 6755.54 | 0.00400959 | 未验证 |
| cluster / static_cfg2sp2 | video | completed | 1 | 1 | 249401 | 0.00400959 | 未验证 |
| cluster / static_sp3 | 全部 | completed | 2 | 1 | 304208 | 0.00625266 | 未验证 |
| cluster / static_sp3 | image | completed | 1 | 1 | 6749.79 | 0.00312632 | 未验证 |
| cluster / static_sp3 | video | completed | 1 | 1 | 319864 | 0.00312632 | 未验证 |
| cluster / static_sp4 | 全部 | completed | 2 | 1 | 292883 | 0.00649476 | 未验证 |
| cluster / static_sp4 | image | completed | 1 | 1 | 6794.56 | 0.00324737 | 未验证 |
| cluster / static_sp4 | video | completed | 1 | 1 | 307941 | 0.00324737 | 未验证 |

SLO 达成率为 0–1 比例。按时 goodput 只统计完成且按时的请求；资格通过 goodput 还要求已配置资格通过。
二者都使用所记录的观测窗口，有限 trace 的值不代表稳定容量；资格缺失不能用按时 goodput 代替。

阶段 `cluster` 资格：numeric=unverified, semantic=unverified, performance=unverified。

清理：确认删除 0 个文件，回收 0 字节，保留 0 个文件。
运行原始记录及未闭合依赖仍保留在工作目录；此摘要不包含逐请求、事件、采样序列或源码。
