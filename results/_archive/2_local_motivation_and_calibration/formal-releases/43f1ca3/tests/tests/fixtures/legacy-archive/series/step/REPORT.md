# Historic CPU fixture: failed request

## 目的与方法

比较固定请求流下各调度策略的参考行为。

使用保存的 workload、成本模型和全部到达请求，比较完成、超时、SLO 与参考状态一致性。

工作簿：[results.xlsx](results.xlsx)。可重建数据：[data.json.gz](data.json.gz)。

Config 表保存逐实验配置；Results 保存逐策略汇总。字段后缀 `_ms`、`_seconds`、`_bytes`、`_mib` 分别表示毫秒、秒、字节、MiB；`goodput_per_sec` 为请求/秒。

## 实际结果

保存 1 条策略/配置结果，参考状态一致性失败计数为 1。时间、资源和 goodput 均来自合成成本；它们不能作为真实 GPU 性能或感知质量证据。

保存 1 份原始小文件、1 条结果记录；另有 1 份二进制产物仅登记哈希、不嵌入归档。
状态计数：`{"failed": 1}`。空白测项表示未知或不适用，不填零。

| case | completed | parity_failures |
| --- | --- | --- |
| edf | 0 | 1 |

## 结论边界与保留策略

本归档保留表格、报告和小型历史记录，不包含原始数组、latent 或完整 trace。合成结果不是真机性能，参考一致性不等于感知质量，未知质量不计入 SLO 成功。

原始 JSON 与报告文本原样存于压缩数据中；重建报告无需 GPU、模型或原始数组。
