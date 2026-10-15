# 缓存与 K4 在线对照续跑

本入口只补 `{off, paired_008} × {step, K4}` 在线实验。用户已明确授权“取消8小时GPU小时硬上限，GPU无上限，开始在线对照”。本次不设 GPU 小时上限，仍完整记录用量；原来的实验及账本保持只读。准备和 CPU 检查不构成 GPU 实验结果。

## 固定实验

- Sana-Sprint-0.6B：1024²、2 步、BF16、eager、batch=1。
- Wan2.1-T2V-1.3B：480×832、81 帧、32 步、Flash、eager；VAE eager，CPU FP32 文本编码。
- 缓存固定为 `paired_008`，阈值 0.08，最大连续复用 1；不修改原质量门槛。
- 使用原 `seed102/load0.5/burst` 与 `seed101/load0.5/ordinary` 的完整请求列表；到达后才读取并编码 prompt。
- 每条 trace 按 `off_step → off_k4 → paired_step → paired_k4` 和反序执行，各 24 个请求。完整矩阵为 16 次回放、384 个请求；每组 96 个请求，包括 88 个图像与 8 个视频。
- 四组共享 EDF、到达时间和 deadline；K4 可跨 CFG 分支，每量子最多推进 4 个实际 block。暂停不回滚共享随机数或 manager。

## 来源与预热

复用 `results/optimization-screen/run01/cache_k4/attempt00` 已完成的本地资格：20 组质量配对、正反序加速检查、96 次逐步恢复对照、9 次真实 Sana 插入。续跑前绑定并验证原结果、配置、账本、代码快照及当前 26 个数学依赖文件的哈希。新报告明确记为引用旧证据，新增本地资格样本数为零。

新进程重新加载权重、运行完整无缓存共驻检查、K4 与 paired full/reuse 路径预热和真实解码。每次回放前对称运行两次 Sana 预热。这些时间全部计费，测量时钟在对应预热完成后启动。

## 独立预算与执行

新结果目录为 `results/optimization-cache-online/run01`，新账本为 `results/optimization-cache-online/gpu-budget.json`。两个旧账本只作为不可变来源。配置中保留的旧 8 小时字段用于证明科学配置一致；本次有效额度由单独的用户授权文件和新账本确定，不转入旧预算余额。

完整固定到达窗口下限约 2.12 小时；包含服务时间裕量、加载、预热与清理的启动估计为 2.54 GPU 小时。该值仅为耗时估计；当前授权没有 GPU 小时截止时间。实际时间以新账本为准。

```bash
PYTHONPATH=python python3 -m ditserve optimize-serving cache-online inspect
PYTHONPATH=python python3 -m ditserve optimize-serving cache-online run \
  --authorization /absolute/path/to/user-authorized-budget.json \
  --wait-idle-seconds 7200
PYTHONPATH=python python3 -m ditserve optimize-serving cache-online report
```

授权文件包含 `approved_by_user: true`、真实的 `user_instruction` 和 `scope: "cache_online_factorial_only"`。当前无上限授权显式使用 `budget_unlimited: true` 与 `authorized_gpu_seconds: null`，账本也记录 null 上限，不向 JSON 写入 Infinity。有限额度模式仍要求正数 GPU 秒数。

运行前和运行中检查 A100 独占状态；V100 不参与本实验。等待空闲期间不加载模型。已占卡的加载、预热、等待、失败和清理均计费。仅清理本次 worker 进程组；清理无法确认时，账本保持活动状态，阻止盲目重试。失败重试需要已结算且完成清理的同配置来源。取消小时上限不会取消父进程存活检查、GPU 隔离或失败后的资格限制。

## 验收和产物

报告同时给出抢占效果、step 下的缓存效果、K4 下的缓存效果、缓存下的抢占效果。晋级判断固定比较 `off_k4` 与 `paired_k4`：goodput 提升至少 5% 或图像端到端 p95 降低至少 10%，同时满足每条 trace 的 goodput、图像 p95 和视频按时完成率保护门槛。只有局部加速不能晋级。

图像 p95 从逐请求数据合并计算，不平均各行 p95。失败和未完成请求保留，部分矩阵单独标记，不能被当成完整结果。日志记录开销按组报告，不从耗时中扣除。只报告均值、p95 与样本数，不宣称稳定 p99 或统计显著性。

新报告和 Excel 保存配置、指标、失败、来源哈希及预算。事件正文与旧原始数组不复制进 Excel。原始日志在归因及续跑依赖结束前保留；后续清理需先导出并回读校验，再按明确清单记录删除证据。旧 D1 或旧缓存阶段结论不会被此次结果改写。

清理仅允许本轮完成后的 `cache-online.json`、`partial.json` 和 `isolation.jsonl`，并须绑定最终独立审计、主 Excel、日志摘要归档及删除回执。清理后的 `report` 从已核验归档读取冻结指标，明确显示 `archive_only`，不重新评测原事件或重新授予资格；账本、授权、配置、终局监督记录和来源快照仍逐项校验。任一来源变化、回执缺失或账本重新活动时拒绝读取该归档结论。旧原始数组不在本轮清理范围内。
