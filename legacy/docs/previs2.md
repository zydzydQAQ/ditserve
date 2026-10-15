# PreViS run02：质量约束下的状态动作验证

本轮使用 `python/ditserve/experiments/previs2_*` 实现。历史 [run01](/home/zyd/code/dit/results/previs/run01) 与 [run02](/home/zyd/code/dit/results/previs/run02/CONCLUSIONS.md) 留在旧仓库，只作只读证据；当前重写后的源码不能续接历史预算或复用历史正确性资格。新实验使用当前仓库独立输出目录。原 worker 的五个接口 `prepare/step/snapshot/restore/finish` 继续复用；测量源码、依赖版本、提示词、种子、形状候选和预算协议在第一次 GPU 运行前归档并绑定 SHA256。

## 实验语义

- 模型固定为 Wan2.1-1.3B；FP16 DiT、FP32 VAE、CPU FP32 文本编码及相同的 MATH attention 路径。
- 采样更新始终执行。缓存仅省略部分 DiT 计算，使用真实 DBCache；连续复用上限固定为 1。
- 首发候选为全算及阈值 0.02、0.04、0.08。每个配置必须具备完整的四个开发提示词、两个种子及分阶段计时。质量控制采用开发样本的最大整段视频平均 LPIPS，阈值 0.05。
- 配置比较共用无缓存文本准备阶段的中位耗时，再加各配置的逐步耗时和解码耗时，避免 CPU 编码波动改变首发选择。
- 恢复标定以完整前缀加后缀轨迹评价质量；覆盖完成第 8、16、24 步以及 keep/hot/cold 和继续缓存/剩余全算的组合。未完成的标定单元不会获得资格。
- 未覆盖位置允许完整 keep/hot 延续已有策略，保留原来的整轨质量预测。未覆盖的 cold 或策略改变被排除；最多一次有损恢复选择。暂停本身不造成 timestep 漂移。
- LPIPS 是相对同输入无缓存参考的经验约束，不是 VBench、人工偏好或逐请求质量保证。测试集不参与首发和恢复表拟合。

## 容量与执行

安全余量取 `max(512 MiB, 5% × 总显存)`。计算下一请求所需空间时，使用实测的增量峰值，并只减去已驻留且已计入 allocated 的下一请求状态一次。权重常驻量不会重复计入。

候选容量估计为驱动空闲量加 allocator 未使用的 reserved 空间，再扣安全余量。hot/cold 可以加上预计释放的请求存储，但不会假定这些字节立即归还驱动。实际复制完成后，分配或恢复下一请求前重新读取容量；不足时记录资源拒绝。完成下一请求的真实分配、计算和解码，才构成可复用容量的运行证据。

首版 snapshot 同步等待传输完成后才替换 GPU 状态。它不实现异步流水线，不声称换出期间显存已经可用。未知字段、版本错误、复制失败、OOM 和未知质量均保留为失败或未知证据。

## 对照与门槛

固定状态组和联合组共享 EDF、首发配置、质量控制、输入、deadline、batch=1 和资源约束。固定组在 keep 可行时保留状态，否则使用开发集选出的固定换出模式；联合组枚举合法状态和恢复后缀。已选择后缀的剩余成本曲线和整轨质量预测继续沿用，不重置质量预算。

双请求实验使用定点到达，资源检查使用真实容量；两个属性分别记录。每种状态模式内都由同一质量控制器选择合法后缀。紧急请求计时包含决策和换出等待。局部恢复成本另报，不替代紧急请求端到端延迟。

只有双请求机制证据完整，且观察到真实容量限制或两个重复中均出现可复现的状态成本交叉，才运行在线强基线对照。成本交叉的保守判据为两次局部恢复成本均低于 keep 的 95%，这是小样本工程门槛，不是统计显著性。完整在线对照的项目扩展门槛另为：同质量约束下 goodput 提升至少 10%，三个 trace 方向一致且可归因到状态决策。

在线 trace 固定为负载 1.2、四次双请求突发、三个种子、每条八个请求。两组配对推进；单个未完成的配对不进入配对提升统计。文本编码计入 E2E；原始输出持久化和离线质量评估在一条 trace 完成后执行，持卡评估时间仍计入 GPU 预算。

## 预算、原始输出和复核

独立预算上限为两卡累计 43,200 秒。阶段累计上限分别为：正确性 0.5 小时、形状探测 1.5 小时、标定 6 小时、机制 7 小时、在线 11 小时、V100 复验 12 小时。加载、预热、失败、清理及持卡评分都收费。预算停止和阶段跳过会保留明确记录；预算未耗尽不意味着应继续扩大实验。

原始 FP32 输出和 latent 保存为 NPY。专用 SSD 上的文件通过单文件符号链接及 `artifact-storage.json` 注册；索引包含完整 NPY 文件哈希，实验记录另外包含数组数据哈希。禁止覆盖已有原始数组，不以预览图代替原始输出。

在项目根目录设置模块路径，先检查计划和启动条件，再运行已通过门槛的阶段。下面的 suite 和数组目录必须使用未占用的新名称，所有阶段沿用完全相同的参数：

```bash
export PYTHONPATH=python
python3 -m ditserve.experiments.run_previs2 --dry-run
python3 -m ditserve.experiments.run_previs2 --preflight --stage correctness --output results/previs/NEW-RUN-ID --history-run01 /home/zyd/code/dit/results/previs/run01 --raw-storage-root /home/zyd/code/dit-rewrite/.cache/runs/NEW-RUN-ID-arrays
.venv-previs/bin/python -m ditserve.experiments.run_previs2 --stage correctness --gpu both --gpu-hours 12 --output results/previs/NEW-RUN-ID --history-run01 /home/zyd/code/dit/results/previs/run01 --raw-storage-root /home/zyd/code/dit-rewrite/.cache/runs/NEW-RUN-ID-arrays
.venv-previs/bin/python -m ditserve.experiments.previs2_report results/previs/NEW-RUN-ID --verify-raw --previews
```

`--dry-run` 仅输出计划，明确列出未检查项，不等于启动验收。`--preflight` 不加载模型、不访问 GPU、不创建输出目录；它检查历史 manifests、固定依赖版本与源码身份、离线模型文件、LPIPS 主干权重、输出/数组路径和已有预算。正常启动自动执行同样的检查。CPU preflight 通过仍不代表 CUDA 可用或正确性通过。

`--history-run01` 未指定时，优先查当前仓库的 `results/previs/run01`，不存在时查同级 `dit/results/previs/run01`；检查从不复制或修改历史结果。已有 suite 绑定历史目录和文件哈希，不能换一份内容相同的目录继续运行。`--python` 指定测量环境，默认当前仓库 `.venv-previs/bin/python`，保留虚拟环境入口的符号链接语义。

固定配置中的历史 `raw_storage_root` 在启动时自动替换为新 suite 的同级 `<输出目录名>-arrays`，例如 `results/previs/NEW-RUN-ID-arrays`；也可使用 `--raw-storage-root` 指定新的真实目录。实际路径进入冻结协议，恢复时必须一致。历史 SSD 数组目录即使已清空也禁止重用；有其他实验数据而没有当前 suite 索引的数组目录同样会被拒绝。离线模型使用当前仓库 `.cache/huggingface/hub` 下固定 revision；需要 LPIPS 的阶段还要求 `.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` 已就绪。preflight 不下载权重，路径检查也不替代完整权重哈希或数值验证。

导出需在 GPU 测量全部停止后执行。报告区分 completed、failed、infeasible、unknown 和 not_run，并核对预算、来源、完整配对和原始数组。预览预先固定为新留出集的 tram/aquarium、种子 201；缺测时不挑选其他样例替代。

修改已冻结测量源码会阻止继续运行。失败重试必须保留原收费、原日志和源版本，不能删除 manifest 或重置预算来重跑。非空 suite 缺少 `manifest.json` 时，preflight 和取得独占锁后的启动检查都会拒绝继续，要求恢复原账本。
