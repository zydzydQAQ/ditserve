# CFG 协调缓存预实验

本轮是新的研究协议 `configs/research_cfg.json`。历史双卡协议和 run03 的失败质量结论保持原样；新运行不能继承其缓存、后端成本或质量资格。目标是先验证 CFG 协调缓存的质量和成本潜力，再决定是否开展 V100→A100 迁移机制实验。这里描述执行规则，不代表实验已完成或策略已合格。

## 固定输入与策略

模型仅从 `/mnt/data/zyd-preempt-phase16-01a1296d/dit-runtime/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b` 离线加载。模型清单、文件内容 SHA256、依赖与源码随协议封印；不下载、不回退、不复制模型。规格固定为 256×448×17、32 步、CFG=5、DiT FP16、VAE FP32。A100 使用 Flash，V100 使用 EFFICIENT；两卡均为 SP1，不自动切换后端或规格。

| 策略 | 实现身份 | 用途 |
|---|---|---|
| `off` | `native` | 同输入无缓存参考 |
| `independent_008` | `native` | 复现与诊断，不能重新授予旧资格 |
| `independent_004` | `native` | 全程 0.04/m1 保守候选 |
| `paired_008` | `oracle` | 两分支联合准入的可回滚参考原型 |
| `paired_008` | `split_prefix` | 原型筛选通过后才能实现和执行的前缀分离版本 |

`cfg_policy.resolve_policy(config, policy_id, implementation=None)` 返回独立字典及摘要，摘要包含执行实现。`t008` 和其他未知策略直接拒绝。能够解析 `split_prefix` 不表示该实现已存在或合格，执行器必须检查原型继续门槛。请求、检查点、迁移状态和质量记录均绑定策略摘要；原型资格不能移给高效实现。

原型在采样器更新前保存可回滚缓存、管理器、计数、配置、历史及 RNG；先按原顺序执行两分支。判断不一致时恢复并重算两分支，保留原生判断的记账副作用，然后覆盖为不复用。一次 step 只提交一次 UniPC 更新。只有证明缓存没有原地写入后才可以只快照 tensor 引用。

高效版本若获准开发，分别执行两分支前 8 个 block，在隔离上下文中获取判断，再按原顺序提交联合选择；临时前缀不越过 step 边界，暂停、导出及迁移仅发生于完整 step 边界。所有前缀驻留、同步、判断和重复计算成本计入测量，不修改安装的依赖源码。

## 五阶段与继续条件

独立入口为 `research-two-gpu-cfg`，结果默认写入 `results/research-cfg/run01`。使用已安装的 `.venv-previs` 环境；以下是阶段顺序，各阶段是否启动 GPU 由其已封印前置证据与预算决定：

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg preflight --validate-cpu
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg probe
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg calibrate
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg mechanism
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg report
```

`preflight` 验证模型、依赖、设备、现有账本和 CPU 正确性后冻结新协议。与旧账本追加协议修订绑定，不能重置历史支出、重新分配阶段上限或改写旧报告。

原型筛选通过、高效 adapter 实现后，以独立运行绑定原型证据并重新验证：

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg preflight --output results/research-cfg/run02 --prototype-run results/research-cfg/run01 --validate-cpu
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg probe --output results/research-cfg/run02 --kind correctness
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg calibrate --output results/research-cfg/run02
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg mechanism --output results/research-cfg/run02
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg report --output results/research-cfg/run02
```

`prototype_continuation` 冻结父协议、源码、筛选单元及决定的摘要，仅授权继续实现。父数组不复制，父质量与性能资格不转移。新运行每卡新生成一条当前源码的 oracle 轨迹，与高效版本的全部完整步骤及最终输出对照；正式开发集和留出集仍全部重跑。`execution-plan.json` 根据已测完整单元成本、加载和清理开销，以至少 1.3 安全系数检查完整矩阵；阶段上限保持原值，实际执行仍逐单元准入。

### 已完成开发矩阵后的实现修订

run02 的两个候选均完成开发集，但没有通过双卡正反序均加速 5% 的门槛。run03 只修订 `paired_008` 的执行开销，不增加策略或重选阈值。`implementation_refinement` 绑定 run02 的已关闭单元、账本、失败选择和原型授权；新的质量与性能资格必须重新取得。

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg preflight --output results/research-cfg/run03 --refinement-run results/research-cfg/run02 --cpu-embeddings /mnt/data/zyd-preempt-phase16-01a1296d/dit-research-artifacts/cfg-inputs/run03 --validate-cpu
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg probe --output results/research-cfg/run03 --kind correctness
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-two-gpu-cfg calibrate --output results/research-cfg/run03
```

优化仅在固定依赖、推理状态、原生空 HookRegistry 及无编译／卸载／前向 hook 条件下，省去每步四次空缓存上下文遍历；每步检查模块结构、调用入口和空 registry 是否保持不变，检查成本计入耗时。未知 hook、别名或结构变化立即拒绝。原生策略和 oracle 保留原上下文调用。

文本编码在 GPU 会话开始前用 CPU FP32 完成，绑定模型、输入、源码与 tensor/file SHA256；GPU worker 显式安装，经验证缺失或不符即拒绝，无在线编码回退。此步骤的 CPU 时间单独记录，GPU 加载、安装和持卡时间仍全部计费。编码缓存与上一轮 128 条开发轨迹的正负文本 tensor hash 一致；这仅证明输入相同，不继承输出资格。

每卡正确性使用三条当前源码完整轨迹：带同输入影子诊断的连续高效实现、第 16 步暂停恢复的无诊断高效实现、无诊断 oracle。比较 32 步完整状态及最终 latent／视频，要求影子计算实际执行且每步守卫证据完整。此正确性单元不另跑无缓存轨迹，不能取得逐步无缓存轨迹误差或性能资格。正式矩阵仍包含每卡 8 个开发输入的正反序配对以及门槛通过后的 4 个留出输入，新生成全部无缓存参考。

预算用上一轮封印完整单元估计，不预扣本次预期加速；计划额外计入影子计算、安装、加载、评价和清理。阶段余额不足即停止，不借用其他阶段。

`probe` 区分正确性和诊断的预算归属。筛选输入固定为 origami/101、origami/102、kettle/101，两卡均运行。记录每步条件／无条件计算或复用路径、判断值、耗时；复用时在相同输入 latent 上隔离计算完整预测，记录两分支与 CFG 合成的相对 L2 和最大绝对误差。另记录相对无缓存轨迹的逐步 latent 误差。影子计算不能改变实际状态、RNG 或轨迹，其运行不取得性能资格；常规逐步张量只在内存中使用，保存标量。

任一筛选输入平均 LPIPS>0.05，则对应候选停止。协调原型还须在两卡上均有至少 5% 的乐观加速空间，扣除项只能是明确测量的快照和重复计算；该值仅决定是否继续实现，不能称为已实现收益。

`calibrate` 对最终实现重新进行两卡各 8 个开发输入的 off／候选正反序完整配对。逐输入平均 LPIPS≤0.05，且每卡每顺序的总生成时间均至少改善 5%，才能进入迁移。两个候选合格时，选两卡归一化总时间均值最小者，精确并列选 `independent_004`；随后冻结选择，在 tram／aquarium、201／202 的两卡留出上验证。留出失败即停止，不重选。

`mechanism` 仅接受已冻结并通过留出的共同策略。固定第 16 步完成后最多迁移一次，比较 stay、full 和 cold；不新增精确省略资格。迁移候选分别相对全程 A100/off 和相同迁移路径/off 完成开发及留出质量检查，路径参考先通过共同参考检查。目标繁忙条件使用实际请求队列，从 1、2、4、8 选择首个覆盖盈亏平衡等待时间 1.25 倍的长度；没有覆盖就报告缺测，不人为等待。正反序收益包含等待、搬运、构造、提交、后缀和全部参与请求的解码完成，两次均改善至少 5% 才通过。

`report` 区分已通过、已否定和证据不足，提供 CFG 分支与误差、质量—时间、协调成本与显存、迁移成本交叉四组图。LPIPS 及 5% 均为工程过滤门槛，不代表绝对质量保证或统计显著性。若完整迁移加剩余时间估计即可解释收益，只支持普通异构迁移。协调减少复用次数时，没有相近成本对照就不能把质量改善单独归因于 CFG 对齐。

## 预算与证据留存

继续使用 `results/research-pilot/gpu-budget.json`，累计上限 28,800 GPU·秒，保留原阶段上限。每个完整配对单元按新测费用至少乘 1.3，加载、文本编码准备、评价、持卡空闲、失败和清理全部计费。正确性预算用于 GPU 状态与等价检查；标定预算用于诊断、筛选、正式开发及留出；迁移与多请求使用各自预算。容量阶段额度本轮保留，不挪用。任一阶段不足即停止，不减少质量样本；GPU 执行需要独占窗口。

机制实验使用冻结 CPU FP32 文本编码；这不构成在线 E2E，后者还需真实编码和排队。SP、LP、动态 batch、容量筛选及在线控制器均后置。

保留 `summary_and_key_samples`：全量指标、逐帧质量、分支事件、预算与隔离日志留在 results；关键原始状态、开发中位／最差、留出最差及失败样本和必要参考放在 `/mnt/data/zyd-preempt-phase16-01a1296d/dit-research-artifacts`。完成评价和封印且没有活动引用后才能整理。原始数组缺失必须由索引明确声明，保留旧指标不能称为重新评价原始数据。
