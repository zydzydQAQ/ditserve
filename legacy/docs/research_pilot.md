# 质量约束下的缓存、抢占与 SP 研究试验

本试验使用独立的 `ditserve.research` 包、`configs/research_pilot.json`、源码归档和 GPU 账本。旧 PreViS 协议、历史结果和旧预算不会被修改。当前工作是可验证的研究基础与实验执行器，不代表论文命题已经成立。

## 实现范围

| 组成 | 行为与边界 |
|---|---|
| 状态盘点 | 按底层 storage 去重，保留所有引用路径与角色；必要态和特征共享 storage 时不能将其算作可释放特征。支持 UniPC timestep 历史的已知标量 view，以 storage 为单位拷贝并重建别名；其余不同 tensor view 共享 storage 的搬运暂时拒绝。 |
| 刷新证书 | 对固定 cache-dit 1.5 控制流进行只读判断；两 CFG 分支下一步自然触发连续复用上限时，才可提出回收候选。绑定请求、step、配置、布局、状态内容。 |
| 精确回收 | 验证执行器显式允许后，只清空已认证特征，保留 counters、scheduler、RNG 和决策元数据；不设置 cold 的 `needs_anchor`。GPU 完整轨迹验证之前不能称其已具备精确资格。 |
| 近似调整 | 32 步请求最多在完成第 16 步后调整一次；完整前缀、状态动作、后缀与布局历史共同构成证据身份。关闭缓存不消除前缀误差。 |
| SP 基础 | 预建单卡、相邻双卡、四卡组；归约限定请求组；CFG、step、cache presence、shape、layout epoch 不一致时统一失败。一次安装 Ulysses/Flash 布局接口，SP1 显式绕过。 |
| SP 迁移 | 已实现无缓存 Wan 请求的真实跨进程 SP4→SP2→SP4 搬运；完成第 16／24 步切换，搬运 latent、conditioning、UniPC 和 RNG，预提交确认后释放旧态。CPU/Gloo 已验证真实采样器后缀逐位一致。缓存特征迁移暂不开放；GPU 数值、质量与收益仍待四张同型卡验证。 |
| SP 测量 | 计费入口依次测 off 和 0.08/m1 的 SP1、两个独立 SP2、SP4，随后按预算尝试无缓存弹性对照。保存真实 compute／mixed／reuse 路径和完整视频；未出现的路径保留缺测。Wan 的四个视频残差按 token 轴切分，两个 encoder 残差保持复制。每次重载模型均计费，缓存跨布局迁移仍拒绝执行。 |
| 证据和选参 | 三个初始 recipe × 八个开发输入；按初选 prefix 生成去重 suffix。最多两个联合候选补齐八个开发单元，开发合格后才测四个留出单元。缺测、质量失败及容量失败不从分母中消失。 |

模型固定 Wan2.1-T2V-1.3B revision `0fad780a534b6463e45facd96134c9f345acfa5b`，32 步、CFG=5、DiT FP16、VAE FP32、原生 SDPA Flash。初始缓存为 off、0.04/m1、0.08/m1。不启用 Ring、TP、通信量化、编译或 CUDA Graph。LP 始终指 Latent Parallelism，留待主线机制证据通过后开展。

## 使用

在仓库根目录运行，先激活已有依赖环境或使用下列解释器；所有模型文件均来自固定本地快照。

```bash
export PYTHONPATH=python
export DIT_RESEARCH_RUN=results/research-pilot/new-run

# 创建源码/协议归档，执行 CPU/Gloo 测试和只读 GPU 发现；不加载模型。
.venv-previs/bin/python -m ditserve research preflight --validate-cpu --devices 0 --output "$DIT_RESEARCH_RUN"

# 驱动与 CPU 门槛通过后，probe 先执行并验证 GPU correctness，再探测帧数。
.venv-previs/bin/python -m ditserve research probe --devices 0 --output "$DIT_RESEARCH_RUN"
.venv-previs/bin/python -m ditserve research calibrate --devices 0 --output "$DIT_RESEARCH_RUN"
.venv-previs/bin/python -m ditserve research mechanism --devices 0 --kind state --output "$DIT_RESEARCH_RUN"

# 只有四张同型号卡及前置证据齐备时才会分配四卡。
.venv-previs/bin/python -m ditserve research mechanism --devices 0,1,2,3 --kind sp --output "$DIT_RESEARCH_RUN"

# 不占 GPU，核验证据并生成报告和四组 SVG 图。
.venv-previs/bin/python -m ditserve research report --output "$DIT_RESEARCH_RUN"
```

五个公开阶段为 `preflight / probe / calibrate / mechanism / report`。可指定 `--protocol`、`--output`、`--ledger`、`--model-path`；预算固定为 `--budget-gpu-hours 8`。`preflight` 会冻结源码与依赖实现，修改代码后请使用新的输出目录，继续指向同一个累计账本。已有 GPU 阶段不会直接重跑覆盖证据。预检被阻断时退出码为 2，已生成的诊断与零消耗账本仍可读。

默认输出 `results/research-pilot/run01`，唯一新试验累计账本 `results/research-pilot/gpu-budget.json`。不要通过改变账本路径或删除账本重置预算。`protocol.json` 绑定账本绝对路径、源码及依赖 hash；归档内保留实际源码。`report` 可读取冻结归档，不要求当前实现仍与归档相同。

## GPU 计费和停止

总上限为 **28,800 GPU·秒**。正确性 1,800；状态探测 5,400；标定与抢占机制共 9,000；四卡 SP 9,000；保留复核余量 3,600。余量不会自动借给其他阶段。四卡运行一秒计四秒。

必要复测可显式指定 `--use-reserve`，使用同一累计账本的复核桶；需要新的输出目录和该源码的 CPU 验证。该选项不重置已用预算、不借用 SP 桶，也不允许覆盖旧证据。

父进程从模型加载前持有独占账本锁，直到所有 GPU 子进程退出才结算。加载、预热、失败、持卡质量评价、空闲等待和清理均计费。子进程核对父 PID、活动预算 session、设备和协议，不接受无账本的独立执行。下一完整配对单元连同清理预计放不下时停止；如果实测超过估算，未完成单元没有资格。崩溃恢复保守计费，未清理的进程不能变成免费时段。

探测按 480×832、17→33→49→81 帧递增，遇容量失败停止上探。四卡未到位时保持 SP 预算，不用 A100/V100 混合替代。CPU/Gloo 测试需要本机回环 socket；环境禁止 socket 时它应明确失败，而非跳过后记为通过。

成本实验要求选定 GPU 没有其他计算进程。预检会拒绝忙碌设备；运行期间父进程每约两秒检查 GPU 进程是否属于本次 worker 进程组。发现外部占用或无法确认归属时，仅停止本研究进程、继续计费至清理完成，并将本轮成本证据保守标为无效。`measurement-environment.json` 与逐阶段监测日志保留原始观察。此检查是采样监测，不能替代实际独占设备调度，也不保证捕获采样间隔内的短暂干扰。

标定形状按完整实验费用选择，并非直接使用能装进显存的最大形状。`cost-plan.json` 用探测得到的整次生成、归档和整视频 LPIPS 墙钟时间，保守预留模型／编码初始化、转移和清理，选择剩余 9,000 秒标定桶能容纳的最大形状及一或两个候选。成本对照只使用最小探测形状和选定形状；只有一种条件时不能证明成本交叉。预测是工程预留，实测账本与逐单元停止仍是硬约束。

正确性基准每次执行真实文本编码，并增加 off／0.08 两种配置的冻结文本路径逐步等价检查。后续机制阶段使用冻结的 CPU FP32 文本编码，按需编码新提示词，保存内容 hash 并绑定完整轨迹身份；留出提示词只在留出阶段编码。编码初始化和持卡等待仍计入 GPU 账本，单请求机制耗时不重复计入已完成的编码。所有 GPU 子进程固定使用 8 个 CPU 计算线程；这些结果不能作为包含在线编码和排队的 E2E 性能。

## 验收与报告

同布局 hot 与合法回收，对照连续执行逐步 latent、完整 sampler/cache/RNG 状态、分支路径、计数与最终视频。cold 和强度变化使用同输入无缓存视频作完整轨迹参考，保留每帧、平均和 P95 LPIPS、时序差分。平均 LPIPS≤0.05 是首轮经验门槛，不是绝对画质或逐请求保证。

状态量是 storage 盘点，峰值 allocated 还包括权重、工作区与其他分配；reserved 下降不等于实际可接纳容量。内存命题必须由 keep 的真实容量失败以及回收后真实新请求完成共同支持。抢占命题要求两种反序重复均改善至少 5%，且不同条件下有不同动作占优。弹性 SP 命题要求 compute/reuse 的优选布局不同，收益在扣除实测迁移与 collective 成本后仍有至少 5%。

报告将没有测量的图标为缺测，不绘制假数据；缺测结论为“证据不足”，并非“否定”。CPU 单元测试、Gloo 通信或静态 SP 数值日志都不会自动打开在线联合控制器。后续在线控制器、五个配对 trace seeds 的完整论文评估和 LP 分支按原计划门槛推进，完整论文评测另行预算。

## 2026-10-13 实施检查点

最新已执行目录为 `results/research-pilot/run04`，完整 CPU 回归 619 项、其中研究专项 CPU/Gloo 160 项均通过，零跳过。该轮没有修改冻结源码。初次忙碌预检保存在 `preflight-initial-blocked.json`；A100 空闲后重新预检并完成正确性、形状探测和全部初始标定。

| 阶段 | 实际结果与资格边界 |
|---|---|
| GPU 正确性 | off／0.08 与官方流程、冻结文本路径逐步等价；keep／hot 及合法自然刷新边界的精确回收通过 latent、采样器、缓存分支、计数、RNG 和最终输出对照。cold 仅通过结构检查，未取得完整轨迹质量资格。 |
| 状态探测 | 480×832 的 17／33／49／81 帧均完成，无容量失败。81 帧的特征缓存为 386.906 MiB；单请求探测不能证明释放后接纳新请求的收益。 |
| 初始标定 | 4 个开发提示词 × 2 个种子 × 3 个策略，共 24 条完整轨迹。0.08/m1 在八个输入上的整视频平均 LPIPS 均≤0.05，最差为 origami/102 的 0.048490；该视频逐帧 P95 为 0.085329，最大单帧为 0.120059，8/33 帧高于 0.05。0.04/m1 均未触发复用，输出与 off 一致。完整证据及原始选择保留；这些质量结果不覆盖联合动作或留出输入。 |
| 联合动作 | 初始选择为 0.08/m1，成本计划锁定 33 帧和 1 个联合候选。筛选期间出现外部 A100 进程，占用守卫立即中止本研究进程；没有完成任何动作筛选单元，开发联合轨迹、留出、反序成本对照与真实准入均缺测。 |
| SP | 仍只有一张 A100 和一张 V100，未执行混合设备替代试验。四卡 SP 和缓存跨布局迁移的 GPU 验证均未完成。 |

`run04` 的正确性、探测和标定各自保留了已完成的独占采样日志，但当前执行器采取整轮保守规则：后续机制阶段争用将 `measurement-environment.json` 标为 `contended`，`cost_measurements_valid=false`，报告不再采用该轮任何性能数据作收益结论。原始标定中位去噪加解码时间为 off 42.640 秒、0.08/m1 34.594 秒，差约 18.87%；这只是原始观察，**不是最终有效性能资格**，也不是在线 E2E 结果。数值正确性与完整视频质量记录仍保留，不被资源争用抹除。

机制的最终状态以 `mechanism-attempt.json` 的 `blocked_contention` 为准；`mechanism.json` 是中止前的运行检查点，其 `running` 不表示后台仍有本研究进程。账本活动会话已清空，失败及清理均已计费。`run01/02` 缺少独占监测，`run03/04` 已确认发生争用，均没有有效机制收益结论。

独立账本累计消耗 **4,093.812 GPU·秒 / 28,800 GPU·秒**，剩余 **24,706.188 GPU·秒**：

| 分项 | 已消耗 GPU·秒 | 剩余 GPU·秒 |
|---|---:|---:|
| 正确性 | 1,176.757 | 623.243 |
| 状态探测 | 729.708 | 4,670.292 |
| 标定与抢占机制 | 1,226.675 | 7,773.325 |
| 四卡 SP | 0 | 9,000 |
| 复核余量 | 960.673 | 2,639.327 |

本次质量标定完成，但内存、抢占和弹性 SP 三项研究门槛均仍为“证据不足”，没有因此开启在线控制器或 LP 分支。

为避免根分区被视频数组填满，仅将新研究 `run01`—`run04` 的 `arrays` 移至 `/mnt/flexllmgen_ssd/preempt-dual-fp16-20260914/dit-rewrite-research-pilot-20261013/`，原路径保留符号链接。搬运时逐文件校验 SHA256，各目录的 `array-storage.json` 保存搬运清单；后来生成的数组由轨迹记录继续绑定内容 hash。源码、账本、原始记录及旧 PreViS 结果保持原位置。这是证据归档位置调整，不是模型状态的 NVMe 卸载。

下一次需要可靠的 A100 独占窗口；只在预检瞬间空闲不足以保证一轮实验完成。已有 GPU 阶段不能原地重跑，须使用新目录并沿用同一累计账本；新一轮将按剩余预算重新规划形状和候选数，不直接沿用 run04 的成本资格。必要正确性复测仍用复核余量，避免超出正确性分项预算：

```bash
export PYTHONPATH=python
export DIT_RESEARCH_RUN=results/research-pilot/run05
.venv-previs/bin/python -m ditserve research preflight --validate-cpu --devices 0 --output "$DIT_RESEARCH_RUN"
.venv-previs/bin/python -m ditserve research probe --devices 0 --output "$DIT_RESEARCH_RUN" --use-reserve
.venv-previs/bin/python -m ditserve research calibrate --devices 0 --output "$DIT_RESEARCH_RUN"
.venv-previs/bin/python -m ditserve research mechanism --devices 0 --kind state --output "$DIT_RESEARCH_RUN"
.venv-previs/bin/python -m ditserve research report --output "$DIT_RESEARCH_RUN"
```

新运行目录也应为视频数组预留足够数据盘空间。后续阶段只有在前置记录、完整 GPU 生命周期及独占采样日志均通过时才会执行。四张同型卡到位前保留 SP 预算；缺测不作为否定研究命题的依据。
