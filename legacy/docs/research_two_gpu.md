# 双卡预实验与结果留存

新协议为 `configs/research_two_gpu.json`，与历史 PreViS/四卡协议独立。两个常驻 SP1 worker 通过 CPU 字节流搬运已暂停的请求；不下载、不复制模型，不启用 SP、LP 或在线控制器。

固定模型：`/mnt/data/zyd-preempt-phase16-01a1296d/dit-runtime/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b`。预检校验完整分片并计算 SHA256，后续启动核对清单和文件身份；不接受其他路径、revision 或自动下载。关键原始证据放在 `/mnt/data/zyd-preempt-phase16-01a1296d/dit-research-artifacts`，结构化结果和报告放在 `results/research-two-gpu`。

使用已安装环境执行五阶段：

```bash
export PYTHONPATH=python
.venv-previs/bin/python -m ditserve research-two-gpu preflight --devices 0,1 --validate-cpu
.venv-previs/bin/python -m ditserve research-two-gpu probe --devices 0,1
.venv-previs/bin/python -m ditserve research-two-gpu calibrate --devices 0,1
.venv-previs/bin/python -m ditserve research-two-gpu mechanism --devices 0,1
.venv-previs/bin/python -m ditserve research-two-gpu report
```

`mechanism --kind migration|mechanism|capacity` 可分别运行三个有依赖的子阶段。GPU 阶段需要独占观察窗口。每张卡的加载、空闲持卡、预热、评价、失败和清理均计费。单卡标定、容量筛选只启动并计费对应设备。原 `results/research-pilot/gpu-budget.json` 的历史金额不会清零；活动会话结算后，首次启动追加剩余额度分配记录（8/20/30/22/5/15%）。

`probe` 测量 MATH/Flash 共同对照、V100 原生 EFFICIENT 两次反序比较，锁定后端；480×832×17 完整矩阵不满足预算时统一降到 256×448×17 并重新验证。仍不满足时停止，保留缺测。只使用 off 与 0.08/m1，32 步、CFG=5、DiT FP16、VAE FP32。完整质量样本固定为八个开发输入与四个留出输入，不用留出选参。

失败或预算停止后的 probe 必须保留原目录。修订执行器后可在新目录预检时传入 `--retry-from results/research-two-gpu/run01`，只引用旧运行独立封印的主规格成本进行准入估算；不会继承质量或后端资格。后续命令均使用新目录的 `--output`。复测使用既定 15% reserve，先保护后续初始化、评价与清理费用；不足则停止。账本追加协议修订链，原分配、上限、历史支出和会话不变。

迁移仅在第 16 步完成后发生一次。完整迁移保留 UniPC、RNG、CFG 元数据及已知特征 buffer；cold 显式强制下一次计算；精确省略必须有紧邻导出的源端证书。目标状态未验证不能接管；源释放后不能回滚执行。目标失败且源尚未释放时源可继续。相同设备检查逐步等价；跨设备先检查无缓存路径，再对共同参考与路径参考分别评价完整视频。

LPIPS≤0.05 是首轮经验过滤器。逐帧、P95、最大值和时序差分均保存；不累加不同参考的误差，不把关闭缓存解释成恢复无缓存轨迹。不同前缀、设备路径、动作和后端的证据不复用。独立完整配对单元各自封印，争用只作废未完成单元，不追溯改变历史性能资格。

多请求实验使用实际解码完成的目标队列，长度从 1/2/4/8 选择；记录所有参与请求的完成时间和完整 makespan。内存筛选暂停 1/2/4/8 个真实请求；只有 keep 的真实新请求准入失败时才测试换出与精确回收，不填充显存。所有原请求与新请求都必须完成并通过质量检查。

结果整理分为生成文件清单、删除前校验、按清单执行三步：

```bash
.venv-previs/bin/python -m ditserve.research.retention plan --results results --plan /tmp/dit-retention-plan.json \
  --project-external-root /mnt/flexllmgen_ssd/preempt-dual-fp16-20260914/dit-rewrite-research-pilot-20261013
.venv-previs/bin/python -m ditserve.research.retention verify --plan /tmp/dit-retention-plan.json --verification /tmp/dit-retention-verified.json
.venv-previs/bin/python -m ditserve.research.retention apply --plan /tmp/dit-retention-plan.json --verification /tmp/dit-retention-verified.json
```

活动账本、待续跑引用、共享文件和归属不明文件受到保护。最新可核验运行中保留正确性对照、开发中位/最差、留出最差、质量失败最差样本及必要参考；原始内容按 SHA256 去重。普通 final latent 只记录 hash；正确性关键样本保存原始状态。清理索引声明每个已删除原始文件，历史 JSON/JSONL、协议和报告保持不变。`raw_available` 表示原始证据可重新评价；`retained_metrics_only` 只能查看已有指标，不能称为新的原始数据复核。未声明文件缺失仍然报错。

报告中的收益门槛是工程继续条件。即使迁移收益超过 5%，若平均剩余成本加完整迁移已解释全部收益，只支持普通异构迁移。SP、LP、在线 goodput 和论文统计显著性都需要后续独立实验。
