# 无缓存跨卡数值诊断

本入口专门定位 CFG run03 的 `kettle/102` 无缓存迁移路径相对全程 A100 参考的质量失败。保留 `kettle/101` 为通过对照，不修改旧协议、报告或缓存资格。新诊断不取得缓存、迁移性能或在线 goodput 资格。

保持指定 `/mnt/data` Wan1.3B revision、离线加载、256×448×17、32 步、CFG=5、DiT FP16、VAE FP32，以及已校验的 CPU FP32 文本编码。模型不复制，编码仅作为相同输入复用，输出全部新生成。

## 对照与执行门槛

两种配置分别独立执行：原 A100/Flash、V100/EFFICIENT；诊断对照 A100/MATH、V100/MATH。MATH 不替换正式配置。每种配置对两个输入各执行正反两个顺序，每个完整单元包含五条轨迹：A100 连续、V100 连续、第16步 V100→A100 完整迁移、A100 本地导出恢复、V100 本地导出恢复。记录完整32步状态摘要，重点保留第1、16、17、32步数学状态签名。

每单元额外生成一份 V100 第16步状态。真实源保持暂停，分别导入两卡的独立诊断副本，仅执行一次原生第17步，返回条件／无条件／CFG预测和前后 latent 的标量误差。临时张量只用于本单元计算，不保存常规逐步数组。副本不能进入普通步进、迁移或生成流程；副本运行后再次验证源仍暂停且签名不变。两份副本结果须分别匹配相同设备路径真实轨迹的第17步状态，否则诊断无效。

只有原配置所有本地恢复、源暂停隔离及影子等价检查通过，才启动 MATH。跨设备首次计算前的数学状态、tensor 与 UniPC 别名必须一致；首次计算后的数值差异允许存在并如实测量。LPIPS 超过0.05是本轮要定位的观测，不能让预期失败样本提前跳过诊断矩阵。本地恢复或影子等价失败则停止后续对照并优先修正实现。

## 使用

```bash
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-migration-diagnostic preflight --validate-cpu
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-migration-diagnostic probe --kind original
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-migration-diagnostic probe --kind math
PYTHONPATH=python .venv-previs/bin/python -m ditserve research-migration-diagnostic report
```

默认新结果目录是 `results/research-migration-diagnostic/run01`。继续保留五阶段入口，但 `calibrate`、`mechanism` 在诊断协议中明确后置，不能凭少量诊断样本重新授予正式资格。

## 预算与留存

全部新 GPU 占用计入同一28,800 GPU·秒账本的原迁移阶段，不借用正确性、标定、机制、容量或余量额度。完整矩阵按旧同形状有效封印单元估价；旧 MATH 成本仅用于预算，不继承质量或性能。新增状态测量开销单独列为保守估计，至少1.3安全系数；加载、输入安装、评价、双卡空闲持有、失败与清理全部计费。每个完整单元前检查阶段生命周期剩余时间，费用不足则缺测保留未验证。

全部指标、逐帧质量、状态摘要、源码引用及账本保留。完成并封印后，仅保留复现失败对照、本地恢复正确性和第16步关键状态及配套参考；其余已评价原始数组按照清单校验后清理。对旧运行只查看封印指标与声明保留的关键数组，不重新评价已删除原始数据。
