# DitServe 离线写作与 Rebuttal 完整补充归档 (`_archive/`)

本目录将原本分散在本地 `.local/` 以及远端 `jjjkairos` 集群上的全部可用实验数据（覆盖至 **2026-11-17 08:46** 最后一批正式实验）统一汇总至 `ditserve/results/_archive/`，确保无 GPU 平台离线写作、Rebuttal 核查或重算任意分位数时数据 **100% 自包含**。

## 目录结构（共 `3,240` 个文件，`210.04 MB`）

1. `1_local_formal_v2/`（`88` 文件，`4.87 MB`）
   - 来源：本地 `.local/formal-v2-cb-refresh-jobs/` 及 `.local/ditserve-kairos-rewrite-runtime/results/`（2026-11-15 ~ 2026-11-17 五份阶段报告）
   - 内容：42 组正式实验经验 CDF、各维度汇总 JSON、8×A100 校准包（`calibration-bundles/d20261106-formal-v2-wan22-b{1,2}-21a1ca3-r1`）及 11-15~11-17 阶段归档报告。
2. `2_local_motivation_and_calibration/`（`441` 文件，`39.18 MB`）
   - 来源：本地 `.local/ditserve-kairos-rewrite-runtime/{evidence,formal-releases}/`
   - 内容：§3.1 8×A100 SP×CFG 全矩阵实测台账、§3.2/§3.3/§5.7 时间线/显存墙/连续批处理/GFC/PCIe 分解 JSON、校准数据及正式发布 commit 单测清单。
3. `3_remote_kairos_outputs/`（`2,541` 文件，`91.25 MB`）
   - 来源：远端 `jjjkairos:/data/share/jjjkairos/output/DITSERVE/`
   - 内容：截至 `2026-11-17 08:46` 的全部可读 `results/`、`acceptance/`（含 `720da72` 全矩阵/扩展性/`SBP+c2`/Soak、`mot-cv`、ISCA'27 多随机种子与 121f 弹性、`3878f01` 5 级阶梯）、`external-tasks/` 及 `20261104-sd35-continuous-batch-sweep/`。
   - 说明：容器以 `root` `0600` 写入而 SSH 账号无读权限的 180 份 `results/**/summary.json` 记录于 `_skipped_unreadable_files.txt`；其对应的可读 Markdown 报告均在 `results/` 中，且完整指标与逐请求明细均已从 100% 可读的 `run.json` 无损提取至 `4_per_request_ledgers/`。
4. `4_per_request_ledgers/`（`167` 份 `.ledger.json.gz` + `index.json`，`74.74 MB`）
   - 来源：远端 `jjjkairos:/data/share/jjjkairos/output/DITSERVE/.cache/results-work/formal-wan22-*/**/run.json`
   - 内容：按运行切片提取并压缩的**逐请求完整调度与抢占时间线明细**（覆盖至 `2026-11-17 08:46`，含 `720da72` 全矩阵、`SBP+c2` 32 组、500 请求五阶段 Soak、ISCA'27 多随机种子/121f 弹性以及 `3878f01` 5 级阶梯），可直接无卡重算任意分位数、筛选子集或绘制微观调度甘特图。索引见 `index.json`。

## 集群环境复原

如需在 Rebuttal 前将代码、模型与镜像环境重新恢复到 `jjjkairos` 集群，直接按仓库根目录 [`MIGRATION.md`](../../MIGRATION.md) 执行即可（所需模型 SHA256 清单、提示词数据集及 Kairos 发布/作业脚本已脱敏归档于 [`deploy/kairos/`](../../deploy/kairos/)）。
