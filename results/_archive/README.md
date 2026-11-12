# DitServe 离线写作完整补充归档 (`_archive/`)

本目录将原本分散在本地隐藏目录 `.local/` 以及远端 `jjjkairos` 集群上的全部可用实验数据统一汇总至 `ditserve/results/_archive/`，确保迁移至无 GPU 平台写作、改稿或审稿回复（Rebuttal）时数据 **100% 自包含**。

## 目录结构

1. `1_local_formal_v2/`
   - 来源：本地 `.local/formal-v2-cb-refresh-jobs/`
   - 内容：42 组正式实验的全量经验 CDF 分布（`results/latency_cdf_summary.json`）、各实验维度汇总 JSON、Markdown 报告及 8×A100 校准包（`calibration-bundles/`）。
2. `2_local_motivation_and_calibration/`
   - 来源：本地 `.local/ditserve-kairos-rewrite-runtime/{evidence,formal-releases}/`
   - 内容：§3.1 8×A100 SP×CFG 全矩阵实测台账（`20261111-mot1-sp-cfg`）、§3.2/§3.3/§5.7 时间线/显存墙/连续批处理/GFC/PCIe 分解 JSON（`20261104-motivation-local`）、校准数据（`20261103-cal8-51740c0`）及 6 个正式发布 commit 的单测与输入清单。
3. `3_remote_kairos_outputs/`
   - 来源：远端 `jjjkairos:/data/share/jjjkairos/output/DITSERVE/`
   - 内容：`results/`（含 `Wan2.1-T2V-1.3B` Suite 2、全部 `formal-wan22-test-*` 官方归档及 Roofline 分析）、`acceptance/`（全部质量门禁 PSNR/SSIM/无损抢占契约与作业验收 JSON）、`external-tasks/`（xDiT/TetriServe 对齐、`cumem` 生命周期、GFC 2/8 卡微基准 JSON）及 `20261104-sd35-continuous-batch-sweep/`。
4. `4_per_request_ledgers/`
   - 来源：远端 `jjjkairos:/data/share/jjjkairos/output/DITSERVE/.cache/results-work/formal-wan22-test-*/**/run.json`
   - 内容：按运行切片提取并压缩（`.ledger.json.gz`）的**逐请求完整调度与时间线明细**（到达时刻、排队耗时、服务耗时、SLO、GPU 组分配、并行形态切换与抢占/续跑量子时间线），可供后续无卡重算任意分位数或绘制微观调度甘特图。索引见 `index.json`。

## 无卡平台迁移清单

迁移时只需获取以下两项即可：
- `ditserve/`：直接从 GitHub `origin/main` 克隆/拉取（`results/` 及本 `_archive/` 已完整纳入版本控制）
- `ditserve-writing/`（或根目录已同步刷新的 `ditserve-writing.zip`）
