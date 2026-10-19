# 本机 A100 预实验汇总（2026-10-18）

本机为 1×A100 PCIe 40GB（GPU0，`GPU-3ab2fedc…`），V100 不参与。目标机为 8×A100 PCIe 40GB（无 NVLink）。主线为 **多卡 block 级抢占 + 弹性 Ulysses SP**，见[迁移与验收](migration_8xa100.md)和论文草稿 `dit_serving_sections.tex`。每个实验的长期产物只有 `results/<experiment>/<run>/results.xlsx` 与 `REPORT.md`；原始数组在导出校验后已按白名单清理，只能查看历史指标，不能重新评测。所有结果均为单输入或单 seed 的描述性数据，不作显著性结论。

## 结果一览

| 实验 | 问题 | 结论 | 证据 |
| --- | --- | --- | --- |
| E0 验收 | 精简后的代码能否在真实 GPU 上运行 | Wan 1.3B 状态往返后与官方管线逐位一致；Wan 与 Sana 两个 worker 同卡混合回放完成 4/4 | `results/a100-e0-acceptance/run01`，106 GPU·s |
| E1 历史复现 | 新的原生回放能否复现 block-4 相对 step 的收益（`optimization-cache-online` 的 off_k4/off_step） | ordinary trace 可复现：b4 goodput 偏差 −0.8%，step −10%。burst trace 不可复现：偏差 −29%/−39%；输出不落盘后由 −59%/−47% 收窄到这个范围。两条 trace 上 b4 都优于 step | `a100-e1-parity/run01`（写输出）、`a100-e1-parity-check/run01`（`output_mode=check`），2180 + 2161 GPU·s |
| E2 成本画像 | 调度与模拟需要的实测成本 | 480p 每步 0.58/1.23/4.16 s，720p 每步 1.72/4.14/16.8 s（17/33/81 帧）；block 耗时 = step/60；解码 1.5–15.9 s；暂停态 16–86 MB，恢复 < 10 ms；UMT5 在 CPU 上每个 prompt 18–22 s；Sana 每张 0.30 s | `a100-e2-cost-profile/run01`，`configs/studies/local-a100/profile_a100_measured.json`，377 GPU·s |
| E4 推演 | 8×A100 PCIe 上 block 级抢占是否值得做 | 2026-10-19 在修复后的规划器（f26e9e0 起）上重新模拟。原规划器会让处在 step 边界的视频回到另一个视频停在 step 中途占着的卡上，SP worker 实际不允许这样做；全部 8 个 block 配置都出现过这种情况，static 与 step 配置没有。以下区间都是全部 16 个负载/混合单元（p2p_switch）上的范围：视频 step 长时（外推的 14B），work-conserving 的 step 级弹性 SP 总按时率为 0.46–0.73；为图像预留 1 张卡后为 0.78–1.00；`block_elastic` 不预留为 0.82–1.00，视频按时率比预留基线高 0–23 个百分点，16 个单元都不更差。lazy 与 eager 抢占差别很小（总按时率 −1.7 到 +0.7 个百分点）。按 E3 实测的服务时间（图像 406 ms、解码 ×1.7）校准后重跑（`cal-*`），block 相对预留基线的优势为总按时率 +0–9.3、视频 +0–18.8 个百分点，结论不变。修复前的报告值为 0.49–0.72、0.78–0.99、0.84–0.998、+5–17 个百分点（cal：+1.2–8.0、+5.7–16.2），统计口径未记录，已不再使用 | `a100-e4-whatif/*`，仅 CPU；14B 为 FLOP 外推，PCIe 带宽为假设 |
| E5 block 精确恢复 | step 内部的 block 边界是否是合法的暂停点（GF-DiT 认为不是） | 6 种形状 × 两个 CFG 分支，导出后在新进程恢复，视频和 latent 逐位一致。packet 28–269 MiB，导出 0.15–1.3 s，恢复 0.09–0.45 s | `a100-e5-block-exact/run01`，515 GPU·s |
| E6 图像批处理 | step 级基线是否需要图像批处理 | Sana 每张图成本：batch 1/2/4/8 为 309/238/210/204 ms；batch 8 的整批延迟为 1.63 s。因此基线与本系统都应支持 batch 2–4 的图像批处理 | `a100-e6-sana-batch/run01`，35 GPU·s |
| E3 单卡负载曲线 | `cluster` 执行器在真实后端上的表现，以及模拟器的误差 | 30 个请求（80% 图像，10% 17 帧、10% 33 帧视频），负载 0.4/0.6/0.8。按时率：static_sp1 为 0.73/0.67/0.50；block_elastic lazy 为 0.90/0.73/0.80，eager 为 0.93/0.80/0.83。block 在每个负载下都优于 step；lazy 抢占在单卡上没有收益。在线回放核对（负载 0.6）中 edf_step 为 0.57，edf_b4 为 0.87。feasible-first EDF 会把已经超时的图像排到视频之后，高负载下 block 策略的图像 p95 达 74–118 s，这是用尾延迟换按时率。未校准的模拟器比实测高 0–17 个百分点（平均 11.5）；按实测图像服务时间 406 ms、解码耗时 ×1.7 校准后，平均误差 6.7、最大 16.7 个百分点（每个 cell 30 个请求，一个请求占 3.3 个百分点），策略排序一致 | `a100-e3-load-curves/run01`，2689 GPU·s |
| E7 可抢占解码 | block 抢占后，不可抢占的 VAE 解码是否成为新的队头阻塞 | burst trace，24 个请求。按时数：static_sp1 从 5 增到 8，block_elastic 从 7 增到 10（goodput 0.0171 → 0.0245），可见可抢占解码对两种策略都有帮助（step 级策略下图像同样能抢占正在解码的卡；该实验配置说明中的相反说法有误）。剩下的迟到主要来自 feasible-first EDF：一批图像等完当前 step 时已经错过 0.79 s 的 deadline，被降级后又多等约 80 s | `a100-e7-chunked-decode/run01`，1851 GPU·s |
| E8 共置 | Mistral-7B 与 Wan、Sana 同卡时的显存余量和互相拖慢的程度 | 三个进程加载后驱动剩余 14.4 GiB，各自执行一次后剩 7.4 GiB；每个 3.9k token 的会话 KV 为 488 MiB，增加 6 个会话后只剩 0.53 GiB。三个进程的分配器缓存合计约 11 GiB，彼此无法共享。时间片并发下：Wan 每步 4.19 s，与 LLM 同时为 +22%，与图像同时为 +6%，三者同时为 +28%；LLM 每 token 34–41 ms，与 Wan 同时为 63–66 ms；图像 0.30 s，与 Wan 同时为 0.58 s，三者同时为 0.77 s。KV 的 offload/restore 为 564/168 ms，Wan 42 MB 状态的 offload/restore 为 24/3.9 ms | `a100-e8-colocation/run01`，103 GPU·s |

| E9 迟到短请求不降级 | 真实 GPU 上验证 `late_short_ms=2000` | burst trace（分块解码）：block_elastic 按时数 11/24，图像 p95 1.75 s（E7 为 10/24、90.8 s）；static_sp1 为 5/24、3.9 s（E7 为 8/24、87.8 s）。负载 0.6：block_elastic 27/30，p95 1.76 s（E3 为 24/30、86 s）；static_sp1 为 18/30、1.7 s（E3 为 20/30、104 s）。尾延迟降到 2 s 以内，block 的按时数反而上升，与模拟器预测一致 | `a100-e9-late-short/run01`，1358 GPU·s |
| 镜像冒烟 | 由 c35a811 构建的镜像 `ditserve:rewrite-runtime`（b7aae1591953，7.01 GB）能否在 GPU 上运行 | 不加 `--pid=host` 时因 PID 命名空间误判 GPU 被占而失败（smoke01）；加上后数值验收逐位一致，混合回放 4/4 完成（smoke02） | `results/a100-image-smoke/` |

GPU 用量合计约 11,500 GPU·s（约 3.2 h），在 12 GPU·h 预算以内。

## 主要发现

1. **block 级抢占仍未被现有工作覆盖。** 已核对的 DiT serving 工作（GenServe、TetriServe、GF-DiT、FlashDiff）都只在 step 边界调度。E5 给出了反驳 GF-DiT 论点的直接证据。
2. **收益取决于 step 长度。** 1.3B 480p 17 帧一步只要 0.58 s，step 级抢占已经够用。81 帧和 720p 的 step 达数秒到十几秒，14B 估计在 25–88 s。所以论文主实验应以长 step 的视频模型为主，并与 step 级弹性 SP 加图像预留的强基线比较。
3. **serving 路径上的实际服务时间比孤立测量的长。** E3 中图像的中位服务时间为 406 ms（孤立测量 298 ms），视频解码是孤立测量的约 1.7 倍，主要来自 CPU 上 UMT5 编码对其他进程的争用。模拟器与 E4 推演应采用这些服务路径上的数值；目标机上需要重新测量。
4. **抢占之后还剩下其他阻塞源。** VAE 解码（数秒）、CPU 上的文本编码（约 20 s，并且与图像请求争用 CPU）、输出落盘（每张图约 45–75 ms）都会成为新的队头阻塞。可抢占解码（E7）和 `output_mode=check` 处理了其中两项。目标机为 40GB，UMT5 只能放在 CPU 上，编码延迟需要预取来掩盖。
5. **E1 的 burst trace 尚未复现。** 新路径与历史的退休路径在进程结构上不同（Wan 与 Sana 分属两个进程，并通过 IPC 交互），对 burst 到达的时序很敏感。在论文里只能使用新路径重新测得的数据，不能混用历史数值。

6. **时间片共置保护不了短请求。** 与 Wan 同时运行时，图像延迟几乎翻倍，LLM 每 token 耗时增加约 70%。block 抢占的等待只有一个 block（约 70 ms），而且不会拖慢视频。40GB 卡上放下 LLM 与两个 DiT 后几乎没有余量，分配器缓存又分散在各进程里，所以暂停态和 KV 的放置（保留、offload、丢弃）在目标机上是真实存在的问题。
7. **feasible-first EDF 需要修正。** 已超时的短请求不应排到长视频之后。新增的 `late_short_ms`（默认 2 s）让剩余时间短的迟到请求保留 EDF 位置。在校准后的模拟器中，这使图像 p95 从 70–140 s 降到 0.8–4.8 s，代价是按时数变化 −4 到 +5 个（每组 30 或 24 个请求）。真实 GPU 上的 E9 证实：图像 p95 降到 2 s 以内，block 的按时数上升，static_sp1 略降。

## 局限与待办

- 14B 成本和 PCIe all-to-all 带宽是估计值。到货后按验收清单第 2、6 步实测，再重跑 E4。
- SP 组内的 block 执行和 step 边界换组尚未在 GPU 上实现。目前真实后端固定为 SP1、不换组。
- 所有 GPU 数据都只有一个输入或一个 seed。负载曲线（E3）仍需增加 seed 和形状。
- 评测中同时报告按时率和 p99；`late_short_ms` 的取值需要在目标机上做敏感性分析。
- E4 的 14B 成本与 PCIe 带宽仍为估计值；到货后用目标机实测替换并重跑。

## 收尾清理凭据（2026-10-18）

有 14 个原始文件没有登记为运行产物，因此 finalizer 没有删除它们：

- E1 run01 预热阶段写出的视频和图像。当时预热还没有使用 `output_mode=check`。
- E5 跨进程使用的 block packet。当时 `block_export` 的输出还没有登记为产物。

两个原因都已在代码中修正。这些文件都没有下游使用：对应运行已全部完成，不会续跑，工作簿与报告已导出并校验过。清理前逐个记录了大小和 SHA256，然后按显式白名单删除，共回收 2462944993 字节。删除后只能查看历史指标，这些原始数据无法恢复，也不能重新评测。

| 路径 | 字节 | SHA256 |
| --- | --- | --- |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-0/artifacts/wan/output-3485f442c3bd5349-4.npy` | 388178048 | `dc14b01537c2a5b9c5039eb25fd6f0aa31565ed8fff57f2dab7a4fc83e042f66` |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-1/artifacts/wan/output-3485f442c3bd5349-4.npy` | 388178048 | `dc14b01537c2a5b9c5039eb25fd6f0aa31565ed8fff57f2dab7a4fc83e042f66` |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-2/artifacts/wan/output-3485f442c3bd5349-4.npy` | 388178048 | `dc14b01537c2a5b9c5039eb25fd6f0aa31565ed8fff57f2dab7a4fc83e042f66` |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-3/artifacts/wan/output-3485f442c3bd5349-4.npy` | 388178048 | `dc14b01537c2a5b9c5039eb25fd6f0aa31565ed8fff57f2dab7a4fc83e042f66` |
| `.cache/results-work/a100-e5-block-exact/run01/transfer/cand-480x832x17.pt` | 43970663 | `5e9ae64332c076d8dc2130f4b55ab8f7b8e2f045133aea7ec28eeed7e074b76e` |
| `.cache/results-work/a100-e5-block-exact/run01/transfer/cand-480x832x33.pt` | 72924644 | `4a3f9e36ff88b91435c2ed7d51b84f8ae98449845e930a3f5f6733001bb08578` |
| `.cache/results-work/a100-e5-block-exact/run01/transfer/cand-480x832x81.pt` | 152596583 | `41768fc5ce85143958de0003c7800ae1108b64652c11ab3bb2fdddc134319ad3` |
| `.cache/results-work/a100-e5-block-exact/run01/transfer/cand-720x1280x17.pt` | 90665444 | `058663000c7a807fbc68edc9b0dbc316f17357daf2a8bf20950fdc5425b06bdf` |
| `.cache/results-work/a100-e5-block-exact/run01/transfer/cand-720x1280x33.pt` | 151029863 | `58926cca1763be16952ec5859c6b3a01bd6829a76e105b7ffa786ce4e4b6a66f` |
| `.cache/results-work/a100-e5-block-exact/run01/transfer/cand-720x1280x81.pt` | 348713444 | `2b439277a745501488d72e844cf44f6c2d28f1b1e74164ad9bfc5d116c4de448` |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-0/artifacts/sana/output-be808f9763ab27c9-3.npy` | 12583040 | `fdbcefd6bacfe268c51ce5ce0b5761ab302f08b6004ff47736be3b52e806bf1f` |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-1/artifacts/sana/output-be808f9763ab27c9-3.npy` | 12583040 | `fdbcefd6bacfe268c51ce5ce0b5761ab302f08b6004ff47736be3b52e806bf1f` |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-2/artifacts/sana/output-be808f9763ab27c9-3.npy` | 12583040 | `fdbcefd6bacfe268c51ce5ce0b5761ab302f08b6004ff47736be3b52e806bf1f` |
| `.cache/results-work/a100-e1-parity/run01/attempts/replay/attempt-1/cell-3/artifacts/sana/output-be808f9763ab27c9-3.npy` | 12583040 | `fdbcefd6bacfe268c51ce5ce0b5761ab302f08b6004ff47736be3b52e806bf1f` |
