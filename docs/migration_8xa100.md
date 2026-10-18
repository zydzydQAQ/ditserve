# 8×A100 PCIe 迁移与到货验收

目标机为单机 8 张 A100 PCIe 40GB（无 NVLink，用户 2026-10-18 确认）。研究主线是多卡 block 级抢占 + 弹性 Ulysses SP（[实验配置](experiments.md#多-gpu-调度与模拟)）。容器镜像的构建与运行见 [容器镜像](container.md)；本文只规定模型集、目标机检查顺序和通过标准。

本机（1×A100 PCIe 40GB）已验证与未验证的范围：

| 项目 | 本机状态 | 目标机需要做的 |
| --- | --- | --- |
| C++ 控制面、`cluster` 执行器、模拟器 | CPU 测试；mock 8 worker 并发 | 真实 8 slot 回放 |
| Wan 1.3B / Sana 单卡后端 | A100 数值验收与混合回放（`results/a100-e0-acceptance`） | 每张卡重复 E0 |
| SP 对齐进程组（2/4/8） | 四进程布局与原实现相同；8 进程只经 CPU 契约 | NCCL 初始化、等价性、扩展曲线 |
| SP 内 block 执行、跨组状态迁移 | 已实现：block quanta 的组内安全点、step 边界换组（host-staged 导出/导入），`cluster` 执行器 `sp_transport=export`；只有 Gloo/CPU 契约（与 SP1 逐位一致） | 按第 5 步验收 |
| PCIe 拓扑与 all-to-all 带宽 | 无法在本机测 | 第 1–2 步 |

## 模型集

镜像不含权重。权重只读挂载到 `/models`，容器内保持 `HF_HUB_OFFLINE=1`；下载在容器外进行，记录 revision 与逐文件 SHA256 后再挂载。

| 模型 | 用途 | 来源 |
| --- | --- | --- |
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers @ `0fad780a534b6463e45facd96134c9f345acfa5b` | 小视频模型；与本机结果对齐 | 从本机复制（约 27GB，UMT5 FP32 占约 22.7GB） |
| Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers @ `aa76e7f4f4928f378716b6716a2130fba3caf5b1` | 短图像请求 | 从本机复制（约 7.6GB） |
| Wan-AI/Wan2.2-TI2V-5B-Diffusers | **主视频模型**（用户 2026-10-18 确定）：GenServe 与 GF-DiT 均用它，便于直接比较；单卡 40GB 可放下 | 目标机下载；到货后记录 revision |
| Wan-AI/Wan2.1-T2V-14B-Diffusers | 可选的长 step 视频模型（block 级抢占收益最大）；40GB 上 720p 需 SP≥2；不在默认下载集 | 需要时在目标机下载 |
| black-forest-labs/FLUX.1-dev | 标准图像 DiT（`models/flux.py` 已有整管线后端） | 目标机下载；需在 HF 接受许可 |

Mistral-7B 与 LongLive 不进入默认迁移集：会话与流式不在主线内，LongLive 还需要独立 torch 2.8 环境（约 +7.5GB）。本机 `/mnt/data/Mistral-7B-Instruct-v0.2` 可在需要 LLM 共置或显存压力实验时单独复制。

**Wan2.2-TI2V-5B 的移植工作**（到货后、正式实验前完成，并在 CPU tiny 模型上测试）。现有 Wan 路径按 Wan2.1-1.3B 的形状写死，需要按 5B 的配置文件放开：

- `models/wan/block.py` 的 `_validate_model` 固定为 30 层、12 头、16 输入通道；
- `models/wan/parallel.py` 的 `_validate_cache_off_values` 固定 latent 为 16 通道、8 倍空间压缩；
- `models/pipeline.py` 的 `ChunkedDecode` 拒绝带 `patch_size` 的 VAE（Wan2.2 的 VAE 压缩率更高，解码后需要 unpatchify）；
- `evaluation/cost_profile.py` 的 `ARCHITECTURES` 需要加入从配置文件读出的 5B 维度。

改完后，用 E2 和 E5 的方法在目标机上重新测量成本、验证逐位一致，再把成本画像换成 5B 的实测值。

容器外下载示例（下载机需外网；完成后生成清单再挂载）：

```bash
export MODELS=/data/models
HF_HUB_OFFLINE=0 huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "$MODELS/wan2.2-ti2v-5b"
HF_HUB_OFFLINE=0 huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir "$MODELS/flux1-dev"
(cd "$MODELS" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$MODELS/SHA256SUMS"
```

14B DiT 的 BF16/FP16 权重约 28.6GB。在 40GB 卡上 UMT5 只能放在 CPU（与 1.3B 现状相同，本机实测每个 prompt 约 20 秒，8 线程），720p 长视频要靠 SP 分摊激活与 VAE 解码峰值；成本画像用类别的 `min_degree` 表达这一约束。同一 40GB 约束下，block 边界暂停的大模型视频（14B 720p 约 0.9GB/请求，按 1.3B 实测 260 MiB 乘以隐藏维比例估算）会与权重争用显存，暂停态的放置因此成为真实问题。

## 到货验收顺序

每一步的配置和结果按 save-experiment-results 约定保存为 `results/<experiment>/<run>/results.xlsx` + `REPORT.md`；未通过的步骤保持 failed/unverified，不进入下一步的结论。

1. **拓扑与 P2P。** 记录 `nvidia-smi -L`、`nvidia-smi topo -m`、`nvidia-smi topo -p2p r`、`nvidia-smi -q | grep -i -A2 "bus id"`。确认 8 张卡同型号、同显存、驱动 ≥ 570.181。CUDA 设备序号要按拓扑顺序排列，使对齐的 pair 同 PCIe switch、quad 同 socket；否则在 `cluster.gpus` 与 SP rank 顺序里显式重排。若 P2P 读不可用，检查 PCIe switch 的 ACS（`lspci -vvv | grep -i acsctl`）和 IOMMU 设置（需宿主管理员）。通过标准：拓扑记录完整，序号映射已写入配置。
2. **all-to-all 带宽。** 容器以 `--gpus all --pid=host --ipc=host --shm-size=32g --ulimit memlock=-1` 启动。`--pid=host` 是必需的：GPU 隔离检查把 `nvidia-smi` 报告的宿主 PID 与 worker PID 比对，容器的 PID 命名空间会让自己的 worker 被误判为外来进程，运行因此以 "GPU isolation lost" 失败（本机 A100 冒烟已复现并验证此参数，见 `results/a100-image-smoke`）。运行
   `torchrun --nproc_per_node 8 -m ditserve.evaluation.collectives --output /data/collectives.json`。
   该工具测 pair、quad 和 8 卡的 concurrent/isolated 每 rank 带宽，结果替换模拟器成本画像中的 `links`。通过标准：所有组完成；记录各档 GB/s。
3. **单卡冒烟。** `ditserve doctor`；每张卡跑一次 `configs/studies/target-8xa100/acceptance.json`（替换 GPU UUID；模型路径已按 `/models` 挂载布局填写）。通过标准：数值阶段精确一致，混合回放全部完成。
4. **8 slot 并发回放（SP1）。** `configs/studies/target-8xa100/cluster_sp1.json`：每张卡一个 Wan block worker + 一个图像 worker，按混合负载运行 `static_sp1` 与 `block_elastic`（约 300 ms quantum，可抢占解码）。通过标准：全部请求完成，账本结算，GPU 隔离观测为 exclusive。
5. **SP 等价性。** SP 组内的 block 执行与 step 边界换组已有 CPU 契约实现，用 1.3B 和 Wan2.2-5B（如做 14B 也一并）比较 SP1 与 SP2/4/8 的输出：latent 在 BF16/FP16 容差内一致，且有明确记录的最大误差。通过前不得报告 SP 下的任何性能结论。
6. **SP 扩展曲线。** 对 {480p, 720p} × {17, 33, 81} 帧 × SP{1,2,4,8} 测每步时间，拟合成本画像中的 `sp_efficiency`、`links` 和 `comm_mb_per_step`，与本机推演（Phase 4）的预测比较。通过标准：模拟器对留出形状的每步时间误差 ≤10%。
7. **主实验。** 按负载曲线比较 `static_spK`、`step_elastic`、`block_elastic`（含 lazy/eager 抢占消融），多 seed，报告 SLO 达成率、goodput、分类别 p50/p95/p99。

## 不随迁移携带

本机 `results/` 历史结果、旧归档读取器的 fixture 和 `.migration/` 中旧 `python/` 布局的暂存包都不是目标机的输入。历史结论在论文中引用本机工作簿与报告。
