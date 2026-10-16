# Wan profiler 修正与补采

本实验补采 Wan2.1-T2V-1.3B 的真实 GPU 事件与硬件采样。原有
`results/wan-breakdown/run01`、`analysis01` 保持只读；旧的嵌套标记聚合结果
不能作为 kernel 活跃率或 Attention / FFN 占比使用。

## 统计口径

- **真实 kernel**：只接受 Kineto Chrome trace 的 `cat=kernel`。
  `gpu_user_annotation` 是标记，不计 kernel；memcpy / memset 单独统计。
- **活跃率**：真实 kernel 时间区间并集 / 明确的观察窗口。
  分别报告完整请求窗口、请求内首次至最后一次 GPU 活动窗口。
  跨 stream 重叠只计一次。此比例不等于 SM 利用率。
- **Attention / FFN**：用 External id / correlation 关联 CPU 调用，再读取
  同线程嵌套标记；每个 kernel 仅归属一个类别。分母为 Transformer 内真实
  kernel 累计时长，self-attention、cross-attention、FFN、其他分别保存。
  Attention 包括对应模块内投影和归一化，不能把所有 GEMM 当作 FFN。
- **不完整证据**：缺少关联、歧义、多 GPU、非法时间轴等保留未验证状态。
  未归因 memcpy / memset 会使阶段窗口和阶段活跃率不可用，不能填零。
- **硬件指标**：独立 Nsight Systems 采样 SM Active、Tensor Active、DRAM
  Read / Write Bandwidth，保留工具原名。后两者表示 DRAM 接口读写活跃周期比例，
  不换算成 GB/s，也不冒充 Nsight Compute 的峰值吞吐百分比。
  按 NVTX 范围积分采样值；覆盖不足 95%、不足 10 个样本或存在异常时不发布均值。
  完整请求 NVTX 结束前执行 GPU 同步；denoise NVTX 是 CPU 去噪循环范围，
  不额外同步尾部 GPU 工作，因此两个范围分别解释。

## 配置与执行

配置：`configs/studies/wan_profile_corrected.json`。
480 × 832，17 / 33 / 65 帧，32 步，CFG 5，batch 1；同 prompt、seed 0。
UMT5 在 CPU FP32、8 线程；DiT 在 A100 FP16 / Flash-only；VAE 在 GPU FP32。
每种帧数先 warmup，再 clean，再 trace 或 hardware；每种采集只有一个样本，
不能替代多次重复的稳定性结论。采集输出必须与同输入 clean 输出逐字节一致。

原生 C++ 控制器负责两个阶段与同一累计账本：

1. `software_trace`：等待 A100 连续两次空闲，最长 3600 秒；等候不加载模型、不计 GPU 时间。
2. `hardware_metrics`：软件阶段结束后，用独立 Nsight 进程采样；准入仍需 GPU 空闲。

总上限 3600 GPU 秒，其中 300 秒维修预留；单阶段 watchdog 1800 秒。
模型加载、持有 GPU、同步和清理均计费。重试沿用账本，不通过换 run 清零。
执行中每 2 秒复核 GPU 所有权；检测外来作业后只停止自己的进程。

启动软件补采：

```bash
cd /home/zyd/code/dit-rewrite
.cache/build/ditserve run \
  --config configs/studies/wan_profile_corrected.json \
  --run-id run01 --phase software_trace --allow-gpu
```

这条初次启动命令不能重复用于已有 run。续跑应使用工作目录中冻结的 Python
与 `sources/ditserve-core`，并显式指定原工作目录、结果根目录和累计账本。
采集包装器根据冻结清单选择入口：旧来源保留原 Python CLI，新来源直接运行冻结的原生程序并传入冻结包路径；不把当前源码混入旧实验。

硬件采集工具为 `ditserve.evaluation.wan_profile.capture`。它先复核软件阶段、来源绑定、
账本和 GPU 准入，再请求 sudo 认证。只有安装在 `/opt/nvidia` 下的 Nsight 以
管理员权限运行，使用 `--run-as=zyd` 降权启动控制器与模型；后端还校验 UID 1002。
不修改驱动权限设置。密码仅在本机终端输入。

```bash
PYTHONPATH=workers .venv-previs/bin/python -m ditserve.evaluation.wan_profile.capture --check-only
# 软件补采完成且 A100 空闲后，在本机终端去掉 --check-only 执行。
```

采集结束后自动调用同一模块组中的 `publish`，以 CPU 刷新图表和归档。
这些专用工具通过 Python 模块启动；正式实验入口仍是 C++ `ditserve run`。

## 来源与保存

以下 Python 模块均位于 `workers/ditserve/`。`evaluation/cuda_profile.py` 负责真实事件聚合，`models/wan/profile.py` 负责
模块标记和一致性校验，`evaluation/nsys_metrics.py` 读取已验证 SQLite schema，
`evaluation/wan_profile/figure.py` 生成统一统计图。CPU 回归测试使用合成 trace / SQLite
和伪 GPU 查询，测试通过不代表实际 GPU 补采完成。

完整 trace、Nsight 报告和 SQLite 暂存工作目录，直到归因、采样覆盖、数值一致性
与图中数字全部复核通过。之后保存精确摘要、单位、来源大小和 SHA256 到
`results.xlsx` 与 `REPORT.md`，回读验证后按白名单清理。用户要求的统一图额外保存
PNG / SVG；缺失测量明确标为待补采，不能用旧错误统计补齐。
