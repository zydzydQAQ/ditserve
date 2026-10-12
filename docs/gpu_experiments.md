# 真实 GPU profiling 与后续实验

已有 V100 单卡真实 Wan 配对记录，当前完整状态与数值见 [正式配对报告](../../results/wan-paired/03-formal-pairs/REPORT.md) 和 [环境归档](../../results/wan-paired/01-environment/REPORT.md)，未完成的测量不算成功。设备为物理 GPU 1（Tesla V100 PCIe 32GB）；此前沙箱内检测失败不能说明宿主机没有 CUDA。CPU 参考执行与调度模拟仍使用 synthetic 成本；标准库测试中的假事件只验证计时、参数和 JSON 契约，不能作为实验数据。本工具测量单请求或固定离线 batch，不实现真实在线 batching、完整缓存状态迁移或集群调度。

## V100 固定实验

历史实验使用独立环境 `.venv-v100`，Python 3.10.20。当前整理不创建此环境；重新执行时按下面的固定版本准备。固定依赖入口在 `benchmarks/configs/requirements-v100.txt`；PyTorch 必须是 `2.10.0+cu126`，不能使用不含 `sm_70` 的构建。历史完整版本保存在环境步骤的 `data.json.gz` 中。安装方式：

```bash
uv venv --python /home/zyd/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10 .venv-v100
uv pip install --python .venv-v100/bin/python 'https://download.pytorch.org/whl/cu126/torch-2.10.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl' -r benchmarks/configs/requirements-v100.txt --index-url https://pypi.org/simple
mkdir -p .cache/runs/v100/environment
uv pip freeze --python .venv-v100/bin/python > .cache/runs/v100/environment/requirements-lock.txt
```

`--preset v100-wan` 使用 FP16 DiT、FP32 VAE、CPU FP32 文本编码器。每次请求重新编码文本、传输 embeddings，均计入端到端耗时并单列；不复用跨请求 embeddings。两组均使用 cache-dit native attention 和 PyTorch SDPA math，关闭编译、量化与额外 attention 扩展。绑定物理 GPU 1 的 UUID，进程内使用 `cuda:0`；preflight 会验证 V100、原生 BF16 能力、构建架构，并实际执行小规模矩阵乘与 math SDPA。

`scripts/run_v100.py` 编排冒烟和正式实验。先做 256×256、17 帧、12 步两组冒烟；正式输入为 480×832、17 帧、50 步、CFG=5、batch=1，三个固定提示词各配 seed 0/1。每个样本、每组各预热一次，再分别进行无观测计时和同 seed 诊断，交替缓存组顺序。缓存参数固定 Fn=8、Bn=0、warmup=8、threshold=0.08、上游 `max_continuous_cached_steps=3`，沿用 cache-dit 1.5.0 的计数逻辑。命令接口以脚本 `--help` 为准。

非有限值触发两组同配置 FP32 重测，并保留 FP16 失败记录。OOM 保留原配置失败，不缩小某一组输入。所有失败和拒绝都出现在编排记录中。运行及补测期间保存原始数组、视频、逐步缓存证据和环境日志，最终归档后删除大数组；显存峰值差不是完整缓存状态大小，输出差异不是感知质量指标。六个样本不能支持稳定 p99，也不能验证在线 serving 的 20% 有效吞吐门槛。

## 环境与权重

`python3 -m ditserve profile --help` 和以下检查无需预先安装重依赖，也不加载模型。缺依赖或 CUDA 时输出 `measured: false`，退出码为 2；不会生成仿真性能数据。

```bash
python3 -m ditserve profile --preflight --model flux
python3 -m ditserve profile --preflight --model wan --cache cache-dit --attention cache-dit-native
```

历史真实测量使用上述 `.venv-v100` 独立环境。在其他机器复现时，另建 Python 3.10+ 环境，安装适合 GPU 驱动的 CUDA 版 PyTorch，以及 diffusers、transformers、accelerate、numpy、sentencepiece、protobuf、ftfy 和可选 cache-dit。以各项目兼容要求为准，并用 `python -m pip freeze > environment.txt` 保存确切版本；不要把 Python 3.8 的标准库测试环境当成 GPU 依赖组合。运行前确认目标 GPU 可用与显存需求；本工具不启用模型动态 CPU offload、量化或编译，V100 preset 的 CPU 文本编码器为固定放置。

FLUX.1-dev 是需要申请访问的模型。先在 [模型页面](https://huggingface.co/black-forest-labs/FLUX.1-dev) 阅读并接受访问条件，使用已有 Hugging Face 登录凭据；不要把 token 写进命令或报告。Wan 使用 [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)。用 `--revision` 指定实际权重 commit，或使用 `--model-id` 指向已固定的本地快照并加 `--local-files-only`。未指定 revision 时，报告会如实记为 null，不声称固定了模型版本。

### 下载固定 Wan 权重

安装上述独立环境后，在仓库根目录运行：

```bash
HF_XET_HIGH_PERFORMANCE=0 HF_XET_FIXED_DOWNLOAD_CONCURRENCY=64 .venv-v100/bin/python scripts/download_v100_weights.py
```

脚本直接读取版本控制中的 `benchmarks/configs/v100.json`，无需预先准备 `results/`。它先调用官方 `HfApi.model_info(..., files_metadata=True)` 核对固定 commit，生成 `.cache/runs/v100/environment/model-revision.json` 和 `model-files.json`，再用 `snapshot_download` 下载同一版本的 model index、scheduler、tokenizer、text encoder、transformer 和 VAE。清单记录各文件字节数、Hub 提供的 LFS SHA256 及选中文件总字节数；非 LFS 文件的 SHA256 为 null，这不是已完成本地逐文件哈希校验的声明。

默认缓存固定在项目 `.cache/huggingface/hub`；下载成功后，同一 environment 目录写入 `model-path.txt`，可供 profiler 的 `--model-id` 使用。下载中断后可重新运行，继续利用已有缓存。对于当前 `hf-xet 1.6.0`，请同时保持 `HF_XET_HIGH_PERFORMANCE=0` 和 `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=64`：开启 HIGH_PERFORMANCE 会覆盖固定并发设置，在本环境中退回 4。不要在已有下载仍运行时再启动第二份。

## 汇总与归档

历史实验已按环境、冒烟、正式配对、阶段补测整理为 Excel 和中文报告。`data.json.gz` 保留小型 profile、manifest、预算、视觉检查记录和来源；原始 NPY 与完整 trace 在预览校验后删除，历史像素哈希不能当作当前可重新核验的数组。

```bash
python3 scripts/archive_results.py verify --root .
python3 scripts/archive_results.py rebuild results/wan-paired/03-formal-pairs
```

新实验的 raw 文件暂存在 `.cache/runs/v100`。`scripts/summarize_v100.py` 汇总活动 raw suite；需要依赖数组的阶段补测先执行，再最终归档。预览用于查看，不能恢复 FP32 输出或替代独立质量评价。

已有助手视觉检查仅代表记录中覆盖的帧与配对，不是独立真人盲评。`perceptual_quality_validated=false`、`research_gate_eligible=false` 保持不变；未完成、失败及未知结果不转换为成功。

## 配对测量

主配对结束后，可追加一个独立的阶段诊断配对：

```bash
.venv-v100/bin/python scripts/profile_v100_phases.py --plan
.venv-v100/bin/python scripts/profile_v100_phases.py --suite-dir .cache/runs/v100/paired --output .cache/runs/v100/phases
```

实际执行要求完整主 suite 已结束成功，每组新进程先完整预热一次，再运行 object/seed0。补测在 Wan 真实去噪循环入口/出口同步计时，并观察已填充缓存的内部 `reset_context`；它不是 DBCache 残差重算、缓存迁移或恢复耗时。补测保存原始数组、与既有同组输出的 hash 对比和 GPU 采样。汇总报告会单列这些带观测的阶段数据，不把它们加入原六对无观测 E2E 性能统计。

以下命令分别使用官方 diffusers pipeline 和 cache-dit 的官方 `DBCacheConfig`、`enable_cache` 与 `summary`。缓存算法是 DBCache，不冒用 TeaCache 名称。两组都显式选择 `cache-dit-native` attention，使 attention 配置一致；只比较 cache 开关与显式记录的固定参数。若只需原始 diffusers 结果，可用 `--cache none --attention default`，但不要与改变 attention 后端的结果混为单一缓存收益。

```bash
python3 -m ditserve profile --model flux --cache none --attention cache-dit-native --height 1024 --width 1024 --steps 28 --seed 0 --warmup 1 --repeats 3 --output .cache/runs/flux-none.json --artifact-dir .cache/runs/flux-none
python3 -m ditserve profile --model flux --cache cache-dit --attention cache-dit-native --height 1024 --width 1024 --steps 28 --seed 0 --warmup 1 --repeats 3 --output .cache/runs/flux-cache.json --artifact-dir .cache/runs/flux-cache
python3 -m ditserve profile --model wan --cache none --attention cache-dit-native --height 480 --width 832 --frames 81 --steps 50 --seed 0 --warmup 1 --repeats 3 --output .cache/runs/wan-none.json --artifact-dir .cache/runs/wan-none
python3 -m ditserve profile --model wan --cache cache-dit --attention cache-dit-native --height 480 --width 832 --frames 81 --steps 50 --seed 0 --warmup 1 --repeats 3 --output .cache/runs/wan-cache.json --artifact-dir .cache/runs/wan-cache
```

正式实验补上相同的 `--revision` 与 prompt。默认 DBCache 使用 Fn=8、Bn=0、warmup=8、阈值 0.08、固定上游 `max_continuous_cached_steps=3`，没有额外 calibrator。可通过 `--cache-*` 固定参数；完整实际配置进入报告。Wan 使用 FP32 VAE，去噪器采用 `--dtype`；默认 GPU 文本编码器采用该精度，V100 preset 则固定为 CPU FP32。视频帧数满足 `frames=4*k+1`。每次都新建同 seed 的 CPU generator；相同 seed 不保证跨 GPU、软件版本、batch shape 的位级一致。

每个 repeat 做两次独立完整生成：先无观测 hooks 测端到端耗时与显存峰值，再以相同 seed、重新刷新缓存做诊断。诊断记录每次 transformer forward 的 CUDA event 耗时及其所属去噪步；Wan CFG 可能每步有两次 forward，不能把 forward 次数直接当步数。诊断 callback 只收集张量元数据，不复制 GPU 内容或逐步同步。诊断运行本身仍有 Python/event 开销，因此性能比较使用 `uninstrumented.e2e_ms`，不要用诊断总耗时替代它。

端到端时间包括文本编码、去噪、VAE 和转成 CPU NumPy 输出，以及 cache-dit v1.5.0 在完整 pipeline 调用内部进行的自动缓存重置与内部统计。下载加载、warmup、调用后 summary 提取、hash 和文件写入不在计时内。完整 pipeline 使用非持久上下文，官方外部 `refresh_context` 对它不执行刷新，因此工具不额外调用或计费这个空操作。`cache_refresh` 明确标记为 pipeline 内自动重置、不能单独观测；兼容字段 `pipeline_plus_refresh_ms` 等于 `e2e_ms`，不表示实际重置成本为零。仍不包括排队、请求解析或冷加载。显存是 PyTorch allocator 的 allocated/reserved 峰值，不是整卡或整进程显存；也不要把缓存开关的峰值差解释为精确 cache 大小。

## 数据和完整性边界

- JSON 为 `ditserve.gpu-profile.v1`，真实成功结果有 `kind: gpu_profile` 和 `measured: true`，记录软件版本、模型/采样器配置、设备、精度、分辨率、seed、缓存实现及逐步数据。
- 缓存统计来自 `cache_dit.summary(..., details=True, logging=False)` 的公开结构化结果，保留各块列表及 CFG 统计；不同 entries 可能包含块级和 transformer 级累计值，不能盲目相加。API 不兼容则报错，不猜测命中率。
- `diagnostic.cache_step_mapping` 与逐步 `cache_evidence` 将公开的 `cached_steps`/`cfg_cached_steps` 映射到 callback 步号。上游 `CachedContext` 以 `executed_steps-1` 记录零起始去噪步；仅当该 entry 的累计去噪步数、forward 数与观测一致且索引合法时映射，否则保留原始统计并标记不支持。每项只指向原始 summary entry，不能假定它们互斥。`reported_cached=true` 说明该 scope 报告使用缓存，不等于整个 transformer 跳过；false 只表示未报告缓存，不能推导精确计算量。DBCache 的 Fn/Bn 块仍可执行，所以不强制给整步贴 compute/reuse 二分类。
- callback 的 `latents.bytes` 和 `sampler.bytes` 是可观察张量的逻辑 payload 大小。重复引用按对象去重，不同 view 可能共享存储；这些字节数不是显存占用或迁移成本。scheduler 只检查实例字段内的 dict/list/tuple 张量，跳过配置并列出无法遍历的对象；不会声称已保存完整 RNG、求解器或请求状态。
- cache-dit 未提供本工具已验证的完整 per-request 缓存快照导出/导入契约，因此缓存状态大小为 `null`、`status: unsupported`。恢复、抢占、迁移延迟均未实测，不能用 latent 大小替代完整缓存大小。
- 报告保留无观测/诊断两遍输出 hash 和 `instrumentation_output_exact_match`。若不同，先排查版本、缓存 reset、随机性和 instrumentation；不据此宣称性能或质量等价。缓存与无缓存输出本来就可能不同，不能要求 hash 相等。
- `--artifact-dir` 在活动实验期间保存原始解码结果 `.npy`，独立质量评价必须在最终清理前完成。归档后只保留视频预览与历史记录，无法重新计算原数组指标。至少评估图像语义符合度和视觉质量；视频还需时序一致性和运动幅度。仅有延迟、hash 或 cache 命中数不能证明质量维持。

正式采样使用足够多 prompt 和 seed，固定方法参数、运行顺序并重复，报告分布与置信区间。3 次默认重复仅适合冒烟检查；不要从它推导稳定 p99。

## 后续单机 4 GPU 流程

1. 先在一张卡完成模型加载、单请求无缓存/固定缓存、重复输出和缓存 reset 检查，人工打开保存的产物确认未损坏。
2. 使用相同数据，逐卡通过 `--device cuda:0` 至 `cuda:3` 收集独立单卡 profile，确认异质性、显存和固定 batch 大小的可行范围。固定 batch 要单独重测，不能把单请求耗时直接除以 batch size。
3. 在独立后续实现中接入真实请求队列、GPU worker 与固定生成/评分策略，测多请求到达下资源调度。当前 profiler 只提供服务时间等输入；仿真回放不构成在线系统验证。
4. 加入统一时钟 trace、每阶段排队/执行事件、显存、吞吐与 SLO 指标，对比静态资源分配及动态策略。若研究 verifier 资源分配，还需要独立真实 verifier profile；当前工具没有 verifier 测量。
5. 若研究迁移，先为固定版本 cache-dit 实现并验证完整 state adapter，对照不中断结果，再测实际 GPU/CPU 传输；在此之前保持迁移测项“不支持”。

实现依据：[Diffusers callback 文档](https://huggingface.co/docs/diffusers/using-diffusers/callback)、[Wan pipeline 文档](https://huggingface.co/docs/diffusers/api/pipelines/wan)、[Cache-DiT 官方 API](https://cache-dit.readthedocs.io/en/latest/user_guide/API_DOCS/)、[缓存配置源码](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_contexts/cache_config.py)、[公开统计源码](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/summary.py)、[attention 配置](https://cache-dit.readthedocs.io/en/stable/user_guide/ATTENTION/)。这些链接跟随上游变化；正式 GPU 实验应保存实际安装版本和权重 revision。
