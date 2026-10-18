# 迁移与复现手册

在一台新机器上按本文从上到下执行，即可复现本项目的全部实验。同一套代码、镜像和实验配置适用于三种机器，只有环境名不同：

| 环境名 | 机器 | 状态 |
| --- | --- | --- |
| `a100-1x40g` | 开发机：1×A100 PCIe 40GB（GPU0；GPU1 的 V100 属于其他项目，已在环境文件里禁用） | E0–E9 预实验已完成，见 [docs/a100_preexperiments.md](docs/a100_preexperiments.md) |
| `rtx4090-2x24g` | 2×RTX 4090 24GB，消费卡，无 GPU P2P | 下一站：按本文复现 |
| `a100-8x40g` | 8×A100 PCIe 40GB，无 NVLink | 到货后按本文复现，另见第 8 节 |

环境文件 `configs/env/<环境名>.json` 写明该机器用哪些 GPU、模型放在哪、worker 的环境变量、实测成本画像和请求类别。实验配置 `configs/studies/*.json` 与机器无关；`ditserve run --env <环境名>` 在运行前把它们绑定到本机，并把绑定结果记录进冻结配置。

## 0. 前置条件

| 项目 | 要求 | 检查命令 |
| --- | --- | --- |
| 驱动 | ≥ 560（镜像里的 torch 2.10.0+cu126 自带 CUDA 12.6 用户态库，不需要在宿主机装 CUDA） | `nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv` |
| 容器 | Docker 和 nvidia-container-toolkit | `docker info \| grep -i nvidia` |
| 内存 | 每个 Wan worker 在 CPU 上常驻约 23GiB 的 FP32 UMT5：2×4090 需 ≥ 64GiB，8×A100 需 ≥ 256GiB | `free -g` |
| 磁盘 | 模型约 35GB，镜像约 7GB，结果与工作目录几 GB | `df -h` |
| 网络 | GitHub、PyPI、Docker Hub、Hugging Face。不通时可用镜像站（见第 2 步），也可以从开发机拷贝镜像和模型 | |

下面的命令都在仓库根目录执行，并共用这些变量：

```bash
export ENV=rtx4090-2x24g            # 开发机用 a100-1x40g，8 卡机用 a100-8x40g
export DATA=$HOME/ditserve-data     # 结果、账本、缓存；容器内是 /data
export MODELS=$HOME/models          # 模型；容器内只读挂到 /models
export IMAGE=ditserve:rewrite-runtime
export C=/opt/ditserve/share/ditserve/configs   # 容器内的 configs（下面挂载为本仓库的 configs/）
mkdir -p "$DATA" "$MODELS"

# GPU 运行。--pid=host 是必需的：GPU 隔离检查要把 nvidia-smi 报告的宿主 PID 和 worker 对上，
# 容器自己的 PID 命名空间会让 worker 被当成外来进程（报 "GPU isolation lost"）。
# 多卡 NCCL 需要 --ipc=host、足够的 /dev/shm 和不受限的 memlock。
ds() {
  docker run --rm --gpus all --pid=host --ipc=host --shm-size=32g --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" -e DITSERVE_ENV="$ENV" \
    -v "$DATA":/data -v "$MODELS":/models:ro -v "$PWD/configs":"$C" "$IMAGE" "$@"
}
```

挂载本仓库的 `configs/` 是为了让第 6 步生成的成本画像和对环境文件的修改不用重建镜像就能生效。

## 1. 代码与镜像

```bash
git clone -b 8a100 https://github.com/zydzydQAQ/ditserve.git && cd ditserve
DOCKER_BUILDKIT=0 docker build --target runtime --build-arg GIT_REVISION="$(git rev-parse HEAD)" -t "$IMAGE" .
docker run --rm "$IMAGE" ditserve --version
```

镜像本身与 GPU 型号无关：

- 没有任何自编译的 CUDA kernel。
- torch wheel 带 sm_80、sm_86 和 sm_90 的 SASS：A100 是 sm_80；4090 是 sm_89，sm_86 的 cubin 可以直接在 sm_89 上运行。
- Triton 和 Inductor 会在本机现场编译。

所以 4090 和 A100 用同一个镜像。构建时会编译 C++ 并跑 ctest；依赖按 `docker/artifacts-runtime.json` 逐个校验 SHA256，不会解析出锁文件以外的版本。

PyPI 或 Docker Hub 不通时，有两条退路：

- **用 wheelhouse 离线构建。** 把锁文件里列出的 wheel 放进一个目录，然后执行：
  ```bash
  python3 -m http.server 18765 --bind 127.0.0.1 --directory /path/to/wheelhouse &
  DOCKER_BUILDKIT=0 docker build --network host --target runtime \
    --build-arg PYTHON_BASE=<本地已有的 python:3.10.20-slim 镜像> \
    --build-arg DITSERVE_WHEELHOUSE_URL=http://127.0.0.1:18765 \
    --build-arg GIT_REVISION="$(git rev-parse HEAD)" -t "$IMAGE" .
  ```
- **直接搬开发机的镜像。** 开发机上 `delivery/` 目录里有 `docker save` 导出的包和 `SHA256SUMS`。拷过来后执行：
  ```bash
  sha256sum -c SHA256SUMS && docker load -i ditserve-rewrite-runtime.tar.gz
  ```

## 2. 模型

只用两个模型：Wan2.1-T2V-1.3B（视频）和 SANA-Sprint 0.6B（图像）。版本固定在环境文件里，逐文件 SHA256 在 `configs/models/`。

```bash
hf() { docker run --rm --user "$(id -u):$(id -g)" -e HF_HUB_OFFLINE=0 -e HF_HOME=/models/.hf \
         ${HF_ENDPOINT:+-e HF_ENDPOINT=$HF_ENDPOINT} -v "$MODELS":/models "$IMAGE" huggingface-cli "$@"; }
# 需要镜像站时：export HF_ENDPOINT=https://hf-mirror.com
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --revision 0fad780a534b6463e45facd96134c9f345acfa5b \
   --local-dir /models/wan2.1-t2v-1.3b
hf download Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers --revision aa76e7f4f4928f378716b6716a2130fba3caf5b1 \
   --local-dir /models/sana-sprint-0.6b
for m in wan2.1-t2v-1.3b sana-sprint-0.6b; do (cd "$MODELS/$m" && sha256sum --quiet -c "$OLDPWD/configs/models/$m.sha256") && echo "$m ok"; done
```

通过标准：两行都输出 `ok`。下载不了时，可以从开发机拷贝快照（`rsync -L` 跟随 HF 的符号链接），源目录在：

- `/mnt/data/zyd-preempt-phase16-01a1296d/dit-runtime/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a…`
- `/mnt/data/zyd-preempt-phase16-01a1296d/dit-directions-assets/huggingface/hub/models--Efficient-Large-Model--Sana_Sprint_0.6B_1024px_diffusers/snapshots/aa76e7f4…`

## 3. CPU 契约测试（不占 GPU）

```bash
docker run --rm --user "$(id -u):$(id -g)" -e CUDA_VISIBLE_DEVICES= "$IMAGE" \
  python -m unittest discover -s /opt/ditserve/share/ditserve/tests/python
```

通过标准：以 `OK` 结束，跳过数等于镜像里不带的旧格式结果测试。C++ 测试已在构建时运行。

## 4. 机器检查

```bash
ds ditserve doctor --env "$ENV" --allow-gpu
for s in acceptance cost_profile block_exact serving sp_equivalence; do
  ds ditserve resolve --config "$C/studies/$s.json" --env "$ENV" > /dev/null && echo "$s resolves"; done
```

通过标准：

- doctor 的 `status` 是 `ok`，每项检查都是 `pass`。
  - `--allow-gpu` 只让 CUDA 看见环境文件选中的 GPU。
  - 4090 上 `peer_access` 显示 false 是正常的，环境文件已设 `NCCL_P2P_DISABLE=1`，此项为 `warn`。
- 第 4 步的 `resolve` 预期结果：
  - **成本画像尚未测量（第 6 步之前）：** 4090 和 8 卡机上的 `serving` 会报 `has no cost_profile`。
  - **单卡环境：** `sp_equivalence` 会报 `no phase of this study fits`。
  - 其他实验都应输出 `resolves`。

## 5. 验收（`acceptance`，每张卡都跑）

```bash
ds ditserve run --config "$C/studies/acceptance.json" --allow-gpu --run-id run01
```

内容与门槛：

- **numeric：** stepwise worker 与 diffusers 官方循环逐位一致。
- **block：** 在 step 内暂停，导出到文件，再导入，最后按 latent 帧分块解码，结果与不中断的运行逐位一致。
- **mixed：** 两个视频和两张图像经 cluster 执行器全部完成，无失败。

结果在 `$DATA/results/$ENV-acceptance/run01/`（`results.xlsx` + `REPORT.md`），报告首表各阶段的数值资格应为 passed。GPU 预算上限写在配置里（`budget.limit_gpu_seconds`），A100 上约需 5 分钟。

## 6. 成本画像（`cost_profile`）与 all-to-all 带宽

```bash
ds ditserve run --config "$C/studies/cost_profile.json" --allow-gpu --run-id run01
# 两卡及以上：测 SP 组的 all-to-all 带宽（torchrun 能看见所有 GPU）
ds torchrun --nproc_per_node "$(nvidia-smi -L | wc -l)" -m ditserve.evaluation.collectives --output /data/collectives-$ENV.json
# 生成画像（单卡省略 --links；没有 collectives 时可写场景名，如 --links host_staged）
ds python -m ditserve.evaluation.cost_profile \
  --run-json /data/.cache/results-work/$ENV-cost-profile/run01/run.json \
  --hardware "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)" \
  --links /data/collectives-$ENV.json --out "$C/profiles/$ENV.json"
```

然后编辑 `configs/env/$ENV.json`：

- **`cost_profile`：** 填上 `"$ENV"`，即刚生成的画像文件名。
- **`links`：** 因为画像里已经带了实测链路，删掉这个字段（它原本是假设的场景）。
- **显存（24GB 卡要看这一项）：** CLI 打印的每个类别都带 `memory.peak_allocated_bytes` 和 `peak_reserved_bytes`。
  - Wan 最大形状的峰值加上 Sana 的峰值，再加两个 CUDA 上下文（约 1GiB），必须小于单卡显存。
  - `rtx4090-2x24g.json` 已按 A100 上的模拟设了上限：Wan `memory_fraction` 0.54（12.95GiB），Sana 0.40（9.6GiB），并开启 `expandable_segments`。
  - A100 上不加上限时，Wan 480p 峰值分配 11.4GiB、预留 15.0GiB，Sana 分别为 8.2GiB 和 10.2GiB，按预留合计超过 24GB。加上上限后，两者预留只到 12.85GiB 和 8.55GiB，速度不变（`results/a100-as-24g-cost-profile/emu01`）。
  - 4090 上的 kernel workspace 可能不同：如果这里的峰值接近上限，或者 cost_profile 报分配失败，就从 `serving_classes` 里去掉 81 帧，或者调整两个上限。

改完后，再跑一次 `ds ditserve resolve --config "$C/studies/serving.json" --env "$ENV"`，确认能解析。

## 7. 主要实验

```bash
ds ditserve run --config "$C/studies/block_exact.json"    --allow-gpu --run-id run01   # E5：同卡跨进程；两卡以上加跨卡
ds ditserve run --config "$C/studies/serving.json"        --allow-gpu --run-id run01   # E3/E7/E9：负载曲线 + 突发
ds ditserve simulate --config "$C/simulation.json" --run-id run01                     # E4：8 卡推演，只用 CPU，约 30 分钟
ds ditserve run --config "$C/studies/sp_equivalence.json" --allow-gpu --run-id run01   # 实验性：两卡以上的第一次 SP2 实跑
```

| 实验 | 回答的问题 | 通过标准 / 读法 |
| --- | --- | --- |
| block_exact | step 内的 block 边界是不是合法的暂停点 | 480p×{17,33,81} 帧，每种都在条件分支和无条件分支中途导出；新进程（两卡时为另一张卡）恢复后，视频和 latent 逐位一致 |
| serving | block 级抢占 + 可抢占解码 + 迟到短请求不降级，相比 step 级 static_sp1 的收益 | 负载 0.4/0.6/0.8 和突发 trace，比较 `by_kind` 按时率、图像 p95/p99。每个 cell 的 `simulated` 是模拟器对同一 trace 的复算，用来校准（A100 上平均误差 6.7 个百分点） |
| simulate | 8×A100 PCIe 上 block_elastic 对 step 级弹性 SP 的收益 | 与机器无关，只要结果复现；14B 成本是外推值，链路是假设场景 |
| sp_equivalence | Ulysses SP2 的数值是否与 SP1 一致 | 记录最大误差（容差 0.05）。没通过之前，不报告任何 SP 性能结论 |

## 8. 结果与收尾

结果只长期保留两个文件：`$DATA/results/<环境名>-<实验>/<run>/results.xlsx` 和 `REPORT.md`。

- 原始数组、block 包等工作数据在运行收尾时按清单删除，删除凭据写进工作簿。
- `.cache/results-work/.../run.json` 和 `.state/` 下的账本属于来源与续跑控制文件，要保留。
- 校验：`ds ditserve results verify /data/results/$ENV-acceptance/run01`。
- 中断后可以续跑：`--resume --run-id run01`。换了环境文件或机器再续跑会被拒绝，因为冻结配置里记录了 GPU 清单和环境文件的哈希。

把新机器上产生的配置提交回仓库，结果则拷回开发机归档（`results/` 不进 git）：

```bash
git add configs/profiles/$ENV.json configs/env/$ENV.json
git commit -m "Measured $ENV cost profile" && git push origin HEAD:8a100
rsync -a "$DATA/results/" <开发机>:/home/zyd/code/dit-rewrite/results/
```

## 9. 新机器的环境文件（以 8×A100 到货为例）

1. **拓扑与 P2P。**
   - 记录 `nvidia-smi topo -m`、`nvidia-smi topo -p2p r` 和 `nvidia-smi -q | grep -i -A2 "bus id"` 的输出。
   - 8 张卡应同型号、同显存。
   - P2P 读不可用时，查 PCIe switch 的 ACS（`lspci -vvv | grep -i acsctl`）和 IOMMU。
2. **GPU 顺序。**
   - 在 `gpus.uuids` 里按拓扑顺序列出 UUID，使对齐的 pair 位于同一 PCIe switch 下、quad 位于同一 socket。
   - 调度器的 SP 组就是按这个顺序取的对齐组。
3. **链路。** 第 6 步的 collectives 结果替换假设的 `links`。之后用实测画像重跑 `simulate`，对照到货前的推演。
4. **SP。** 真实 cluster 执行器目前固定 SP1。SP 组内的 block 执行和 step 边界换组要先实现，再用 `sp_equivalence` 验收，之后才能做 SP 实验。
5. **Wan2.2-TI2V-5B**（8 卡阶段的主视频模型，2026-10-18 确定）。代码目前按 Wan2.1-1.3B 的形状写死，移植时要放开这几处：
   - `models/wan/block.py` 的 `_validate_model`：30 层、12 头、16 通道；
   - `models/wan/parallel.py` 的 `_validate_values`：latent 16 通道、8 倍空间压缩；
   - `models/wan/sp.py` 的形状检查；
   - `models/pipeline.py` 的 `ChunkedDecode`：Wan2.2 的 VAE 带 `patch_size`，解码后需要 unpatchify；
   - `evaluation/cost_profile.py` 的 `ARCHITECTURES`。

   改完先在 CPU tiny 模型上测试，再按第 5–6 步重新验收、重测成本。

## 10. 在开发机上不用容器运行

开发机已有锁定版本的 venv（`.venv-previs`）。模型通过 `.cache/models` 下的符号链接指向共享快照，CLI 用 `DITSERVE_MODELS_ROOT` 找到它们：

```bash
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release && cmake --build .cache/build -j8 && (cd .cache/build && ctest)
PYTHONPATH=workers CUDA_VISIBLE_DEVICES= .venv-previs/bin/python -m unittest discover -s tests/python
mkdir -p .cache/models   # 链接到上面第 2 步列出的两个快照目录，链接名分别为 wan2.1-t2v-1.3b 和 sana-sprint-0.6b
export DITSERVE_MODELS_ROOT=$PWD/.cache/models
.cache/build/ditserve doctor --env a100-1x40g --allow-gpu
.cache/build/ditserve run --config configs/studies/acceptance.json --env a100-1x40g --allow-gpu --run-id run01
```

这台机器的 A100 和其他用户共享。运行前控制器会等 GPU 空闲（`admission_wait_seconds`），运行中一旦发现外来进程就中止。
