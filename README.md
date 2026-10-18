# DiT Serving Lab

研究 DiT 视频/图像混合负载的服务调度：在 Wan DiT 的 step 内按 block 抢占，VAE 解码按 latent 帧分块并可抢占，调度采用可行优先 EDF（迟到的短请求保留 EDF 位置），目标是多卡 PCIe 上的弹性 Ulysses SP。

C++17 控制器负责工作负载、调度、阶段、预算和进程监督；常驻 Python worker 持有模型和张量，二者经 JSONL 交换不透明的状态 ID。

```text
src/ditserve/          C++：control/（study、cluster 调度、environment 绑定、budget、dispatch）、interface/、runtime/
include/ditserve/      C++ 接口
cmd/                   main
benchmarks/            集群模拟器，与真实执行器共用 control/cluster.cpp
workers/ditserve/      Python：models/（wan、sana、mock）、runtime/、evaluation/（数值对比、成本画像、collectives）、results/
configs/env/           每台机器一个环境文件：GPU、模型路径、worker 环境、成本画像、请求类别
configs/studies/       与机器无关的实验：acceptance、cost_profile、block_exact、serving、sp_equivalence、mock
configs/profiles/      实测成本画像与假设链路；configs/simulation.json 是 8 卡推演
configs/models/        模型文件的 SHA256
tests/                 C++ ctest 与 Python CPU 契约测试
docs/                  预实验结论、实验配置说明、结果处理、论文草稿
```

模型：Wan2.1-T2V-1.3B（视频）和 SANA-Sprint 0.6B（图像）。依赖锁定为 torch 2.10.0+cu126、diffusers 0.36.0、cache-dit 1.5.0。

## 新机器

按 [MIGRATION.md](MIGRATION.md) 执行：拉取代码、构建镜像、下载并校验模型、机器检查，再依次跑验收、成本画像和主要实验。开发机（1×A100）、2×RTX 4090、8×A100 共用同一套代码和配置，只换 `--env`。

## 开发

需要 Linux、C++17、CMake 3.16+、OpenSSL、Python 3.8+。nlohmann/json 固定为 3.11.3 并校验 SHA256；编译保留 `-ffp-contract=off`。

```bash
cmake -S . -B .cache/build -DCMAKE_BUILD_TYPE=Release && cmake --build .cache/build -j8
(cd .cache/build && ctest --output-on-failure)
PYTHONPATH=workers CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .venv-previs/bin/python -m unittest discover -s tests/python
```

没装 torch/diffusers 时，涉及模型的数值用例会跳过。

```bash
.cache/build/ditserve run --config configs/studies/mock.json --run-id demo     # CPU：两张 mock 卡的调度冒烟
.cache/build/ditserve simulate --count 32 --run-id demo-sim                    # CPU：合成成本的模拟
.cache/build/ditserve resolve --config configs/studies/serving.json --env a100-1x40g   # 看实验在本机绑定成什么
.cache/build/ditserve doctor --env a100-1x40g [--allow-gpu]
.cache/build/ditserve run --config configs/studies/acceptance.json --env a100-1x40g --allow-gpu --run-id run01
```

真实模型运行必须加 `--allow-gpu`，并在配置里给出 GPU 秒预算。从阶段开始到清理确认，所选 GPU 全程计费，失败也计入。每次运行只长期保留 `results/<环境>-<实验>/<run>/results.xlsx` 和 `REPORT.md`。

[预实验结论](docs/a100_preexperiments.md) · [实验配置](docs/experiments.md) · [结果处理](docs/results.md) · [设计与精简记录](docs/rewrite.md)
