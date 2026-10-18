# 容器镜像与迁移

`Dockerfile` 只构建一个运行镜像 `runtime`：C++ `ditserve`（构建阶段跑 ctest 后 `cmake --install --strip`）、`/opt/ditserve/share/ditserve` 下的 Python worker 与 configs，以及按 `docker/artifacts-runtime.json` 逐个校验 SHA256 安装的 `/opt/venv`。不含 tests、docs、results、`.cache`、`.migration` 或可编辑安装。

精简范围：

- 体积约 89% 是 torch 2.10.0+cu126、nvidia-* 与 triton wheel；`libtorch_cuda.so` 直接链接 cuSPARSELt、NVSHMEM、cuFile 等库，不能删。
- 锁文件去掉当前代码不引用的 imageio、imageio-ffmpeg 和 wheel；安装后只删除 Triton 静态 CUPTI 归档 `*.a`。不要删各包 `tests/`：`numpy.testing` 依赖 `numpy._core.tests`，diffusers/cache-dit 导入会失败。
- 保留 gcc/g++ 与 Python 头文件，torch.compile/Triton 运行时需要。
- LongLive 需要另一套 torch 2.8 环境（约 +7.5 GB），不在默认镜像中。

## 构建

依赖网络不可用时，用本机 wheelhouse 离线构建（文件名须与锁文件一致）：

```bash
python3 -m http.server 18765 --bind 127.0.0.1 --directory /path/to/wheelhouse &
DOCKER_BUILDKIT=0 docker build --network host --target runtime \
  --build-arg PYTHON_BASE=ditserve-python-base:3.10.20-verified \
  --build-arg DITSERVE_WHEELHOUSE_URL=http://127.0.0.1:18765 \
  --build-arg GIT_REVISION="$(git rev-parse HEAD)" \
  -t ditserve:rewrite-runtime .
```

不传 `PYTHON_BASE` 时使用固定 digest 的官方 `python:3.10.20-slim-bookworm`；不传 `DITSERVE_WHEELHOUSE_URL` 时从锁文件中的官方 URL 下载。旧版 builder 把 build-arg 计入缓存键，`DITSERVE_WHEELHOUSE_URL` 与上次不同就会重装依赖层（结果相同，均按 SHA256 校验）。

## 打包与加载

```bash
docker save ditserve:rewrite-runtime | pigz -6 > ditserve-rewrite-runtime.tar.gz
sha256sum ditserve-rewrite-runtime.tar.gz > SHA256SUMS
# 目标机器
sha256sum -c SHA256SUMS && docker load -i ditserve-rewrite-runtime.tar.gz
```

## 运行

`DITSERVE_ROOT=/data`，`results/`、`.state/`、`.cache/results-work/` 都写在其下，需挂载才能保留；模型放 `/models`。

```bash
docker run --rm --gpus all --pid=host --user "$(id -u):$(id -g)" \
  -v /host/ditserve-data:/data -v /host/models:/models:ro \
  ditserve:rewrite-runtime ditserve run --config /data/study.json --run-id trial-01 --allow-gpu
```

`--pid=host` 是 GPU 实验的必需参数：GPU 隔离检查会把 `nvidia-smi` 报告的宿主 PID 与 worker PID 比对。在独立 PID 命名空间里，自己的 worker 会被当成外来进程，运行以 "GPU isolation lost" 失败。本机 A100 冒烟测试已复现这个问题，并验证加上该参数后通过（`results/a100-image-smoke/smoke01`、`smoke02`）。

多卡（SP、NCCL all-to-all、`torchrun`）须加 `--ipc=host --shm-size=32g --ulimit memlock=-1`，否则 NCCL/共享内存会报错或退化。8×A100 到货验收步骤见 [migration_8xa100.md](migration_8xa100.md)。镜像不含模型权重，需在容器外预下载到 `/models` 挂载目录；容器内保持 `HF_HUB_OFFLINE=1`。
