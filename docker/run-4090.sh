#!/usr/bin/env bash
# Run the pinned runtime against this checkout's two-4090 configuration.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
image=${DITSERVE_IMAGE:-ditserve:rewrite-runtime}
docker_host=${DITSERVE_DOCKER_HOST:-${DOCKER_HOST:-unix:///run/ditserve-docker/docker.sock}}
models_dir=${DITSERVE_MODELS_DIR:-/mnt/ssd0/zyd/model}
data_dir=${DITSERVE_DATA_DIR:-/mnt/ssd0/zyd/ditserve-data}
container_configs=/opt/ditserve/share/ditserve/configs

if ! command -v docker >/dev/null 2>&1; then
    printf '%s\n' 'docker is not installed or is not on PATH.' >&2
    exit 1
fi
if [[ ! -d "$models_dir" ]]; then
    printf 'Model directory does not exist: %s\n' "$models_dir" >&2
    exit 1
fi

mkdir -p -- "$data_dir"/{results,.state,tmp} \
    "$data_dir"/cache/{huggingface,torch,torch_extensions,torchinductor,triton,cuda,xdg,pip}
models_dir=$(cd -- "$models_dir" && pwd -P)
data_dir=$(cd -- "$data_dir" && pwd -P)

# Keep the real utility separate from NVIDIA Container Toolkit's own mounts.
nvidia_smi=${DITSERVE_NVIDIA_SMI-"$data_dir/tools/nvidia-smi"}
host_tool_args=()
if [[ ${DITSERVE_NVIDIA_SMI+x} && ( ! -f "$nvidia_smi" || ! -x "$nvidia_smi" ) ]]; then
    printf 'DITSERVE_NVIDIA_SMI must name an executable file: %s\n' "$nvidia_smi" >&2
    exit 1
fi
if [[ -f "$nvidia_smi" && -x "$nvidia_smi" ]]; then
    nvidia_smi=$(realpath -- "$nvidia_smi")
    host_tool_args=(
        -v "$nvidia_smi:/usr/local/bin/nvidia-smi:ro"
        -e PATH=/opt/ditserve/bin:/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
    )
fi

# GPU work remains explicit: callers add --allow-gpu to ditserve commands.
if [[ $# -eq 0 ]]; then
    set -- ditserve doctor
fi

exec docker --host "$docker_host" run --rm --gpus all --network host --pid=host --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" --workdir /data \
    -e DITSERVE_ENV=rtx4090-2x24g \
    -e DITSERVE_ROOT=/data -e DITSERVE_MODELS_ROOT=/models \
    -e NCCL_P2P_DISABLE=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HOME=/data -e TMPDIR=/data/tmp \
    -e HF_HOME=/data/cache/huggingface \
    -e HF_HUB_CACHE=/data/cache/huggingface/hub \
    -e TORCH_HOME=/data/cache/torch \
    -e TORCH_EXTENSIONS_DIR=/data/cache/torch_extensions \
    -e TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    -e TRITON_CACHE_DIR=/data/cache/triton \
    -e CUDA_CACHE_PATH=/data/cache/cuda \
    -e XDG_CACHE_HOME=/data/cache/xdg -e PIP_CACHE_DIR=/data/cache/pip \
    -v "$models_dir:/models:ro" \
    -v "$data_dir:/data" \
    -v "$repo_root/configs:$container_configs" \
    "${host_tool_args[@]}" \
    "$image" "$@"
