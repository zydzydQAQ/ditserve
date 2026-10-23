#!/usr/bin/env bash
# Isolated runner for the opt-in workflow experiments and hardware counters.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source_dir=${DITSERVE_WORKFLOW_SOURCE_DIR:-$repo_root}
data_dir=${DITSERVE_DATA_DIR:-/mnt/ssd0/zyd/ditserve-data}
models_dir=${DITSERVE_MODELS_DIR:-/mnt/ssd0/zyd/model}
docker_host=${DITSERVE_DOCKER_HOST:-unix:///run/ditserve-docker/docker.sock}
image=${DITSERVE_IMAGE:-ditserve:rewrite-runtime}
counter_args=()
if [[ ${1-} == --counters ]]; then
    counter_args=(--user 0:0 --cap-add SYS_ADMIN)
    shift
else
    counter_args=(--user "$(id -u):$(id -g)")
fi
if [[ $# == 0 ]]; then
    printf '%s\n' 'Usage: run-workflow-profile.sh [--counters] COMMAND [ARG ...]' >&2
    exit 2
fi
mkdir -p -- "$data_dir/workflow" "$data_dir/tmp" "$data_dir/cache/torch" "$data_dir/cache/huggingface"
name_args=()
if [[ -n ${DITSERVE_WORKFLOW_CONTAINER_NAME-} ]]; then
    name_args=(--name "$DITSERVE_WORKFLOW_CONTAINER_NAME")
fi
exec docker --host "$docker_host" run --rm --gpus all --network none \
    --pid=host --shm-size=32g --ulimit memlock=-1 --workdir /data \
    "${counter_args[@]}" "${name_args[@]}" \
    -e PYTHONPATH=/workspace/workers:/opt/ditserve/share/ditserve/python \
    -e PATH=/opt/ditserve/bin:/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
    -e TOKENIZERS_PARALLELISM=false -e NCCL_P2P_DISABLE=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e DITSERVE_MODELS_ROOT=/models -e DITSERVE_ROOT=/data \
    -e HF_HOME=/data/cache/huggingface -e TORCH_HOME=/data/cache/torch \
    -e TMPDIR=/data/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$source_dir:/workspace:ro" -v "$data_dir:/data" -v "$models_dir:/models:ro" \
    -v "$data_dir/tools/nvidia-smi:/usr/local/bin/nvidia-smi:ro" \
    -v /opt/nvidia/nsight-systems/2023.4.4:/opt/nvidia/nsight-systems/2023.4.4:ro \
    -v /opt/nvidia/nsight-compute/2024.1.0:/opt/nvidia/nsight-compute/2024.1.0:ro \
    "$image" "$@"
