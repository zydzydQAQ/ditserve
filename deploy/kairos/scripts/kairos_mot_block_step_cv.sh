#!/bin/bash
# Motivation Step- & Block-Level Predictability (CV%) Collector on 8x A100-PCIE-40GB:
# Runs either Shard s0 (Wan2.2 720p x 7 forms) or Shard s1 (Wan2.2 256p/480p x 7 + SD3.5 x 15)
set -euo pipefail

check_storage() {
  df -h /shared
  df -BG /shared | python3 -c 'import sys; r=sys.stdin.read().splitlines()[-1].split(); assert int(r[2].rstrip("G"))<=805 and int(r[4].rstrip("%"))<81, f"shared storage stop line reached: used={r[2]} pct={r[4]}"'
}

check_storage
source "$(dirname "$0")/kairos_common.sh"
check_storage

: "${DITSERVE_MOT_CV_SHARD:?set DITSERVE_MOT_CV_SHARD to s0, s1, or all}"
export PYTHONPATH=$PACKAGE/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi -L
nvidia-smi topo -m | head -12
df -h /dev/shm /shared

timeout "${DITSERVE_MOT_CV_TIMEOUT_S:-3600}s" /opt/venv/bin/torchrun --standalone --nproc_per_node=8 \
  "$SCRIPTS/run_mot_block_step_cv.py" \
  --output-dir "$RUN_DIR" \
  --shard "$DITSERVE_MOT_CV_SHARD" \
  --report-name "20261116-mot-block-step-cv-${DITSERVE_MOT_CV_SHARD}.md"
