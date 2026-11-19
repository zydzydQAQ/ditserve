#!/bin/bash
# Item 7 (review 20261105): interleaved Ulysses all-to-all routing (GFC vs NCCL) and bitwise exactness,
# Wan2.1-T2V-1.3B, SP2, cfg=1, real WanBackend workers called directly. One dated report + summary.json
# under /shared/output/DITSERVE/acceptance/$DITSERVE_RUN_ID/; block exports stay in container /dev/shm.
source "$(dirname "$0")/kairos_common.sh"
export PYTHONPATH=$PACKAGE/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
nvidia-smi topo -m | head -4
df -h /dev/shm /shared
used=$(df -BG /shared | awk 'NR==2{gsub("G","",$3); print $3}'); pct=$(df -P /shared | awk 'NR==2{gsub("%","",$5); print $5}')
if [ "$used" -gt 790 ] || [ "$pct" -ge 80 ]; then echo "shared storage gate: used=${used}G pct=${pct}%"; exit 4; fi
timeout "${DITSERVE_VAL_TIMEOUT_S:-3300}s" /opt/venv/bin/python -u "$SCRIPTS/run_val_i7_interleave.py" \
  --output-dir "$RUN_DIR" --report-name "20261105-val-i7-interleaved-ulysses-${DITSERVE_RUN_ID}.md" \
  ${DITSERVE_VAL_I7_ARGS:-}
