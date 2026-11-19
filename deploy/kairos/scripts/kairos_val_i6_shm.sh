#!/bin/bash
# Item 6 (review 20261105): shared /dev/shm encoder-pool race on real SD3.5 workers, 1 process per GPU.
# Calls the worker backends directly (no native study config). Writes one dated report + summary.json
# under /shared/output/DITSERVE/acceptance/$DITSERVE_RUN_ID/; scratch stays in container /dev/shm.
source "$(dirname "$0")/kairos_common.sh"
export PYTHONPATH=$PACKAGE/python:$PACKAGE/benchmarks
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
trap 'rm -rf /dev/shm/ditserve-val-i6-"$DITSERVE_RUN_ID" /dev/shm/ditserve-hetero-embeds "$DITSERVE_TRANSFER_DIR" 2>/dev/null || true; cp -a "$LEDGERS/." "$RUN_DIR/ledgers/" 2>/dev/null || true; chmod -R a+rX "$RUN_DIR" 2>/dev/null || true' EXIT
nvidia-smi -L
df -h /dev/shm /shared
echo "memlock=$(ulimit -l) nproc=$(nproc) cgroup_memory_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"
# Operator storage gate (stricter than the AGENTS 810G/82% stop line).
used=$(df -BG /shared | awk 'NR==2{gsub("G","",$3); print $3}'); pct=$(df -P /shared | awk 'NR==2{gsub("%","",$5); print $5}')
if [ "$used" -gt 790 ] || [ "$pct" -ge 80 ]; then echo "shared storage gate: used=${used}G pct=${pct}%"; exit 4; fi
timeout "${DITSERVE_VAL_TIMEOUT_S:-3300}s" /opt/venv/bin/python -u "$SCRIPTS/run_val_i6_shm_pool.py" \
  --output-dir "$RUN_DIR" --report-name "20261105-val-i6-encoder-shm-race-${DITSERVE_RUN_ID}.md" \
  --timeout-s "${DITSERVE_VAL_INNER_TIMEOUT_S:-3000}" ${DITSERVE_VAL_I6_ARGS:-}
