#!/bin/bash
# Challenge 1 & Motivation 1 Pre-Experiment on 8x A100-PCIE-40GB (PHB):
# - Pure SP (SP1, SP2, SP4, SP8) scaling & communication breakdown for Image (SD3.5-Medium) & Video (Wan2.2-5B)
# - CFG2 / CFG2xSP vs 1-GPU Fused CFG & 1-GPU Continuous Batching + cu_seqlens Packing
source "$(dirname "$0")/kairos_common.sh"
export PYTHONPATH=$PACKAGE/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -L
nvidia-smi topo -m | head -12
df -h /dev/shm /shared
used=$(df -BG /shared | awk 'NR==2{gsub("G","",$3); print $3}'); pct=$(df -P /shared | awk 'NR==2{gsub("%","",$5); print $5}')
if [ "$used" -gt 790 ] || [ "$pct" -ge 80 ]; then echo "shared storage gate: used=${used}G pct=${pct}%"; exit 4; fi
timeout "${DITSERVE_MOT1_TIMEOUT_S:-3300}s" /opt/venv/bin/torchrun --standalone --nproc_per_node=8 \
  "$SCRIPTS/run_mot1_sp_cfg_study.py" \
  --output-dir "$RUN_DIR" \
  --report-name "20261111-mot1-sp-cfg-study.md"
