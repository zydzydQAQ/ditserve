#!/bin/bash
set -euo pipefail

check_storage() {
  df -h /shared
  df -BG /shared | python3 -c 'import sys; r=sys.stdin.read().splitlines()[-1].split(); assert int(r[2].rstrip("G"))<=790 and int(r[4].rstrip("%"))<80, "shared storage stop line reached"'
}

check_storage

# Step 1: Run formal serve batch via the immutable 720da72 operator script
# (also materializes $OUTPUT/acceptance/$DITSERVE_RUN_ID/bundle.json).
bash "${DITSERVE_KAIROS_RUNTIME_ROOT}/scripts/kairos_formal_benchmark.sh"

if [ -z "${DITSERVE_SOAK_POLICIES:-}" ]; then
  exit 0
fi

# Step 2: Run 5-phase 500-request sustained SLA & HBM/SHM stability soak policies
OUTPUT=/shared/output/DITSERVE
DATA=/shared/data/DITSERVE
MODELS=$DATA/models
CODE=/shared/code/DITSERVE
BUILD=$CODE/builds/$DITSERVE_BUILD
PACKAGE=$BUILD/share/ditserve
RUN_DIR=$OUTPUT/acceptance/$DITSERVE_RUN_ID
LEDGERS=/tmp/ditserve-ledgers-soak
export DITSERVE_TRANSFER_DIR=/dev/shm/ditserve-transfer-${DITSERVE_RUN_ID}-soak
export PATH=$BUILD/bin:$PATH
export DITSERVE_PACKAGE_ROOT=$PACKAGE/python
export PYTHONPATH=$PACKAGE/python
export DITSERVE_ROOT=$OUTPUT HOME=$OUTPUT
export HF_HOME=$OUTPUT/cache/huggingface TORCH_HOME=$MODELS/torch
export TORCHINDUCTOR_CACHE_DIR=$OUTPUT/cache/torchinductor TRITON_CACHE_DIR=$OUTPUT/cache/triton
export MPLCONFIGDIR=$OUTPUT/cache/matplotlib
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
umask 0000

mkdir -p "$RUN_DIR/ledgers" "$LEDGERS" "$DITSERVE_TRANSFER_DIR"
exec > >(tee -a "$RUN_DIR/kairos_soak_5phase.log") 2>&1

cleanup_soak() {
  if [ -n "${HW_MON_PID:-}" ]; then
    kill "$HW_MON_PID" 2>/dev/null || true
    wait "$HW_MON_PID" 2>/dev/null || true
  fi
  rm -rf "$DITSERVE_TRANSFER_DIR" 2>/dev/null || true
  cp -a "$LEDGERS/." "$RUN_DIR/ledgers/" 2>/dev/null || true
  chmod -R a+rX "$RUN_DIR" 2>/dev/null || true
}
trap cleanup_soak EXIT

start_hw_monitor() {
  local csv_out=$1
  python3 - "$csv_out" <<'PY' &
import pathlib, subprocess, sys, time

out_path = pathlib.Path(sys.argv[1])
t0 = time.monotonic()
with out_path.open("w", encoding="utf-8", buffering=1) as f:
    f.write("elapsed_s,gpu_used_mib_min,gpu_used_mib_mean,gpu_used_mib_max,gpu_util_mean_pct,shm_used_mb,shm_files,host_rss_mb\n")
    while True:
        try:
            el = round(time.monotonic() - t0, 1)
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True,
                timeout=10,
            ).strip().splitlines()
            mems, utils = [], []
            for ln in smi:
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) >= 2:
                    mems.append(float(parts[0]))
                    utils.append(float(parts[1]))
            g_min = round(min(mems), 1) if mems else 0.0
            g_mean = round(sum(mems) / len(mems), 1) if mems else 0.0
            g_max = round(max(mems), 1) if mems else 0.0
            u_mean = round(sum(utils) / len(utils), 1) if utils else 0.0

            shm_df = subprocess.check_output(["df", "-BM", "/dev/shm"], text=True, timeout=5).splitlines()[-1].split()
            shm_mb = int(shm_df[2].rstrip("M"))
            shm_files = sum(1 for _ in pathlib.Path("/dev/shm").iterdir())

            ps_out = subprocess.check_output(["ps", "-eo", "rss="], text=True, timeout=5).splitlines()
            rss_mb = round(sum(int(x.strip()) for x in ps_out if x.strip().isdigit()) / 1024.0, 1)

            f.write(f"{el},{g_min},{g_mean},{g_max},{u_mean},{shm_mb},{shm_files},{rss_mb}\n")
        except Exception:
            pass
        time.sleep(5.0)
PY
  HW_MON_PID=$!
}

IFS=',' read -ra soak_pols <<< "$DITSERVE_SOAK_POLICIES"
for pol in "${soak_pols[@]}"; do
  [ -n "$pol" ] || continue
  check_storage
  pol_slug=${pol//_/-}
  cfg="$RUN_DIR/serving-soak-5phase-${pol_slug}.json"
  hw_csv="$RUN_DIR/soak-hw-telemetry-${pol_slug}.csv"
  run_sub_id="${DITSERVE_RUN_ID}-soak-${pol_slug}"

  echo "== building 5-phase 500-req soak config for policy=$pol ($(date -Is))"
  python3 /shared/data/DITSERVE/scripts/build_soak_5phase_config.py \
    --benchmarks-dir "$DITSERVE_BENCHMARK_ROOT" \
    --protocol "$DITSERVE_FORMAL_PROTOCOL" \
    --model wan22 \
    --bundle "$RUN_DIR/bundle.json" \
    --datasets-dir "$DATA/datasets" \
    --manifests "$DITSERVE_MODEL_MANIFESTS" \
    --policy "$pol" \
    --seed "${DITSERVE_SEED:-301}" \
    --slo-scale 1.5 \
    --output "$cfg"

  experiment=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment"])' "$cfg")
  HW_MON_PID=
  start_hw_monitor "$hw_csv"

  echo "== running 5-phase 500-req soak: run_id=$run_sub_id experiment=$experiment ($(date -Is))"
  (
    trap 'chmod -R a+rX "$OUTPUT/.cache/results-work/$experiment/$run_sub_id" "$OUTPUT/results/$experiment/$run_sub_id" 2>/dev/null || true' EXIT
    ditserve run --config "$cfg" --run-id "$run_sub_id" --allow-gpu --ledger "$LEDGERS/$experiment.json"
    python3 -c 'import sys; from pathlib import Path; from ditserve.results import archive_run; archive_run(Path(sys.argv[1]), Path(sys.argv[2]))' \
      "$OUTPUT/.cache/results-work/$experiment/$run_sub_id" \
      "$OUTPUT/results/$experiment/$run_sub_id"
    ditserve results verify "$OUTPUT/results/$experiment/$run_sub_id"
  )

  if [ -n "${HW_MON_PID:-}" ]; then
    kill "$HW_MON_PID" 2>/dev/null || true
    wait "$HW_MON_PID" 2>/dev/null || true
    HW_MON_PID=
  fi
  chmod a+r "$hw_csv" "$cfg" 2>/dev/null || true
  echo "== completed 5-phase 500-req soak for policy=$pol ($(date -Is))"
done
