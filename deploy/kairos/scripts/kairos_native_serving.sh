#!/bin/bash
source "$(dirname "$0")/kairos_common.sh"
export PYTHONPATH=$PACKAGE/python
if [ "${DITSERVE_CALIBRATE:-0}" = 1 ]; then
  kairos_calibrate_native_profile
  export DITSERVE_PROFILE=$RUN_DIR/profile.json DITSERVE_REQUIRE_PROFILE_MATCH=1
fi
: "${DITSERVE_PROFILE:?set DITSERVE_PROFILE to the fitted profile in shared storage}"
case "$WAN_MODEL" in
  wan) MODEL=wan21 ;;
  wan2.2-ti2v-5b) MODEL=wan22 ;;
  *) echo "unknown WAN_MODEL $WAN_MODEL"; exit 1 ;;
esac
seed=${DITSERVE_SEED:-201}
load=${DITSERVE_LOAD:-1.0}
pattern=${DITSERVE_PATTERN:-poisson}
cv=${DITSERVE_CV:-1.0}
num_gpus=${DITSERVE_NUM_GPUS:-}
fraction=${DITSERVE_VIDEO_FRACTION:-0.2}
count=${DITSERVE_COUNT:-40}
suite=${DITSERVE_SUITE:-main}
policies=${DITSERVE_POLICIES:-}
forms=${DITSERVE_FORMS:-'sp1,cfg2,sp2,cfg2sp2,sp3,cfg2sp3,sp4,cfg2sp4,sp6,sp8'}
if [ "$suite" = preempt ] || [ "${DITSERVE_NO_PREEMPTION:-0}" = 1 ]; then
  forms=${DITSERVE_FORMS:-cfg2}
fi
if [ -n "${DITSERVE_QUALITY_ARM:-}" ]; then
  forms=${DITSERVE_FORMS:-cfg2sp3}
fi
if [ "${DITSERVE_NO_PREEMPTION:-0}" = 1 ]; then
  load=0.1 pattern=poisson fraction=1.0
fi
quality=${DITSERVE_QUALITY:-0}
video_slo=${DITSERVE_VIDEO_SLO_SCALE:-1.5}
image_slo=${DITSERVE_IMAGE_SLO_SCALE:-2.5}
experiment=${DITSERVE_EXPERIMENT:-kairos-$MODEL-native-serving-s${seed}-l${load//./p}}
args=(
  --model "$MODEL"
  --experiment "$experiment"
  --profile "$DITSERVE_PROFILE"
  --datasets "$DATA/datasets"
  --seed "$seed"
  --load "$load"
  --pattern "$pattern"
  --cv "$cv"
  --video-fraction "$fraction"
  --count "$count"
  --forms "$forms"
  --video-slo-scale "$video_slo"
  --image-slo-scale "$image_slo"
  --quality-per-kind "$quality"
  --output-mode "${DITSERVE_OUTPUT_MODE:-check}"
  --collective-backend "${DITSERVE_COLLECTIVE_BACKEND:-nccl}"
  --image-batch "${DITSERVE_IMAGE_BATCH:-1}"
  --policy-order "${DITSERVE_POLICY_ORDER:-random}"
  --order-seed "${DITSERVE_ORDER_SEED:-$seed}"
  --output "$RUN_DIR/study.json"
)
if [ -n "${DITSERVE_QUALITY_ARM:-}" ]; then
  args+=(--quality-arm "$DITSERVE_QUALITY_ARM")
fi
if [ "${DITSERVE_NO_PREEMPTION:-0}" = 1 ]; then
  args+=(--no-preemption-diagnostic)
fi
if [ -n "$num_gpus" ]; then
  args+=(--num-gpus "$num_gpus")
fi
if [ -n "${DITSERVE_QUALITY_ARM:-}" ]; then
  : # The quality arm owns its policy and fold configuration.
elif [ -n "$policies" ]; then
  args+=(--policies "$policies")
else
  args+=(--suite "$suite")
fi
if [ "${DITSERVE_CURVATURE:-0}" = 1 ]; then
  args+=(--curvature)
fi
if [ -n "${DITSERVE_FOLD_STEPS:-}" ]; then
  args+=(--fold-steps "$DITSERVE_FOLD_STEPS")
fi
if [ "${DITSERVE_DOORBELL:-0}" = 1 ]; then
  args+=(--doorbell)
fi
if [ "${DITSERVE_REQUIRE_PROFILE_MATCH:-0}" = 1 ]; then
  args+=(--require-profile-match)
fi
python3 "$PACKAGE/benchmarks/native_serving.py" "${args[@]}"
nvidia-smi -L | tee "$RUN_DIR/gpu-list.txt"
nvidia-smi topo -m | tee "$RUN_DIR/topology.txt"
df -h /shared | tee "$RUN_DIR/shared-storage.txt"
monitor_pid=
cleanup() {
  rc=$?
  if [ -n "${monitor_pid:-}" ]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  cp -a "$LEDGERS/." "$RUN_DIR/ledgers/" || true
  chmod -R a+rX "$OUTPUT"/.cache/results-work/*/"$DITSERVE_RUN_ID" "$RUN_DIR" 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT
python3 "$SCRIPTS/kairos_runtime_audit.py" --memory-summary "$RUN_DIR/gpu-memory-summary.json" &
monitor_pid=$!
sleep 0.2
kill -0 "$monitor_pid" || { echo "nvidia-smi memory sampler exited early"; exit 1; }
rc=0
ditserve run --config "$RUN_DIR/study.json" --run-id "$DITSERVE_RUN_ID" --allow-gpu \
  --ledger "$LEDGERS/$experiment.json" || rc=$?
if [ "$rc" -ne 0 ] && [ -n "${DITSERVE_QUALITY_ARM:-}" ]; then
  work=$OUTPUT/.cache/results-work/$experiment/$DITSERVE_RUN_ID
  python3 "$PACKAGE/benchmarks/native_quality_pending.py" \
    --run "$work/run.json" --journal "$work/.finalization.json" && rc=0
fi
if [ "$rc" -eq 0 ]; then
  ditserve results verify "$OUTPUT/results/$experiment/$DITSERVE_RUN_ID" || rc=$?
fi
if [ "$rc" -eq 0 ] && { [ "$suite" = preempt ] || [ "${DITSERVE_NO_PREEMPTION:-0}" = 1 ]; }; then
  exact_args=(--run "$OUTPUT/.cache/results-work/$experiment/$DITSERVE_RUN_ID/run.json"
              --output "$RUN_DIR/exact-summary.json")
  if [ "${DITSERVE_NO_PREEMPTION:-0}" = 1 ]; then
    exact_args+=(--no-preemption)
  fi
  python3 "$PACKAGE/benchmarks/native_exact.py" "${exact_args[@]}" || rc=$?
fi
exit "$rc"
