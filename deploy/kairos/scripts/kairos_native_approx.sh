#!/bin/bash
source "$(dirname "$0")/kairos_common.sh"
base=$DITSERVE_RUN_ID
case "$WAN_MODEL" in wan) MODEL=wan21 ;; *) MODEL=wan22 ;; esac
load=${DITSERVE_LOAD:-1.0}
experiment=${DITSERVE_EXPERIMENT:-kairos-$MODEL-native-serving-s${DITSERVE_SEED:-201}-l${load//./p}}
export DITSERVE_EXPERIMENT=$experiment
export DITSERVE_SUITE=main DITSERVE_NO_PREEMPTION=0
export DITSERVE_VIDEO_FRACTION=${DITSERVE_VIDEO_FRACTION:-1.0}
export DITSERVE_COUNT=${DITSERVE_COUNT:-32}
export DITSERVE_QUALITY=$DITSERVE_COUNT
export DITSERVE_OUTPUT_MODE=check
if [ "${DITSERVE_CALIBRATE:-1}" = 1 ]; then
  kairos_calibrate_native_profile
  export DITSERVE_PROFILE=$RUN_DIR/profile.json
else
  : "${DITSERVE_PROFILE:?set DITSERVE_PROFILE to a GPU-bound profile}"
fi
export DITSERVE_CALIBRATE=0 DITSERVE_REQUIRE_PROFILE_MATCH=1
# These are pilot reference-relative gates, not calibrated semantic-quality guarantees.
gate_args=(
  --min-psnr "${DITSERVE_MIN_PSNR:-30.5}"
  --min-ssim "${DITSERVE_MIN_SSIM:-0.94}"
  --max-lpips "${DITSERVE_MAX_LPIPS:-0.10}"
  --lpips-device "${DITSERVE_LPIPS_DEVICE:-cpu}"
  --bootstrap-samples "${DITSERVE_BOOTSTRAP_SAMPLES:-2000}"
)

# The exact reference and every approximate arm start from cfg2sp3. Fixed-N uses
# the first N steps of the same 12..18 window; step 19 remains an exact anchor.
DITSERVE_RUN_ID=$base-exact DITSERVE_FORMS=cfg2sp3 \
  DITSERVE_QUALITY_ARM=exact DITSERVE_FOLD_STEPS= DITSERVE_CURVATURE=0 \
  bash "$SCRIPTS/kairos_native_serving.sh"
budgets=${DITSERVE_FIXED_BUDGETS:-"1 2 3 4 5 6 7"}
declare -A seen=()
for budget in $budgets; do
  if [[ ! $budget =~ ^[1-7]$ ]] || [ -n "${seen[$budget]:-}" ]; then
    echo "DITSERVE_FIXED_BUDGETS must contain unique integers in 1..7" >&2
    exit 2
  fi
  seen[$budget]=1
  steps=$(seq -s, 12 $((11 + budget)))
  DITSERVE_RUN_ID=$base-fixed$budget DITSERVE_FORMS=cfg2sp3 \
    DITSERVE_QUALITY_ARM=fixed DITSERVE_FOLD_STEPS=$steps DITSERVE_CURVATURE=0 \
    bash "$SCRIPTS/kairos_native_serving.sh"
done
DITSERVE_RUN_ID=$base-curvature DITSERVE_FORMS=cfg2sp3 \
  DITSERVE_QUALITY_ARM=curvature DITSERVE_FOLD_STEPS= DITSERVE_CURVATURE=0 \
  bash "$SCRIPTS/kairos_native_serving.sh"

export PYTHONPATH=$PACKAGE/python
work=$OUTPUT/.cache/results-work/$experiment
reference_run=$work/$base-exact/run.json
quality_rc=0
for candidate in 0 $budgets; do
  if [ "$candidate" = 0 ]; then
    candidate_run=$reference_run
  else
    candidate_run=$work/$base-fixed$candidate/run.json
  fi
  if [ ! -f "$reference_run" ] || [ ! -f "$candidate_run" ]; then
    echo "quality pending: source run.json unavailable for fixed$candidate; qualification is unverified" >&2
    quality_rc=1
    continue
  fi
  # Fixed controls retain pairs so a later curvature run can be paired by identity.
  python3 "$PACKAGE/benchmarks/native_quality.py" \
    --reference-run "$reference_run" --candidate-run "$candidate_run" \
    --output "$RUN_DIR/quality-fixed$candidate.json" \
    --expected-fold-count "$candidate" --keep-pairs "${gate_args[@]}" || quality_rc=$?
done
controls=()
for candidate in 0 $budgets; do
  if [ -f "$RUN_DIR/quality-fixed$candidate.json" ]; then
    controls+=(--control-quality "$RUN_DIR/quality-fixed$candidate.json")
  fi
done
curvature_run=$work/$base-curvature/run.json
if [ -f "$reference_run" ] && [ -f "$curvature_run" ]; then
  python3 "$PACKAGE/benchmarks/native_quality.py" \
    --reference-run "$reference_run" --candidate-run "$curvature_run" \
    --output "$RUN_DIR/quality-curvature.json" \
    "${controls[@]}" "${gate_args[@]}" || quality_rc=$?
else
  echo "quality pending: source run.json unavailable for curvature; qualification is unverified" >&2
  quality_rc=1
fi
for arm in exact curvature; do
  run=$work/$base-$arm/run.json
  if [ -f "$run" ]; then
    python3 "$PACKAGE/benchmarks/native_metrics.py" --run "$run" --output "$RUN_DIR/metrics-$arm.json"
  fi
done
for budget in $budgets; do
  run=$work/$base-fixed$budget/run.json
  if [ -f "$run" ]; then
    python3 "$PACKAGE/benchmarks/native_metrics.py" --run "$run" --output "$RUN_DIR/metrics-fixed$budget.json"
  fi
done

if [ "${DITSERVE_MIXED:-1}" = 1 ]; then
  mixed_count=${DITSERVE_MIXED_COUNT:-$DITSERVE_COUNT}
  mixed_fraction=${DITSERVE_MIXED_VIDEO_FRACTION:-0.5}
  for arm in exact fold-fixed7 fission-fixed7 fission-curvature; do
    policy=block-cfg2sp3
    fold_steps=
    curvature=0
    case "$arm" in
      fold-fixed7) policy=fold-cfg2sp3; fold_steps=12,13,14,15,16,17,18 ;;
      fission-fixed7) policy=approx-cfg2sp3; fold_steps=12,13,14,15,16,17,18 ;;
      fission-curvature) policy=approx-cfg2sp3; curvature=1 ;;
    esac
    DITSERVE_RUN_ID=$base-mixed-$arm DITSERVE_POLICIES=$policy \
      DITSERVE_FORMS=cfg2sp3,sp3 DITSERVE_QUALITY_ARM= \
      DITSERVE_FOLD_STEPS=$fold_steps DITSERVE_CURVATURE=$curvature \
      DITSERVE_COUNT=$mixed_count DITSERVE_QUALITY=0 DITSERVE_VIDEO_FRACTION=$mixed_fraction \
      DITSERVE_IMAGE_BATCH=1 DITSERVE_DOORBELL=1 DITSERVE_OUTPUT_MODE=check \
      bash "$SCRIPTS/kairos_native_serving.sh"
    python3 "$PACKAGE/benchmarks/native_metrics.py" \
      --run "$work/$base-mixed-$arm/run.json" --output "$RUN_DIR/metrics-mixed-$arm.json"
  done
fi
exit "$quality_rc"
