#!/bin/bash
# Same frozen trace twice in one job: collective_backend=nccl, then collective_backend=gfc, sharing one
# same-allocation calibration; then digest-compare the fixed-form policies (exactness) and keep both
# runs' serving metrics (performance). Run only after the gfc microbenchmark passed on this platform.
source "$(dirname "$0")/kairos_common.sh"
base=$DITSERVE_RUN_ID
export DITSERVE_OUTPUT_MODE=digest DITSERVE_DOORBELL=${DITSERVE_DOORBELL:-1}
export DITSERVE_POLICIES=${DITSERVE_POLICIES:-static-cfg2sp3,edf-cfg2}
for required in static-cfg2sp3 edf-cfg2; do
  case ",$DITSERVE_POLICIES," in
    *",$required,"*) ;;
    *) echo "GFC comparison requires policy $required; no GPU run started" >&2; exit 1 ;;
  esac
done
if [ "${DITSERVE_CALIBRATE:-1}" = 1 ]; then
  kairos_calibrate_native_profile
  export DITSERVE_PROFILE=$RUN_DIR/profile.json
fi
export DITSERVE_CALIBRATE=0 DITSERVE_REQUIRE_PROFILE_MATCH=1
DITSERVE_RUN_ID=$base-nccl DITSERVE_COLLECTIVE_BACKEND=nccl bash "$SCRIPTS/kairos_native_serving.sh"
DITSERVE_RUN_ID=$base-gfc DITSERVE_COLLECTIVE_BACKEND=gfc bash "$SCRIPTS/kairos_native_serving.sh"
case "$WAN_MODEL" in wan) MODEL=wan21 ;; *) MODEL=wan22 ;; esac
load=${DITSERVE_LOAD:-1.0}
experiment=${DITSERVE_EXPERIMENT:-kairos-$MODEL-native-serving-s${DITSERVE_SEED:-201}-l${load//./p}}
WORK=$OUTPUT/.cache/results-work/$experiment
export PYTHONPATH=$PACKAGE/python
python3 "$PACKAGE/benchmarks/compare_digests.py" \
  --reference "$WORK/$base-nccl/run.json" --candidate "$WORK/$base-gfc/run.json" \
  --policies static-cfg2sp3,edf-cfg2 --output "$RUN_DIR/digest-compare.json"
