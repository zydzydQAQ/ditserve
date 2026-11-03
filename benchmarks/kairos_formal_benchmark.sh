#!/bin/bash
# Same-allocation, shape-specific calibration precedes every formal run.
set -euo pipefail
storage_blocked=0
check_storage() {
  if df -h /shared && df -BG /shared | python3 -c 'import sys; r=sys.stdin.read().splitlines()[-1].split(); assert int(r[2].rstrip("G"))<=800 and int(r[4].rstrip("%"))<80, "shared storage stop line reached"'; then
    return 0
  fi
  storage_blocked=1
  # In particular, do not copy ledgers to the shared filesystem on EXIT.
  trap - EXIT
  return 75
}
# The pod may start long after the submission-time check. The common script
# creates shared directories, copies BUILD_INFO and starts a shared log writer.
check_storage || exit $?
source "$(dirname "$0")/kairos_common.sh"
export PYTHONPATH=$PACKAGE/python
: "${DITSERVE_MODEL_MANIFESTS:?provide frozen model manifest mapping}"
: "${DITSERVE_PROTOCOL_SHA256:?bind the frozen protocol}"
protocol=$PACKAGE/configs/benchmarks/formal-mixed-v1.json
python3 - "$protocol" "$DITSERVE_PROTOCOL_SHA256" <<'PY'
import hashlib, json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
raw = json.dumps(p, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2], 'protocol differs'
PY
case "$WAN_MODEL" in
  wan) model=wan21 ;;
  wan2.2-ti2v-5b) model=wan22 ;;
  *) exit 2 ;;
esac
action=${DITSERVE_FORMAL_ACTION:-p0}
benchmarks=$PACKAGE/benchmarks
if [ "$action" = p1_isolated ] || [ "$action" = serve ]; then
  : "${DITSERVE_BENCHMARK_ROOT:?provide the separately frozen benchmark snapshot}"
  : "${DITSERVE_KAIROS_RUNTIME_ROOT:?provide the immutable operator release}"
  [ "$DITSERVE_BENCHMARK_ROOT" = "$DITSERVE_KAIROS_RUNTIME_ROOT/benchmarks" ] || {
    echo 'isolated benchmark path differs from the submitted operator release' >&2; exit 2;
  }
  benchmarks=$DITSERVE_BENCHMARK_ROOT
fi
if [ "$action" = serve ]; then
  [ "$SCRIPTS" = "$DITSERVE_KAIROS_RUNTIME_ROOT/scripts" ] || {
    echo 'serving compiler must belong to the executing operator release' >&2; exit 2;
  }
  : "${DITSERVE_FORMAL_BENCHMARK_SHA256:?bind the committed serving compiler}"
  [[ "$DITSERVE_FORMAL_BENCHMARK_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo 'invalid serving compiler SHA256' >&2; exit 2;
  }
  printf '%s  %s\n' "$DITSERVE_FORMAL_BENCHMARK_SHA256" "$benchmarks/formal_benchmark.py" | sha256sum -c -
  echo "serving_operator=$benchmarks compiler_sha256=$DITSERVE_FORMAL_BENCHMARK_SHA256"
fi
case "$action" in
  p0) scope=development ;;
  calibration|serve|p1|p1_isolated) scope=formal ;;
  *) echo "unknown formal action: $action" >&2; exit 2 ;;
esac
# Use the frozen protocol's order and require complete coverage. A new shape
# must never silently inherit another shape's calibration or remain unprofiled.
if [ "$action" = p1 ]; then
  : "${DITSERVE_P1_PAIR:?provide the frozen P1 shape pair}"
  : "${DITSERVE_P1_ROLE:?provide the frozen P1 role}"
  : "${DITSERVE_P1_PLAN_SHA256:?bind the frozen P1 plan}"
  p1_plan=$PACKAGE/configs/benchmarks/p1-qualification-v1.json
  shape_pairs=$(python3 - "$PACKAGE" "$protocol" "$p1_plan" "$DITSERVE_P1_PAIR" "$DITSERVE_P1_ROLE" "$DITSERVE_P1_PLAN_SHA256" "$model" <<'PY'
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'benchmarks'))
from p1_benchmark import validate_plan
from ditserve.evaluation.common import canonical_hash
p, plan = (json.loads(pathlib.Path(path).read_bytes()) for path in sys.argv[2:4])
validate_plan(plan, p)
if canonical_hash(plan) != sys.argv[6]:
    raise ValueError('P1 plan differs from submitted identity')
pair = next(item for item in plan['shape_pairs'] if item['id'] == sys.argv[4])
if pair['model'] != sys.argv[7] or sys.argv[5] not in plan['roles']:
    raise ValueError('P1 model or role differs')
print(pair['video_shape'], pair['image_shape'], sep='\t')
PY
)
else
shape_pairs=$(python3 - "$protocol" "$model" "$scope" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
def selected(model, kind):
    return [key for key, value in p['shapes'].items()
            if value['model'] == model and value['kind'] == kind and value['scope'] == sys.argv[3]]
videos, images = selected(sys.argv[2], 'video'), selected('sd35', 'image')
if not videos or len(videos) != len(images):
    raise ValueError('each selected video and image shape requires an independent paired calibration')
for video, image in zip(videos, images):
    print(video, image, sep='\t')
PY
)
fi
videos=(); images=()
while IFS=$'\t' read -r video image; do
  videos+=("$video"); images+=("$image")
done <<< "$shape_pairs"
run_config() {
  local config=$1 id=$2 experiment status=0
  last_run=
  check_storage || return $?
  experiment=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment"])' "$config") || return $?
  # The common EXIT handler covers the parent run ID. Each child replay owns
  # this additional handler, including when run/verification fails or exits.
  (
    trap 'chmod -R a+rX "$OUTPUT/.cache/results-work/$experiment/$id" 2>/dev/null || true' EXIT
    ditserve run --config "$config" --run-id "$id" --allow-gpu --ledger "$LEDGERS/$experiment.json" || exit $?
    ditserve results verify "$OUTPUT/results/$experiment/$id" || exit $?
  ) || status=$?
  last_run=$OUTPUT/.cache/results-work/$experiment/$id/run.json
  return "$status"
}
check_storage
preflight_scope=$scope
if [ "$action" = p1 ] || [ "$action" = p1_isolated ]; then preflight_scope=development; fi
if [ "$action" = p1_isolated ]; then
  # The helper and benchmark tools belong to the new immutable operator.
  # GPU core/workers remain bound to the separately verified P0 build.
  helper=$SCRIPTS/../tools/p1_isolated_job.py
  export PYTHONPATH=$benchmarks:$PACKAGE/python
  python3 "$helper" preflight --model "$model" --protocol "$protocol" \
    --benchmark-root "$benchmarks" \
    --pair-plan "$PACKAGE/configs/benchmarks/p1-qualification-v1.json" \
    --model-manifests "$DITSERVE_MODEL_MANIFESTS" --build "$BUILD/BUILD_INFO.json" \
    --job-spec "${DITSERVE_JOB_SPEC_PATH:?provide immutable submitted spec}" \
    --datasets "$DATA/datasets" --package "$PACKAGE/python" --core "$(command -v ditserve)" \
    --operator-script "$0" --output "$RUN_DIR/isolated-preflight.json"
fi
python3 "$benchmarks/formal_benchmark.py" --protocol "$protocol" preflight \
  --model "$model" --scope "$preflight_scope" --datasets "$DATA/datasets" \
  --model-manifests "$DITSERVE_MODEL_MANIFESTS" --output "$RUN_DIR/input-preflight.json"
python3 "$SCRIPTS/kairos_runtime_audit.py"
calibrations=()
for i in "${!videos[@]}"; do
  cfg=$RUN_DIR/calibration-$i.json
  python3 "$benchmarks/formal_benchmark.py" --protocol "$protocol" calibrate \
    --model "$model" --video-shape "${videos[$i]}" --image-shape "${images[$i]}" \
    --datasets "$DATA/datasets" --model-manifests "$DITSERVE_MODEL_MANIFESTS" --output "$cfg"
  run_config "$cfg" "$DITSERVE_RUN_ID-cal$i"
  calibrations+=("$last_run")
done
python3 "$benchmarks/formal_benchmark.py" --protocol "$protocol" assemble --model "$model" \
  --runs "${calibrations[@]}" --runtime-facts "$RUN_DIR/runtime-facts.json" --output "$RUN_DIR/bundle.json"
if [ "$action" = calibration ]; then exit 0; fi
if [ "$action" = p1_isolated ]; then
  prepared=$RUN_DIR/isolated
  python3 "$helper" prepare --preflight "$RUN_DIR/isolated-preflight.json" \
    --benchmark-root "$benchmarks" \
    --bundle "$RUN_DIR/bundle.json" --runtime-facts "$RUN_DIR/runtime-facts.json" \
    --calibrations "${calibrations[@]}" --package "$PACKAGE/python" --core "$(command -v ditserve)" \
    --output-dir "$prepared"
  cases=$(python3 - "$prepared/execution-plan.json" <<'PY'
import json, sys
for case in json.load(open(sys.argv[1]))['cases']:
    print(*(case[key] for key in ('pair_id', 'inputs', 'inputs_sha256', 'config', 'run_id')), sep='\t')
PY
)
  while IFS=$'\t' read -r pair inputs inputs_sha cfg id; do
    status=0
    run_config "$prepared/$cfg" "$id" || status=$?
    if [ "$storage_blocked" -ne 0 ]; then exit "$status"; fi
    check_storage
    coverage=(--inputs "$prepared/$inputs" --inputs-sha256 "$inputs_sha" \
              --config "$prepared/$cfg" --output "$prepared/$pair-coverage.json")
    if [ -n "$last_run" ] && [ -f "$last_run" ]; then coverage+=(--run "$last_run"); fi
    python3 "$benchmarks/p1_isolated_batch.py" "${coverage[@]}"
    # Failed calibration or replay never advances to another measured case.
    # Both planned coverage files already preserve all 192 intended requests.
    if [ "$status" -ne 0 ]; then exit "$status"; fi
  done <<< "$cases"
  exit 0
fi
if [ "$action" = p1 ]; then
  seeds=$(python3 -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["seeds"])))' "$p1_plan")
  for seed in $seeds; do
    cfg=$RUN_DIR/p1-$DITSERVE_P1_ROLE-s$seed.json
    python3 "$benchmarks/p1_benchmark.py" --protocol "$protocol" --plan "$p1_plan" \
      --pair "$DITSERVE_P1_PAIR" --role "$DITSERVE_P1_ROLE" --seed "$seed" \
      --bundle "$RUN_DIR/bundle.json" --datasets "$DATA/datasets" \
      --model-manifests "$DITSERVE_MODEL_MANIFESTS" --output "$cfg"
    run_config "$cfg" "$DITSERVE_RUN_ID-s$seed"
  done
  # Source-bound offline P1 evaluation covers all cases, including zero-stop
  # and negative outcomes. Successful replay is not scientific acceptance.
  exit 0
fi
cases=(serving)
if [ "$action" = p0 ]; then cases=(contention no_preemption); fi
if [ -n "${DITSERVE_P0_CASE:-}" ]; then
  [ "$action" = p0 ] || { echo 'P0 case only applies to P0' >&2; exit 2; }
  case "$DITSERVE_P0_CASE" in contention|no_preemption) cases=("$DITSERVE_P0_CASE") ;;
    *) echo 'unknown P0 case' >&2; exit 2 ;; esac
fi
for diagnostic in "${cases[@]}"; do
  cfg=$RUN_DIR/$diagnostic.json
  args=(serve --model "$model" --bundle "$RUN_DIR/bundle.json" --datasets "$DATA/datasets"
        --model-manifests "$DITSERVE_MODEL_MANIFESTS" --output "$cfg")
  if [ "$action" = p0 ]; then
    args+=(--split development --seed 201 --load 1.2 --video-fraction 0.5 --pattern bursty --diagnostic "$diagnostic")
  else
    args+=(--split test --seed "${DITSERVE_SEED:-301}" --load "${DITSERVE_LOAD:-0.6}"
           --video-fraction "${DITSERVE_VIDEO_FRACTION:-0.2}" --pattern "${DITSERVE_PATTERN:-poisson}"
           --suite "${DITSERVE_SUITE:-main}")
  fi
  python3 "$benchmarks/formal_benchmark.py" --protocol "$protocol" "${args[@]}"
  run_config "$cfg" "$DITSERVE_RUN_ID-$diagnostic"
  if [ "$action" = p0 ]; then
    exact=(--run "$last_run" --output "$RUN_DIR/exact-$diagnostic.json")
    if [ "$diagnostic" = no_preemption ]; then exact+=(--no-preemption); fi
    python3 "$benchmarks/native_exact.py" "${exact[@]}"
  fi
done
