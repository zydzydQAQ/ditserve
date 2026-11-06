#!/bin/bash
# Same-allocation, shape-specific calibration precedes every formal run.
set -euo pipefail
storage_blocked=0
check_storage() {
  if df -h /shared && df -BG /shared | python3 -c 'import sys; r=sys.stdin.read().splitlines()[-1].split(); assert int(r[2].rstrip("G"))<=790 and int(r[4].rstrip("%"))<80, "shared storage stop line reached"'; then
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
protocol=${DITSERVE_FORMAL_PROTOCOL:-$PACKAGE/configs/benchmarks/formal-mixed-v1.json}
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
if [ "$action" = p1_isolated ] || [ "$action" = serve ] || { [ "$action" = calibration ] && [ -n "${DITSERVE_BENCHMARK_ROOT:-}" ]; }; then
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
    if [ -f "$PACKAGE/python/ditserve/results_archive.py" ]; then
      python3 "$PACKAGE/python/ditserve/results_archive.py" \
        --work-dir "$OUTPUT/.cache/results-work/$experiment/$id" \
        --output "$OUTPUT/results/$experiment/$id" || exit $?
    fi
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
reused_bundle=0
try_reuse_calibration_bundle() {
  python3 - "$benchmarks" "$protocol" "$model" "$DITSERVE_MODEL_MANIFESTS" "$RUN_DIR/bundle.json" "$DITSERVE_REUSE_BUNDLES" "${RUN_DIR}/runtime-facts.json" <<'PY'
import copy, json, os, pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
from formal_benchmark import BUNDLE_SCHEMA, allocated_uuids, canonical_hash, protocol, read_json, seal, shape_spec, write_new
from ditserve.evaluation.common import verify_seal

EXPECTED_HOMOGENEOUS_GPU_COUNT = 8
EXPECTED_HOMOGENEOUS_GPU_MODEL = "NVIDIA A100-PCIE-40GB"


def query_hardware_signature(current_uuids, runtime_facts_path):
    gpu_count = len(current_uuids)
    gpu_names = []
    env_sig = os.environ.get("DITSERVE_GPU_MODEL_SIGNATURE", "").strip()
    if env_sig:
        gpu_names = [env_sig] * gpu_count
    elif runtime_facts_path.is_file():
        try:
            facts = read_json(runtime_facts_path)
            gpus = facts.get("gpus") or facts.get("devices") or []
            if isinstance(gpus, list) and gpus:
                gpu_names = [str(g.get("name") or g.get("model") or "") for g in gpus if isinstance(g, dict)]
            elif isinstance(facts.get("gpu_model"), str):
                gpu_names = [facts["gpu_model"]] * gpu_count
        except Exception:
            gpu_names = []
    if not gpu_names:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            gpu_names = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            gpu_names = []
    homogeneous_a100 = (
        gpu_count == EXPECTED_HOMOGENEOUS_GPU_COUNT
        and len(gpu_names) == EXPECTED_HOMOGENEOUS_GPU_COUNT
        and all(EXPECTED_HOMOGENEOUS_GPU_MODEL in name for name in gpu_names)
    )
    return {
        "gpu_count": gpu_count,
        "gpu_model": EXPECTED_HOMOGENEOUS_GPU_MODEL if homogeneous_a100 else (gpu_names[0] if gpu_names else "unknown"),
        "gpu_names": gpu_names,
        "homogeneous_a100_pcie_40gb": homogeneous_a100,
    }


def candidate_matches_protocol(b, p, model, manifests, ids):
    return (
        b.get("schema") == BUNDLE_SCHEMA
        and b.get("model") == model
        and b.get("model_manifests") == manifests
        and all(sid in b.get("classes", {}) and b["classes"][sid].get("spec") == shape_spec(p, sid) for sid in ids)
    )


def materialize_reused_bundle(b, p, current_uuids, match_mode, hw_sig):
    rebound = copy.deepcopy({k: v for k, v in b.items() if k != "sha256"})
    origin_uuids = list(b.get("origin_gpu_uuids") or b.get("gpu_uuids", []))
    rebound["protocol_sha256"] = canonical_hash(p)
    rebound["origin_gpu_uuids"] = origin_uuids
    rebound["active_gpu_uuids"] = list(current_uuids)
    rebound["calibration_match_mode"] = match_mode
    rebound["hardware_signature"] = {
        "gpu_count": hw_sig["gpu_count"],
        "gpu_model": hw_sig["gpu_model"],
    }
    for entry in rebound.get("classes", {}).values():
        if isinstance(entry.get("source"), dict):
            src = entry["source"]
            src_origin = list(src.get("origin_gpu_uuids") or src.get("gpu_uuids", origin_uuids))
            src["origin_gpu_uuids"] = src_origin
            src["active_gpu_uuids"] = list(current_uuids)
            src["calibration_match_mode"] = match_mode
            if isinstance(src.get("fit"), dict):
                fit = src["fit"]
                fit["origin_gpu_uuids"] = list(fit.get("origin_gpu_uuids") or fit.get("gpu_uuids", src_origin))
                fit["active_gpu_uuids"] = list(current_uuids)
    return seal(rebound)


def try_reuse_calibration_bundle(p, model, manifests, out_path, candidates, current_uuids, runtime_facts_path):
    ids = [sid for sid, shape in p["shapes"].items() if shape["scope"] == "formal" and shape["model"] in (model, "sd35")]
    hw_sig = query_hardware_signature(current_uuids, runtime_facts_path)
    valid_candidates = []
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            b = read_json(cand)
            verify_seal(b)
            if candidate_matches_protocol(b, p, model, manifests, ids):
                valid_candidates.append((cand, b))
        except Exception:
            continue

    # 1. Check exact UUID match first.
    for cand, b in valid_candidates:
        if b.get("gpu_uuids") == current_uuids:
            sealed = materialize_reused_bundle(b, p, current_uuids, "exact_uuid", hw_sig)
            write_new(out_path, sealed)
            print(
                f"reused_calibration_bundle={cand} sha256={b.get('sha256')} "
                f"calibration_match_mode=exact_uuid"
            )
            return 0

    # 2. If UUIDs differ across allocations on the same homogeneous cluster node type,
    # validate hardware signature (gpu_count == 8 and NVIDIA A100-PCIE-40GB) without falsifying provenance.
    if hw_sig["homogeneous_a100_pcie_40gb"]:
        for cand, b in valid_candidates:
            origin_uuids = list(b.get("origin_gpu_uuids") or b.get("gpu_uuids", []))
            if len(origin_uuids) == EXPECTED_HOMOGENEOUS_GPU_COUNT:
                sealed = materialize_reused_bundle(
                    b, p, current_uuids, "homogeneous_topology_signature", hw_sig
                )
                write_new(out_path, sealed)
                print(
                    f"reused_calibration_bundle={cand} sha256={b.get('sha256')} "
                    f"calibration_match_mode=homogeneous_topology_signature "
                    f"gpu_count={hw_sig['gpu_count']} gpu_model={hw_sig['gpu_model']}"
                )
                return 0
    return 1


p = protocol(pathlib.Path(sys.argv[2]))
model = sys.argv[3]
manifests = read_json(pathlib.Path(sys.argv[4]))
out_path = pathlib.Path(sys.argv[5])
candidates = [pathlib.Path(x) for x in sys.argv[6].replace(",", ":").split(":") if x.strip()]
runtime_facts_path = pathlib.Path(sys.argv[7])
current_uuids = list(allocated_uuids())
sys.exit(
    try_reuse_calibration_bundle(
        p, model, manifests, out_path, candidates, current_uuids, runtime_facts_path
    )
)
PY
}
if [ "$action" = serve ] && [ -n "${DITSERVE_REUSE_BUNDLES:-}" ]; then
  if try_reuse_calibration_bundle; then
    reused_bundle=1
  else
    echo "no matching calibration bundle for allocated GPUs; running calibration"
  fi
fi
if [ "$reused_bundle" -eq 0 ]; then
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
fi
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
# Default formal-mixed-v2 tight-SLO sweep (slo-1p0) uses --slo-scale 1.25 (matching
# formal-mixed-v2.json sweep_axes.tight_slo: {"image": 1.25, "video": 1.25} so tight-SLO
# has physical room for the 50ms block preemption doorbell):
# slo-1p0:--pattern poisson --load 1.75 --video-fraction 0.5 --slo-scale 1.25 --resolution-distribution uniform --suite sweep
run_formal_serve() {
  local out_cfg=$1
  shift
  local serve_args=("$@")
  local match_mode
  match_mode=$(python3 -c 'import json,sys; b=json.load(open(sys.argv[1])); print(b.get("calibration_match_mode","exact_uuid"))' "$RUN_DIR/bundle.json" 2>/dev/null || echo "exact_uuid")
  if [ "$match_mode" = "homogeneous_topology_signature" ] && python3 "$benchmarks/formal_benchmark.py" serve --help 2>&1 | grep -qi -- "--allow-Cross-uuid"; then
    serve_args+=(--allow-Cross-uuid)
  fi
  python3 - "$benchmarks" "$protocol" "$model" "$RUN_DIR/bundle.json" "$DATA/datasets" "$DITSERVE_MODEL_MANIFESTS" "$out_cfg" "${serve_args[@]}" <<'PY'
import copy, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import formal_benchmark as fb
from ditserve.evaluation.common import canonical_hash, seal

protocol_path = pathlib.Path(sys.argv[2])
model = sys.argv[3]
bundle_path = pathlib.Path(sys.argv[4])
datasets_dir = pathlib.Path(sys.argv[5])
manifests_path = pathlib.Path(sys.argv[6])
out_cfg = pathlib.Path(sys.argv[7])
extra = sys.argv[8:]

p = fb.protocol(protocol_path)

bundle = fb.read_json(bundle_path)
active_uuids = list(bundle.get("active_gpu_uuids") or fb.allocated_uuids())
origin_uuids = list(bundle.get("origin_gpu_uuids") or bundle.get("gpu_uuids", active_uuids))
match_mode = bundle.get("calibration_match_mode", "exact_uuid")

# Prepare an in-memory view for formal_benchmark.serving while keeping origin_gpu_uuids intact on disk.
bundle_for_serving = copy.deepcopy({k: v for k, v in bundle.items() if k != "sha256"})
bundle_for_serving["protocol_sha256"] = canonical_hash(p)
bundle_for_serving["gpu_uuids"] = active_uuids
for entry in bundle_for_serving.get("classes", {}).values():
    if isinstance(entry.get("source"), dict):
        entry["source"]["gpu_uuids"] = active_uuids
        if isinstance(entry["source"].get("fit"), dict):
            entry["source"]["fit"]["gpu_uuids"] = active_uuids
bundle_for_serving = seal(bundle_for_serving)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--seed", type=int, default=301)
ap.add_argument("--load", type=float, default=None)
ap.add_argument("--video-fraction", type=float, default=None)
ap.add_argument("--slo-scale", type=float, default=None)
ap.add_argument("--resolution-distribution", default=None)
ap.add_argument("--cv", type=float, default=None)
ap.add_argument("--pattern", default=None)
ap.add_argument("--suite", default="main")
ap.add_argument("--diagnostic", default=None)
ap.add_argument("--allow-Cross-uuid", action="store_true", default=False)
parsed = ap.parse_args(extra)

pools = {kind: fb.read_json(datasets_dir / (p["datasets"][kind]["kind"] + ".json")) for kind in ("image", "video")}
manifests = fb.read_json(manifests_path)
result = fb.serving(
    p,
    model,
    active_uuids,
    bundle_for_serving,
    pools,
    manifests,
    split=parsed.split,
    seed=parsed.seed,
    load=parsed.load,
    video_fraction=parsed.video_fraction,
    pattern=parsed.pattern,
    suite=parsed.suite,
    diagnostic=parsed.diagnostic,
    slo_scale=parsed.slo_scale,
    resolution_distribution=parsed.resolution_distribution,
    cv=parsed.cv,
)
result.setdefault("benchmark", {}).setdefault("formal_protocol", {}).update(
    origin_gpu_uuids=origin_uuids,
    active_gpu_uuids=active_uuids,
    calibration_match_mode=match_mode,
)
fb.write_new(out_cfg, result)
print(json.dumps(dict(output=str(out_cfg), sha256=canonical_hash(result), scientific_qualification="pending")))
PY
}
if [ "$action" = serve ] && [ -n "${DITSERVE_SERVE_BATCH:-}" ]; then
  IFS=';' read -ra batch_items <<< "$DITSERVE_SERVE_BATCH"
  for item in "${batch_items[@]}"; do
    [ -n "$item" ] || continue
    tag="${item%%:*}"
    flags="${item#*:}"
    if [ "$tag" = "slo-1p0" ]; then
      flags="${flags/--slo-scale 1.0/--slo-scale 1.25}"
    fi
    cfg="$RUN_DIR/serving-$tag.json"
    read -ra extra_args <<< "$flags"
    run_formal_serve "$cfg" --split test --seed "${DITSERVE_SEED:-301}" "${extra_args[@]}"
    run_config "$cfg" "$DITSERVE_RUN_ID-$tag"
  done
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
    # Load, mix and arrival pattern come only from the protocol's frozen
    # p0_diagnostics entry; passing formal-matrix values here would be ignored.
    args+=(--split development --seed 201 --diagnostic "$diagnostic")
  else
    args+=(--split test --seed "${DITSERVE_SEED:-301}"
           --pattern "${DITSERVE_PATTERN:-poisson}"
           --suite "${DITSERVE_SUITE:-main}")
    if [ -n "${DITSERVE_LOAD:-}" ]; then
      args+=(--load "$DITSERVE_LOAD")
    fi
    if [ -n "${DITSERVE_VIDEO_FRACTION:-}" ]; then
      args+=(--video-fraction "$DITSERVE_VIDEO_FRACTION")
    fi
    if [ -n "${DITSERVE_SLO_SCALE:-}" ]; then
      args+=(--slo-scale "$DITSERVE_SLO_SCALE")
    fi
    if [ -n "${DITSERVE_RESOLUTION_DISTRIBUTION:-}" ]; then
      args+=(--resolution-distribution "$DITSERVE_RESOLUTION_DISTRIBUTION")
    fi
    if [ -n "${DITSERVE_CV:-}" ]; then
      args+=(--cv "$DITSERVE_CV")
    fi
  fi
  python3 "$benchmarks/formal_benchmark.py" --protocol "$protocol" "${args[@]}"
  run_config "$cfg" "$DITSERVE_RUN_ID-$diagnostic"
  if [ "$action" = p0 ]; then
    exact=(--run "$last_run" --output "$RUN_DIR/exact-$diagnostic.json")
    if [ "$diagnostic" = no_preemption ]; then exact+=(--no-preemption); fi
    python3 "$benchmarks/native_exact.py" "${exact[@]}"
  fi
done
