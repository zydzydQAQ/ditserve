#!/bin/bash
# ISCA'27 revision targeted GPU supplement runner on verified 720da72 build.
# Enforces AGENTS.md shared storage hard stop (< 810G and < 82%; gated at <= 805G and < 81%).
set -euo pipefail

storage_blocked=0
check_storage() {
  if df -h /shared && df -BG /shared | python3 -c 'import sys; r=sys.stdin.read().splitlines()[-1].split(); assert int(r[2].rstrip("G"))<=805 and int(r[4].rstrip("%"))<81, f"shared storage stop line approached: {r[2]} {r[4]}"'; then
    return 0
  fi
  storage_blocked=1
  trap - EXIT
  return 75
}

check_storage || exit $?

: "${DITSERVE_KAIROS_RUNTIME_ROOT:?provide the immutable 720da72 operator release}"
: "${DITSERVE_BENCHMARK_ROOT:?provide the frozen benchmark snapshot}"
: "${DITSERVE_MODEL_MANIFESTS:?provide frozen model manifest mapping}"
: "${DITSERVE_PROTOCOL_SHA256:?bind the frozen protocol}"
: "${DITSERVE_FORMAL_BENCHMARK_SHA256:?bind the committed serving compiler}"
: "${DITSERVE_REUSE_BUNDLES:?provide verified 8x A100 calibration bundle}"

source "${DITSERVE_KAIROS_RUNTIME_ROOT}/scripts/kairos_common.sh"
export PYTHONPATH=$PACKAGE/python

protocol=${DITSERVE_FORMAL_PROTOCOL:-$DITSERVE_KAIROS_RUNTIME_ROOT/configs/benchmarks/formal-mixed-v2.json}
benchmarks=$DITSERVE_BENCHMARK_ROOT

python3 - "$protocol" "$DITSERVE_PROTOCOL_SHA256" <<'PY'
import hashlib, json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
raw = json.dumps(p, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2], 'protocol differs'
PY

printf '%s  %s\n' "$DITSERVE_FORMAL_BENCHMARK_SHA256" "$benchmarks/formal_benchmark.py" | sha256sum -c -
echo "serving_operator=$benchmarks compiler_sha256=$DITSERVE_FORMAL_BENCHMARK_SHA256"

model=wan22
python3 "$benchmarks/formal_benchmark.py" --protocol "$protocol" preflight \
  --model "$model" --scope "formal" --datasets "$DATA/datasets" \
  --model-manifests "$DITSERVE_MODEL_MANIFESTS" --output "$RUN_DIR/input-preflight.json"
python3 "${DITSERVE_KAIROS_RUNTIME_ROOT}/scripts/kairos_runtime_audit.py"

# Reuse verified 8x NVIDIA A100-PCIE-40GB calibration bundle with hardware signature check
python3 - "$benchmarks" "$protocol" "$model" "$DITSERVE_MODEL_MANIFESTS" "$RUN_DIR/bundle.json" "$DITSERVE_REUSE_BUNDLES" "${RUN_DIR}/runtime-facts.json" <<'PY'
import copy, json, pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
from formal_benchmark import BUNDLE_SCHEMA, allocated_uuids, canonical_hash, protocol, read_json, seal, shape_spec, write_new
from ditserve.evaluation.common import verify_seal

p = protocol(pathlib.Path(sys.argv[2]))
model = sys.argv[3]
manifests = read_json(pathlib.Path(sys.argv[4]))
out_path = pathlib.Path(sys.argv[5])
cand = pathlib.Path(sys.argv[6])
current_uuids = list(allocated_uuids())
assert len(current_uuids) == 8, f"expected 8 GPUs, got {len(current_uuids)}"
gpu_names = [
    ln.strip()
    for ln in subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10).splitlines()
    if ln.strip()
]
assert len(gpu_names) == 8 and all("NVIDIA A100-PCIE-40GB" in n for n in gpu_names), f"unexpected GPU names: {gpu_names}"

b = read_json(cand)
verify_seal(b)
ids = [sid for sid, shape in p["shapes"].items() if shape["scope"] == "formal" and shape["model"] in (model, "sd35")]
assert b.get("schema") == BUNDLE_SCHEMA and b.get("model") == model and b.get("model_manifests") == manifests
assert all(sid in b.get("classes", {}) and b["classes"][sid].get("spec") == shape_spec(p, sid) for sid in ids)

match_mode = "exact_uuid" if b.get("gpu_uuids") == current_uuids else "homogeneous_topology_signature"
origin_uuids = list(b.get("origin_gpu_uuids") or b.get("gpu_uuids", []))
rebound = copy.deepcopy({k: v for k, v in b.items() if k != "sha256"})
rebound["protocol_sha256"] = canonical_hash(p)
rebound["origin_gpu_uuids"] = origin_uuids
rebound["active_gpu_uuids"] = current_uuids
rebound["calibration_match_mode"] = match_mode
rebound["hardware_signature"] = {"gpu_count": 8, "gpu_model": "NVIDIA A100-PCIE-40GB"}
for entry in rebound.get("classes", {}).values():
    if isinstance(entry.get("source"), dict):
        src = entry["source"]
        src_origin = list(src.get("origin_gpu_uuids") or src.get("gpu_uuids", origin_uuids))
        src["origin_gpu_uuids"] = src_origin
        src["active_gpu_uuids"] = current_uuids
        src["calibration_match_mode"] = match_mode
        if isinstance(src.get("fit"), dict):
            src["fit"]["origin_gpu_uuids"] = list(src["fit"].get("origin_gpu_uuids") or src["fit"].get("gpu_uuids", src_origin))
            src["fit"]["active_gpu_uuids"] = current_uuids
write_new(out_path, seal(rebound))
print(f"reused_calibration_bundle={cand} sha256={b.get('sha256')} match_mode={match_mode}")
PY

run_config() {
  local config=$1 id=$2 experiment status=0
  check_storage || return $?
  experiment=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment"])' "$config") || return $?
  echo "== running id=$id experiment=$experiment ($(date -Is))"
  (
    trap 'chmod -R a+rX "$OUTPUT/.cache/results-work/$experiment/$id" "$OUTPUT/results/$experiment/$id" 2>/dev/null || true' EXIT
    ditserve run --config "$config" --run-id "$id" --allow-gpu --ledger "$LEDGERS/$experiment.json" || exit $?
    PYTHONPATH="$PACKAGE/python:${PYTHONPATH:-}" python3 -c 'import sys; from pathlib import Path; from ditserve.results import archive_run; archive_run(Path(sys.argv[1]), Path(sys.argv[2]))' \
      "$OUTPUT/.cache/results-work/$experiment/$id" \
      "$OUTPUT/results/$experiment/$id" || exit $?
    ditserve results verify "$OUTPUT/results/$experiment/$id" || exit $?
  ) || status=$?
  cp -a "$LEDGERS/." "$RUN_DIR/ledgers/" 2>/dev/null || true
  return "$status"
}

run_formal_serve() {
  local out_cfg=$1
  shift
  python3 - "$benchmarks" "$protocol" "$model" "$RUN_DIR/bundle.json" "$DATA/datasets" "$DITSERVE_MODEL_MANIFESTS" "$out_cfg" "$@" <<'PY'
import argparse, copy, json, pathlib, sys
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
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--seed", type=int, default=301)
rate_group = ap.add_mutually_exclusive_group()
rate_group.add_argument("--load", type=float, default=None)
rate_group.add_argument("--rate-rpm", type=float, default=None)
ap.add_argument("--video-fraction", type=float, default=None)
ap.add_argument("--slo-scale", type=float, default=None)
ap.add_argument("--resolution-distribution", default=None)
ap.add_argument("--cv", type=float, default=None)
ap.add_argument("--pattern", default=None)
ap.add_argument("--suite", default="main")
ap.add_argument("--policies", default=None)
parsed = ap.parse_args(extra)

bundle = fb.read_json(bundle_path)
active_uuids = list(bundle.get("active_gpu_uuids") or fb.allocated_uuids())[:8]
origin_uuids = list(bundle.get("origin_gpu_uuids") or bundle.get("gpu_uuids", active_uuids))
match_mode = bundle.get("calibration_match_mode", "exact_uuid")

bundle_for_serving = copy.deepcopy({k: v for k, v in bundle.items() if k != "sha256"})
bundle_for_serving["protocol_sha256"] = canonical_hash(p)
bundle_for_serving["gpu_uuids"] = active_uuids
for entry in bundle_for_serving.get("classes", {}).values():
    if isinstance(entry.get("source"), dict):
        entry["source"]["gpu_uuids"] = active_uuids
        if isinstance(entry["source"].get("fit"), dict):
            entry["source"]["fit"]["gpu_uuids"] = active_uuids
bundle_for_serving = seal(bundle_for_serving)

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
    rate_rpm=parsed.rate_rpm,
    video_fraction=parsed.video_fraction,
    pattern=parsed.pattern,
    suite=parsed.suite,
    policies=parsed.policies,
    slo_scale=parsed.slo_scale,
    resolution_distribution=parsed.resolution_distribution,
    cv=parsed.cv,
)
result.setdefault("benchmark", {}).setdefault("formal_protocol", {}).update(
    origin_gpu_uuids=origin_uuids,
    active_gpu_uuids=active_uuids,
    calibration_match_mode=match_mode,
    num_gpus=len(active_uuids),
)
fb.write_new(out_cfg, result)
print(json.dumps(dict(output=str(out_cfg), experiment=result["experiment"], sha256=canonical_hash(result))))
PY
}

if [ -n "${DITSERVE_SERVE_BATCH:-}" ]; then
  IFS=';' read -ra batch_items <<< "$DITSERVE_SERVE_BATCH"
  for item in "${batch_items[@]}"; do
    [ -n "$item" ] || continue
    tag="${item%%:*}"
    flags="${item#*:}"
    cfg="$RUN_DIR/serving-$tag.json"
    read -ra extra_args <<< "$flags"
    run_formal_serve "$cfg" "${extra_args[@]}"
    run_config "$cfg" "$DITSERVE_RUN_ID-$tag"
  done
fi

if [ "${DITSERVE_RUN_121F_ELASTIC:-0}" = "1" ]; then
  check_storage
  cfg121="$RUN_DIR/serving-121f-720p-elastic.json"
  python3 /shared/data/DITSERVE/scripts/build_121f_elastic_config.py \
    --benchmarks-dir "$benchmarks" \
    --protocol "$protocol" \
    --bundle-121f "${DITSERVE_BUNDLE_121F:?provide 121f bundle path}" \
    --bundle-81f "$RUN_DIR/bundle.json" \
    --datasets-dir "$DATA/datasets" \
    --manifests "$DITSERVE_MODEL_MANIFESTS" \
    --policies "${DITSERVE_121F_POLICIES:-ditserve,block-cfg2,sbp-esp-cfg2}" \
    --seed 301 \
    --count "${DITSERVE_121F_COUNT:-12}" \
    --rate-rpm "${DITSERVE_121F_RATE_RPM:-1.5}" \
    --video-slo-scale "${DITSERVE_121F_VIDEO_SLO:-0.50}" \
    --image-slo-scale 1.50 \
    --output "$cfg121"
  run_config "$cfg121" "$DITSERVE_RUN_ID-121f-elastic"
fi

echo "== completed kairos_rev_supplement.sh run=$DITSERVE_RUN_ID ($(date -Is))"
