#!/bin/bash
# Run `abl-first-fit` with `continuous_batch=0` (`+Block` w/o `PairFit`, w/o `CB`)
# on both Poisson and Gamma CV=2.0 at (rate_rpm=76, video_fraction=0.3, slo_scale=1.25)
# using published build 3878f01.
set -euo pipefail

check_storage() {
  df -h /shared
  df -BG /shared | python3 -c 'import sys; r=sys.stdin.read().splitlines()[-1].split(); assert int(r[2].rstrip("G"))<=805 and int(r[4].rstrip("%"))<81, f"shared storage stop line approached: {r[2]} {r[4]}"'
}

check_storage
source "${DITSERVE_KAIROS_RUNTIME_ROOT}/scripts/kairos_common.sh"
export PYTHONPATH=$PACKAGE/python

protocol=${DITSERVE_FORMAL_PROTOCOL:-$DITSERVE_KAIROS_RUNTIME_ROOT/configs/benchmarks/formal-mixed-v2.json}
benchmarks=$DITSERVE_BENCHMARK_ROOT
model=wan22

python3 "$benchmarks/formal_benchmark.py" --protocol "$protocol" preflight \
  --model "$model" --scope "formal" --datasets "$DATA/datasets" \
  --model-manifests "$DITSERVE_MODEL_MANIFESTS" --output "$RUN_DIR/input-preflight.json"
python3 "${DITSERVE_KAIROS_RUNTIME_ROOT}/scripts/kairos_runtime_audit.py"

python3 - "$benchmarks" "$protocol" "$model" "$DITSERVE_MODEL_MANIFESTS" "$DATA/datasets" "$DITSERVE_REUSE_BUNDLES" "$RUN_DIR" <<'PY'
import copy, json, pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
import formal_benchmark as fb
from ditserve.evaluation.common import canonical_hash, seal, verify_seal

p = fb.protocol(pathlib.Path(sys.argv[2]))
model = sys.argv[3]
manifests = fb.read_json(pathlib.Path(sys.argv[4]))
datasets_dir = pathlib.Path(sys.argv[5])
cand = pathlib.Path(sys.argv[6])
run_dir = pathlib.Path(sys.argv[7])

current_uuids = list(fb.allocated_uuids())[:8]
assert len(current_uuids) == 8
gpu_names = [
    ln.strip()
    for ln in subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10).splitlines()
    if ln.strip()
]
assert len(gpu_names) == 8 and all("NVIDIA A100-PCIE-40GB" in n for n in gpu_names)

b = fb.read_json(cand)
verify_seal(b)
origin_uuids = list(b.get("origin_gpu_uuids") or b.get("gpu_uuids", []))
match_mode = "exact_uuid" if b.get("gpu_uuids") == current_uuids else "homogeneous_topology_signature"

rebound = copy.deepcopy({k: v for k, v in b.items() if k != "sha256"})
rebound["protocol_sha256"] = canonical_hash(p)
rebound["gpu_uuids"] = current_uuids
rebound["origin_gpu_uuids"] = origin_uuids
rebound["active_gpu_uuids"] = current_uuids
rebound["calibration_match_mode"] = match_mode
for entry in rebound.get("classes", {}).values():
    if isinstance(entry.get("source"), dict):
        entry["source"]["gpu_uuids"] = current_uuids
        if isinstance(entry["source"].get("fit"), dict):
            entry["source"]["fit"]["gpu_uuids"] = current_uuids
bundle_for_serving = seal(rebound)
fb.write_new(run_dir / "bundle.json", bundle_for_serving)

pools = {kind: fb.read_json(datasets_dir / (p["datasets"][kind]["kind"] + ".json")) for kind in ("image", "video")}
for tag, pat, cv_val in [("ff0-p", "poisson", None), ("ff0-g2", "gamma", 2.0)]:
    cfg = fb.serving(
        p, model, current_uuids, bundle_for_serving, pools, manifests,
        split="test", seed=301, rate_rpm=76.0, video_fraction=0.3,
        pattern=pat, cv=cv_val, slo_scale=1.25, resolution_distribution="uniform",
        suite="ablation", policies="abl-first-fit",
        continuous_batch=0, sequence_packing=False, bubble_free=False,
    )
    cfg["experiment"] = f"formal-wan22-test-{pat}-s301-r76rpm-v0p3-slo1p25-{tag}"
    cfg.setdefault("benchmark", {}).setdefault("formal_protocol", {}).update(
        origin_gpu_uuids=origin_uuids,
        active_gpu_uuids=current_uuids,
        calibration_match_mode=match_mode,
        num_gpus=8,
    )
    out_cfg = run_dir / f"serving-{tag}.json"
    fb.write_new(out_cfg, cfg)
    print("Wrote", out_cfg, cfg["experiment"])
PY

for tag in ff0-p ff0-g2; do
  check_storage
  cfg="$RUN_DIR/serving-$tag.json"
  id="$DITSERVE_RUN_ID-$tag"
  experiment=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment"])' "$cfg")
  echo "== running id=$id experiment=$experiment ($(date -Is))"
  (
    trap 'chmod -R a+rX "$OUTPUT/.cache/results-work/$experiment/$id" "$OUTPUT/results/$experiment/$id" 2>/dev/null || true' EXIT
    ditserve run --config "$cfg" --run-id "$id" --allow-gpu --ledger "$LEDGERS/$experiment.json"
    PYTHONPATH="$PACKAGE/python:${PYTHONPATH:-}" python3 -c 'import sys; from pathlib import Path; from ditserve.results import archive_run; archive_run(Path(sys.argv[1]), Path(sys.argv[2]))' \
      "$OUTPUT/.cache/results-work/$experiment/$id" \
      "$OUTPUT/results/$experiment/$id"
    ditserve results verify "$OUTPUT/results/$experiment/$id"
  )
  cp -a "$LEDGERS/." "$RUN_DIR/ledgers/" 2>/dev/null || true
done

echo "== completed kairos_staircase_ff_nocb.sh run=$DITSERVE_RUN_ID ($(date -Is))"
