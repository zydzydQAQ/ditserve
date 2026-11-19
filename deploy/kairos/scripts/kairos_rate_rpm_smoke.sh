#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/kairos_common.sh"
: "${DITSERVE_TEST_ROOT:?set the versioned Python test directory}"
: "${DITSERVE_BENCHMARK_ROOT:?set the versioned benchmark directory}"
: "${DITSERVE_TEST_REVISION:?set the full tested commit}"
: "${DITSERVE_JOB_SPEC_PATH:?set the immutable submitted job spec}"
export PATH=$BUILD/bin:/opt/venv/bin:$PATH
export PYTHONPATH=$PACKAGE/python:$DITSERVE_BENCHMARK_ROOT:$DITSERVE_TEST_ROOT
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ditserve doctor --python /opt/venv/bin/python > "$RUN_DIR/doctor.json"
export DITSERVE_CORE=$BUILD/bin/ditserve
python3 - "$RUN_DIR" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import torch

run_dir = pathlib.Path(sys.argv[1])
core_path = pathlib.Path(os.environ["DITSERVE_CORE"])
build = json.loads((core_path.parents[1] / "BUILD_INFO.json").read_text())
spec_path = pathlib.Path(os.environ["DITSERVE_JOB_SPEC_PATH"])
spec_raw = spec_path.read_bytes()
spec = json.loads(spec_raw)
assert spec["workers"] == 1 and spec["gpus_per_worker"] == 1 and spec["mount_shared"] is True
assert spec["name"] == os.environ["DITSERVE_RUN_ID"] and spec["image"].endswith("@" + os.environ["DITSERVE_IMAGE_DIGEST"])
assert build["commit"] == os.environ["DITSERVE_TEST_REVISION"]
assert build["core_sha256"] == hashlib.sha256(core_path.read_bytes()).hexdigest()
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
value = (torch.arange(1024, device="cuda", dtype=torch.float32) + 1).sum()
matrix = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
torch.cuda.synchronize()
assert value.item() == 524800 and torch.equal(matrix, torch.full_like(matrix, 32))
result = {
    "schema": "ditserve.rate-rpm-smoke.v1",
    "status": "cuda_passed_contract_pending",
    "run_id": os.environ["DITSERVE_RUN_ID"],
    "commit": build["commit"],
    "core_sha256": build["core_sha256"],
    "image_digest": os.environ["DITSERVE_IMAGE_DIGEST"],
    "image_job_spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
    "gpu": torch.cuda.get_device_name(0),
    "devices": 1,
    "sum": value.item(),
    "matmul_value": 32,
    "scope": "CUDA arithmetic and deployed runtime; Archive contract and serving performance pending",
}
(run_dir / "cuda-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
export CUDA_VISIBLE_DEVICES=''
python3 -B "$DITSERVE_KAIROS_RUNTIME_ROOT/tools/rate_rpm_contract.py" "$RUN_DIR" | tee "$RUN_DIR/rate-rpm-contract.log"
python3 - "$RUN_DIR/cuda-smoke.json" "$RUN_DIR/runtime-smoke.json" <<'PY'
import json
import pathlib
import sys

source, target = map(pathlib.Path, sys.argv[1:])
result = json.loads(source.read_text())
result["status"] = "passed"
result["scope"] = "CUDA arithmetic, deployed runtime and frozen-input integer-rate contract; serving performance pending"
result["rate_contract"] = "rate_rpm_contract.py passed"
with target.open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result))
PY
