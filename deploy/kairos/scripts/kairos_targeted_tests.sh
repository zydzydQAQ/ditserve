#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/kairos_common.sh"
: "${DITSERVE_TEST_ROOT:?set DITSERVE_TEST_ROOT to the versioned Python test directory}"
export PYTHONPATH=$PACKAGE/python:$DITSERVE_TEST_ROOT
export DITSERVE_KAIROS_RUNTIME_ROOT=${DITSERVE_KAIROS_RUNTIME_ROOT:-$(cd "$SCRIPTS/.." && pwd)}
export DITSERVE_CORE=$(command -v ditserve)
export PATH=$BUILD/bin:/opt/venv/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
# This is a real allocated-GPU runtime probe, followed by CPU-only contracts.
# It does not establish model correctness, quality or serving performance.
ditserve doctor --python /opt/venv/bin/python > "$RUN_DIR/doctor.json"
python3 "$SCRIPTS/kairos_runtime_audit.py"
python3 - "$RUN_DIR" <<'PY'
import hashlib, json, os, pathlib, re, sys
import torch
root = pathlib.Path(sys.argv[1])
info = json.loads((root / 'BUILD_INFO.json').read_text())
assert info['commit'].startswith(os.environ['DITSERVE_BUILD'])
image_identity = {}
if os.environ.get('DITSERVE_FORMAL_PROTOCOL'):
    digest = os.environ.get('DITSERVE_IMAGE_DIGEST', '')
    execution_image = os.environ.get('DITSERVE_EXECUTION_IMAGE', '')
    assert re.fullmatch(r'sha256:[0-9a-f]{64}', digest), 'missing verified immutable image digest'
    assert execution_image.endswith('@' + digest), 'job must pin its image digest'
    spec_path = pathlib.Path(os.environ['DITSERVE_JOB_SPEC_PATH'])
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    assert spec['name'] == os.environ['DITSERVE_RUN_ID'] and spec['image'] == execution_image
    assert spec['workers'] == 1 and spec['gpus_per_worker'] == 1 and spec['mount_shared'] is True
    spec_env = {item['name']: item['value'] for item in spec['env']}
    assert spec_env['DITSERVE_BUILD'] == os.environ['DITSERVE_BUILD']
    assert os.environ['DITSERVE_TEST_REVISION'] == info['commit'], 'tests must bind the full build commit'
    assert spec_env['DITSERVE_TEST_REVISION'] == info['commit']
    image_identity = dict(image_digest=digest, execution_image=execution_image,
        image_identity_source='submitted_job_spec_digest_pin',
        image_job_spec_sha256=hashlib.sha256(spec_raw).hexdigest())

assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected one allocated CUDA device'
binary = pathlib.Path(os.environ['DITSERVE_CORE'])
core = hashlib.sha256(binary.read_bytes()).hexdigest()
assert core == info['core_sha256'], 'deployed core identity differs'
package = pathlib.Path(os.environ['DITSERVE_PACKAGE_ROOT'])
python_sources = {}
for name in ('ditserve', 'xfuser'):
    files = sorted((package / name).rglob('*.py'))
    assert files and all(p.is_file() and not p.is_symlink() for p in files), 'missing or linked Python source'
    python_sources.update({p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
x = torch.arange(1024, device='cuda', dtype=torch.float32)
value = (x + 1).sum()
matrix = torch.ones((32, 32), device='cuda') @ torch.ones((32, 32), device='cuda')
torch.cuda.synchronize()
assert torch.isfinite(matrix).all().item() and torch.equal(matrix, torch.full_like(matrix, 32))
assert value.item() == 524800
result = dict(schema='ditserve.runtime-smoke.v1', status='passed',
              run_id=os.environ['DITSERVE_RUN_ID'], commit=info['commit'], core_sha256=core,
              python_sources=python_sources,
              test_revision=os.environ.get('DITSERVE_TEST_REVISION', os.environ['DITSERVE_BUILD']),
              previous_contract=os.environ.get('DITSERVE_PREVIOUS_CONTRACT'),
              runtime_image=info.get('runtime_image'), torch=torch.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name(0), devices=1, sum=value.item(), matmul_value=32,
              scope='CUDA arithmetic and deployed runtime; model and serving qualification pending', **image_identity)
with (root / 'runtime-smoke.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({key: value for key, value in result.items() if key != 'python_sources'}))
PY
export CUDA_VISIBLE_DEVICES=''
python3 -c 'import openpyxl; print("openpyxl", openpyxl.__version__)'
if [ -n "${DITSERVE_QUALITY_PENDING_RUN:-}" ]; then
  python3 "$PACKAGE/benchmarks/native_quality_pending.py" \
    --run "$DITSERVE_QUALITY_PENDING_RUN/run.json" \
    --journal "$DITSERVE_QUALITY_PENDING_RUN/.finalization.json"
fi
if [ "${DITSERVE_LPIPS_SELFTEST:-0}" = 1 ]; then
  python3 "$PACKAGE/benchmarks/p4_offline_lpips.py" selftest | tee "$RUN_DIR/lpips-selftest.json"
fi
case "${DITSERVE_CONTRACT_RECHECK:-all}" in
  sd35)
    : "${DITSERVE_PREVIOUS_CONTRACT:?bind the prior completed contract attempt}"
    echo "recheck=sd35 prior=$DITSERVE_PREVIOUS_CONTRACT tests=${DITSERVE_TEST_REVISION:?}"
    modules=(test_xdit_sd35)
    ;;
  all) modules=( \
  test_domain_control \
  test_domain_block_executor \
  test_domain_memory \
  test_domain_collectives \
  test_domain_archive ) ;;
  *) echo 'unsupported contract recheck scope' >&2; exit 2 ;;
esac
# tee must finish before hashing the log. The enclosing subshell and pipefail
# propagate either suite's failure; a CUDA success never creates this receipt.
(
  set -euo pipefail
  timeout 1800s python3 -m unittest -v "${modules[@]}"
) 2>&1 | tee "$RUN_DIR/contract.log"
# The exact package tested must still be present when a receipt is published.
python3 - "$RUN_DIR/runtime-smoke.json" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(os.environ['DITSERVE_PACKAGE_ROOT'])
current = {}
for name in ('ditserve', 'xfuser'):
    files = sorted((root / name).rglob('*.py'))
    assert files and all(p.is_file() and not p.is_symlink() for p in files)
    current.update({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
assert current == json.loads(pathlib.Path(sys.argv[1]).read_bytes())['python_sources'], 'Python source changed during contracts'
PY
if [ -n "${DITSERVE_FORMAL_PROTOCOL:-}" ]; then
  [ "${DITSERVE_CONTRACT_RECHECK:-all}" = all ] || {
    echo 'partial contract recheck cannot produce a formal receipt' >&2; exit 1;
  }
  PYTHONPATH="$PACKAGE/benchmarks:$PYTHONPATH" python3 "$SCRIPTS/kairos_contract_receipt.py" \
    --run-dir "$RUN_DIR" --protocol "$DITSERVE_FORMAL_PROTOCOL" \
    --job-spec "${DITSERVE_JOB_SPEC_PATH:?set immutable submitted job spec path}"
else
  echo 'diagnostic contracts complete; no formal protocol/image-bound receipt requested'
fi
