#!/bin/bash
# Diagnostic tooling validation against the frozen 517 core/worker package.
set -euo pipefail
check_storage() {
  if df -h /shared && df -BG /shared | python3 -c 'import sys; r=sys.stdin.read().splitlines()[-1].split(); assert int(r[2].rstrip("G"))<=790 and int(r[4].rstrip("%"))<80'; then
    return 0
  fi
  trap - EXIT
  return 75
}
check_storage || exit $?
runtime=$(cd "$(dirname "$0")/.." && pwd)
: "${DITSERVE_TOOLS_RUNNER_SHA256:?bind the diagnostic runner before execution}"
runner=$runtime/tools/kairos_tools_validation.py
[ "$(sha256sum "$runner" | cut -d' ' -f1)" = "$DITSERVE_TOOLS_RUNNER_SHA256" ] || exit 2
python3 "$runner" preflight || exit $?
source "$runtime/scripts/kairos_common.sh"
export DITSERVE_CORE=$BUILD/bin/ditserve
export PYTHONPATH=$DITSERVE_TEST_ROOT:$DITSERVE_BENCHMARK_ROOT:$PACKAGE/python
export PATH=$BUILD/bin:/opt/venv/bin:$PATH
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if python3 "$runner" run; then
  check_storage || exit $?
else
  code=$?
  if [ "$code" = 75 ]; then trap - EXIT; exit 75; fi
  check_storage || exit $?
  exit "$code"
fi
