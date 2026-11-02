import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import native_quality_pending


PENDING = [{"consumer": "native_quality", "closed": False}]


def completed_record():
    return {
        "schema": "ditserve.run.v1", "run_id": "quality", "status": "completed",
        "source": {"core_sha256": "a" * 64},
        "config": {
            "benchmark": {"native_serving": True, "quality_arm": "exact", "output_mode": "check",
                          "trace": {"manifest": {"quality_requests": {"video": 1}}}},
            "phases": [{"artifact_dependencies": PENDING}],
        },
        "phases": [{"status": "completed", "cells": [{"status": "completed"}]}],
        "requests": [{"status": "completed", "outputs": [{"output": {}}]}],
        "metrics": {"requests": 1, "completed": 1, "failed": 0, "incomplete": 0},
        "failures": [], "artifacts": [{"dependencies": PENDING}],
    }


class NativeQualityPendingTest(unittest.TestCase):
    def test_only_expected_pending_evaluation_is_accepted(self):
        record = completed_record()
        journal = {"status": "deferred", "errors": [], "deleted": []}
        result = native_quality_pending.quality_deferral(record, journal)
        self.assertEqual(result["status"], "quality_evaluation_pending")
        self.assertFalse(result["quality_qualified"])
        mutations = []
        for path, value in [
            (("status",), "failed"),
            (("metrics", "failed"), 1),
            (("config", "benchmark", "quality_arm"), None),
            (("config", "phases", 0, "artifact_dependencies"), []),
            (("artifacts", 0, "dependencies"), []),
        ]:
            changed = copy.deepcopy(record)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            mutations.append(changed)
        changed = copy.deepcopy(record)
        changed["requests"][0]["status"] = "failed"
        mutations.append(changed)
        for changed in mutations:
            with self.subTest(change=changed), self.assertRaises(ValueError):
                native_quality_pending.quality_deferral(changed, journal)
        for field, value in (("status", "protected"), ("errors", ["read failed"]), ("deleted", [{}])):
            with self.subTest(journal=field), self.assertRaises(ValueError):
                native_quality_pending.quality_deferral(record, {**journal, field: value})

    def test_serving_wrapper_does_not_swallow_execution_failure(self):
        runtime = Path(os.environ.get("DITSERVE_KAIROS_RUNTIME_ROOT",
            ROOT.parent / ".local/ditserve-kairos-rewrite-runtime"))
        source = runtime / "scripts/kairos_native_serving.sh"
        if not source.is_file():
            self.skipTest("local operator scripts")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, package, binary = root / "scripts", root / "package/benchmarks", root / "bin"
            scripts.mkdir()
            package.mkdir(parents=True)
            binary.mkdir()
            (scripts / source.name).write_bytes(source.read_bytes())
            for name in ("native_quality_pending.py", "native_exact.py", "native_metrics.py"):
                (package / name).write_bytes((ROOT / "benchmarks" / name).read_bytes())
            (scripts / "kairos_common.sh").write_text('''set -euo pipefail
export OUTPUT=$FIXTURE/output PACKAGE=$FIXTURE/package SCRIPTS=$FIXTURE/scripts
export RUN_DIR=$OUTPUT/acceptance/$DITSERVE_RUN_ID LEDGERS=$FIXTURE/ledgers
export DATA=$FIXTURE/data WAN_MODEL=wan
mkdir -p "$RUN_DIR" "$LEDGERS"
''')
            (scripts / "kairos_runtime_audit.py").write_text('import time\ntime.sleep(60)\n')
            (package / "native_serving.py").write_text('''import json, sys
path = sys.argv[sys.argv.index('--output') + 1]
with open(path, 'w') as stream:
    json.dump({'experiment': 'quality'}, stream)
''')
            (binary / "ditserve").write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
if sys.argv[1] == 'results':
    print(json.dumps({'status': 'verified'}))
    raise SystemExit(0)
run_id = sys.argv[sys.argv.index('--run-id') + 1]
work = pathlib.Path(os.environ['FIXTURE']) / 'output/.cache/results-work/quality' / run_id
work.mkdir(parents=True, exist_ok=True)
pending = [{'consumer': 'native_quality', 'closed': False}]
failed = os.environ.get('CASE') == 'failed'
record = {
    'schema': 'ditserve.run.v1', 'run_id': run_id, 'status': 'completed',
    'source': {'core_sha256': 'a' * 64},
    'config': {'benchmark': {'native_serving': True, 'quality_arm': 'exact',
        'quality_per_kind': 1, 'output_mode': 'check',
        'trace': {'manifest': {'quality_requests': {'video': 1}}}},
        'phases': [{'artifact_dependencies': pending}]},
    'phases': [{'status': 'completed', 'cells': [{'status': 'completed'}]}],
    'requests': [{'status': 'failed' if failed else 'completed', 'outputs': [{}]}],
    'metrics': {'requests': 1, 'completed': int(not failed), 'failed': int(failed), 'incomplete': 0},
    'failures': [], 'artifacts': [{'dependencies': pending}],
}
(work / 'run.json').write_text(json.dumps(record))
(work / '.finalization.json').write_text(json.dumps({'status': 'deferred', 'errors': [], 'deleted': []}))
print(json.dumps({'status': 'completed', 'archive': {'status': 'deferred'}}))
raise SystemExit(1)
''')
            for name in ("ditserve", "nvidia-smi", "df"):
                path = binary / name
                if not path.exists():
                    path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
            environment = dict(os.environ, FIXTURE=str(root), PATH=str(binary) + os.pathsep + os.environ['PATH'],
                DITSERVE_CALIBRATE='0', DITSERVE_PROFILE='fixture', DITSERVE_EXPERIMENT='quality',
                DITSERVE_QUALITY='1', DITSERVE_QUALITY_ARM='exact', DITSERVE_COUNT='1')
            for case, expected in (("pending", 0), ("failed", 1)):
                with self.subTest(case=case):
                    result = subprocess.run(['bash', str(scripts / source.name)], text=True, capture_output=True,
                        env={**environment, 'DITSERVE_RUN_ID': 'quality-' + case, 'CASE': case}, timeout=30)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                    self.assertEqual('quality_evaluation_pending' in result.stdout, case == 'pending')


if __name__ == '__main__':
    unittest.main()
