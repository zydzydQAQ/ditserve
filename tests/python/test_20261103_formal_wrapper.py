"""The formal wrapper must cover frozen shapes and preserve failed evidence."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get('DITSERVE_KAIROS_RUNTIME_ROOT', ROOT.parent / '.local/ditserve-kairos-rewrite-runtime'))
SOURCE = LOCAL / 'scripts/kairos_formal_benchmark.sh'


@unittest.skipUnless(SOURCE.is_file(), 'operator tooling not installed')
class FormalWrapperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.scripts = self.root / 'scripts'
        self.package = self.root / 'package'
        for path in (self.scripts, self.package / 'configs/benchmarks', self.package / 'benchmarks',
                     self.root / 'bin', self.root / 'run', self.root / 'output', self.root / 'ledgers'):
            path.mkdir(parents=True)
        shutil.copyfile(SOURCE, self.scripts / SOURCE.name)
        (self.scripts / 'kairos_common.sh').write_text('''set -euo pipefail
PACKAGE=$FIXTURE/package
SCRIPTS=$FIXTURE/scripts
RUN_DIR=$FIXTURE/run
OUTPUT=$FIXTURE/output
LEDGERS=$FIXTURE/ledgers
DATA=$FIXTURE/data
trap 'touch "$FIXTURE/parent-exited"' EXIT
''')
        (self.scripts / 'kairos_runtime_audit.py').write_text('')
        (self.package / 'benchmarks/formal_benchmark.py').write_text('''import json, pathlib, sys
args = sys.argv
action = args[3]
root = pathlib.Path(__import__('os').environ['FIXTURE'])
with (root/'calls.jsonl').open('a') as stream:
    stream.write(json.dumps(args[1:])+'\\n')
output = pathlib.Path(args[args.index('--output')+1])
if action == 'preflight':
    if __import__('os').environ.get('FAIL_PREFLIGHT') == '1':
        raise ValueError('insufficient unseen prompts')
    output.write_text('{}')
elif action == 'calibrate':
    shape = args[args.index('--video-shape')+1]
    output.write_text(json.dumps({'experiment': 'cal-'+shape}))
elif action == 'assemble':
    output.write_text('{}')
else:
    raise ValueError('unexpected fixture action')
''')
        self.executable('df', '#!/bin/sh\nprintf "Filesystem Size Used Avail Use%% Mounted\\nfixture 1000G 20G 980G 2%% /shared\\n"\n')
        self.executable('ditserve', '''#!/usr/bin/env python3
import json, os, pathlib, sys
root = pathlib.Path(os.environ['FIXTURE'])
args = sys.argv
if args[1] == 'run':
    config = json.loads(pathlib.Path(args[args.index('--config')+1]).read_bytes())
    run_id = args[args.index('--run-id')+1]
    path = root/'output/.cache/results-work'/config['experiment']/run_id/'run.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    path.chmod(0o600)
    if os.environ.get('FAIL_REPLAY') == '1':
        sys.exit(9)
''')
        self.protocol = {'shapes': {
            'new-short-video': {'model': 'wan21', 'kind': 'video', 'scope': 'formal'},
            'new-long-video': {'model': 'wan21', 'kind': 'video', 'scope': 'formal'},
            'new-small-image': {'model': 'sd35', 'kind': 'image', 'scope': 'formal'},
            'new-large-image': {'model': 'sd35', 'kind': 'image', 'scope': 'formal'},
        }}

    def executable(self, name, content):
        path = self.root / 'bin' / name
        path.write_text(content)
        path.chmod(0o755)

    def test_p1_runs_one_formal_pair_and_all_frozen_seeds(self):
        import ditserve
        (self.package / 'python').symlink_to(Path(ditserve.__file__).resolve().parents[1], target_is_directory=True)
        plan = dict(schema='ditserve.p1-plan.v1', seeds=[201, 202, 203],
                    roles={'contention': {}, 'no_preemption': {}}, shape_pairs=[dict(
                        id='pair-a', model='wan21', video_shape='new-long-video', image_shape='new-large-image')])
        (self.package / 'configs/benchmarks/p1-qualification-v1.json').write_text(json.dumps(plan))
        # The actual plan validator/generator has separate adversarial tests.
        # Here the stub records shell routing and preserves exact arguments.
        (self.package / 'benchmarks/p1_benchmark.py').write_text('''def validate_plan(plan, protocol):
    assert plan['schema'] == 'ditserve.p1-plan.v1'
if __name__ == '__main__':
    import json, os, pathlib, sys
    root = pathlib.Path(os.environ['FIXTURE'])
    args = sys.argv
    with (root/'p1-calls.jsonl').open('a') as stream:
        stream.write(json.dumps(args[1:])+'\\n')
    pathlib.Path(args[args.index('--output')+1]).write_text(json.dumps({'experiment': 'p1-fixture'}))
''')
        digest = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False,
                                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        result = self.run_wrapper(DITSERVE_FORMAL_ACTION='p1', DITSERVE_P1_PAIR='pair-a',
                                  DITSERVE_P1_ROLE='contention', DITSERVE_P1_PLAN_SHA256=digest)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        calibrations = [row for row in calls if 'calibrate' in row]
        self.assertEqual(len(calibrations), 1)
        self.assertIn('new-long-video', calibrations[0])
        self.assertIn('new-large-image', calibrations[0])
        p1_calls = [json.loads(line) for line in (self.root / 'p1-calls.jsonl').read_text().splitlines()]
        self.assertEqual([row[row.index('--seed')+1] for row in p1_calls], ['201', '202', '203'])
        self.assertTrue(all(row[row.index('--role')+1] == 'contention' for row in p1_calls))

    def run_wrapper(self, **extra):
        (self.package / 'configs/benchmarks/formal-mixed-v1.json').write_text(json.dumps(self.protocol))
        digest = hashlib.sha256(json.dumps(self.protocol, sort_keys=True, ensure_ascii=False,
                                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        env = dict(os.environ, FIXTURE=str(self.root), PATH=str(self.root/'bin')+os.pathsep+os.environ['PATH'],
                   DITSERVE_MODEL_MANIFESTS='fixture', DITSERVE_PROTOCOL_SHA256=digest,
               DITSERVE_RUN_ID='parent', WAN_MODEL='wan', DITSERVE_FORMAL_ACTION='calibration')
        env.update(extra)
        return subprocess.run(['bash', str(self.scripts / SOURCE.name)], env=env,
                              text=True, capture_output=True, timeout=30)

    def test_uses_every_protocol_shape_and_preserves_parent_cleanup(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in (self.root/'calls.jsonl').read_text().splitlines()]
        pairs = [(args[args.index('--video-shape')+1], args[args.index('--image-shape')+1])
                 for args in calls if args[2] == 'calibrate']
        self.assertEqual(pairs, [('new-short-video', 'new-small-image'), ('new-long-video', 'new-large-image')])
        raw = list((self.root/'output').rglob('run.json'))
        self.assertEqual(len(raw), 2)
        self.assertTrue(all(path.stat().st_mode & 0o444 == 0o444 for path in raw))
        self.assertTrue((self.root/'parent-exited').is_file())

    def test_failed_replay_evidence_remains_readable(self):
        result = self.run_wrapper(FAIL_REPLAY='1')
        self.assertEqual(result.returncode, 9, result.stderr)
        raw = list((self.root/'output').rglob('run.json'))
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0].stat().st_mode & 0o444, 0o444)
        self.assertTrue((self.root/'parent-exited').is_file())
        self.assertFalse((self.root/'run/bundle.json').exists())

    def test_unpaired_shapes_stop_before_replay(self):
        del self.protocol['shapes']['new-large-image']
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('independent paired calibration', result.stderr)
        self.assertFalse((self.root/'calls.jsonl').exists())

    def test_input_preflight_failure_precedes_gpu_replay(self):
        result = self.run_wrapper(FAIL_PREFLIGHT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('insufficient unseen prompts', result.stderr)
        self.assertEqual(list((self.root/'output').rglob('run.json')), [])
        calls = [json.loads(line) for line in (self.root/'calls.jsonl').read_text().splitlines()]
        self.assertEqual([args[2] for args in calls], ['preflight'])


if __name__ == '__main__':
    unittest.main()
