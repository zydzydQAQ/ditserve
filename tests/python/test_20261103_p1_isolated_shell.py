"""Local operator-copy integration; all external commands are CPU-only fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT=Path(__file__).resolve().parents[2]
RUNTIME=Path(os.environ.get('DITSERVE_KAIROS_RUNTIME_ROOT', ROOT.parent/'.local/ditserve-kairos-rewrite-runtime'))
OPERATOR=RUNTIME/'scripts/kairos_formal_benchmark.sh'
sys.path[:0]=[str(ROOT/'benchmarks'),str(ROOT/'workers')]
from ditserve.evaluation.common import canonical_hash

STUB=r'''import json,os,pathlib,sys
args=sys.argv[1:]; name=pathlib.Path(sys.argv[0]).name
def arg(key): return args[args.index(key)+1]
def log(value):
    with open(os.environ['MOCK_LOG'],'a') as out: out.write(value+'\n')
def write(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
mode=os.environ['CASE']
if name=='kairos_runtime_audit.py':
    log('runtime');write(pathlib.Path(os.environ['RUN_DIR'])/'runtime-facts.json',{})
elif name=='formal_benchmark.py':
    action=next(v for v in args if v in ('preflight','calibrate','assemble','serve'))
    log('formal-'+action)
    write(arg('--output'),dict(experiment='fixture'))
elif name=='p1_isolated_job.py':
    action=args[0];log('helper-'+action)
    if action=='preflight':
        if mode=='preflight': sys.exit(17)
        write(arg('--output'),{})
    elif action=='prepare':
        root=pathlib.Path(arg('--output-dir'));cases=[]
        for i in range(2):
            pair='wan21-'+str(i)
            case=dict(pair_id=pair,inputs=pair+'-inputs.json',inputs_sha256='a'*64,
                config=pair+'-config.json',run_id=os.environ['DITSERVE_RUN_ID']+'-'+pair)
            cases.append(case);write(root/case['config'],dict(experiment='fixture'))
            write(root/case['inputs'],{});write(root/(pair+'-planned-coverage.json'),dict(unexecuted=96))
        write(root/'execution-plan.json',dict(cases=cases))
elif name=='p1_isolated_batch.py':
    log('coverage-'+pathlib.Path(arg('--output')).stem)
    write(arg('--output'),dict(had_run='--run' in args))
elif name=='ditserve':
    if args[0]=='run':
        run=arg('--run-id');log('run-'+run)
        failed=(mode=='calibration' and run.endswith('-cal0')) or (mode=='replay' and run.endswith('-wan21-0'))
        if mode!='missing-run':
            write(pathlib.Path(os.environ['OUTPUT'])/'.cache/results-work/fixture'/run/'run.json',dict(status='failed' if failed else 'completed'))
        if failed or (mode=='missing-run' and run.endswith('-wan21-0')): sys.exit(19)
    else:
        run=pathlib.Path(args[-1]).name;log('verify-'+run)
        if mode=='verify' and run.endswith('-wan21-0'): sys.exit(23)
'''

@unittest.skipUnless(OPERATOR.is_file(), "Kairos operator snapshot is unavailable")
class ShellIntegration(unittest.TestCase):
    def run_case(self, mode):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            scripts=root/'release/scripts';scripts.mkdir(parents=True)
            tools=root/'release/tools';tools.mkdir()
            package=root/'package';(package/'benchmarks').mkdir(parents=True)
            (package/'configs/benchmarks').mkdir(parents=True)
            protocol=json.loads((ROOT/'configs/benchmarks/formal-mixed-v1.json').read_bytes())
            (package/'configs/benchmarks/formal-mixed-v1.json').write_text(json.dumps(protocol))
            (package/'configs/benchmarks/p1-qualification-v1.json').write_text('{}')
            shutil.copyfile(OPERATOR,scripts/OPERATOR.name)
            common='''set -euo pipefail
echo common-entered >> "$MOCK_LOG"
export OUTPUT="$MOCK_ROOT/output" DATA="$MOCK_ROOT/data" PACKAGE="$MOCK_ROOT/package"
export BUILD="$MOCK_ROOT/build" SCRIPTS="$MOCK_ROOT/release/scripts"
export RUN_DIR="$OUTPUT/acceptance/$DITSERVE_RUN_ID" LEDGERS="$MOCK_ROOT/ledgers"
mkdir -p "$RUN_DIR" "$LEDGERS"
trap 'echo trap-shared-write >> "$MOCK_LOG"' EXIT
'''
            (scripts/'kairos_common.sh').write_text(common)
            for path in (scripts/'kairos_runtime_audit.py',tools/'p1_isolated_job.py',
                         package/'benchmarks/formal_benchmark.py',package/'benchmarks/p1_isolated_batch.py'):
                path.write_text(STUB)
            binpath=root/'bin';binpath.mkdir()
            core=binpath/'ditserve';core.write_text('#!'+sys.executable+'\n'+STUB);core.chmod(0o755)
            df=binpath/'df';df.write_text('#!'+sys.executable+'\n'+'''import os,pathlib,sys
used,percent=700,70
if '-BG' in sys.argv:
    path=pathlib.Path(os.environ['MOCK_ROOT'])/'df-count'
    count=int(path.read_text())+1 if path.exists() else 1;path.write_text(str(count))
    mode=os.environ['CASE']
    if mode=='storage-initial-threshold' and count==1: used=791
    if mode=='storage-initial-percentage' and count==1: percent=80
    if mode=='storage-initial-unavailable' and count==1: sys.exit(1)
    if mode=='storage-at-replay' and count==5: used=791
    if mode=='storage-after-replay' and count==6: percent=80
    if mode=='storage-unavailable' and count==5: sys.exit(1)
print('Filesystem 1G-blocks Used Available Use% Mounted')
print(f'fixture 1000G {used}G 300G {percent}% /shared')
''');df.chmod(0o755)
            env=dict(os.environ,CASE=mode,MOCK_ROOT=str(root),MOCK_LOG=str(root/'events'),
                PATH=str(binpath)+':'+os.environ['PATH'],WAN_MODEL='wan',DITSERVE_RUN_ID='fixture',
                DITSERVE_MODEL_MANIFESTS=str(root/'models.json'),DITSERVE_PROTOCOL_SHA256=canonical_hash(protocol),
                DITSERVE_FORMAL_ACTION='p1_isolated',DITSERVE_JOB_SPEC_PATH=str(root/'spec.json'))
            result=subprocess.run(['bash',str(scripts/OPERATOR.name)],env=env,capture_output=True,text=True)
            events=(root/'events').read_text().splitlines() if (root/'events').exists() else []
            if mode.startswith('storage-initial-'):
                self.assertFalse((root/'output').exists(), 'common wrote shared output before storage permission')
                self.assertFalse((root/'ledgers').exists(), 'common ran before storage permission')
            prepared=root/'output/acceptance/fixture/isolated'
            coverage={p.name:json.loads(p.read_text()) for p in prepared.glob('*coverage.json')}
            return result,events,coverage

    def test_calibrations_precede_both_abbas(self):
        result,events,coverage=self.run_case('pass');self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([v for v in events if v.startswith('run-')],
            ['run-fixture-cal0','run-fixture-cal1','run-fixture-wan21-0','run-fixture-wan21-1'])
        self.assertLess(events.index('helper-preflight'),events.index('formal-calibrate'))
        self.assertLess(events.index('verify-fixture-cal1'),events.index('helper-prepare'))
        self.assertEqual(len(coverage),4)

    def test_preflight_or_calibration_failure_cannot_start_abba(self):
        for mode,code in [('preflight',17),('calibration',19)]:
            with self.subTest(mode=mode):
                result,events,coverage=self.run_case(mode);self.assertEqual(result.returncode,code,result.stderr)
                self.assertNotIn('helper-prepare',events);self.assertFalse(coverage)
                self.assertNotIn('verify-fixture-cal0',events)

    def test_replay_or_verification_failure_keeps_all_planned_denominators_and_stops(self):
        for mode,code in [('replay',19),('verify',23),('missing-run',19)]:
            with self.subTest(mode=mode):
                result,events,coverage=self.run_case(mode);self.assertEqual(result.returncode,code,result.stderr)
                self.assertNotIn('run-fixture-wan21-1',events)
                self.assertEqual(coverage['wan21-1-planned-coverage.json']['unexecuted'],96)
                self.assertEqual(coverage['wan21-0-coverage.json']['had_run'],mode!='missing-run')
                if mode=='replay': self.assertNotIn('verify-fixture-wan21-0',events)

    def test_storage_stop_before_or_after_replay_never_writes_coverage_or_exit_ledgers(self):
        for mode in ('storage-at-replay','storage-after-replay','storage-unavailable'):
            with self.subTest(mode=mode):
                result,events,coverage=self.run_case(mode)
                self.assertEqual(result.returncode,75,result.stderr)
                self.assertFalse(any(v.startswith('coverage-') for v in events))
                self.assertNotIn('run-fixture-wan21-1',events)
                self.assertNotIn('trap-shared-write',events)
                self.assertEqual(set(coverage),{'wan21-0-planned-coverage.json','wan21-1-planned-coverage.json'})
                self.assertEqual('run-fixture-wan21-0' in events,mode=='storage-after-replay')

    def test_initial_storage_failure_precedes_all_common_script_writes(self):
        for mode in ('storage-initial-threshold','storage-initial-percentage','storage-initial-unavailable'):
            with self.subTest(mode=mode):
                result,events,coverage=self.run_case(mode)
                self.assertEqual(result.returncode,75,result.stderr)
                self.assertEqual(events,[])
                self.assertEqual(coverage,{})

if __name__=='__main__':unittest.main()
