"""CPU-only admission and coordination checks for incremental postprocessing."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('workflow_watch',Path(__file__).resolve().parents[2]/'docker/watch-workflow-postprocess.py')
watch=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'run01';self.root.mkdir()
        self.manifest=self.root/'jobs-trace.json'
        self.jobs=[]
        self.args=SimpleNamespace(root=self.root,jobs=[self.manifest],nsys=Path('/fake/nsys'),
                                  ncu=Path('/fake/ncu'),timeout=10,reparse=False)

    def job(self,name='T0',status='completed'):
        path=self.root/'traces'/f'{name}.nsys-rep';path.parent.mkdir(exist_ok=True);path.write_bytes(b'capture')
        stamp=path.stat().st_mtime_ns
        job=dict(id='trace-'+name,source='/frozen/v2',command=['/tools/nsys','profile',
             '--output=/data/workflow/run01/traces/'+name,'python','-m','ditserve.evaluation.workflow_encoder',
             '--child',name,'--output','/data/workflow/run01/profiling/encoder'])
        value=dict(id=job['id'],status=status,returncode=0,source=job['source'],command=['wrapper','--counters',*job['command']],
            started_unix_ns=stamp-1000000,finished_unix_ns=stamp+1000000,
            gpu_monitor=dict(admission='passed',conflicts=[],errors=[]))
        ledger=self.root/'launches'/job['id']/'launch.json';ledger.parent.mkdir(parents=True);ledger.write_text(json.dumps(value))
        self.jobs.append(job);self.manifest.write_text(json.dumps(dict(root=str(self.root),jobs=self.jobs)))
        return job,path,ledger,value

    def candidate(self,job):
        return watch.completed_capture(self.root,job,watch.launch_state(self.root,job))[0]

    def test_writing_failed_and_unmonitored_captures_are_never_admitted(self):
        for index,status in enumerate(('running','failed','timeout','interrupted','gpu_monitor_unavailable')):
            job,path,_,_=self.job('T'+str(index),status)
            self.assertTrue(path.exists())
            self.assertIsNone(self.candidate(job))
        job,_,ledger,value=self.job('T5')
        for monitor in (dict(admission='passed',conflicts=[{'pid':9}],errors=[]),
                        dict(admission='passed',conflicts=[],errors=['lost NVML']),None):
            value['gpu_monitor']=monitor;ledger.write_text(json.dumps(value))
            self.assertIsNone(self.candidate(job))

    def test_exact_command_source_and_capture_launch_window_required(self):
        job,path,ledger,value=self.job()
        self.assertIsNotNone(self.candidate(job))
        for change in (dict(returncode=1),dict(source='/frozen/other'),dict(command=['different']),
                       dict(started_unix_ns=path.stat().st_mtime_ns+1),dict(finished_unix_ns=path.stat().st_mtime_ns-1)):
            ledger.write_text(json.dumps(dict(value,**change)))
            self.assertIsNone(self.candidate(job),change)
        ledger.write_text(json.dumps(value))
        saved=path.with_suffix('.saved');path.rename(saved);path.symlink_to(saved)
        self.assertIsNone(self.candidate(job))

    def test_launch_restart_and_shared_output_writer_invalidate_candidate(self):
        job,_,ledger,value=self.job();candidate=self.candidate(job)
        value['status']='running';ledger.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError,'became ineligible'):
            watch.validate_candidate(self.root,[self.manifest],candidate)
        value['status']='completed';ledger.write_text(json.dumps(value));candidate=self.candidate(job)
        self.jobs.append(dict(job,id='trace-retry'))
        self.manifest.write_text(json.dumps(dict(root=str(self.root),jobs=self.jobs)))
        with self.assertRaisesRegex(RuntimeError,'unfinished job shares'):
            watch.validate_candidate(self.root,[self.manifest],candidate)

    def test_changed_launch_before_publish_leaves_no_parsed_evidence(self):
        job,path,ledger,value=self.job();candidate=self.candidate(job)
        def parse(*_):
            value['status']='running';ledger.write_text(json.dumps(value))
            return dict(schema_version='ditserve.workflow_trace.v1',status='completed')
        with patch.object(watch.HELPER,'parse_nsys',side_effect=parse):
            action=watch.process_pass(self.args,[candidate],threading.Event())
        self.assertEqual(action['processed'][0]['status'],'failed')
        self.assertFalse(path.with_suffix('.parsed.json').exists())

    def test_shared_lock_is_nonblocking_for_watcher(self):
        job,_,_,_=self.job();candidate=self.candidate(job)
        with (self.root/'.postprocess.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with patch.object(watch.HELPER,'process_capture') as process:
                result=watch.process_pass(self.args,[candidate],threading.Event())
            self.assertEqual(result['status'],'lock_busy');process.assert_not_called()

    def test_primary_waits_for_current_capture_without_failing_pipeline(self):
        lock=(self.root/'.postprocess.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        observed=[]
        def release():
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                if (self.root/'.postprocess-primary-waiting.json').exists():
                    observed.append(True);break
                time.sleep(.01)
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        thread=threading.Thread(target=release);thread.start()
        with patch('builtins.print'):
            result=watch.HELPER.main(['--root',str(self.root),'--lock-wait','2'])
        thread.join(3)
        self.assertEqual(result,0);self.assertEqual(observed,[True])
        self.assertFalse((self.root/'.postprocess-primary-waiting.json').exists())

    def test_primary_zero_wait_reports_contention_immediately(self):
        with (self.root/'.postprocess.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            with self.assertRaisesRegex(RuntimeError,'another postprocessor'):
                watch.HELPER.main(['--root',str(self.root),'--lock-wait','0'])
        self.assertFalse((self.root/'.postprocess-primary-waiting.json').exists())

    def test_watcher_yields_between_captures_and_keeps_completed_result(self):
        first,_,_,_=self.job('T0');second,_,_,_=self.job('T1')
        candidates=[self.candidate(first),self.candidate(second)]
        def parsed(kind,path,args,previous,validate):
            identity=watch.process_identity(os.getpid())
            watch.HELPER.atomic_json(self.root/'.postprocess-primary-waiting.json',
                dict(pid=identity['pid'],start_ticks=identity['start_ticks'],deadline_unix_ns=time.time_ns()+10**9))
            return dict(capture=str(path),status='parsed')
        with patch.object(watch.HELPER,'process_capture',side_effect=parsed) as process,patch.object(watch,'run_child') as child:
            result=watch.process_pass(self.args,candidates,threading.Event(),preview=True)
        self.assertEqual(result['status'],'yielded_primary');self.assertEqual(process.call_count,1);child.assert_not_called()
        ledger=json.loads((self.root/'postprocess.json').read_text())
        self.assertEqual(len(ledger['captures']),1)

    def test_historical_failure_does_not_stop_live_resume_or_suite(self):
        job,_,_,_=self.job(status='gpu_monitor_unavailable')
        states=[watch.launch_state(self.root,job)]
        for command in (["bash",str(self.root/'resume-after-reboot.sh')],
                        ['python','docker/run-workflow-suite.py','--jobs',str(self.manifest)]):
            runners=watch.active_runners(self.root,[self.manifest],states,
                identities=[dict(pid=100,start_ticks=10,command=command)])
            self.assertTrue(runners)
            self.assertEqual(watch.terminal_status(states,runners,0,1000,90),(None,None))
        self.assertEqual(watch.terminal_status(states,[],0,1000,90)[0],'pipeline_terminal')
        self.assertEqual(watch.terminal_status([dict(status='not_started')],[],0,1000,90)[0],'pipeline_stopped_incomplete')

    def test_preview_command_is_cpu_report_only(self):
        with patch.object(watch,'run_child',return_value=[]) as child:
            result=watch.process_pass(self.args,[],threading.Event(),preview=True)
        command=child.call_args.args[0]
        self.assertEqual(command[1:3],['-m','ditserve.evaluation.workflow_report'])
        self.assertIn('--preview',command);self.assertNotIn('--destination',command)
        self.assertEqual(result['preview']['status'],'completed')

    def test_main_requires_two_stable_polls_then_exits_after_terminal_grace(self):
        self.job()
        clock=[0.]
        class Event:
            def is_set(self):return False
            def wait(self,seconds):clock[0]+=seconds
        def process(args,candidates,stop,preview=False):
            return dict(status='completed',processed=[dict(capture=row['path'],status='parsed') for row in candidates],
                        preview=dict(status='completed'))
        with patch.object(watch.os,'sched_getaffinity',return_value={10}),patch.object(watch.os,'sched_setaffinity') as affinity, \
             patch.object(watch.os,'getpriority',return_value=19),patch.object(watch.signal,'signal'), \
             patch.object(watch.threading,'Event',Event),patch.object(watch.time,'monotonic',side_effect=lambda:clock[0]), \
             patch.object(watch,'active_runners',return_value=[]),patch.object(watch,'process_pass',side_effect=process) as run, \
             patch('builtins.print'):
            self.assertEqual(watch.main(['--root',str(self.root),'--jobs',str(self.manifest)]),0)
        self.assertEqual([len(call.args[1]) for call in run.call_args_list],[0,1])
        self.assertEqual(clock[0],90)
        affinity.assert_called_once_with(0,{10})
        state=json.loads((self.root/'watch-postprocess.json').read_text())
        self.assertEqual(state['status'],'pipeline_terminal')
        self.assertEqual(state['identity']['pid'],os.getpid())
        self.assertEqual(len(state['passes']),3)


if __name__=='__main__':unittest.main()
