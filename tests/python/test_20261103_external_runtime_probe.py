"""Safety-relevant failures in the actual runtime probe wrapper."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('external_runtime_probe',ROOT/'benchmarks/external_runtime_probe.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ExternalRuntimeProbeTest(unittest.TestCase):
    def test_uninventoried_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root/'probe.py').write_text('pass\n')
            files = {'probe.py':m.sha(root/'probe.py')}
            m.verify_inventory(root,files)
            (root/'shadow.py').write_text('pass\n')
            with self.assertRaisesRegex(ValueError,'unmanifested'):
                m.verify_inventory(root,files)

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()
            (root/'probe.py').write_text('pass\n')
            (root/'shadow.py').symlink_to(root/'probe.py')
            with self.assertRaises(ValueError):
                m.verify_inventory(root,{'shadow.py':m.sha(root/'probe.py')})

    def test_exit_two_preserves_real_blocker(self):
        result=m.run_command([sys.executable,'-B','-c',
            'import json; print(json.dumps({"qualified":False,"blocker":"missing model"})); raise SystemExit(2)'],10,os.environ.copy())
        self.assertEqual(result['returncode'],2)
        self.assertFalse(result['result']['qualified'])
        self.assertEqual(result['result']['blocker'],'missing model')

    def test_timeout_terminates_actual_process(self):
        with patch.object(m,'storage',return_value={}):
            result=m.run_command([sys.executable,'-B','-c','import time; time.sleep(30)'],.1,os.environ.copy())
        self.assertTrue(result['timed_out'])
        self.assertNotEqual(result['returncode'],0)

    def test_storage_threshold_fails_closed(self):
        for used,percent,fail in ((800,79,False),(801,79,True),(800,80,True)):
            result=subprocess.CompletedProcess([],0,f'Filesystem 1G-blocks Used Available Use% Mounted on\n/dev/sdb 1024G {used}G 234G {percent}% /shared\n','')
            with patch.object(m.subprocess,'run',return_value=result):
                if fail:
                    with self.assertRaises(SystemExit) as caught:m.storage()
                    self.assertEqual(caught.exception.code,75)
                else:self.assertEqual(m.storage()['used_gib'],800)

    def test_missing_df_fails_closed(self):
        with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','error')):
            with self.assertRaises(SystemExit) as caught:m.storage()
            self.assertEqual(caught.exception.code,75)

if __name__=='__main__':unittest.main()
