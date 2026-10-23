"""Weight-free workload-matrix and exactness checks; no torch/CUDA needed."""

import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.evaluation import workflow_encoder, workflow_overlap


class E4OutputRetentionTests(unittest.TestCase):
    def test_default_checks_pixels_without_writing_and_explicit_npy_preserves_them(self):
        import numpy as np
        pixels = np.arange(24, dtype=np.float32).reshape(1, 2, 2, 2, 3)
        cpu = SimpleNamespace(numpy=lambda: pixels)
        timeline = SimpleNamespace(phase=lambda *_: contextlib.nullcontext())
        with tempfile.TemporaryDirectory() as directory:
            args = workflow_overlap.parse_args(['--experiment', 'E4', '--out', directory])
            path = Path(directory) / 'output.npy'
            checked = workflow_overlap._persist_output(cpu, None, None, path, timeline, 'test', args.output_mode)
            self.assertFalse(path.exists())
            written = workflow_overlap._persist_output(cpu, None, None, path, timeline, 'test', 'npy')
            self.assertEqual(checked['sha256'], written['sha256'])
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), pixels)


class FakeQueue:
    def __init__(self):
        self.rows=[]

    def get(self,timeout=None):
        return self.rows.pop(0)


class FakeContext:
    def __init__(self, corrupt_other=False):
        self.corrupt_other=corrupt_other

    def Barrier(self,size):
        return object()

    def Queue(self):
        return FakeQueue()

    def Process(self,target,args):
        config,device,indices,preencoded,barrier,results,prefix=args
        mode=prefix.rsplit("/",1)[-1]
        milliseconds={"single":100,"single_other":120,"dual":70}[mode]
        requests=[dict(index=index,sha256="video-"+str(index),
                       latent=dict(sha256="latent-"+str(index)+("-changed" if mode=="single_other" and self.corrupt_other else "")))
                  for index in indices]
        value=dict(device=device,start_ns=1_000_000,end_ns=1_000_000+milliseconds*1_000_000,requests=requests)
        return SimpleNamespace(start=lambda:results.rows.append(dict(ok=True,result=value)),
            join=lambda timeout=None:None,exitcode=0,is_alive=lambda:False)


class E2MatrixTests(unittest.TestCase):
    def args(self,out):
        return SimpleNamespace(devices="0,1",device="cuda:1",requests=4,repeats=3,
                               timeout=10,out=Path(out))

    def test_three_arms_cover_both_single_cards_and_counterbalance_positions(self):
        with tempfile.TemporaryDirectory() as out, patch.object(workflow_overlap.multiprocessing,"get_context",return_value=FakeContext()):
            result=workflow_overlap.experiment_e2(self.args(out))
        self.assertEqual(len(result["pairs"]),6)
        for preencoded in (True,False):
            rows=[row for row in result["pairs"] if row["preencoded"]==preencoded]
            for arm in ("single","single_other","dual"):
                self.assertEqual({row["order"].index(arm) for row in rows},{0,1,2})
            for row in rows:
                self.assertEqual(row["modes"]["single"]["devices"],["cuda:1"])
                self.assertEqual(row["modes"]["single_other"]["devices"],["cuda:0"])
                self.assertEqual(row["modes"]["dual"]["devices"],["cuda:0","cuda:1"])
                self.assertEqual(row["speedup"],100/70)
                self.assertEqual(row["other_single_speedup"],120/70)
                self.assertTrue(all(check["exact"] for check in row["correctness"]))
                self.assertTrue(all(check["exact"] for check in row["other_single_correctness"]))

    def test_other_card_latent_difference_is_rejected(self):
        with tempfile.TemporaryDirectory() as out, patch.object(workflow_overlap.multiprocessing,"get_context",return_value=FakeContext(True)):
            with self.assertRaisesRegex(RuntimeError,"output differs"):
                workflow_overlap.experiment_e2(self.args(out))


class FakeArray:
    def __init__(self,value):
        self.value=value

    def tobytes(self):
        return self.value.encode()


class FakeTensor:
    def __init__(self,value):
        self.value=value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return FakeArray(self.value)


class T5ExactnessTests(unittest.TestCase):
    def run_queue(self,corrupt_latent=False):
        def prepare(spec,payload):
            value="latent-"+str(spec["seed"])
            if corrupt_latent and "prefetch" in spec["request_id"]:
                value+="-changed"
            return SimpleNamespace(spec=spec,latents=FakeTensor(value))
        worker=SimpleNamespace(_owner=object(),prepare_encoded=prepare,step=lambda state:None,release=lambda state:None)
        recorder=SimpleNamespace(phase=lambda *args,**kwargs:contextlib.nullcontext())
        args=SimpleNamespace(frames=[17],queue_repeats=1,steps=2,pipeline_requests=2)
        encode=lambda *arguments:((object(),object()),1.)
        with patch.object(workflow_encoder,"_decode_separate",side_effect=lambda worker,state,*rest:FakeArray("video-"+str(state.spec["seed"]))):
            return workflow_encoder._dedicated_queue(worker,encode,recorder,args)

    def test_video_format_preserved_and_latent_hashes_added(self):
        rows=self.run_queue()
        self.assertEqual(len(rows),2)
        self.assertIsNone(rows[0]["paired_exact"])
        self.assertTrue(rows[1]["paired_exact"])
        self.assertEqual(rows[0]["video_sha256"],rows[1]["video_sha256"])
        self.assertEqual(rows[0]["latent_sha256"],rows[1]["latent_sha256"])
        self.assertTrue(all(isinstance(value,str) for value in rows[0]["video_sha256"]))

    def test_matching_video_does_not_hide_changed_latent(self):
        with self.assertRaisesRegex(RuntimeError,"video or latent"):
            self.run_queue(corrupt_latent=True)


if __name__=="__main__":
    unittest.main()
