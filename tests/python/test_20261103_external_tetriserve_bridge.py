"""Bridge metadata checks plus optional real CPU Gloo process-group exercise."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_tetriserve_bridge as bridge

HAS_TORCH = importlib.util.find_spec('torch') is not None
SHARED = os.environ.get('DITSERVE_SHARED_PYTHON_ROOT')


def request(batch=2, start=0, end=2):
    return SimpleNamespace(input_config=SimpleNamespace(batch_size=batch, prompt=['fox'] * batch,
        negative_prompt='blurry, low quality', guidance_scale=4.5, max_sequence_length=256,
        height=512, width=512, num_inference_steps=end-start, seed=101),
        sub_req_ids=list(range(1,batch+1)), origin_steps=[28]*batch,
        start_step=start, end_step=end, is_last_steps=[end==28]*batch)


def gloo_child(rank, store, shared, queue):
    # Actual Gloo groups exercise the borrowing contract; this is not a mocked
    # collective or a substitute for official NCCL/LatentManager GPU tests.
    try:
        sys.path.insert(0, shared)
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        dist.init_process_group('gloo', init_method='file://' + store, rank=rank,
                                world_size=2, timeout=timedelta(seconds=30))
        mapping, cpus = {}, {}
        for ranks in [(0,), (1,), (0,1)]:
            mapping[str(list(ranks))] = dist.new_group(list(ranks), backend='gloo')
            cpus[str(list(ranks))] = dist.new_group(list(ranks), backend='gloo')
        class Owner:
            generated = True
            all_process_groups_map = mapping
            all_cpu_groups_map = cpus
            def get_process_group(self, ranks, device='cuda'):
                return (mapping if device == 'cuda' else cpus)[str(sorted(ranks))]
        original = dist.new_group
        def forbidden(*a, **k):
            raise AssertionError('bridge attempted to allocate a process group')
        dist.new_group = forbidden
        try:
            pool = bridge.borrowed_pool(Owner(), rank, 2, 'cpu')
            executions = 0
            for selected in [(0,), (0,1), (1,), (0,1)]:
                if rank in selected:
                    pool.activate(selected)
                    from xfuser.core.distributed.parallel_state import get_sp_group
                    output = get_sp_group().all_gather(torch.tensor([float(rank+1)]))
                    if not torch.equal(output, torch.tensor([float(r+1) for r in selected])):
                        raise AssertionError('borrowed Gloo rank order/data differs')
                    executions += 1
                dist.barrier()
            pool.close()
            # Closing a borrower must leave the owner's groups usable.
            value = torch.tensor([float(rank+1)])
            dist.all_reduce(value, group=cpus['[0, 1]'])
            if value.item() != 3.:
                raise AssertionError('borrower destroyed or corrupted owner group')
            queue.put({'rank':rank, 'status':'passed', 'executions':executions, 'new_groups':0})
        finally:
            dist.new_group = original
            dist.destroy_process_group()
    except Exception as exc:
        queue.put({'rank':rank, 'status':'failed', 'error':repr(exc)})
        raise


class TetriBridgeTests(unittest.TestCase):
    def test_sorted_native_rank_contract_and_group_inventory(self):
        self.assertEqual(bridge.ranks_tuple([0,1],2), (0,1))
        self.assertEqual(bridge.ranks_tuple([1],2), (1,))
        for ranks, world in [([1,0],2), ([0,0],2), ([True],2), ([2],2), ([0],8)]:
            with self.assertRaises(ValueError): bridge.ranks_tuple(ranks,world)
        owner = SimpleNamespace(generated=True, all_process_groups_map={'[0]':1,'[1]':2,'[0, 1]':3},
                                 all_cpu_groups_map={'[0]':4,'[1]':5,'[0, 1]':6})
        self.assertEqual(set(bridge.existing_groups(owner,2)), {(0,), (1,), (0,1)})
        del owner.all_cpu_groups_map['[1]']
        with self.assertRaises(ValueError): bridge.existing_groups(owner,2)

    def test_native_future_cursor_must_match_new_absolute_round(self):
        bridge.validate_request(request())
        req = request(start=2,end=28)
        futures = {i:SimpleNamespace(sub_req_id=i,step_range=(0,2)) for i in req.sub_req_ids}
        bridge.validate_request(req,futures)
        for change in ('missing','replay','gap','identity'):
            bad = deepcopy(futures)
            if change == 'missing': del bad[1]
            if change == 'replay': bad[1].step_range = (0,1)
            if change == 'gap': bad[1].step_range = (0,3)
            if change == 'identity': bad[1].sub_req_id = 7
            with self.assertRaises(ValueError): bridge.validate_request(req,bad)
        with self.assertRaisesRegex(ValueError,'replay'):
            bridge.validate_request(request(), futures)

    def test_batch_identity_guidance_and_final_marker_fail_closed(self):
        for mutation in ('prompts','guidance','negative','steps','origins','final','duplicates'):
            req = request()
            if mutation == 'prompts': req.input_config.prompt = ['fox']
            if mutation == 'guidance': req.input_config.guidance_scale = 7.
            if mutation == 'negative': req.input_config.negative_prompt = ['x','y']
            if mutation == 'steps': req.input_config.num_inference_steps = 1
            if mutation == 'origins': req.origin_steps = [28]
            if mutation == 'final': req.is_last_steps = [True,False]
            if mutation == 'duplicates': req.sub_req_ids = [1,1]
            with self.assertRaises(ValueError): bridge.validate_request(req)

    @unittest.skipUnless(HAS_TORCH, 'real Torch required; no fake tensor validation')
    def test_real_cpu_tensors_reject_mixed_schedules_and_broken_cfg_pairs(self):
        import torch
        req=request()
        arrays=[[torch.zeros(2,3,4) for _ in range(2)], [torch.zeros(2,5) for _ in range(2)],
                [torch.zeros(1,16,64,64) for _ in range(2)],
                [torch.linspace(1000,1,28) for _ in range(2)], [torch.linspace(1,0,29) for _ in range(2)]]
        bridge.validate_transferred_batch(req,*arrays)
        arrays[3][1] = arrays[3][1] + 1.
        with self.assertRaisesRegex(ValueError,'different timestep'):
            bridge.validate_transferred_batch(req,*arrays)
        arrays[3][1] = arrays[3][0].clone()
        arrays[0][1] = torch.zeros(1,3,4)
        with self.assertRaisesRegex(ValueError,'unconditional, conditional'):
            bridge.validate_transferred_batch(req,*arrays)

    @unittest.skipUnless(HAS_TORCH and SHARED, 'set DITSERVE_SHARED_PYTHON_ROOT in real Torch environment for CPU Gloo test')
    def test_real_gloo_sp1_sp2_rank_switch_without_new_groups(self):
        import torch.multiprocessing as mp
        bridge.verify_shared_sources(SHARED)
        with TemporaryDirectory() as directory:
            context=mp.get_context('spawn')
            queue=context.Queue()
            processes=[context.Process(target=gloo_child,args=(rank,str(Path(directory)/'store'),SHARED,queue))
                       for rank in range(2)]
            for process in processes: process.start()
            try:
                records=[queue.get(timeout=60) for _ in processes]
                for process in processes: process.join(timeout=10)
                self.assertTrue(all(p.exitcode == 0 for p in processes),records)
                self.assertEqual(sorted((r['rank'],r['status'],r['executions'],r['new_groups']) for r in records),
                                 [(0,'passed',3,0),(1,'passed',3,0)])
            finally:
                for process in processes:
                    if process.is_alive(): process.terminate()
                for process in processes: process.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
