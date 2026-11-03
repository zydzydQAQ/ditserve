"""CPU-only validation of native distributed plans and rank evidence gates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import json
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_xdit_distributed as distributed


def receipts(world=2):
    request=dict(height=480,width=832,num_frames=81,num_inference_steps=50,guidance_scale=5.,
                 negative_prompt='',prompt='Unit fixture, not generated output.',seed=42)
    row=dict(status='native_rank_completed',cleanup_completed=True,planned_returncode=0,
        world_size=world,actual_sp_degree=world,actual_dp_degree=1,qualification_eligible=False,
        request=request,effective_request=dict(request),model_identity={'manifest_sha256':'a'*64},
        source={'commit':'unit fixture'},protocol_sha256='b'*64,adapter={'sha':'fixture'},native_dtype='bfloat16',
        output=dict(shape=[81,480,832,3],finite=True,outputs=1,floating_point_frames=True,dtype='float32',sha256='c'*64),
        loaded_source_sha256={'xfuser/fixture.py':'d'*64})
    return [dict(deepcopy(row),rank=i,local_rank=i,gpu_uuid='601e4969-39bd-d7bc-af70-7e99a34cd52'+str(i)) for i in range(world)]


class DistributedNativeTests(unittest.TestCase):
    def test_only_valid_single_node_torchrun_topology(self):
        for world in (2,4):
            for rank in range(world):
                env=dict(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(world),LOCAL_WORLD_SIZE=str(world))
                self.assertEqual(distributed.topology(world,env),rank)
        for world in (True,1,3,8):
            with self.assertRaises(ValueError):distributed.topology(world,{})
        for env in ({},dict(RANK='0',LOCAL_RANK='0',WORLD_SIZE='4',LOCAL_WORLD_SIZE='2'),
                    dict(RANK='1',LOCAL_RANK='0',WORLD_SIZE='2',LOCAL_WORLD_SIZE='2')):
            with self.assertRaises(ValueError):distributed.topology(2,env)

    def test_parallel_plan_changes_only_native_ulysses_and_disables_ray(self):
        request=receipts()[0]['request'];identity={'flow_shift':3}
        for world in (2,4):
            actual=distributed.distributed_config('wan21',request,identity,Path('/unit-fixture'),world)
            expected=distributed.native.runner_config('wan21',request,identity,Path('/unit-fixture'),False)
            expected.update(ulysses_degree=world,ring_degree=1,use_ray=False)
            self.assertEqual(actual,expected)
            self.assertFalse(actual['enable_model_cpu_offload'])
            self.assertEqual(actual['num_inference_steps'],50)
            self.assertEqual(actual['data_parallel_degree'],1)

    def test_complete_unique_ranks_only_count_one_logical_request(self):
        for world in (2,4):
            result=distributed.aggregate(receipts(world),world)
            self.assertEqual(result['completed_requests'],1)
            self.assertEqual(result['selected_output_rank'],0)
            self.assertFalse(result['torchrun_exit_verified'])
            self.assertFalse(result['qualification_eligible'])
            self.assertTrue(result['cross_rank_output_digest_equal'])

    def test_missing_failed_or_mismatched_rank_never_passes(self):
        mutations=[lambda r:r.pop(),lambda r:r[1].update(rank=0),
            lambda r:r[1].update(gpu_uuid=r[0]['gpu_uuid']),
            lambda r:r[1].update(cleanup_completed=False),lambda r:r[1].update(planned_returncode=1),
            lambda r:r[1].update(status='native_rank_failed'),lambda r:r[1].update(actual_sp_degree=1),
            lambda r:r[1].update(actual_dp_degree=2),lambda r:r[1].update(loaded_source_sha256={}),
            lambda r:r[1]['source'].update(commit='other'),lambda r:r[1]['effective_request'].update(num_inference_steps=1),
            lambda r:r[1]['output'].update(shape=[1,480,832,3]),lambda r:r[1]['output'].update(finite=False),
            lambda r:r[1]['output'].update(floating_point_frames=False)]
        for mutate in mutations:
            rows=receipts();mutate(rows)
            with self.assertRaises(ValueError):distributed.aggregate(rows,2)

    def test_different_rank_output_fails_without_claiming_sp1_equivalence(self):
        for field,value in [('sha256','f'*64),('dtype','float16')]:
            rows=receipts();rows[1]['output'][field]=value
            with self.assertRaisesRegex(ValueError,'SP ranks disagree'):
                distributed.aggregate(rows,2)

    def test_actual_bare_or_gpu_prefixed_uuid_normalizes_and_detects_aliases(self):
        rows=receipts();rows[0]['gpu_uuid']='GPU-'+rows[0]['gpu_uuid']
        result=distributed.aggregate(rows,2)
        self.assertEqual(result['normalized_gpu_uuids'][0],rows[0]['gpu_uuid'][4:])
        rows[1]['gpu_uuid']=rows[0]['gpu_uuid'][4:]
        with self.assertRaisesRegex(ValueError,'distinct'):distributed.aggregate(rows,2)
        for value in (None,'','GPU-unit-fixture-0','not-a-uuid'):
            with self.assertRaises(ValueError):distributed.normalize_uuid(value)

    def test_rank_zero_failure_returns_without_waiting_for_blocked_peers(self):
        with TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'rank-0').mkdir()
            failed=dict(rank=0,planned_returncode=1,error='actual unit fault',failed_stage='load_model')
            distributed.publish_json(output/'rank-0'/'summary.json',failed)
            with patch.object(distributed.time,'sleep',side_effect=AssertionError('must not wait for peers')):
                result=distributed.collect_rank_summary(output,failed,4,float('inf'))
            self.assertEqual(result['status'],'native_sp_request_failed')
            self.assertEqual(result['completed_requests'],0)
            self.assertEqual(result['available_rank_receipts'],[failed])
            self.assertEqual(result['failed_stage'],'load_model')
            self.assertFalse(result['qualification_eligible'])

    def test_atomic_receipt_publication_never_exposes_partial_json(self):
        with TemporaryDirectory() as directory:
            target=Path(directory)/'summary.json'
            payload={'unit_fixture':'x'*40000}
            original=Path.rename
            def checked_rename(source,destination):
                self.assertFalse(target.exists())
                self.assertEqual(json.loads(source.read_bytes()),payload)
                return original(source,destination)
            with patch.object(Path,'rename',checked_rename):
                distributed.publish_json(target,payload)
            self.assertEqual(json.loads(target.read_bytes()),payload)
            before=target.read_bytes()
            with self.assertRaises(FileExistsError):distributed.publish_json(target,{'changed':True})
            self.assertEqual(target.read_bytes(),before)


if __name__=='__main__':unittest.main()
