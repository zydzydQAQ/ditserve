"""CPU contracts for topology-only reuse of the frozen first-step observer."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_wan_topology_step as t
from test_20261104_external_wan_first_step import complete_fixture


def geometry_oracle():
    return dict(ranks=[dict(boundaries=[
        dict(name='prepare_latents.0.out',tensor=dict(shape=[1,48,31,32,56])),
        dict(name='transformer.0.in.timestep',tensor=dict(shape=[1,13888]))])])


def rows(world):
    all_rows=[]
    for rank in range(world):
        r=complete_fixture();r.update(rank=rank,world_size=world,launcher={'pin':'immutable'},
            reference_summary_sha256='a'*64,artifact_bytes=0,topology_contract={'world_size':world})
        r['runtime'].update(sp=world,gpu_uuid='00000000-0000-0000-0000-'+str(rank+1).zfill(12))
        if rank==0:
            for b in r['boundaries']:b['comparison']=dict(bitwise_equal=True,within_tolerance=True)
        all_rows.append(r)
    return all_rows


class TopologyStepTests(unittest.TestCase):
    def test_original_observer_and_helpers_remain_byte_identical(self):
        identity=t.verify_observer_sources()
        self.assertEqual(identity['observer_sources'],t.PINNED)
        self.assertEqual(identity['observer_commit'],'45685bea933790a5f335f614cd1e96d7e2d6f44f')
        with patch.dict(t.PINNED,{'external_wan_first_step.py':'f'*64}):
            with self.assertRaisesRegex(ValueError,'bytes differ'):t.verify_observer_sources()

    def test_sp2_and_sp8_require_exact_single_node_rank_allocation(self):
        for world in (2,8):
            for rank in range(world):
                env={k:str(v) for k,v in dict(RANK=rank,LOCAL_RANK=rank,WORLD_SIZE=world,LOCAL_WORLD_SIZE=world).items()}
                self.assertEqual(t.topology(world,env),rank)
                env['LOCAL_RANK']=str(world)
                with self.assertRaises(ValueError):t.topology(world,env)
        for world in (1,4,True,0):
            with self.assertRaises(ValueError):t.topology(world,{})

    def test_config_is_original_sp2_and_only_changes_degree_for_sp8(self):
        request={key:1 for key in ('height','width','num_frames','num_inference_steps','guidance_scale','negative_prompt','prompt','seed')}
        identity={'flow_shift':5.}
        original=t.observer.distributed.distributed_config('wan22',request,identity,Path('/shared/output/DITSERVE/test'),2)
        self.assertEqual(t.runner_config(request,identity,Path('/shared/output/DITSERVE/test'),2),original)
        expected=dict(original,ulysses_degree=8)
        self.assertEqual(t.runner_config(request,identity,Path('/shared/output/DITSERVE/test'),8),expected)
        self.assertFalse(expected['use_cfg_parallel']);self.assertFalse(expected['enable_model_cpu_offload'])

    def test_heads_sequence_and_expanded_timestep_support_sp8_without_padding(self):
        config=dict(num_attention_heads=24,patch_size=[1,2,2])
        for world in (2,8):
            contract=t.geometry_contract(config,geometry_oracle(),world)
            self.assertEqual(contract['sequence_tokens'],13888)
            self.assertEqual(contract['tokens_per_rank'],13888//world)
            self.assertEqual(contract['heads_per_rank'],24//world)
            self.assertEqual(contract['expected_padding_tokens'],0)
        for bad in (dict(config,num_attention_heads=23),dict(config,patch_size=[1,1,2])):
            with self.assertRaises(ValueError):t.geometry_contract(bad,geometry_oracle(),8)
        oracle=geometry_oracle();oracle['ranks'][0]['boundaries'][1]['tensor']['shape']=[1,13887]
        with self.assertRaises(ValueError):t.geometry_contract(config,oracle,8)

    def test_all_eighteen_boundaries_and_all_ranks_required_for_both_worlds(self):
        for world in (2,8):
            r=rows(world);summary=t.aggregate(r,r[0]['binding'],world,{'pin':'immutable'},'a'*64)
            self.assertTrue(summary['diagnostic_completed']);self.assertFalse(summary['qualification_eligible'])
            self.assertEqual(summary['completed_requests'],0);self.assertEqual(summary['observed_boundaries_per_rank'],18)
            bad=deepcopy(r);bad[0]['boundaries'][0].pop('comparison')
            with self.assertRaisesRegex(ValueError,'18'):t.aggregate(bad,r[0]['binding'],world,{'pin':'immutable'},'a'*64)
            with self.assertRaises(ValueError):t.aggregate(r[:-1],r[0]['binding'],world,{'pin':'immutable'},'a'*64)

    def test_rank0_metrics_cannot_represent_a_different_rank_payload(self):
        r=rows(8);r[7]['boundaries'][0]['tensor']['sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'differ across ranks'):t.aggregate(r,r[0]['binding'],8,{'pin':'immutable'},'a'*64)

    def test_launcher_reference_and_no_new_arrays_are_bound(self):
        for field,value in (('launcher',{'pin':'changed'}),('reference_summary_sha256','b'*64),('artifact_bytes',1)):
            r=rows(2);r[1][field]=value
            with self.assertRaises(ValueError):t.aggregate(r,r[0]['binding'],2,{'pin':'immutable'},'a'*64)


if __name__=='__main__':unittest.main()
