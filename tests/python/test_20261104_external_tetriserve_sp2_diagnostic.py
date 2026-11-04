"""CPU evidence gates; actual SP2 inference is exclusively a two-GPU job."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_tetriserve_sp2_diagnostic as diagnostic


def dispatches():
    rows=[]; start=0
    for end in diagnostic.EXPECTED_ENDS:
        rows.append(dict(request_ids=[1],start=start,end=end,height=1024,width=1024,
                         batch_size=1,ranks=[0] if start<10 else [0,1]))
        start=end
    return rows


def receipt():
    tensors={k:dict(shape=shape,dtype='torch.bfloat16',finite=True,sha256='a'*64,
              minimum=-1.,maximum=1.,retained_array=False) for k,shape in
              diagnostic.numerics.raw_shapes(dict(height=1024,width=1024)).items()}
    outputs={name:deepcopy(tensors) for name in diagnostic.TRAJECTORIES}
    comparisons={name:{k:dict(elements=math.prod(v['shape']),atol=1e-5,rtol=1e-5,
        passed=True,bitwise_equal=True,max_abs_error=0.,rmse=0.,relative_l2_error=0.,cosine_similarity=1.)
        for k,v in tensors.items()} for name in diagnostic.PAIRS}
    rows=[dict(rank=rank,status='completed',gpu_uuid='GPU-'+str(rank),distributed_closed=True,
        groups=[0,1],actual_sp=2,actual_dp=1,chunk_ends=list(diagnostic.EXPECTED_ENDS),
        transformer_calls={n:28 for n in diagnostic.TRAJECTORIES},outputs=deepcopy(outputs),
        initial_five_tensors={n:dict(shape=shape,sha256='f'*64,finite=True,minimum=0.,maximum=1.,
            dtype='torch.float32' if n in ('timesteps','sigmas') else 'torch.bfloat16') for n,shape in
            dict(prompt_embeds=[2,333,4096],pooled_prompt_embeds=[2,2048],latents=[1,16,128,128],timesteps=[28],sigmas=[29]).items()}) for rank in (0,1)]
    rows[0].update(comparisons=comparisons,comparison_inputs={n:deepcopy(tensors) for n in ('sp1_reference','service')})
    return rows


class SP2DiagnosticTests(unittest.TestCase):
    def test_actual_r2_order_and_chunk_boundaries(self):
        self.assertEqual(diagnostic.intervals_from_dispatches(dispatches(),1),diagnostic.EXPECTED_ENDS)
        for mutation in ('gap','overlap','missing','extra','rank','shape'):
            rows=dispatches()
            if mutation=='gap':rows[4]['start']=11
            if mutation=='overlap':rows[4]['start']=9
            if mutation=='missing':rows.pop()
            if mutation=='extra':rows.append(deepcopy(rows[-1]))
            if mutation=='rank':rows[4]['ranks']=[0]
            if mutation=='shape':rows[0]['height']=512
            with self.assertRaises(ValueError,msg=mutation):diagnostic.intervals_from_dispatches(rows,1)

    def test_completion_requires_every_rank_trajectory_and_full_28_steps(self):
        self.assertTrue(diagnostic.assess(receipt()))
        for mutation in ('missingrank','duplicate','gpu','steps','trajectory','cleanup','outputs','initial','pair','elements','cfg','schedule','actualsp','actualdp'):
            rows=receipt()
            if mutation=='missingrank':rows.pop()
            if mutation=='duplicate':rows[1]['rank']=0
            if mutation=='gpu':rows[1]['gpu_uuid']=rows[0]['gpu_uuid']
            if mutation=='steps':rows[1]['transformer_calls']['normal_sp2']=27
            if mutation=='trajectory':del rows[1]['outputs']['round_sp2_full']
            if mutation=='cleanup':rows[1]['distributed_closed']=False
            if mutation=='outputs':rows[1]['outputs']['normal_sp2']['latents']['sha256']='b'*64
            if mutation=='initial':rows[1]['initial_five_tensors']['latents']['sha256']='b'*64
            if mutation=='actualsp':rows[0]['actual_sp']=1
            if mutation=='actualdp':rows[0]['actual_dp']=2
            if mutation=='cfg':rows[0]['initial_five_tensors']['prompt_embeds']['shape'][0]=1
            if mutation=='schedule':rows[0]['initial_five_tensors']['sigmas']['shape']=[28]
            if mutation=='pair':rows[0]['comparisons'].pop(diagnostic.PAIRS[-1])
            if mutation=='elements':rows[0]['comparisons'][diagnostic.PAIRS[0]]['latents']['elements']=1
            self.assertFalse(diagnostic.assess(rows),mutation)

    def test_diagnostic_can_complete_with_real_disagreement_without_qualification(self):
        rows=receipt()
        rows[0]['comparison_inputs']['sp1_reference']['latents']['sha256']='b'*64
        for name in diagnostic.PAIRS:
            if name.startswith('sp1_reference_'):
                rows[0]['comparisons'][name]['latents'].update(passed=False,bitwise_equal=False,
                    max_abs_error=2.89453125,rmse=.21898,relative_l2_error=.19916,cosine_similarity=.9802)
        self.assertTrue(diagnostic.assess(rows))
        self.assertFalse(all(c['passed'] for pair in rows[0]['comparisons'].values() for c in pair.values()))

    def test_false_pass_or_corrupt_numerical_receipts_are_rejected(self):
        for mutation in ('falsepass','nan','inf','rmse','tol','dtype','shape','digest','extrema','bitwise'):
            rows=receipt(); c=rows[0]['comparisons'][diagnostic.PAIRS[0]]['latents']
            if mutation=='falsepass':c.update(max_abs_error=1.,rmse=.1)
            if mutation=='nan':c['rmse']=float('nan')
            if mutation=='inf':c['relative_l2_error']=float('inf')
            if mutation=='rmse':c['rmse']=1.
            if mutation=='tol':c['atol']=1.
            if mutation=='dtype':rows[0]['comparison_inputs']['service']['decoded']['dtype']='torch.float32'
            if mutation=='shape':rows[0]['comparison_inputs']['service']['decoded']['shape']=[1]
            if mutation=='digest':rows[0]['comparison_inputs']['service']['decoded']['sha256']='z'*64
            if mutation=='extrema':rows[0]['comparison_inputs']['service']['decoded']['maximum']=float('inf')
            if mutation=='bitwise':c['bitwise_equal']=False
            self.assertFalse(diagnostic.assess(rows),mutation)

    def test_changed_prior_summary_is_rejected_before_loading_or_writing(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'numerics').mkdir()
            (root/'summary.json').write_text('{}')
            (root/'numerics/summary.json').write_text('{}')
            before=sorted(p.relative_to(root).as_posix() for p in root.rglob('*'))
            with self.assertRaisesRegex(ValueError,'summaries differ'):
                diagnostic.bind_prior(root,'0'*64,'0'*64)
            self.assertEqual(before,sorted(p.relative_to(root).as_posix() for p in root.rglob('*')))


if __name__=='__main__':unittest.main()
