"""Small CPU arrays only: full-payload, fixed-tolerance and ownership contracts."""
from copy import deepcopy
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'benchmarks'))
import external_xdit_numerics as numerics
import external_xdit_native as native
import external_xdit_distributed as distributed
from test_20261103_external_xdit_distributed import receipts


def request(frames=2, height=2, width=3):
    return dict(num_frames=frames, height=height, width=width, prompt='CPU fixture, not GPU evidence.',
                seed=42, num_inference_steps=50, guidance_scale=5., negative_prompt='', num_outputs=1)


def contract(root, req):
    return dict(schema='ditserve.xdit-decoded-contract.v1', space=numerics.SPACE,
        atol=1e-5, rtol=1e-5, max_total_bytes=2*1024**3, member_budget_bytes=768*1024**2,
        image_digest=numerics.IMAGE, build='51740c0', request=req,
        members={role:dict(run_id='fixture-'+role, world_size=world,
                           output_dir=str(root/'external-tasks'/('fixture-'+role)/'result'))
                 for role,world in [('sp1',1),('sp4',4)]},
        protocol_sha256='a'*64, official_source_manifest_sha256='b'*64, model_manifest_sha256='c'*64,
        code_sha256={name:numerics.sha_bytes(Path(numerics.__file__).with_name(name).read_bytes())
                     for name in numerics.CODE_FILES})


class NumericsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve(); self.output = self.root/'output'; self.packages=self.root/'packages'
        self.output.mkdir(); self.packages.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(numerics,'OUTPUT_ROOT',self.output).start()
        patch.object(numerics,'PACKAGE_ROOT',self.packages).start()
        self.guard=patch.object(numerics,'check_storage',return_value={'used_gib':1,'percent':1}).start()

    def fixture(self, role='sp1', req=None):
        req=req or request(); con=contract(self.output,req); package=self.packages/role;package.mkdir()
        cp=package/'numerics-contract.json';cp.write_bytes(numerics.json_bytes(con));cs=numerics.sha_bytes(cp.read_bytes())
        manifest=dict(image_digest=numerics.IMAGE,files={**con['code_sha256'],cp.name:cs})
        mp=package/'package-manifest.json';mp.write_bytes(numerics.json_bytes(manifest));ms=numerics.sha_bytes(mp.read_bytes())
        env=dict(DITSERVE_RUN_ID=con['members'][role]['run_id'],DITSERVE_BUILD='51740c0',
            DITSERVE_EXTERNAL_IMAGE_DIGEST=numerics.IMAGE,DITSERVE_EXTERNAL_PROBE_ROOT=str(package),
            DITSERVE_EXTERNAL_PROBE_MANIFEST_SHA256=ms)
        out=Path(con['members'][role]['output_dir']);out.mkdir(parents=True)
        with patch.dict(os.environ,env):
            ctx=numerics.prepare(cp,cs,out,req,{'manifest_sha256':'c'*64},{'manifest_sha256':'b'*64},'a'*64,
                                 con['members'][role]['world_size'],0)
        numerics.begin(ctx)
        data=np.arange(np.prod(numerics.dimensions(req)[0]),dtype=np.float32).reshape(numerics.dimensions(req)[0])/100
        summary=native.summarize_video([data],req)
        return ctx,data,summary

    def test_full_payload_and_native_prefix_sha_are_independently_verified(self):
        ctx,data,summary=self.fixture(); saved=numerics.capture([data],summary,ctx)
        raw=Path(ctx['output_dir'])/'decoded-numerics/decoded.float32.raw'
        self.assertEqual(raw.read_bytes(),data.tobytes())
        self.assertEqual(saved['file_sha256'],hashlib.sha256(data.tobytes()).hexdigest())
        self.assertEqual(saved['native_output_sha256'],summary['sha256'])
        self.assertNotEqual(saved['file_sha256'],summary['sha256'])
        self.assertEqual(saved['payload_bytes'],data.nbytes)
        self.assertTrue((raw.parent/'DEPENDENCY-PROTECTED.json').exists())
        self.assertGreaterEqual(self.guard.call_count,len(data)+4)
        with self.assertRaises(FileExistsError):numerics.capture([data],summary,ctx)
        self.assertEqual(raw.read_bytes(),data.tobytes())

    def test_declared_large_geometry_budget_and_contract_tampering(self):
        self.assertEqual(numerics.dimensions(request(121,512,896))[1],666107904)
        self.assertLess(2*666107904+2*1024**2,numerics.TOTAL_BUDGET)
        for req in [request(True,2,3), request(121,1024,1024),request(0,2,3)]:
            with self.assertRaises(ValueError):numerics.dimensions(req)
        valid=contract(self.output,request());numerics.validate_contract(valid)
        for field,value in [('atol',1e-4),('rtol',0.),('max_total_bytes',4*1024**3),('member_budget_bytes',True)]:
            bad=deepcopy(valid);bad[field]=value
            with self.assertRaises(ValueError):numerics.validate_contract(bad)
        bad=deepcopy(valid);bad['code_sha256']['external_xdit_native.py']='0'*64
        with self.assertRaisesRegex(ValueError,'adapter source'):numerics.validate_contract(bad)

    def test_paths_cannot_escape_follow_symlink_or_reuse_job_output(self):
        for path in [self.root/'elsewhere',self.output/'..'/'escape',self.output,Path('relative')]:
            with self.assertRaises(ValueError):numerics.safe_path(path,self.output)
        (self.output/'alias').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):numerics.safe_path(self.output/'alias'/'artifact',self.output)
        ctx,_,_=self.fixture()
        with self.assertRaises(FileExistsError):numerics.begin(ctx)

    def test_nonowner_cannot_save_duplicate_and_rank_gate_closes_denominator(self):
        ctx,data,summary=self.fixture('sp4');other=dict(ctx,rank=1)
        numerics.begin(other)
        with self.assertRaisesRegex(ValueError,'owner'):numerics.capture([data],summary,other)
        rows=receipts(4)
        req=rows[0]['request']
        for row in rows:
            row['decoded_capture']=dict(enabled=True,contract_sha256='e'*64,role='sp4',world_size=4,
                owner_rank=0,artifact_expected_on_this_rank=row['rank']==0)
            row['performance_eligible']=False;row['output']['retained_arrays']=row['rank']==0
        rows[0]['decoded_artifact']=dict(owner_rank=0,contract_sha256='e'*64,
            native_output_sha256=rows[0]['output']['sha256'],payload_bytes=numerics.dimensions(req)[1])
        self.assertEqual(distributed.aggregate(rows,4)['completed_requests'],1)
        for mutate in [lambda r:r[1].update(decoded_artifact=r[0]['decoded_artifact']),
                       lambda r:r[2].pop('decoded_capture'),lambda r:r[3]['decoded_capture'].update(owner_rank=3),
                       lambda r:r[0]['output'].update(retained_arrays=False),lambda r:r[0]['decoded_artifact'].update(payload_bytes=1)]:
            bad=deepcopy(rows);mutate(bad)
            with self.assertRaises(ValueError):distributed.aggregate(bad,4)

    def test_nonfinite_wrong_dtype_shape_and_digest_leave_protected_failure(self):
        for mode in ['finite','dtype','shape','sha']:
            with self.subTest(mode=mode):
                # Unique fixture package/run roots per scenario.
                root=self.output;self.output=root/mode;self.output.mkdir()
                patch.object(numerics,'OUTPUT_ROOT',self.output).start()
                self.packages=self.packages/mode;self.packages.mkdir();patch.object(numerics,'PACKAGE_ROOT',self.packages).start()
                ctx,data,summary=self.fixture()
                if mode=='finite':data[1,0,0,0]=np.nan
                elif mode=='dtype':data=data.astype(np.float16)
                elif mode=='shape':data=data[:,:,:2,:]
                else:summary['sha256']='f'*64
                with self.assertRaises(ValueError):numerics.capture([data],summary,ctx)
                directory=Path(ctx['output_dir'])/'decoded-numerics'
                self.assertTrue((directory/'DEPENDENCY-PROTECTED.json').exists())
                self.assertTrue((directory/'decoded.float32.raw.partial').exists())
                self.assertFalse((directory/'artifact.json').exists())

    def test_storage_gate_during_write_keeps_partial_and_never_publishes_complete(self):
        ctx,data,summary=self.fixture()
        with patch.object(numerics,'check_storage',side_effect=[{}, {}, ValueError('hard stop')]):
            with self.assertRaisesRegex(ValueError,'hard stop'):numerics.capture([data],summary,ctx)
        directory=Path(ctx['output_dir'])/'decoded-numerics'
        self.assertEqual((directory/'decoded.float32.raw.partial').stat().st_size,data[0].nbytes)
        self.assertFalse((directory/'artifact.json').exists())

    def raw_pair(self,a,b):
        paths=[self.root/'a.raw',self.root/'b.raw'];shape=list(a.shape)
        prefix=json.dumps({'shape':shape,'dtype':'float32'},sort_keys=True).encode();expected=[]
        for path,array in zip(paths,[a,b]):
            data=array.tobytes();path.write_bytes(data)
            expected.append(dict(file_sha256=hashlib.sha256(data).hexdigest(),
                native_output_sha256=hashlib.sha256(prefix+data).hexdigest()))
        return paths,shape,expected

    def test_full_fixed_reference_elementwise_tolerance_and_fp64_metrics(self):
        a=np.array([0.,1.,10000.,-1.,0.,1.],dtype=np.float32).reshape(1,1,2,3)
        b=a.copy();b.flat[0]=2e-5 # A global refmax gate would incorrectly allow this.
        paths,shape,expected=self.raw_pair(a,b)
        result=numerics.compare_files(*paths,shape,expected)
        errors=np.abs(a.astype(np.float64)-b.astype(np.float64))
        self.assertEqual(result['total_elements'],6);self.assertEqual(result['failed_elements'],1)
        self.assertFalse(result['strict_decoded_consistency_passed'])
        self.assertEqual(result['max_abs'],float(errors.max()))
        self.assertEqual(result['mean_abs'],float(errors.mean()))
        self.assertEqual(result['rmse'],float(np.sqrt(np.mean(errors**2))))
        paths,shape,expected=self.raw_pair(a,a)
        self.assertTrue(numerics.compare_files(*paths,shape,expected)['strict_decoded_consistency_passed'])

    def test_full_file_and_native_hash_length_finite_and_mutation_checks(self):
        a=np.zeros((2,1,1,3),dtype=np.float32)
        paths,shape,expected=self.raw_pair(a,a)
        for key in ['file_sha256','native_output_sha256']:
            bad=deepcopy(expected);bad[1][key]='f'*64
            with self.assertRaisesRegex(ValueError,'SHA256'):numerics.compare_files(*paths,shape,bad)
        paths[1].write_bytes(a.tobytes()[:-1])
        with self.assertRaisesRegex(ValueError,'byte count'):numerics.compare_files(*paths,shape,expected)
        nan=a.copy();nan.flat[0]=np.inf;paths,shape,expected=self.raw_pair(a,nan)
        with self.assertRaisesRegex(ValueError,'nonfinite'):numerics.compare_files(*paths,shape,expected)
        paths,shape,expected=self.raw_pair(a,a)
        def touch():
            stamp=paths[0].stat().st_mtime_ns+1_000_000
            os.utime(paths[0],ns=(stamp,stamp));return {}
        with patch.object(numerics,'check_storage',side_effect=touch):
            with self.assertRaisesRegex(ValueError,'changed during'):numerics.compare_files(*paths,shape,expected)

    def test_source_and_launch_identity_are_required_before_capture(self):
        ctx,_,_=self.fixture();con=ctx['contract'];cp=self.packages/'sp1/numerics-contract.json'
        args=[cp,ctx['contract_sha256'],Path(ctx['output_dir']),ctx['request'],ctx['model_identity'],ctx['source'],ctx['protocol_sha256'],1,0]
        with self.assertRaisesRegex(ValueError,'launch identity'):numerics.prepare(*args)
        bad=deepcopy(args);bad[4]['manifest_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'preregistered inputs'):numerics.prepare(*bad)

    def completed_fixture(self, role):
        ctx,data,summary=self.fixture(role)
        saved=numerics.capture([data],summary,ctx)
        shape=summary['shape'];world=ctx['world_size']
        if role=='sp4':
            rows=receipts(4)
            for row in rows:
                row.update(request=ctx['request'],effective_request=ctx['request'],model_identity=ctx['model_identity'],
                    source=ctx['source'],output=dict(summary,retained_arrays=row['rank']==0),
                    decoded_capture=numerics.capture_receipt(dict(ctx,rank=row['rank'])),performance_eligible=False)
            rows[0]['decoded_artifact']=saved
            child=distributed.aggregate(rows,4)
        else:
            child=dict(status='native_request_completed',completed_requests=1,requested_requests=1,
                request=ctx['request'],model_identity=ctx['model_identity'],source=ctx['source'],
                output=dict(summary,retained_arrays=True),decoded_capture=numerics.capture_receipt(ctx),
                decoded_artifact=saved)
        output=Path(ctx['output_dir']); raw=numerics.json_bytes(child);(output/'summary.json').write_bytes(raw)
        outer=dict(run_id=ctx['run_id'],binding=dict(manifest_sha256=ctx['package_manifest_sha256'],
            image='192.168.1.114:30500/jjjkairos/ditserve@'+numerics.IMAGE),
            execution=dict(returncode=0,timed_out=False,child_summary=child,child_summary_sha256=numerics.sha_bytes(raw)))
        (output.parent/'summary.json').write_bytes(numerics.json_bytes(outer))
        return ctx,saved

    def test_real_capture_receipt_chain_and_complete_comparison_cli(self):
        contexts=[self.completed_fixture(role) for role in ['sp1','sp4']]
        c=contexts[0][0];self.assertEqual(c['contract_sha256'],contexts[1][0]['contract_sha256'])
        for role,(ctx,saved) in zip(['sp1','sp4'],contexts):
            artifact,path=numerics.verify_artifact(c['contract'],c['contract_sha256'],role,saved['manifest_sha256'])
            self.assertEqual(artifact['file_sha256'],saved['file_sha256'])
            self.assertEqual(path.stat().st_size,144)
        output=self.output/'comparison'
        with contextlib.redirect_stdout(io.StringIO()):
            status=numerics.main(['--contract',str(self.packages/'sp1/numerics-contract.json'),
                '--contract-sha256',c['contract_sha256'],'--sp1-manifest-sha256',contexts[0][1]['manifest_sha256'],
                '--sp4-manifest-sha256',contexts[1][1]['manifest_sha256'],'--output-dir',str(output)])
        self.assertEqual(status,0)
        report=json.loads((output/'summary.json').read_bytes())
        self.assertTrue(report['strict_decoded_consistency_passed']);self.assertEqual(report['total_elements'],36)
        self.assertEqual(report['failed_elements'],0);self.assertFalse(report['semantic_quality_evaluated'])
        self.assertEqual(report['shape'],[2,2,3,3]);self.assertEqual(report['payload_bytes'],288)
        self.assertFalse(report['qualification_eligible']);self.assertEqual(len(report['input_receipts']),2)

    def test_execution_and_protection_pollution_is_rejected_after_recapture(self):
        ctx,saved=self.completed_fixture('sp1');out=Path(ctx['output_dir'])
        pp=out/'decoded-numerics/DEPENDENCY-PROTECTED.json';original=pp.read_bytes()
        for field,value in [('cleanup_allowed',True),('automatic_cleanup',True),('run_id','foreign'),('expected_payload_bytes',1)]:
            changed=json.loads(original);changed[field]=value;pp.write_bytes(numerics.json_bytes(changed))
            with self.assertRaisesRegex(ValueError,'protection'):
                numerics.verify_artifact(ctx['contract'],ctx['contract_sha256'],'sp1',saved['manifest_sha256'])
        pp.write_bytes(original)
        outerpath=out.parent/'summary.json';outer=json.loads(outerpath.read_bytes());outer['execution']['returncode']=1
        outerpath.write_bytes(numerics.json_bytes(outer))
        with self.assertRaisesRegex(ValueError,'execution binding'):
            numerics.verify_artifact(ctx['contract'],ctx['contract_sha256'],'sp1',saved['manifest_sha256'])

    def test_updated_storage_hard_line_is_800_gib_or_80_percent(self):
        for used,pct,ok in [(800,79,True),(801,79,False),(779,80,False)]:
            result=type('Result',(),{'returncode':0,'stdout':f'header\n/dev/sdb 1024G {used}G 200G {pct}% /shared\n'})()
            with patch.object(native.subprocess,'run',return_value=result):
                if ok:self.assertEqual(native.check_storage()['used_gib'],used)
                else:
                    with self.assertRaises(ValueError):native.check_storage()


if __name__=='__main__':unittest.main()
