"""CPU checks of original GF-DiT migration code; no fabricated GPU success."""
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
import external_gfdit_migration as port

SOURCE=Path(__file__).resolve().parents[3]/'.local/ditserve-kairos-rewrite-runtime/external-systems/20261103/GF-DiT'
MANIFEST=SOURCE.with_suffix('.source.json')
MANIFEST_SHA='e1b54cf09241ff70fc27e9d1666a44cbe38e00eaf1713b57cd5c9ade9c9db3f4'


def pure_layout_math(control):
    # These exact official functions contain only integer/layout operations.
    # Execute their AST for CPU validation without inventing a torch module.
    # Production always imports and executes the complete original data_plane.
    names={'tensor_numel','validate_migration_field_pair','intersect_field_slices',
           'select_source_fields_for_dst','select_replicated_source_field','build_layout_migration_schedule'}
    source=(SOURCE/port.MODULES['data_plane.py']).read_text()
    nodes=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(nodes)!=len(names): raise AssertionError('official layout function inventory differs')
    module=ast.parse('from __future__ import annotations\n')
    module.body.extend(nodes)
    namespace=dict(vars(control.protocol))
    exec(compile(ast.fix_missing_locations(module),'<original-pure-layout-math>','exec'),namespace)
    return SimpleNamespace(**{name:namespace[name] for name in names})


def describe(control,codec,command,rank):
    p=control.protocol
    src,dst=[],[]
    for spec in command.plan.artifacts:
        if rank in command.src_group.ranks:
            layout=codec.describe_output(group=command.src_group,group_rank=command.src_group.ranks.index(rank),
                request_metadata=command.plan.request_metadata,artifact=spec.effective_source_handle)
            src.append(p.OutputArtifactLayout(spec.handle,layout.tensors))
        if rank in command.dst_group.ranks:
            dst.append(codec.describe_output(group=command.dst_group,group_rank=command.dst_group.ranks.index(rank),
                request_metadata=command.plan.request_metadata,artifact=spec.handle))
    return control.transport_protocol.MigrateArtifactsRankResult(command.migrate_id,command.plan.request_id,
        rank,command.phase,tuple(src),tuple(dst))


class GFMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_dir(): raise unittest.SkipTest('real pinned official GF-DiT source required')
        cls.directory=TemporaryDirectory()
        cls.root=Path(cls.directory.name)
        cls.derived=port.apply(SOURCE,MANIFEST,MANIFEST_SHA,cls.root/'derived')
        cls.control=port.load(cls.derived['root'],cls.derived['manifest_sha256'])
        cls.math=pure_layout_math(cls.control)
        cls.codec=port.fixture_codec(cls.control)

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name==port.NAMESPACE or name.startswith(port.NAMESPACE+'.'): del sys.modules[name]
        cls.directory.cleanup()

    def command(self,case=None,index=0,phase='build_schedule',mid='unit-migration'):
        case=case or port.scenarios()[0]
        plan=port.plan_for(self.control,case,index)
        groups=port.groups(self.control)
        return self.control.transport_protocol.MigrateArtifactsCommand(mid,phase,plan,
            groups[plan.src_group_id],groups[plan.dst_group_id])

    def test_source_algorithms_only_import_changes_and_real_message_classes(self):
        manifest=self.derived['manifest']
        self.assertEqual(len(manifest['import_only_files']),7)
        for target,row in manifest['import_only_files'].items():
            raw=(SOURCE/row['upstream_path']).read_bytes()
            derived,algorithm=port.relocate(raw)
            self.assertEqual(derived,(Path(self.derived['root'])/port.NAMESPACE/target).read_bytes())
            self.assertEqual(algorithm,row['normalized_algorithm_ast_sha256'])
        with self.assertRaisesRegex(ValueError,'worker implementation'):
            port.relocate(b'from vllm_omni.diffusion.runtime_v2.multiproc_worker import MultiprocWorkerPool\n')
        with self.assertRaisesRegex(ValueError,'execution dependency'):
            port.relocate(b'from vllm.distributed import get_world_group\n')
        self.assertFalse(manifest['qualification_eligible'])
        self.assertFalse(manifest['native_multiproc_worker_reproduced'])

    def test_native_layout_math_covers_all_36_renamed_artifact_transitions(self):
        count=0
        large_chunk=False
        for case in port.scenarios():
            for index in range(len(port.CHAIN)-1):
                command=self.command(case,index)
                replies=tuple(describe(self.control,self.codec,command,rank) for rank in (0,1))
                schedule=self.math.build_layout_migration_schedule(replace(command,layout_results=replies))
                self.assertEqual(schedule.participant_ranks,(0,1))
                self.assertTrue(all(edge.artifact_id=='state-'+str(index+1) for edge in schedule.edges))
                # Replicated [0,1] -> rank1 can reuse the declared fields
                # locally. The original worker adds a separate leader->rank1
                # edge for solver history during BUILD_SCHEDULE on the GPU.
                if index==5:
                    self.assertTrue(all(edge.local_copy for edge in schedule.edges))
                else:
                    self.assertTrue(any(not edge.local_copy for edge in schedule.edges))
                for rank in (0,1):
                    counts=port.edge_counts(schedule,rank)
                    if counts['send_bytes']>port.BUFFER_BYTES and counts['send_calls']>4:
                        large_chunk=True
                count+=1
        self.assertEqual(count,36)
        self.assertTrue(large_chunk)
        expected=port.expected_totals()
        self.assertEqual(expected['phase_executions'],252)
        self.assertEqual(expected['byte_checks'],270)
        self.assertEqual(expected['per_rank'][0],dict(phase_executions=132,destination_states=24,byte_checks=120))
        self.assertEqual(expected['per_rank'][1],dict(phase_executions=120,destination_states=30,byte_checks=150))

    def test_layout_dtype_and_missing_source_fail_before_payload(self):
        command=self.command()
        replies=[describe(self.control,self.codec,command,rank) for rank in (0,1)]
        with self.assertRaisesRegex(RuntimeError,'source layouts missing'):
            self.math.build_layout_migration_schedule(replace(command,layout_results=(replies[1],)))
        layout=replies[1].dst_layouts[0]
        bad=replace(layout,tensors=(replace(layout.tensors[0],dtype='torch.float32'),*layout.tensors[1:]))
        replies[1]=replace(replies[1],dst_layouts=(bad,))
        with self.assertRaisesRegex(RuntimeError,'dtype mismatch'):
            self.math.build_layout_migration_schedule(replace(command,layout_results=tuple(replies)))

    def engine(self):
        queue,done,failed=[],[],[]
        engine=self.control.migration_engine.MigrationEngine(
            build_command=lambda job,phase:port.build_command(self.control,job,phase),
            send=lambda command,ranks:queue.append((command,ranks)))
        def begin(mid):
            command=self.command(mid=mid)
            engine.begin(plan=command.plan,src_group=command.src_group,dst_group=command.dst_group,
                participant_ranks=(0,1),src_leader_rank=0,on_done=lambda schedule:done.append(mid),
                on_error=lambda error:failed.append((mid,str(error))),deadline=10.,migrate_id=mid)
        return engine,queue,done,failed,begin

    def replies(self,command,targets):
        if command.phase=='describe_layout':
            return [describe(self.control,self.codec,command,rank) for rank in targets]
        schedule=self.math.build_layout_migration_schedule(command) if command.phase=='build_schedule' else None
        return [self.control.transport_protocol.MigrateArtifactsRankResult(command.migrate_id,
            command.plan.request_id,rank,command.phase,schedule=schedule) for rank in targets]

    def test_original_engine_four_phases_and_fifo_lifecycle(self):
        engine,queue,done,failed,begin=self.engine()
        begin('first'); begin('second')
        engine.pump([],0.)
        trace=[]
        while queue:
            command,targets=queue.pop(0)
            trace.append((command.migrate_id,command.phase,targets))
            engine.pump(self.replies(command,targets),1.)
        self.assertEqual(done,['first','second'])
        self.assertFalse(failed)
        self.assertEqual([(mid,phase) for mid,phase,_ in trace],
                         [(mid,phase) for mid in ('first','second') for phase in port.PHASES])
        self.assertFalse(engine.has_pending())
        self.assertFalse(engine._jobs)
        self.assertTrue(all(not queue for queue in engine._rank_queues.values()))

    def test_timeout_does_not_release_ranks_inside_actual_transfer_phase(self):
        engine,queue,done,failed,begin=self.engine()
        begin('held'); begin('next')
        engine.pump([],0.)
        while queue[0][0].phase!='execute_transfer':
            command,targets=queue.pop(0)
            engine.pump(self.replies(command,targets),1.)
        transfer,targets=queue.pop(0)
        engine.pump([],20.)
        # A queued handshake may time out, but the active transfer must retain
        # its ranks until both real worker results arrive.
        self.assertTrue(engine.has_pending())
        self.assertIn('held',engine._jobs)
        self.assertFalse(done)
        self.assertNotIn('held',[mid for mid,_ in failed])
        engine.pump(self.replies(transfer,targets),21.)
        self.assertEqual(done,['held'])
        self.assertFalse(engine.has_pending())

    def test_native_handshake_error_and_stale_phase_fail_once(self):
        for mode in ('timeout','worker_error','stale'):
            engine,queue,done,failed,begin=self.engine()
            begin('broken'); engine.pump([],0.)
            command,targets=queue.pop(0)
            if mode=='timeout': engine.pump([],20.)
            else:
                result=self.replies(command,targets)[0]
                result=replace(result,error='explicit unit failure') if mode=='worker_error' else replace(result,phase='accept_schedule')
                engine.pump([result],1.)
                engine.pump([result],2.)
            self.assertFalse(done)
            self.assertEqual(len(failed),1)
            self.assertFalse(engine.has_pending())
            self.assertTrue(all(not q for q in engine._rank_queues.values()))

    def test_full_rank_denominator_and_qualification_gate(self):
        expected=port.expected_totals()
        records=[dict(rank=rank,status='passed',cleanup_verified=True,gpu_uuid='unit-'+str(rank),
            source_binding_sha256='unit-source-binding',migrations=36,
            phase_trace=port.expected_phase_trace() if rank==0 else [],
            p2p=dict(send_calls=1,recv_calls=1,send_bytes=1,recv_bytes=1),
            expected_p2p=dict(send_calls=1,recv_calls=1,send_bytes=1,recv_bytes=1),
            **expected['per_rank'][rank]) for rank in (0,1)]
        result=port.assess(records)
        self.assertFalse(result['qualification_eligible'])
        self.assertFalse(result['model_executed'])
        for mode in ('missing','failed','cleanup','uuid','source','bytes','phase','trace','migration','fallback'):
            bad=deepcopy(records)
            if mode=='missing': bad.pop()
            elif mode=='failed': bad[1]['status']='failed'
            elif mode=='cleanup': bad[1]['cleanup_verified']=False
            elif mode=='uuid': bad[1]['gpu_uuid']=bad[0]['gpu_uuid']
            elif mode=='source': bad[1]['source_binding_sha256']='different'
            elif mode=='bytes': bad[1]['byte_checks']-=1
            elif mode=='phase': bad[1]['phase_executions']-=1
            elif mode=='trace': bad[0]['phase_trace'].pop()
            elif mode=='migration': bad[1]['migrations']-=1
            elif mode=='fallback': bad[1]['p2p']['recv_calls']=0
            with self.assertRaises(ValueError,msg=mode): port.assess(bad)

    def test_derived_extra_file_mutation_and_existing_output_are_rejected(self):
        root=Path(self.derived['root'])
        with self.assertRaises(FileExistsError): port.apply(SOURCE,MANIFEST,MANIFEST_SHA,root)
        with self.assertRaisesRegex(ValueError,'disjoint'): port.apply(SOURCE,MANIFEST,MANIFEST_SHA,SOURCE/'no-write')
        with self.assertRaisesRegex(ValueError,'manifest SHA'): port.load(root,'0'*64)
        path=root/'extra.so'
        try:
            path.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'inventory'): port.load(root,self.derived['manifest_sha256'])
        finally: path.unlink()


if __name__=='__main__': unittest.main()
