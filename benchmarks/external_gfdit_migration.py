"""Real two-GPU GF-DiT migration diagnostic using its original data plane.

The original migration engine, layout math, codec and collective runtime remain
unchanged except for imports/logging. A small actual distributed controller
replaces the native vLLM multiprocessing transport. CUDA tensor payloads use
GFC exclusively; NCCL world operations carry bounded control messages only.
Synthetic state and solver-history tensors are not model inference or a serving
benchmark. Every report remains ineligible for paper/performance conclusions.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import timedelta
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import sys
import threading
import time
from types import SimpleNamespace

import external_gfdit as upstream
import external_gfc_smoke as gfc_check

sys.dont_write_bytecode = True
NAMESPACE = 'gfdit_migration_runtime'
PREFIX = 'vllm_omni.diffusion.runtime_v2'
SOURCE_PREFIX = PREFIX.replace('.', '/')
SCHEMA = 'ditserve.gfdit-migration-source.v1'
MODULES = {name: SOURCE_PREFIX + '/' + name for name in (
    '_env.py', 'protocol.py', 'codec.py', 'adapters/common.py', 'data_plane.py', 'migration_engine.py')}
MODULES['collective_runtime.py'] = 'vllm_omni/diffusion/distributed/collective_runtime.py'
PHASE_NAMES = ('_MIGRATE_PHASE_DESCRIBE_LAYOUT', '_MIGRATE_PHASE_BUILD_SCHEDULE',
               '_MIGRATE_PHASE_ACCEPT_SCHEDULE', '_MIGRATE_PHASE_EXECUTE_TRANSFER')
PHASES = ('describe_layout', 'build_schedule', 'accept_schedule', 'execute_transfer')
MESSAGE_CLASSES = ('MigrateArtifactsCommand', 'MigrateArtifactsRankResult')
BUFFER_BYTES = 65536
FIELDS = ('latents', 'conditioning', 'timesteps', 'sigmas', 'history')
GROUPS = (
    ('one0', (0,), 'replicated'), ('one1', (1,), 'replicated'),
    ('split01', (0, 1), 'sharded'), ('split10', (1, 0), 'sharded'),
    ('copy01', (0, 1), 'replicated'),
)
CHAIN = ('one0', 'one1', 'split01', 'split10', 'one0', 'copy01', 'one1')
CONTEXT_SOURCE = '''"""Actual logical-group binding for the original migration helpers."""
from types import SimpleNamespace
from .collective_runtime import make_logical_group, runtime_v2_uses_gfc
_active = None

def activate(group, rank):
    global _active
    if not runtime_v2_uses_gfc():
        raise RuntimeError('this diagnostic requires the actual GFC backend')
    if rank not in group.ranks or group.parallel_spec.sp != len(group.ranks):
        raise ValueError('logical group/rank mismatch')
    if group.parallel_spec.tp != 1 or group.parallel_spec.cfg != 1:
        raise ValueError('migration diagnostic supports SP only')
    handle = make_logical_group(group.ranks, group_id=group.group_id, register_gfc=True)
    # data_plane expects a coordinator exposing .device_group. Bind that to
    # the real original LogicalGroupHandle, never an invented ProcessGroup.
    _active = SimpleNamespace(world_size=handle.world_size, rank_in_group=handle.rank_in_group,
                              device_group=handle, logical_handle=handle)
    return _active

def get_sp_group():
    if _active is None:
        raise RuntimeError('no actual logical GFC group has been activated')
    return _active
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


class ImportNormalizer(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module == 'vllm.logger':
            if [(n.name, n.asname) for n in node.names] != [('init_logger', None)]:
                raise ValueError('unrecognized logging import')
            node.module, node.names = 'logging', [ast.alias(name='getLogger', asname='init_logger')]
        elif node.module == 'vllm_omni.diffusion.distributed.collective_runtime':
            node.module = NAMESPACE + '.collective_runtime'
        elif node.module == 'vllm_omni.diffusion.distributed':
            if [(n.name, n.asname) for n in node.names] != [('parallel_state', 'omni_parallel_state')]:
                raise ValueError('unrecognized distributed execution import')
            node.module, node.names = NAMESPACE, [ast.alias(name='group_context', asname='omni_parallel_state')]
        elif node.module and node.module.startswith(PREFIX + '.'):
            node.module = NAMESPACE + node.module[len(PREFIX):]
            if node.module == NAMESPACE + '.multiproc_worker':
                if any(item.name not in MESSAGE_CLASSES for item in node.names):
                    raise ValueError('native worker implementation is not a transport message')
                node.module = NAMESPACE + '.transport_protocol'
            if node.module == NAMESPACE + '.data_plane' and {n.name for n in node.names} == set(PHASE_NAMES):
                node.module = NAMESPACE + '.migration_phases'
        return node


def relocate(raw):
    tree = ImportNormalizer().visit(ast.parse(raw))
    # Unparse only after checking that the only transformations are imports.
    for node in ast.walk(tree):
        names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                 [n.name for n in node.names] if isinstance(node, ast.Import) else [])
        if any(name == 'vllm' or name.startswith(('vllm.', 'vllm_omni.')) for name in names):
            raise ValueError('unported vLLM execution dependency')
    content = ('# SPDX-License-Identifier: Apache-2.0\n' + ast.unparse(tree) + '\n').encode()
    algorithm = ast.dump(tree, include_attributes=False)
    if ast.dump(ast.parse(content), include_attributes=False) != algorithm:
        raise ValueError('AST changed beyond the explicit import transform')
    return content, sha(algorithm.encode())


def extract_nodes(raw, names, kind):
    text = raw.decode()
    selected = []
    for name in names:
        found = [node for node in ast.parse(text).body if isinstance(node, kind) and
                 ((getattr(node, 'name', None) == name) if kind is ast.ClassDef else
                  any(isinstance(target, ast.Name) and target.id == name for target in node.targets))]
        if len(found) != 1:
            raise ValueError('official declaration missing or ambiguous: ' + name)
        node = found[0]
        start = min([node.lineno] + [item.lineno for item in getattr(node, 'decorator_list', [])])
        fragment = '\n'.join(text.splitlines()[start-1:node.end_lineno]) + '\n'
        if ast.dump(ast.parse(fragment).body[0], include_attributes=False) != ast.dump(node, include_attributes=False):
            raise ValueError('official declaration extraction changed its AST')
        selected.append(fragment)
    return '\n'.join(selected).encode()


def source_plan(source_root, manifest_path, manifest_sha256):
    manifest = upstream.verify_source(source_root, manifest_path, manifest_sha256)
    root = Path(source_root).resolve(strict=True)
    files, changes = {}, {}
    for target, origin in MODULES.items():
        raw = (root / origin).read_bytes()
        content, algorithm_sha = relocate(raw)
        files[NAMESPACE + '/' + target] = content
        changes[target] = dict(upstream_path=origin, upstream_sha256=sha(raw),
            derived_sha256=sha(content), normalized_algorithm_ast_sha256=algorithm_sha)
    worker_path = SOURCE_PREFIX + '/multiproc_worker.py'
    messages = extract_nodes((root / worker_path).read_bytes(), MESSAGE_CLASSES, ast.ClassDef)
    header = ('# SPDX-License-Identifier: Apache-2.0\nfrom __future__ import annotations\n'
              'from dataclasses import dataclass\nfrom .protocol import MigrateArtifactsPlan, '
              'ExecutionGroupSpec, ArtifactMigrationSchedule, OutputArtifactLayout\n\n').encode()
    files[NAMESPACE + '/transport_protocol.py'] = header + messages
    files[NAMESPACE + '/migration_phases.py'] = extract_nodes(
        (root / MODULES['data_plane.py']).read_bytes(), PHASE_NAMES, ast.Assign)
    files[NAMESPACE + '/group_context.py'] = CONTEXT_SOURCE.encode()
    for name in ('__init__.py', 'adapters/__init__.py'):
        files[NAMESPACE + '/' + name] = b'# GF-DiT migration source port; not a vLLM module.\n'
    files['UPSTREAM_LICENSE'] = (root / 'LICENSE').read_bytes()
    record = dict(schema=SCHEMA, upstream_commit=upstream.COMMIT,
        upstream_manifest_sha256=manifest_sha256, import_only_files=changes,
        extracted_messages=dict(source=worker_path, source_sha256=manifest['files'][worker_path],
                                classes=list(MESSAGE_CLASSES), declaration_sha256=sha(messages)),
        group_context='actual original LogicalGroupHandle/GFC session binding',
        files={name: sha(raw) for name, raw in files.items()},
        native_multiproc_worker_reproduced=False, model_executed=False,
        qualification_eligible=False, allows=[], port_sha256=sha(Path(__file__).read_bytes()))
    return record, files


def apply(source_root, manifest_path, manifest_sha256, destination):
    destination, source = Path(destination).resolve(), Path(source_root).resolve(strict=True)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('output must be disjoint from official source')
    if destination.exists():
        raise FileExistsError('previous derived evidence must be preserved')
    record, files = source_plan(source, manifest_path, manifest_sha256)
    destination.mkdir(parents=True)
    for name, raw in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
    raw = encoded(record)
    with (destination / 'PORT_MANIFEST.json').open('xb') as stream:
        stream.write(raw)
    return dict(root=str(destination), manifest_sha256=sha(raw), manifest=record)


def load(root, manifest_sha256, *, gpu=False):
    root = Path(root).resolve(strict=True)
    raw = (root / 'PORT_MANIFEST.json').read_bytes()
    if sha(raw) != manifest_sha256:
        raise ValueError('derived manifest SHA256 differs')
    manifest = upstream.read_json(raw)
    if manifest.get('schema') != SCHEMA or manifest.get('upstream_commit') != upstream.COMMIT:
        raise ValueError('derived identity differs')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('linked derived source')
        if path.is_file(): actual.add(path.relative_to(root).as_posix())
    if actual != set(manifest['files']) | {'PORT_MANIFEST.json'}:
        raise ValueError('derived source inventory differs')
    for name, expected in manifest['files'].items():
        posix = PurePosixPath(name)
        if posix.is_absolute() or '..' in posix.parts or str(posix) != name or '\\' in name:
            raise ValueError('unsafe source path')
        if sha((root / name).read_bytes()) != expected:
            raise ValueError('derived source differs: ' + name)
    for name, module in list(sys.modules.items()):
        if name == NAMESPACE or name.startswith(NAMESPACE + '.'):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise ValueError('another GF-DiT migration namespace is imported')
    sys.path.insert(0, str(root))
    names = ['protocol', 'codec', 'adapters.common', 'transport_protocol', 'migration_engine']
    if gpu: names += ['collective_runtime', 'group_context', 'data_plane']
    return SimpleNamespace(**{name.rsplit('.',1)[-1]: importlib.import_module(NAMESPACE + '.' + name) for name in names})


def scenarios():
    return [dict(id=f'{dtype}-{size}', dtype=dtype, shape=list(shape), seed=17+index*31)
            for index,(dtype,size,shape) in enumerate(
                (dtype,size,shape) for dtype in ('uint8','bfloat16','float32')
                for size,shape in [('small',(7,13)),('large',(257,257))])]


def groups(control):
    p = control.protocol
    return {name: p.ExecutionGroupSpec(name, ranks, p.ParallelSpec(sp=len(ranks)), (p.TaskKind.DIT_STEP_CHUNK,))
            for name,ranks,_ in GROUPS}


def metadata(case):
    return dict(shape=case['shape'], dtype='torch.'+case['dtype'], cursor=7, seed=case['seed'],
                label='synthetic-gfdit-migration-state', layout_modes={n:m for n,_,m in GROUPS})


def handle(control, request_id, index):
    p = control.protocol
    return p.ArtifactHandle(request_id, 'state-'+str(index), p.ArtifactKind.REQUEST_STATE,
                            p.ArtifactLayout.WORKER_LOCAL, codec_id='migration-fixture-v1')


def plan_for(control, case, index):
    p = control.protocol
    return p.MigrateArtifactsPlan(case['id'], CHAIN[index], CHAIN[index+1],
        (p.MigrateArtifactSpec(handle(control, case['id'], index+1), handle(control, case['id'], index)),),
        metadata(case))


def build_command(control, job, phase):
    # Same field mapping as the official MultiprocWorkerPool command builder.
    return control.transport_protocol.MigrateArtifactsCommand(job.migrate_id, phase, job.plan,
        job.src_group, job.dst_group, tuple(job.layout_results), job.schedule, job.metadata_payloads)


def expected_totals():
    specs = {name:ranks for name,ranks,_ in GROUPS}
    per_rank = {rank:dict(phase_executions=0, destination_states=0, byte_checks=0) for rank in (0,1)}
    for case in scenarios():
        for source,destination in zip(CHAIN,CHAIN[1:]):
            for rank in set(specs[source]) | set(specs[destination]):
                per_rank[rank]['phase_executions'] += 3 + (rank == specs[source][0])
            for rank in specs[destination]:
                per_rank[rank]['destination_states'] += 1
                per_rank[rank]['byte_checks'] += len(FIELDS)
    return dict(migrations=len(scenarios())*(len(CHAIN)-1), per_rank=per_rank,
                byte_checks=sum(row['byte_checks'] for row in per_rank.values()),
                phase_executions=sum(row['phase_executions'] for row in per_rank.values()))


def expected_phase_trace():
    specs={name:ranks for name,ranks,_ in GROUPS}
    return [dict(migrate_id=case['id']+'-move-'+str(index),phase=phase,
                 targets=[specs[CHAIN[index]][0]] if phase=='build_schedule' else [0,1])
            for case in scenarios() for index in range(len(CHAIN)-1) for phase in PHASES]


def fixture_codec(control):
    p, common, base = control.protocol, control.common, control.codec.ArtifactLayoutCodec
    class Codec(base):
        codec_id = 'migration-fixture-v1'
        def describe_output(self, *, group, group_rank, request_metadata, artifact):
            shape, dtype = tuple(request_metadata['shape']), request_metadata['dtype']
            if request_metadata['layout_modes'][group.group_id] == 'sharded':
                latent = common.sharded_tensor_field('latents', shape, dtype, shard_dim=0,
                    shard_world_size=len(group.ranks), shard_rank=group_rank)
            else:
                latent = common.replicated_tensor_field('latents', shape, dtype)
            return p.OutputArtifactLayout(artifact, (latent,
                common.replicated_tensor_field('conditioning',(2,8),dtype),
                common.replicated_tensor_field('timesteps',(5,),'torch.float32'),
                common.replicated_tensor_field('sigmas',(6,),'torch.float32')))
        def migration_extra_fields(self, *, value, layout):
            tensor = value['history'][0]
            return (common.replicated_tensor_field(('history','0'),tuple(tensor.shape),str(tensor.dtype)),)
        def pack_metadata(self, *, value, layout):
            # Preserve history through original data_plane extra edges, not a
            # CUDA pickle/D2H shortcut. Base codec strips all declared tensors.
            skeleton = dict(value, history=[None])
            raw = super().pack_metadata(value=skeleton, layout=layout)
            no_tensors(pickle.loads(raw))
            if len(raw) > 4096:
                raise ValueError('unexpectedly large state metadata')
            return raw
        def assemble(self, *, metadata_bytes, tensors, layout, device=None):
            value = super().assemble(metadata_bytes=metadata_bytes,tensors=tensors,layout=layout,device=device)
            value['history'][0] = tensors[('history','0')]
            return value
    return Codec()


def no_tensors(value):
    """Bounded control channel; payload tensors must use native GFC migration."""
    torch = sys.modules.get('torch')
    if torch is not None and isinstance(value, torch.Tensor):
        raise ValueError('tensor payload attempted to enter the control channel')
    if is_dataclass(value):
        for name in value.__dataclass_fields__: no_tensors(getattr(value,name))
    elif isinstance(value, dict):
        for item in value.values(): no_tensors(item)
    elif isinstance(value, (list,tuple)):
        for item in value: no_tensors(item)
    elif isinstance(value, bytes) and len(value) > 4096:
        raise ValueError('oversized opaque control payload')


def make_worker(control, rank, specs, codec):
    class Worker(control.data_plane.MigrationDataPlaneMixin):
        def __init__(self):
            self.worker_rank = rank
            self.codec_registry = control.codec.ArtifactLayoutCodecRegistry()
            self.codec_registry.register(codec)
            self.local_artifacts, self._migration_schedules = {}, {}
            self._artifacts_lock = threading.RLock()
            self.results = []
        def _artifact_key(self, request_id, group_id, artifact_id):
            return request_id,group_id,artifact_id
        def _group_spec(self, group_id):
            return specs[group_id]
        def _maybe_activate_group_session(self, group_id):
            control.group_context.activate(specs[group_id], rank)
        def _send_result(self, result):
            no_tensors(result)
            self.results.append(result)
    return Worker()


def pattern(torch, shape, dtype, seed, device):
    return ((torch.arange(math.prod(shape),device=device,dtype=torch.int64)+seed) % 251).to(dtype).reshape(shape)


def initial_state(torch, case, device):
    dtype = getattr(torch,case['dtype'])
    return dict(latents=pattern(torch,case['shape'],dtype,case['seed'],device),
        conditioning=pattern(torch,(2,8),dtype,case['seed']+13,device),
        timesteps=pattern(torch,(5,),torch.float32,case['seed']+29,device),
        sigmas=pattern(torch,(6,),torch.float32,case['seed']+43,device),
        history=[pattern(torch,(33,),torch.float32,case['seed']+59,device)],
        cursor=7,label='synthetic-gfdit-migration-state',seed=case['seed'])


def validate_state(torch, control, value, case, group, rank, device):
    reference = initial_state(torch,case,device)
    if metadata(case)['layout_modes'][group.group_id] == 'sharded':
        layout = control.common.sharded_tensor_field('latents',tuple(case['shape']),str(reference['latents'].dtype),
            shard_dim=0,shard_world_size=len(group.ranks),shard_rank=group.ranks.index(rank))
        reference['latents'] = reference['latents'].narrow(0,layout.global_offset[0],layout.local_shape[0])
    for name in FIELDS:
        actual, expected = (value[name][0],reference[name][0]) if name=='history' else (value[name],reference[name])
        if (actual.device != device or actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual.contiguous().view(torch.uint8),expected.contiguous().view(torch.uint8))):
            raise ValueError('actual migrated state bytes/device/shape differ: '+name)
    if any(value.get(name) != reference[name] for name in ('cursor','label','seed')):
        raise ValueError('migration lost non-tensor continuation metadata')
    return len(FIELDS)


@contextmanager
def forbid_payload_fallback(dist):
    names = ('new_group','send','recv','isend','irecv','batch_isend_irecv')
    original = {name:getattr(dist,name) for name in names}
    def forbidden(*unused,**kw):
        raise RuntimeError('NCCL payload/subgroup fallback is forbidden in GFC migration diagnostic')
    try:
        for name in names: setattr(dist,name,forbidden)
        yield
    finally:
        for name,value in original.items(): setattr(dist,name,value)


def instrument_p2p(runtime, counts):
    for name,key in (('p2p_put','send'),('p2p_put_recv','recv')):
        original = getattr(runtime,name)
        def wrapped(peer, tensor, _original=original, _key=key):
            counts[_key+'_calls'] += 1
            counts[_key+'_bytes'] += tensor.numel()*tensor.element_size()
            return _original(peer,tensor)
        setattr(runtime,name,wrapped)


def edge_counts(schedule,rank):
    sizes = {'torch.uint8':1,'torch.bfloat16':2,'torch.float32':4}
    result = dict(send_calls=0,recv_calls=0,send_bytes=0,recv_bytes=0)
    for edge in schedule.edges:
        if edge.local_copy: continue
        size = math.prod(edge.src_slice.shape)*sizes[edge.src_slice.dtype]
        for key,participant in (('send',edge.src_rank),('recv',edge.dst_rank)):
            if participant==rank:
                result[key+'_bytes'] += size
                result[key+'_calls'] += (size+BUFFER_BYTES-1)//BUFFER_BYTES
    return result


def migrate(control,worker,case,index,dist,rank,record):
    specs = groups(control)
    plan = plan_for(control,case,index)
    src,dst = specs[plan.src_group_id],specs[plan.dst_group_id]
    participants = tuple(sorted(set(src.ranks)|set(dst.ranks)))
    queued,done,failed = deque(),[],[]
    engine = control.migration_engine.MigrationEngine(build_command=lambda job,phase:build_command(control,job,phase),
        send=lambda command,ranks:queued.append((command,ranks))) if rank==0 else None
    mid = case['id']+'-move-'+str(index)
    if engine:
        engine.begin(plan=plan,src_group=src,dst_group=dst,participant_ranks=participants,
            src_leader_rank=src.ranks[0],on_done=done.append,on_error=lambda exc:failed.append(str(exc)),
            deadline=time.monotonic()+90,migrate_id=mid)
        engine.pump([],time.monotonic())
    deadline = time.monotonic()+180
    schedule = None
    while True:
        if time.monotonic()>deadline: raise TimeoutError('migration controller diagnostic deadline')
        packet = [None]
        if rank==0:
            if failed: packet[0] = ('failed',failed[0])
            elif done: packet[0] = ('done',done[0])
            elif queued: packet[0] = ('command',queued.popleft())
            else: raise RuntimeError('original engine made no progress')
            no_tensors(packet[0])
        dist.broadcast_object_list(packet,src=0)  # Bounded layout/metadata only.
        action,payload = packet[0]
        if action=='failed': raise RuntimeError(payload)
        if action=='done':
            schedule=payload
            break
        command,targets = payload
        no_tensors(command)
        if command.migrate_id!=mid or command.phase not in PHASES:
            raise ValueError('unexpected original migration command identity')
        local = None
        if rank in targets:
            worker._handle_migrate_artifacts(command)
            if len(worker.results)!=1: raise RuntimeError('native mixin emitted wrong result count')
            local=worker.results.pop()
            if (local.worker_rank!=rank or local.migrate_id!=mid or local.phase!=command.phase):
                raise ValueError('native mixin result identity differs')
            record['phase_executions'] += 1
        replies=[None,None]
        dist.all_gather_object(replies,local)
        if rank==0:
            if {r.worker_rank for r in replies if r is not None} != set(targets):
                raise ValueError('migration phase result denominator differs')
            record['phase_trace'].append(dict(migrate_id=mid,phase=command.phase,targets=list(targets)))
            engine.pump([r for r in replies if r is not None],time.monotonic())
    if rank==0 and (engine.has_pending() or engine._jobs or any(engine._rank_queues.values())):
        raise ValueError('original engine did not release migration lifecycle')
    if worker._migration_schedules:
        raise ValueError('native data plane retained completed schedule')
    return schedule,plan


def assess(records):
    expected=expected_totals()
    if len(records)!=2 or sorted(row.get('rank') for row in records)!=[0,1]:
        raise ValueError('two actual rank records required')
    uuids={row.get('gpu_uuid') for row in records}
    if len(uuids)!=2 or None in uuids or 'unavailable' in uuids:
        raise ValueError('two distinct actual GPU UUIDs required')
    bindings={row.get('source_binding_sha256') for row in records}
    if len(bindings)!=1 or not next(iter(bindings)):
        raise ValueError('rank source bindings differ')
    for row in records:
        count=expected['per_rank'][row['rank']]
        if row.get('status')!='passed' or row.get('cleanup_verified') is not True:
            raise ValueError('rank failed or cleanup missing')
        for name in ('phase_executions','destination_states','byte_checks'):
            if type(row.get(name)) is not int or row[name]!=count[name]:
                raise ValueError('incomplete rank '+name)
        if row.get('migrations')!=expected['migrations'] or row.get('p2p')!=row.get('expected_p2p'):
            raise ValueError('actual migration/P2P denominator differs')
        if any(row['p2p'][name]<=0 for name in ('send_calls','recv_calls','send_bytes','recv_bytes')):
            raise ValueError('actual GFC payload path not covered')
        if row.get('phase_trace') != (expected_phase_trace() if row['rank']==0 else []):
            raise ValueError('original controller phase sequence differs')
    return dict(schema='ditserve.gfdit-migration-check.v1',status='passed',expected=expected,ranks=records,
        byte_checks=expected['byte_checks'],migrations=expected['migrations'],
        migration_engine_executed=True,original_data_plane_executed=True,
        synthetic_state_only=True,model_executed=False,native_multiproc_worker_reproduced=False,
        serving_executed=False,performance_measurements=False,qualification_eligible=False,allows=[],
        cleanup_scope='GFC runtime and local artifacts; also require platform Succeeded and both final rank files')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','source-manifest','gfc-source-root','gfc-source-manifest','shared-python-root','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--gfc-source-manifest-sha256',required=True)
    args=parser.parse_args(argv)
    rank=int(os.environ.get('RANK','-1'))
    world=int(os.environ.get('WORLD_SIZE','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','-1'))
    output=args.output_dir.resolve()
    record=dict(rank=rank,status='started',phase_executions=0,destination_states=0,byte_checks=0,migrations=0,
        phase_trace=[],p2p=dict(send_calls=0,recv_calls=0,send_bytes=0,recv_bytes=0),
        expected_p2p=dict(send_calls=0,recv_calls=0,send_bytes=0,recv_bytes=0),cleanup_verified=False,
        qualification_eligible=False,allows=[])
    owned=False
    def checkpoint():
        gfc_check.storage_guard()
        temporary=output/f'rank-{rank}.{os.getpid()}.tmp'
        with temporary.open('xb') as stream: stream.write(encoded(record))
        os.replace(temporary,output/f'rank-{rank}.json')
    try:
        if world!=2 or rank not in (0,1) or local_rank!=rank:
            raise ValueError('use single-node torchrun --nproc_per_node=2')
        if not output.is_relative_to(Path('/shared/output/DITSERVE')):
            raise ValueError('GPU output must stay under /shared/output/DITSERVE')
        if any(name.startswith('SYMM_COLL_') for name in os.environ):
            raise ValueError('uncontrolled GFC environment override')
        for name,value in {'VLLM_RUNTIME_V2_MIGRATE_GFC_P2P':'1','VLLM_RUNTIME_V2_MIGRATE_TRACE':'0',
                           'VLLM_RUNTIME_V2_MIGRATE_NSYS':'0'}.items():
            if name in os.environ and os.environ[name]!=value: raise ValueError('uncontrolled migration flag '+name)
            os.environ[name]=value
        gfc_source=gfc_check.verify_sources(args.gfc_source_root,args.gfc_source_manifest,
            args.gfc_source_manifest_sha256,args.shared_python_root)
        upstream.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        sys.path.insert(0,str(args.shared_python_root.resolve()))
        import torch
        import torch.distributed as dist
        import gfc
        if Path(gfc.__file__).resolve()!=args.shared_python_root.resolve()/'gfc/__init__.py':
            raise ValueError('actual GFC import source differs')
        if torch.cuda.device_count()!=2: raise ValueError('exactly two visible GPUs required')
        torch.cuda.set_device(rank)
        device=torch.device('cuda',rank)
        if not torch.cuda.can_device_access_peer(rank,1-rank): raise ValueError('actual GPU peer access required')
        record.update(gpu_uuid=str(torch.cuda.get_device_properties(rank).uuid),torch=torch.__version__,
                      cuda=torch.version.cuda,gfc_file=str(Path(gfc.__file__).resolve()))
        gfc_check.storage_guard()
        if rank==0:
            output.mkdir(parents=True,exist_ok=False)
            derived=apply(args.source_root,args.source_manifest,args.source_manifest_sha256,output/'derived')
        dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=180))
        binding=[derived if rank==0 else None]
        dist.broadcast_object_list(binding,src=0)
        derived=binding[0]
        owned=True
        control=load(derived['root'],derived['manifest_sha256'],gpu=True)
        source=dict(gfdit_manifest_sha256=args.source_manifest_sha256,gfc=gfc_source,
                    derived_manifest_sha256=derived['manifest_sha256'],port_sha256=sha(Path(__file__).read_bytes()))
        record.update(source=source,source_binding_sha256=sha(encoded(source)))
        checkpoint()
        with forbid_payload_fallback(dist):
            runtime=control.collective_runtime.init_runtime_v2_collective_runtime(
                backend='gfc',device=device,max_group_size=2,max_collective_bytes=BUFFER_BYTES)
            migration=control.collective_runtime.get_gfc_migrate_runtime()
            if migration is runtime or migration.stream==runtime.stream:
                raise ValueError('native migration runtime must have independent stream/epochs')
            if not control.collective_runtime.runtime_v2_uses_gfc() or not control.data_plane._MIGRATE_GFC_P2P:
                raise ValueError('actual native data plane is not using GFC')
            record['gfc_config']=asdict(migration.config)
            if (migration.config.max_collective_bytes!=BUFFER_BYTES or migration.config.max_group_size!=2
                    or migration.config.use_tma or migration.config.use_copy_engine or migration.config.enable_fused_path):
                raise ValueError('unexpected native GFC diagnostic configuration')
            instrument_p2p(migration,record['p2p'])
            specs=groups(control)
            worker=make_worker(control,rank,specs,fixture_codec(control))
            for case in scenarios():
                if rank==0:
                    first=handle(control,case['id'],0)
                    worker.local_artifacts[(case['id'],CHAIN[0],first.artifact_id)]=control.protocol.ArtifactValue(
                        first,initial_state(torch,case,device))
                for index in range(len(CHAIN)-1):
                    record['current_migration']=case['id']+'-'+str(index)
                    checkpoint()
                    schedule,plan=migrate(control,worker,case,index,dist,rank,record)
                    for name,count in edge_counts(schedule,rank).items(): record['expected_p2p'][name]+=count
                    destination=specs[plan.dst_group_id]
                    if rank in destination.ranks:
                        key=(case['id'],destination.group_id,plan.artifacts[0].handle.artifact_id)
                        record['byte_checks']+=validate_state(torch,control,worker.local_artifacts[key].value,
                                                              case,destination,rank,device)
                        record['destination_states']+=1
                    # Every rank independently validated bytes before any old
                    # source is evicted or reused by a later migration.
                    dist.barrier()
                    source_handle=plan.artifacts[0].effective_source_handle
                    worker.local_artifacts.pop((case['id'],plan.src_group_id,source_handle.artifact_id),None)
                    record['migrations']+=1
                worker.local_artifacts.clear()
            if worker.local_artifacts or worker._migration_schedules or worker.results:
                raise ValueError('worker lifecycle did not settle')
            if record['p2p']!=record['expected_p2p']: raise ValueError('GFC actual calls/bytes differ from native schedule')
            control.collective_runtime.shutdown_runtime_v2_collective_runtime()
        torch.cuda.synchronize()
        load(derived['root'],derived['manifest_sha256'],gpu=True)
        if gfc_check.verify_sources(args.gfc_source_root,args.gfc_source_manifest,
                args.gfc_source_manifest_sha256,args.shared_python_root)!=gfc_source:
            raise ValueError('actual GFC source changed during migration')
        upstream.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        record.update(status='passed',cleanup_verified=True)
        record.pop('current_migration',None)
        checkpoint()
        records=[None,None]
        dist.all_gather_object(records,record)
        summary=assess(records)
        if rank==0:
            gfc_check.storage_guard()
            with (output/'summary.json').open('xb') as stream: stream.write(encoded(summary))
        dist.barrier()
        dist.destroy_process_group()
        print(json.dumps(dict(rank=rank,status='passed',byte_checks=record['byte_checks'],qualification_eligible=False)))
        return 0
    except Exception as exc:
        record.update(status='failed',error=repr(exc),cleanup_verified=False)
        if owned and not isinstance(exc,gfc_check.StorageStop):
            try: checkpoint()
            except gfc_check.StorageStop: pass
        print(json.dumps(record,sort_keys=True,allow_nan=False),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
