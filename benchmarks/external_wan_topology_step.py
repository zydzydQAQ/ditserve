"""SP2/SP8 launcher for the byte-frozen Wan first-step observer and SP1 oracles.

Only the requested Ulysses topology changes. The original observer, tensor
capture, scheduler, model, source verification and tolerance remain immutable.
No full request, quality or performance claim follows from this diagnostic.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import external_wan_first_step as observer

SCHEMA = 'ditserve.wan-topology-first-step.v1'
OBSERVER_COMMIT = '45685bea933790a5f335f614cd1e96d7e2d6f44f'
CROSS_SP_THRESHOLDS = {'relative_l2_max': 0.25, 'cosine_similarity_min': 0.95}
COSINE_BASIS = ('measured cosine when the boundary row carries one; otherwise the exact lower bound '
                'cos >= sqrt(1 - relative_l2^2), valid for relative_l2 < 1 (the frozen observer emits no cosine)')
PINNED = {
    'external_wan_first_step.py':'25969edc40b315baabc726503cf59b9920fd0913d1ea34f0eca47abfbffb6b8b',
    'external_xdit_native.py':'b1a2fec877a063d0a750015c4bbbca6105cc32dd7eb904a340884dab0a99f839',
    'external_xdit_distributed.py':'e0c737925505e1b3e1668d2295d158e12a47e64f7800dbab0da38a45ac68b0cd',
    'external_xdit.py':'719ff99683a5a29341233254e622b9deb4bb7b374e9bba0ba749940adceddbbb',
}


def verify_observer_sources():
    package = Path(__file__).resolve().parent
    modules = (observer, observer.native, observer.distributed, observer.native.source)
    actual = {}
    for module in modules:
        path = Path(module.__file__).resolve(strict=True)
        if path.parent != package or path.name not in PINNED:
            raise ValueError('observer helper import escaped immutable package')
        actual[path.name] = observer.sha(path.read_bytes())
    if actual != PINNED:
        raise ValueError('original observer/helper bytes differ; reference compatibility denied')
    return dict(observer_commit=OBSERVER_COMMIT, observer_sources=actual,
        launcher_sha256=observer.sha(Path(__file__).read_bytes()),
        compatibility='Original observer and all binding helpers are byte-identical; no adapter binding relaxed')


def topology(world, environ):
    if type(world) is not int or world not in (2,8): raise ValueError('only SP2/SP8 are admitted')
    values = {}
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE'):
        value = environ.get(key,'')
        if not isinstance(value,str) or not value.isdecimal(): raise ValueError('missing or invalid torchrun topology: '+key)
        values[key] = int(value)
    rank = values['RANK']
    if (values['WORLD_SIZE'] != world or values['LOCAL_WORLD_SIZE'] != world
            or not 0 <= rank < world or values['LOCAL_RANK'] != rank):
        raise ValueError('requires one node and exact rank/local-rank/world agreement')
    return rank


def runner_config(request, identity, output, world):
    if type(world) is not int or world not in (2,8): raise ValueError('only SP2/SP8 are admitted')
    config = observer.native.runner_config('wan22',request,identity,output,cpu_offload=False)
    # Same three fields as the original distributed_config; its CLI supports 2/4.
    config.update(ulysses_degree=world,ring_degree=1,use_ray=False)
    return config


def geometry_contract(config, oracle, world):
    if type(world) is not int or world not in (2,8): raise ValueError('invalid target degree')
    heads, patch = config.get('num_attention_heads'), config.get('patch_size')
    if heads != 24 or type(heads) is not int or patch != [1,2,2] or heads % world:
        raise ValueError('verified Wan22 attention heads/patch do not support requested SP')
    boundaries = {r['name']:r['tensor'] for r in oracle['ranks'][0]['boundaries']}
    latent = boundaries['prepare_latents.0.out']['shape']
    if latent != [1,48,31,32,56]: raise ValueError('oracle is not the frozen Wan22-small latent geometry')
    tokens = (latent[2]//patch[0])*(latent[3]//patch[1])*(latent[4]//patch[2])
    if tokens % world or boundaries['transformer.0.in.timestep']['shape'] != [1,tokens]:
        raise ValueError('full sequence/expanded timestep cannot be evenly partitioned')
    return dict(world_size=world,heads=heads,heads_per_rank=heads//world,patch_size=patch,
        latent_shape=latent,sequence_tokens=tokens,tokens_per_rank=tokens//world,
        expected_padding_tokens=0,model_config_sha256=observer.sha(observer.canonical(config)))


def boundary_gate(comparison, thresholds=None):
    """Check every declared cross-SP threshold for one boundary; unknown metrics fail closed."""
    limits = CROSS_SP_THRESHOLDS if thresholds is None else thresholds
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if comparison.get('bitwise_equal') is True:
        relative_l2, cosine, basis = 0.0, 1.0, 'bitwise_equal'
    else:
        relative_l2 = comparison.get('relative_l2')
        relative_l2 = float(relative_l2) if finite(relative_l2) and relative_l2 >= 0 else None
        cosine, basis = comparison.get('cosine_similarity'), 'measured'
        if finite(cosine):
            cosine = float(cosine)
        elif relative_l2 is not None and relative_l2 < 1:
            cosine, basis = math.sqrt(1.0 - relative_l2 * relative_l2), 'lower_bound_from_relative_l2'
        else:
            cosine, basis = None, 'unavailable'
    failures = []
    if relative_l2 is None or relative_l2 > limits['relative_l2_max']:
        failures.append('relative_l2')
    if cosine is None or cosine < limits['cosine_similarity_min']:
        failures.append('cosine_similarity')
    return dict(passed=not failures, relative_l2=relative_l2, cosine_similarity=cosine,
                cosine_basis=basis, failed_thresholds=failures)


def aggregate(rows, binding, world, launcher, reference_sha256):
    if world not in (2,8): raise ValueError('unsupported aggregate topology')
    # Reuse every old UUID/source/cleanup/call-count check, then close its
    # world==4-only comparison check for every topology admitted here.
    result = observer.aggregate(rows,binding,world)
    first = sorted(rows,key=lambda r:r['rank'])[0]
    if not result['cross_rank_full_boundary_hashes_equal']:
        raise ValueError('full boundaries differ across ranks; rank0 metrics cannot stand for all ranks')
    for row in rows:
        if (row.get('launcher') != launcher or row.get('reference_summary_sha256') != reference_sha256
                or row.get('topology_contract') != first['topology_contract']
                or row['topology_contract']['world_size'] != world
                or row.get('artifact_bytes') != 0 or len(row['boundaries']) != 18):
            raise ValueError('launcher/reference/geometry/full-boundary inventory differs')
        compared = [b for b in row['boundaries'] if 'comparison' in b]
        if len(compared) != (18 if row['rank']==0 else 0):
            raise ValueError('only rank0 must compare all 18 complete boundaries')
    named = [(b['name'], boundary_gate(b['comparison'])) for b in first['boundaries'] if 'comparison' in b]
    max_rel_l2 = max((g['relative_l2'] for _, g in named if g['relative_l2'] is not None), default=None)
    min_cosine = min((g['cosine_similarity'] for _, g in named if g['cosine_similarity'] is not None), default=None)
    failed_boundaries = {name: g['failed_thresholds'] for name, g in named if not g['passed']}
    bounded_drift_passed = bool(named) and not failed_boundaries
    result.update(schema=SCHEMA,diagnostic_completed=True,launcher=launcher,
        reference_summary_sha256=reference_sha256,topology_contract=first['topology_contract'],
        original_observer_schema=observer.SCHEMA,observed_boundaries_per_rank=18,
        compatibility_scope='same byte-frozen observer and original source binding; topology-only launcher',
        cross_sp_acceptance_gate=dict(
            mode='semantic_quality_equivalence',
            reference_topology='sp1',
            candidate_topology=f'sp{world}',
            passed=bounded_drift_passed,
            max_relative_l2=max_rel_l2,
            min_cosine_similarity=min_cosine,
            cosine_basis=COSINE_BASIS,
            failed_boundaries=failed_boundaries,
            relative_l2_max=CROSS_SP_THRESHOLDS['relative_l2_max'],
            cosine_similarity_min=CROSS_SP_THRESHOLDS['cosine_similarity_min'],
            metrics_not_computed=['PSNR', 'SSIM', 'LPIPS', 'VBench'],
            strict_elementwise_1e5_is_diagnostic_only=True,
            all_boundaries_within_1e5_tolerance=result.get('all_boundaries_within_tolerance'),
            rationale=(
                f'Cross-SP (sp{world} vs sp1) boundary drift in BF16 Ulysses attention is evaluated '
                'as bounded numerical drift (relative L2 and cosine per boundary) rather than 1e-5 elementwise '
                'equality; no perceptual or semantic quality metric is computed here'
            ),
        ))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('plan','run'))
    p.add_argument('--world-size',type=int,choices=(2,8),required=True)
    for name in ('source-root','source-manifest','model-root','protocol','output-dir','reference-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','protocol-sha256','prompt','reference-summary-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--models',type=Path,default=observer.native.DEFAULT_MODELS)
    p.add_argument('--models-sha256',default=observer.native.DEFAULT_MODELS_SHA256)
    p.add_argument('--model',choices=('wan22',),required=True);p.add_argument('--shape',choices=('wan22-small',),required=True)
    p.add_argument('--seed',type=int,choices=(42,43,44,45),default=42)
    p.add_argument('--timeout-seconds',type=int,default=900)
    args=p.parse_args(argv)
    if not 30<=args.timeout_seconds<=1200: p.error('timeout must be 30..1200 seconds')
    output=args.output_dir
    if output.resolve()!=output or not output.is_relative_to('/shared/output/DITSERVE'):
        raise ValueError('resolved project output required')
    for path in (args.source_root,args.model_root,args.reference_dir):
        if output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output):
            raise ValueError('output must be disjoint from inputs and protected reference')
    launcher=verify_observer_sources();rank=topology(args.world_size,os.environ) if args.command=='run' else 0
    deadline=time.monotonic()+args.timeout_seconds;stop=threading.Event();created=False
    result=dict(schema=SCHEMA,rank=rank,world_size=args.world_size,status='failed',cleanup_completed=False,
        completed_requests=0,qualification_eligible=False,performance_eligible=False,launcher=launcher,
        reference_summary_sha256=args.reference_summary_sha256)
    try:
        if args.command=='run':
            observer.native.check_storage();output.mkdir(parents=True,exist_ok=True)
            rankdir=output/f'rank-{rank}';rankdir.mkdir(exist_ok=False);created=True
            def guard():
                while not stop.wait(5):
                    try:
                        observer.native.check_storage()
                        if time.monotonic()>=deadline: raise TimeoutError('topology first-step deadline')
                    except Exception: os._exit(75)
            threading.Thread(target=guard,daemon=True).start()
        manifest=observer.native.source.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        request=observer.native.request_plan(observer.native.bound_json(args.protocol,args.protocol_sha256),
            'wan22','wan22-small','formal',args.prompt,args.seed)
        identity=observer.native.verify_model('wan22',args.model_root,args.models,args.models_sha256)
        result.update(request=request,model_identity=identity,protocol_sha256=args.protocol_sha256,
            source=dict(commit=observer.native.source.COMMIT,manifest_sha256=args.source_manifest_sha256,
                        archive_sha256=observer.native.source.ARCHIVE_SHA256),adapter=dict(PINNED))
        binding=observer.reference_binding(result);result['binding']=binding
        # The original loader still demands exact full source/request/adapter
        # identity and verifies every protected safetensor before CUDA inference.
        reference=observer.load_reference(args.reference_dir,args.reference_summary_sha256,binding) if rank==0 else None
        oracle=observer.native.source.read_json(observer.read_bound(args.reference_dir/'summary.json',args.reference_summary_sha256))
        if oracle.get('binding')!=binding or oracle.get('world_size')!=1 or oracle.get('status')!='first_step_observed':
            raise ValueError('source-bound SP1 oracle identity differs')
        config=runner_config(request,identity,output,args.world_size);result['runner_config']=config
        model_config=observer.native.source.read_json((args.model_root/'transformer/config.json').read_bytes())
        result['topology_contract']=geometry_contract(model_config,oracle,args.world_size)
        if args.command=='plan':print(json.dumps(dict(result,status='planned'),sort_keys=True,allow_nan=False));return 0
        recorder=observer.Recorder(output,rank,reference)
        with redirect_stdout(sys.stderr):observer.execute(args,result,config,identity,manifest,recorder)
        observer.native.source.verify_source(args.source_root,args.source_manifest,args.source_manifest_sha256)
        if verify_observer_sources()!=launcher:raise ValueError('observer or launcher changed during run')
        observer.validate_rank(result)
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()[-8000:])
    if created:
        observer.native.check_storage();observer.distributed.publish_json(rankdir/'summary.json',result)
        if rank==0:
            try:
                if result['status']!='first_step_observed':raise RuntimeError('rank0 failed; do not wait for collective peers')
                paths=[output/f'rank-{i}'/'summary.json' for i in range(args.world_size)]
                while not all(path.exists() for path in paths):
                    if time.monotonic()>=deadline:raise TimeoutError('missing final rank receipt')
                    observer.native.check_storage();time.sleep(.5)
                rows=[observer.native.source.read_json(path.read_bytes()) for path in paths]
                summary=aggregate(rows,result['binding'],args.world_size,launcher,args.reference_summary_sha256)
            except Exception as exc:
                summary=dict(schema=SCHEMA,status='failed',diagnostic_completed=False,error=repr(exc),
                    completed_requests=0,qualification_eligible=False,ranks=[result])
            observer.native.check_storage();observer.distributed.publish_json(output/'summary.json',summary)
            print(json.dumps(dict(summary=str(output/'summary.json'),status=summary['status'],sha256=observer.sha((output/'summary.json').read_bytes()))))
            if summary['status']=='failed':result['status']='failed'
    stop.set();return 0 if result['status']=='first_step_observed' else 1


if __name__=='__main__':raise SystemExit(main())
