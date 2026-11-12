"""Real pinned GFC smoke: one-rank imports, then two/eight-rank data movement.

Run through torchrun with --expected-world-size 1, 2 or 8. Source and shared
package hashes are checked before imports. Multi-rank tests use GFC exclusively
for payload movement; NCCL world operations coordinate/bootstrap the test.
No model, serving or performance qualification follows from this smoke.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
from datetime import timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True

COMMIT = '9aadcdc6d4abc30af13feac908fbb4be98313398'
ARCHIVE_SHA256 = '17e584a929c1555c9cec6789500526265e157b06524cfce5b28bce126a7c893f'
REPOSITORY = 'SJTU-Liquid/group-free-collectives'
DTYPES = ('uint8', 'bfloat16', 'float32')
SIZES = (256, 65536)


class StorageStop(RuntimeError):
    pass


def storage_guard():
    """The mounted /shared is Kairos /data; never write after its stop line.

    The submitting wrapper additionally monitors every five seconds and stops
    the process group while a CUDA call is blocked. This check also protects
    direct invocation before writes and between collective calls.
    """
    try:
        proc = subprocess.run(['df', '-P', '-B1', '/shared'], text=True, capture_output=True,
                              check=True, timeout=5)
        lines = proc.stdout.strip().splitlines()
        fields = lines[-1].split()
        if len(lines) != 2 or len(fields) != 6:
            raise ValueError('unexpected df output')
        used, percent = int(fields[2]), int(fields[4].removesuffix('%'))
        if used < 0 or not 0 <= percent <= 100:
            raise ValueError('invalid df values')
        if used > 790 * 1024**3 or percent >= 80:
            raise StorageStop(f'shared storage stop line reached: used_bytes={used}, used_percent={percent}')
        return {'used_bytes': used, 'used_percent': percent}
    except StorageStop:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise StorageStop('cannot verify shared storage: ' + str(exc)) from exc


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def verify_sources(source_root, manifest_path, manifest_sha256, shared_python_root):
    raw = Path(manifest_path).read_bytes()
    if not isinstance(manifest_sha256, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha256) or digest(raw) != manifest_sha256:
        raise ValueError('GFC source manifest SHA256 differs')
    manifest = read_json(raw)
    if (manifest.get('schema') != 'ditserve.official-source-archive.v1'
            or manifest.get('repository') != REPOSITORY or manifest.get('commit') != COMMIT
            or manifest.get('archive_sha256') != ARCHIVE_SHA256):
        raise ValueError('official GFC identity differs')
    root, shared = Path(source_root).resolve(strict=True), Path(shared_python_root).resolve(strict=True)
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('empty GFC inventory')
    for name, expected in files.items():
        parts = PurePosixPath(name)
        if (not name or parts.is_absolute() or '..' in parts.parts or '\\' in name or str(parts) != name
                or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)):
            raise ValueError('invalid GFC inventory path/hash: ' + name)
        if any(root.joinpath(*parts.parts[:n]).is_symlink() for n in range(1, len(parts.parts) + 1)):
            raise ValueError('symlink in official GFC source')
        if digest((root / name).read_bytes()) != expected:
            raise ValueError('official GFC file hash differs: ' + name)
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink in official GFC source')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError('official GFC inventory differs')
    package_files = {n: h for n, h in files.items() if n.startswith('gfc/') and n.endswith('.py')}
    if not package_files or 'gfc/__init__.py' not in package_files:
        raise ValueError('GFC package absent from official inventory')
    observed = {}
    for path in (shared / 'gfc').rglob('*'):
        if path.is_symlink() or not path.resolve().is_relative_to(shared):
            raise ValueError('symlink/escape in shared GFC package')
        if path.is_file() and path.suffix in ('.py', '.pyc', '.so', '.pyd'):
            observed[path.relative_to(shared).as_posix()] = digest(path.read_bytes())
    if observed != package_files:
        raise ValueError('shared GFC importable files differ from pinned upstream')
    return {'repository': REPOSITORY, 'commit': COMMIT, 'archive_sha256': ARCHIVE_SHA256,
            'manifest_sha256': manifest_sha256, 'verified_official_files': len(files),
            'verified_package_files': package_files, 'shared_python_root': str(shared)}


def cases(world, iterations=2):
    if type(world) is not int or world not in (1, 2, 8):
        raise ValueError('expected world size must be 1, 2 or 8')
    if type(iterations) is not int or not 1 <= iterations <= 10:
        raise ValueError('iterations must be an integer in [1, 10]')
    if world == 1:
        return []
    full = tuple(range(world))
    groups = [full, full[::-1]]
    if world == 8:
        groups += [(1, 2, 4), (0, 7), (0, 2, 4, 6), (1, 3, 5, 7)]
    groups += [full]  # Return to the original order after overlapping regroup.
    result = []
    for stage, group in enumerate(groups):
        for dtype in DTYPES:
            for size in SIZES:
                for iteration in range(iterations):
                    for op in ('all_gather', 'all2all'):
                        result.append(dict(op=op, ranks=list(group), stage=stage, dtype=dtype,
                                           bytes_per_peer=size, iteration=iteration))
    for src in range(world):
        dst = world - 1 - src
        for size in SIZES:
            for iteration in range(iterations):
                for op in ('p2p_put', 'p2p_get'):
                    result.append(dict(op=op, ranks=[src, dst], stage=len(groups), dtype='uint8',
                                       bytes_per_peer=size, iteration=iteration))
    for index, case in enumerate(result):
        case['id'] = index
    return result


def expected_counts(plan, world):
    counts = []
    for rank in range(world):
        participating = [c for c in plan if rank in c['ranks']]
        checked = [c for c in participating if c['op'] in ('all_gather', 'all2all') or rank == c['ranks'][1]]
        counts.append({'rank': rank, 'executions': len(participating), 'byte_checks': len(checked)})
    return counts


@contextlib.contextmanager
def forbid_subgroups(dist):
    """Fail if any tested path tries to replace dynamic GFC with a Torch group."""
    original = dist.new_group
    def forbidden(*args, **kwargs):
        raise RuntimeError('torch.distributed.new_group is forbidden in GFC smoke')
    dist.new_group = forbidden
    try:
        yield
    finally:
        dist.new_group = original


def summarize(records, world, plan_hash, expected, source_hash):
    if len(records) != world or {r.get('rank') for r in records} != set(range(world)):
        raise ValueError('missing or duplicate rank records')
    for record in records:
        rank = record['rank']
        if (record.get('status') != 'passed' or record.get('world_size') != world
                or record.get('plan_sha256') != plan_hash or record.get('source_binding_sha256') != source_hash
                or type(record.get('executions')) is not int or type(record.get('byte_checks')) is not int
                or record['executions'] != expected[rank]['executions']
                or record['byte_checks'] != expected[rank]['byte_checks']
                or record.get('cleanup_verified') is not True
                or record.get('subgroup_process_groups_created') != 0):
            raise ValueError('incomplete or inconsistent rank record: ' + str(rank))
    uuids = [r.get('runtime', {}).get('gpu_uuid') for r in records]
    if any(not isinstance(u, str) or not u or u == 'unavailable' for u in uuids) or len(set(uuids)) != world:
        raise ValueError('missing or duplicate actual GPU UUIDs')
    return {'status': 'passed', 'world_size': world, 'plan_sha256': plan_hash,
            'source_binding_sha256': source_hash, 'ranks': sorted(records, key=lambda r: r['rank']),
            'executions': sum(r['executions'] for r in records),
            'byte_checks': sum(r['byte_checks'] for r in records),
            'collectives_executed': world > 1, 'serving_executed': False,
            'qualification_eligible': False, 'allows': [], 'measurements': []}


def run_case(torch, runtime, case, rank, device):
    """Deterministic exact-byte oracle; expected payload never uses NCCL."""
    ranks, nbytes = case['ranks'], case['bytes_per_peer']
    dtype = getattr(torch, case['dtype'])
    producer = torch.cuda.Stream(device=device)
    def pattern(src, dst):
        # Arbitrary bit patterns, including NaNs when viewed as floating point;
        # comparisons below inspect bytes rather than floating point equality.
        indices = torch.arange(nbytes, dtype=torch.int64, device=device)
        return ((indices * 17 + src * 29 + dst * 7 + case['id'] * 13) & 255).to(torch.uint8)
    checked = False
    if case['op'] in ('all_gather', 'all2all'):
        group = runtime.register_group(tuple(ranks))
        with torch.cuda.stream(producer):
            raw = pattern(rank, 0) if case['op'] == 'all_gather' else torch.cat([pattern(rank, dst) for dst in ranks])
            inp = raw.view(dtype)
            ready = torch.cuda.Event()
            ready.record(producer)
        out_bytes = len(ranks) * nbytes
        out = torch.empty(out_bytes // inp.element_size(), dtype=dtype, device=device)
        runtime.wait_for_external(ready)
        if case['op'] == 'all_gather':
            runtime.all_gather(inp, out, group)
            expected = torch.cat([pattern(src, 0) for src in ranks])
        else:
            runtime.all2all(inp, out, group, slice_bytes=nbytes)
            expected = torch.cat([pattern(src, rank) for src in ranks])
        torch.cuda.current_stream(device).wait_event(runtime.record_event())
        checked = True
        if not torch.equal(out.view(torch.uint8), expected):
            raise RuntimeError('GFC exact-byte mismatch at case ' + str(case['id']))
    else:
        src, dst = ranks
        if rank == src:
            with torch.cuda.stream(producer):
                inp = pattern(src, dst)
                ready = torch.cuda.Event()
                ready.record(producer)
            runtime.wait_for_external(ready)
            if case['op'] == 'p2p_put':
                runtime.p2p_put(dst_rank=dst, src=inp)
            else:
                runtime.p2p_get_serve(dst_rank=dst, src=inp)
        else:
            out = torch.empty(nbytes, dtype=torch.uint8, device=device)
            if case['op'] == 'p2p_put':
                runtime.p2p_put_recv(src_rank=src, dst=out)
            else:
                runtime.p2p_get(src_rank=src, dst=out)
            torch.cuda.current_stream(device).wait_event(runtime.record_event())
            checked = True
            if not torch.equal(out, pattern(src, dst)):
                raise RuntimeError('GFC P2P exact-byte mismatch at case ' + str(case['id']))
    runtime.stream.synchronize()
    return int(checked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--shared-python-root', type=Path, required=True)
    parser.add_argument('--expected-world-size', type=int, required=True, choices=(1, 2, 8))
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    plan = cases(args.expected_world_size, args.iterations)
    record = {'schema': 'ditserve.external-gfc-smoke-rank.v1', 'rank': rank, 'world_size': world,
              'status': 'started', 'executions': 0, 'byte_checks': 0, 'cleanup_verified': False,
              'subgroup_process_groups_created': 0, 'plan_sha256': digest(canonical(plan)),
              'cleanup_scope': 'GFC runtime shutdown; platform success and final rank files must also be verified',
              'qualification_eligible': False, 'serving_executed': False, 'allows': [], 'measurements': []}
    output = args.output_dir / ('rank-' + str(rank) + '.json')
    owned = False
    runtime = None
    def checkpoint():
        storage_guard()
        temporary = output.with_name(output.name + '.' + str(os.getpid()) + '.tmp')
        with temporary.open('x') as stream:
            json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        os.replace(temporary, output)
    try:
        if world != args.expected_world_size or not 0 <= rank < world or local_rank != rank:
            raise ValueError('single-node torchrun rank/world allocation differs')
        inherited = sorted(k for k in os.environ if k.startswith('SYMM_COLL_'))
        if inherited:
            raise ValueError('uncontrolled GFC environment overrides: ' + ','.join(inherited))
        source = verify_sources(args.source_root, args.source_manifest, args.source_manifest_sha256, args.shared_python_root)
        record.update(source=source, source_binding_sha256=digest(canonical(source)))
        if not args.output_dir.resolve().is_relative_to(Path('/shared/output/DITSERVE')):
            raise ValueError('GPU smoke outputs must stay under /shared/output/DITSERVE')
        record['initial_storage'] = storage_guard()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(record, stream)
        owned = True
        sys.path.insert(0, str(args.shared_python_root.resolve()))
        torch, triton, gfc = (importlib.import_module(n) for n in ('torch', 'triton', 'gfc'))
        if Path(gfc.__file__).resolve() != args.shared_python_root.resolve() / 'gfc/__init__.py':
            raise ValueError('GFC import source differs')
        if not torch.cuda.is_available() or torch.cuda.device_count() != world:
            raise ValueError('actual visible GPU count differs')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        prop = torch.cuda.get_device_properties(local_rank)
        record['runtime'] = {'torch': torch.__version__, 'triton': triton.__version__, 'cuda': torch.version.cuda,
                             'gfc_file': str(Path(gfc.__file__).resolve()), 'gfc_init_sha256': digest(Path(gfc.__file__).read_bytes()),
                             'gpu_name': prop.name, 'gpu_uuid': str(getattr(prop, 'uuid', 'unavailable')),
                             'peer_access': [i == local_rank or torch.cuda.can_device_access_peer(local_rank, i) for i in range(world)]}
        sm = importlib.import_module('torch.distributed._symmetric_memory')
        if not all(hasattr(sm, n) for n in ('empty', 'rendezvous')):
            raise ValueError('torch symmetric memory API missing')
        config = gfc.SymmetricCollectiveConfig(max_group_size=world, max_collective_bytes=1024 * 1024,
                                              use_tma=False, use_copy_engine=False, enable_fused_path=False,
                                              enable_debug_checks=False, timeout_ms=15000)
        record['config'] = dataclasses.asdict(config)
        if world > 1:
            if not all(record['runtime']['peer_access']):
                raise RuntimeError('GFC requires GPU peer access; no fallback allowed')
            dist = torch.distributed
            dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=90))
            with forbid_subgroups(dist):
                record['phase'] = 'symmetric_memory_initialization'
                checkpoint()
                runtime = gfc.SymmetricCollectiveRuntime(config, device=device)
                if dataclasses.asdict(runtime.config) != record['config']:
                    raise RuntimeError('actual GFC config differs')
                last_stage = None
                last_storage_check = time.monotonic()
                for case in plan:
                    if time.monotonic() - last_storage_check >= 5:
                        storage_guard()
                        last_storage_check = time.monotonic()
                    if rank in case['ranks']:
                        record.update(phase='collectives', current_case=case)
                        if case['stage'] != last_stage:
                            checkpoint()
                            last_stage = case['stage']
                        record['byte_checks'] += run_case(torch, runtime, case, rank, device)
                        record['executions'] += 1
                    dist.barrier()  # World control only, never a payload reference.
                runtime.shutdown()
                runtime = None
            torch.cuda.synchronize()
        if verify_sources(args.source_root, args.source_manifest, args.source_manifest_sha256, args.shared_python_root) != source:
            raise ValueError('source changed during smoke')
        record.update(status='passed', phase='completed', cleanup_verified=True, collectives_executed=world > 1)
        record.pop('current_case', None)
        checkpoint()
        records = [record]
        if world > 1:
            records = [None] * world
            torch.distributed.all_gather_object(records, record)
        summary = summarize(records, world, record['plan_sha256'], expected_counts(plan, world), record['source_binding_sha256'])
        summary['schema'] = 'ditserve.external-gfc-smoke.v1'
        summary['scope'] = 'GFC import/communication correctness smoke; no model or serving measurements'
        if rank == 0:
            storage_guard()
            with (args.output_dir / 'summary.json').open('x') as stream:
                json.dump(summary, stream, sort_keys=True, indent=2, allow_nan=False)
        if world > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        print(json.dumps({'rank': rank, 'status': 'passed', 'executions': record['executions'], 'byte_checks': record['byte_checks']}))
        return 0
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:3000], cleanup_verified=False)
        if owned and not isinstance(exc, StorageStop):
            try:
                checkpoint()
            except StorageStop as stop:
                record['storage_stop'] = str(stop)
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        # Do not attempt further distributed operations after a CUDA/peer fault;
        # torchrun tears down siblings. Missing rank/summary evidence is failure.
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
