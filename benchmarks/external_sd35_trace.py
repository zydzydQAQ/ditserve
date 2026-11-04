"""Opt-in, bounded SD3.5 observation; hashing can synchronize CUDA, never timing evidence."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
import re

TRAJECTORIES = ('normal_sp2', 'round_sp2_full', 'round_sp2_chunked')
CHUNK_ENDS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
REFERENCE_SHA256 = 'a91c5bb3541702314baff6b98ccb4dfe4d2cd072e4670d23838eb936d4db958a'
MAX_BYTES = 1024 * 1024
REFERENCE_MAX_BYTES = 128 * 1024
METHODS = ('_sync_pipeline', '_backbone_forward', '_scheduler_step', '_naive_forward', '_async_pipeline')
RUNTIME_FIELDS = ('split_text_embed_in_sp', 'split_latents_by_rows', 'text_embed_sp_pad',
    'patch_mode', 'pipeline_patch_idx', 'num_pipeline_patch', 'pp_patches_height',
    'pp_patches_start_idx_local', 'pp_patches_start_end_idx_global',
    'pp_patches_token_start_idx_local', 'pp_patches_token_start_end_idx_global', 'pp_patches_token_num')


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n').encode()


def bounded_read(path, limit):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('trace evidence path must be an unlinked bounded regular file')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('trace evidence exceeded its byte budget')
    return raw


def bind_reference(path, expected_sha256, binding, assess):
    if expected_sha256 != REFERENCE_SHA256:
        raise ValueError('trace requires the fixed completed r3 reference SHA256')
    raw = bounded_read(path, REFERENCE_MAX_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('trace reference SHA256 differs')
    row = json.loads(raw)
    if (row.get('schema') != 'ditserve.tetriserve-sp2-diagnostic.v1'
            or row.get('status') != 'diagnostic_completed' or row.get('diagnostic_completed') is not True
            or row.get('numerical_check_passed') is not False or row.get('qualification_eligible') is not False
            or row.get('binding') != binding or not assess(row.get('ranks', []))):
        raise ValueError('trace reference is not the completed r3 diagnosis with identical prior bindings')
    return row


def tensor_description(tensor):
    from external_sd35_round_check import tensor_summary
    if tensor.ndim != 0:
        return tensor_summary(tensor)
    # A scalar cannot change element size through view(uint8) directly. Keep its
    # actual [] shape in the same description+byte hash used by tensor_summary.
    import torch
    value = tensor.detach().contiguous().cpu()
    if not torch.isfinite(value).all().item():
        raise ValueError('nonfinite scalar tensor')
    description = dict(shape=[], dtype=str(value.dtype))
    digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(description, sha256=digest.hexdigest(), finite=True,
                minimum=float(value), maximum=float(value), retained_array=False)


def valid_tensor(row):
    return (isinstance(row, dict) and isinstance(row.get('shape'), list)
        and all(type(v) is int and v > 0 for v in row['shape'])
        and isinstance(row.get('dtype'), str) and row.get('finite') is True
        and row.get('retained_array') is False
        and all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ('minimum', 'maximum'))
        and row['minimum'] <= row['maximum']
        and isinstance(row.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None)


class TraceObserver:
    """Observe exact original calls; restore instance overrides including absence."""
    def __init__(self, pipe, runtime_state, rank, *, summarize=tensor_description):
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError('trace requires one of the two actual ranks')
        self.pipe, self.runtime_state, self.summarize = pipe, runtime_state, summarize
        self.record = dict(schema='ditserve.sd35-trace.v1', rank=rank, complete=False,
            observation_only=True, performance_measurements=False, arrays_retained=False,
            hashing_may_synchronize_cuda=True, logical_step_rule='chunk_start + local_index',
            methods={}, schedulers={}, trajectories={})
        self.source_cache = {}
        self.scheduler_cache = {}
        self.active = None
        self.chunk = None

    def _source(self, target):
        target = getattr(target, '__func__', target)
        path = inspect.getsourcefile(target)
        if not path:
            raise ValueError('observer needs inspectable Python implementation source')
        path = str(Path(path).resolve(strict=True))
        if path not in self.source_cache:
            self.source_cache[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return dict(qualified_name=target.__module__ + '.' + target.__qualname__,
                    file=path, file_sha256=self.source_cache[path])

    def _scheduler(self):
        scheduler = self.pipe.scheduler
        step = scheduler.step
        inner = vars(scheduler).get('module')
        configs = dict(actual=dict(scheduler.config), inner=dict(inner.config) if inner is not None else None)
        key = (type(scheduler), getattr(step, '__func__', step), encoded(configs))
        if key not in self.scheduler_cache:
            metadata = dict(type=type(scheduler).__module__ + '.' + type(scheduler).__qualname__,
                class_source=self._source(type(scheduler)), bound_step=self._source(step),
                unwrapped_step=self._source(inspect.unwrap(step)), config=configs['actual'])
            if inner is not None:
                metadata['inner'] = dict(type=type(inner).__module__ + '.' + type(inner).__qualname__,
                    class_source=self._source(type(inner)), bound_step=self._source(inner.step),
                    unwrapped_step=self._source(inspect.unwrap(inner.step)), config=configs['inner'])
            raw = encoded(metadata)
            if len(raw) > 16384:
                raise ValueError('scheduler description exceeds bounded metadata budget')
            identity = hashlib.sha256(raw).hexdigest()
            self.record['schedulers'][identity] = metadata
            self.scheduler_cache[key] = identity
        index = getattr(scheduler, 'step_index', None)
        begin = getattr(scheduler, 'begin_index', None)
        if any(v is not None and type(v) is not int for v in (index, begin)):
            raise ValueError('scheduler indices must be integer metadata, not logical step identifiers')
        return dict(identity=self.scheduler_cache[key], step_index=index, begin_index=begin)

    def _runtime(self):
        state = self.runtime_state()
        row = {name: getattr(state, name, None) for name in RUNTIME_FIELDS}
        row['parallel'] = {name: getattr(state.parallel_config, name, None)
                           for name in ('pp_degree', 'cfg_degree', 'sp_degree', 'dp_degree')}
        # JSON roundtrip detaches lists that the actual runtime might mutate.
        return json.loads(encoded(row))

    def _tensor(self, value):
        row = self.summarize(value)
        if not valid_tensor(row):
            raise ValueError('invalid observer tensor summary')
        return row

    def _check_size(self):
        if len(encoded(self.record)) > MAX_BYTES:
            raise ValueError('trace exceeds fixed one MiB per-rank budget')

    @contextmanager
    def trajectory(self, name):
        if self.active is not None or name not in TRAJECTORIES or name in self.record['trajectories']:
            raise ValueError('invalid or repeated observation trajectory')
        row = dict(complete=False, sync=[], backbone=[], scheduler=[], forbidden=[])
        self.record['trajectories'][name] = row
        self.active = name
        originals = {}
        try:
            for method in METHODS:
                original = getattr(self.pipe, method)
                originals[method] = (method in vars(self.pipe), vars(self.pipe).get(method), original)
                self.record['methods'][method] = self._source(original)
                signature = inspect.signature(original)
                def wrapped(*args, _method=method, _original=original, _signature=signature, **kwargs):
                    bound = _signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    return self._call(_method, _original, args, kwargs, bound.arguments)
                setattr(self.pipe, method, wrapped)
            yield
            if not self._coverage(name, row):
                raise ValueError('incomplete observation or unexpected naive/async dispatch')
            row['complete'] = True
        finally:
            for method, (present, value, unused) in originals.items():
                if present:
                    setattr(self.pipe, method, value)
                else:
                    delattr(self.pipe, method)
            self.active = None
            self.chunk = None
            self._check_size()

    def _call(self, method, original, args, kwargs, values):
        row = self.record['trajectories'][self.active]
        if method in ('_naive_forward', '_async_pipeline'):
            row['forbidden'].append(method)
            return original(*args, **kwargs)
        if method == '_sync_pipeline':
            ends = CHUNK_ENDS if self.active == 'round_sp2_chunked' else [28]
            ordinal = len(row['sync'])
            if self.chunk is not None or ordinal >= len(ends):
                raise ValueError('unexpected sync observation')
            start, end = (ends[ordinal - 1] if ordinal else 0), ends[ordinal]
            if values.get('sync_only') is not True or len(values['timesteps']) != end - start:
                raise ValueError('observed sync interval differs from the unchanged trajectory')
            event = dict(chunk_start=start, chunk_end=end, returned=False,
                inputs={k:self._tensor(values[k]) for k in ('latents', 'prompt_embeds', 'pooled_prompt_embeds', 'timesteps')},
                scheduler=self._scheduler(), runtime=self._runtime())
            event['inputs']['sigmas'] = self._tensor(self.pipe.scheduler.sigmas)
            row['sync'].append(event)
            self.chunk = dict(start=start, end=end, backbone=0, scheduler=0, backbone_returned=0)
            self._check_size()
            try:
                result = original(*args, **kwargs)
                event['returned'] = True
                return result
            finally:
                self.chunk = None
        chunk = self.chunk
        if chunk is None:
            raise ValueError('step observed outside a synchronous chunk')
        kind = 'backbone' if method == '_backbone_forward' else 'scheduler'
        local = chunk[kind]
        if local >= chunk['end'] - chunk['start']:
            raise ValueError('too many observed steps in chunk')
        if ((kind == 'backbone' and chunk['scheduler'] != local)
                or (kind == 'scheduler' and chunk['backbone_returned'] != local + 1)):
            raise ValueError('observed backbone/scheduler ordering differs')
        fields = ('latents', 'encoder_hidden_states', 'pooled_prompt_embeds', 't') if kind == 'backbone' else ('noise_pred', 'latents', 't')
        event = dict(chunk_start=chunk['start'], local_index=local, logical_step=chunk['start'] + local,
            returned=False, inputs={k:self._tensor(values[k]) for k in fields},
            scheduler_before=self._scheduler(), runtime=self._runtime())
        row[kind].append(event)
        chunk[kind] += 1
        self._check_size()
        result = original(*args, **kwargs)
        if kind == 'backbone':
            if not isinstance(result, tuple) or len(result) != 2 or result[1] is not None:
                raise ValueError('SP2/PP1 backbone must return (noise tensor, None)')
            event['output'] = dict(noise_pred=self._tensor(result[0]), encoder_hidden_states=None)
            chunk['backbone_returned'] += 1
        else:
            event['output'] = self._tensor(result)
        event.update(returned=True, scheduler_after=self._scheduler())
        self._check_size()
        return result

    @staticmethod
    def _coverage(name, row):
        ends = CHUNK_ENDS if name == 'round_sp2_chunked' else [28]
        starts = [0] + ends[:-1]
        if row.get('forbidden') != [] or len(row.get('sync', [])) != len(ends):
            return False
        for event, start, end in zip(row['sync'], starts, ends):
            if (event.get('chunk_start') != start or event.get('chunk_end') != end
                    or event.get('returned') is not True
                    or set(event.get('inputs', {})) != {'latents', 'prompt_embeds', 'pooled_prompt_embeds', 'timesteps', 'sigmas'}
                    or not all(valid_tensor(v) for v in event['inputs'].values())
                    or event['inputs']['timesteps']['shape'] != [end - start]
                    or event['inputs']['sigmas']['shape'] != [end - start + 1]):
                return False
        expected = [(start, local, start + local) for start, end in zip(starts, ends) for local in range(end-start)]
        for kind in ('backbone', 'scheduler'):
            if len(row.get(kind, [])) != 28:
                return False
            for event, (start, local, step) in zip(row[kind], expected):
                fields = {'latents', 'encoder_hidden_states', 'pooled_prompt_embeds', 't'} if kind == 'backbone' else {'noise_pred', 'latents', 't'}
                if (event.get('chunk_start') != start or event.get('local_index') != local
                        or event.get('logical_step') != step or event.get('returned') is not True
                        or set(event.get('inputs', {})) != fields
                        or not all(valid_tensor(v) for v in event['inputs'].values())):
                    return False
                output = event.get('output')
                if kind == 'backbone':
                    if not isinstance(output, dict) or set(output) != {'noise_pred', 'encoder_hidden_states'} or output['encoder_hidden_states'] is not None or not valid_tensor(output['noise_pred']):
                        return False
                elif not valid_tensor(output):
                    return False
        for backbone, scheduler in zip(row['backbone'], row['scheduler']):
            if (backbone['inputs']['t'] != scheduler['inputs']['t']
                    or backbone['output']['noise_pred'] != scheduler['inputs']['noise_pred']):
                return False
        return True

    def finish(self):
        self.record['complete'] = (set(self.record['trajectories']) == set(TRAJECTORIES)
            and all(row.get('complete') is True and self._coverage(name, row)
                    for name, row in self.record['trajectories'].items()))
        self.record['complete'] = self.record['complete'] and valid_trace(self.record)
        self._check_size()
        return self.record['complete']

    def write(self, directory, storage):
        self.finish()
        raw = encoded(self.record)
        path = Path(directory).absolute() / ('rank-' + str(self.record['rank']) + '.trace.json')
        if path.parent.resolve(strict=True) != path.parent or path.is_symlink():
            raise ValueError('linked trace output path')
        storage()
        with path.open('xb') as stream:
            stream.write(raw)
        return dict(file=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                    complete=self.record['complete'], performance_measurements=False)


def valid_trace(record):
    def source(row):
        return (isinstance(row, dict) and isinstance(row.get('qualified_name'), str)
            and isinstance(row.get('file'), str) and isinstance(row.get('file_sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', row['file_sha256']) is not None)
    def implementation(row):
        return (isinstance(row, dict) and isinstance(row.get('type'), str)
            and isinstance(row.get('config'), dict)
            and all(source(row.get(k)) for k in ('class_source', 'bound_step', 'unwrapped_step')))
    if (record.get('schema') != 'ditserve.sd35-trace.v1' or record.get('complete') is not True
            or type(record.get('rank')) is not int or record['rank'] not in (0, 1)
            or record.get('observation_only') is not True or record.get('performance_measurements') is not False
            or record.get('arrays_retained') is not False or record.get('hashing_may_synchronize_cuda') is not True
            or set(record.get('methods', {})) != set(METHODS)
            or not all(source(value) for value in record['methods'].values())
            or set(record.get('trajectories', {})) != set(TRAJECTORIES)):
        return False
    schedulers = record.get('schedulers', {})
    if not schedulers:
        return False
    for identity, item in schedulers.items():
        if (identity != hashlib.sha256(encoded(item)).hexdigest() or not implementation(item)
                or ('inner' in item and not implementation(item['inner']))):
            return False
    for name, row in record['trajectories'].items():
        if row.get('complete') is not True or not TraceObserver._coverage(name, row):
            return False
        for kind in ('sync', 'backbone', 'scheduler'):
            for event in row[kind]:
                runtime = event.get('runtime', {})
                if (set(runtime) != set(RUNTIME_FIELDS) | {'parallel'}
                        or runtime.get('parallel', {}).get('pp_degree') != 1
                        or runtime['parallel'].get('cfg_degree') != 1):
                    return False
                keys = ('scheduler',) if kind == 'sync' else ('scheduler_before', 'scheduler_after')
                for key in keys:
                    state = event.get(key, {})
                    if (set(state) != {'identity', 'step_index', 'begin_index'}
                            or state['identity'] not in schedulers
                            or any(state[k] is not None and (type(state[k]) is not int or state[k] < 0)
                                   for k in ('step_index', 'begin_index'))):
                        return False
    return True


def assess_observation(records, directory, reference):
    """All six full output descriptors on both ranks must exactly reproduce r3."""
    reproduced = (len(records) == 2 and {r.get('rank') for r in records} == {0, 1}
        and all(row.get('outputs') == next(r['outputs'] for r in reference['ranks'] if r['rank'] == row['rank'])
                for row in records))
    complete = reproduced
    for row in records:
        receipt = row.get('trace', {})
        path = Path(directory) / ('rank-' + str(row.get('rank')) + '.trace.json')
        try:
            raw = bounded_read(path, MAX_BYTES)
            trace = json.loads(raw)
            good = (receipt.get('file') == path.name and receipt.get('bytes') == len(raw)
                and receipt.get('sha256') == hashlib.sha256(raw).hexdigest()
                and receipt.get('complete') is True and trace.get('complete') is True
                and trace.get('rank') == row['rank'] and valid_trace(trace))
        except (OSError, ValueError, TypeError, KeyError):
            good = False
        complete = complete and good
    return dict(observer_reproduced=bool(reproduced), observation_admissible=bool(complete),
                trace_reference_sha256=REFERENCE_SHA256, performance_measurements=False)
