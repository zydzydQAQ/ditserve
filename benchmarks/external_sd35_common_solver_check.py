"""Experimental normal-SP2 raw-solver selection; native defaults stay unchanged.

This restores scheduler object/attribute identity only. Normal inference still
updates the selected module's ordinary schedule and step state.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json

from external_sd35_trace import NATIVE_SOLVER_SHA256, RAW_SOLVER_SHA256

WRAPPER_TYPE = 'xfuser.model_executor.schedulers.scheduling_flow_match_euler_discrete.xFuserFlowMatchEulerDiscreteSchedulerWrapper'
RAW_TYPE = 'diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler'
BOUND_WRAPPER_SHA256 = '570aa9cc5806b6dfc01eb66228deed5c41ffb0465947e27cfd42534493a58897'
RECIPE = 'normal_sp2_uses_original_scheduler_module; round_paths_unchanged'


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def source(method):
    from pathlib import Path
    method = getattr(method, '__func__', method)
    path = Path(inspect.getsourcefile(method)).resolve(strict=True)
    return dict(qualified_name=method.__module__+'.'+method.__qualname__, file=str(path),
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def describe_solver(scheduler):
    raw = vars(scheduler).get('module')
    if raw is None:
        raise ValueError('common solver selection requires the existing native wrapper module')
    configuration = dict(scheduler.config)
    if configuration != dict(raw.config) or configuration.get('stochastic_sampling') is not False:
        raise ValueError('actual wrapper/raw deterministic solver configuration differs')
    row = dict(previous_type=type(scheduler).__module__+'.'+type(scheduler).__qualname__,
        selected_type=type(raw).__module__+'.'+type(raw).__qualname__,
        previous_bound_step=source(scheduler.step), previous_unwrapped_step=source(inspect.unwrap(scheduler.step)),
        selected_step=source(raw.step), configuration=configuration, configuration_sha256=canonical(configuration))
    if not valid_description(row):
        raise ValueError('common solver source differs from the frozen r4 implementations')
    return row


def valid_description(row):
    try:
        return (row['previous_type'] == WRAPPER_TYPE and row['selected_type'] == RAW_TYPE
            and row['previous_bound_step']['file_sha256'] == BOUND_WRAPPER_SHA256
            and row['previous_unwrapped_step']['file_sha256'] == NATIVE_SOLVER_SHA256
            and row['selected_step']['file_sha256'] == RAW_SOLVER_SHA256
            and row['configuration'].get('stochastic_sampling') is False
            and row['configuration_sha256'] == canonical(row['configuration']))
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def valid_runtime(row):
    return (isinstance(row,dict) and set(row) == {'pp_degree','cfg_degree','sp_degree','dp_degree','patch_mode'}
        and all(type(row[k]) is int for k in ('pp_degree','cfg_degree','sp_degree','dp_degree'))
        and row == dict(pp_degree=1,cfg_degree=1,sp_degree=2,dp_degree=1,patch_mode=False)
        and row['patch_mode'] is False)


def runtime_description(state):
    row = {name:getattr(state.parallel_config,name) for name in ('pp_degree','cfg_degree','sp_degree','dp_degree')}
    row['patch_mode'] = state.patch_mode
    if not valid_runtime(row):
        raise ValueError('common solver requires actual SP2/DP1/PP1/CFG1 and non-patched execution')
    return row


@contextmanager
def select(pipe, runtime_state):
    """Use the original raw module only within this normal-pipeline invocation."""
    previous = pipe.scheduler
    description = describe_solver(previous)
    raw = vars(previous)['module']
    pipe_present, pipe_value = 'scheduler' in vars(pipe), vars(pipe).get('scheduler')
    step_present, step_value = 'step' in vars(raw), vars(raw).get('step')
    step = raw.step
    row = dict(schema='ditserve.sd35-common-solver.v1', recipe=RECIPE, complete=False,
        solver=description, selected_existing_module=True, new_scheduler_constructed=False,
        scheduler_state_restored=False, pipeline_scheduler_identity_restored=False,
        raw_step_attribute_restored=False, started_calls=0, returned_calls=0,
        runtime=runtime_description(runtime_state()), qualification_eligible=False)
    def counted(*args, **kwargs):
        if runtime_description(runtime_state()) != row['runtime']:
            raise ValueError('common solver runtime changed during the normal trajectory')
        row['started_calls'] += 1
        result = step(*args, **kwargs)
        row['returned_calls'] += 1
        return result
    try:
        raw.step = counted
        pipe.scheduler = raw
        yield row
        if pipe.scheduler is not raw or raw.step is not counted:
            raise ValueError('normal pipeline replaced the selected common solver')
        if row['started_calls'] != 28 or row['returned_calls'] != 28:
            raise ValueError('common solver must execute exactly 28 actual steps')
        if dict(raw.config) != description['configuration']:
            raise ValueError('common solver configuration changed during inference')
        row['complete'] = True
    finally:
        if step_present:
            raw.step = step_value
        else:
            delattr(raw,'step')
        if pipe_present:
            pipe.scheduler = pipe_value
        else:
            delattr(pipe,'scheduler')
        row['pipeline_scheduler_identity_restored'] = (pipe.scheduler is previous
            and ('scheduler' in vars(pipe)) == pipe_present
            and (not pipe_present or vars(pipe)['scheduler'] is pipe_value))
        row['raw_step_attribute_restored'] = (('step' in vars(raw)) == step_present
            and (not step_present or vars(raw)['step'] is step_value))
        row['complete'] = (row['complete'] and row['pipeline_scheduler_identity_restored']
                           and row['raw_step_attribute_restored'])


def valid_receipt(row):
    return (row.get('schema') == 'ditserve.sd35-common-solver.v1' and row.get('recipe') == RECIPE
        and row.get('complete') is True and valid_description(row.get('solver',{}))
        and row.get('selected_existing_module') is True and row.get('new_scheduler_constructed') is False
        and row.get('scheduler_state_restored') is False and row.get('pipeline_scheduler_identity_restored') is True
        and row.get('raw_step_attribute_restored') is True and row.get('qualification_eligible') is False
        and type(row.get('started_calls')) is int and row['started_calls'] == 28
        and type(row.get('returned_calls')) is int and row['returned_calls'] == 28
        and valid_runtime(row.get('runtime')))


def assess(records, completed):
    """Only within-SP2 normal/common/round equivalence, never SP1 or paper qualification."""
    valid = (completed is True and len(records) == 2 and {r.get('rank') for r in records} == {0,1}
        and all(valid_receipt(r.get('common_solver',{})) for r in records))
    names = {'normal_sp2','round_sp2_full','round_sp2_chunked'}
    equivalent = valid and all(set(r.get('outputs',{})) == names
        and all(set(value) == {'latents','decoded'} for value in r['outputs'].values())
        and r['outputs']['normal_sp2'] == r['outputs']['round_sp2_full'] == r['outputs']['round_sp2_chunked']
        for r in records)
    return dict(common_solver_diagnostic_completed=bool(valid), common_solver_equivalence_passed=bool(equivalent),
        common_solver_scope='normal raw solver versus full/chunk round at fixed SP2; SP1 comparisons remain separate',
        qualification_eligible=False)
