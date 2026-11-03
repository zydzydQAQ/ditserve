"""Execute one source-bound external diagnostic with a storage/time watchdog.

Only committed, manifested adapters are executed. Source/model validation is
performed again by the actual child. This wrapper never grants performance or
quality qualification, even when a native request/collective completes.
"""
import json
import os
from pathlib import Path
import re
import sys

import external_runtime_probe as probe


def main():
    initial = probe.storage()
    expected = int(os.environ['DITSERVE_EXTERNAL_GPUS'])
    if expected not in (1,2,8):
        raise ValueError('unsupported diagnostic allocation')
    root, shared, plan, binding = probe.bind_sources('external_task_runner.py', expected)
    if (plan.get('schema') != 'ditserve.external-task-plan.v1'
            or plan.get('gpus') != expected or type(plan.get('timeout_seconds')) is not int
            or not 30 <= plan['timeout_seconds'] <= 14400):
        raise ValueError('invalid source-bound diagnostic plan')
    run_id = os.environ['DITSERVE_RUN_ID']
    if not re.fullmatch('[a-z][a-z0-9-]{1,51}', run_id):
        raise ValueError('invalid run identity')
    import torch
    if torch.cuda.device_count() != expected:
        raise ValueError('actual GPU visibility differs from submitted allocation')
    output = probe.OUTPUT/'external-tasks'/run_id
    output.mkdir(parents=True,exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(output/'cache/hf'),XDG_CACHE_HOME=str(output/'cache/xdg'),
        TORCH_HOME=str(output/'cache/torch'),TORCHINDUCTOR_CACHE_DIR=str(output/'cache/inductor'),
        TRITON_CACHE_DIR=str(output/'cache/triton'),MPLCONFIGDIR=str(output/'cache/matplotlib'),
        OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(shared),DITSERVE_PACKAGE_ROOT=str(shared))
    mapping={'python':sys.executable,'package':str(root),'output':str(output/'result'),
             'shared_python':str(shared)}
    argv=[part.format(**mapping) for part in plan['argv']]
    result=probe.run_command(argv,plan['timeout_seconds'],env)
    probe.storage()
    child=output/'result/summary.json'
    if child.is_file():
        result['child_summary_path']=str(child)
        result['child_summary_sha256']=probe.sha(child)
        result['child_summary']=json.loads(child.read_bytes())
    else:
        result['child_summary_missing']=True
    summary={'schema':'ditserve.external-task.v1','run_id':run_id,'binding':binding,
        'allocation_gpus':expected,'initial_storage':initial,'final_storage':probe.storage(),
        'task':plan['task'],'execution':result,'qualification_eligible':False,'allows':[],
        'scope':'Native implementation diagnostic; not full-system serving or quality qualification.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (output/'summary.json').chmod(0o644)
    print(json.dumps({'summary':str(output/'summary.json'),'sha256':probe.sha(output/'summary.json'),
        'returncode':result['returncode'],'timed_out':result['timed_out'],
        'child_summary_present':child.is_file(),'qualification_eligible':False}),flush=True)
    return 0 if result['returncode']==0 and not result['timed_out'] and child.is_file() else 1

if __name__=='__main__':
    raise SystemExit(main())
