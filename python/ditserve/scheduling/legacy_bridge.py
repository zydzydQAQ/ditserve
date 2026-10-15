"""Standalone bridge: run this file with the *historical* ditserve PYTHONPATH.

No scheduling package is imported. The subprocess remains in the supervisor's
charged process group and uses only archived model and D1 execution modules.
"""

import argparse
import gc
import importlib.metadata
import os
from pathlib import Path
import sys
import time
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--deadline", required=True, type=float)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--process-group", required=True, type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("legacy bridge output already exists")
    # These resolve from the archived PYTHONPATH, never this bridge's package.
    from ditserve.research.common import atomic_json, digest, file_digest, read_json
    from ditserve.directions import direction1 as d1
    from ditserve.directions.budget import BudgetStop
    from ditserve.directions.block_executor import WanBlockExecutor
    from ditserve.directions.sana import SanaAdapter
    from ditserve.gpu.worker import WanStepWorker

    plan = read_json(args.plan)
    source = plan["source"]
    archive = Path(source["archived_execution_root"]).resolve()
    data = dict(schema_version="ditserve.scheduling.legacy-bridge.v1", status="running", records=[],
                plan_sha256=plan["sha256"], source=source, started_at=time.time())
    worker = executor = sana = None
    code = 1

    def save():
        atomic_json(args.output, data)

    def check():
        if os.getppid() != args.parent_pid or os.getpgrp() != args.process_group:
            raise BudgetStop("legacy bridge lost its charged parent or process group")
        if time.monotonic() >= args.deadline:
            raise BudgetStop("legacy bridge charged deadline reached")

    try:
        check()
        if plan["sha256"] != digest({key: value for key, value in plan.items() if key != "sha256"}):
            raise ValueError("legacy bridge plan digest mismatch")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != source["device_uuid"]:
            raise ValueError("legacy bridge hardware visibility changed")
        manifest_path = archive / "manifest.json"
        if file_digest(manifest_path) != source["manifest_sha256"]:
            raise ValueError("legacy bridge archived manifest changed")
        manifest = read_json(manifest_path)
        for relative, expected in manifest["source_files"].items():
            if file_digest(archive / relative) != expected:
                raise ValueError("legacy bridge archived file changed: " + relative)
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name.startswith("ditserve") and path and archive not in Path(path).resolve().parents:
                raise ValueError("legacy bridge imported current source: " + name)
        dependencies = {name: importlib.metadata.version(name) for name in source["dependencies"]}
        if dependencies != source["dependencies"]:
            raise ValueError("legacy bridge dependency versions differ")
        data["dependencies"] = dependencies
        save()
        config, opts = plan["config"], plan["options"]
        worker = WanStepWorker(device="cuda:0", model_path=d1._model_path(config, "wan"),
                               attention=opts.get("attention", config.get("attention", "flash")))
        executor = WanBlockExecutor(worker)
        check()
        if "A100" not in worker.torch.cuda.get_device_name("cuda:0").upper():
            raise ValueError("legacy bridge requires A100 hardware")
        sana = SanaAdapter(d1._model_path(config, "sana"), device=worker.device)
        check()
        for index in range(opts.get("online_image_warmups", 2)):
            sana.generate(opts["prompts"][0], seed=opts["seeds"][0] + index, check_budget=check)
        # Match the archived online warmup, without replacing frozen service or
        # trace deadlines with today's measurement.
        reference = worker.reference(d1._spec(opts, plan["profile"]["selected_shape"], "legacy-online-warmup"))
        data["warmup_reference_ms"] = reference["elapsed_ms"]
        del reference
        check()
        save()
        for policy in plan["policies"]:
            check()
            def partial(rows, elapsed):
                data["active_trace"] = dict(policy=policy, trace_sha256=plan["trace_sha256"],
                                            complete=False, rows=rows, elapsed_ms=elapsed)
                save()
            replay = d1._run_trace(worker, executor, sana, opts, plan["profile"], plan["target_requests"],
                                   policy, check, partial)
            data.pop("active_trace", None)
            data["records"].append(dict(plan["target"], policy=policy, trace_sha256=plan["trace_sha256"],
                source_sha256=source["source_sha256"], config_sha256=source["config_sha256"],
                device_identity=dict(uuid=source["device_uuid"], name=worker.torch.cuda.get_device_name("cuda:0")),
                dependencies=dependencies, **replay))
            save()
        data.update(status="complete", complete=True,
                    quality_scope="historical D1 correctness prerequisite; archived online replays do not retain generated arrays")
        code = 0
    except BaseException as error:
        data.update(status="incomplete" if isinstance(error, BudgetStop) else "failed",
                    reason=str(error), error_type=type(error).__name__, traceback=traceback.format_exc(),
                    budget_stopped=isinstance(error, BudgetStop))
        code = 2 if isinstance(error, BudgetStop) else 1
    finally:
        if worker is not None:
            if executor is not None:
                for continuation in list(executor._states.values()):
                    if not continuation.closed:
                        executor.close(continuation)
            for state in list(worker._states.values()):
                worker.release(state)
        if sana is not None:
            sana.close()
        if worker is not None:
            worker.torch.cuda.synchronize()
            worker.torch.cuda.empty_cache()
        gc.collect()
        data["finished_at"] = time.time()
        save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
