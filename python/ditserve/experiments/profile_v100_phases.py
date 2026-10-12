#!/usr/bin/env python3
"""独立补测真实 Wan 去噪循环/填充后内部 reset；--plan 不访问 GPU。

仅在既有完整 suite 已结束且成功后运行。每个 arm 使用新进程，完整 warmup
seed1 一次，再对 seed0 的完整真实请求观测阶段边界；不计算额外 E2E 收益。
"""

from ..util import ROOT, read_json, child_environment, archive_run

import argparse
import ast
import contextlib
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

from ditserve.gpu import profiling as p
from . import run_v100 as runner

UUID = "GPU-a058066e-bde9-c534-efb1-d2a68c117681"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def validate_suite(suite_dir, require_finished=True):
    """A running/partial suite cannot authorize another GPU workload."""
    suite_dir = Path(suite_dir).resolve()
    manifest = read_json(suite_dir / "manifest.json")
    if require_finished and (manifest.get("status") != "completed" or not manifest.get("finished_at_utc")):
        raise ValueError("既有 suite 尚未完整结束成功；拒绝启动补测")
    if manifest.get("selected_stages") != ["smoke", "formal"]:
        raise ValueError("补测要求既有完整 smoke+formal suite")
    dtype = manifest.get("selected_dtype")
    if dtype not in ("float16", "float32"):
        raise ValueError("缺少既有 suite 的有效选定 dtype")
    if runner.canonical_gpu_uuid((manifest.get("gpu") or {}).get("uuid")) != runner.canonical_gpu_uuid(UUID):
        raise ValueError("既有 suite GPU UUID 不符合固定 V100")
    plan = manifest.get("fp32_plan") if manifest.get("fp32_rerun") else manifest.get("initial_plan")
    if not isinstance(plan, list) or len(plan) != 14 or any(job.get("dtype") != dtype for job in plan):
        raise ValueError("既有完整 suite 必须包含选定 dtype 的 14 个 arm")
    expected_jobs = {("smoke", "object", 0, arm) for arm in ("none", "cache-dit")}
    expected_jobs.update(("formal", prompt, seed, arm) for prompt in ("object", "motion", "water")
                         for seed in (0, 1) for arm in ("none", "cache-dit"))
    if {(job.get("stage"), job.get("prompt_id"), job.get("seed"), job.get("arm")) for job in plan} != expected_jobs:
        raise ValueError("既有 suite 的 14 个任务身份不完整或重复")
    if require_finished:
        records = read_json(suite_dir / "records.json")
        fields = ("dtype", "stage", "prompt_id", "seed", "arm")
        for job in plan:
            matches = [r for r in records if all(r.get(k) == job.get(k) for k in fields)]
            if len(matches) != 1 or matches[0].get("status") != "completed":
                raise ValueError("既有 suite 有缺失、重复或未完成 arm")
    references = {}
    for arm in ("none", "cache-dit"):
        path = suite_dir / dtype / "formal" / ("object-seed0-" + arm) / "profile.json"
        profile = read_json(path)
        config = profile.get("config", {})
        expected = dict(model="wan", preset="v100-wan", dtype=dtype, cache=arm,
                        height=480, width=832, frames=17, steps=50, seed=0, batch_size=1,
                        warmup=1, repeats=1, device="cuda:0", text_encoder_device="cpu",
                        text_encoder_dtype="float32", vae_dtype="float32", attention="cache-dit-native",
                        local_files_only=True, revision=REVISION, cache_fn=8, cache_bn=0,
                        cache_warmup=8, cache_threshold=0.08, cache_max_continuous=3)
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("既有 object seed0 配置不符合固定协议: " + arm)
        runs = profile.get("runs", [])
        if (profile.get("status") != "completed" or profile.get("measured") is not True or
                len(runs) != 1 or runs[0].get("status") != "completed" or
                runs[0].get("output", {}).get("finite") is not True):
            raise ValueError("既有 object seed0 缺少完整有限输出: " + arm)
        references[arm] = {"path": str(path), "profile": profile}
    allowed = {"cache", "output", "artifact_dir"}
    configs = [references[arm]["profile"]["config"] for arm in ("none", "cache-dit")]
    if any(configs[0].get(k) != configs[1].get(k) for k in set(configs[0]) | set(configs[1]) if k not in allowed):
        raise ValueError("既有两 arm 配置不一致")
    return manifest, references


def validate_device(environment, expected_uuid=UUID):
    target = runner.canonical_gpu_uuid(expected_uuid)
    if (not environment.get("ready") or target is None or
            runner.canonical_gpu_uuid(environment.get("cuda_visible_devices")) != target or
            runner.canonical_gpu_uuid(environment.get("gpu", {}).get("uuid")) != target or
            environment.get("device") != "cuda:0" or
            environment.get("cuda_device_order") != "PCI_BUS_ID"):
        raise p.ProfilingError("补测实际 CUDA 映射/UUID 与既有物理 V100 不符")


def source_contract(diffusers):
    """Pin the exact unique context enclosing Wan's complete denoising loop."""
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(diffusers.WanPipeline.__call__)))
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             isinstance(node.func, ast.Attribute) and node.func.attr == "progress_bar"]
    if (len(calls) != 1 or calls[0].args or len(calls[0].keywords) != 1 or
            calls[0].keywords[0].arg != "total" or
            not isinstance(calls[0].keywords[0].value, ast.Name) or
            calls[0].keywords[0].value.id != "num_inference_steps" or
            not any(isinstance(node, ast.With) and any(item.context_expr is calls[0] for item in node.items)
                    for node in ast.walk(tree))):
        raise p.ProfilingError("Wan progress_bar 源码边界不符合固定 0.36.0 契约")
    return {"wan_call_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "boundary": "unique with self.progress_bar(total=num_inference_steps) body",
            "includes": "all denoising iterations, both CFG forwards, guidance, scheduler.step, progress updates; boundary synchronization",
            "excludes": "prompt encoding, latent/timestep preparation, VAE decode, cache reset, outer cache statistics hooks"}


class LoopObserver:
    def __init__(self, torch, pipe, expected_steps, device="cuda:0", clock=time.perf_counter):
        self.torch, self.pipe, self.expected_steps = torch, pipe, expected_steps
        self.device, self.clock, self.records = device, clock, []

    def __enter__(self):
        self.original = self.pipe.progress_bar
        self.had_instance = "progress_bar" in vars(self.pipe)
        self.old_instance = vars(self.pipe).get("progress_bar")
        observer = self

        @contextlib.contextmanager
        def observed(*args, **kwargs):
            if args or kwargs != {"total": observer.expected_steps} or observer.records:
                raise p.ProfilingError("未知或重复的 Wan progress_bar 调用")
            record = {"status": "running", "wall_ms": None, "progress_updates": 0,
                      "instrumented": True}
            observer.records.append(record)
            with observer.original(*args, **kwargs) as progress:
                if (len(observer.pipe.scheduler.timesteps) != observer.expected_steps or
                        observer.pipe.scheduler.order != 1 or observer.pipe.interrupt):
                    record["status"] = "failed"
                    raise p.ProfilingError("Wan 实际 timesteps/order/interrupt 不符合完整循环契约")
                class Proxy:
                    def update(self, n=1):
                        result = progress.update(n)
                        record["progress_updates"] += n
                        return result

                    def __getattr__(self, name):
                        return getattr(progress, name)
                observer.torch.cuda.synchronize(observer.device)
                start = observer.clock()
                record["started_at_unix_ns"] = time.time_ns()
                record["started_monotonic_ns"] = time.monotonic_ns()
                try:
                    yield Proxy()
                except BaseException:
                    record["status"] = "failed"
                    raise
                else:
                    record["status"] = "failed" if observer.pipe.interrupt else "completed"
                finally:
                    try:
                        observer.torch.cuda.synchronize(observer.device)
                        record["wall_ms"] = (observer.clock() - start) * 1000
                        record["finished_at_unix_ns"] = time.time_ns()
                        record["finished_monotonic_ns"] = time.monotonic_ns()
                    except BaseException:
                        record["status"] = "failed"
                        raise
        self.pipe.progress_bar = observed
        return self

    def __exit__(self, *exc):
        if self.had_instance:
            self.pipe.progress_bar = self.old_instance
        else:
            del self.pipe.progress_bar

    def report(self):
        if (len(self.records) != 1 or self.records[0]["status"] != "completed" or
                self.records[0]["progress_updates"] != self.expected_steps):
            raise p.ProfilingError("完整去噪循环阶段记录缺失、失败或步数不符")
        return self.records[0]


class ResetObserver:
    """Time only real pipeline-triggered resets of previously populated contexts."""
    def __init__(self, torch, pipe, expected_steps, device="cuda:0", clock=time.perf_counter):
        self.torch, self.pipe, self.expected_steps = torch, pipe, expected_steps
        self.device, self.clock, self.records = device, clock, []

    def _before(self, name):
        # The returned record contains primitives only. In particular no old
        # context/tensor references survive into reset_context's clear_buffers.
        context = self.manager.get_context(name)
        return {"context_name": name, "executed_steps": context.executed_steps,
                "transformer_executed_steps": context.transformer_executed_steps,
                "buffer_inventory": p.tensor_inventory(context.buffers, self.torch.is_tensor)}

    def __enter__(self):
        self.manager = self.pipe._context_manager
        if self.manager is not self.pipe.transformer._context_manager or self.manager.persistent_context:
            raise p.ProfilingError("需要真实 pipeline 与 transformer 共享的非持久 context manager")
        self.names = set(self.manager._cached_context_manager)
        if not self.names:
            raise p.ProfilingError("warmup 后没有已填充 context")
        self.original = self.manager.reset_context
        self.had_instance = "reset_context" in vars(self.manager)
        self.old_instance = vars(self.manager).get("reset_context")

        def observed(cached_context, *args, **kwargs):
            # Match upstream reset_context(cached_context, **context_kwargs):
            # its kwargs also contain name=... for the replacement context.
            name = cached_context
            if not isinstance(name, str) or name not in self.names or any(r["context_name"] == name for r in self.records):
                raise p.ProfilingError("未知、重复或非名称形式的内部 reset")
            before = self._before(name)
            tensors = before["buffer_inventory"]["tensors"]
            record = {"context_name": name, "manager_id": id(self.manager), "status": "running",
                      "before": before, "call_wall_ms": None, "synchronized_wall_ms": None,
                      "post_call_synchronize_ms": None}
            self.records.append(record)
            if (before["executed_steps"] != self.expected_steps or
                    before["transformer_executed_steps"] != 2 * self.expected_steps or
                    not any(t["device"] == self.device and t["bytes"] > 0 for t in tensors)):
                record["status"] = "rejected_unpopulated_or_incomplete"
                raise p.ProfilingError("内部 reset 前没有完整 warmup 的 CUDA tensor/步骤证据")
            self.torch.cuda.synchronize(self.device)
            record["started_monotonic_ns"] = time.monotonic_ns()
            record["started_at_unix_ns"] = time.time_ns()
            start = self.clock()
            try:
                result = self.original(name, *args, **kwargs)
                returned = self.clock()
                record["call_wall_ms"] = (returned - start) * 1000
                self.torch.cuda.synchronize(self.device)
                synchronized = self.clock()
                record["synchronized_wall_ms"] = (synchronized - start) * 1000
                record["post_call_synchronize_ms"] = (synchronized - returned) * 1000
                record["finished_monotonic_ns"] = time.monotonic_ns()
                record["finished_at_unix_ns"] = time.time_ns()
                record["status"] = "completed"
                return result
            except BaseException:
                record["status"] = "failed"
                raise
        # Patch one manager instance once; transformer/block summary aliases
        # never generate extra resets or timings.
        self.manager.reset_context = observed
        return self

    def __exit__(self, *exc):
        if self.had_instance:
            self.manager.reset_context = self.old_instance
        else:
            del self.manager.reset_context

    def report(self):
        if ({r["context_name"] for r in self.records} != self.names or
                len(self.records) != len(self.names) or any(r["status"] != "completed" for r in self.records)):
            raise p.ProfilingError("真实已填充内部 reset 记录不完整")
        return {"status": "measured_populated_pipeline_reset", "instrumented": True,
                "unique_manager_count": 1, "context_count": len(self.names), "calls": self.records,
                "call_wall_ms_sum": sum(r["call_wall_ms"] for r in self.records),
                "synchronized_wall_ms_sum": sum(r["synchronized_wall_ms"] for r in self.records),
                "scope": "actual internal reset_context calls only; excludes enter_context/ExitStack and stats hooks; synchronization overhead separately recorded",
                "buffer_inventory_is_complete_migratable_state": False}


def worker(args, manifest, reference):
    output = args.output / args.worker_arm
    output.mkdir(parents=True, exist_ok=False)
    report = {"kind": "gpu_phase_supplement", "schema_version": "ditserve.v100-phases.v1",
              "status": "running", "measured": False, "instrumented": True,
              "arm": args.worker_arm, "reference_profile": reference["path"],
              "scope": "one full warmup + one instrumented full request; no new E2E performance comparison",
              "research_gate_eligible": False, "perceptual_quality_validated": False}
    loop = reset = telemetry = None
    try:
        runner.resolve_gpu(UUID)
        if any(os.environ.get(key) != value for key, value in {
                "CUDA_VISIBLE_DEVICES": UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}.items()):
            raise p.ProfilingError("补测进程设备绑定/CPU 线程环境与既有正式配置不符")
        report["process_environment"] = {key: os.environ[key] for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        telemetry = runner.GpuTelemetry(output, UUID, 3.0)
        telemetry.start()
        config = dict(reference["profile"]["config"])
        config.update(output=output / "profile.json", artifact_dir=output / "artifacts")
        profile_args = argparse.Namespace(**config)
        report["config"] = p.json_safe(config)
        environment = p.preflight(profile_args)
        report["environment"] = environment
        validate_device(environment)
        expected_versions = reference["profile"]["environment"]["packages"]
        if environment["packages"] != expected_versions or environment["packages"].get("diffusers") != "0.36.0" or environment["packages"].get("cache-dit") != "1.5.0":
            raise p.ProfilingError("补测依赖版本与已验证正式测量不一致")
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        report["denoising_source_contract"] = source_contract(diffusers)
        torch.cuda.set_device(profile_args.device)
        pipe = p._pipeline(torch, diffusers, profile_args)
        if str(pipe.device) != "cuda:0" or str(pipe._execution_device) != "cuda:0":
            raise p.ProfilingError("实际 pipeline 执行设备错误")
        backend = p.CacheBackend(pipe, profile_args)
        if args.worker_arm == "cache-dit":
            reset_source = inspect.getsource(type(pipe._context_manager).reset_context)
            report["reset_source_contract"] = {
                "implementation": "cache_dit CachedContextManager.reset_context on actual pipeline manager",
                "source_sha256": hashlib.sha256(reset_source.encode()).hexdigest(),
                "boundary": "real internal reset_context(name, **kwargs) clearing populated old buffers and creating the next context",
                "not_measured": "DBCache residual recomputation, complete state migration, allocator reserved-memory release"}
        report["components"] = p._component_metadata(pipe)
        report["cache_config"] = p.json_safe(backend.config)
        report["revision"] = p._revision_metadata(profile_args, report["components"])
        report["scheduler"] = {"class": type(pipe.scheduler).__name__, "config": p.json_safe(dict(pipe.scheduler.config))}
        prepare = lambda kwargs: p._prepare_prompt_kwargs(torch, pipe, profile_args, kwargs)
        with torch.inference_mode(), p._sdpa_math_context(torch):
            print("完整 warmup seed1；填充真实缓存", flush=True)
            warmup = report["warmup"] = {"seed": 1, "status": "running"}
            value, timing = p.timed_call(torch, pipe, p._call_kwargs(torch, profile_args, 1), "cuda:0", prepare_kwargs=prepare)
            warmup["timing_context_only"] = timing
            p._record_output(value, profile_args, warmup, "warmup-seed1")
            warmup["status"] = "completed"
            warmup["cache_statistics"] = backend.stats()
            del value
            print("完整 phase 诊断 seed0；记录 loop 与真实内部 reset", flush=True)
            phase = report["phase_run"] = {"seed": 0, "status": "running"}
            loop = LoopObserver(torch, pipe, 50)
            reset = ResetObserver(torch, pipe, 50) if args.worker_arm == "cache-dit" else None
            with loop, (reset if reset is not None else contextlib.nullcontext()):
                value, timing = p.timed_call(torch, pipe, p._call_kwargs(torch, profile_args, 0), "cuda:0", prepare_kwargs=prepare)
            phase["instrumented_pipeline_timing_context_only"] = timing
            report["measured"] = True
            p._record_output(value, profile_args, phase, "phase-seed0")
            del value
            phase["denoising_loop"] = loop.report()
            phase["internal_cache_reset"] = reset.report() if reset else {"status": "not_applicable", "calls": [], "call_wall_ms_sum": None}
            if reset is not None:
                if not all(r["finished_monotonic_ns"] <= phase["denoising_loop"]["started_monotonic_ns"] for r in reset.records):
                    raise p.ProfilingError("观测到 loop 内部或之后的 reset，不能标记为请求入口重置")
                phase["internal_cache_reset"]["all_resets_precede_denoising_loop"] = True
            phase["cache_statistics"] = backend.stats()
            reference_hash = reference["profile"]["runs"][0]["output"]["sha256"]
            phase["output_vs_existing_same_arm"] = {"reference_sha256": reference_hash,
                "phase_sha256": phase["output"]["sha256"],
                "exact_match": phase["output"]["sha256"] == reference_hash,
                "interpretation": "instrumentation repeatability evidence only; mismatch retained, not a perceptual quality verdict"}
            phase["status"] = "completed"
            report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc),
                      "category": getattr(exc, "category", "phase_error"), "details": p.json_safe(getattr(exc, "details", {}))})
        if loop is not None:
            report["partial_loop_records"] = loop.records
        if reset is not None:
            report["partial_reset_records"] = reset.records
    finally:
        if telemetry is not None:
            try:
                report["telemetry"] = telemetry.stop()
            except Exception as exc:
                report["telemetry"] = {"status": "failed", "type": type(exc).__name__, "message": str(exc)}
        report["telemetry_complete"] = (report.get("telemetry") or {}).get("status") == "completed"
        report["finished_at_utc"] = runner.utc_now()
        runner.write_json(output / "profile.json", report)
    return 0 if report["status"] == "completed" and report["telemetry_complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=ROOT / ".cache/runs/v100/paired")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/runs/v100/phases")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--worker-arm", choices=("none", "cache-dit"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.suite_dir, args.output = args.suite_dir.resolve(), args.output.resolve()
    try:
        manifest, references = validate_suite(args.suite_dir, require_finished=not args.plan)
        plan = {"kind": "phase_supplement_plan", "suite_status": manifest["status"],
                "execution_requires_finished_full_suite": True, "gpu_uuid": UUID,
                "dtype": manifest["selected_dtype"], "arms_in_order": ["none", "cache-dit"],
                "fresh_process_per_arm": True, "full_calls_per_arm": 2,
                "warmup_seed": 1, "phase_seed": 0, "shape": [480, 832, 17], "steps": 50,
                "references": {arm: ref["path"] for arm, ref in references.items()},
                "output": str(args.output), "instrumented": True, "new_e2e_speedup_computed": False}
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        if args.worker_arm:
            return worker(args, manifest, references[args.worker_arm])
        runner.resolve_gpu(UUID)
        args.output.mkdir(parents=True, exist_ok=False)
        plan.update(status="running", started_at_utc=runner.utc_now(), workers=[])
        runner.write_json(args.output / "manifest.json", plan)
        env = child_environment()
        env.update(CUDA_VISIBLE_DEVICES=UUID, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="8",
                   MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1")
        try:
            for arm in ("none", "cache-dit"):
                validate_suite(args.suite_dir)
                runner.resolve_gpu(UUID)
                command = [args.python, "-m", "ditserve.experiments.profile_v100_phases", "--suite-dir", str(args.suite_dir),
                           "--output", str(args.output), "--worker-arm", arm]
                with (args.output / (arm + ".log")).open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                plan["workers"].append({"arm": arm, "returncode": result.returncode, "command": command})
                runner.write_json(args.output / "manifest.json", plan)
            plan["status"] = "completed" if all(x["returncode"] == 0 for x in plan["workers"]) else "failed"
        except BaseException as exc:
            plan.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
        finally:
            plan["finished_at_utc"] = runner.utc_now()
            runner.write_json(args.output / "manifest.json", plan)
        archive_run(args.output, "wan-phases", args.results_root)
        return 0 if plan["status"] == "completed" else 2
    except (OSError, ValueError, p.ProfilingError) as exc:
        print("拒绝或无法执行补测: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
