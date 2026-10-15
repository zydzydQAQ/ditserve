#!/usr/bin/env python3
"""V100 单卡 Wan 缓存对照编排；实际测量只由 ditserve profile 执行。

固定物理 GPU 1，以 UUID 绑定 CUDA_VISIBLE_DEVICES，子进程使用 cuda:0。
每个 arm 使用独立进程，保留 profiler JSON、原始日志和输出数组。
--dry-run 只打印实验计划，不查询 GPU、不创建文件、不启动 profiler。
"""

from ..util import ROOT, CONFIGS, atomic_json as write_json, utc_now, canonical_gpu_uuid, child_environment, archive_run

import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


DEFAULT_CONFIG = CONFIGS / "v100.json"
NONFINITE_CATEGORIES = {"nonfinite_output", "nonfinite_embeddings", "nonfinite_latents", "nonfinite_preflight"}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal", "all"), default="all")
    parser.add_argument("--python", default=sys.executable, help="运行 profiler 的 Python 解释器")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="新建结果目录；拒绝覆盖已有非空目录")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--gpu-uuid", help="可显式指定 GPU UUID；执行前核对它确实属于物理 GPU 1")
    parser.add_argument("--model-id", help="兼容的本地 checkpoint 或 Hugging Face 模型 ID")
    parser.add_argument("--revision", help="模型 commit hash；建议显式固定")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--timeout-seconds", type=float, help="每个 profiler 进程的超时；默认不设限制")
    parser.add_argument("--telemetry-interval-seconds", type=float, default=3.0,
                        help="GPU 采样间隔，允许 2–5 秒；默认 3 秒")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不访问 GPU 或写入文件")
    parser.add_argument("--no-export-media", action="store_true", help="保留 npy，但不额外导出 mp4 和配对 contact sheet")
    return parser


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("physical_gpu_index") != 1 or config.get("preset") != "v100-wan":
        raise ValueError("本协议固定 physical_gpu_index=1、preset=v100-wan")
    if config.get("arms") != ["none", "cache-dit"] or config.get("warmup") != 1 or config.get("repeats") != 1:
        raise ValueError("本协议固定两 arm、warmup=1、repeats=1")
    prompts = config.get("prompts", [])
    if [prompt.get("id") for prompt in prompts] != ["object", "motion", "water"]:
        raise ValueError("需要固定 object、motion、water 三类 prompt")
    if any(not isinstance(prompt.get("text"), str) or not prompt["text"].strip() for prompt in prompts):
        raise ValueError("prompt 不得为空")
    expected = {
        "smoke": (256, 256, 17, 12, ["object"], [0]),
        "formal": (480, 832, 17, 50, ["object", "motion", "water"], [0, 1]),
    }
    for stage, values in expected.items():
        spec = config["stages"][stage]
        actual = tuple(spec[key] for key in ("height", "width", "frames", "steps", "prompt_ids", "seeds"))
        if actual != values:
            raise ValueError("%s 的形状、步数、prompt 和 seed 必须符合固定协议" % stage)
    cache = config["cache"]
    for key in ("fn", "max_continuous"):
        if type(cache[key]) is not int or cache[key] < 1:
            raise ValueError("cache %s 必须是正整数" % key)
    for key in ("bn", "warmup"):
        if type(cache[key]) is not int or cache[key] < 0:
            raise ValueError("cache %s 必须是非负整数" % key)
    threshold = cache["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("cache threshold 必须是有限非负数")
    return config


def selected_stages(stage):
    return ["smoke", "formal"] if stage == "all" else [stage]


def resolve_gpu(expected_uuid=None):
    """Never fall back from physical GPU 1 to another visible device."""
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    candidates = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 5:
            continue
        index, uuid, name, pci_bus_id, memory = [value.strip() for value in row]
        if index == "1":
            candidates.append({"physical_index": 1, "uuid": uuid, "name": name,
                               "pci_bus_id": pci_bus_id, "memory_total_mib": float(memory)})
    if len(candidates) != 1:
        raise ValueError("无法唯一找到物理 GPU 1，拒绝选择其他 GPU")
    gpu = candidates[0]
    if "V100" not in gpu["name"]:
        raise ValueError("物理 GPU 1 不是 V100: " + gpu["name"])
    if expected_uuid and gpu["uuid"] != expected_uuid:
        raise ValueError("指定 UUID 与物理 GPU 1 不一致")
    gpu["selection_command"] = command
    return gpu


def build_plan(args, config, dtype, output):
    prompts = {prompt["id"]: prompt["text"] for prompt in config["prompts"]}
    jobs = []
    for stage in selected_stages(args.stage):
        shape = config["stages"][stage]
        pair_index = 0
        for prompt_id in shape["prompt_ids"]:
            for seed in shape["seeds"]:
                # Alternate within-pair order to reduce systematic arm-order bias.
                arms = config["arms"] if pair_index % 2 == 0 else list(reversed(config["arms"]))
                for order, arm in enumerate(arms):
                    case_id = "%s-seed%d-%s" % (prompt_id, seed, arm)
                    directory = output / dtype / stage / case_id
                    command = [args.python, "-m", "ditserve", "profile", "--preset", "v100-wan",
                               "--device", "cuda:0", "--model", "wan", "--dtype", dtype,
                               "--model-id", args.model_id or config["model_id"],
                               "--cache", arm, "--attention", "cache-dit-native", "--batch-size", "1",
                               "--warmup", "1", "--repeats", "1", "--seed", str(seed),
                               "--prompt", prompts[prompt_id], "--output", str(directory / "profile.json"),
                               "--artifact-dir", str(directory / "artifacts")]
                    for name in ("height", "width", "frames", "steps"):
                        command.extend(["--" + name, str(shape[name])])
                    for name, value in config["cache"].items():
                        command.extend(["--cache-" + name.replace("_", "-"), str(value)])
                    if args.revision:
                        command.extend(["--revision", args.revision])
                    if args.local_files_only:
                        command.append("--local-files-only")
                    jobs.append({"stage": stage, "dtype": dtype, "prompt_id": prompt_id,
                                 "prompt": prompts[prompt_id], "seed": seed, "arm": arm,
                                 "pair_index": pair_index, "arm_order": order,
                                 "shape": {name: shape[name] for name in ("height", "width", "frames", "steps")},
                                 "directory": str(directory), "command": command})
                pair_index += 1
    return jobs


def classify_result(returncode, profile, log_text):
    failure = profile.get("failure", {}) if isinstance(profile, dict) else {}
    category = failure.get("category", "") if isinstance(failure, dict) else ""
    if category in NONFINITE_CATEGORIES:
        return "nonfinite", category
    if category == "cuda_oom":
        return "oom", category
    lowered = log_text.lower()
    if returncode != 0 and any(marker in lowered for marker in tuple(NONFINITE_CATEGORIES) + ("non-finite output", "nonfinite output")):
        return "nonfinite", category or "nonfinite_log_marker"
    if returncode != 0 and ("out of memory" in lowered or "cuda_oom" in lowered):
        return "oom", category or "oom_log_marker"
    if returncode != 0 or not isinstance(profile, dict) or profile.get("status") == "failed":
        return "failed", category or "profiler_failed"
    if profile.get("measured") is not True or profile.get("kind") != "gpu_profile" or not profile.get("runs"):
        return "failed", "missing_measured_profile"
    return "completed", None


class GpuTelemetry:
    """Bounded target-only queries; no persistent nvidia-smi child to orphan."""

    FIELDS = ("utilization_gpu_percent", "utilization_memory_percent", "temperature_gpu_c",
              "power_draw_w", "memory_used_mib", "memory_total_mib")

    def __init__(self, directory, gpu_uuid, interval_seconds=3.0):
        if canonical_gpu_uuid(gpu_uuid) is None:
            raise ValueError("GPU telemetry requires a full GPU UUID")
        if not math.isfinite(interval_seconds) or not 2 <= interval_seconds <= 5:
            raise ValueError("GPU telemetry interval must be between 2 and 5 seconds")
        self.path = Path(directory) / "gpu-telemetry.jsonl"
        self.gpu_uuid = gpu_uuid
        self.interval_seconds = interval_seconds
        self.command = ["nvidia-smi", "--id=" + gpu_uuid,
                        "--query-gpu=uuid,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total",
                        "--format=csv,noheader,nounits"]
        self._stop = threading.Event()
        self._thread = None
        self._samples = []
        self._thread_error = None
        self.started_at_utc = None
        self.finished_at_utc = None

    def sample(self):
        sample = {"query_started_at_utc": utc_now(), "query_started_unix_ns": time.time_ns(),
                  "query_started_monotonic_ns": time.monotonic_ns(), "requested_gpu_uuid": self.gpu_uuid}
        try:
            result = subprocess.run(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, timeout=2.0)
            sample.update(returncode=result.returncode, raw_stdout=result.stdout, raw_stderr=result.stderr)
            if result.returncode != 0:
                raise ValueError("nvidia-smi exited with code %s" % result.returncode)
            rows = list(csv.reader(result.stdout.strip().splitlines()))
            if len(rows) != 1 or len(rows[0]) != 7:
                raise ValueError("expected exactly one target GPU telemetry row")
            values = [value.strip() for value in rows[0]]
            if canonical_gpu_uuid(values[0]) != canonical_gpu_uuid(self.gpu_uuid):
                raise ValueError("telemetry reported an unexpected GPU UUID")
            sample["reported_gpu_uuid"] = values[0]
            unavailable = []
            for field, raw in zip(self.FIELDS, values[1:]):
                try:
                    value = float(raw)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("invalid numeric telemetry")
                    sample[field] = value
                except ValueError:
                    sample[field] = None
                    unavailable.append(field)
            sample["status"] = "partial_failure" if unavailable else "completed"
            sample["unavailable_metrics"] = unavailable
        except Exception as exc:
            sample.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        sample.update(query_finished_at_utc=utc_now(), query_finished_unix_ns=time.time_ns(),
                      query_finished_monotonic_ns=time.monotonic_ns())
        return sample

    def _run(self):
        try:
            with self.path.open("w", encoding="utf-8") as handle:
                while not self._stop.is_set():
                    next_sample = time.monotonic() + self.interval_seconds
                    sample = self.sample()
                    self._samples.append(sample)
                    handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    if self._stop.wait(max(0.0, next_sample - time.monotonic())):
                        break
        except Exception as exc:
            self._thread_error = {"type": type(exc).__name__, "message": str(exc)}

    def start(self):
        self.started_at_utc = utc_now()
        self._thread = threading.Thread(target=self._run, name="v100-gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            # Each query is killed and reaped by subprocess.run after at most 2s.
            self._thread.join()
        self.finished_at_utc = utc_now()
        completed = sum(sample["status"] == "completed" for sample in self._samples)
        status = "completed" if self._samples and completed == len(self._samples) and not self._thread_error else "partial_failure" if completed else "failed"
        metrics = {}
        for field in self.FIELDS:
            values = [sample[field] for sample in self._samples if sample.get(field) is not None]
            metrics[field] = {"samples": len(values), "min": min(values) if values else None,
                              "max": max(values) if values else None,
                              "mean": statistics.mean(values) if values else None}
        return {"status": status, "path": str(self.path), "command": self.command,
                "gpu_uuid": self.gpu_uuid, "interval_seconds": self.interval_seconds,
                "query_timeout_seconds": 2.0, "samples": len(self._samples),
                "complete_samples": completed, "incomplete_samples": len(self._samples) - completed,
                "started_at_utc": self.started_at_utc, "finished_at_utc": self.finished_at_utc,
                "thread_error": self._thread_error, "metrics": metrics,
                "scope": "entire profiler subprocess, including model loading, warmup and diagnostics; align individual query windows with profiler timestamps"}


def validate_profile_contract(profile, job, expected_environment):
    """A successful exit must still match the scheduled experiment arm."""
    config = profile.get("config", {})
    expected = dict(job["shape"], dtype=job["dtype"], cache=job["arm"], seed=job["seed"],
                    warmup=1, repeats=1, batch_size=1, device="cuda:0")
    for field, value in expected.items():
        if config.get(field) != value:
            return "profile config mismatch for %s: expected %r, received %r" % (field, value, config.get(field))
    # Config cuda:0 alone does not identify physical GPU 1 after CUDA filtering.
    target = canonical_gpu_uuid(expected_environment.get("CUDA_VISIBLE_DEVICES"))
    if target is None:
        return "expected environment must bind exactly one full GPU UUID"
    environment = profile.get("environment", {})
    if not isinstance(environment, dict):
        return "profile is missing reported environment"
    gpu = environment.get("gpu", {})
    if not isinstance(gpu, dict) or canonical_gpu_uuid(gpu.get("uuid")) != target:
        return "profile reported GPU UUID does not match the requested physical GPU"
    if canonical_gpu_uuid(environment.get("cuda_visible_devices")) != target:
        return "profile reported CUDA_VISIBLE_DEVICES does not match the requested GPU UUID"
    if environment.get("device") != "cuda:0":
        return "profile reported logical device must be cuda:0"
    if environment.get("cuda_device_order") != expected_environment.get("CUDA_DEVICE_ORDER"):
        return "profile reported CUDA_DEVICE_ORDER does not match child environment"
    if len(profile.get("runs", [])) != 1:
        return "expected exactly one measured repeat"
    aggregate = profile.get("aggregate", {})
    for field in ("e2e_ms_median", "pipeline_plus_refresh_ms_median", "peak_allocated_bytes_max"):
        value = aggregate.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return "missing or invalid aggregate metric: " + field
    return None


def run_one(job, args, environment):
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    record = dict(job, started_at_utc=utc_now(), status="running",
                  process_environment={key: environment[key] for key in (
                      "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    write_json(directory / "record.json", record)
    start = time.monotonic()
    log_path = directory / "raw.log"
    profile_path = directory / "profile.json"
    print("开始 %s %s %s seed=%s arm=%s" % (job["dtype"], job["stage"], job["prompt_id"], job["seed"], job["arm"]), flush=True)
    execution_failure = None
    telemetry = GpuTelemetry(directory, environment["CUDA_VISIBLE_DEVICES"], args.telemetry_interval_seconds)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            telemetry.start()
            record["profiler_started_unix_ns"] = time.time_ns()
            record["profiler_started_monotonic_ns"] = time.monotonic_ns()
            process = subprocess.run(job["command"], cwd=str(ROOT), env=environment,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout_seconds)
            returncode = process.returncode
        except subprocess.TimeoutExpired as exc:
            returncode, execution_failure = None, "subprocess_timeout"
            log.write("\nOrchestrator timeout: %s\n" % exc)
        except OSError as exc:
            returncode, execution_failure = None, "subprocess_launch_error"
            log.write("\nOrchestrator launch error: %s\n" % exc)
        except KeyboardInterrupt:
            returncode, execution_failure = None, "subprocess_interrupted"
            log.write("\nOrchestrator interrupted by user.\n")
        finally:
            record["profiler_finished_unix_ns"] = time.time_ns()
            record["profiler_finished_monotonic_ns"] = time.monotonic_ns()
            record["telemetry"] = telemetry.stop()
    profile = None
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            record["profile_read_error"] = str(exc)
    status, failure_category = classify_result(returncode, profile, log_path.read_text(encoding="utf-8", errors="replace"))
    if status == "completed":
        mismatch = validate_profile_contract(profile, job, environment)
        if mismatch:
            status, failure_category = "failed", "profile_contract_mismatch"
            record["profile_contract_error"] = mismatch
    if execution_failure:
        status = "interrupted" if execution_failure == "subprocess_interrupted" else "failed"
        failure_category = execution_failure
    record.update(status=status, failure_category=failure_category, returncode=returncode,
                  finished_at_utc=utc_now(), subprocess_wall_seconds=time.monotonic() - start,
                  raw_log=str(log_path), profile_file=str(profile_path) if profile_path.exists() else None,
                  quality_status="unknown", research_gate_eligible=False)
    if isinstance(profile, dict):
        record["failure"] = profile.get("failure")
        record["aggregate"] = profile.get("aggregate", {})
        record["instrumentation_output_exact_match"] = [run.get("instrumentation_output_exact_match") for run in profile.get("runs", [])]
        record["artifacts"] = [run.get("output", {}).get("artifact") for run in profile.get("runs", [])]
    write_json(directory / "record.json", record)
    print("结束 %s: %s%s" % (job["arm"], status, (" / " + failure_category) if failure_category else ""), flush=True)
    return record


def execute_suite(plan, args, environment, records, output):
    for stage in selected_stages(args.stage):
        stage_records = []
        for job in [job for job in plan if job["stage"] == stage]:
            record = run_one(job, args, environment)
            records.append(record)
            stage_records.append(record)
            write_json(output / "records.json", records)
            if record["status"] == "nonfinite":
                return {"status": "nonfinite", "failed_stage": stage}
            if record["status"] == "interrupted":
                return {"status": "interrupted", "failed_stage": stage}
        if stage == "smoke" and any(record["status"] != "completed" for record in stage_records):
            return {"status": "smoke_failed", "failed_stage": stage,
                    "skipped_stages": [name for name in selected_stages(args.stage) if name != "smoke"]}
    return {"status": "completed" if all(record["status"] == "completed" for record in records if record["dtype"] == plan[0]["dtype"]) else "incomplete"}


def export_media(output, records, dtype, args, environment):
    items = []
    for record in records:
        if record["dtype"] != dtype or record["status"] != "completed":
            continue
        for artifact in record.get("artifacts", []):
            if artifact:
                items.append({"stage": record["stage"], "prompt_id": record["prompt_id"],
                              "seed": record["seed"], "arm": record["arm"], "artifact": artifact})
    if not items:
        return {"status": "no_completed_artifacts", "quality_status": "unknown"}
    media = output / "media" / dtype
    spec_path, report_path = output / "media-export-input.json", output / "media-export.json"
    write_json(spec_path, {"items": items, "dtype": dtype, "output": str(media), "report": str(report_path)})
    log_path = output / "media-export.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run([args.python, "-m", "ditserve.gpu.media", str(spec_path)], cwd=str(ROOT),
                                 env=environment, stdout=log, stderr=subprocess.STDOUT)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"status": "failed", "quality_status": "unknown",
                  "reason": "媒体导出依赖或进程失败；原始 npy 保留"}
    report.update(returncode=process.returncode, log_file=str(log_path))
    write_json(report_path, report)
    return report


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.timeout_seconds is not None and (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        parser.error("--timeout-seconds 必须是有限正数")
    if not math.isfinite(args.telemetry_interval_seconds) or not 2 <= args.telemetry_interval_seconds <= 5:
        parser.error("--telemetry-interval-seconds 必须在 2–5 秒之间")
    config = load_config(args.config)
    args.revision = args.revision or config.get("revision")
    args.gpu_uuid = args.gpu_uuid or config.get("gpu_uuid")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / ".cache" / "runs" / ("v100-" + timestamp)).resolve()
    plan = build_plan(args, config, args.dtype, output)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "physical_gpu_index": 1,
                          "CUDA_VISIBLE_DEVICES": args.gpu_uuid or "<运行时查询物理 GPU 1 的 UUID>",
                          "selected_stages": selected_stages(args.stage), "initial_dtype": args.dtype,
                          "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                          "telemetry_interval_seconds": args.telemetry_interval_seconds,
                          "telemetry_target": args.gpu_uuid,
                          "fp32_nonfinite_fallback": args.dtype == "float16", "jobs": plan},
                         indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error("结果目录非空，拒绝覆盖，请指定新的 --output")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at_utc": utc_now(), "status": "initializing", "config": config,
                "config_file": str(args.config.resolve()), "selected_stages": selected_stages(args.stage),
                "initial_dtype": args.dtype, "selected_dtype": args.dtype, "fp32_rerun": False,
                "requested_revision": args.revision, "python": args.python,
                "model_id": args.model_id or config["model_id"], "initial_plan": plan,
                "telemetry_interval_seconds": args.telemetry_interval_seconds,
                "quality_status": "unknown", "research_gate_eligible": False}
    write_json(output / "manifest.json", manifest)
    records = []
    try:
        gpu = resolve_gpu(args.gpu_uuid)
        manifest["gpu"] = gpu
        environment = child_environment()
        environment["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["PYTHONUNBUFFERED"] = "1"
        environment["OMP_NUM_THREADS"] = "8"
        environment["MKL_NUM_THREADS"] = "8"
        manifest["process_environment"] = {key: environment[key] for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        manifest["status"] = "running"
        write_json(output / "manifest.json", manifest)
        result = execute_suite(plan, args, environment, records, output)
        if result["status"] == "nonfinite" and args.dtype == "float16":
            manifest["fp16_suite_result"] = result
            manifest["selected_dtype"] = "float32"
            manifest["fp32_rerun"] = True
            plan = build_plan(args, config, "float32", output)
            manifest["fp32_plan"] = plan
            write_json(output / "manifest.json", manifest)
            print("检测到 FP16 非有限值，保留原始结果；全部所选案例和两 arm 改用 FP32 重测。", flush=True)
            result = execute_suite(plan, args, environment, records, output)
        manifest["suite_result"] = result
        manifest["status"] = result["status"]
        if not args.no_export_media and result["status"] != "interrupted":
            try:
                manifest["media_export"] = export_media(output, records, manifest["selected_dtype"], args, environment)
            except Exception as exc:
                manifest["media_export"] = {"status": "failed", "type": type(exc).__name__, "message": str(exc)}
        else:
            manifest["media_export"] = {"status": "disabled_or_interrupted"}
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["orchestrator_failure"] = "用户中断；已完成记录与当前案例原始日志保留"
    except Exception as exc:
        manifest["status"] = "orchestrator_failed"
        manifest["orchestrator_failure"] = {"type": type(exc).__name__, "message": str(exc)}
    manifest["finished_at_utc"] = utc_now()
    manifest["telemetry_complete"] = bool(records) and all(record.get("telemetry", {}).get("status") == "completed" for record in records)
    write_json(output / "manifest.json", manifest)
    write_json(output / "records.json", records)
    from .summarize_v100 import build_summary, write_report
    write_report(build_summary(output, ROOT / ".cache/runs/v100/environment"), output)
    try:
        archived = archive_run(output, "wan-paired", args.results_root)
        print("实验报告：" + archived["report"], flush=True)
    except (OSError, ValueError, ImportError) as exc:
        print("归档失败，原始记录保留于 %s: %s" % (output, exc), file=sys.stderr)
        return 2
    media_status = manifest.get("media_export", {}).get("status")
    media_ok = media_status not in ("failed", "partial_failure", "no_completed_artifacts")
    return 0 if manifest["status"] == "completed" and manifest["telemetry_complete"] and media_ok else 2


if __name__ == "__main__":
    sys.exit(main())
