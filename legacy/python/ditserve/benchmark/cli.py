"""Public entry points for datasets, frozen workloads, execution and evaluation."""

import argparse
import contextlib
import datetime
import json
import signal
import sys
import uuid
from pathlib import Path

from ..util import CONFIGS, ROOT
from .budget import GpuBudget
from .common import atomic_json, read_json, seal, verify_seal


RESULTS = ROOT / "results" / "video-benchmark"


def config_path(value):
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    preset = CONFIGS / "benchmark" / (value + ".json")
    if preset.is_file():
        return preset
    raise ValueError("unknown configuration: " + value)


def _new_file(path, value):
    if Path(path).exists():
        raise ValueError("refusing to overwrite frozen artifact: " + str(path))
    return atomic_json(path, value)


def _new_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("output directory must be new or empty: " + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require_absent(path):
    if Path(path).exists():
        raise ValueError("attempt directory already exists; use a new attempt: " + str(path))


def _attempt_path():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS / (stamp + "-" + uuid.uuid4().hex[:8])


def _backend_options(parser, choices=("cpu-test", "wan")):
    parser.add_argument("--backend", choices=choices, default="cpu-test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--attention", choices=("math", "flash", "efficient"), default="math",
                        help="Explicit DiT SDPA backend; VAE stays MATH; no fallback")
    parser.add_argument("--budget-path", type=Path,
                        help="Use the existing authoritative ledger; never clone or reset a historical budget")


def _budget(backend, args=None):
    if backend != "wan":
        return None
    path = getattr(args, "budget_path", None)
    if path is None:
        local = RESULTS / "gpu-budget.json"
        historical = ROOT.parent / "dit/results/video-benchmark/gpu-budget.json"
        if local.is_file() and historical.is_file() and local.resolve() != historical.resolve():
            raise ValueError("two GPU ledgers exist; select the authoritative cumulative --budget-path")
        path = historical if historical.is_file() else local
    return GpuBudget(path, hours=2)


@contextlib.contextmanager
def _interruptible():
    """Translate a normal termination request into a persisted interrupted attempt."""
    def interrupted(signum, frame):
        raise KeyboardInterrupt("received signal %s" % signum)

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _exit_status(result):
    state = result.get("status", "completed")
    return 0 if state in ("completed", "ok", "passed", "exported", "dry_run") else 3


def _save_trace(destination, trace):
    from .workload import to_core_trace, validate_trace
    validate_trace(trace)
    destination = _new_directory(destination)
    _new_file(destination / "trace.json", trace)
    _new_file(destination / "core-trace.json", to_core_trace(trace))
    with (destination / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for request in trace["requests"]:
            stream.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
    return str(destination / "trace.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible video DiT benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze public prompt data; downloads are opt-in")
    prepare.add_argument("--dataset", choices=("vbench", "vidprom"), required=True)
    prepare.add_argument("--source", type=Path)
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--revision")
    prepare.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "benchmark-data")
    prepare.add_argument("--dev-count", type=int, default=32)
    prepare.add_argument("--sample-count", type=int, default=4096)
    prepare.add_argument("--seed", type=int, default=20261012)
    prepare.add_argument("--output", type=Path, required=True)

    profile = commands.add_parser("profile", help="Profile development samples independently")
    profile.add_argument("--config", default="smoke")
    profile.add_argument("--dataset", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    _backend_options(profile, ("cpu-test", "core", "wan"))

    trace = commands.add_parser("trace", help="Freeze one open-loop workload")
    trace.add_argument("--config", default="smoke")
    trace.add_argument("--dataset", type=Path, required=True)
    trace.add_argument("--profile", type=Path, required=True)
    trace.add_argument("--pattern", choices=("poisson", "gamma", "foreground_burst", "offline_all", "replay"))
    trace.add_argument("--rate-rps", type=float)
    trace.add_argument("--load", type=float, default=1.)
    trace.add_argument("--cv", type=float, default=2.)
    trace.add_argument("--seed", type=int, default=20261012)
    trace.add_argument("--count", type=int)
    trace.add_argument("--replay", type=Path)
    trace.add_argument("--time-scale", type=float, default=1.)
    trace.add_argument("--vbench-manifest", type=Path, help="Separate official quality generation protocol")
    trace.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run", help="Replay exactly one frozen trace across policies")
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--policies", default="fcfs,edf,srtf")
    run.add_argument("--output", type=Path)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--text-prefetch", type=int, default=0,
                     help="Bounded CPU encoding pipeline: 0 disables, 2 allows two outstanding requests")
    run.add_argument("--quality-reference", type=Path,
                     help="Require held-out LPIPS against this measured reference run before online timing")
    run.add_argument("--quality-epsilon", type=float, default=.05)
    _backend_options(run, ("cpu-test", "core", "wan"))

    evaluate = commands.add_parser("evaluate", help="Evaluate reference fidelity or a separate official VBench suite")
    evaluate.add_argument("--run", type=Path)
    evaluate.add_argument("--reference-run", type=Path)
    evaluate.add_argument("--lpips", action="store_true")
    evaluate.add_argument("--epsilon", type=float, default=.05)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--vbench-info", type=Path)
    evaluate.add_argument("--config", default="formal", help="Generation settings for a new VBench manifest")
    evaluate.add_argument("--shape", default="long", help="One configured shape for the VBench suite")
    evaluate.add_argument("--vbench-manifest", type=Path)
    evaluate.add_argument("--dimensions", help="Comma-separated official dimensions; omission means all")
    evaluate.add_argument("--videos-dir", type=Path)
    evaluate.add_argument("--repo", type=Path)
    evaluate.add_argument("--python", dest="python_executable")
    evaluate.add_argument("--export-videos", action="store_true")
    evaluate.add_argument("--policy", default="fcfs")

    report = commands.add_parser("report", help="Rebuild auditable reports from immutable run records")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path)

    plan = commands.add_parser("optimize-plan", help="Freeze a paired math/flash/prefetch serving experiment")
    plan.add_argument("--config", default="optimization")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--math-profile", type=Path, required=True)
    plan.add_argument("--flash-profile", type=Path, required=True)
    plan.add_argument("--stage", choices=("pilot", "expanded"), default="pilot")
    plan.add_argument("--remaining-gpu-seconds", type=float)
    plan.add_argument("--output", type=Path, required=True)

    comparison = commands.add_parser("optimize-report", help="Compare complete paired runs and gate follow-up work")
    comparison.add_argument("--runs", type=Path, nargs="+", required=True)
    comparison.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "optimize-plan":
            from .optimization import build_plan
            _require_absent(args.output)
            plan = build_plan(read_json(config_path(args.config)), read_json(args.dataset),
                {"math_serial": read_json(args.math_profile), "flash_serial": read_json(args.flash_profile)},
                stage=args.stage, remaining_gpu_seconds=args.remaining_gpu_seconds)
            args.output.mkdir(parents=True)
            atomic_json(args.output / "plan.json", plan)
            for group in plan["groups"]:
                for variant, frozen in group["traces"].items():
                    _save_trace(args.output / group["group_id"] / variant, frozen)
            result = dict(status="completed", plan=str(args.output / "plan.json"), groups=len(plan["groups"]),
                          deferred_groups=len(plan["deferred_groups"]), budget=plan["budget"], gpu_loaded=False)
        elif args.command == "optimize-report":
            from .optimization import summarize_runs
            result = summarize_runs(args.runs, output_path=args.output)
        elif args.command == "prepare":
            from .datasets import prepare_dataset
            if args.output.exists():
                raise ValueError("dataset output already exists")
            dataset = prepare_dataset(args.dataset, source=args.source, revision=args.revision,
                                      dev_count=args.dev_count, seed=args.seed, download=args.download,
                                      cache_dir=args.cache_dir, sample_count=args.sample_count)
            _new_file(args.output, dataset)
            result = {"status": "completed", "dataset": str(args.output),
                      "samples": len(dataset["samples"]), "sha256": dataset["sha256"]}
        elif args.command == "profile":
            from .runner import build_profile
            config, dataset = read_json(config_path(args.config)), read_json(args.dataset)
            verify_seal(dataset)
            _require_absent(args.output)
            budget = _budget(args.backend, args)
            with _interruptible(), budget.session() if budget else contextlib.nullcontext():
                result = build_profile(config, dataset, args.backend, args.output,
                                       device=args.device, model_path=args.model_path, budget=budget,
                                       attention=args.attention)
                if budget and result.get("status") != "completed":
                    budget.finish(result.get("status", "failed"))
        elif args.command == "trace":
            from .workload import build_trace
            if args.vbench_manifest:
                from .protocols import build_vbench_trace
                if any(value is not None for value in (args.pattern, args.rate_rps, args.count, args.replay)) or args.time_scale != 1 or args.load != 1 or args.cv != 2:
                    raise ValueError("VBench generation manifest fixes all requests; serving arrival overrides do not apply")
                trace = build_vbench_trace(read_json(config_path(args.config)), read_json(args.dataset),
                                          read_json(args.profile), read_json(args.vbench_manifest))
            else:
                trace = build_trace(read_json(config_path(args.config)), read_json(args.dataset),
                                read_json(args.profile), pattern=args.pattern or "poisson", rate_rps=args.rate_rps,
                                load=args.load, cv=args.cv, seed=args.seed, count=args.count,
                                replay=read_json(args.replay) if args.replay else None,
                                time_scale=args.time_scale)
            result = {"status": "completed", "trace": _save_trace(args.output, trace),
                      "sha256": trace["sha256"], "requests": len(trace["requests"])}
        elif args.command == "run":
            from .workload import validate_trace
            frozen = read_json(args.trace)
            validate_trace(frozen)
            from .runner import validate_execution_options
            validate_execution_options(frozen, args.backend, args.attention, args.text_prefetch)
            policies = [name.strip() for name in args.policies.split(",")]
            if not all(policies) or len(set(policies)) != len(policies):
                raise ValueError("policies must be nonempty and unique")
            if args.dry_run:
                from .backends import core_policy
                from .runner import POLICIES
                if frozen["profile"].get("backend") != args.backend:
                    raise ValueError("profile backend does not match execution backend")
                for policy in policies:
                    if args.backend == "core":
                        core_policy(policy)
                    elif policy not in POLICIES:
                        raise ValueError("Unsupported policy: " + policy)
                result = {"status": "dry_run", "backend": args.backend, "policies": policies,
                          "execution_options": dict(attention=args.attention, text_prefetch=args.text_prefetch),
                          "trace_sha256": frozen["sha256"], "requests": len(frozen["requests"]),
                          "gpu_loaded": False, "gpu_budget_hours": 2,
                          "notice": "Validation only; no measurements or quality claims."}
            else:
                from .runner import run_suite
                budget = _budget(args.backend, args)
                destination = args.output or _attempt_path()
                _require_absent(destination)
                with _interruptible(), budget.session() if budget else contextlib.nullcontext():
                    result = run_suite(frozen, policies, args.backend, destination,
                                       device=args.device, model_path=args.model_path, budget=budget,
                                       attention=args.attention, text_prefetch=args.text_prefetch,
                                       quality_reference=args.quality_reference, quality_epsilon=args.quality_epsilon)
                    if budget and result.get("status") != "completed":
                        budget.finish(result.get("status", "failed"))
        elif args.command == "evaluate":
            from . import evaluation as e
            if args.vbench_info:
                from .protocols import generation_settings
                if not args.output:
                    raise ValueError("--vbench-info requires --output manifest.json")
                if args.output.exists():
                    raise ValueError("VBench plan output already exists")
                result = e.export_vbench_manifest(args.vbench_info, args.output,
                    dimensions=args.dimensions.split(",") if args.dimensions else None,
                    generation_settings=generation_settings(read_json(config_path(args.config)), args.shape))
            elif args.export_videos:
                if not all((args.run, args.vbench_manifest, args.videos_dir)):
                    raise ValueError("video export requires --run --vbench-manifest --videos-dir")
                result = e.export_vbench_videos(args.run, args.vbench_manifest, args.videos_dir, policy=args.policy)
            elif args.vbench_manifest:
                if not all((args.videos_dir, args.repo)):
                    raise ValueError("VBench scoring requires --videos-dir and --repo")
                budget = _budget("wan")
                with budget.session():
                    result = e.run_vbench(args.vbench_manifest, args.videos_dir, args.repo,
                        output_dir=args.output, python_executable=args.python_executable,
                        timeout=max(1, int(budget.check())))
            elif args.run:
                result = e.evaluate_run(args.run, reference_run_path=args.reference_run,
                    with_lpips=args.lpips, epsilon=args.epsilon, output_path=args.output)
            else:
                raise ValueError("choose --run, --vbench-info or --vbench-manifest")
        else:
            from .reporting import write_report
            result = write_report(args.run, output_dir=args.output)
        # Large immutable artifacts stay on disk; stdout is a compact operation receipt.
        receipt = {key: result[key] for key in ("schema_version", "status", "backend", "measurement_kind",
                   "measured", "sha256", "trace_sha256", "dataset", "output_dir", "output_path",
                   "reason", "error", "notice", "gpu_loaded", "gpu_budget_hours", "scope") if key in result}
        for key in ("requests", "samples", "trace"):
            if isinstance(result.get(key), (str, int)):
                receipt[key] = result[key]
        print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
        return _exit_status(result)
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as error:
        print("benchmark: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
