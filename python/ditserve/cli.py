"""Command line interface: doctor / simulate / sweep / archive."""

import argparse
import json
from pathlib import Path
import sys

from . import api, archive
from .core import POLICIES
from .util import read_json


def _csv(value, convert=str):
    try:
        result = [convert(item.strip()) for item in value.split(",")]
        if not result or any(item == "" for item in result):
            raise ValueError()
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("Expected a comma-separated list, got %r" % value)


def _shapes(value):
    try:
        shapes = [tuple(int(d) for d in item.split("x")) for item in value.split(",")]
        if any(len(s) != 3 or min(s) <= 0 for s in shapes):
            raise ValueError()
        return shapes
    except ValueError:
        raise argparse.ArgumentTypeError("Use HxWxframes, e.g. 512x512x1,768x768x1")


def _policies(value):
    result = list(POLICIES) if value == "all" else _csv(value)
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Duplicate policy in " + value)
    return result


def _common(parser):
    parser.add_argument("--output", type=Path, help="Raw directory; defaults to .cache/runs/<unique run>")
    parser.add_argument("--keep-raw", action="store_true", help="Retain the working directory after verified archival")
    parser.add_argument("--results-root", type=Path, help="Excel/report archive root; defaults to ./results")
    parser.add_argument("--cost-model", type=Path, help="Synthetic cost JSON; GPU profiles require calibration")
    parser.add_argument("--policies", type=_policies, default=list(POLICIES))
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shapes", type=_shapes, default=[(512, 512, 1), (768, 768, 1)])
    parser.add_argument("--steps", type=lambda v: _csv(v, int), default=[20, 30])
    parser.add_argument("--cache-threshold", type=float, default=0.15)
    parser.add_argument("--slo-scale", type=float, default=2.0)
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--host-memory-mb", type=float, default=65536.0)
    parser.add_argument("--skip-parity", action="store_true", help="Unknown quality never counts as successful goodput")


def _shared(args):
    return {"count": args.count, "seed": args.seed, "slo_scale": args.slo_scale, "shapes": args.shapes,
            "step_choices": args.steps, "cache_threshold": args.cache_threshold, "policies": args.policies,
            "max_batch": args.max_batch, "host_memory_mb": args.host_memory_mb,
            "cost_model": read_json(args.cost_model) if args.cost_model else None,
            "verify_outputs": not args.skip_parity, "output": args.output,
            "results_root": args.results_root, "keep_raw": args.keep_raw}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "benchmark":
        from .benchmark.cli import main as benchmark_main
        return benchmark_main(argv[1:])
    if argv and argv[0] == "profile":
        from .gpu.profiling import main as profile_main
        return profile_main(argv[1:])
    parser = argparse.ArgumentParser(prog="ditserve",
                                     description="C++ reference control plane with a Python experiment API")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect local prerequisites without loading models")
    sub.add_parser("profile", help="Measure real FLUX/Wan pipelines (forwarded; needs the GPU stack)")
    sub.add_parser("benchmark", help="Frozen online traces, Wan execution and quality reports (forwarded)")
    simulate = sub.add_parser("simulate", help="Execute synthetic reference states on an analytic timeline")
    _common(simulate)
    simulate.add_argument("--trace", type=Path, help="JSON RequestSpec array with absolute arrival/deadline milliseconds")
    simulate.add_argument("--arrival-rate", type=float, default=8.0)
    simulate.add_argument("--pattern", choices=("steady", "bursty"), default="steady")
    simulate.add_argument("--num-gpus", type=int, default=1)
    simulate.add_argument("--memory-mb", type=float, default=256.0)
    sweep = sub.add_parser("sweep", help="Replay workloads across synthetic resource budgets")
    _common(sweep)
    sweep.add_argument("--rates", type=lambda v: _csv(v, float), default=[2.0, 8.0, 16.0])
    sweep.add_argument("--memories", type=lambda v: _csv(v, float), default=[128.0, 256.0])
    sweep.add_argument("--gpu-counts", type=lambda v: _csv(v, int), default=[1, 4])
    sweep.add_argument("--patterns", type=_csv, default=["steady", "bursty"])
    archive_cmd = sub.add_parser("archive", help="Verify or rebuild archived experiment steps")
    archive_sub = archive_cmd.add_subparsers(dest="archive_command", required=True)
    archive_verify = archive_sub.add_parser("verify", help="Verify evidence bytes, workbook cells and report links")
    archive_verify.add_argument("--root", type=Path, default=Path("results"))
    archive_rebuild = archive_sub.add_parser("rebuild", help="Regenerate results.xlsx and REPORT.md from data.json.gz")
    archive_rebuild.add_argument("step", type=Path, help="Archive step directory, e.g. results/synthetic/<run>")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(api.doctor(), indent=2))
        return 0
    try:
        if args.command == "archive":
            if args.archive_command == "verify":
                result = archive.verify(args.root)
            else:
                result = archive.rebuild(args.step)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "simulate":
            result = api.simulate(arrival_rate=args.arrival_rate, pattern=args.pattern,
                                  num_gpus=args.num_gpus, memory_mb=args.memory_mb,
                                  trace=args.trace, **_shared(args))
        else:
            result = api.sweep(rates=args.rates, memories=args.memories,
                               gpu_counts=args.gpu_counts, patterns=args.patterns, **_shared(args))
        print("Synthetic reference experiment. Report: " + result["archive"]["report"])
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
