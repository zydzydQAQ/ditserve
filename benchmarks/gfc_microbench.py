#!/usr/bin/env python3
"""Group-free collectives (GF-DiT's GFC) vs NCCL on this node: setup cost, latency, exactness.

Launch on one 8-GPU node:
    python -m torch.distributed.run --standalone --nproc_per_node 8 gfc_microbench.py --output OUT.json

Measures, per rank 0 report (all ranks participate):
  1. Peer access matrix (torch.cuda.can_device_access_peer) and GFC runtime init (symmetric-memory
     rendezvous) time and memory.
  2. Group setup: torch.distributed.new_group + first collective (NCCL communicator creation) versus
     gfc register_group + first collective, for aligned and non-aligned rank sets. This is the
     measurement GF-DiT reports as ~778 ms vs ~60 us on its hardware; numbers here are this
     platform's own and are not expected to match.
  3. Steady latency of all-to-all and all-gather at Wan-relevant sizes: NCCL, GFC vector path and
     GFC copy-engine path (a second runtime with use_copy_engine=True).
  4. Bitwise equality of every GFC result with the NCCL result for the same inputs (both are pure
     data movement, so any difference is a correctness failure).
Fails closed: any GFC initialization error is recorded with its message and the script exits
non-zero after writing the report; nothing silently falls back to NCCL.
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
PACKAGE = os.environ.get("DITSERVE_PACKAGE_ROOT")
if PACKAGE:
    sys.path.insert(0, PACKAGE)

GROUPS = {
    "pair-aligned": (0, 1),
    "pair-unaligned": (1, 2),
    "pair-far": (0, 7),
    "quad-aligned": (0, 1, 2, 3),
    "quad-unaligned": (1, 2, 3, 4),
    "triple": (3, 4, 5),
    "hex-unaligned": (1, 2, 3, 4, 5, 6),
    "octet": tuple(range(8)),
}
# Bytes per rank: CFG prediction exchange (Wan2.1 latent, fp32), Ulysses QKV exchange per layer for
# cfg2sp2 / cfg2sp4 at 480x832x33f (fp16), and the Wan2.2 sp2 exchange (512x896x49f).
SIZES = {
    "cfg-prediction-3.6MB": 3_594_240,
    "ulysses-wan21-cfg2sp4-32MB": 3 * 3510 * 1536 * 2,
    "ulysses-wan21-cfg2sp2-64MB": 3 * 7020 * 1536 * 2,
    "small-256KB": 262_144,
}


def sync():
    torch.cuda.synchronize()


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        sync()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples), "iters": iters}


def gather_reports(local):
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, local)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--setup-reps", type=int, default=3)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise SystemExit("gfc_microbench expects one 8-GPU node")
    report = {
        "schema": "ditserve.gfc-microbench.v1",
        "scope": "platform microbenchmark; not a serving result",
        "torch": torch.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "peer_access": [[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(torch.cuda.device_count())]
                        for i in range(torch.cuda.device_count())],
        "errors": [],
    }
    try:
        import triton
        report["triton"] = triton.__version__
    except Exception as error:  # recorded, GFC init below will fail
        report["errors"].append("triton import: " + repr(error))

    # --- 1. GFC runtime init (collective) ---
    runtimes = {}
    try:
        import gfc
        report["gfc_vendored"] = str(Path(gfc.__file__).parent)
        for path, copy_engine in (("vector", False), ("copy_engine", True)):
            before = torch.cuda.memory_reserved(device)
            dist.barrier()
            started = time.perf_counter()
            runtimes[path] = gfc.SymmetricCollectiveRuntime(
                gfc.SymmetricCollectiveConfig(max_group_size=world, max_collective_bytes=args.max_bytes,
                                              use_copy_engine=copy_engine), device=device)
            sync()
            elapsed = (time.perf_counter() - started) * 1000
            report.setdefault("gfc_init", {})[path] = {
                "init_ms_per_rank": [r["ms"] for r in gather_reports({"ms": elapsed})],
                "reserved_bytes_added_rank0": torch.cuda.memory_reserved(device) - before,
            }
    except Exception as error:
        report["errors"].append("gfc init: " + "".join(traceback.format_exception_only(type(error), error)).strip())

    # --- 2. Group setup cost ---
    setup = {}
    probe = torch.ones(1024, device=device)
    for name, ranks in GROUPS.items():
        rows = {"ranks": list(ranks), "nccl_new_group_first_ms": [], "gfc_register_us": [], "gfc_first_collective_ms": []}
        for _ in range(args.setup_reps):
            dist.barrier()
            sync()
            started = time.perf_counter()
            pg = dist.new_group(list(ranks))
            if rank in ranks:
                dist.all_reduce(probe, group=pg)
                sync()
            elapsed = (time.perf_counter() - started) * 1000
            dist.barrier()
            if rank in ranks:
                rows["nccl_new_group_first_ms"].append(elapsed)
            dist.destroy_process_group(pg)
            if "vector" in runtimes and rank in ranks:
                runtime = runtimes["vector"]
                started = time.perf_counter()
                descriptor = runtime.register_group(ranks)
                rows["gfc_register_us"].append((time.perf_counter() - started) * 1e6)
                out = torch.empty(len(ranks) * probe.numel(), device=device)
                started = time.perf_counter()
                runtime.all_gather(probe, out, descriptor)
                runtime.stream.synchronize()
                rows["gfc_first_collective_ms"].append((time.perf_counter() - started) * 1000)
                runtime.unregister_group(descriptor)
            dist.barrier()
        setup[name] = gather_reports(rows if rank in ranks else None)
    report["group_setup"] = {name: [row for row in rows if row] for name, rows in setup.items()}

    # --- 3/4. Latency and exactness ---
    latency, exact = {}, {}
    for gname in ("pair-aligned", "pair-unaligned", "quad-aligned", "quad-unaligned", "octet"):
        ranks = GROUPS[gname]
        pg = dist.new_group(list(ranks))
        descriptors = {path: runtime.register_group(ranks) for path, runtime in runtimes.items()} if rank in ranks else {}
        for sname, nbytes in SIZES.items():
            key = f"{gname}/{sname}"
            if nbytes * len(ranks) > args.max_bytes:
                continue
            row = {}
            if rank in ranks:
                elements = (nbytes // 2 // len(ranks)) * len(ranks)
                generator = torch.Generator(device=device).manual_seed(1000 + rank)
                source = torch.randn(elements, dtype=torch.float16, device=device, generator=generator)
                nccl_a2a, nccl_ag = torch.empty_like(source), torch.empty(len(ranks) * elements, dtype=source.dtype, device=device)
                row["nccl_all2all"] = timed(lambda: dist.all_to_all_single(nccl_a2a, source, group=pg), args.warmup, args.iters)
                row["nccl_all_gather"] = timed(lambda: dist.all_gather_into_tensor(nccl_ag, source, group=pg), args.warmup, args.iters)
                for path, runtime in runtimes.items():
                    d = descriptors[path]
                    out_a2a, out_ag = torch.empty_like(source), torch.empty_like(nccl_ag)
                    try:
                        row[f"gfc_{path}_all2all"] = timed(
                            lambda: (runtime.all2all(source, out_a2a, d, slice_bytes=source.nbytes // len(ranks)),
                                     runtime.stream.synchronize()), args.warmup, args.iters)
                        row[f"gfc_{path}_all_gather"] = timed(
                            lambda: (runtime.all_gather(source, out_ag, d), runtime.stream.synchronize()), args.warmup, args.iters)
                        exact.setdefault(key, {})[path] = bool(torch.equal(out_a2a, nccl_a2a) and torch.equal(out_ag, nccl_ag))
                    except Exception as error:
                        row[f"gfc_{path}_error"] = repr(error)
            reports = gather_reports(row if rank in ranks else None)
            latency[key] = {"bytes_per_rank": nbytes, "rank0_of_group": next(r for r in reports if r is not None)}
            dist.barrier()
        if rank in ranks:
            for path, runtime in runtimes.items():
                runtime.unregister_group(descriptors[path])
        dist.destroy_process_group(pg)
    report["latency"] = latency
    exact_all = gather_reports(exact)
    report["bitwise_equal_to_nccl"] = all(all(v for per in item.values() for v in per.values()) for item in exact_all) if runtimes else None
    report["exact_detail_rank0"] = exact

    for runtime in runtimes.values():
        runtime.shutdown()
    status = "completed" if runtimes and report["bitwise_equal_to_nccl"] and not report["errors"] else "failed"
    report["status"] = status
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "status": status, "errors": report["errors"]}))
    dist.barrier()
    dist.destroy_process_group()
    if status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
