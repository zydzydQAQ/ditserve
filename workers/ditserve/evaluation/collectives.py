"""Per-rank all-to-all bandwidth on aligned rank groups (Ulysses SP pattern).

Run on the target node under torchrun, one process per GPU in topology order:

    torchrun --nproc_per_node 8 -m ditserve.evaluation.collectives --output /data/collectives.json

Groups are the aligned pairs, quads and the whole node used by the SP layout.
``concurrent`` runs every group of a degree at once (the contended case when
several SP requests share PCIe switches); ``isolated`` runs only the first
group while the other ranks wait. CPU Gloo runs are protocol checks only.
"""

import argparse
import json
import os
import statistics
import time

from ..models.wan.parallel import aligned_groups


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(*, sizes_mb=(16, 64, 256), iterations=5, warmup=2, backend=None, device=None):
    """Collective call: every rank must call with identical arguments."""
    import torch
    dist = torch.distributed
    world, rank = dist.get_world_size(), dist.get_rank()
    backend = backend or dist.get_backend()
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
    groups = {ranks: dist.new_group(list(ranks), backend=backend) for ranks in aligned_groups(world) if len(ranks) > 1}
    rows = []
    for degree in sorted({len(ranks) for ranks in groups}):
        mine = next(ranks for ranks in groups if len(ranks) == degree and rank in ranks)
        first = next(ranks for ranks in groups if len(ranks) == degree)
        for mode in ("concurrent", "isolated"):
            active = mode == "concurrent" or rank in first
            for size_mb in sizes_mb:
                elements = int(size_mb * 2 ** 20) // 2 // degree * degree
                send = torch.ones(elements, dtype=torch.float16, device=device)
                receive = torch.empty_like(send)
                samples = []
                for index in range(warmup + iterations):
                    dist.barrier()
                    if active:
                        _sync(torch, device)
                        started = time.perf_counter()
                        dist.all_to_all_single(receive, send, group=groups[mine])
                        _sync(torch, device)
                        if index >= warmup:
                            samples.append((time.perf_counter() - started) * 1000)
                if active and rank == mine[0]:
                    sent_mb = elements * 2 / 2 ** 20 * (degree - 1) / degree
                    median = statistics.median(samples)
                    rows.append(dict(group=list(mine), degree=degree, mode=mode, buffer_mb=size_mb,
                                     per_rank_send_mb=sent_mb, iterations=iterations, ms_median=median,
                                     ms_max=max(samples), per_rank_gbps=sent_mb / 1024 / (median / 1000)))
    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    return [row for part in gathered for row in part]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=[16, 64, 256])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--backend", default="nccl", choices=("nccl", "gloo"))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    import torch
    if args.backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.distributed.init_process_group(backend=args.backend)
    try:
        rows = measure(sizes_mb=args.sizes_mb, iterations=args.iterations, backend=args.backend)
        if torch.distributed.get_rank() == 0:
            report = dict(schema="ditserve.collectives.v1", world_size=torch.distributed.get_world_size(),
                          backend=args.backend, torch=torch.__version__,
                          nccl=".".join(map(str, torch.cuda.nccl.version())) if args.backend == "nccl" else None,
                          scope="all_to_all_single fp16; per-rank send volume / median wall time", rows=rows)
            text = json.dumps(report, indent=2)
            if args.output:
                with open(args.output, "x", encoding="utf-8") as stream:
                    stream.write(text)
            print(text)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
