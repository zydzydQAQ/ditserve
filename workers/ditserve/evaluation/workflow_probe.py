"""Small, explicitly invoked CUDA/NVTX profiler compatibility check."""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seconds", type=float, default=3.)
    args = parser.parse_args()
    import torch
    torch.cuda.set_device(args.device)
    torch.manual_seed(101)
    a = torch.randn(2048, 2048, device=args.device, dtype=torch.float16)
    b = torch.randn_like(a)
    for _ in range(5):
        out = a @ b
    torch.cuda.synchronize()
    begin = time.perf_counter()
    count = 0
    torch.cuda.nvtx.range_push("workflow/probe/matmul")
    while time.perf_counter() - begin < args.seconds:
        for _ in range(20):
            out = a @ b
            count += 1
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("workflow/probe/output_d2h")
    cpu = out.cpu()
    torch.cuda.nvtx.range_pop()
    print(json.dumps(dict(device=args.device, iterations=count, finite=bool(cpu.isfinite().all()),
                          wall_s=time.perf_counter()-begin, torch=torch.__version__)))


if __name__ == "__main__":
    main()
