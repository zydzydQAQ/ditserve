#!/usr/bin/env python3
"""Small CUDA probes; independent of doctor, with no model loads or downloads."""
import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed", action="store_true", help="Run with torchrun --standalone --nproc_per_node=2")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0")) if args.distributed else 0
    report = {"mode": "nccl" if args.distributed else "kernels", "checks": []}
    def check(name, operation):
        started = time.monotonic()
        item = {"name": name, "status": "pass"}
        try:
            operation()
        except Exception as error:
            item.update(status="fail", error=f"{type(error).__name__}: {error}")
        item["seconds"] = round(time.monotonic() - started, 3)
        report["checks"].append(item)
        return item["status"] == "pass"
    try:
        import torch
        import torch.distributed as dist
        report["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "triton": importlib.metadata.version("triton")}
        require(torch.__version__ == "2.10.0+cu126" and torch.version.cuda == "12.6", "Expected torch 2.10.0+cu126")
        require(torch.cuda.is_available() and torch.cuda.device_count() == 2, "Expected exactly two visible CUDA GPUs")
        report["devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                              "capability": list(torch.cuda.get_device_capability(i))} for i in range(2)]
        require(all("RTX 4090" in d["name"] and d["capability"] == [8, 9] for d in report["devices"]), "Expected two RTX 4090 GPUs")
        report["versions"]["nccl"] = list(torch.cuda.nccl.version())
        if args.distributed:
            require(int(os.environ.get("WORLD_SIZE", "0")) == 2, "Use torchrun --standalone --nproc_per_node=2")
            local_rank = int(os.environ["LOCAL_RANK"])
            require(local_rank in (0, 1), "LOCAL_RANK must be 0 or 1")
            require(os.environ.get("NCCL_P2P_DISABLE") == "1", "Set NCCL_P2P_DISABLE=1 for this 4090 profile")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            def initialize():
                dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30), device_id=device)
            def reduce():
                value = torch.full((8,), rank + 1., device=device)
                dist.all_reduce(value)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(value.cpu(), torch.full((8,), 3.), rtol=0, atol=0)
            def exchange():
                value = torch.tensor([10 * rank + i for i in range(4)], dtype=torch.float32, device=device)
                output = torch.empty_like(value)
                dist.all_to_all_single(output, value)
                torch.cuda.synchronize(device)
                expected = torch.tensor([2 * rank, 2 * rank + 1, 10 + 2 * rank, 11 + 2 * rank], dtype=torch.float32)
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            try:
                if check("nccl_init", initialize):
                    if check("all_reduce", reduce) and check("all_to_all_single", exchange):
                        def gather():
                            reports = [None, None]
                            dist.all_gather_object(reports, {"rank": rank, "checks": report["checks"]})
                            report["ranks"] = reports
                        check("gather_rank_results", gather)
            finally:
                if dist.is_initialized():
                    check("destroy_process_group", dist.destroy_process_group)
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            for index in range(2):
                device = torch.device("cuda", index)
                torch.cuda.set_device(index)
                def matmul():
                    value = torch.ones((16, 16), dtype=torch.bfloat16, device=device)
                    output = value @ value
                    torch.cuda.synchronize(device)
                    torch.testing.assert_close(output.float().cpu(), torch.full((16, 16), 16.), rtol=0, atol=0)
                def attention():
                    query = torch.zeros((1, 2, 16, 64), dtype=torch.bfloat16, device=device)
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        output = torch.nn.functional.scaled_dot_product_attention(query, query, torch.full_like(query, 2.))
                    torch.cuda.synchronize(device)
                    torch.testing.assert_close(output.float().cpu(), torch.full((1, 2, 16, 64), 2.), rtol=0, atol=0)
                def compile_pointwise():
                    with torch._dynamo.config.patch(disable=False, suppress_errors=False):
                        function = torch.compile(lambda value: value * 3 + 2, backend="inductor", fullgraph=True)
                        for fill in (1., 2.):
                            output = function(torch.full((64, 64), fill, device=device))
                            torch.cuda.synchronize(device)
                            torch.testing.assert_close(output.cpu(), torch.full((64, 64), fill * 3 + 2), rtol=0, atol=0)
                check(f"cuda:{index}/bf16_matmul", matmul)
                check(f"cuda:{index}/flash_sdpa", attention)
                check(f"cuda:{index}/compile_pointwise_twice", compile_pointwise)
    except Exception as error:
        report["checks"].append({"name": "setup", "status": "fail", "error": f"{type(error).__name__}: {error}"})
    passed = all(item["status"] == "pass" for item in report["checks"])
    report["status"] = "pass" if passed else "fail"
    if rank == 0:
        print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
