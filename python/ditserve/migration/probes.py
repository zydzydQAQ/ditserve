"""Bounded CUDA/model probes, invoked only by the migration supervisor."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time


def cuda_probe():
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (8, 0):
        raise ValueError("probe requires exactly one visible A100")
    if torch.cuda.get_device_properties(0).total_memory < 39000 * 1024 ** 2:
        raise ValueError("probe requires a full 40GB/80GB A100; MIG slices are unsupported")
    torch.cuda.set_device(0)
    outcomes = {}
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(64, 64, device="cuda", dtype=dtype)
            if not torch.isfinite(x @ x).all():
                raise ValueError("nonfinite matrix multiplication")
            q = torch.randn(1, 2, 32, 64, device="cuda", dtype=dtype)
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
                output = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            if not torch.isfinite(output).all():
                raise ValueError("nonfinite Flash SDPA")
            outcomes[str(dtype)] = "passed"
        x = torch.randn(256, device="cuda")
        def kernel(tensor):
            return tensor.sin() + tensor.square()
        compiled = torch.compile(kernel, backend="inductor", fullgraph=True)
        expected = kernel(x)
        for _ in range(2):
            torch.testing.assert_close(compiled(x), expected)
        torch.cuda.synchronize()
    return dict(status="passed", name=torch.cuda.get_device_name(0), dtypes=outcomes,
                cuda=torch.version.cuda, torch=torch.__version__, cuda_inductor="passed")


def nccl_probe():
    import torch
    import torch.distributed as dist
    rank, count = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if torch.cuda.device_count() != count:
        raise ValueError("CUDA device count does not match NCCL ranks")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        value = torch.tensor([float(rank)], device="cuda")
        dist.all_reduce(value)
        torch.cuda.synchronize()
        expected = count * (count - 1) / 2
        if value.item() != expected:
            raise ValueError("incorrect NCCL sum")
        dist.barrier()
        return dict(status="passed", rank=rank, ranks=count, sum=value.item(), scope="communication_only")
    finally:
        dist.destroy_process_group()


def wan_probe(config):
    import numpy as np
    from ..gpu.worker import WanStepWorker
    from ..gpu.quality import evaluate_quality
    worker = WanStepWorker(model_path=config["models"]["wan"]["path"], attention="flash")
    spec = dict(request_id="migration-smoke", prompt="A red ball on a table.", seed=7,
                height=256, width=448, frames=17, steps=4, cache=False)
    state = worker.prepare(spec)
    try:
        while state.step_index < spec["steps"]:
            worker.step(state)
        video = worker.finish(state)
        if not np.isfinite(video).all():
            raise ValueError("nonfinite video")
        quality = evaluate_quality(video[:, :1], video[:, :1], lpips_device="cpu")
        if quality["status"] != "ok" or quality["lpips_mean"] != 0:
            raise ValueError("offline LPIPS self-comparison failed: " + str(quality))
        return dict(status="passed", shape=list(video.shape), finite=True, steps=4,
                    lpips_self=quality["lpips_mean"], scope="load/encode/denoise/decode, not quality qualification")
    finally:
        worker.release(state)


def sana_probe(config):
    from ..directions.sana import SanaAdapter
    adapter = SanaAdapter(config["models"]["sana"]["path"])
    try:
        image, details = adapter.generate("A red ball on a table.", seed=7)
        return dict(status="passed", shape=list(image.shape), **details)
    finally:
        adapter.close()


def sessions_probe(config):
    import numpy as np
    from ..sessions.runtime import MistralRuntime, SanaRuntime, cache_comparison
    prompt = "A red ball on a table."
    llm = MistralRuntime(config["models"]["mistral"]["path"], profile="a100_fp32")
    messages = [dict(role="user", content="Write a short image prompt about a red ball.")]
    state = llm.prepare(messages, max_new_tokens=4)
    try:
        llm.advance(state)
        before = llm.snapshot(state)
        llm.offload_to_cpu(state)
        llm.restore(state)
        after = llm.snapshot(state)
        comparison = cache_comparison(before["cache"], after["cache"])
        if not comparison["passed"] or not comparison["exact"]:
            raise ValueError("KV offload/restore is not exact")
        while not state.done:
            llm.advance(state)
        text = llm.finish(state)
        tokens = list(state.generated_ids)
        reference = llm.reference(messages, max_new_tokens=4)
        if tokens != reference["token_ids"]:
            raise ValueError("resumed tokens differ from native reference")
        prompt = text.strip() or prompt
    finally:
        llm.release(state)
        llm.close()
    # Load separately to keep smoke memory bounded even on A100 40GB.
    sana = SanaRuntime(config["models"]["sana"]["path"], profile="a100_fp32")
    state = sana.prepare(prompt, seed=7)
    try:
        sana.advance(state)
        sana.offload_to_cpu(state)
        sana.restore(state)
        while not state.done:
            sana.advance(state)
        actual = sana.finish(state)
        expected = sana.reference(prompt, seed=7)["output"]
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
            raise ValueError("resumed Sana output differs from native reference")
        return dict(status="passed", token_ids=tokens, kv_exact=True, sana_shape=list(actual.shape),
                    profile="a100_fp32", scope="environment_only; new study qualification still required")
    finally:
        sana.release(state)
        sana.close()


def longlive_probe(config):
    import importlib.metadata
    import sys
    import torch
    import flash_attn
    for package, expected in (("torch", "2.8.0+cu126"), ("diffusers", "0.31.0"),
                              ("flash-attn", "2.8.3")):
        if importlib.metadata.version(package).split("+cu12torch")[0] != expected:
            raise ValueError("LongLive environment version mismatch: " + package)
    if Path(sys.prefix).resolve() != Path(config["models"]["longlive"]["python"]).parent.parent.resolve():
        raise ValueError("LongLive must use its isolated environment")
    if not torch._C._GLIBCXX_USE_CXX11_ABI or flash_attn.__version__ != "2.8.3":
        raise ValueError("LongLive FlashAttention ABI/version mismatch")
    import numpy as np
    from ..directions.direction3 import NativeLongLive
    started = time.monotonic()
    def check():
        if time.monotonic() - started > 600:
            raise TimeoutError("LongLive acceptance deadline reached")
    native = NativeLongLive(config, check)
    session = native.new_session("migration", 7, 1)
    try:
        native.condition(session, "A red ball rests on a plain white table.", 0)
        _, pixels, metrics = native.chunk(session)
        if not np.isfinite(pixels).all():
            raise ValueError("nonfinite LongLive output")
        return dict(status="passed", identity=native.identity, shape=list(pixels.shape), metrics=metrics,
                    scope="one native block; no semantic or research qualification")
    finally:
        native.release(session)
        native.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=("cuda", "nccl", "wan", "sana", "sessions", "longlive"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.probe in ("cuda", "nccl"):
        result = globals()[args.probe + "_probe"]()
    else:
        if args.config is None:
            parser.error("model probes require --config")
        config = json.loads(args.config.read_text())
        if config.get("schema_version") != "ditserve.migration.environment.v1":
            raise ValueError("expected a migration environment configuration")
        os.environ["TORCH_HOME"] = config["torch_home"]
        result = globals()[args.probe + "_probe"](config)
    print("MIGRATION_RESULT=" + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
