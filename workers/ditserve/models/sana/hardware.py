"""Explicit hardware variants. Importing this module does not initialize CUDA."""

from contextlib import nullcontext
import importlib
import inspect
import time
from pathlib import Path

from .runtime import SanaAdapter


class SchedulingSana(SanaAdapter):
    def __init__(self, model_path, device="cuda:0", profile="a100_bf16"):
        self.profile = profile
        if profile == "a100_bf16":
            super().__init__(model_path, device)
            return
        if profile != "v100_fp32":
            raise ValueError("unknown explicit Sana hardware profile")
        self.torch = importlib.import_module("torch")
        self.device = device
        if self.torch.cuda.get_device_capability(device) != (7, 0):
            raise ValueError("v100_fp32 requires a Volta sm70 device")
        path = Path(model_path).resolve()
        if not (path / "model_index.json").is_file():
            raise ValueError("local Sana snapshot missing")
        diffusers = importlib.import_module("diffusers")
        self.pipe = diffusers.SanaSprintPipeline.from_pretrained(
            str(path), torch_dtype=self.torch.float32, local_files_only=True)
        if (type(self.pipe.scheduler).__name__ != "SCMScheduler"
                or type(self.pipe.transformer).__name__ != "SanaTransformer2DModel"):
            raise ValueError("unexpected Sana model or sampler")
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        for component in (self.pipe.text_encoder, self.pipe.transformer, self.pipe.vae):
            if any(p.is_floating_point() and p.dtype != self.torch.float32 for p in component.parameters()):
                raise ValueError("V100 Sana component did not remain FP32")
        self._instruction = inspect.signature(self.pipe.__call__).parameters["complex_human_instruction"].default
        self._owner, self.model_path = object(), str(path)

    def _attention(self):
        if self.profile == "v100_fp32":
            return self.torch.nn.attention.sdpa_kernel([self.torch.nn.attention.SDPBackend.MATH])
        return nullcontext()

    def encode(self, prompt, check_budget=lambda: None):
        with self._attention():
            return super().encode(prompt, check_budget)

    def generate(self, prompt, *, seed=0, encoded=None, check_budget=lambda: None):
        with self._attention():
            if encoded is not None:
                return super().generate(prompt, seed=seed, encoded=encoded, check_budget=check_budget)
            started = time.perf_counter()
            encoded = super().encode(prompt, check_budget)
            encode_end = (time.perf_counter() - started) * 1000
            image, service = super().generate(prompt, seed=seed, encoded=encoded, check_budget=check_budget)
            service.update(ms=(time.perf_counter() - started) * 1000, prompt_preencoded=False,
                           encode_ms=encode_end, encode_start_offset_ms=0., encode_end_offset_ms=encode_end,
                           encoding_device=self.device)
            return image, service
