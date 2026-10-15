"""Local-only official SANA-Sprint 0.6B adapter for direction 1."""

import dataclasses
import importlib
import inspect
import time
from pathlib import Path


MODEL_ID = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
RECIPE = dict(height=1024, width=1024, num_inference_steps=2, guidance_scale=4.5,
              max_timesteps=1.57080, intermediate_timesteps=1.3,
              num_images_per_prompt=1, clean_caption=False, use_resolution_binning=True,
              max_sequence_length=300)


@dataclasses.dataclass
class SanaPrompt:
    prompt: str
    embeddings: object
    attention_mask: object
    encode_ms: float
    owner: object = dataclasses.field(repr=False)


class SanaAdapter:
    """Resident BF16 components; no implicit download, offload or quantization."""

    def __init__(self, model_path, device="cuda:0"):
        path = Path(model_path).resolve()
        if not (path / "model_index.json").is_file():
            raise ValueError("Sana requires a complete local model snapshot")
        self.torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
        self.device = device
        if self.torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("SANA-Sprint BF16 requires Ampere or newer; do not run on V100")
        self.pipe = diffusers.SanaSprintPipeline.from_pretrained(
            str(path), torch_dtype=self.torch.bfloat16, local_files_only=True)
        if (type(self.pipe.scheduler).__name__ != "SCMScheduler"
                or type(self.pipe.transformer).__name__ != "SanaTransformer2DModel"):
            raise ValueError("unexpected SANA-Sprint 0.6B scheduler/model configuration")
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self._instruction = inspect.signature(self.pipe.__call__).parameters["complex_human_instruction"].default
        self._owner = object()
        self.model_path = str(path)

    def _sync(self):
        self.torch.cuda.synchronize(self.device)

    def encode(self, prompt, check_budget=lambda: None):
        check_budget()
        self._sync()
        started = time.perf_counter()
        with self.torch.inference_mode():
            embeddings, mask = self.pipe.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1,
                clean_caption=False, max_sequence_length=300, complex_human_instruction=self._instruction)
        self._sync()
        result = SanaPrompt(prompt, embeddings, mask, (time.perf_counter() - started) * 1000, self._owner)
        check_budget()
        return result

    def generate(self, prompt, *, seed=0, encoded=None, check_budget=lambda: None):
        """Return image and synchronized wall cost; record preencoded scope explicitly."""
        if encoded is not None and (encoded.owner is not self._owner or encoded.prompt != prompt):
            raise ValueError("encoded Sana prompt belongs to another request/adapter")
        check_budget()
        arguments = dict(RECIPE)
        if encoded is None:
            arguments["prompt"] = prompt
        else:
            arguments.update(prompt_embeds=encoded.embeddings, prompt_attention_mask=encoded.attention_mask)
        generator = self.torch.Generator(device="cpu").manual_seed(seed)
        def boundary(pipe, index, timestep, kwargs):
            check_budget()
            return kwargs
        self._sync()
        started = time.perf_counter()
        with self.torch.inference_mode():
            result = self.pipe(**arguments, generator=generator, output_type="np", callback_on_step_end=boundary)
        self._sync()
        duration = (time.perf_counter() - started) * 1000
        numpy = importlib.import_module("numpy")
        if not numpy.isfinite(result.images).all():
            raise RuntimeError("nonfinite Sana image")
        check_budget()
        return result.images, dict(ms=duration, prompt_preencoded=encoded is not None,
                                   encode_ms=None if encoded is None else encoded.encode_ms,
                                   recipe=dict(RECIPE), finite=True)

    def close(self):
        self._sync()
        self.pipe = None
