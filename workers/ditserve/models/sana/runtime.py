"""Local-only official SANA-Sprint 0.6B adapter."""

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

    def storage_details(self):
        """Conservatively charge backing storage, including noncompact views.

        Aliased tensors are charged separately: this may overestimate persistent
        storage but cannot admit a view by counting only its visible elements.
        This is a tensor-storage bound, not a CUDA allocator-reservation bound.
        """
        details = []
        for name in ("embeddings", "attention_mask"):
            tensor = getattr(self, name)
            shape = list(tensor.shape)
            numel, element_size = tensor.numel(), tensor.element_size()
            storage_bytes = tensor.untyped_storage().nbytes()
            if (any(type(size) is not int or size < 0 for size in shape)
                    or type(numel) is not int or numel < 1
                    or type(element_size) is not int or element_size < 1
                    or type(storage_bytes) is not int or storage_bytes < 0):
                raise ValueError("invalid Sana prompt tensor storage metadata")
            product = 1
            for size in shape:
                product *= size
            if product != numel:
                raise ValueError("Sana prompt tensor shape does not match numel")
            logical_bytes = numel * element_size
            details.append(dict(name=name, shape=shape, numel=numel,
                                element_size=element_size, storage_bytes=storage_bytes,
                                logical_bytes=logical_bytes,
                                accounted_bytes=max(storage_bytes, logical_bytes)))
        return details


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

    def encode_and_generate(self, prompt, *, seed=0, on_encoded=None, check_budget=lambda: None):
        """Submit a cache miss on one stream, synchronizing only its endpoints.

        ``on_encoded`` admits/evicts cache entries before generation, while the
        new packet remains live for this request even if admission is bypassed.
        Both reported components are disjoint host wall intervals. Encoding is
        submission time, not completed GPU time; generation includes its tail,
        cache admission, final synchronization, and output/budget checks.
        """
        check_budget()
        self._sync()
        started = time.perf_counter()
        stream = self.torch.cuda.current_stream(self.device)
        try:
            with self.torch.cuda.stream(stream), self.torch.inference_mode():
                embeddings, mask = self.pipe.encode_prompt(
                    prompt=prompt, device=self.device, num_images_per_prompt=1,
                    clean_caption=False, max_sequence_length=300, complex_human_instruction=self._instruction)
                encode_ms = (time.perf_counter() - started) * 1000
                encoded = SanaPrompt(prompt, embeddings, mask, encode_ms, self._owner)
                check_budget()
                if on_encoded is not None:
                    on_encoded(encoded)
                check_budget()
                arguments = dict(RECIPE, prompt_embeds=encoded.embeddings,
                                 prompt_attention_mask=encoded.attention_mask)
                generator = self.torch.Generator(device="cpu").manual_seed(seed)

                def boundary(pipe, index, timestep, kwargs):
                    check_budget()
                    return kwargs

                result = self.pipe(**arguments, generator=generator, output_type="np", callback_on_step_end=boundary)
        finally:
            # Also drain submitted encoding if admission, budget, or generation
            # fails before the caller clears its cache and releases this packet.
            self._sync()
        numpy = importlib.import_module("numpy")
        if not numpy.isfinite(result.images).all():
            raise RuntimeError("nonfinite Sana image")
        check_budget()
        generation_ms = (time.perf_counter() - started) * 1000 - encode_ms
        return encoded, result.images, dict(
            ms=encode_ms + generation_ms, encode_ms=encode_ms, generation_ms=generation_ms,
            prompt_preencoded=True, recipe=dict(RECIPE), finite=True,
            timing_scope="fused_miss_host_wall",
            encode_timing_scope="host encoding submission; GPU encoding may still be running",
            generation_timing_scope="remaining wall including encoding tail, cache admission, generation, final sync, finite and budget checks")

    def close(self):
        self._sync()
        self.pipe = None
