"""Wan's official atomic pipeline, including the original offline batch path."""

import time

from ..flux import FluxBackend


class WanPipelineBackend(FluxBackend):
    model = "wan"
    output_field = "frames"
    defaults = dict(FluxBackend.defaults, height=480, width=832, frames=81,
                    steps=50, guidance_scale=5.)

    def _arguments(self, spec):
        arguments = super()._arguments(spec)
        arguments.update(num_frames=spec["frames"], negative_prompt=[spec["negative_prompt"]] * spec["batch_size"])
        return arguments

    def _encode(self, arguments):
        if self.config.get("text_encoder_device", "cpu") != "cpu":
            return {}
        started = time.perf_counter()
        prompt, negative = self.pipe.encode_prompt(prompt=arguments.pop("prompt"),
            negative_prompt=arguments.pop("negative_prompt"),
            do_classifier_free_guidance=arguments["guidance_scale"] > 1,
            num_videos_per_prompt=1, max_sequence_length=512, device="cpu", dtype=self.torch.float32)
        for value in (prompt, negative):
            if value is not None and not bool(self.torch.isfinite(value).all()):
                raise ValueError("nonfinite CPU prompt embeddings")
        encoded = time.perf_counter()
        prompt = prompt.to(device=self.device, dtype=self.pipe.transformer.dtype)
        if negative is not None:
            negative = negative.to(device=self.device, dtype=self.pipe.transformer.dtype)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        for value in (prompt, negative):
            if value is not None and not bool(self.torch.isfinite(value).all()):
                raise ValueError("nonfinite transferred prompt embeddings")
        transferred = time.perf_counter()
        arguments.update(prompt_embeds=prompt, negative_prompt_embeds=negative)
        return dict(cpu_prompt_encode_ms=(encoded - started) * 1000,
                    prompt_embeddings_h2d_ms=(transferred - encoded) * 1000,
                    prompt_embeddings_reused=False)
