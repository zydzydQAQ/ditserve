"""Pinned model loading and decoding; no experiment CLI or orchestration."""
import importlib

def _sdpa_math_context(torch):
    """Pin every GPU execution arm to SDPA MATH, including warmup/preflight."""
    attention = getattr(getattr(torch, "nn", None), "attention", None)
    if attention is not None and hasattr(attention, "sdpa_kernel"):
        return attention.sdpa_kernel([attention.SDPBackend.MATH])
    options = dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    try:
        return torch.backends.cuda.sdp_kernel(enable_cudnn=False, **options)
    except TypeError:
        return torch.backends.cuda.sdp_kernel(**options)

def _wan_pipeline_class(base_class):
    """Keep denoising on the DiT device with an explicitly resident CPU encoder.

    DiffusionPipeline.device chooses the first module in a signature-derived
    set; with mixed placement that may be the CPU encoder. Only these placement
    properties change. The inherited Wan call/encoding/sampling are unchanged.
    The Wan prefix also preserves cache-dit's documented adapter discovery.
    """
    class WanExplicitExecutionPipeline(base_class):
        @property
        def device(self):
            return self.transformer.device

        @property
        def _execution_device(self):
            return self.transformer.device

    return WanExplicitExecutionPipeline

def _pipeline(torch, diffusers, args):
    load = {"torch_dtype": getattr(torch, args.dtype), "local_files_only": args.local_files_only}
    if args.revision:
        load["revision"] = args.revision
    if args.model in ("flux", "sd35"):
        pipeline_class = diffusers.FluxPipeline if args.model == "flux" else diffusers.StableDiffusion3Pipeline
        pipe = pipeline_class.from_pretrained(args.model_id, **load)
    else:
        vae_args = dict(load, torch_dtype=getattr(torch, args.vae_dtype))
        vae = diffusers.AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", **vae_args)
        # Load directly in the intended precision: FP16 loading followed by a
        # cast back to FP32 would irreversibly round the encoder weights.
        transformers = importlib.import_module("transformers")
        encoder_args = dict(load, torch_dtype=getattr(torch, args.text_encoder_dtype))
        encoder = transformers.UMT5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder", **encoder_args)
        pipeline_class = (_wan_pipeline_class(diffusers.WanPipeline)
                          if args.text_encoder_device == "cpu" else diffusers.WanPipeline)
        components = {}
        if getattr(args, "engine", "legacy") == "xdit":
            from .xdit.wan import load_wan_transformer
            components["transformer"] = load_wan_transformer(
                args.model_id, getattr(torch, args.dtype), local_files_only=args.local_files_only)
        pipe = pipeline_class.from_pretrained(args.model_id, vae=vae, text_encoder=encoder, **components, **load)
    if args.model == "wan":
        # Never pipe.to(cuda) here: it transiently puts the large CPU encoder on
        # the GPU and can OOM before it is moved back.
        pipe.transformer.to(device=args.device, dtype=getattr(torch, args.dtype))
        if getattr(pipe, "transformer_2", None) is not None:
            pipe.transformer_2.to(device=args.device, dtype=getattr(torch, args.dtype))
        pipe.vae.to(device=args.device, dtype=getattr(torch, args.vae_dtype))
        encoder_device = "cpu" if args.text_encoder_device == "cpu" else args.device
        pipe.text_encoder.to(device=encoder_device, dtype=getattr(torch, args.text_encoder_dtype))
    else:
        pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    return pipe



def decode_final_latent(torch, pipe, latent):
    """Exact arithmetic/order from diffusers 0.36 WanPipeline's decode tail."""
    latent = latent.to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
    inverse_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
    normalized = latent / inverse_std + mean
    video = pipe.vae.decode(normalized, return_dict=False)[0]
    array = pipe.video_processor.postprocess_video(video, output_type="np")
    del video, normalized, mean, inverse_std
    if any(value is not None for name in ("_feat_map", "_enc_feat_map")
           for value in getattr(pipe.vae, name, [])):
        raise RuntimeError("VAE retained temporal feature caches after decode")
    return array


class ChunkedDecode:
    """Wan VAE decode suspendable at latent-frame boundaries.

    AutoencoderKLWan decodes causally one latent frame at a time and carries
    its temporal feature cache as an explicit argument. This replays diffusers
    0.36 ``_decode`` in the same order with the cache held per request, so the
    GPU can run other work between chunks; the result matches
    ``decode_final_latent``. The Wan2.2 VAE's pixel patches are folded back after
    the last chunk, as ``_decode`` does. Tiled decoding is not supported.
    """

    def __init__(self, torch, pipe, latent):
        vae = pipe.vae
        if getattr(vae, "use_tiling", False) or latent.shape[0] != 1:
            raise ValueError("chunked decode supports the untiled single-video Wan VAE")
        self.torch, self.pipe = torch, pipe
        latent = latent.to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        inverse_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        self.x = vae.post_quant_conv(latent / inverse_std + mean)
        self.frames, self.index, self.chunks = self.x.shape[2], 0, []
        self.cache = [None] * vae._conv_num

    @property
    def done(self):
        return self.index == self.frames

    def advance(self, count=1):
        """Decode up to ``count`` latent frames; returns True after the last one."""
        if type(count) is not int or count < 1:
            raise ValueError("decode chunk count must be a positive integer")
        decoder = self.pipe.vae.decoder
        for _ in range(min(count, self.frames - self.index)):
            i = self.index
            chunk = self.x[:, :, i:i + 1, :, :]
            if i == 0:
                out = decoder(chunk, feat_cache=self.cache, feat_idx=[0], first_chunk=True)
            else:
                out = decoder(chunk, feat_cache=self.cache, feat_idx=[0])
            self.chunks.append(out)
            self.index += 1
        return self.done

    def finish(self):
        if not self.done:
            raise ValueError("decode has remaining latent frames")
        video = self.torch.cat(self.chunks, 2)
        patch = self.pipe.vae.config.patch_size
        if patch is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
            video = unpatchify(video, patch_size=patch)
        video = self.torch.clamp(video, min=-1.0, max=1.0)
        array = self.pipe.video_processor.postprocess_video(video, output_type="np")
        self.x = self.cache = self.chunks = None
        return array
