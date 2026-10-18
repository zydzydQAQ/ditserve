"""A weight-free Wan worker and request on CPU, for codec and executor tests."""

import threading


def fixture(*, boundary=16):
    """A registered request after ``boundary`` synthetic UniPC steps."""
    import torch
    import cache_dit
    from diffusers import UniPCMultistepScheduler
    from ditserve.models.wan.runtime import WanStepWorker, RequestState, normalize_spec
    worker = WanStepWorker.__new__(WanStepWorker)
    worker.torch, worker.cache_dit, worker.device = torch, cache_dit, "cpu"
    worker._owner, worker._states = object(), {}
    worker._lock = threading.RLock()
    worker._sync = lambda: None
    worker._scheduler_type = UniPCMultistepScheduler
    spec = normalize_spec(dict(request_id="request-a", prompt="a kettle", seed=101,
                               height=32, width=32, frames=5, steps=32, guidance_scale=5.))
    scheduler = UniPCMultistepScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(32, device="cpu")
    state = RequestState(spec["request_id"], spec, scheduler=scheduler,
                         latents=torch.arange(512, dtype=torch.float32).reshape(1, 16, 2, 4, 4) / 512,
                         prompt_embeds=torch.ones(1, 512, 4096, dtype=torch.float16),
                         negative_prompt_embeds=torch.zeros(1, 512, 4096, dtype=torch.float16),
                         generator=torch.Generator().manual_seed(spec["seed"]), owner=worker._owner)
    worker._states[state.request_id] = state
    for _ in range(boundary):
        index = state.step_index
        prediction = torch.full_like(state.latents, .05 + .001 * index, dtype=torch.float16)
        state.latents = state.scheduler.step(prediction, state.scheduler.timesteps[index], state.latents,
                                            return_dict=False)[0]
        state.step_index += 1
    return worker, state
