"""Wan execution through the pinned xFuser transformer wrapper."""

from pathlib import Path

from .runtime import XDitRuntime


def wrapper_class():
    from xfuser.model_executor.models.transformers.transformer_wan import xFuserWanTransformer3DWrapper
    return xFuserWanTransformer3DWrapper


def load_wan_transformer(model_path, dtype, *, local_files_only=True):
    path = Path(model_path).resolve() / "transformer"
    return wrapper_class().from_pretrained(
        str(path), torch_dtype=dtype, local_files_only=local_files_only)


def convert_wan_transformer(transformer):
    cls = wrapper_class()
    if type(transformer) is cls:
        return transformer
    from diffusers import WanTransformer3DModel
    if type(transformer) is not WanTransformer3DModel:
        raise ValueError("xDiT Wan conversion requires an unmodified WanTransformer3DModel")
    converted = cls.from_config(transformer.config)
    converted.load_state_dict(transformer.state_dict())
    converted.to(device=transformer.device, dtype=transformer.dtype)
    return converted.train(transformer.training)


class XDitWanEngine:
    def __init__(self, worker, groups=None):
        self.worker = worker
        self.groups = groups
        self.runtime = None
        self._layout = None
        self._closed = False
        worker.pipe.transformer = convert_wan_transformer(worker.pipe.transformer)
        self._start(groups)

    @property
    def model(self):
        return self.worker.pipe.transformer

    @property
    def revision(self):
        return self.runtime.revision

    def _start(self, groups):
        self.runtime = XDitRuntime(
            self.worker.pipe,
            model="wan",
            device=self.worker.device,
            attention=self.worker.attention,
            groups=groups,
            dtype=self.model.dtype,
        )

    def attach_groups(self, groups):
        self.runtime.close()
        self.groups = groups
        self._layout = None
        self._start(groups)

    def activate(self, state, layout=None):
        from ..wan.parallel import FixedLayout
        if layout is None:
            layout = self._layout or FixedLayout((self.runtime.pool.rank,))
        self.runtime.activate(
            layout,
            height=state.spec["height"],
            width=state.spec["width"],
            num_frames=state.spec["frames"],
            num_inference_steps=state.spec["steps"],
        )
        self._layout = layout
        return layout

    def _timestep(self, state):
        from ..wan.runtime import model_timestep
        return model_timestep(
            self.worker.torch,
            state.latents,
            state.scheduler.timesteps[state.step_index],
            self.worker.expand_timesteps,
        )

    def forward(self, state, branch):
        self.activate(state)
        encoder = state.prompt_embeds if branch == "cond" else state.negative_prompt_embeds
        with self.worker.pipe.transformer.cache_context(branch):
            return self.worker.pipe.transformer(
                hidden_states=state.latents.to(self.model.dtype),
                timestep=self._timestep(state),
                encoder_hidden_states=encoder,
                attention_kwargs=None,
                return_dict=False,
            )[0]

    def begin_packet(self, state, branch):
        self.activate(state)
        encoder = state.prompt_embeds if branch == "cond" else state.negative_prompt_embeds
        return self.model.begin_packet(
            state.latents.to(self.model.dtype),
            self._timestep(state),
            encoder,
        )

    def forward_block(self, packet, index):
        return self.model.forward_block(packet, index)

    def prediction(self, packet):
        return self.model.prediction(packet)

    def info(self):
        return self.runtime.info()

    def close(self):
        if not self._closed:
            self.runtime.close()
            self._closed = True


class XDitWanParallelBackend:
    capabilities = dict(
        backend="xdit-xfuser-usp",
        fixed_sp_degrees=[1, 2, 4, 8],
        explicit_sp1_bypass=True,
        local_layout_hook_binding=True,
        request_group_cache_gate=True,
        distributed_request_transport=False,
        gpu_validated=False,
        cache_off_group01_transport=True,
        cached_feature_transport=False,
        elastic_model_equivalence_validated=False,
        block_quantum_safe_points=True,
        step_boundary_regroup="host_staged_file",
        cfg_parallel=True,
    )

    def __init__(self, handle, groups):
        from ..wan.parallel import RequestGroupGate
        self.handle = handle
        self.groups = groups
        self.gate = RequestGroupGate(handle, groups)
        self.worker = None
        self._binding = None
        self._bound_epoch = None

    def attach(self, worker):
        from ..wan.parallel import ParallelContractError, bind_request_group_similarity
        if self.worker is not None:
            raise ParallelContractError("backend is already attached")
        if worker.engine_name != "xdit" or worker.xdit is None:
            raise ParallelContractError("the xDiT parallel backend requires engine=xdit")
        torch = worker.torch
        contract = self.groups.backend == "gloo" and torch.device(worker.device).type == "cpu"
        if not contract and (self.groups.backend != "nccl" or not torch.cuda.is_available()):
            raise ParallelContractError("Wan xDiT execution requires NCCL and an available GPU; Gloo is CPU-only")
        if worker.attention != "flash" and not contract:
            raise ParallelContractError("Wan xDiT SP requires explicit SDPA Flash attention")
        self.worker = worker
        try:
            if worker.xdit.groups is not self.groups:
                worker.xdit.attach_groups(self.groups)
            self._binding = bind_request_group_similarity(worker.manager, self.gate)
        except BaseException:
            self.close()
            raise
        return self

    def _abort(self, error):
        if self.handle.active is not None and not self.handle.failed:
            try:
                self.gate.consensus(dict(kind="abort", reason=type(error).__name__))
            except BaseException:
                pass
        self.handle.fail()

    def _require_active(self):
        from ..wan.parallel import ParallelContractError
        if (self.worker is None or self.handle.active is None or self.handle.failed
                or self._bound_epoch != self.handle.current.epoch):
            raise ParallelContractError("Wan execution is outside its committed step")

    def exchange_predictions(self, prediction):
        from ..wan.parallel import ParallelContractError
        self._require_active()
        layout = self.handle.current
        if layout.cfg != 2:
            raise ParallelContractError("prediction exchange needs a CFG-parallel layout")
        torch = self.worker.torch
        try:
            chunks = [torch.empty_like(prediction) for _ in layout.ranks]
            torch.distributed.all_gather(chunks, prediction.contiguous(), group=self.groups.group(layout.ranks))
        except Exception as error:
            self._abort(error)
            raise ParallelContractError("CFG prediction exchange failed: " + str(error)) from error
        return chunks[0], chunks[layout.sp_degree]

    def begin_step(self, state, *, pair=False):
        from ..wan.parallel import ParallelContractError, cache_manager_manifest
        if self.worker is None:
            raise ParallelContractError("attach a worker first")
        layout = self.handle.current
        self.worker.xdit.activate(state, layout)
        self.handle.begin_step(state.request_id, state.step_index,
                               cfg="paired" if state.spec["guidance_scale"] > 1 else "cond", pair=pair)
        self._bound_epoch = layout.epoch
        try:
            context_manifest = {}
            for name, context in state.contexts.items():
                with self.worker.manager.enter_context(context):
                    context_manifest[name] = cache_manager_manifest(self.worker.manager)
            descriptor = dict(contexts=context_manifest, cache_enabled=state.cache_enabled,
                              spec=state.spec, latents_shape=list(state.latents.shape),
                              latents_dtype=str(state.latents.dtype), needs_anchor=state.needs_anchor)
            self.gate.pre_step(descriptor)
            if any("invalid" in value for value in context_manifest.values()):
                raise ParallelContractError("unsupported cache state in pre-step consensus")
        except BaseException as error:
            self._abort(error)
            raise

    def end_step(self, request_id=None):
        self._require_active()
        try:
            self.worker._sync()
        except BaseException as error:
            self._abort(error)
            raise
        self.gate.consensus(dict(kind="step_complete") if request_id is None else dict(kind="step_complete", request_id=request_id))
        self.handle.end_step(request_id)
        if self.handle.active is None:
            self._bound_epoch = None

    def reduce_flag(self, flag):
        self._require_active()
        layout = self.handle.current
        if layout.degree > 1:
            torch = self.worker.torch
            try:
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX,
                                             group=self.gate.groups.group(layout.ranks))
            except Exception as error:
                self.handle.fail()
                from ..wan.parallel import ParallelContractError
                raise ParallelContractError("request-group doorbell reduction failed: " + str(error)) from error
        return flag

    def quantum_point(self, position):
        self._require_active()
        self.gate.consensus(dict(kind="block_quantum", position=position))

    def step(self, state):
        from ..wan.parallel import ParallelContractError
        if self.worker is None:
            raise ParallelContractError("attach a worker first")
        if self.handle.current.cfg == 2:
            raise ParallelContractError("CFG-parallel layouts run through the block executor")
        with self.worker._lock:
            self.begin_step(state)
            try:
                result = self.worker.step(state)
                self.end_step()
                return result
            except BaseException as error:
                self._abort(error)
                raise

    def migrate_cache_off(self, state, source_ranks, target_ranks, *, request_id, step_index, expected_epoch):
        from ..wan.parallel import CacheOffGroupMigration, ParallelContractError
        if self.worker is None:
            raise ParallelContractError("attach a worker before migrating")
        with self.worker._lock:
            return CacheOffGroupMigration(self.worker, self.handle, self.groups).migrate(
                state, source_ranks, target_ranks, request_id=request_id,
                step_index=step_index, expected_epoch=expected_epoch)

    def close(self):
        if self._binding is not None:
            self._binding.close()
            self._binding = None
        self.worker = None


def pipeline_class():
    import inspect
    from xfuser.core.distributed import get_runtime_state
    from xfuser.model_executor.pipelines.base_pipeline import xFuserPipelineBaseWrapper
    from .pipeline import ServingPipelineWrapper

    class WanXDitPipeline(ServingPipelineWrapper, xFuserPipelineBaseWrapper):
        def convert_transformer(self, transformer):
            return convert_wan_transformer(transformer)

        def _convert_scheduler(self, scheduler):
            if type(scheduler).__name__ != "UniPCMultistepScheduler":
                raise ValueError("DITSERVE atomic Wan requires the pinned UniPC scheduler")
            return scheduler

        def __call__(self, *args, **kwargs):
            import torch

            arguments = inspect.signature(self.module.__call__).bind(*args, **kwargs)
            arguments.apply_defaults()
            values = arguments.arguments
            prompt = values.get("prompt")
            embeds = values.get("prompt_embeds")
            batch = len(prompt) if isinstance(prompt, list) else 1 if embeds is None else embeds.shape[0]
            get_runtime_state().set_video_input_parameters(
                height=values["height"],
                width=values["width"],
                num_frames=values["num_frames"],
                batch_size=batch,
                num_inference_steps=values["num_inference_steps"],
                split_text_embed_in_sp=False,
            )
            with torch.no_grad():
                return self.module(*args, **kwargs)

    return WanXDitPipeline


def attach_pipeline(pipeline, device, *, attention="math", groups=None):
    return XDitRuntime(
        pipeline,
        model="wan",
        device=device,
        attention=attention,
        groups=groups,
        dtype=pipeline.transformer.dtype,
        wrapper=pipeline_class(),
    )
