"""Opt-in real-model CPU/GPU overlap experiments; never changes serving defaults.

Run inside the pinned runtime image, with model snapshots already available::

    python -m ditserve.evaluation.workflow_overlap --experiment E1 --frames 17 \
        --device cuda:1 --model-root /models --out /data/workflow/e1

Host intervals describe submission/waiting, not GPU execution. Use the emitted
NVTX ranges with a CUDA trace to establish kernel/memcpy overlap. All comparisons
include output correctness; no speedup is accepted merely from timers.
"""

import argparse
import contextlib
import ctypes
import dataclasses
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import statistics
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


PROMPTS = (
    "A small sailboat crossing a calm blue lake, mountains in the background, cinematic daylight.",
    "A red fox walking through a snowy forest, steady camera, natural light.",
    "A colorful kite flying over a green meadow, a gentle breeze, wide shot.",
    "A waterfall flowing between moss covered rocks, soft morning light, fixed camera.",
)


def bind_cpu_to_gpu(device,count=8):
    """Bind this process's threads to physical cores local to a visible GPU.

    Runs before torch/model loading. NVML's CPU affinity supplies the PCI/NUMA
    locality and sysfs collapses SMT siblings. CUDA_VISIBLE_DEVICES numeric and
    GPU UUID remapping is honored; unsupported/MIG mappings fail explicitly.
    """
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("CPU affinity requires an explicit cuda:N device")
    visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    ordinal=int(device[5:])
    identifier=str(ordinal)
    if visible is not None:
        identifiers=[item.strip() for item in visible.split(",") if item.strip()]
        if ordinal>=len(identifiers):
            raise ValueError("CUDA device outside CUDA_VISIBLE_DEVICES")
        identifier=identifiers[ordinal]
    library=ctypes.CDLL("libnvidia-ml.so.1")
    def call(name,argtypes,*arguments):
        function=getattr(library,name)
        function.restype=ctypes.c_int
        function.argtypes=argtypes
        code=function(*arguments)
        if code:
            raise RuntimeError(f"{name} failed with NVML status {code}")
    call("nvmlInit_v2",[])
    try:
        handle=ctypes.c_void_p()
        if identifier.startswith("GPU-"):
            call("nvmlDeviceGetHandleByUUID",[ctypes.c_char_p,ctypes.POINTER(ctypes.c_void_p)],identifier.encode(),ctypes.byref(handle))
        elif identifier.isdigit():
            call("nvmlDeviceGetHandleByIndex_v2",[ctypes.c_uint,ctypes.POINTER(ctypes.c_void_p)],int(identifier),ctypes.byref(handle))
        else:
            raise ValueError("unsupported CUDA_VISIBLE_DEVICES identifier: "+identifier)
        bits=ctypes.sizeof(ctypes.c_ulong)*8
        words=((os.cpu_count() or 1)+bits-1)//bits
        mask=(ctypes.c_ulong*words)()
        call("nvmlDeviceGetCpuAffinity",[ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(ctypes.c_ulong)],handle,words,mask)
        local={index for index in range(words*bits) if mask[index//bits]&(1<<(index%bits))}
        allowed=set(os.sched_getaffinity(0))
        cores={}
        for cpu in sorted(local&allowed):
            topology=Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key=(int((topology/"physical_package_id").read_text()),int((topology/"core_id").read_text()))
            cores.setdefault(key,cpu)
        selected=sorted(cores.values())[:count]
        if len(selected)!=count:
            raise RuntimeError(f"need {count} local physical CPU cores, have {len(selected)}")
        for task in Path("/proc/self/task").iterdir():
            try:
                os.sched_setaffinity(int(task.name),selected)
            except ProcessLookupError:
                pass
        uuid=ctypes.create_string_buffer(96)
        call("nvmlDeviceGetUUID",[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_uint],handle,uuid,len(uuid))
        return dict(policy="first local physical cores",device=device,gpu_uuid=uuid.value.decode(),
                    cuda_visible_devices=visible,nvml_identifier=identifier,
                    allowed_before=sorted(allowed),local_cpu_mask=sorted(local),bound_cpus=selected,
                    physical_core_count=count,source="NVML CPU affinity + Linux sysfs package/core topology")
    finally:
        call("nvmlShutdown",[])


def spec_for(args, index, prefix="request"):
    return dict(request_id=f"{prefix}-{index}", prompt=PROMPTS[index % len(PROMPTS)],
                negative_prompt="", seed=1234 + index, height=args.height, width=args.width,
                frames=args.frames, steps=args.steps, guidance_scale=5.0)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Timeline:
    """Thread-safe host ranges plus NVTX; never inserts device synchronization."""
    def __init__(self, device=None):
        from .workflow_metrics import PhaseRecorder
        self.recorder = PhaseRecorder(device, synchronize=False, reset_peaks=False)
        self.records = self.recorder.records
        self.device = device
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def phase(self, name, **metadata):
        with self.recorder.phase(name, **metadata):
            yield


def _hash_array(array):
    import numpy as np
    array = np.ascontiguousarray(array)
    if not np.isfinite(array).all():
        raise RuntimeError("nonfinite model output")
    return dict(sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                shape=list(array.shape), dtype=str(array.dtype), finite=True)


def _worker(args, device=None):
    device=device or args.device
    affinity=(bind_cpu_to_gpu(device,args.cpu_threads) if args.cpu_affinity=="auto"
              else dict(policy="none",bound_cpus=sorted(os.sched_getaffinity(0))))
    import torch
    from ditserve.models.wan.runtime import WanStepWorker
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(device)
    worker=WanStepWorker(device,str(Path(args.model_root)/"wan2.1-t2v-1.3b"),"flash")
    worker.workflow_cpu_affinity=affinity
    atomic_json(args.out/("cpu-affinity-"+device.replace(":","-")+".json"),affinity)
    return worker


def _encode(worker, spec, timeline, prefix):
    with timeline.phase(prefix + "/cpu_encode"):
        return worker.encode_prompt(spec)


def _finish_request(worker, state, timeline, prefix):
    with timeline.phase(prefix + "/vae_decode_and_d2h"):
        array = worker.finish(state)
    with timeline.phase(prefix + "/output_check"):
        result = _hash_array(array)
        result["latent"] = _hash_array(state.latents.detach().float().cpu().numpy())
    worker.release(state)
    return result


def _run_prepared(worker, spec, payload, timeline, prefix):
    with timeline.phase(prefix + "/prepare"):
        state = worker.prepare_encoded(spec, payload)
    for step in range(spec["steps"]):
        with timeline.phase(prefix + f"/dit.step.{step}"):
            worker.step(state)
    return _finish_request(worker, state, timeline, prefix)


def _warm(worker, args, timeline):
    spec = spec_for(args, 0, "warm")
    return _run_prepared(worker, spec, _encode(worker, spec, timeline, "warm"), timeline, "warm")


def _queue(worker, args, mode, timeline, prefix, *, indices=None, preencoded=False):
    indices = list(range(args.requests) if indices is None else indices)
    specs = [spec_for(args, index, prefix.replace("/", "-")) for index in indices]
    encoded = [_encode(worker, spec, timeline, prefix + f"/setup.{i}") for i, spec in enumerate(specs)] if preencoded else None
    records = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="workflow-encoder") as executor:
        start = time.perf_counter_ns()
        future = executor.submit(_encode, worker, specs[0], timeline, prefix + "/request.0") if mode == "prefetch" and not preencoded else None
        for i, spec in enumerate(specs):
            request_start = time.perf_counter_ns()
            name = prefix + f"/request.{i}"
            if encoded is not None:
                payload = encoded[i]
            elif mode == "prefetch":
                with timeline.phase(name + "/wait_encode"):
                    payload = future.result()
                if i + 1 < len(specs):
                    future = executor.submit(_encode, worker, specs[i+1], timeline, prefix + f"/request.{i+1}")
            else:
                payload = _encode(worker, spec, timeline, name)
            result = _run_prepared(worker, spec, payload, timeline, name)
            end = time.perf_counter_ns()
            records.append(dict(index=indices[i], start_ns=request_start, complete_ns=end,
                                queue_response_ms=(end-start)/1e6,
                                service_ms=(end-request_start)/1e6, **result))
        end = time.perf_counter_ns()
    return dict(start_ns=start, end_ns=end, wall_ms=(end-start)/1e6, requests=records,
                throughput_per_second=len(specs)/((end-start)/1e9),
                timing_scope="all queued at start; output checks included; model load and warmup excluded",
                preencoded=preencoded, mode=mode)


def _check_pair(reference, candidate):
    ref = {row["index"]: row for row in reference["requests"]}
    checked = []
    for row in candidate["requests"]:
        expected = ref[row["index"]]
        equal = row["sha256"] == expected["sha256"] and row["latent"]["sha256"] == expected["latent"]["sha256"]
        checked.append(dict(index=row["index"], exact=equal))
    if len(checked) != len(ref) or not all(row["exact"] for row in checked):
        raise RuntimeError("paired workload output differs: " + json.dumps(checked))
    return checked


def experiment_e1(args):
    timeline = Timeline(args.device)
    worker = _worker(args)
    _warm(worker, args, timeline)
    pairs = []
    for repeat in range(args.repeats):
        results = {}
        order = ("serial", "prefetch") if repeat % 2 == 0 else ("prefetch", "serial")
        for mode in order:
            results[mode] = _queue(worker, args, mode, timeline, f"E1/r{repeat}/{mode}")
            atomic_json(args.out / f"E1-r{repeat}-{mode}.json", results[mode])
        pairs.append(dict(repeat=repeat, order=order, modes=results,
                          correctness=_check_pair(results["serial"], results["prefetch"]),
                          speedup=results["serial"]["wall_ms"]/results["prefetch"]["wall_ms"]))
    return dict(pairs=pairs, phases=timeline.records)


def _e2_child(config, device, indices, preencoded, barrier, result_queue, prefix):
    args = argparse.Namespace(**config)
    timeline = Timeline(device)
    try:
        worker = _worker(args, device)
        _warm(worker, args, timeline)
        # Keep encoder work outside the common start only in the preencoded arm.
        cached = {}
        original_encode = worker.encode_prompt
        if preencoded:
            for index in indices:
                spec = spec_for(args, index, "E2")
                cached[index] = original_encode(spec)
            def reuse(spec):
                index = int(spec["request_id"].rsplit("-", 1)[-1])
                return dataclasses.replace(cached[index], spec=spec, consumed=False)
            worker.encode_prompt = reuse
        barrier.wait(timeout=args.timeout)
        result = _queue(worker, args, "prefetch", timeline, prefix, indices=indices)
        result.update(device=device, phases=timeline.records, preencoded=preencoded)
        result_queue.put(dict(ok=True, result=result))
    except BaseException:
        result_queue.put(dict(ok=False, device=device, error=traceback.format_exc()))
        try:
            barrier.abort()
        except Exception:
            pass


def experiment_e2(args):
    devices = ["cuda:" + part for part in args.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("E2 requires two distinct --devices")
    if args.device not in devices:
        raise ValueError("E2 selected single device must belong to --devices")
    other_device=next(device for device in devices if device!=args.device)
    context = multiprocessing.get_context("spawn")
    rows = []
    for preencoded in (True, False):
        for repeat in range(args.repeats):
            modes = {}
            # Every arm occupies every position over three repeats. Reverse
            # the next three-repeat cycle to counterbalance directed ordering.
            arms=("single","single_other","dual")
            offset=repeat%3
            order=arms[offset:]+arms[:offset]
            if (repeat//3)%2:
                order=tuple(reversed(order))
            for mode in order:
                active = ([args.device] if mode=="single" else
                          [other_device] if mode=="single_other" else devices)
                barrier, results = context.Barrier(len(active)), context.Queue()
                processes = [context.Process(target=_e2_child, args=(vars(args), device,
                    list(range(rank, args.requests, len(active))), preencoded, barrier, results,
                    f"E2/preencoded{preencoded}/r{repeat}/{mode}"))
                    for rank, device in enumerate(active)]
                for process in processes:
                    process.start()
                try:
                    outputs = [results.get(timeout=args.timeout) for _ in processes]
                    if not all(row["ok"] for row in outputs):
                        raise RuntimeError(json.dumps(outputs))
                    for process in processes:
                        process.join(timeout=60)
                        if process.exitcode != 0:
                            raise RuntimeError(f"E2 child did not exit cleanly: {process.exitcode}")
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join(30)
                workers = [row["result"] for row in outputs]
                start, end = min(row["start_ns"] for row in workers), max(row["end_ns"] for row in workers)
                modes[mode] = dict(workers=workers, devices=active, start_ns=start, end_ns=end, wall_ms=(end-start)/1e6,
                                   requests=sorted([request for worker in workers for request in worker["requests"]], key=lambda row:row["index"]))
                atomic_json(args.out / f"E2-preencoded{preencoded}-r{repeat}-{mode}.json", modes[mode])
            rows.append(dict(preencoded=preencoded, repeat=repeat, order=order, modes=modes,
                             correctness=_check_pair(modes["single"], modes["dual"]),
                             other_single_correctness=_check_pair(modes["single"], modes["single_other"]),
                             speedup=modes["single"]["wall_ms"]/modes["dual"]["wall_ms"],
                             other_single_speedup=modes["single_other"]["wall_ms"]/modes["dual"]["wall_ms"],
                             single_device=args.device,other_single_device=other_device,dual_devices=devices))
    return dict(pairs=rows)


def _step_async(worker, state, *, fixed_attention=False):
    """Same eager arithmetic as runtime.step, without device-global barriers.

    Caller owns this state and must wait before consuming output. The ordinary
    worker remains unchanged. Fixed attention is used only by E5, where global
    SDPA switches must not change while two model streams are active.
    """
    torch = worker.torch
    with torch.inference_mode(), (contextlib.nullcontext() if fixed_attention else worker._dit_attention_context()):
        timestep_scalar = state.scheduler.timesteps[state.step_index]
        timestep = timestep_scalar.expand(state.latents.shape[0])
        model_input = state.latents.to(worker.pipe.transformer.dtype)
        transformer = worker.pipe.transformer
        with transformer.cache_context("cond"):
            prediction = worker._original_forward(hidden_states=model_input, timestep=timestep,
                encoder_hidden_states=state.prompt_embeds, attention_kwargs=None, return_dict=False)[0]
        if state.spec["guidance_scale"] > 1:
            with transformer.cache_context("uncond"):
                unconditional = worker._original_forward(hidden_states=model_input, timestep=timestep,
                    encoder_hidden_states=state.negative_prompt_embeds, attention_kwargs=None, return_dict=False)[0]
            prediction = unconditional + state.spec["guidance_scale"] * (prediction - unconditional)
        state.latents = state.scheduler.step(prediction, timestep_scalar, state.latents, return_dict=False)[0]
        state.step_index += 1


def _decode_gpu(worker, state, *, fixed_attention=False):
    """Official VAE and postprocessing arithmetic, retain output on CUDA."""
    torch, pipe = worker.torch, worker.pipe
    with torch.inference_mode(), (contextlib.nullcontext() if fixed_attention else worker.p._sdpa_math_context(torch)):
        latent = state.latents.to(pipe.vae.dtype)
        mean = torch.tensor(pipe.vae.config.latents_mean, device=latent.device, dtype=latent.dtype).view(1,pipe.vae.config.z_dim,1,1,1)
        inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1,pipe.vae.config.z_dim,1,1,1)
        video = pipe.vae.decode(latent / inv_std + mean, return_dict=False)[0]
        result = pipe.video_processor.postprocess_video(video, output_type="pt")
        if result.ndim != 5:
            raise RuntimeError("unexpected pinned postprocess_video pt shape")
        return result.permute(0,1,3,4,2).contiguous()


def _host_inputs(worker, spec, payload, *, pinned):
    torch = worker.torch
    generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
    with torch.inference_mode():
        latent = worker.pipe.prepare_latents(1, worker.pipe.transformer.config.in_channels,
            spec["height"], spec["width"], spec["frames"], torch.float32, "cpu", generator, None)
        tensors = [payload.prompt_embeds, payload.negative_prompt_embeds, latent]
        return tuple(t.pin_memory() if pinned and t is not None else t for t in tensors)


def _install_inputs(worker, spec, host_inputs, *, stream=None):
    from ditserve.models.wan.runtime import RequestState, normalize_spec
    torch = worker.torch
    spec = normalize_spec(spec)
    with torch.inference_mode(), (torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()):
        # Both arms transfer FP32 and convert on CUDA, keeping conversion fixed.
        copied = [None if value is None else value.to(worker.device, non_blocking=stream is not None) for value in host_inputs]
        prompt, negative, latent = copied
        prompt = prompt.to(torch.float16)
        negative = None if negative is None else negative.to(torch.float16)
        scheduler = worker._scheduler_type.from_config(worker._scheduler_config)
        scheduler.set_timesteps(spec["steps"], device=worker.device)
        generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
        state = RequestState(spec["request_id"], spec, latents=latent, scheduler=scheduler,
            generator=generator, prompt_embeds=prompt, negative_prompt_embeds=negative, owner=worker._owner)
        worker._states[state.request_id] = state
        event = torch.cuda.Event()
        event.record(stream)
    return state, event


def experiment_e3(args):
    import torch
    worker, timeline = _worker(args), Timeline(args.device)
    official_reference=_warm(worker, args, timeline)
    specs = [spec_for(args, i, "E3") for i in range(args.requests)]
    payloads = [_encode(worker, spec, timeline, f"E3/input.{i}") for i,spec in enumerate(specs)]
    rows = []
    for repeat in range(args.repeats):
        modes = {}
        for mode in (("sync", "async") if repeat%2 == 0 else ("async", "sync")):
            setup = time.perf_counter_ns()
            hosts = [_host_inputs(worker, spec, payload, pinned=mode=="async") for spec,payload in zip(specs,payloads)]
            setup_ms = (time.perf_counter_ns()-setup)/1e6
            copy_stream = torch.cuda.Stream(device=args.device) if mode=="async" else None
            pending, records = {}, []
            start = time.perf_counter_ns()
            for i,spec in enumerate(specs):
                prefix = f"E3/r{repeat}/{mode}/request.{i}"
                if i not in pending:
                    with timeline.phase(prefix + "/h2d", bytes=sum(t.numel()*t.element_size() for t in hosts[i] if t is not None)):
                        pending[i] = _install_inputs(worker,spec,hosts[i],stream=copy_stream)
                state, ready = pending.pop(i)
                compute_stream=torch.cuda.current_stream(args.device)
                compute_stream.wait_event(ready)
                for tensor in (state.latents,state.prompt_embeds,state.negative_prompt_embeds):
                    if tensor is not None:
                        tensor.record_stream(compute_stream)
                handle = None
                if mode=="async" and i+1 < len(specs):
                    def transfer_next(module, inputs, output, next_index=i+1):
                        if next_index not in pending:
                            with timeline.phase(f"E3/r{repeat}/{mode}/request.{next_index}/h2d", bytes=sum(t.numel()*t.element_size() for t in hosts[next_index] if t is not None)):
                                pending[next_index] = _install_inputs(worker,specs[next_index],hosts[next_index],stream=copy_stream)
                    handle = worker.pipe.transformer.blocks[0].register_forward_hook(transfer_next)
                try:
                    for step in range(args.steps):
                        with timeline.phase(prefix + f"/dit.step.{step}"):
                            _step_async(worker,state)
                        if handle is not None:
                            handle.remove()
                            handle=None
                    torch.cuda.current_stream(args.device).synchronize()
                    result = _finish_request(worker,state,timeline,prefix)
                    if i==0 and (result["sha256"]!=official_reference["sha256"] or result["latent"]["sha256"]!=official_reference["latent"]["sha256"]):
                        raise RuntimeError("E3 input transport arithmetic differs from ordinary worker")
                    records.append(dict(index=i,complete_ns=time.perf_counter_ns(),**result))
                finally:
                    if handle is not None:
                        handle.remove()
            end = time.perf_counter_ns()
            modes[mode] = dict(start_ns=start,end_ns=end,wall_ms=(end-start)/1e6,requests=records,
                               input_setup_ms=setup_ms,including_input_setup_ms=setup_ms+(end-start)/1e6,
                               preencoded=True,transfer_dtype="float32",gpu_cast_dtype="float16")
            atomic_json(args.out/f"E3-r{repeat}-{mode}.json",modes[mode])
        rows.append(dict(repeat=repeat,modes=modes,correctness=_check_pair(modes["sync"],modes["async"]),
                         speedup=modes["sync"]["wall_ms"]/modes["async"]["wall_ms"]))
    return dict(pairs=rows,phases=timeline.records)


def _persist_output(cpu, ready, retained_source, path, timeline, prefix, output_mode):
    if ready is not None:
        with timeline.phase(prefix + "/wait_d2h"):
            ready.synchronize()
    with timeline.phase(prefix + "/output_check"):
        array = cpu.numpy()
        result = _hash_array(array)
    if output_mode=="npy":
        import numpy as np
        with timeline.phase(prefix + "/write_npy"):
            np.save(path,array,allow_pickle=False)
    result["complete_ns"] = time.perf_counter_ns()
    return result


def experiment_e4(args):
    import torch
    worker,timeline = _worker(args),Timeline(args.device)
    official_reference=_warm(worker,args,timeline)
    specs = [spec_for(args,i,"E4") for i in range(args.requests)]
    payloads = [_encode(worker,spec,timeline,f"E4/input.{i}") for i,spec in enumerate(specs)]
    hosts = [_host_inputs(worker,spec,payload,pinned=False) for spec,payload in zip(specs,payloads)]
    rows=[]
    for repeat in range(args.repeats):
        modes={}
        order=("sync","cpu_async","d2h_async") if repeat%2==0 else ("d2h_async","cpu_async","sync")
        for mode in order:
            copy_stream=torch.cuda.Stream(device=args.device)
            futures=[]
            latent_results=[]
            with ThreadPoolExecutor(max_workers=1,thread_name_prefix="workflow-output") as executor:
                start=time.perf_counter_ns()
                for i,spec in enumerate(specs):
                    # Bounded queue: at most two outputs may retain pinned/GPU buffers.
                    if len(futures)>=2:
                        futures[-2].result()
                    prefix=f"E4/r{repeat}/{mode}/request.{i}"
                    with timeline.phase(prefix+"/prepare"):
                        state,ready_inputs=_install_inputs(worker,spec,hosts[i])
                        torch.cuda.current_stream(args.device).wait_event(ready_inputs)
                    for step in range(args.steps):
                        with timeline.phase(prefix+f"/dit.step.{step}"):
                            _step_async(worker,state)
                    with timeline.phase(prefix+"/vae"):
                        video=_decode_gpu(worker,state)
                    latent_results.append(_hash_array(state.latents.detach().float().cpu().numpy()))
                    ready=None
                    with timeline.phase(prefix+"/d2h",bytes=video.numel()*video.element_size()):
                        if mode=="d2h_async":
                            produced=torch.cuda.Event()
                            produced.record()
                            cpu=torch.empty(video.shape,dtype=video.dtype,device="cpu",pin_memory=True)
                            with torch.cuda.stream(copy_stream):
                                copy_stream.wait_event(produced)
                                cpu.copy_(video,non_blocking=True)
                                video.record_stream(copy_stream)
                                ready=torch.cuda.Event()
                                ready.record()
                        else:
                            cpu=video.cpu()
                    path=args.out/f"E4-r{repeat}-{mode}-request{i}.npy"
                    future=executor.submit(_persist_output,cpu,ready,video,path,timeline,prefix,args.output_mode)
                    futures.append(future)
                    if mode=="sync":
                        future.result()
                    # Avoid runtime.release's device-global barrier; state tensor
                    # destruction is ordered on its producing stream.
                    worker._states.pop(state.request_id)
                    del state,video,cpu
                records=[]
                for i,future in enumerate(futures):
                    result=future.result()
                    if i==0 and (result["sha256"]!=official_reference["sha256"] or latent_results[i]["sha256"]!=official_reference["latent"]["sha256"]):
                        raise RuntimeError("E4 split decode/output differs from ordinary worker")
                    records.append(dict(index=i,latent=latent_results[i],**result))
                torch.cuda.synchronize(args.device)
                end=time.perf_counter_ns()
            modes[mode]=dict(start_ns=start,end_ns=end,wall_ms=(end-start)/1e6,requests=records,
                             output_mode=args.output_mode,queue_depth=2,preencoded=True,
                             write_completion="buffered np.save returned; no fsync" if args.output_mode=="npy" else "finite/hash validation completed")
            atomic_json(args.out/f"E4-r{repeat}-{mode}.json",modes[mode])
        rows.append(dict(repeat=repeat,order=order,modes=modes,
            correctness={mode:_check_pair(modes["sync"],modes[mode]) for mode in ("cpu_async","d2h_async")}))
    return dict(pairs=rows,phases=timeline.records)


def experiment_e5(args):
    """Direct eager dual-stream fallback; fixed SDPA policy, no global sync in hotpaths."""
    import torch
    from ditserve.models.sana.runtime import SanaAdapter,RECIPE
    timeline=Timeline(args.device)
    worker=_worker(args)
    # One process owns both models; production's per-process 0.54/0.40 quotas
    # do not apply. The experiment's explicit combined cap is 95%.
    torch.cuda.set_per_process_memory_fraction(0.95,args.device)
    sana=SanaAdapter(str(Path(args.model_root)/"sana-sprint-0.6b"),args.device)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    spec=spec_for(args,0,"E5")
    payload=_encode(worker,spec,timeline,"E5/input")
    host=_host_inputs(worker,spec,payload,pinned=False)
    torch.cuda.synchronize(args.device)
    residency=dict(allocated_bytes=torch.cuda.memory_allocated(args.device),
                   reserved_bytes=torch.cuda.memory_reserved(args.device),
                   total_bytes=torch.cuda.get_device_properties(args.device).total_memory)
    rows=[]
    for stage in ("dit","vae"):
        modes={}
        # Standalone and paired arms are repeated in alternating order. Warm
        # outputs use the same arithmetic and are recorded separately below.
        for repeat in range(-1,args.repeats):
            order=("wan_only","sana_only","interleaved","streams") if repeat%2==0 else ("streams","interleaved","sana_only","wan_only")
            current={}
            for mode in order:
                prefix=f"E5/{stage}/r{repeat}/{mode}"
                state,ready=_install_inputs(worker,spec,host)
                main=torch.cuda.current_stream(args.device)
                main.wait_event(ready)
                if stage=="vae":
                    for _ in range(args.steps):
                        _step_async(worker,state,fixed_attention=True)
                torch.cuda.synchronize(args.device)
                wan_stream=torch.cuda.Stream(device=args.device)
                sana_stream=torch.cuda.Stream(device=args.device) if mode=="streams" else wan_stream
                # Models and prepared state were last used on the setup stream.
                ready=torch.cuda.Event()
                ready.record(main)
                wan_stream.wait_event(ready)
                sana_stream.wait_event(ready)
                for tensor in (state.latents,state.prompt_embeds,state.negative_prompt_embeds):
                    if tensor is not None:
                        tensor.record_stream(wan_stream)
                images=[]
                image_events=[]
                wan_end=None
                video=None
                decode=raw=None
                torch.cuda.reset_peak_memory_stats(args.device)
                start=time.perf_counter_ns()
                def generate(index):
                    with torch.cuda.stream(sana_stream),torch.inference_mode(),timeline.phase(prefix+f"/sana.{index}"):
                        generator=torch.Generator(device="cpu").manual_seed(9000+index)
                        output=sana.pipe(prompt="A watercolor landscape with mountains",generator=generator,
                                         output_type="pt",**RECIPE).images
                        images.append(output)
                        end=torch.cuda.Event(enable_timing=True)
                        end.record(sana_stream)
                        image_events.append(end)
                with torch.cuda.stream(wan_stream):
                    anchor=torch.cuda.Event(enable_timing=True)
                    anchor.record(wan_stream)
                if mode=="sana_only":
                    for i in range(10):
                        generate(i)
                elif stage=="dit":
                    emitted=0
                    for step in range(args.steps):
                        with torch.cuda.stream(wan_stream),timeline.phase(prefix+f"/dit.step.{step}"):
                            _step_async(worker,state,fixed_attention=True)
                            if step+1==args.steps:
                                wan_end=torch.cuda.Event(enable_timing=True)
                                wan_end.record(wan_stream)
                        if mode!="wan_only":
                            wanted=(step+1)*10//args.steps
                            while emitted<wanted:
                                generate(emitted)
                                emitted+=1
                else:
                    if mode=="interleaved":
                        # The current serving scheduler interleaves at causal
                        # VAE latent-frame boundaries, rather than whole decode.
                        from ditserve.models.pipeline import ChunkedDecode
                        with torch.cuda.stream(wan_stream),torch.inference_mode():
                            decode=ChunkedDecode(torch,worker.pipe,state.latents)
                            emitted=0
                            while not decode.done:
                                with timeline.phase(prefix+f"/vae.chunk.{decode.index}"):
                                    decode.advance(1)
                                wanted=decode.index*10//decode.frames
                                while emitted<wanted:
                                    generate(emitted)
                                    emitted+=1
                            raw=torch.clamp(torch.cat(decode.chunks,2),min=-1.0,max=1.0)
                            video=worker.pipe.video_processor.postprocess_video(raw,output_type="pt").permute(0,1,3,4,2).contiguous()
                            wan_end=torch.cuda.Event(enable_timing=True)
                            wan_end.record(wan_stream)
                    else:
                        with torch.cuda.stream(wan_stream),timeline.phase(prefix+"/vae"):
                            video=_decode_gpu(worker,state,fixed_attention=True)
                            wan_end=torch.cuda.Event(enable_timing=True)
                            wan_end.record(wan_stream)
                        if mode!="wan_only":
                            for i in range(10):
                                generate(i)
                torch.cuda.synchronize(args.device)
                end=time.perf_counter_ns()
                result=dict(wall_ms=(end-start)/1e6,start_ns=start,end_ns=end,
                    image_complete_ms=[anchor.elapsed_time(event) for event in image_events],
                    wan_stream_complete_ms=None if wan_end is None else anchor.elapsed_time(wan_end),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                    images=[_hash_array(image.float().cpu().numpy()) for image in images])
                if mode!="sana_only":
                    result["latent"]=_hash_array(state.latents.detach().float().cpu().numpy())
                    if video is not None:
                        result["video"]=_hash_array(video.float().cpu().numpy())
                worker._states.pop(state.request_id)
                # ChunkedDecode owns temporal caches, latent input and output
                # chunks. Release them after validation, outside the timer, so
                # subsequent arms do not inherit interleaved VAE allocations.
                del state,images,video,decode,raw
                current[mode]=result
                atomic_json(args.out/f"E5-{stage}-r{repeat}-{mode}.json",result)
            if repeat>=0:
                correctness={}
                for mode in ("interleaved","streams"):
                    value=current[mode]
                    correct=(value["latent"]["sha256"]==current["wan_only"]["latent"]["sha256"]
                        and [image["sha256"] for image in value["images"]]==[image["sha256"] for image in current["sana_only"]["images"]])
                    if stage=="vae":
                        correct=correct and value["video"]["sha256"]==current["wan_only"]["video"]["sha256"]
                    correctness[mode]=correct
                if not all(correctness.values()):
                    raise RuntimeError("E5 changed model outputs: "+json.dumps(correctness))
                rows.append(dict(stage=stage,repeat=repeat,order=order,modes=current,correctness=correctness,
                    streams_vs_interleaved=current["interleaved"]["wall_ms"]/current["streams"]["wall_ms"]))
    return dict(pairs=rows,phases=timeline.records,residency=residency,
                executor="single process dual CUDA stream fallback; private MPS is a separate launch option",
                attention_policy="fixed flash+math enabled, efficient+cudnn disabled; no threaded context mutation",
                memory_fraction=0.95,overlap_requires_cuda_trace=True)


def _e6_child(config,rank,init_method,result_queue):
    args=argparse.Namespace(**config)
    device="cuda:"+args.devices.split(",")[rank]
    timeline=Timeline(device)
    try:
        import torch
        import datetime
        from ditserve.models.wan.parallel import LayoutHandle,build_fixed_groups
        from ditserve.models.wan.sp import PinnedWanUlyssesBackend
        from .workflow_attention import attention_variant
        os.environ.setdefault("NCCL_P2P_DISABLE","1")
        if os.environ["NCCL_P2P_DISABLE"]!="1":
            raise RuntimeError("E6 requires NCCL_P2P_DISABLE=1 to preserve host-staged topology")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group("nccl",init_method=init_method,rank=rank,world_size=2,
            timeout=datetime.timedelta(seconds=120))
        worker=_worker(args,device)
        spec=spec_for(args,0,"E6")
        payload=_encode(worker,spec,timeline,"E6/input")
        qualification=dict(spec,steps=4,request_id="E6-qualification")
        state=worker.prepare_encoded(qualification,dataclasses.replace(payload,spec=qualification,consumed=False))
        for step in range(4):
            with timeline.phase(f"E6/SP1-qualification/dit.step.{step}"):
                worker.step(state)
        reference_latent=state.latents.detach().float().cpu().numpy().copy()
        reference_video=worker.finish(state)
        worker.release(state)
        groups=build_fixed_groups(timeout_seconds=60,backend="nccl")
        backend=PinnedWanUlyssesBackend(LayoutHandle(rank,(0,1)),groups).attach(worker)
        variants=("cache_dit","baseline","qkv_pipeline","heads_pipeline")
        reports=[]
        def run(spec,prefix):
            state=worker.prepare_encoded(spec,dataclasses.replace(payload,spec=spec,consumed=False))
            torch.distributed.barrier()
            start=time.perf_counter_ns()
            for step in range(spec["steps"]):
                with timeline.phase(prefix+f"/dit.step.{step}"):
                    backend.step(state)
            end=time.perf_counter_ns()
            latent=state.latents.detach().float().cpu().numpy().copy()
            with timeline.phase(prefix+"/vae"):
                video=worker.finish(state)
            complete=time.perf_counter_ns()
            worker.release(state)
            return dict(start_ns=start,end_ns=end,dit_ms=(end-start)/1e6,
                        finish_ms=(complete-end)/1e6,wall_ms=(complete-start)/1e6),latent,video
        for mode in variants:
            context=contextlib.nullcontext() if mode=="cache_dit" else attention_variant(backend,mode)
            with context:
                result,latent,video=run(qualification,f"E6/{mode}/qualification")
                import numpy as np
                errors=dict(latent_max_abs=float(np.max(np.abs(latent-reference_latent))),
                    video_max_abs=float(np.max(np.abs(video-reference_video))),
                    finite=bool(np.isfinite(latent).all() and np.isfinite(video).all()),
                    latent_shape=list(latent.shape),video_shape=list(video.shape))
                passed=errors["finite"] and errors["latent_max_abs"]<=0.05 and errors["video_max_abs"]<=0.05
                decision=torch.tensor(int(passed),device=device,dtype=torch.int32)
                torch.distributed.all_reduce(decision,op=torch.distributed.ReduceOp.MIN)
                report=dict(mode=mode,qualification=dict(steps=4,passed=bool(decision.item()),**errors),runs=[])
                if report["qualification"]["passed"]:
                    # Qualification warms kernels at the target shape. A full
                    # workload warmup precedes the timed production-step repeats.
                    for repeat in range(-1,args.repeats):
                        result,latent,video=run(spec,f"E6/{mode}/r{repeat}")
                        result.update(repeat=repeat,warmup=repeat<0,latent=_hash_array(latent),video=_hash_array(video))
                        report["runs"].append(result)
                reports.append(report)
                atomic_json(args.out/f"E6-rank{rank}-{mode}.json",report)
        backend.close()
        torch.distributed.destroy_process_group()
        result_queue.put(dict(ok=True,result=dict(rank=rank,device=device,variants=reports,phases=timeline.records)))
    except BaseException:
        result_queue.put(dict(ok=False,rank=rank,error=traceback.format_exc()))


def experiment_e6(args):
    devices=args.devices.split(",")
    if len(devices)!=2 or len(set(devices))!=2:
        raise ValueError("E6 requires two distinct devices")
    context=multiprocessing.get_context("spawn")
    results=context.Queue()
    rendezvous=args.out/"nccl-rendezvous"
    if rendezvous.exists():
        raise FileExistsError(rendezvous)
    processes=[context.Process(target=_e6_child,args=(vars(args),rank,"file://"+str(rendezvous.resolve()),results)) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        outputs=[results.get(timeout=args.timeout) for _ in processes]
        if not all(row["ok"] for row in outputs):
            raise RuntimeError(json.dumps(outputs))
        for process in processes:
            process.join(timeout=60)
            if process.exitcode!=0:
                raise RuntimeError("E6 worker did not exit cleanly")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(30)
    ranks=sorted([row["result"] for row in outputs],key=lambda row:row["rank"])
    return dict(ranks=ranks,qualification_threshold=0.05,
        timing_scope="SP2 eager DiT including consensus; VAE replicated; CPU encoding excluded",
        qualification_shape=[args.height,args.width,args.frames],overlap_requires_cuda_trace=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment",choices=[f"E{i}" for i in range(1,7)],required=True)
    parser.add_argument("--frames",type=int,choices=[5,17,33],default=17)
    parser.add_argument("--height",type=int,default=480)
    parser.add_argument("--width",type=int,default=832)
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--requests",type=int,default=4)
    parser.add_argument("--device",default="cuda:1")
    parser.add_argument("--devices",default="0,1")
    parser.add_argument("--cpu-threads",type=int,default=8)
    parser.add_argument("--cpu-affinity",choices=["auto","none"],default="auto")
    parser.add_argument("--model-root",type=Path,default=Path("/models"))
    parser.add_argument("--output-mode",choices=["check","npy"],default="check",
                        help="Validate outputs without retaining video arrays by default; npy opts into disk-write measurements.")
    parser.add_argument("--timeout",type=int,default=7200)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    if min(args.repeats,args.requests,args.steps,args.cpu_threads)<1:
        parser.error("repeats, requests, steps and cpu-threads must be positive")
    if args.height%16 or args.width%16:
        parser.error("height/width must be multiples of 16")
    if args.experiment=="E2" and args.requests<2:
        parser.error("E2 requires at least two requests")
    return args


def main(argv=None):
    args=parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/"run.json"
    if manifest.exists():
        raise FileExistsError("refusing to overwrite an experiment: "+str(manifest))
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    result=dict(schema="ditserve.workflow-overlap.v1",experiment=args.experiment,config=config,
                status="running",started_ns=time.perf_counter_ns(),
                evidence_contract="Host timers do not prove device overlap; inspect matching NVTX/CUDA trace.")
    atomic_json(manifest,result)
    try:
        result.update(globals()["experiment_"+args.experiment.lower()](args))
        result["status"]="complete"
    except BaseException:
        result.update(status="failed",error=traceback.format_exc())
        raise
    finally:
        result["finished_ns"]=time.perf_counter_ns()
        atomic_json(manifest,result)
    print(json.dumps(dict(experiment=args.experiment,status=result["status"],manifest=str(manifest))))


if __name__=="__main__":
    main()
