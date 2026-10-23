"""CPU doubles exercise E5 arm boundaries without model loads or CUDA."""

import contextlib
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

from ditserve.evaluation import workflow_overlap as workflow
from ditserve.models import pipeline
from ditserve.models.sana import runtime as sana_runtime


class Tensor:
    def record_stream(self, *_):
        pass

    def float(self):
        return self

    cpu = float
    contiguous = float
    detach = float

    def numpy(self):
        return 0

    def permute(self, *_):
        return self


class Stream:
    def wait_event(self, *_):
        pass


class Event:
    def __init__(self, **_):
        pass

    def record(self, *_):
        pass

    def elapsed_time(self, *_):
        return 0.


class Generator:
    def __init__(self, **_):
        pass

    def manual_seed(self, _):
        return self


class Timeline:
    def __init__(self, *_):
        self.records = []

    def phase(self, *_):
        return contextlib.nullcontext()


class E5LifetimeTests(unittest.TestCase):
    def test_chunked_decode_buffers_are_released_before_the_next_arm(self):
        decoder_refs, raw_refs, starts = [], [], []

        class Decode:
            def __init__(self, *_):
                decoder_refs.append(weakref.ref(self))
                self.x, self.cache, self.chunks = Tensor(), [Tensor()], []
                self.index, self.frames = 0, 2

            @property
            def done(self):
                return self.index == self.frames

            def advance(self, *_):
                self.chunks.append(Tensor())
                self.index += 1

        def cat(*_):
            raw = Tensor()
            raw_refs.append(weakref.ref(raw))
            return raw

        def start_arm(*_):
            starts.append(dict(decoders=sum(ref() is not None for ref in decoder_refs),
                               raw_outputs=sum(ref() is not None for ref in raw_refs)))

        cuda = types.SimpleNamespace(
            synchronize=lambda *_: None, set_per_process_memory_fraction=lambda *_: None,
            memory_allocated=lambda *_: 0, memory_reserved=lambda *_: 0,
            max_memory_allocated=lambda *_: 0, max_memory_reserved=lambda *_: 0,
            get_device_properties=lambda *_: types.SimpleNamespace(total_memory=24 * 1024 ** 3),
            Stream=lambda **_: Stream(), current_stream=lambda *_: Stream(),
            stream=lambda *_: contextlib.nullcontext(), Event=Event,
            reset_peak_memory_stats=start_arm)
        torch = types.SimpleNamespace(cuda=cuda, Generator=Generator,
            inference_mode=contextlib.nullcontext, cat=cat, clamp=lambda value, **_: value,
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(**{
                name: lambda *_: None for name in (
                    "enable_flash_sdp", "enable_math_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp")})))
        processor = types.SimpleNamespace(postprocess_video=lambda *_args, **_kwargs: Tensor())
        worker = types.SimpleNamespace(pipe=types.SimpleNamespace(video_processor=processor), _states={})

        def install(_worker, spec, _host):
            state = types.SimpleNamespace(latents=Tensor(), prompt_embeds=Tensor(),
                negative_prompt_embeds=Tensor(), request_id=spec["request_id"])
            worker._states[state.request_id] = state
            return state, Event()

        sana = types.SimpleNamespace(pipe=lambda **_: types.SimpleNamespace(images=Tensor()))
        replacements = dict(_worker=lambda *_: worker, _encode=lambda *_: object(),
            _host_inputs=lambda *_args, **_kwargs: (), _install_inputs=install,
            _step_async=lambda *_args, **_kwargs: None,
            _decode_gpu=lambda *_args, **_kwargs: Tensor(),
            _hash_array=lambda *_: dict(sha256="same"), atomic_json=lambda *_: None,
            Timeline=Timeline)
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", torch=torch))
            stack.enter_context(patch.object(sana_runtime, "SanaAdapter", lambda *_: sana))
            stack.enter_context(patch.object(pipeline, "ChunkedDecode", Decode))
            for name, value in replacements.items():
                stack.enter_context(patch.object(workflow, name, value))
            args = types.SimpleNamespace(device="cuda:1", model_root=Path(directory),
                out=Path(directory), repeats=1, steps=2, height=480, width=832, frames=17)
            result = workflow.experiment_e5(args)
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(len(decoder_refs), 2)  # One warmup and one measured interleaved arm.
        self.assertEqual(len(starts), 16)
        self.assertTrue(all(row == dict(decoders=0, raw_outputs=0) for row in starts), starts)
        self.assertFalse(any(ref() is not None for ref in decoder_refs + raw_refs))


if __name__ == "__main__":
    unittest.main()
