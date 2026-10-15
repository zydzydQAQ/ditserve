"""External regional Wan compilation; original model modules remain untouched.

Only block tensor arithmetic is compiled. Request ownership, synchronization,
budget checks, CFG combination and UniPC commits remain in the eager executor.
"""

import contextlib
import time

from ..directions.block_executor import WanBlockExecutor


class _NativeEagerMetadata:
    """Reporting identity only; deliberately has no tensor execution method."""

    name, compiled, group_size = "eager", False, 1

    def report(self):
        return dict(name=self.name, compiled=False, external_callable=False,
                    model_forward_replaced=False, original_advance=True,
                    executor="ditserve.directions.block_executor.WanBlockExecutor.advance")


class NativeWanExecutor(WanBlockExecutor):
    """Original strong eager baseline, with metadata and no execution override."""

    supports_host_observation = False

    def __init__(self, worker):
        super().__init__(worker)
        self.backend = _NativeEagerMetadata()


class BlockBackend:
    """Request-independent callable registry, with explicit fallback detection."""

    def __init__(self, worker, *, compiled=False, group_size=1, compiler=None):
        if type(group_size) is not int or group_size not in (1, 4, 8):
            raise ValueError("regional block group must be 1, 4 or 8")
        self.worker, self.model = worker, worker.pipe.transformer
        self.compiled, self.group_size = bool(compiled), group_size
        self.name = "inductor-block%d" % group_size if compiled else "eager"
        self._compiler = compiler
        self._callables = {}
        self.compilations = 0
        self.graph_invocations = 0
        self.compile_backend_ms = 0.
        self.compile_attempts = []
        self.calls = 0
        self.first_calls = []
        self.steady = False

    def freeze_compilation(self):
        self.steady = True

    def steady_context(self):
        return (self.worker.torch.compiler.set_stance("fail_on_recompile")
                if self.compiled and self.steady and self._compiler is None else contextlib.nullcontext())

    def width(self, index, remaining, stop=30, alignment_offset=0):
        # A tail uses individual blocks: never extend a quantum or CFG branch
        # just to fill a compiled region. In particular K4 at cond28 executes
        # cond28,29 then uncond0,1, exactly as the original block executor.
        available = min(remaining, stop - index)
        aligned = (index + alignment_offset) % self.group_size == 0
        return self.group_size if aligned and available >= self.group_size else 1

    def _inductor(self, graph, inputs):
        from torch._dynamo.backends.registry import lookup_backend
        started = time.perf_counter()
        attempt = dict(status="running")
        self.compile_attempts.append(attempt)
        try:
            executable = lookup_backend("inductor")(graph, inputs,
                config_patches={"triton.cudagraphs": False})
            self.compilations += 1
            attempt["status"] = "completed"
        except BaseException as error:
            attempt.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            attempt["backend_ms"] = (time.perf_counter() - started) * 1000
            self.compile_backend_ms += attempt["backend_ms"]

        # Do not copy Inductor's callable metadata: functools.wraps would copy
        # _torchdynamo_orig_callable, so Dynamo's disable()/innermost_fn() would
        # unwrap past this counter and falsely report an eager fallback.
        def counted(*args, **kwargs):
            self.graph_invocations += 1
            return executable(*args, **kwargs)
        return counted

    def _build(self, index, count):
        blocks = tuple(self.model.blocks[index:index + count])

        def tensor_region(hidden, encoder, projected, rotary):
            for block in blocks:
                hidden = block(hidden, encoder, projected, rotary)
            return hidden

        if not self.compiled:
            return tensor_region
        if self._compiler is not None:  # Explicit test/instrumentation backend.
            return self._compiler(tensor_region)
        return self.worker.torch.compile(tensor_region, backend=self._inductor,
                                         fullgraph=True, dynamic=False)

    def run(self, index, count, hidden, encoder, projected, rotary):
        if (type(index) is not int or type(count) is not int or count < 1 or
                index < 0 or index + count > len(self.model.blocks)):
            raise ValueError("invalid regional block interval")
        key = (index, count)
        first = key not in self._callables
        if first:
            self._callables[key] = self._build(index, count)
        before = self.graph_invocations
        started = time.perf_counter()
        row = dict(start_block=index, blocks=count, status="running",
            scope="host invocation including lazy compilation; not synchronized GPU latency")
        try:
            with self.steady_context():
                value = self._callables[key](hidden, encoder, projected, rotary)
            self.calls += 1
            if self.compiled and self._compiler is None and self.graph_invocations == before:
                raise RuntimeError("compiled Wan region silently executed without a compiled graph")
            row["status"] = "completed"
            return value
        except BaseException as error:
            row.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            if first:
                row["host_call_ms"] = (time.perf_counter() - started) * 1000
                self.first_calls.append(row)

    def report(self):
        return dict(name=self.name, compiled=self.compiled, group_size=self.group_size,
                    requested_regions=len(self._callables), compilations=self.compilations,
                    graph_invocations=self.graph_invocations, calls=self.calls,
                    compile_backend_ms=self.compile_backend_ms, compile_attempts=list(self.compile_attempts),
                    first_calls=list(self.first_calls),
                    cuda_graphs=False, fullgraph=True, dynamic=False,
                    external_callable=True, model_forward_replaced=False,
                    test_compiler=self._compiler is not None)


class WanCompiledExecutor(WanBlockExecutor):
    """Same cache-off continuation and cross-CFG K semantics, external backend."""

    supports_host_observation = True

    def __init__(self, worker, backend=None):
        super().__init__(worker)  # Preserve all original pinned-model guards.
        self.backend = backend or BlockBackend(worker)
        if self.backend.worker is not worker or self.backend.model is not self.model:
            raise ValueError("regional backend belongs to another model/worker")
        from ..research.cfg_empty_context import EmptyCacheContextGuard
        self._model_guard = EmptyCacheContextGuard(self.model)

    def advance(self, state, blocks=1, *, observe=False, observe_host=False, check_budget=lambda: None):
        if type(blocks) is not int or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        worker, torch = self.worker, self.worker.torch
        with worker._lock, torch.inference_mode(), worker._dit_attention_context():
            whole_started = time.perf_counter() if observe_host else None
            host_started = whole_started
            host_ms = 0.
            self._check(state)
            if self._advancing:
                raise RuntimeError("overlapping/reentrant GPU block execution")
            if state.base.step_index >= state.base.spec["steps"]:
                raise ValueError("request has no remaining solver steps")
            self._model_guard.validate()
            if observe_host:
                host_ms += (time.perf_counter()-host_started)*1000
            check_budget()
            step, branch, index = state.committed_step, state.branch, state.next_block
            events, completed, executed = [], False, 0
            worker._sync()
            started = time.perf_counter()
            self._advancing = True
            try:
                while executed < blocks:
                    check_budget()
                    if observe_host:
                        host_started = time.perf_counter()
                    needs_packet = state.packet is None
                    if observe_host:
                        host_ms += (time.perf_counter()-host_started)*1000
                    if needs_packet:
                        self._begin(state)
                    if observe_host:
                        host_started = time.perf_counter()
                    packet, current = state.packet, state.next_block
                    # Align regions by logical cond[0:30]+uncond[30:60], so a
                    # continuous step and repeated K4 use identical regions.
                    width = self.backend.width(current, blocks-executed,
                        alignment_offset=0 if state.branch=="cond" else len(self.model.blocks))
                    if observe_host:
                        host_ms += (time.perf_counter()-host_started)*1000
                    if observe:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                    packet["hidden"] = self.backend.run(current, width, packet["hidden"],
                        packet["encoder"], packet["projected"], packet["rotary"])
                    if observe:
                        end.record()
                        events.append((state.branch, current, width, start, end))
                    if observe_host:
                        host_started = time.perf_counter()
                    state.next_block += width
                    executed += width
                    branch_finished = state.next_block == len(self.model.blocks)
                    if observe_host:
                        host_ms += (time.perf_counter()-host_started)*1000
                    if branch_finished:
                        completed = self._finish_branch(state)
                        if completed:
                            break
                worker._sync()
                row = dict(kind="block_quantum", step_index=step, branch=branch,
                    start_block=index, next_block=state.next_block, next_branch=state.branch,
                    blocks=executed, step_completed=completed,
                    ms=(time.perf_counter() - started) * 1000,
                    block_cuda_ms=[dict(branch=b, block=i, blocks=n, ms=float(a.elapsed_time(z)))
                                   for b, i, n, a, z in events],
                    pure_host_control_ms=host_ms if observe_host else None,
                    quantum_total_wall_ms=(time.perf_counter()-whole_started)*1000 if observe_host else None,
                    host_scope="lower bound: request/schema/model guards, packet-needed test, region selection and block counters; excludes tensor calls, CUDA APIs, synchronization, budget callback and evidence IO; may overlap GPU execution",
                    extra_state_bytes=self.extra_state_bytes(state), backend=self.backend.name,
                    boundary_semantics="at_most_K_actual_blocks_cross_CFG_then_stop_at_solver_commit")
                state.events.append(row)
                return row
            except BaseException:
                state.failed = True
                state.packet = state.conditional = None
                raise
            finally:
                self._advancing = False

