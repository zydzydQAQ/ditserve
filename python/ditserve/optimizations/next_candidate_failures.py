"""Narrow, evidence-backed screening failures for the selected Sana candidate.

This module neither imports torch nor changes the supervisor's success policy.
Only an unsupported compiler/capture or an explicit compile-mechanics rejection
inside the candidate itself can become a completed negative screening result.
"""

from pathlib import Path

from ..research.common import file_digest
from .budget import BudgetStop


def _exceptions(error):
    pending, values, seen = [error], [], set()
    while pending and len(values) < 32:
        item = pending.pop()
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        values.append(item)
        seen.add(id(item))
        pending.extend((item.__cause__, item.__context__, getattr(item, "inner_exception", None)))
    return values if not pending else []


def _site(errors, candidate):
    expected = "_CompiledRegion" if candidate == "vae_compile" else "_TransformerGraph"
    for error in errors:
        trace = error.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            owner = frame.f_locals.get("self")
            if (frame.f_globals.get("__name__") == "ditserve.optimizations.next_sana"
                    and type(owner).__name__ == expected
                    and frame.f_code.co_name in (("__call__", "_backend") if candidate == "vae_compile"
                                                else ("_capture",))):
                source = Path(frame.f_code.co_filename).resolve(strict=True)
                return dict(module=frame.f_globals["__name__"], owner=expected,
                            operation=frame.f_code.co_name, source_path=str(source),
                            source_sha256=file_digest(source))
            trace = trace.tb_next
    return None


def classify(error, candidate, active_unit, mechanics):
    """Return a rejection category, or None for a failure requiring attention."""
    if (candidate not in ("vae_compile", "transformer_graph")
            or not isinstance(active_unit, dict)
            or active_unit.get("kind") not in ("warmup", "measurement")
            or active_unit.get("arm") != candidate
            or mechanics.get("candidate") != candidate):
        return None
    errors = _exceptions(error)
    if not errors:
        return None
    fatal = ("out of memory", "illegal memory access", "device-side assert",
             "unspecified launch failure", "misaligned address", "driver shutting down",
             "invalid device function", "launch timeout")
    for item in errors:
        if (isinstance(item, (BudgetStop, KeyboardInterrupt, SystemExit, ValueError,
                              MemoryError, OSError)) or type(item).__name__ == "OutOfMemoryError"
                or any(word in str(item).lower() for word in fatal)):
            return None
    site = _site(errors, candidate)
    if site is None:
        return None
    category = None
    if candidate == "vae_compile":
        # A BackendCompilerFailed wrapper is accepted only when its underlying
        # exception is itself one of these explicit unsupported/recompile types.
        # Arbitrary backend bugs, resource errors and kernel failures propagate.
        accepted = {("torch._dynamo.exc", "Unsupported"): "compiler_unsupported",
                    ("torch._dynamo.exc", "RecompileError"): "measurement_recompile",
                    ("torch._dynamo.exc", "FailOnRecompileLimitHit"): "measurement_recompile"}
        for item in errors:
            for kind in type(item).__mro__:
                match = accepted.get((kind.__module__, kind.__name__))
                if match and (match != "measurement_recompile" or active_unit["kind"] == "measurement"):
                    category = match
            message = str(item)
            if type(item) is RuntimeError:
                if message in ("Sana decoder executed without a compiled graph",
                               "Sana post_quant_conv executed without a compiled graph"):
                    category = "compiled_graph_not_executed"
                elif active_unit["kind"] == "measurement" and (
                        message in ("Sana decoder compiled during steady measurement",
                                    "Sana post_quant_conv compiled during steady measurement")
                        or message.startswith("Detected recompile when torch.compile stance is 'fail_on_recompile'. ")):
                    category = "measurement_recompile"
    elif type(error) is RuntimeError and (str(error).splitlines() or [""])[0] in (
            "CUDA error: operation not permitted when stream is capturing",
            "CUDA error: operation not permitted when stream is capturing."):
        category = "cuda_capture_unsupported"
    if category is None:
        return None
    return dict(classification=category, operation=site,
                error_type=type(error).__name__, module=type(error).__module__, message=str(error),
                exception_chain=[dict(type=type(item).__module__ + "." + type(item).__name__,
                                      type_mro=[kind.__module__ + "." + kind.__name__ for kind in type(item).__mro__],
                                      message=str(item)) for item in errors],
                scope="candidate screening rejection; no eager fallback result or serving qualification")


def restore_and_verify(adapter):
    """Restore candidate hooks/release graph state; never hide cleanup failure.

    The phase supervisor must still verify process exit, GPU release, immutable
    sources and complete accounting before any next phase may start.
    """
    from .next_sana import _CompiledRegion, _TransformerGraph
    originals = list(adapter._next_originals)
    adapter.set_candidate("eager")
    graph = adapter._transformer_graph
    if graph is not None:
        graph.release()
    if (adapter._cleanup_failed or adapter._busy or adapter._next_lock.locked()
            or adapter._steady_depth or adapter._next_originals
            or adapter.candidate != "eager" or adapter.backend != "eager"
            or adapter.pipe.transformer is not adapter._eager_transformer
            or adapter._eager_transformer.forward != adapter._transformer_forward):
        raise RuntimeError("candidate cleanup ownership or transformer restoration unconfirmed")
    if any(getattr(module, name) != original for module, name, original in originals):
        raise RuntimeError("candidate cleanup did not restore original module hooks")
    for name in ("decoder", "post_quant_conv"):
        module = getattr(adapter.pipe.vae, name, None)
        if isinstance(getattr(module, "forward", None), (_CompiledRegion, _TransformerGraph)):
            raise RuntimeError("candidate cleanup left a computation wrapper installed")
    if graph is not None and any(getattr(graph, key) is not None for key in
                                 ("graph", "static_inputs", "static_output", "signature", "static_signature")):
        raise RuntimeError("candidate cleanup left captured graph state allocated")
    return dict(passed=True, hooks_restored=True, graph_state_released=True,
                scope="adapter ownership only; full process/GPU cleanup remains mandatory in supervisor")
