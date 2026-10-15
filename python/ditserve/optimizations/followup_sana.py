"""Opt-in Sana VAE compilation plus eager transformer CUDA graph.

This adapter only provides execution mechanics. Quality, resident-memory and
paired online checks belong to the follow-up runner; selection grants no
qualification. Existing experiment modules are left unchanged.
"""

from contextlib import contextmanager, nullcontext

from .next_sana import NextSanaAdapter, _CompiledRegion, _TransformerGraph


class FollowupSanaAdapter(NextSanaAdapter):
    """Add ``vae_graph`` while preserving the original three candidate APIs."""

    _CANDIDATES = ("eager", "vae_compile", "transformer_graph", "vae_graph")
    _GRAPH_CANDIDATES = ("transformer_graph", "vae_graph")

    def _release_inactive_graph(self, target):
        # A baseline without graph replay must not retain the candidate's live
        # capture pool and static tensors. Allocator-reserved memory is still
        # reported by the runner rather than hidden with empty_cache().
        graph = self._transformer_graph
        if target in self._GRAPH_CANDIDATES or graph is None:
            return
        try:
            graph.release()
        except BaseException as error:
            self._candidate_failed = self._cleanup_failed = True
            self._restore()
            self._failures.append(dict(candidate=self.candidate,
                target_candidate=target, phase="candidate_graph_teardown",
                error=type(error).__name__ + ": " + str(error),
                transformer_graph=graph.summary(),
                vae_regions=[region.summary() for region in self._vae_regions.values()]))
            self._vae_regions = {}
            raise

    def set_candidate(self, candidate):
        if candidate not in self._CANDIDATES:
            raise ValueError("unknown Sana candidate")
        if self._busy or self._next_lock.locked():
            raise RuntimeError("cannot change candidate during a Sana invocation")
        if self._cleanup_failed:
            raise RuntimeError("Sana graph teardown failed; a fresh process is required")
        self._release_inactive_graph(candidate)
        if candidate != "vae_graph":
            return super().set_candidate(candidate)

        self._restore()
        self.candidate, self._candidate_failed = candidate, False
        self.pipe.transformer = self._eager_transformer
        self.backend = "eager"
        try:
            decoder = getattr(self.pipe.vae, "decoder", None)
            if decoder is None or not callable(getattr(decoder, "forward", None)):
                raise RuntimeError("Sana VAE has no pure decoder module")
            for name in ("decoder", "post_quant_conv"):
                module = getattr(self.pipe.vae, name, None)
                if module is None:
                    continue
                if name not in self._vae_regions:
                    self._vae_regions[name] = _CompiledRegion(self, name, module.forward)
                self._install(module, "forward", self._vae_regions[name])
            if self._transformer_graph is None:
                self._transformer_graph = _TransformerGraph(self, self._transformer_forward)
            self._install(self._eager_transformer, "forward", self._transformer_graph)
        except BaseException as error:
            self._fail(error)
            raise
        return self.candidate_summary()

    @contextmanager
    def steady_context(self):
        self._steady_depth += 1
        try:
            # The graph remains eager. Only VAE pure regions are compiled, and
            # neither initial compilation nor recapture is allowed in steady
            # measurements. _CompiledRegion also verifies backend execution.
            stance = (self.torch.compiler.set_stance("fail_on_recompile")
                      if self.candidate in ("vae_compile", "vae_graph")
                      else nullcontext())
            with stance:
                yield
        finally:
            self._steady_depth -= 1

    def candidate_summary(self):
        summary = super().candidate_summary()
        summary.update(
            combination=self.candidate == "vae_graph",
            graph_released_when_inactive=True,
            qualification_scope="execution mechanics only; combined quality, capacity and online results required",
        )
        return summary
