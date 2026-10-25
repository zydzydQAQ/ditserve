"""Pipeline wrappers that keep DITSERVE's batching and execution recipes."""


class ServingPipelineWrapper:
    def _naive_forward(self, *args, **kwargs):
        self.activate_request(args, kwargs)
        return super()._naive_forward(*args, **kwargs)

    def _convert_transformer_backbone(self, transformer, pipe=None, enable_torch_compile=False,
                                      enable_onediff=False, cache_args=None):
        if enable_torch_compile or enable_onediff or cache_args:
            raise ValueError("pipeline compilation and xDiT-internal caches are configured by DITSERVE")
        return self.convert_transformer(transformer)
