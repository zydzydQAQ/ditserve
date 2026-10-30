"""Deprecated compatibility layer for the former cache adapter API."""
import warnings


def apply_cache_on_transformer(transformer, *args, **kwargs):
    warnings.warn(
        "xfuser.model_executor.cache.diffusers_adapters is deprecated; "
        "use xfuser.model_executor.cache.adapters instead.",
        FutureWarning,
        stacklevel=2,
    )
    use_cache = kwargs.get("use_cache", "Fb")
    from xfuser.model_executor.cache.adapters.flux import (
        apply_fbcache,
        apply_teacache,
    )

    if use_cache == "Tea":
        kwargs.pop("use_cache", None)
        return apply_teacache(transformer, *args, **kwargs)
    return apply_fbcache(transformer, *args, **kwargs)
