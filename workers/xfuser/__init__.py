try:
    from xfuser.model_executor.pipelines import (
        xFuserStableDiffusion3Pipeline,
    )
    from xfuser.config import xFuserArgs, EngineConfig
    from xfuser.parallel import xDiTParallel
except ImportError:
    xFuserStableDiffusion3Pipeline = None
    xFuserArgs = None
    EngineConfig = None
    xDiTParallel = None

__all__ = [
    "xFuserStableDiffusion3Pipeline",
    "xFuserArgs",
    "EngineConfig",
    "xDiTParallel",
]
