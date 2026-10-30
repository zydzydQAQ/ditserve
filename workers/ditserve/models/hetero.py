"""Unified HeteroDiT optimization facade exporting all v1/v2/v3 runtime modules."""

from .encoder_pool import ProactiveEncoderPool
from .sd35 import SD35Backend
from .sd35_cache import (
    configure_sd35_fast_cache,
    install_sd35_fast_cache_hooks,
    pop_sd35_reused_steps,
)
from .wan.backend import WanBackend
from .wan.cfg_reuse import AsymmetricCFGStepper
from .wan.ring_quant import AsymmetricRingQuantizer
from .wan.taylor_cache import TaylorResidualCompensator
from .wan.tiered_state import PinnedSlabPool, TieredVideoStateManager
from .wan.vae_overlap import ConcurrentVAEDecodeStreamer

# Convenient aliases across v1/v2/v3 scripts
SharedEncoderPool = ProactiveEncoderPool
PinnedStatePool = PinnedSlabPool
TieredWanStateManager = TieredVideoStateManager
RingKVQuantizer = AsymmetricRingQuantizer
StreamIsolated3DVAEStreamer = ConcurrentVAEDecodeStreamer

__all__ = [
    "SD35Backend",
    "WanBackend",
    "ProactiveEncoderPool",
    "SharedEncoderPool",
    "PinnedSlabPool",
    "PinnedStatePool",
    "TieredVideoStateManager",
    "TieredWanStateManager",
    "AsymmetricRingQuantizer",
    "RingKVQuantizer",
    "TaylorResidualCompensator",
    "AsymmetricCFGStepper",
    "ConcurrentVAEDecodeStreamer",
    "StreamIsolated3DVAEStreamer",
    "install_sd35_fast_cache_hooks",
    "configure_sd35_fast_cache",
    "pop_sd35_reused_steps",
]
