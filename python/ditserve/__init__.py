"""Lightweight DiT serving lab: C++ control plane behind a Python API."""

from .api import doctor, simulate, sweep
from .core import CoreBridge, CoreError, POLICIES, SCHEMA_VERSION

__version__ = "0.3.0"
__all__ = ["CoreBridge", "CoreError", "POLICIES", "SCHEMA_VERSION", "doctor", "simulate", "sweep", "__version__"]
