"""Dependency-free USP process-group slots shared with xFuser."""

from types import SimpleNamespace


# ALL_TO_ALL: optional override fn(flat_tensor, ulysses_pg) -> tensor | None, installed when a
# non-NCCL collective backend (group-free collectives) carries the Ulysses exchange.
PROCESS_GROUP = SimpleNamespace(ULYSSES_PG=None, RING_PG=None, ALL_TO_ALL=None)
