"""Controller-owned stop doorbells for continuous block execution.

The controller keeps one 8-byte little-endian counter per GPU in a shared-memory
file and increments it when it wants the work running on that GPU to stop at
its next block boundary. A continuous advance carries the counter value seen at
dispatch (its epoch); the worker stops once the file holds a larger value.
Reading the counter is a memory load: no syscall, no lock, no RPC.
"""

import mmap
import os
import struct

_SIZE = 8
_open = {}


class Doorbell:
    def __init__(self, path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            if os.fstat(descriptor).st_size < _SIZE:
                raise ValueError("doorbell file is shorter than 8 bytes: " + path)
            self._map = mmap.mmap(descriptor, _SIZE, prot=mmap.PROT_READ)
        finally:
            os.close(descriptor)
        self.path = path

    def value(self):
        return struct.unpack_from("<Q", self._map, 0)[0]

    def close(self):
        self._map.close()


def doorbell(path):
    """The resident mapping of one doorbell file (a worker reuses it for every dispatch)."""
    if not isinstance(path, str) or not path:
        raise ValueError("doorbell must be a file path")
    bell = _open.get(path)
    if bell is None:
        bell = _open[path] = Doorbell(path)
    return bell


def rung(request):
    """A callable that is true once the request's doorbell passed its dispatch epoch, or None."""
    path = request.get("doorbell")
    if path is None:
        return None
    epoch = request.get("doorbell_epoch")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("doorbell_epoch must be a nonnegative integer")
    bell = doorbell(path)
    return lambda: bell.value() > epoch


def close_all():
    for bell in _open.values():
        bell.close()
    _open.clear()
