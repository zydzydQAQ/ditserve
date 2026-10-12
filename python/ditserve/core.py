"""Subprocess bridge to the C++ control plane (schema v2 JSON stdio protocol)."""

import json
import os
from pathlib import Path
import subprocess

from .util import ROOT

SCHEMA_VERSION = 2
POLICIES = ("cache_only", "batch_only", "naive", "edf", "path_aware", "state_aware")


class CoreError(ValueError):
    """The C++ core rejected the request or answered with invalid JSON."""


def _non_finite(value):
    raise ValueError("non-finite JSON number: " + value)


class CoreBridge:
    """One core command per call: generate | simulate. Diagnostics stay on stderr."""

    def __init__(self, binary=None):
        self.binary = Path(binary or os.environ.get("DITSERVE_CORE", str(ROOT / ".cache" / "build" / "ditserve-core")))

    def available(self):
        return self.binary.is_file()

    def call(self, command, payload):
        if not self.binary.is_file():
            raise CoreError("C++ core not found at %s; build with cmake or set DITSERVE_CORE" % self.binary)
        result = subprocess.run([str(self.binary), command], input=json.dumps(payload, allow_nan=False),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            raise CoreError(result.stderr.strip() or "C++ core exited %s" % result.returncode)
        try:
            return json.loads(result.stdout, parse_constant=_non_finite)
        except ValueError as error:
            raise CoreError("C++ core returned invalid JSON: %s" % error)

    def generate(self, generator, cost_model=None):
        return self.call("generate", {"schema_version": SCHEMA_VERSION, "cost_model": cost_model or {},
                                      "generator": generator})

    def simulate(self, config, workload, cost_model=None):
        return self.call("simulate", {"schema_version": SCHEMA_VERSION, "config": config,
                                      "cost_model": cost_model or {}, "workload": workload})
