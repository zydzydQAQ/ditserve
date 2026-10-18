"""Optional Python client for the native CLI and JSON RPC interface."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import PACKAGE_ROOT, ROOT, strict_json

CLUSTER_POLICIES = ("static_sp1", "static_sp2", "static_sp4", "static_sp8", "step_elastic", "block_elastic")


def native_binary(binary=None):
    selected = binary or os.environ.get("DITSERVE_CORE")
    if selected:
        if isinstance(selected, os.PathLike) or os.path.dirname(selected):
            return Path(selected).resolve()
        located = shutil.which(selected)
        return Path(located).resolve() if located else Path(selected)
    repository = ROOT / ".cache/build/ditserve"
    for candidate in (repository, Path(sys.prefix) / "bin/ditserve"):
        if candidate.is_file():
            return candidate
    located = shutil.which("ditserve")
    return Path(located).resolve() if located else repository


def native_environment():
    environment = dict(os.environ)
    for key, value in (("DITSERVE_PYTHON", sys.executable), ("DITSERVE_PACKAGE_ROOT", PACKAGE_ROOT),
                       ("DITSERVE_ROOT", ROOT)):
        environment.setdefault(key, str(value))
    return environment


def _json_value(value):
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class CoreError(ValueError):
    """The native process rejected a request or returned invalid JSON."""


class CoreBridge:
    """One native RPC per call. Python does not supply experiment defaults."""

    def __init__(self, binary=None):
        self.binary = native_binary(binary)

    def available(self):
        return self.binary.is_file()

    def call(self, command, payload):
        if not self.available():
            raise CoreError("C++ core not found at %s; build with cmake or set DITSERVE_CORE" % self.binary)
        result = subprocess.run([str(self.binary), "rpc", command],
            input=json.dumps(_json_value(payload), allow_nan=False),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=native_environment())
        if result.returncode not in (0, 1) or (result.returncode == 1 and not result.stdout.strip()):
            raise CoreError(result.stderr.strip() or "C++ core exited %s" % result.returncode)
        try:
            return strict_json(result.stdout)
        except ValueError as error:
            raise CoreError("C++ core returned invalid JSON: %s" % error)

    def generate(self, generator):
        return self.call("generate", {"generator": generator})

    def simulate(self, config):
        """One cluster simulation: workload or generator, profile, policy, gpus, options."""
        return self.call("simulate", config)


def _call(command, core, options):
    return (core or CoreBridge()).call(command, _json_value({key: value for key, value in options.items()
                                                          if value is not None}))


def doctor(core=None):
    return _call("doctor", core, {})


def run(config, *, core=None, **options):
    return _call("run", core, dict(options, config=config))


def simulate(config=None, *, core=None, **options):
    return _call("simulate-study", core, dict(options, config=config))
