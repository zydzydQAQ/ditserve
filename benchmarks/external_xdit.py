"""Audit and import the real pinned xDiT, separately from DitServe's xfuser fork.

No model is loaded and no scheduler, process group, or service is started. The
native runner calls complete pipelines; passing these diagnostics cannot qualify
a mixed-workload benchmark. All imports execute in fresh isolated interpreters.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

COMMIT = "16c34e7f070de65d9d2c67a6d74e0007dd8b426a"
REPOSITORY = "xdit-project/xDiT"
ARCHIVE_SHA256 = "c09e48fa4792290b299ab3cfc0afc9c11ce60ac24affee933b1a0ce146e52f5b"
MODEL_IDS = {
    "wan21": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
}
RUNNER_FILES = ("xfuser/model_executor/models/runner_models/wan.py",
                "xfuser/model_executor/models/runner_models/stable_diffusion.py")
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs/benchmarks/formal-mixed-v1.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode())


def read_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def verify_source(source_root, manifest_path, manifest_sha256):
    root = Path(source_root)
    if root.is_symlink():
        raise ValueError("symlink source root")
    root = root.resolve(strict=True)
    raw = Path(manifest_path).read_bytes()
    if not re.fullmatch("[0-9a-f]{64}", manifest_sha256) or digest(raw) != manifest_sha256:
        raise ValueError("official source manifest SHA256 differs")
    manifest = read_json(raw)
    if (manifest.get("schema") != "ditserve.official-source-archive.v1"
            or manifest.get("repository") != REPOSITORY
            or manifest.get("commit") != COMMIT
            or manifest.get("archive_sha256") != ARCHIVE_SHA256):
        raise ValueError("official source identity differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("official source inventory is empty")
    for relative, expected in files.items():
        parts = PurePosixPath(relative)
        if (parts.is_absolute() or ".." in parts.parts or str(parts) != relative
                or not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected)):
            raise ValueError("invalid source inventory entry: " + relative)
        path = root / relative
        if any(root.joinpath(*parts.parts[:n]).is_symlink() for n in range(1, len(parts.parts) + 1)):
            raise ValueError("symlink in source: " + relative)
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError("official source file SHA256 differs: " + relative)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in source: " + str(path))
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError("official source inventory differs (extra or missing files)")
    return manifest


def python_inventory(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("symlink xfuser root")
    root = root.resolve(strict=True)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in xfuser inventory: " + str(path))
        if path.is_file() and path.suffix == ".py":
            files[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    if "__init__.py" not in files:
        raise ValueError("xfuser package is missing")
    return files


def compare_backends(source_root, shared_xfuser_root):
    upstream = python_inventory(Path(source_root) / "xfuser")
    shared = python_inventory(shared_xfuser_root)
    common = upstream.keys() & shared.keys()
    modified = {p: {"official_sha256": upstream[p], "shared_sha256": shared[p]}
                for p in sorted(common) if upstream[p] != shared[p]}
    same = sorted(p for p in common if upstream[p] == shared[p])
    omitted = sorted(upstream.keys() - shared.keys())
    extra = sorted(shared.keys() - upstream.keys())
    return {
        "scope": "complete Python source inventories; runtime loaded-module identities collected separately",
        "official_python_files": len(upstream), "shared_python_files": len(shared),
        "identical_count": len(same), "modified_count": len(modified),
        "omitted_count": len(omitted), "shared_only_count": len(extra),
        "modified": modified, "omitted": omitted, "shared_only": extra,
        "official_inventory_sha256": canonical(upstream),
        "shared_inventory_sha256": canonical(shared), "shared_files": shared,
        "identical_python_source": upstream == shared,
        "numerical_equivalence_verified": False,
        "revision_label_is_sufficient_identity": False,
    }


def registered_models(source_root):
    registry = {}
    for relative in RUNNER_FILES:
        parsed = ast.parse((Path(source_root) / relative).read_text(), filename=relative)
        for node in parsed.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for deco in node.decorator_list:
                if (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name)
                        and deco.func.id == "register_model" and len(deco.args) == 1):
                    name = ast.literal_eval(deco.args[0])
                    if name in registry:
                        raise ValueError("duplicate official model registration: " + name)
                    registry[name] = {"class": node.name, "file": relative}
    return registry


def audit(source_root, manifest_path, manifest_sha256, shared_xfuser_root, protocol_path=DEFAULT_PROTOCOL):
    manifest = verify_source(source_root, manifest_path, manifest_sha256)
    registry = registered_models(source_root)
    protocol_raw = Path(protocol_path).read_bytes()
    protocol = read_json(protocol_raw)
    if protocol.get("schema") != "ditserve.formal-protocol.v1" or protocol.get("id") != "formal-mixed-v1":
        raise ValueError("formal protocol identity differs")
    current = {}
    for name, model in MODEL_IDS.items():
        current[name] = {"model_id": model, "registered": model in registry,
                         "native_class": registry.get(model), "runtime_verified": False,
                         "formal_eligible": False}
    current["wan21"].update(native_parallelism="Ulysses/ring SP; runner does not advertise CFG parallelism",
                             native_dtype="bfloat16", native_default_frames=81)
    current["wan22"].update(native_parallelism="Ulysses/ring SP; CFG parallelism is explicitly rejected for task=t2v",
                             native_dtype="bfloat16", native_default_height=736, native_default_frames=121)
    current["sd35"].update(blocker="official runner registers SD3.5 large (38 heads), not current medium; validated model port required")
    paths = (*RUNNER_FILES, "xfuser/runner.py", "xfuser/cli.py", "entrypoints/launch.py")
    return {
        "schema": "ditserve.external-xdit-probe.v1", "status": "formal_adapter_incomplete",
        "official_source": {"repository": REPOSITORY, "commit": COMMIT,
            "manifest_sha256": manifest_sha256, "archive_sha256": ARCHIVE_SHA256,
            "verified_files": len(manifest["files"]),
            "execution_source_sha256": {p: manifest["files"][p] for p in paths}},
        "formal_protocol": {"id": protocol["id"], "raw_sha256": digest(protocol_raw)},
        "shared_xfuser": compare_backends(source_root, shared_xfuser_root),
        "current_models": current, "registered_models": registry,
        "official_entrypoints": {"cli": "xdit = xfuser.cli:main (launches torchrun xfuser/runner.py)",
            "python": "xfuser.runner.xFuserModelRunner: preprocess_args -> initialize -> run -> cleanup",
            "http": "entrypoints/launch.py (Ray/FastAPI, image-only model map)"},
        "native_execution_unit": "complete diffusion pipeline call; no resumable service-step API in the runner",
        "blockers": ["SD3.5 medium is absent from the official runner registry",
            "official CLI model selection is registry-based and runner settings use Hugging Face repository IDs; local offline weight mapping must be verified",
            "shared xfuser is a modified subset of upstream; imports must never silently mix the two trees",
            "native dtype, scheduler, attention backend, checkpoint hashes and workload parameters require matched validation",
            "persistent mixed-model request replay, deadline accounting and original-denominator output adapter are not implemented"],
        "qualification_eligible": False, "allows": [], "measurements": [], "serving_executed": False,
    }


_IMPORT_PROBE = r'''
import contextlib, hashlib, importlib, importlib.metadata, io, json, pathlib, sys, traceback
root = pathlib.Path(sys.argv[1]).resolve()
mode = sys.argv[2]
sys.path.insert(0, str(root if mode == 'official' else root.parent))
package_root = root / 'xfuser' if mode == 'official' else root
result = {'python':sys.version.split()[0], 'mode':mode, 'imports':[], 'packages':{}}
for package in ['torch','diffusers','transformers','cache-dit','numpy','ray','fastapi','flash-attn','distvae']:
    try: result['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: result['packages'][package] = None
names = ['xfuser', 'xfuser.runner', 'xfuser.model_executor.models.runner_models.wan',
         'xfuser.model_executor.models.runner_models.stable_diffusion', 'entrypoints.launch'] if mode == 'official' else [
         'xfuser', 'xfuser.model_executor.models.transformers.transformer_wan',
         'xfuser.model_executor.models.transformers.transformer_sd3',
         'xfuser.model_executor.pipelines.pipeline_stable_diffusion_3']
for name in names:
    entry = {'module':name, 'status':'import_failed'}
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            module = importlib.import_module(name)
        path = pathlib.Path(module.__file__).resolve()
        expected = root / 'entrypoints' if name.startswith('entrypoints.') else package_root
        if not path.is_relative_to(expected): raise ValueError('module imported outside verified source: '+str(path))
        entry.update(status='imported', path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if name == 'xfuser.runner':
            entry['registered_models'] = sorted(module.MODEL_REGISTRY)
    except Exception as exc:
        entry.update(error_type=type(exc).__name__, error=str(exc)[:2000], traceback=traceback.format_exc()[-6000:])
    entry['diagnostic_tail'] = captured.getvalue()[-3000:]
    result['imports'].append(entry)
result['loaded_xfuser_modules'] = {}
result['loaded_xfuser_namespace_packages'] = {}
result['source_violations'] = []
for name, module in sorted(sys.modules.items()):
    if name == 'xfuser' or name.startswith('xfuser.'):
        origin = getattr(module, '__file__', None)
        if origin:
            path = pathlib.Path(origin).resolve()
            result['loaded_xfuser_modules'][name] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            if not path.is_relative_to(package_root): result['source_violations'].append(name)
        else:
            locations = [str(pathlib.Path(p).resolve()) for p in getattr(module, '__path__', [])]
            result['loaded_xfuser_namespace_packages'][name] = locations
            if not locations or any(not pathlib.Path(p).is_relative_to(package_root) for p in locations):
                result['source_violations'].append(name)
print('XDIT_PROBE_JSON=' + json.dumps(result, sort_keys=True))
'''


def import_one(root, mode, *, python=sys.executable, timeout=60):
    if mode not in ("official", "shared"):
        raise ValueError("invalid import mode")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 300):
        raise ValueError("import timeout must be in (0, 300] seconds")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    command = [str(python), "-I", "-B", "-c", _IMPORT_PROBE, str(Path(root).resolve()), mode]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, env=env,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "import_probe_timeout", "mode": mode, "timeout_seconds": timeout,
                "serving_executed": False}
    marker = "XDIT_PROBE_JSON="
    reports = [line[len(marker):] for line in proc.stdout.splitlines() if line.startswith(marker)]
    if proc.returncode != 0 or len(reports) != 1:
        return {"status": "import_probe_process_failed", "mode": mode, "returncode": proc.returncode,
                "diagnostic_tail": (proc.stdout + proc.stderr)[-6000:], "serving_executed": False}
    result = read_json(reports[0])
    result["status"] = ("imports_passed" if not result["source_violations"]
                        and all(i["status"] == "imported" for i in result["imports"]) else "imports_failed")
    result["serving_executed"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "import-probe"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--shared-xfuser-root", type=Path, required=True)
    parser.add_argument("--shared-xfuser-inventory-sha256")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=60, help="timeout per isolated interpreter")
    args = parser.parse_args(argv)
    try:
        result = audit(args.source_root, args.source_manifest, args.source_manifest_sha256,
                       args.shared_xfuser_root, args.protocol)
        before = result["shared_xfuser"]["shared_inventory_sha256"]
        if args.shared_xfuser_inventory_sha256 and args.shared_xfuser_inventory_sha256 != before:
            raise ValueError("shared xfuser inventory SHA256 differs")
        if args.command == "import-probe":
            result["runtime_probes"] = [import_one(args.source_root, "official", python=args.python, timeout=args.timeout),
                import_one(args.shared_xfuser_root, "shared", python=args.python, timeout=args.timeout)]
            verify_source(args.source_root, args.source_manifest, args.source_manifest_sha256)
            if canonical(python_inventory(args.shared_xfuser_root)) != before:
                raise ValueError("shared xfuser source changed during imports")
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 2  # Source/import diagnostics are not a runnable formal reproduction.
    except (ValueError, OSError, SyntaxError, StopIteration, KeyError) as exc:
        print(json.dumps({"schema": "ditserve.external-xdit-probe.v1", "status": "invalid_evidence",
            "error": str(exc), "qualification_eligible": False, "allows": [], "measurements": [],
            "serving_executed": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
