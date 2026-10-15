"""Generate deployment bindings without modifying frozen scientific protocols."""

import json
from pathlib import Path
import sys

from ..util import ROOT, canonical_gpu_uuid


REVISIONS = {
    "wan": ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"),
    "sana": ("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers", "aa76e7f4f4928f378716b6716a2130fba3caf5b1"),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.2", "63a8b081895390a26e140280378bc85ec8bce07a"),
}
LONG_SOURCE = "c6a27e771e803b1c12ffa28b82ba69b42472020e"
LONG_CHECKPOINT = "cda9138d93872ef109d9a74c270184eb3a5acc6e"


def environment_config(model_root, data_root, gpu_uuid, *, python=None,
                       longlive_python="/opt/longlive-venv/bin/python",
                       longlive_source="/opt/longlive-source"):
    normalized_uuid = canonical_gpu_uuid(gpu_uuid)
    if normalized_uuid is None:
        raise ValueError("provide a full physical GPU UUID from migrate inspect")
    gpu_uuid = "GPU-" + normalized_uuid
    roots = [Path(model_root), Path(data_root), Path(python or sys.executable),
             Path(longlive_python), Path(longlive_source)]
    if not all(path.is_absolute() for path in roots):
        raise ValueError("container model/data/interpreter/source paths must be absolute")
    model_root, data_root, python, longlive_python, longlive_source = map(str, roots)
    models = {name: dict(id=repo, revision=revision, path=str(Path(model_root) / name),
                        verification_manifest=str(Path(model_root) / ".provisioning" / (name + ".json")))
              for name, (repo, revision) in REVISIONS.items()}
    models["longlive"] = dict(id="Efficient-Large-Model/LongLive-1.3B", revision=LONG_SOURCE,
        checkpoint_revision=LONG_CHECKPOINT, path=str(Path(model_root) / "longlive/weights"),
        source_path=longlive_source, python=longlive_python,
        verification_manifest=str(Path(model_root) / ".provisioning/longlive.json"))
    return dict(schema_version="ditserve.migration.environment.v1", device="cuda:0", gpu_uuid=gpu_uuid,
                python=python, models=models, model_root=model_root, data_root=data_root,
                torch_home=str(Path(model_root) / "torch"), results_root=str(Path(data_root) / "results"),
                artifact_root=str(Path(data_root) / "cache"), session_profile="a100_fp32",
                direction3=json.loads((ROOT / "configs/research_directions.json").read_text())["direction3"],
                scope="environment_acceptance_only", research_qualification="not_inherited",
                historical_continuation="V100-only study protocols require a separately qualified new study")


def configure(destination, model_root, data_root, gpu_uuid, **kwargs):
    value = environment_config(model_root, data_root, gpu_uuid, **kwargs)
    destination = Path(destination)
    # Repeated setup is safe; changed identities must use a new config directory.
    scheduling = json.loads((ROOT / "configs/research_scheduling.json").read_text())
    scheduling.update(python=value["python"], v100_python=value["python"], torch_home=value["torch_home"],
        models={name: value["models"][name] for name in ("wan", "sana")}, device_preference=["A100"],
        gpu_uuid=value["gpu_uuid"], legacy_run=str(Path(data_root) / "history/results/research-directions/run01"))
    documents = {"environment.json": value, "scheduling-a100.json": scheduling}
    for name, document in documents.items():
        path = destination / name
        if path.exists() and json.loads(path.read_text()) != document:
            raise ValueError("existing deployment binding differs; use a new directory: " + str(path))
    destination.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        path = destination / name
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(document, indent=2) + "\n")
            temporary.replace(path)
    return dict(status="configured", gpu_work_started=False, files=[str(destination / n) for n in documents],
                scope=value["scope"], research_qualification="not_inherited",
                note="Scheduling starts at a new calibrate phase; no prepare/run commands were executed")


def validate_environment(value):
    """Bind every runtime path and model revision to the verified deployment root."""
    if value.get("schema_version") != "ditserve.migration.environment.v1":
        raise ValueError("generate an environment config using migrate configure first")
    try:
        expected = environment_config(value["model_root"], value["data_root"], value["gpu_uuid"],
            python=value["python"], longlive_python=value["models"]["longlive"]["python"],
            longlive_source=value["models"]["longlive"]["source_path"])
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete environment configuration: " + str(error)) from error
    for key in expected:
        if value.get(key) != expected[key]:
            raise ValueError("stale or changed deployment binding: " + key + "; run migrate configure in a new directory")
    if Path(value["python"]).absolute() != Path(sys.executable).absolute():
        raise ValueError("configured interpreter differs from the running environment; regenerate configuration")
    return value
