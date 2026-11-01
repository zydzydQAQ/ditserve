"""Data preparation and scientific evaluation behind one worker operation.

These adapters do not schedule experiments or allocate GPU time. Downloads
require ``dataset`` with ``download=True``; the official evaluator uses an
already installed, pinned checkout and local weights. Optional numerical and
perceptual libraries are imported only by operations that need them.
"""

from .artifacts import check_backend_quality, evaluate_artifacts, evaluate_run, freeze_outputs
from .datasets import prepare_dataset
from .vbench import export_vbench_manifest, export_vbench_videos, run_vbench


EVALUATORS = {
    "dataset": prepare_dataset,
    "paired": evaluate_artifacts,
    "paired_run": evaluate_run,
    "outputs": freeze_outputs,
    "backend": check_backend_quality,
    "vbench_manifest": export_vbench_manifest,
    "vbench_export": export_vbench_videos,
    "vbench": run_vbench,
}


def evaluate(request):
    """Dispatch explicit evaluator parameters; transport fields are not data."""
    name = request.get("evaluator")
    if name not in EVALUATORS:
        raise ValueError("unknown evaluator: " + str(name))
    parameters = {key: value for key, value in request.items()
                  if key not in {"version", "id", "op", "evaluator", "rank", "world_size", "device"}}
    return EVALUATORS[name](**parameters)


__all__ = ["evaluate", "prepare_dataset", "evaluate_artifacts", "evaluate_run", "freeze_outputs",
           "check_backend_quality", "export_vbench_manifest", "export_vbench_videos", "run_vbench"]
