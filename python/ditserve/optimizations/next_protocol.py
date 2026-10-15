"""Frozen follow-on study: image feasibility, Sana serving, cache load capacity."""
import copy
from pathlib import Path

from ..research.common import digest, file_digest, read_json
from .protocol import ROOT, CONFIG as ORIGINAL_CONFIG, load_config as original_config

CONFIG = ROOT / "configs/serving_next_validation.json"
OUTPUT = ROOT / "results/optimization-next/run01"
LEDGER = ROOT / "results/optimization-next/gpu-budget.json"
AUTHORIZATION = ROOT / "results/optimization-next/authorization.json"
PHASES = ("diagnosis", "sana_vae", "sana_graph", "sana_numerics",
          "cache_load_05", "cache_load_07", "cache_load_09", "conditional")
PARENTS = tuple(ROOT / name for name in (
    "results/research-directions/gpu-budget.json",
    "results/optimization-screen/gpu-budget.json",
    "results/optimization-cache-online/gpu-budget.json"))
DEFAULTS = dict(sana_backend="eager", image_batch_size=1, wan_backend="eager",
                vae_backend="eager", encode_device="cpu", encode_on_scheduler=False,
                policy="block-4")


def default_config():
    base = original_config(ORIGINAL_CONFIG)
    value = copy.deepcopy(base)
    value.update(schema_version="ditserve.optimizations.next_config.v1", order=list(PHASES),
        budget_gpu_seconds=None, budget_unlimited=True,
        stage_budgets_gpu_seconds={stage: None for stage in PHASES},
        artifact_root=str(ROOT / ".cache/optimization-next"),
        scope="finite paired research matrices with unlimited lifetime accounting; no p99")
    value["next_study"] = dict(
        inherited_config_sha256=digest(base),
        historical_ledgers=[dict(path=str(p), sha256=file_digest(p)) for p in PARENTS],
        defaults=DEFAULTS,
        diagnosis=dict(arms=["image_only", "mixed"], orders=[["image_only", "mixed"], ["mixed", "image_only"]],
            wan_weights_resident=True, preserve_image_arrivals_deadlines=True),
        sana=dict(candidates=["vae_compile", "transformer_graph"], local_repeats=3,
            graph_trigger_host_overhead_fraction=.05, qualification="8 inputs, AB/BA, LPIPS <= .01, 5% both orders, capacity then fixed online"),
        cache_loads=[.5, .7, .9], seeds=[101, 102, 103], patterns=["ordinary", "burst"],
        cache_arms=["off_k4", "paired_k4"], eager_sana_for_cache=True,
        conditional=dict(adaptive_k="only after measured conflicting K1/K4/K8 tradeoffs",
            video_vae="only when targeted decode collision evidence is present",
            v100_encoder="only on observed encoder starvation and idle V100",
            cache_memory="only at observed real paused-state capacity boundary"))
    value["next_workloads"] = dict(expected_source_sha256=file_digest(base["old_online"]),
        expected_calibration=read_json(base["old_online"])["frozen_service_ms"])
    return value


def load_config(path=CONFIG):
    value = read_json(path)
    if value != default_config():
        raise ValueError("follow-on protocol or historical bindings changed")
    return value


def authorization(path):
    path = Path(path).resolve(strict=True)
    value = read_json(path)
    if (value.get("scope") != "serving_next_validation" or value.get("approved_by_user") is not True
            or value.get("budget_unlimited") is not True or "authorized_gpu_seconds" not in value
            or value["authorized_gpu_seconds"] is not None
            or "就按照这个继续" not in value.get("user_instruction", "")
            or "取消8小时GPU小时硬上限，GPU无上限，开始在线对照" not in value.get("user_instruction", "")):
        raise ValueError("current continuation must bind the existing unlimited GPU authorization")
    return value, dict(path=str(path), sha256=file_digest(path))


def safe_paths(output, ledger):
    output, ledger = Path(output).resolve(), Path(ledger).resolve()
    for parent in PARENTS:
        if any(path == parent.parent or parent.parent in path.parents for path in (output, ledger)):
            raise ValueError("new study must be outside all historical result trees")
    if ledger.name != "gpu-budget.json" or ledger.parent not in output.parents:
        raise ValueError("independent gpu-budget.json must be above the run directory")
    return output, ledger
