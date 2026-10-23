"""Prepare encoding comparisons on CPU without registering or launching GPU jobs.

The current study and its cumulative budget are read-only inputs. Emitted native
configs still require numerical qualification, a frozen revision, and admission
through the study launcher; this command cannot increase the budget or execute.
"""

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workers"))

from ditserve.evaluation import workflow_contention_study as contention
from ditserve.evaluation import workflow_encoding_optimization_study as encoding
from ditserve.evaluation import workflow_encoding_qualification as qualification
from ditserve.evaluation import workflow_next_study as api


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("write the proposal outside the historical study directory")
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing proposal: " + str(output))
    study_path, budget_path = root / "study.json", root / "budget.json"
    before = {str(path): api.binding(path) for path in (study_path, budget_path)}
    study, budget = api.read(study_path), api.read(budget_path)
    if not isinstance(study, dict) or not isinstance(budget, dict):
        raise ValueError("an existing study and its cumulative budget are required")
    if Path(study["root"]).resolve() != root:
        raise ValueError("study root does not match the requested directory")
    if study.get("active_job") or budget.get("active"):
        raise ValueError("wait for the active study task to settle before preparing a proposal")
    plan = encoding.build_plan(study, budget)
    profile_path, calibration_path = root / "profile.json", root / "calibration.json"
    calibration_job = contention.report.current_job(study, "calibration-video")
    calibration_inputs = (Path(calibration_job["output"]) / "run.json",
                          root / "launches" / calibration_job["id"] / "launch.json")
    for path in (profile_path, calibration_path, *calibration_inputs,
                 Path(study["source"]) / "configs/env/rtx4090-2x24g.json"):
        before[str(path)] = api.binding(path)
    profile, calibration = api.read(profile_path), api.read(calibration_path)
    if not profile or not calibration or calibration.get("profile") != profile:
        raise ValueError("the original frozen profile and calibration must agree")
    _, video_cells, calibration_bindings = contention._calibration(study, "calibration-video")
    if any(before.get(row["path"]) != row for row in calibration_bindings):
        raise RuntimeError("calibration changed while preparing the proposal")
    anchors = contention._anchors(profile, video_cells)
    configs = []
    for optimization, seeds, distributions in (
            ("sana_prompt_cache", encoding.SANA_SEEDS, ("unique", "repeated")),
            ("wan_negative_cache", encoding.WAN_SEEDS, ("mixed_negative",))):
        for repeat, seed in enumerate(seeds):
            rows = contention.contention_trace(profile, seed, anchors)
            for distribution in distributions:
                config = encoding.native_online_config(profile, rows,
                    optimization=optimization, prompt_distribution=distribution, repeat=repeat)
                identifier = "%s-%s-s%d" % (optimization.replace("_", "-"), distribution, seed)
                config["experiment"] = "encoding-optimization-" + identifier
                # A per-task ceiling is not a new cumulative allowance. These
                # drafts cannot be launched until the original study admits them.
                config["budget"] = dict(limit_gpu_seconds=3600, watchdog_seconds=1800,
                                        repair_reserve_gpu_seconds=0, carry="none")
                for phase in config["phases"]:
                    phase["controller_cpu_affinity"] = [59]
                    phase["artifact_dependencies"] = [dict(closed=False,
                        reason="encoding optimization numerical and performance review pending")]
                configs.append((identifier, config))
    environment = api.environment(study, (0, 1))
    qualification_environment = api.environment(study, (1,))
    qualification_configs = [(job["id"], encoding.native_qualification_config(job,
                                gpu_uuid=study["gpu_mapping"][1]["uuid"]))
                             for job in plan["jobs"] if job["stage"] == "qualification"]
    qualification_configs = [(identifier, qualification.normalize_config(config["kind"], config))
                             for identifier, config in qualification_configs]
    inputs = list(before.values())
    sources = [api.binding(REPO / path) for path in (
        "docker/plan-encoding-optimization.py",
        "workers/ditserve/evaluation/workflow_encoding_optimization_study.py",
        "workers/ditserve/evaluation/workflow_encoding_qualification.py",
        "workers/ditserve/models/sana/backend.py", "workers/ditserve/models/sana/runtime.py",
        "workers/ditserve/models/wan/backend.py", "workers/ditserve/models/wan/runtime.py",
        "workers/ditserve/models/wan/prompt_cache.py")]
    if any(api.binding(path) != value for path, value in before.items()):
        raise RuntimeError("input changed while preparing the proposal")
    plan.update(proposal_only=True, gpu_jobs_launched=0, historical_study_modified=False,
                source_freeze_required=True, input_bindings=inputs, development_sources=sources,
                native_configs=[], qualification_configs=[],
                execution_note="Drafts only. Preserve the original ledger and reserve; obtain an explicit "
                "additional allowance if needed, freeze source, qualify numeric gates, and use "
                "the guarded cumulative study launcher. No direct GPU execution is authorized by this plan.")
    output.mkdir(parents=True, exist_ok=False)
    api.atomic_json(output / "environment.json", environment)
    api.atomic_json(output / "qualification-environment.json", qualification_environment)
    plan["environment_bindings"] = [api.binding(output / name) for name in
                                    ("environment.json", "qualification-environment.json")]
    for identifier, config in qualification_configs:
        path = output / (identifier + ".json")
        api.atomic_json(path, config)
        plan["qualification_configs"].append(dict(id=identifier, kind=config["kind"],
            configuration_sha256=api.digest(config), binding=api.binding(path),
            qualification="pending", requires_guarded_launcher=True))
    for identifier, config in configs:
        path = output / (identifier + ".json")
        api.atomic_json(path, config)
        plan["native_configs"].append(dict(id=identifier, configuration_sha256=api.digest(config),
                                          binding=api.binding(path), qualification="pending"))
    api.atomic_json(output / "plan.json", plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Existing study directory, read only")
    parser.add_argument("--output", type=Path, required=True, help="New proposal directory outside the study")
    args = parser.parse_args(argv)
    plan = prepare(args.root, args.output)
    print(json.dumps(dict(proposal=str(args.output.resolve() / "plan.json"),
                          status=plan.get("status"), native_configs=len(plan["native_configs"]),
                          qualification_configs=len(plan["qualification_configs"]),
                          gpu_jobs_launched=0), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
