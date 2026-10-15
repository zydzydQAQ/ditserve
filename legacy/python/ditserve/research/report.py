"""Evidence-only pilot report with separately scoped performance figures."""

import json
import math
from pathlib import Path

from .common import atomic_json, digest, file_digest, read_json


def read_records(output, frozen):
    path = Path(output) / "records.jsonl"
    rows = []
    if not path.exists():
        return rows
    for number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        if (row.get("sha256") != digest({k: v for k, v in row.items() if k != "sha256"})
                or row.get("protocol_sha256") != frozen["sha256"]
                or row.get("source_sha256") != frozen["source_sha256"]):
            raise ValueError("invalid research evidence record at line %d" % number)
        def artifacts(value):
            if isinstance(value, dict):
                if value.get("output_path"):
                    relative = Path(value["output_path"])
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe video artifact path")
                    from .retention import verify_reference
                    verify_reference(output, relative, value.get("output_sha256"))
                for child in value.values():
                    artifacts(child)
            elif isinstance(value, list):
                for child in value:
                    artifacts(child)
        artifacts(row)
        rows.append(row)
    return rows


def read_measurement_environment(output, frozen):
    """Fail closed on missing, damaged, or nonexclusive performance provenance."""
    path = Path(output) / "measurement-environment.json"
    if not path.exists():
        return dict(status="legacy_unmonitored", cost_measurements_valid=False,
                    evidence_status="missing", reason="GPU exclusivity was not monitored for this run")
    try:
        record = read_json(path)
        if not isinstance(record, dict):
            raise ValueError("environment evidence must be an object")
        if (record.get("sha256") != digest({k: v for k, v in record.items() if k != "sha256"})
                or record.get("schema_version") != "ditserve.research.measurement_environment.v1"
                or record.get("protocol_sha256") != frozen["sha256"]
                or record.get("source_sha256") != frozen["source_sha256"]
                or record.get("status") not in ("exclusive_observed", "contended", "unknown")
                or type(record.get("cost_measurements_valid")) is not bool
                or not isinstance(record.get("stages"), dict)):
            raise ValueError("environment schema/hash/source/protocol binding is invalid")
        stages = record["stages"]
        def complete_observation(name, stage):
            log_path = Path(output) / (name + "-gpu-isolation.jsonl")
            return (isinstance(stage, dict) and stage.get("status") == "exclusive_observed"
                    and stage.get("completion_observed") is True and stage.get("worker_returncode") == 0
                    and type(stage.get("observations_count")) is int and stage["observations_count"] >= 2
                    and stage.get("log") == log_path.name and log_path.is_file()
                    and stage.get("log_sha256") == file_digest(log_path))
        exclusive = (bool(stages) and record["status"] == "exclusive_observed"
                     and record["cost_measurements_valid"] is True
                     and all(complete_observation(name, stage) for name, stage in stages.items()))
        return dict(status=record["status"], cost_measurements_valid=exclusive,
                    evidence_status="verified", evidence=record,
                    reason=None if exclusive else "No usable exclusive performance measurement for this run",
                    scope="sampled process ownership; not proof against interference between observations")
    except (OSError, ValueError, TypeError) as error:
        return dict(status="invalid", cost_measurements_valid=False,
                    evidence_status="invalid", reason=str(error))


def quality_cost_points(rows):
    """Keep single-request solver time separate from two-request critical paths."""
    points = dict(calibration=[], joint=[])
    def numeric(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    for row in rows:
        loss = row.get("evidence", {}).get("metrics", {}).get("lpips_mean")
        if not numeric(loss):
            continue
        kind, cost = None, None
        if row.get("type") == "calibration" and row.get("status") == "completed":
            phases = row.get("phase_cost_ms", {})
            if all(numeric(phases.get(name)) for name in ("denoise", "decode")):
                kind, cost = "calibration", phases["denoise"] + phases["decode"]
        elif (row.get("type") == "joint_quality" and row.get("status") in ("completed", "unqualified")
              and row.get("cost_scope") == "full_critical_path"):
            kind, cost = "joint", row.get("cost_ms")
        if kind is not None and numeric(cost) and cost > 0:
            points[kind].append(dict(cost_seconds=cost / 1000, lpips_mean=loss,
                                     split=row.get("split", "unknown"), case_id=row.get("case_id")))
    return points


def make_figures(output, rows, *, environment=None):
    """Missing measurements get a labeled empty panel, never synthetic data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return {"status": "unavailable", "reason": str(error), "paths": []}
    output = Path(output)
    environment = environment or dict(status="legacy_unmonitored", cost_measurements_valid=False)
    performance_valid = environment.get("status") == "exclusive_observed" and environment.get("cost_measurements_valid") is True
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    plots = []
    def save(name, title, draw, observed, timing_scope=None):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        ax.set_title(title)
        usable = bool(observed) and performance_valid
        if usable:
            draw(ax)
        else:
            message = ("No measured GPU evidence\nNot a zero-gain result" if performance_valid else
                       "No usable exclusive performance measurements\nEnvironment: %s\nRaw numerical evidence retained" % environment["status"])
            ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        path = directory / (name + ".svg")
        fig.savefig(path)
        plt.close(fig)
        plots.append({"path": str(path.relative_to(output)), "measured": usable,
                      "raw_measurements_present": bool(observed), "cost_measurements_valid": performance_valid,
                      "environment_status": environment["status"], "timing_scope": timing_scope})

    state_rows = []
    for row in rows:
        if row.get("type") != "probe" or row.get("status") != "completed":
            continue
        for arm in row["arms"]:
            invs = arm["run"].get("inventories", [])
            if not invs:
                continue
            inv = max(invs, key=lambda x: x["inventory"]["cuda_bytes"])["inventory"]
            features = inv["feature_only_cuda_bytes"]
            required = inv["cuda_bytes"] - features
            resident = arm.get("resident_allocated_bytes", 0)
            state_rows.append((str(row["shape"]["frames"]) + "f/" + arm["recipe"]["id"],
                               required, features, resident,
                               max(0, arm["peak_allocated_bytes"] - inv["cuda_bytes"] - resident)))
    def draw_state(ax):
        labels = [r[0] for r in state_rows]
        bottom = [0.] * len(labels)
        for index, label in ((1, "Required request state"), (2, "Feature cache"),
                             (3, "Resident allocation before request"), (4, "Peak remainder: workspace + other")):
            values = [r[index] / 2**30 for r in state_rows]
            ax.bar(labels, values, bottom=bottom, label=label)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_ylabel("GiB; request state sampled at step boundaries")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    save("state-memory", "State / peak allocation decomposition", draw_state, state_rows)

    trials = [r for r in rows if r.get("type") == "state_trial" and r.get("status") == "completed"]
    def draw_cost(ax):
        names = sorted({name for r in trials for name in r["arms"]})
        for name in names:
            indices = [i for i, r in enumerate(trials) if name in r["arms"]]
            ax.plot(indices, [trials[i]["arms"][name]["cost_ms"] / 1000 for i in indices], "o-", label=name)
        ax.set_ylabel("Whole critical path (seconds)")
        ax.set_xlabel("Counterbalanced trial; no interpolation claim")
        ax.legend()
    save("state-crossover", "Preemption action costs", draw_cost, trials, "full_critical_path")

    sp = [r for r in rows if r.get("type") == "sp_trial" and r.get("status") == "completed"]
    def draw_sp(ax):
        for row in sp:
            points = sorted((int(arm["sp_degree"]), arm["cost_ms"]) for arm in row["arms"].values()
                            if "sp_degree" in arm)
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=row["phase"])
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("SP degree")
        ax.set_ylabel("Measured net layout window (ms)")
    save("sp-scaling", "Compute / reuse SP scaling", draw_sp, sp, "net_layout_window")

    scoped_points = quality_cost_points(rows)
    for kind, title, xlabel, scope in (
        ("calibration", "Initial recipe: full-video quality / solver time",
         "32-step denoise + decode (seconds; prepare excluded)", "denoise_plus_decode"),
        ("joint", "Joint policy: full-video quality / interruption cost",
         "Boundary through urgent request and resumed completion (seconds)", "full_critical_path")):
        points = scoped_points[kind]
        def draw_quality(ax):
            for split in sorted({p["split"] for p in points}):
                group = [p for p in points if p["split"] == split]
                ax.scatter([p["cost_seconds"] for p in group], [p["lpips_mean"] for p in group], label=split)
            ax.axhline(.05, linestyle="--", color="gray", label="Pilot mean-LPIPS threshold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Full-video mean LPIPS vs cache off")
            ax.legend(fontsize=8)
        save("quality-time-" + kind, title, draw_quality, points, scope)
    return {"status": "completed", "paths": plots}


def build_report(output, frozen):
    from .evidence import research_gates
    from .runner import _valid_cpu
    from .protocol import source_files
    output = Path(output)
    rows = read_records(output, frozen)
    from .retention import _artifacts, verify_reference
    availability = {}
    for row in rows:
        for item in _artifacts(row, output):
            if item.get("sha256"):
                availability[item["path"]] = verify_reference(output, item["path"], item["sha256"])["status"]
    environment = read_measurement_environment(output, frozen)
    cost_stages = {"probe": "probe", "calibration": "calibrate", "joint_quality": "mechanism",
                   "state_trial": "mechanism", "memory_trial": "mechanism", "sp_trial": "sp"}
    required_observations = {cost_stages[row["type"]] for row in rows if row.get("type") in cost_stages}
    observed_stages = set(environment.get("evidence", {}).get("stages", {}))
    if environment["cost_measurements_valid"] and required_observations - observed_stages:
        environment.update(status="incomplete", cost_measurements_valid=False,
                           reason="Completed isolation observations are missing for measured cost stages",
                           missing_stages=sorted(required_observations - observed_stages))
    performance_rows = rows if environment["cost_measurements_valid"] else []
    gates = research_gates(memory_trials=[r for r in performance_rows if r.get("type") == "memory_trial"],
                           state_trials=[r for r in performance_rows if r.get("type") == "state_trial"],
                           sp_trials=[r for r in performance_rows if r.get("type") == "sp_trial"])
    if not environment["cost_measurements_valid"]:
        for name in ("memory_admission", "state_crossover", "sp_path"):
            gates[name].update(environment_status=environment["status"],
                               blocked_reason="measurement_environment_not_valid_for_performance")
    ledger = read_json(frozen["ledger"])
    preflight = read_json(output / "preflight.json") if (output / "preflight.json").exists() else {}
    figures = make_figures(output, rows, environment=environment)
    generator_files = {name: file_digest(path) for name, path in source_files().items()}
    generator_hash = digest(generator_files)
    result = dict(status="completed", protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"],
                  frozen_source_sha256=frozen["source_sha256"], report_generator_source_sha256=generator_hash,
                  report_generator_source_files=generator_files,
                  report_generator_matches_frozen_source=generator_hash == frozen["source_sha256"],
                  source_scope="Report regeneration never transfers trajectory qualification to a different source",
                  measurement_environment=environment, cost_measurements_valid=environment["cost_measurements_valid"],
                  numerical_correctness=[{key: row[key] for key in ("case_id", "status", "official_equal", "exact_match", "check") if key in row}
                                         for row in rows if row.get("type") == "correctness"],
                  measured_records=len(rows), preflight=preflight,
                  evidence_availability={status: sum(v == status for v in availability.values())
                                         for status in ("raw_available", "retained_metrics_only")},
                  cpu_validation="passed" if _valid_cpu(output / "cpu-validation.json", frozen) else "unverified",
                  gpu_seconds_used=ledger["used_gpu_seconds"],
                  gpu_seconds_remaining=max(0, ledger["limit_gpu_seconds"] - ledger["used_gpu_seconds"]),
                  gates=gates, figures=figures,
                  unresolved=["Only completed same-protocol measurements qualify; partial stages remain unverified",
                              "Four homogeneous GPUs and static numerical validation are required for elastic Wan state handoff",
                              "Online joint controller and LP remain conditional on measured mechanism gates"],
                  interpretation="missing GPU measurements are insufficient evidence, not a negative performance result")
    atomic_json(output / "report.json", result)
    lines = ["# 缓存、抢占与弹性 SP 研究试验", "",
             "CPU 验证：`%s`。GPU 预检：`%s`。" % (result["cpu_validation"], preflight.get("status", "unverified")), "",
             "测量环境：`%s`；成本结论可用：`%s`。资源争用或环境旁证缺失不抹除原始数值正确性记录。" % (
                 environment["status"], result["cost_measurements_valid"]), "",
             "原始证据可重新评价：%d 个文件；仅保留已记录指标、原始数组已清理：%d 个文件。后者不表示新的原始数据复核。" % (
                 result["evidence_availability"]["raw_available"], result["evidence_availability"]["retained_metrics_only"]), "",
             "实验冻结源码：`%s`。报告生成源码：`%s`。两者分别记录，重新生成报告不构成跨源码资格复用。" % (
                 result["frozen_source_sha256"], result["report_generator_source_sha256"]), "",
             "独立账本累计使用 %.3f / %.0f GPU·秒；剩余 %.3f GPU·秒。" % (
                 result["gpu_seconds_used"], ledger["limit_gpu_seconds"], result["gpu_seconds_remaining"]), "",
             "## 研究结论", "", "```json", json.dumps(gates, ensure_ascii=False, indent=2), "```", "",
             "缺测保持证据不足；没有用 CPU 测试、合成张量或空图推断 GPU 收益。LPIPS 是整轨经验过滤器，不能累加前后缀误差或声称绝对画质保证。", "",
             "初始标定图使用完整 32 步去噪与解码耗时；联合动作图使用中断边界起、涵盖紧急请求及恢复完成的关键路径。两图不能合成同一成本 Pareto 前沿，均不表示在线端到端吞吐或 P99。", "",
             "## 验证与资源", "", "- [冻结协议](protocol.json)", "- [CPU 验证](cpu-validation.json)",
             "- [GPU 预检](preflight.json)", "- [实验矩阵](matrix.json)", "",
             "当前执行器涵盖单卡状态与质量试验。原生 Ulysses 后端、组隔离和 storage 重分片已具备 CPU 测试；真实 Wan 跨进程状态交接及弹性性能尚未验证。", "",
             "继续顺序：补齐同协议的单卡正确性、形状探测与标定，再使用四张同型卡验证 SP 数值、请求隔离和重配置成本。各阶段完成情况以本目录证据为准；在线控制器与 LP 分支按机制门槛后置。", "",
             "## 图与数据", ""]
    for item in figures.get("paths", []):
        lines += ["![%s](%s)" % ("有效环境下的实测数据" if item["measured"] else "缺少可用独占测量，不能解读为零收益", item["path"]), ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    return {k: result[k] for k in ("status", "measured_records", "cpu_validation", "gpu_seconds_used", "gpu_seconds_remaining", "gates")}
