"""Sealed numerical-diagnostic summaries; never migration qualification."""

from collections import Counter
import math
from pathlib import Path

from .common import atomic_json, digest, read_json
from .dual_evidence import read_records
from .dual_report import budget_summary
from .migration_diagnostic_experiment import ACTIONS, recompute_checks
from .migration_diagnostic_protocol import verify_protocol
from .retention import _artifacts, idle_guard, verify_reference


PAIRS = {"v100_vs_a100": ("v100_stay", "a100_stay"),
         "migration_vs_a100": ("full_migrate", "a100_stay"),
         "migration_vs_v100": ("full_migrate", "v100_stay")}
ERRORS = ("cond", "uncond", "cfg", "latent_before", "latent_after")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _quality(metrics, candidate, reference):
    frames = metrics.get("lpips_per_frame", [])
    return (metrics.get("status") == "ok" and metrics.get("lpips_status") == "ok"
        and metrics.get("finite") is True and metrics.get("range_valid") is True
        and metrics.get("frame_count") == 17 and len(frames) == 17 and all(_number(x) for x in frames)
        and all(_number(metrics.get(k)) for k in ("lpips_mean", "lpips_p95", "lpips_max", "temporal_delta_rmse"))
        and metrics.get("candidate_output_sha256") == candidate.get("output_sha256")
        and metrics.get("reference_output_sha256") == reference.get("output_sha256")
        and metrics.get("candidate_identity_sha256") == digest(candidate.get("identity"))
        and metrics.get("reference_identity_sha256") == digest(reference.get("identity")))


def _audit_row(row, diagnostic):
    issues = []
    try:
        runs, shadow = row["runs"], row["shadow"]
        computed = recompute_checks(runs, shadow)
        if row.get("checks") != computed or row.get("checks_passed") is not all(computed.values()):
            issues.append("recorded correctness flags differ from complete state evidence")
        if not all(value is True for value in computed.values()):
            issues.append("local restore, migration prefix or shadow neutrality failed")
        if (set(runs) != set(ACTIONS) or row.get("recipe_id") != "off"
                or row.get("status") != "completed" or row.get("timing_eligible") is not False
                or row.get("performance_qualified") is not False or row.get("quality_qualified") is not False):
            issues.append("record is outside the cache-off diagnostic-only contract")
        for action, run in runs.items():
            identity = run.get("identity", {})
            role = "v100" if action == "full_migrate" or action.startswith("v100") else "a100"
            route = ["v100", "a100"] if action == "full_migrate" else [role]
            backends = diagnostic["attention_by_comparison"][row["comparison"]]
            if (identity.get("recipe") != "off" or identity.get("action") != action
                    or identity.get("implementation") != "native" or identity.get("boundary") != 16
                    or identity.get("path") != route or identity.get("attention_path") != [backends[r] for r in route]
                    or identity.get("context_sha256") != row.get("context_sha256")
                    or identity.get("input", {}).get("seed") != row["seed"]
                    or identity.get("input", {}).get("prompt_id") != row["prompt_id"]
                    or len(run.get("traces", [])) != 32 or run.get("boundary_equal") is not True
                    or run.get("boundary_signature") != run.get("signatures", {}).get("16")):
                issues.append("trajectory identity, complete trace or pre-compute boundary proof is missing: " + action)
            for step in (1, 16, 17, 32):
                signature = run.get("signatures", {}).get(str(step), {})
                if signature.get("step_index") != step or not signature.get("state_sha256"):
                    issues.append("native step signature is missing: %s/%s" % (action, step))
        expected = diagnostic["attention_by_comparison"][row["comparison"]]
        source, target = shadow["source"], shadow["target"]
        if (source.get("backend") != expected["v100"] or target.get("backend") != expected["a100"]
                or not source.get("target_device") or not target.get("target_device")
                or source["target_device"] == target["target_device"]
                or source.get("snapshot_sha256") != shadow.get("snapshot_sha256")
                or target.get("snapshot_sha256") != shadow.get("snapshot_sha256")
                or shadow.get("step") != 17):
            issues.append("shadow result lacks distinct device/backend/snapshot binding")
        errors = shadow.get("errors", {})
        if set(errors) != set(ERRORS) or any(not _number(errors[k].get(field))
                for k in ERRORS for field in ("relative_l2", "max_abs")):
            issues.append("same-input step-17 prediction or latent error is missing")
        for name, (candidate, reference) in PAIRS.items():
            if not _quality(row.get("qualities", {}).get(name, {}), runs[candidate], runs[reference]):
                issues.append("complete paired video quality is missing or unbound: " + name)
    except (KeyError, TypeError, ValueError, AttributeError):
        issues.append("incomplete or malformed numerical control record")
        computed = {}
    return computed, sorted(set(issues))


def analyze_rows(rows, diagnostic):
    """Recompute finite matrix completeness and numerical evidence validity."""
    expected = {(comparison, case["prompt_id"], case["seed"], order)
                for comparison in diagnostic["comparisons"] for case in diagnostic["cases"]
                for order in diagnostic["repetitions"]}
    selected, invalid, excluded = {}, [], []
    for row in rows:
        if row.get("type") != "migration_diagnostic":
            continue
        key = (row.get("comparison"), row.get("prompt_id"), row.get("seed"), row.get("order"))
        if row.get("performance_valid") is not True:
            excluded.append(dict(case_id=row.get("case_id"), reason="unit incomplete or isolation-invalidated"))
            continue
        if key not in expected or key in selected:
            invalid.append(dict(case_id=row.get("case_id"), reasons=["unknown or duplicate diagnostic matrix cell"]))
            continue
        checks, issues = _audit_row(row, diagnostic)
        selected[key] = row
        if issues:
            invalid.append(dict(case_id=row.get("case_id"), comparison=row.get("comparison"), reasons=issues))
    phases, samples, repeatability = {}, [], []
    for comparison in diagnostic["comparisons"]:
        group = [row for key, row in selected.items() if key[0] == comparison]
        contexts = {row.get("context_sha256") for row in group}
        if len(contexts) > 1 or group and None in contexts:
            invalid.append(dict(comparison=comparison, reasons=["multiple or missing execution contexts within a backend comparison"]))
        bad = any(item.get("comparison") in (None, comparison) for item in invalid)
        missing = [dict(comparison=c, prompt_id=p, seed=s, order=o)
                   for c, p, s, o in sorted(expected - selected.keys()) if c == comparison]
        phases[comparison] = dict(status="diagnostic_invalid" if bad else "completed" if not missing
            else "partial" if group else "unverified", completed_units=len(group), expected_units=4, missing=missing)
        for row in group:
            samples.append(dict(comparison=comparison, prompt_id=row["prompt_id"], seed=row["seed"], order=row["order"],
                row_sha256=row["sha256"], checks=row.get("checks", {}),
                final_quality=row.get("qualities", {}), step17_errors=row.get("shadow", {}).get("errors", {}),
                trajectory_elapsed_ms={action: run.get("elapsed_ms") for action, run in row.get("runs", {}).items()},
                numerical_attribution_valid=not bad))
        for case in diagnostic["cases"]:
            left = selected.get((comparison, case["prompt_id"], case["seed"], "forward"))
            right = selected.get((comparison, case["prompt_id"], case["seed"], "reverse"))
            if left is None or right is None:
                continue
            values = {}
            for action in ACTIONS:
                a, b = left.get("runs", {}).get(action, {}), right.get("runs", {}).get(action, {})
                values[action] = dict(output_equal=bool(a.get("output_sha256")) and a.get("output_sha256") == b.get("output_sha256"),
                    latent_equal=bool(a.get("latent_sha256")) and a.get("latent_sha256") == b.get("latent_sha256"))
            repeatability.append(dict(comparison=comparison, seed=case["seed"], actions=values,
                                     all_hashes_equal=all(all(v.values()) for v in values.values())))
    crossover = []
    for case in diagnostic["cases"]:
        for order in diagnostic["repetitions"]:
            values = [selected.get((comparison, case["prompt_id"], case["seed"], order)) for comparison in ("original", "math")]
            if any(value is None for value in values):
                continue
            original, math_row = values
            deltas = {}
            for name in PAIRS:
                a = original.get("qualities", {}).get(name, {}).get("lpips_mean")
                b = math_row.get("qualities", {}).get(name, {}).get("lpips_mean")
                deltas[name] = dict(original=a, math=b, original_minus_math=a - b if _number(a) and _number(b) else None)
            error_deltas = {}
            for name in ERRORS:
                a = original.get("shadow", {}).get("errors", {}).get(name, {}).get("relative_l2")
                b = math_row.get("shadow", {}).get("errors", {}).get(name, {}).get("relative_l2")
                error_deltas[name] = dict(original=a, math=b, original_minus_math=a - b if _number(a) and _number(b) else None)
            crossover.append(dict(seed=case["seed"], order=order, lpips=deltas, step17_relative_l2=error_deltas,
                scope="descriptive backend-configuration association; not identification of a pure backend or hardware cause"))
    failures = [selected.get(("original", "kettle", 102, order)) for order in diagnostic["repetitions"]]
    losses = [row.get("qualities", {}).get("migration_vs_a100", {}).get("lpips_mean") if row else None for row in failures]
    if invalid:
        status, reproduction = "diagnostic_invalid", "unverified_due_to_invalid_controls"
        next_step = "先修复状态恢复、别名或影子计算一致性问题，再使用新的冻结运行重复诊断；暂不归因于硬件或后端。"
    elif selected.keys() == expected:
        status = "completed"
        next_step = "根据同源第17步预测误差与完整轨迹结果选择有限的数值稳定性改动；新实现须重新冻结并完整验证，不能恢复旧迁移资格。"
    else:
        status = "insufficient_evidence"
        next_step = "在 migration 原阶段剩余预算允许时补齐完整诊断单元；不足则保留缺测，不扩展参数搜索或迁移机制。"
    if not invalid:
        reproduction = ("unverified" if not all(_number(v) for v in losses)
                        else "reproduced_in_both_orders" if all(v > diagnostic["epsilon"] for v in losses)
                        else "not_reproduced" if all(v <= diagnostic["epsilon"] for v in losses)
                        else "order_dependent")
        if status == "completed" and any(not value["all_hashes_equal"] for value in repeatability):
            next_step = "先定位同后端、同输入的重复执行漂移，再解释设备或后端差异；不能把重复性问题归因于迁移。"
    return dict(status=status, phases=phases, samples=sorted(samples, key=lambda x: (x["comparison"], x["seed"], x["order"])),
        invalid_controls=invalid, excluded_units=excluded, completed_units=len(selected), expected_units=len(expected),
        repeatability=repeatability, backend_comparison=crossover,
        failure_reproduction=dict(status=reproduction, case_id="dev-kettle-102", original_lpips_by_order=dict(zip(diagnostic["repetitions"], losses))),
        next_step=next_step, migration_qualified=False, performance_qualified=False, quality_qualified=False,
        attribution="MATH 同时改变两卡执行路径；误差改善仅为配置对照，不能单独区分纯后端、硬件及采样器误差放大。")


def make_figures(root, analysis):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return dict(status="unavailable", reason=str(error), paths=[])
    directory = Path(root) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    samples, paths = analysis["samples"], []
    labels = ["%s/%s/%s" % (r["comparison"], r["seed"], r["order"]) for r in samples]

    def save(name, title, draw):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.set_title(title)
        if samples:
            draw(ax)
            ax.set_xticks(range(len(samples)), labels, rotation=25, ha="right", fontsize=8)
        else:
            ax.text(.5, .5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        path = directory / (name + ".svg")
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)
        paths.append(dict(path=str(path.relative_to(root)), title=title))

    def checks(ax):
        ax.bar(range(len(samples)), [sum(v is True for v in r["checks"].values()) for r in samples])
        ax.set_ylim(0, 7.5); ax.set_ylabel("Passed state/neutrality checks (7 required)")

    def errors(ax):
        for name in ("cond", "uncond", "cfg", "latent_after"):
            values = [r["step17_errors"].get(name, {}).get("relative_l2") for r in samples]
            values = [value if _number(value) else float("nan") for value in values]
            ax.plot(range(len(samples)), values, "o-", label=name)
        ax.set_ylabel("Relative L2, identical step-16 input"); ax.legend(fontsize=8)

    def quality(ax):
        for name in PAIRS:
            values = [r["final_quality"].get(name, {}).get("lpips_mean") for r in samples]
            values = [value if _number(value) else float("nan") for value in values]
            ax.plot(range(len(samples)), values, "o-", label=name)
        ax.axhline(.05, color="gray", linestyle="--")
        ax.set_ylabel("Full-video mean LPIPS (empirical 0.05 filter)"); ax.legend(fontsize=8)

    def cost(ax):
        for action in ACTIONS:
            values = [r["trajectory_elapsed_ms"].get(action) for r in samples]
            ax.plot(range(len(samples)), [v / 1000 if _number(v) else float("nan") for v in values], "o-", label=action)
        ax.set_ylabel("Instrumented trajectory seconds; not performance"); ax.legend(fontsize=7)

    for args in (("state-checks", "State restoration and shadow neutrality", checks),
                 ("step17-errors", "Same-source prefix: first post-boundary prediction errors", errors),
                 ("final-quality", "Cache-off video quality across device paths", quality),
                 ("diagnostic-cost", "Diagnostic execution cost including instrumentation", cost)):
        save(*args)
    return dict(status="completed", paths=paths)


def report(root):
    root = Path(root).absolute()
    preliminary = read_json(root / "protocol.json")
    with idle_guard(root, [preliminary["ledger"]]):
        frozen = verify_protocol(root / "protocol.json", verify_current=False)
        rows = read_records(root, frozen)
        analysis = analyze_rows(rows, frozen["diagnostic"])
        references = {}
        for row in rows:
            for item in _artifacts(row, root):
                if item.get("sha256"):
                    references[(item["path"], item["sha256"])] = verify_reference(root, item["path"], item["sha256"])
        counts = Counter(item["status"] for item in references.values())
        ledger = read_json(frozen["ledger"])
        budget = budget_summary(ledger)
        budget.update(run_gpu_seconds=sum(s.get("gpu_seconds", 0.) for s in ledger.get("sessions", [])
            if s.get("protocol_sha256") == frozen["sha256"]), budget_stage="migration",
            remaining_migration_gpu_seconds=max(0., ledger["stage_limits_gpu_seconds"]["migration"]
                - ledger["stage_used_gpu_seconds"]["migration"]))
        phases = {}
        for name in ("preflight", "original", "math", "probe", "calibrate", "mechanism"):
            path = root / (name + ".json")
            if path.exists():
                value = read_json(path)
                phases[name] = dict(status=value.get("status"), reason=value.get("reason"))
        result = dict(schema_version="ditserve.research.migration_diagnostic.report.v1", status=analysis["status"],
            protocol_sha256=frozen["sha256"], source_sha256=frozen["source_sha256"], source_archive=frozen["source_archive"],
            analysis=analysis, budget=budget, phase_records=phases, measured_records=len(rows),
            figures=make_figures(root, analysis), parent_failure=frozen["parent_failure"],
            evidence_availability=dict(raw_available=counts["raw_available"], retained_metrics_only=counts["retained_metrics_only"],
                interpretation="declared deletions retain recorded metrics only; no new raw-array quality evaluation"),
            migration_qualified=False, performance_qualified=False, quality_qualified=False,
            quality_scope="LPIPS <= 0.05 is a pre-experiment empirical filter, not absolute quality or statistical significance",
            timing_scope="all observations are instrumented numerical diagnostics, not serving performance qualification")
        result["sha256"] = digest(result)
        atomic_json(root / "report.json", result)
        lines = ["# 无缓存异构迁移数值诊断", "", "状态：`%s`；完成 %d / %d 个完整诊断单元。" %
            (analysis["status"], analysis["completed_units"], analysis["expected_units"]), "",
            "本报告只诊断 kettle/101 与 kettle/102 的无缓存路径。任何本地恢复或影子一致性失败均使诊断无效；完整性不足保留未验证。", "",
            "未授予迁移、性能或质量资格。LPIPS≤0.05 仅是预实验经验过滤条件，不是绝对质量保证或统计显著性结论。", "",
            "| 后端对照 | 状态 | 已完成 / 需要 |", "|---|---|---:|"]
        for name, value in analysis["phases"].items():
            lines.append("| %s | %s | %d / %d |" % (name, value["status"], value["completed_units"], value["expected_units"]))
        lines += ["", "原后端 kettle/102 失败复现：`%s`。" % analysis["failure_reproduction"]["status"], "",
            "| 对照 / 种子 / 顺序 | V100 对 A100 | 迁移对 A100 | 迁移对 V100 | 第17步 CFG 相对 L2 |",
            "|---|---:|---:|---:|---:|"]
        def number(value):
            return "未测" if not _number(value) else "%.7f" % value
        for sample in analysis["samples"]:
            values = [number(sample["final_quality"].get(name, {}).get("lpips_mean")) for name in PAIRS]
            values.append(number(sample["step17_errors"].get("cfg", {}).get("relative_l2")))
            lines.append("| %s/%s/%s | %s |" % (sample["comparison"], sample["seed"], sample["order"], " | ".join(values)))
        lines += ["", analysis["attribution"], "",
            "正反序重复的最终 latent/video hash 全部相同：%d / %d 个输入对照。不同 hash 会如实保留，不自动归因于迁移。" %
            (sum(r["all_hashes_equal"] for r in analysis["repeatability"]), len(analysis["repeatability"])), "",
            "累计结算 %.3f / %.0f GPU·秒；本运行 %.3f GPU·秒；migration 阶段剩余 %.3f GPU·秒。" %
            (budget["settled_gpu_seconds"], budget["limit_gpu_seconds"], budget["run_gpu_seconds"], budget["remaining_migration_gpu_seconds"]), "",
            "原始证据可重新评价：%d 个引用；仅保留已记录指标：%d 个引用。未声明缺失仍报错；本报告没有重新计算 LPIPS。" %
            (counts["raw_available"], counts["retained_metrics_only"]), "", "下一步：" + analysis["next_step"], ""]
        for item in analysis["invalid_controls"]:
            lines.append("- %s：%s" % (item.get("case_id", item.get("comparison", "未知单元")), "; ".join(item["reasons"])))
        for item in result["figures"].get("paths", []):
            lines += ["", "![%s](%s)" % (item["title"], item["path"])]
        (root / "report.md").write_text("\n".join(lines) + "\n")
        return result
