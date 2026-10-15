"""Evidence-only report for the finite, heterogeneous two-GPU pre-experiment."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import time

from .common import atomic_json, digest, read_json
from .dual_evidence import read_records
from .dual_matrix import cache_gate, known_quality, migration_quality_gate
from .dual_protocol import verify_protocol
from .retention import _artifacts, idle_guard, verify_reference


def budget_summary(ledger, now=None):
    """Settled charges and a live estimate are reported separately, never reset."""
    settled = float(ledger["used_gpu_seconds"])
    limit = float(ledger["limit_gpu_seconds"])
    active = ledger.get("active")
    live = 0.
    if active:
        count = active.get("gpu_count", len(active.get("devices", [])))
        live = max(0., (time.time() if now is None else now) - float(active["started_at"])) * count
    return dict(limit_gpu_seconds=limit, settled_gpu_seconds=settled,
                active_estimated_gpu_seconds=live, remaining_after_settled_gpu_seconds=max(0., limit - settled),
                estimated_remaining_gpu_seconds=max(0., limit - settled - live),
                active_session_present=bool(active), accounting="active estimate is not a settled charge")


def _decision(status, reason, **details):
    return dict(status=status, reason=reason, **details)


def _finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def evaluate_hypotheses(rows, config):
    """Only sealed complete units may establish a performance conclusion."""
    valid = [row for row in rows if row.get("performance_valid") is True]
    correctness = {r.get("case_id"): r for r in valid if r.get("type") == "correctness"}
    correctness_passed = (all(correctness.get("local-" + role, {}).get("step_equivalent") is True
                              and correctness.get("local-" + role, {}).get("status") == "passed"
                              for role in ("a100", "v100"))
                          and correctness.get("migration", {}).get("status") == "passed"
                          and correctness.get("migration", {}).get("exact_omit_equal") is True)
    cache = {}
    for role in ("a100", "v100"):
        selected = [r for r in valid if r.get("type") == "calibration" and r.get("device_role") == role]
        contexts = {r.get("context_sha256") for r in selected}
        if len(contexts) != 1:
            cache[role] = _decision("insufficient_evidence", "complete same-configuration calibration is missing")
            continue
        gate = cache_gate(selected, config, role, next(iter(contexts)))
        if not gate["eligible"]:
            cache[role] = _decision("falsified" if gate["status"] == "failed" else "insufficient_evidence",
                                     "cache quality failed on the tested development set" if gate["status"] == "failed"
                                     else "development matrix is incomplete", quality_gate=gate)
            continue
        gains = []
        for row in selected:
            baseline = row.get("reference", {}).get("elapsed_ms")
            candidate = row.get("candidate", {}).get("elapsed_ms")
            if _finite_positive(baseline) and _finite_positive(candidate):
                gains.append(1 - candidate / baseline)
        cache[role] = _decision("supported" if len(gains) == gate["required"] and statistics.mean(gains) > 0
                               else "falsified" if len(gains) == gate["required"] else "insufficient_evidence",
            "quality-qualified observed development cost; not a per-request quality guarantee",
            quality_gate=gate, mean_observed_gain=statistics.mean(gains) if gains else None)
    migration_rows = [r for r in valid if r.get("type") == "migration_quality"]
    actions = {r.get("action") for r in migration_rows}
    contexts = {r.get("context_sha256") for r in migration_rows}
    quality = {}
    if len(actions) == len(contexts) == 1:
        for split in ("dev", "holdout"):
            quality[split] = migration_quality_gate([r for r in migration_rows if r.get("split") == split],
                config, split, next(iter(actions)), next(iter(contexts)))
    quality_passed = len(quality) == 2 and all(g["eligible"] for g in quality.values())
    conditions = []
    for condition in ("idle", "busy"):
        pairs = [r for r in valid if r.get("type") == "mechanism" and r.get("condition") == condition]
        detail = dict(condition=condition, status="insufficient_evidence", gains={}, winners=[])
        queue_covered = condition != "busy" or (len(pairs) == 2 and all(r.get("queue_coverage") is True for r in pairs))
        if not queue_covered:
            detail["reason"] = "both busy paired units must record real queue wait covering 1.25 times migration break-even"
        if (queue_covered and len(pairs) == 2 and pairs[0].get("order") == list(reversed(pairs[1].get("order", [])))
                and all(r.get("quality_passed") is True and r.get("all_requests_decoded") is True for r in pairs)
                and len(contexts) == 1 and all(r.get("context_sha256") in contexts for r in pairs)
                and pairs[0].get("queue_length", 0) == pairs[1].get("queue_length", 0)):
            names = set(pairs[0].get("costs_ms", {}))
            complete = (names == set(pairs[1].get("costs_ms", {})) and "stay" in names and "full_migrate" in names
                        and actions.issubset(names) and set(pairs[0]["order"]) == names
                        and all(_finite_positive(r["costs_ms"][name]) for r in pairs for name in names))
            if complete:
                detail["gains"] = {name: [1 - r["costs_ms"][name] / r["costs_ms"]["stay"] for r in pairs]
                                   for name in sorted(names - {"stay"})}
                detail["winners"] = [min(names, key=lambda name: (r["costs_ms"][name], name)) for r in pairs]
                detail["supported_actions"] = [name for name, gains in detail["gains"].items() if min(gains) >= .05]
                detail["status"] = "supported" if detail["supported_actions"] else "falsified"
        conditions.append(detail)
    if not correctness_passed:
        migration = _decision("insufficient_evidence", "same-device step equivalence and migration correctness are not both verified",
                              quality=quality, conditions=conditions)
    elif not quality_passed:
        failed = any(g["status"] == "failed" for g in quality.values())
        migration = _decision("falsified" if failed else "insufficient_evidence",
            "complete trajectory quality failed" if failed else "eight development and four held-out dual-reference units required",
            quality=quality, conditions=conditions)
    elif any(c["status"] == "supported" for c in conditions):
        migration = _decision("supported", "both opposite orders show at least 5% complete critical-path improvement",
            quality=quality, conditions=conditions, scope="ordinary heterogeneous migration engineering evidence")
    elif all(c["status"] == "falsified" for c in conditions):
        migration = _decision("falsified", "tested idle and real-queue conditions did not meet the 5% gate",
                              quality=quality, conditions=conditions)
    else:
        migration = _decision("insufficient_evidence", "paired real-queue mechanism evidence is incomplete",
                              quality=quality, conditions=conditions)
    # A useful migration is insufficient evidence for cache-aware joint control.
    # This pilot has no independently evaluated simple online decision baseline.
    joint = _decision("insufficient_evidence", "no measured improvement over the average-remaining-cost/full-migration decision baseline",
                      claim="no cache-joint-controller innovation claim")
    memory_rows = [r for r in valid if r.get("type") == "capacity" and r.get("status") == "completed"]
    accepted = []
    for row in memory_rows:
        keep = [k for k in memory_rows if k.get("action") == "keep" and k.get("paused_count") == row.get("paused_count")]
        if (row.get("action") in ("offload", "reclaim") and len(keep) == 1
                and keep[0].get("urgent_capacity_failed") is True and row.get("urgent_capacity_failed") is False
                and row.get("quality_passed") is True and row.get("artificially_filled_memory") is False
                and len(row.get("original_requests", [])) == row.get("paused_count") and row.get("urgent")
                and all(r.get("status") == "completed" and known_quality(r.get("quality", {}))
                        for r in row.get("original_requests", []))
                and row["urgent"].get("status") == "completed" and known_quality(row["urgent"].get("quality", {}))
                and any(r.get("type") == "calibration" and r.get("device_role") == "a100"
                        and r.get("context_sha256") == row.get("context_sha256")
                        and r.get("reference", {}).get("status") == "completed" for r in valid)):
            accepted.append(row["case_id"])
    if accepted and correctness_passed:
        memory = _decision("supported", "real keep admission failed; released state admitted and completed all requests",
                           accepted_cases=accepted)
    elif any(r.get("paused_count") == 8 and r.get("action") == "keep"
             and r.get("urgent_capacity_failed") is False and r.get("quality_passed") is True for r in memory_rows):
        memory = _decision("falsified", "up to eight real paused requests still admitted a new request; no capacity benefit observed",
                           scope="tested shapes and hardware only")
    else:
        memory = _decision("insufficient_evidence", "no completed quality-qualified real admission comparison")
    return dict(cache=cache, migration=migration, memory=memory, cache_joint_control=joint,
                correctness_passed=correctness_passed,
                sp=_decision("deferred", "heterogeneous devices run independent SP1; no SP scaling claim"),
                lp=_decision("deferred", "Latent Parallelism is a later branch"))


def _runs(value):
    if isinstance(value, dict):
        if value.get("output_path"):
            yield value
        for child in value.values():
            yield from _runs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _runs(child)


def make_figures(root, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        return dict(status="unavailable", reason=str(error), paths=[])
    root = Path(root)
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    valid = [r for r in rows if r.get("performance_valid") is True]
    figures = []
    def save(name, title, samples, draw, scope):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
        ax.set_title(title)
        if samples:
            draw(ax)
        else:
            ax.text(.5, .5, "Insufficient completed-unit evidence\nMissing data are not zero gain", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        path = directory / (name + ".svg")
        fig.savefig(path)
        plt.close(fig)
        figures.append(dict(path=str(path.relative_to(root)), measured=bool(samples), scope=scope))
    memory = []
    for row in valid:
        if row.get("type") != "probe":
            continue
        for run in _runs(row):
            states = [i.get("state", i.get("inventory", {})) for i in run.get("inventories", [])]
            if not states:
                continue
            inv = max(states, key=lambda s: s.get("cuda_bytes", 0))
            feature = inv.get("feature_only_cuda_bytes", 0)
            required = inv.get("cuda_bytes", 0) - feature
            peak = run.get("peak_memory_bytes", {}).get("allocated", 0)
            memory.append((row["case_id"], required, feature, max(0, peak - required - feature)))
    def draw_memory(ax):
        bottom = [0.] * len(memory)
        for index, label in ((1, "Required state"), (2, "Feature cache"), (3, "Peak remainder: weights/workspace/other")):
            values = [s[index] / 2**30 for s in memory]
            ax.bar([s[0] for s in memory], values, bottom=bottom, label=label)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_ylabel("GiB (peak remainder is not pure workspace)")
        ax.legend(fontsize=7)
    save("state-workspace", "Request state and allocation decomposition", memory, draw_memory, "sampled request state vs allocator peak")
    trials = [r for r in valid if r.get("type") == "mechanism" and r.get("quality_passed") is True]
    def draw_trials(ax):
        for action in sorted({a for r in trials for a in r.get("costs_ms", {})}):
            points = [(i, r["costs_ms"][action] / 1000) for i, r in enumerate(trials) if action in r.get("costs_ms", {})]
            ax.plot([x for x, _ in points], [y for _, y in points], "o-", label=action)
        ax.set_xticks(range(len(trials)), [r["case_id"] for r in trials], rotation=20)
        ax.set_ylabel("Complete critical path (seconds)")
        ax.legend(fontsize=8)
    save("migration-crossover", "Idle / real-queue migration action costs", trials, draw_trials, "wait/transfer/commit/suffix/all decoding and recorded I/O")
    phases = defaultdict(list)
    for row in valid:
        for run in _runs(row):
            role = run.get("device_role", row.get("device_role", "unknown"))
            recipe = run.get("recipe_id", "off")
            for event in run.get("events", []):
                if event.get("kind") == "step" and _finite_positive(event.get("ms")):
                    phases[role + "/" + recipe + "/" + event.get("path", "unknown")].append(event["ms"])
    def draw_phases(ax):
        labels = sorted(phases)
        ax.bar(labels, [statistics.mean(phases[k]) for k in labels])
        ax.set_ylabel("Observed mean step milliseconds")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    save("device-cache-phases", "Device / cache path costs (SP deferred)", phases, draw_phases, "independent SP1; no cross-device SP inference")
    points = []
    for row in valid:
        if row.get("type") not in ("calibration", "migration_quality"):
            continue
        loss = row.get("quality", row.get("canonical_quality", {})).get("lpips_mean")
        cost = row.get("candidate", {}).get("elapsed_ms")
        if isinstance(loss, (int, float)) and math.isfinite(loss) and _finite_positive(cost):
            points.append((row.get("split", "dev"), cost / 1000, loss))
    def draw_quality(ax):
        for split in sorted({x[0] for x in points}):
            group = [x for x in points if x[0] == split]
            ax.scatter([x[1] for x in group], [x[2] for x in group], label=split)
        ax.axhline(.05, linestyle="--", color="gray")
        ax.set_xlabel("Complete trajectory wall time (seconds)")
        ax.set_ylabel("Mean LPIPS vs canonical cache-off reference")
        ax.legend()
    save("quality-time", "Full-trajectory quality and time", points, draw_quality, "empirical reference-relative filter; not absolute quality guarantee")
    return dict(status="completed", paths=figures)


def report(root):
    """Generate only after accounting is idle; no large verification during runs."""
    root = Path(root).absolute()
    preliminary = read_json(root / "protocol.json")
    with idle_guard(root, [preliminary["ledger"]]):
        frozen = verify_protocol(root / "protocol.json", verify_current=False)
        rows = read_records(root, frozen)
        artifacts = {}
        for row in rows:
            for item in _artifacts(row, root):
                if item.get("sha256"):
                    artifacts[(item["path"], item["sha256"])] = verify_reference(root, item["path"], item["sha256"])
        availability = Counter(item["status"] for item in artifacts.values())
        hypotheses = evaluate_hypotheses(rows, frozen["config"])
        ledger = read_json(frozen["ledger"])
        result = dict(schema_version="ditserve.dual.report.v1", status="completed", protocol_sha256=frozen["sha256"],
            source_sha256=frozen["source_sha256"], source_archive=frozen.get("source_archive"),
            measured_records=len(rows), valid_complete_unit_records=sum(r.get("performance_valid") is True for r in rows),
            hypotheses=hypotheses, budget=budget_summary(ledger), figures=make_figures(root, rows),
            evidence_availability=dict(raw_available=availability["raw_available"], retained_metrics_only=availability["retained_metrics_only"],
                interpretation="retained_metrics_only uses recorded metrics; it is not new raw-data re-evaluation"),
            quality_scope="LPIPS <= 0.05 is an empirical full-trajectory filter, not an absolute or per-request quality guarantee",
            performance_scope="finite frozen-text mechanism; not online E2E throughput or statistical significance",
            later_work="online controller only after mechanism gates; SP and LP deferred")
        result["sha256"] = digest(result)
        atomic_json(root / "report.json", result)
        labels = {"supported": "支持", "falsified": "在已测范围内未通过", "insufficient_evidence": "证据不足", "deferred": "后置"}
        lines = ["# 双卡视频 DiT 预实验", "", "每个完整配对单元独立封印。未测和失效单元不作为零收益结果。", "",
                 "| 假设 | 结论 | 依据 |", "|---|---|---|"]
        for name, decision in [("A100 缓存", hypotheses["cache"]["a100"]), ("V100 缓存", hypotheses["cache"]["v100"]),
                               ("异构迁移", hypotheses["migration"]), ("真实容量改善", hypotheses["memory"]),
                               ("缓存联合控制创新", hypotheses["cache_joint_control"])]:
            lines.append("| %s | %s | %s |" % (name, labels[decision["status"]], decision["reason"]))
        budget = result["budget"]
        lines += ["", "累计已结算 %.3f / %.0f GPU·秒；活动会话估算 %.3f GPU·秒（单列，未结算）。" %
                  (budget["settled_gpu_seconds"], budget["limit_gpu_seconds"], budget["active_estimated_gpu_seconds"]), "",
                  "原始证据可重新评价：%d 个文件引用；仅保留已记录指标、原始数组已清理：%d 个文件引用。" %
                  (availability["raw_available"], availability["retained_metrics_only"]), "",
                  "LPIPS 是首轮完整轨迹经验过滤器。机制成本采用冻结文本编码，不能解读为在线 E2E goodput。SP、LP 和在线控制器后置。", ""]
        for figure in result["figures"].get("paths", []):
            lines.append("![%s](%s)" % (Path(figure["path"]).stem, figure["path"]))
        (root / "report.md").write_text("\n".join(lines) + "\n")
        return result
