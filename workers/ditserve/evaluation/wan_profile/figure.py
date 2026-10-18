"""One auditable nine-panel Wan figure, using preserved scalar evidence only.

No GPU work or profiling is performed here. Missing/unverified quantities stay
null. In particular, no historical profiler active-rate or inclusive annotation
attribution is used. New hardware values must come from nsys_metrics summaries.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..common import file_hash as _hash


_FRAMES = (17, 33, 65)
_HARDWARE_STUDY_UNSET = object()
_COLORS = ("#2563a6", "#d78028", "#2f987b", "#8f79ad")
_PANELS = {
    "e2e": "旧正式样本 E2E 均值，误差线=min/max；每帧数三个不同prompt/seed，非重复计时置信区间",
    "stages": "旧独立计时诊断；CPU文本编码=host span，DiT/VAE=CUDA event累计区间，非纯kernel时间",
    "memory": "旧独立显存诊断；PyTorch allocated绝对峰值，含常驻GPU权重，阶段间不可相加",
    "named_kernels": "旧复核保留的真实Flash和名称匹配GEMM kernel duration之和；Flash含self/cross，GEMM不等于FFN",
    "activity": "新真实kernel时间区间并集/完整请求wall，以及并集/首末真实GPU活动窗口；不是SM利用率",
    "shares": "新仅Transformer真实kernel duration之和为分母；互斥叶子归因，不是wall-time占比",
    "sm": "Nsight Systems设备级SM Active采样，按NVTX区间时间加权；不是计算吞吐率；denoise NVTX不保证包含末尾GPU尾部",
    "tensor": "Nsight Systems设备级Tensor Active采样，按NVTX区间时间加权；不是达到的理论FLOP比例；denoise NVTX不保证包含末尾GPU尾部",
    "dram": "Nsight Systems DRAM读/写接口活跃周期比例采样；不是实测GB/s或理论峰值带宽比例",
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _document(value, label):
    if value is None:
        return {}, {"id": label, "status": "not_provided", "path": None}, ""
    if isinstance(value, dict):
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
        return value, dict(id=label, status="provided_mapping", path=None,
                           sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded)), ""
    path = Path(value).resolve()
    if not path.exists():
        return {}, {"id": label, "status": "missing", "path": str(path)}, ""
    if path.is_dir():
        if (path / "results.xlsx").is_file():
            from ...results import load
            before = _hash(path / "results.xlsx")
            data = load(path)
            source = path / "results.xlsx"
            if _hash(source) != before:
                raise RuntimeError("archive changed while preparing figure: " + str(source))
            return data, dict(id=label, status="loaded", path=str(source), bytes=source.stat().st_size,
                              sha256=before), ""
        else:
            source = path / "run.json"
            if not source.is_file():
                return {}, {"id": label, "status": "missing", "path": str(source)}, ""
    else:
        source = path
    raw = source.read_bytes()
    data = json.loads(raw)
    return data, dict(id=label, status="loaded", path=str(source), bytes=len(raw),
                      sha256=hashlib.sha256(raw).hexdigest()), ""


def _rank_results(study, prefix=""):
    if isinstance(study.get("run"), dict):
        study, prefix = study["run"], prefix + "/run"
    for p, phase in enumerate(study.get("phases", [])):
        for c, cell in enumerate(phase.get("cells", [])):
            for o, operation in enumerate(cell.get("operations", [])):
                for r, rank in enumerate(operation.get("ranks", [])):
                    if isinstance(rank, dict):
                        yield rank, "%s/phases/%d/cells/%d/operations/%d/ranks/%d" % (prefix, p, c, o, r), phase


def _canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hardware_binding(study, capture, source):
    """Bind an independently collected hardware pass to its actual frozen run."""
    run = study.get("run", study)
    reasons = []
    identity = {key: run.get(key) for key in ("run_id", "experiment", "config_sha256", "status")}
    identity["source_sha256"] = _canonical_digest(run["source"]) if isinstance(run.get("source"), dict) else None
    if run.get("schema") != "ditserve.run.v1":
        reasons.append("hardware_study_schema_missing_or_invalid")
    if run.get("status") not in ("completed", "incomplete"):
        reasons.append("hardware_study_not_settled_or_failed")
    if not all(isinstance(run.get(key), str) and run[key] for key in ("run_id", "experiment")):
        reasons.append("hardware_study_identity_missing")
    if not isinstance(run.get("config"), dict) or run.get("config_sha256") != _canonical_digest(run["config"]):
        reasons.append("hardware_study_config_hash_mismatch")
    if not isinstance(run.get("source"), dict) or not run["source"]:
        reasons.append("hardware_study_source_missing")
    if capture.get("schema") != "ditserve.wan_hardware_capture.v1":
        reasons.append("independent_hardware_requires_bound_capture_wrapper")
    preflight = capture.get("preflight", {})
    if not isinstance(preflight, dict):
        preflight = {}
    if preflight.get("ready") is not True:
        reasons.append("hardware_capture_preflight_unverified")
    for key in ("config_sha256", "source_sha256"):
        if not identity.get(key) or preflight.get(key) != identity[key]:
            reasons.append("hardware_capture_" + key + "_mismatch")
    # Older capture wrappers have work_dir but no explicit run_id.  Its final
    # two path components carry the same experiment/run identity.  A declared
    # run_id/experiment is additionally checked, never allowed to contradict it.
    work = preflight.get("work_dir")
    if not isinstance(work, str) or not work:
        reasons.append("hardware_capture_work_dir_missing")
    else:
        work_path = Path(work).resolve()
        if work_path.name != run.get("run_id") or work_path.parent.name != run.get("experiment"):
            reasons.append("hardware_capture_run_identity_mismatch")
        source_path = source.get("path")
        if source_path and Path(source_path).name == "run.json" and Path(source_path).parent != work_path:
            reasons.append("hardware_capture_run_path_mismatch")
    for container in (preflight, capture):
        for key in ("run_id", "experiment", "config_sha256", "source_sha256"):
            if key in container and container[key] != identity.get(key):
                reason = "hardware_capture_" + key + "_mismatch"
                if reason not in reasons:
                    reasons.append(reason)
    return dict(status="verified" if not reasons else "unverified", validation_source="hardware_study",
                identity=identity, reason_codes=reasons)


def build_analysis(legacy_run, legacy_analysis, study=None, hardware=None,
                   hardware_study=_HARDWARE_STUDY_UNSET):
    """Return bounded, JSON-safe plot values, provenance, definitions and gaps."""
    documents, sources = {}, []
    independent_hardware = hardware_study is not _HARDWARE_STUDY_UNSET
    for label, value in (("legacy_run", legacy_run), ("legacy_analysis", legacy_analysis),
                         ("study", study), ("hardware", hardware),
                         ("hardware_study", hardware_study if independent_hardware else None)):
        document, source, unused = _document(value, label)
        documents[label] = document
        sources.append(source)
    rows, warnings = [], []

    def add(panel, series, frames, value, unit, n, source, pointers, **extra):
        good = _finite(value)
        row = dict(panel=panel, series=series, frames=frames, value=float(value) if good else None,
                   unit=unit, n=n, status="measured" if good else "unmeasured", source=source,
                   source_pointers=pointers, definition=_PANELS[panel])
        row.update(extra)
        rows.append(row)

    grouped = defaultdict(list)
    for index, sample in enumerate(documents["legacy_run"].get("samples", [])):
        if (sample.get("mode") in ("baseline", "clean") and not sample.get("error")
                and sample.get("status") not in ("failed", "error") and _finite(sample.get("e2e_ms"))):
            grouped[sample.get("frames", sample.get("spec", {}).get("frames"))].append((index, sample))
    for frames in _FRAMES:
        samples = grouped[frames]
        values = [sample["e2e_ms"] / 1000 for unused, sample in samples]
        add("e2e", "E2E", frames, statistics.mean(values) if values else None, "s", len(values),
            "legacy_run", ["/samples/%d/e2e_ms" % index for index, unused in samples],
            minimum=min(values) if values else None, maximum=max(values) if values else None)

    analysis = documents["legacy_analysis"]
    for panel, field, wanted, key, factor, unit in (
            ("stages", "stages", ("text_encode", "transformer", "vae_decode"), "duration_ms", 1000., "s"),
            ("memory", "memory", ("transformer", "vae_decode"), "peak_allocated_bytes", 1024.**3, "GiB")):
        groups = defaultdict(list)
        for index, row in enumerate(analysis.get(field, [])):
            if row.get("stage") in wanted and _finite(row.get(key)):
                groups[(row.get("frames"), row["stage"])].append((index, row))
        for frames in _FRAMES:
            for stage in wanted:
                evidence = groups[(frames, stage)]
                values = [row[key] / factor for unused, row in evidence]
                value = (max(values) if panel == "memory" else statistics.mean(values)) if values else None
                add(panel, stage, frames, value, unit, len(values), "legacy_analysis",
                    ["/%s/%d/%s" % (field, index, key) for index, unused in evidence],
                    source_fields=[row.get("source_field", key) for unused, row in evidence])
    scaling = analysis.get("summary", {}).get("scaling", {})
    for frames in _FRAMES:
        for series, key in (("Flash Attention", "flash_ms"), ("Named GEMM", "named_gemm_ms")):
            value = scaling.get(key, {}).get(str(frames))
            add("named_kernels", series, frames, value / 1000 if _finite(value) else None, "s",
                1 if _finite(value) else 0, "legacy_analysis", ["/summary/scaling/%s/%s" % (key, frames)])

    new_groups = defaultdict(list)
    hardware_validations = defaultdict(list)
    for rank, pointer, phase in _rank_results(documents["study"]):
        phase_valid = (phase.get("status") == "completed" and phase.get("qualification", {}).get("qualified") is True
                       and not phase.get("error") and not phase.get("final_occupancy", {}).get("foreign"))
        if rank.get("mode") != "trace":
            continue
        frames, profiler = rank.get("spec", {}).get("frames"), rank.get("profiler", {})
        validated = phase_valid and rank.get("numerical_validation", {}).get("passed") is True and not rank.get("error")
        timeline_ok = validated and profiler.get("timeline_status") == "measured"
        attributed = (validated and profiler.get("attribution_status") == "measured"
                      and rank.get("instrumentation_validation", {}).get("passed") is True)
        for key in ("request", "gpu_window"):
            value = profiler.get(key, {}).get("kernel_active_pct") if timeline_ok else None
            new_groups[("activity", key, frames)].append((value,
                pointer + "/profiler/%s/kernel_active_pct" % key, timeline_ok))
        categories = {row.get("name"): (index, row) for index, row in
                      enumerate(profiler.get("transformer_kernel_categories", []))}
        leaves = ("self_attention", "cross_attention", "ffn")
        denominator = profiler.get("transformer_kernel_duration_ms")
        attributed = attributed and _finite(denominator) and denominator > 0
        attributed = attributed and all(_finite(row.get("share_pct")) for unused, row in categories.values())
        attributed = attributed and abs(math.fsum(row["share_pct"] for unused, row in categories.values()) - 100.) < 1e-5
        for category in (*leaves, "other"):
            selected = [(index, row) for name, (index, row) in categories.items()
                        if (name == category if category != "other" else name not in leaves)]
            valid = attributed and bool(selected) and all(_finite(row.get("share_pct")) for unused, row in selected)
            value = math.fsum(row["share_pct"] for unused, row in selected) if valid else None
            pointers = [pointer + "/profiler/transformer_kernel_categories/%d/share_pct" % index
                        for index, unused in selected]
            new_groups[("shares", category, frames)].append((value, pointers, valid))
    for panel, series in (("activity", ("request", "gpu_window")),
                          ("shares", ("self_attention", "cross_attention", "ffn", "other"))):
        for frames in _FRAMES:
            for name in series:
                samples = new_groups[(panel, name, frames)]
                valid = [sample for sample in samples if sample[2] and _finite(sample[0])]
                pointers = [pointer for unused, pointer, good in valid]
                pointers = [value for pointer in pointers for value in (pointer if isinstance(pointer, list) else [pointer])]
                add(panel, name, frames, statistics.mean(sample[0] for sample in valid) if valid else None,
                    "%", len(valid), "study", pointers, excluded_samples=len(samples) - len(valid))
                if len(valid) != len(samples):
                    warnings.append("%s/%s/%s excluded %d unverified trace samples" %
                                    (panel, name, frames, len(samples) - len(valid)))

    validation_source = "hardware_study" if independent_hardware else "study"
    validation_study = documents[validation_source]
    hardware_binding = (_hardware_binding(validation_study, documents["hardware"],
                        next(source for source in sources if source["id"] == "hardware_study"))
                        if independent_hardware else dict(status="shared_study_validation", validation_source="study",
                                                         reason_codes=[]))
    if hardware_binding["reason_codes"]:
        warnings.extend(hardware_binding["reason_codes"])
    clean_hashes = defaultdict(set)
    if independent_hardware:
        for rank, pointer, phase in _rank_results(validation_study):
            if (phase.get("id") == "hardware_metrics" and phase.get("status") == "completed"
                    and rank.get("mode") == "clean" and rank.get("status") == "ok"
                    and rank.get("output", {}).get("finite") is True):
                clean_hashes[_canonical_digest(rank.get("spec", {}))].add(rank.get("output", {}).get("sha256"))
    for rank, pointer, phase in _rank_results(validation_study):
        if rank.get("mode") != "hardware":
            continue
        occupancy = phase.get("final_occupancy", {})
        phase_valid = (phase.get("status") == "completed" and phase.get("qualification", {}).get("qualified") is True
                       and not phase.get("error") and occupancy.get("status") == "exclusive_observed"
                       and not occupancy.get("foreign") and not occupancy.get("owned"))
        numeric = rank.get("numerical_validation", {})
        valid = phase_valid and rank.get("status") == "ok" and numeric.get("passed") is True and not rank.get("error")
        if independent_hardware:
            output = rank.get("output", {})
            output_hash = output.get("sha256")
            valid = (valid and hardware_binding["status"] == "verified" and phase.get("id") == "hardware_metrics"
                     and occupancy.get("foreign") == [] and occupancy.get("owned") == []
                     and output.get("finite") is True and isinstance(output_hash, str) and len(output_hash) == 64
                     and all(char in "0123456789abcdef" for char in output_hash)
                     and numeric.get("reference_sha256") == output_hash
                     and clean_hashes[_canonical_digest(rank.get("spec", {}))] == {output_hash})
        if valid:
            hardware_validations[rank.get("spec", {}).get("frames")].append(pointer)

    hardware_groups = defaultdict(list)
    hardware_document = documents["hardware"]
    hardware_prefix, hardware_capture_valid = "", True
    if hardware_document.get("schema") == "ditserve.wan_hardware_capture.v1":
        hardware_capture_valid = (hardware_document.get("status") == "sampled"
                                  and hardware_document.get("measured") is True
                                  and hardware_document.get("hardware_phase_qualified") is True
                                  and not hardware_document.get("failure"))
        if not hardware_capture_valid:
            warnings.append("hardware_capture_wrapper_unverified; nested metrics cannot override capture failure")
        hardware_document = hardware_document.get("metrics", {})
        hardware_prefix = "/metrics"
    for r, interval in enumerate(hardware_document.get("ranges", [])):
        for m, metric in enumerate(interval.get("metrics", [])):
            hardware_groups[(interval.get("frames"), interval.get("scope"), metric.get("id"))].append(
                (metric, hardware_prefix + "/ranges/%d/metrics/%d/time_weighted_mean_pct" % (r, m),
                 hardware_capture_valid and interval.get("valid", True)))
    for panel, metric_ids in (("sm", ("sm_active_pct",)), ("tensor", ("tensor_active_pct",)),
                             ("dram", ("dram_read_active_pct", "dram_write_active_pct"))):
        for frames in _FRAMES:
            for scope in ("request", "denoise"):
                for metric_id in metric_ids:
                    samples = hardware_groups[(frames, scope, metric_id)]
                    valid = [(metric, pointer) for metric, pointer, range_valid in samples if range_valid
                             and hardware_validations[frames]
                             and metric.get("measured") is True and metric.get("status") == "sampled"
                             and _finite(metric.get("time_weighted_mean_pct"))]
                    add(panel, scope + "/" + metric_id, frames,
                        statistics.mean(metric["time_weighted_mean_pct"] for metric, unused in valid) if valid else None,
                        "%", len(valid), "hardware", [pointer for unused, pointer in valid],
                        excluded_samples=len(samples) - len(valid),
                        coverage_fractions=[metric.get("coverage_fraction") for metric, unused in valid],
                        sample_counts=[metric.get("sample_count") for metric, unused in valid],
                        validation_source=validation_source, validation_source_pointers=hardware_validations[frames],
                        validation_binding_status=hardware_binding["status"])
    implementation = Path(__file__).resolve()
    return dict(schema="ditserve.wan_profile_figure.v1", frames=list(_FRAMES), sources=sources,
                implementation=dict(path=str(implementation), bytes=implementation.stat().st_size,
                                    sha256=_hash(implementation)),
                metrics=rows, panel_definitions=_PANELS, warnings=warnings,
                study_status=documents["study"].get("run", documents["study"]).get("status", "not_provided"),
                hardware_study_status=validation_study.get("run", validation_study).get("status", "not_provided"),
                hardware_binding=hardware_binding,
                historical_invalid_metrics_used=False,
                status="complete" if all(row["value"] is not None for row in rows
                    if row["panel"] not in ("named_kernels",)) else "partial",
                missing_policy="unmeasured/unverified values remain null; never filled with zero")


def render_figure(analysis, output_prefix):
    """Render PNG and SVG, preserving missing data and measurement denominators."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt
    import numpy as np
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        font_name = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        font_name = "sans-serif"
    style = {"font.family": font_name, "font.size": 10, "axes.titlesize": 12,
             "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
             "axes.labelcolor": "#344054", "text.color": "#1f2937", "axes.edgecolor": "#bdc5d1",
             "axes.unicode_minus": False, "svg.fonttype": "path"}
    rows = analysis["metrics"]
    by_panel = lambda name: [row for row in rows if row["panel"] == name]

    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 3, figsize=(18, 14.8), dpi=180)
        fig.subplots_adjust(left=.062, right=.98, top=.86, bottom=.135, wspace=.28, hspace=.68)
        fig.suptitle("Wan 视频 DiT：时间、显存、真实 kernel 归因与硬件活动", x=.062, y=.97,
                     ha="left", fontsize=22, fontweight="bold")
        fig.text(.062, .931, "Wan2.1-T2V-1.3B · A100 40 GB · 480×832 · 32 步 · CFG 5  |  历史测量与修复后采集分别标注",
                 fontsize=11, color="#5b6575")
        if analysis["status"] != "complete":
            waiting = analysis.get("study_status") == "waiting_resource"
            fig.text(.062, .906, ("补采等待 A100 空闲；新增 E–I 指标尚未测得" if waiting else
                     "补采未完成：空白处为待测 / 尚未通过验证的指标"), fontsize=11, color="#a65216", fontweight="bold")

        def prepare(ax, title, ylabel, percent=False):
            ax.set_title(title, loc="left", pad=12)
            ax.set_xlabel("视频帧数")
            ax.set_ylabel(ylabel)
            ax.set_xticks(_FRAMES)
            ax.set_xlim(12, 70)
            ax.grid(axis="y", color="#e8ebef", linewidth=.8)
            ax.set_axisbelow(True)
            if percent:
                ax.set_ylim(0, 105)

        def lines(ax, panel, series, *, annotate=True):
            total = 0
            for index, (key, label) in enumerate(series):
                data = sorted([row for row in by_panel(panel) if row["series"] == key and row["value"] is not None],
                              key=lambda row: row["frames"])
                if not data:
                    continue
                x, y = [row["frames"] for row in data], [row["value"] for row in data]
                if panel == "dram":
                    color = _COLORS[0 if "read" in key else 1]
                    linestyle = "--" if key.startswith("request") else "-"
                else:
                    color, linestyle = _COLORS[index % len(_COLORS)], "-"
                ax.plot(x, y, marker="o", linewidth=2, markersize=5, color=color, linestyle=linestyle, label=label)
                if annotate:
                    for xx, yy in zip(x, y):
                        offset = -15 if (panel in ("activity", "sm", "tensor") and
                            (key == "request" or key.startswith("request/"))) or (
                            xx == 17 and (panel, key) in (("stages", "transformer"),
                                                          ("named_kernels", "Flash Attention"))) else 7
                        text_offset = (22, -10) if xx == 17 and (panel, key) == ("stages", "transformer") else (0, offset)
                        ax.annotate("%.1f" % yy, (xx, yy), xytext=text_offset, textcoords="offset points",
                                    ha="center", fontsize=9, color=color)
                total += len(data)
            if total:
                ax.legend(frameon=False, loc="best", fontsize=9)
            else:
                ax.text(.5, .5, "未测 / 尚无通过验证的数据", transform=ax.transAxes, ha="center",
                        va="center", color="#8b94a3", fontsize=12)
            return total

        ax = axes[0, 0]
        prepare(ax, "A  生成总耗时｜旧样本 n=3", "秒")
        data = [row for row in by_panel("e2e") if row["value"] is not None]
        if data:
            x, y = [row["frames"] for row in data], [row["value"] for row in data]
            error = [[row["value"] - row["minimum"] for row in data],
                     [row["maximum"] - row["value"] for row in data]]
            ax.errorbar(x, y, yerr=error, fmt="o-", color=_COLORS[0], linewidth=2, capsize=5)
            for xx, yy in zip(x, y):
                ax.annotate("%.2f s" % yy, (xx, yy), xytext=(0, 10), textcoords="offset points", ha="center")
            ax.text(.04, .93, "点=均值；误差线=min–max", transform=ax.transAxes, color="#667085", fontsize=9)
        else:
            lines(ax, "e2e", [("E2E", "E2E")])
        prepare(axes[0, 1], "B  阶段耗时｜旧诊断 n=1", "秒")
        lines(axes[0, 1], "stages", [("text_encode", "CPU 文本编码"), ("transformer", "GPU DiT"), ("vae_decode", "GPU VAE")])
        prepare(axes[0, 2], "C  阶段 allocated 峰值｜旧诊断 n=1", "GiB")
        lines(axes[0, 2], "memory", [("transformer", "DiT（含常驻权重）"), ("vae_decode", "VAE（含常驻权重）")])
        prepare(axes[1, 0], "D  真实 kernel 累计耗时｜旧复核 n=1", "秒")
        lines(axes[1, 0], "named_kernels", [("Flash Attention", "Flash Attention"), ("Named GEMM", "名称匹配 GEMM")])
        axes[1, 0].text(.04, .93, "仅 17 / 65 帧；GEMM 不等于 FFN", transform=axes[1, 0].transAxes,
                        color="#667085", fontsize=9)
        prepare(axes[1, 1], "E  真正 kernel 活跃时间｜新 trace", "%", True)
        lines(axes[1, 1], "activity", [("request", "占完整请求时间"), ("gpu_window", "占 GPU 首末活动窗口")])
        ax = axes[1, 2]
        prepare(ax, "F  DiT kernel 耗时构成｜新 trace", "%", True)
        complete_frames = [frames for frames in _FRAMES if all(any(row["frames"] == frames
            and row["series"] == name and row["value"] is not None for row in by_panel("shares"))
            for name in ("self_attention", "cross_attention", "ffn", "other"))]
        if complete_frames:
            bottoms = np.zeros(len(complete_frames))
            for index, (key, label) in enumerate((("self_attention", "Self Attention"),
                    ("cross_attention", "Cross Attention"), ("ffn", "FFN"), ("other", "其他 DiT kernel"))):
                values = [next(row["value"] for row in by_panel("shares")
                               if row["series"] == key and row["frames"] == frames) for frames in complete_frames]
                ax.bar(complete_frames, values, bottom=bottoms, width=7, color=_COLORS[index], label=label)
                for xx, bottom, value in zip(complete_frames, bottoms, values):
                    if value >= 6:
                        ax.text(xx, bottom + value / 2, "%.1f" % value, ha="center", va="center", color="white", fontsize=9)
                bottoms += values
            ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
        else:
            ax.text(.5, .5, "未测 / 归因尚未通过验证", transform=ax.transAxes, ha="center", color="#8b94a3", fontsize=12)
        for ax, panel, title, metric in ((axes[2, 0], "sm", "G  SM Active｜新硬件采样", "sm_active_pct"),
                                        (axes[2, 1], "tensor", "H  Tensor Active｜新硬件采样", "tensor_active_pct")):
            prepare(ax, title, "%", True)
            lines(ax, panel, [("request/" + metric, "完整请求"), ("denoise/" + metric, "去噪 NVTX 区间")])
        prepare(axes[2, 2], "I  DRAM 读写接口活动｜新硬件采样", "%", True)
        lines(axes[2, 2], "dram", [("request/dram_read_active_pct", "请求：读"),
              ("request/dram_write_active_pct", "请求：写"), ("denoise/dram_read_active_pct", "去噪：读"),
              ("denoise/dram_write_active_pct", "去噪：写")], annotate=False)
        dram_values = [row["value"] for row in by_panel("dram") if row["value"] is not None]
        if dram_values:
            axes[2, 2].set_ylim(0, min(100, max(5, math.ceil(max(dram_values) * 1.25 / 5) * 5)))
        for ax, panel in ((axes[0, 0], "e2e"), (axes[0, 1], "stages"),
                          (axes[0, 2], "memory"), (axes[1, 0], "named_kernels")):
            values = [row.get("maximum", row["value"]) for row in by_panel(panel) if row["value"] is not None]
            ax.set_ylim(0, max(values) * 1.22 if values and max(values) > 0 else 1.)

        counts = sorted({row["n"] for row in rows if row["panel"] in ("activity", "shares") and row["n"]})
        hardware_counts = sorted({row["n"] for row in rows if row["panel"] in ("sm", "tensor", "dram") and row["n"]})
        sample_note = "新 trace 每点 n=%s；硬件区间每点 n=%s。" % (
            "/".join(map(str, counts)) if counts else "未测", "/".join(map(str, hardware_counts)) if hardware_counts else "未测")
        notes = ["A–D 来自不可变历史归档；A 的 n=3 是三个配对输入，B–D 各 n=1。" + sample_note,
                 "B 的 CUDA event 区间可能包含 CPU 提交间隙；E 是真实 kernel 区间并集，不等同 SM/Tensor 利用率；F 分母为 DiT kernel duration 总和。",
                 "G–I 是设备级采样估计；去噪 NVTX 区间未显式同步 GPU 尾部，完整请求已同步。DRAM 是接口活跃周期，非 GB/s；未测留空，连线不代表拟合。"]
        for index, note in enumerate(notes):
            fig.text(.062, .078 - index * .022, note, fontsize=9, color="#606b7b")
        prefix = Path(output_prefix).resolve()
        prefix.parent.mkdir(parents=True, exist_ok=True)
        output = {}
        for suffix in ("png", "svg"):
            destination = prefix.with_suffix("." + suffix)
            fig.savefig(destination, facecolor="white", dpi=200)
            output[suffix] = dict(path=str(destination), bytes=destination.stat().st_size, sha256=_hash(destination))
        plt.close(fig)
    return output


def generate_figure(legacy_run, legacy_analysis, output_prefix, *, study=None, hardware=None,
                    hardware_study=_HARDWARE_STUDY_UNSET):
    analysis = build_analysis(legacy_run, legacy_analysis, study=study, hardware=hardware, hardware_study=hardware_study)
    analysis["figures"] = render_figure(analysis, output_prefix)
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", default="results/wan-breakdown/run01")
    parser.add_argument("--legacy-analysis", default="results/wan-breakdown/analysis01")
    parser.add_argument("--study")
    parser.add_argument("--hardware-json")
    parser.add_argument("--hardware-study", help="independent hardware run/archive; never falls back to --study when missing")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--analysis-json", help="optional temporary scalar analysis for archival; not a default extra result")
    args = parser.parse_args()
    options = dict(study=args.study, hardware=args.hardware_json)
    if args.hardware_study is not None:
        options["hardware_study"] = args.hardware_study
    analysis = generate_figure(args.legacy_run, args.legacy_analysis, args.output_prefix, **options)
    if args.analysis_json:
        Path(args.analysis_json).write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=analysis["status"], figures=analysis["figures"],
                          warnings=analysis["warnings"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
