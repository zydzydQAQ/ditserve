"""Strict generic-quantum/decode audit for the separate remaining-online role."""

import math
import statistics

from . import workflow_cache_combination_report as combination
from . import workflow_contention_study as contention
from . import workflow_dispatch_refresh_report as dispatch
from . import workflow_encoding_optimization_report as encoding
from . import workflow_next_report as report
from . import workflow_occupancy_report as occupancy
from . import workflow_remaining_online as study_api
from .workflow_next_study import read
from .workflow_serving_study import digest


ROLE = study_api.ROLE
SCHEMA = "ditserve.workflow.remaining-online-report.v1"
_require = encoding._require
_number = encoding._number


def _feedback(phase, row, event, images):
    required = bool(phase["options"].get("decode_frames"))
    feedback = event.get("quantum_feedback")
    if not required:
        _require(feedback is None, "unexpected feedback without explicit decode-frame instrumentation")
        return
    _require(isinstance(feedback, dict), "missing actual fixed-quantum feedback")
    observed = feedback["observed_at_ms"]
    _require(_number(observed) and observed <= event["start_ms"], "feedback occurs after dispatch")
    waiting, last = feedback["waiting_images"], feedback["last_image_arrival_ms"]
    possible = [r for r in images if r["arrival_ms"] <= observed <= r["start_ms"]]
    _require(type(waiting) is int and 0 <= waiting <= len(possible), "impossible image waiter count")
    _require(last is None or (_number(last) and last <= observed and any(r["arrival_ms"] == last for r in images)),
             "feedback includes an image that has not arrived")
    wait = feedback["oldest_image_wait_ms"]
    _require(_number(wait) and wait <= max((observed-r["arrival_ms"] for r in possible), default=0)
             and (waiting != 0 or wait == 0), "invalid oldest image wait")
    _require(feedback["adaptive"] is False and feedback["mode"] == "fixed"
             and feedback["trigger"] == "fixed" and feedback["hot_window_ms"] == 0,
             "fixed quantum unexpectedly used adaptive feedback")


def _cell(phase, observed):
    """Preserve lifecycle/CPU checks while deriving work from this frozen arm."""
    _require(observed.get("status") == "completed" and len(observed["cells"]) == 1, "incomplete arm")
    cell = observed["cells"][0]
    _require(cell.get("status") == "completed" and cell.get("real_backends_pinned_sp1") is True
             and cell.get("real_sp") is None and cell.get("gpus") == 2
             and cell.get("policy") == phase["policy"] and cell.get("text_prefetch_depth") == 1,
             "actual pinned SP1 execution differs")
    _require(cell.get("scheduler_options") == dict(phase["options"], degrees=[1], pin_started=True),
             "actual scheduler controls differ")
    _require(_number(cell["elapsed_ms"], positive=True), "invalid normal timing interval")
    rows = cell["requests"]
    _require(len(rows) == len(phase["workload"]) and report.frozen_request_map(rows) ==
             report.frozen_request_map(phase["workload"]), "actual requests differ from frozen workload")
    planned = {r["request_id"]: r for r in phase["workload"]}
    images = [r for r in rows if r["kind"] == "image"]
    work = dict(requests=len(rows), images=len(images), videos=len(rows)-len(images),
                observed_executed_blocks=0, step_dispatches=0, derived_step_blocks=0,
                decode_dispatches=0, decoded_latent_frames=0, per_video=[])
    for row in rows:
        _require(row.get("status") == "completed" and all(row.get(k) == planned[row["request_id"]][k]
                 for k in ("kind", "class")), "unfinished or reclassified request")
        _require(all(_number(row[k]) for k in ("arrival_ms", "start_ms", "finish_ms", "deadline_ms"))
                 and row["arrival_ms"] <= row["start_ms"] <= row["finish_ms"], "invalid request timing")
        events = row["dispatch_timeline"]
        _require(events and all(e.get("group") in ([0], [1]) and _number(e["start_ms"])
                 and _number(e["finish_ms"]) and row["start_ms"] <= e["start_ms"] <= e["finish_ms"] <= row["finish_ms"]
                 for e in events), "invalid actual dispatch timeline")
        _require(all(a["finish_ms"] <= b["start_ms"] for a, b in zip(events, events[1:])), "request quanta overlap")
        _require(all(e["group"] == events[0]["group"] for e in events), "pinned request migrated")
        slot = events[0]["group"][0]
        worker = ("sana" if row["kind"] == "image" else "wan") + str(slot)
        _require(row.get("worker_pids") == {worker: cell["worker_runtime"][worker]["pid"]}, "request worker/GPU differs")
        for event in events:
            _feedback(phase, row, event, images)
        if row["kind"] == "image":
            _require(len(events) == 1 and events[0].get("quantum") == "atomic"
                     and row.get("operation_counts") == dict(prepare=1, advance=1, finish=1, release=1)
                     and row.get("quanta") == 1 and row.get("degrees") == [1] and len(row["outputs"]) == 1,
                     "image lifecycle differs")
            _require(_number(row["model_ms"], positive=True) and row["model_ms"] <=
                     events[0]["finish_ms"]-events[0]["start_ms"], "invalid image model service")
            continue
        cost, options = phase["profile"]["classes"][row["class"]], phase["options"]
        wanted = row["spec"]["steps"] * cost["blocks_per_step"]
        step = phase["policy"] == "static_sp1"
        quantum = max(1, min(math.floor(options["quantum_ms"] / (cost["step_compute_ms"] / cost["blocks_per_step"])),
                             cost["blocks_per_step"]))
        blocks = steps = decoded = finishes = advances = 0
        for event in events:
            if event["quantum"] == "finish":
                _require((steps * cost["blocks_per_step"] if step else blocks) == wanted,
                         "decode dispatched before complete DiT work")
                remaining = cost["decode_chunks"] - decoded
                _require(remaining > 0, "extra decode dispatch")
                if options["chunked_decode"]:
                    frames = min(options["decode_frames"], remaining)
                    _require(type(event.get("decode_frames")) is int and event["decode_frames"] == frames,
                             "decode frame quantum differs")
                else:
                    _require("decode_frames" not in event, "unchunked decode acquired a frame limit")
                    frames = remaining
                decoded += frames
                finishes += 1
            else:
                _require(finishes == 0, "DiT resumed after decoding")
                advances += 1
                if step:
                    _require(event["quantum"] == "step" and "executed_blocks" not in event,
                             "step arm acquired block telemetry")
                    steps += 1
                else:
                    _require(type(event["quantum"]) is int and event["quantum"] == quantum
                             and type(event.get("executed_blocks")) is int and 0 < event["executed_blocks"] <= quantum,
                             "actual block quantum differs")
                    if options.get("decode_frames"):
                        _require(event.get("block_count") == quantum and event.get("chosen_quantum_ms") == options["quantum_ms"],
                                 "actual chosen quantum differs")
                    else:
                        _require("block_count" not in event and "chosen_quantum_ms" not in event,
                                 "unexpected uninstrumented fixed-quantum metadata")
                    blocks += event["executed_blocks"]
        _require((steps == row["spec"]["steps"] if step else blocks == wanted)
                 and decoded == cost["decode_chunks"], "incomplete actual video work")
        ops = row["operation_counts"]
        _require(ops.get("prepare") == 1 and ops.get("advance") == advances and ops.get("finish") == finishes
                 and ops.get("release") == 1 and row["quanta"] == len(events) and row["degrees"] == [1],
                 "video lifecycle counts differ")
        work["observed_executed_blocks"] += blocks
        work["step_dispatches"] += steps
        work["derived_step_blocks"] += steps * cost["blocks_per_step"]
        work["decode_dispatches"] += finishes
        work["decoded_latent_frames"] += decoded
        work["per_video"].append(dict(request_id=row["request_id"], observed_executed_blocks=blocks,
            step_dispatches=steps, derived_step_blocks=steps*cost["blocks_per_step"],
            decode_dispatches=finishes, decoded_latent_frames=decoded))
    _require(len(cell["warmup"]) == 4 and [r["worker"] for r in cell["warmup"]] ==
             [r["worker"] for r in phase["warmup"]], "warmup provenance differs")
    _require(set(cell["worker_runtime"]) == {"sana0", "sana1", "wan0", "wan1"}, "missing worker runtime")
    for name, runtime in cell["worker_runtime"].items():
        settings, threads = phase["worker_overrides"][name], 4 if name.startswith("sana") else 8
        binding = runtime["cpu_binding"]
        _require(type(runtime["pid"]) is int and runtime["pid"] > 0 and runtime.get("measured") is True
                 and runtime.get("backend") == ("sana" if threads == 4 else "wan")
                 and binding.get("configured") is True and binding.get("requested_affinity") == settings["cpu_affinity"]
                 and binding.get("effective_affinity") == settings["cpu_affinity"]
                 and binding.get("requested_threads") == threads and binding.get("torch_threads") == threads,
                 "actual worker CPU/process identity differs")
    return cell, work


def _image_service(cell):
    rows = [r for r in cell["requests"] if r["kind"] == "image"]
    series = dict(worker_model_ms=[r["model_ms"] for r in rows],
        host_dispatch_ms=[r["dispatch_timeline"][0]["rpc_completed_ms"]-r["dispatch_timeline"][0]["start_ms"] for r in rows],
        queue_after_admission_ms=[r["dispatch_timeline"][0]["start_ms"]-r["admission_ms"] for r in rows],
        harvest_delay_ms=[r["dispatch_timeline"][0]["harvest_delay_ms"] for r in rows])
    _require(rows and all(_number(v) for values in series.values() for v in values), "invalid service samples")
    return dict(requests=len(rows), timing_scope=combination.TIMING_SCOPE, **{name: dict(
        mean=sum(values)/len(values), p50=report.percentile(values,.5), p95=report.percentile(values,.95),
        maximum=max(values)) for name, values in series.items()})


def audit_result(result, job, frozen):
    checked = dict(passed=False, errors=[], arms={}, exact={}, performance_evidence=False)
    try:
        metadata = job["metadata"]
        _require(metadata["role"] == ROLE and digest(frozen) == job["configuration_sha256"], "role/input digest differs")
        _require(job.get("logical_id",job["id"]) == study_api.job_id(metadata["family"],metadata["seed"]),
                 "job identity differs from frozen family/seed")
        study_api.validate_config(frozen, metadata)
        _require(result.get("schema") == "ditserve.run.v1" and result.get("status") == "completed"
                 and result.get("config", {}).get("phases") == frozen["phases"], "result/frozen phases differ")
        _require([p["id"] for p in result["phases"]] == [p["id"] for p in frozen["phases"]], "missing or reordered arms")
        mapping = {r["ordinal"]: r["uuid"] for r in job["gpu_mapping"]}
        _require(set(mapping) == {0,1} and len(job["gpu_mapping"]) == 2, "GPU mapping differs")
        uuids = {name:mapping[settings["gpu"]] for name,settings in frozen["workers"].items()}
        _require(all(result["config"]["workers"][name]["gpu_uuid"] == uuid for name,uuid in uuids.items()),
                 "resolved GPU UUID differs")
        cells, owners = {}, []
        definitions = {a["id"]:a for a in study_api.arm_definitions(metadata["family"])}
        for phase, actual in zip(frozen["phases"], result["phases"]):
            name = phase["id"]
            cell, work = _cell(phase, actual)
            cells[name] = cell
            enabled = definitions[name]["cache"]
            for slot in (0,1):
                warm, worker = cell["warmup"][slot+2], f"sana{slot}"
                _require(warm.get("worker_pid") == cell["worker_runtime"][worker]["pid"]
                         and warm.get("state_id") == f"warmup-{slot+2}" and warm.get("spec") ==
                         dict(phase["warmup"][slot+2]["spec"], request_id=f"warmup-{slot+2}"), "Sana warmup provenance differs")
            sana = encoding._sana_cache(dict(phase,id="staged" if enabled else "disabled"),cell,"sana_prompt_cache")
            wan = encoding._wan_cache(dict(phase,id="enabled"),cell)
            _require(all(w["counters"][k] == 0 for w in sana.values() for k in ("errors","bypasses")), "Sana cache failed/bypassed")
            if enabled:
                measured = {k:sum(w["measured"][k] for w in sana.values()) for k in ("requests","hits","misses")}
                _require(measured["requests"] == work["images"], "cache omitted measured images")
                if metadata["family"] == "quantum_unique":
                    _require(measured["hits"] == 0 and measured["misses"] == work["images"], "unique cache did not miss every request")
                else:
                    _require(all(w["counters"]["evictions"] == 0 for w in sana.values()), "four-prompt cache unexpectedly evicted")
            _require(all(w["measured"]["bypasses"] == 0 for w in wan.values()), "Wan cache bypassed")
            for row in cell["requests"]:
                hashes = report.request_hashes(row)
                _require(len(hashes) == (2 if row["kind"] == "video" else 1) and all(isinstance(v,str)
                         and len(v)==64 and all(c in "0123456789abcdef" for c in v) for v in hashes.values()), "invalid full output/latent hash")
            checked["arms"][name] = dict(sana=sana,wan=wan,work=work,
                timing=dispatch._telemetry(phase,cell),guard=occupancy.guard_telemetry(phase,cell,uuids),
                image_service=_image_service(cell))
            owners.extend(w["adapter_id"] for w in sana.values())
            owners.extend(w["instance_id"] for w in wan.values())
        _require(len(owners)==len(set(owners)), "cache owner reused across workers/arms")
        for left,right in study_api.pairs(metadata["family"]):
            exact = report.exact_pair(cells[left]["requests"],cells[right]["requests"])
            checked["exact"][left+"/"+right] = exact
            _require(exact["passed"] and exact["requests"] == len(frozen["phases"][0]["workload"]), "paired output/latent differs")
        checked.update(passed=True, performance_evidence=True,
            trigger_observations=contention.trigger_summary(result),
            scope="Step work is counted as step RPCs and derived expected blocks; only integer-quantum blocks are directly observed.")
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError) as error:
        checked["errors"].append(str(error))
    return checked


def _metrics(cell, checked):
    return dict(encoding._metrics(cell,checked),guard=checked["guard"],image_service=checked["image_service"],work=checked["work"])


def paired_changes(arms, family):
    return {left+"/"+right:combination.paired_changes(dict(reference=arms[left],candidate=arms[right]))["reference/candidate"]
            for left,right in study_api.pairs(family)}


def performance_assessment(arms, family):
    out = {}
    for left,right in study_api.pairs(family):
        a,b = arms[left],arms[right]
        image_ratio = b["image"]["latency_p95_ms"]/a["image"]["latency_p95_ms"]
        video_ratio = b["video"]["latency_p95_ms"]/a["video"]["latency_p95_ms"]
        throughput_ratio = b["total"]["throughput_per_second"]/a["total"]["throughput_per_second"]
        safety = dict(image_p95=image_ratio<=1.05,video_p95=video_ratio<=1.05,throughput=throughput_ratio>=.97,
                      image_slo=b["image"]["slo_attainment"]>=a["image"]["slo_attainment"])
        out[left+"/"+right] = dict(safety_passed=all(safety.values()),safety=safety,
            image_p95_reduction_percent=(1-image_ratio)*100,video_p95_ratio=video_ratio,throughput_ratio=throughput_ratio,
            mean_service_reduction_percent=(1-b["image_service"]["worker_model_ms"]["mean"]/a["image_service"]["worker_model_ms"]["mean"])*100,
            safety_only=family=="quantum_unique")
    return out


def interactions(arms, family):
    if family in ("quantum_repeated","quantum_unique","poisson","high_load"):
        a,b,c,d = (arms[name] for name in study_api.FOUR_ARMS)
    elif family == "scheduler":
        a,b,c,d = (arms[name] for name in ("S1","S2","S3","S4"))
    else:
        return {}
    return dict(scope="per-seed difference of differences in milliseconds, not additive percentages",
        image_p95_ms=(d["image"]["latency_p95_ms"]-c["image"]["latency_p95_ms"])-(b["image"]["latency_p95_ms"]-a["image"]["latency_p95_ms"]),
        image_mean_service_ms=(d["image_service"]["worker_model_ms"]["mean"]-c["image_service"]["worker_model_ms"]["mean"])-
                              (b["image_service"]["worker_model_ms"]["mean"]-a["image_service"]["worker_model_ms"]["mean"]))


def family_assessment(rows, family):
    rows = [r for r in rows if r["stage"]=="confirmation" and r["family"]==family]
    complete = len(rows)==3 and {r["seed"] for r in rows}==set(study_api.SEEDS[family]) and all(r["qualified"] for r in rows)
    out = dict(qualified=complete,comparisons={},confirmation_seeds=list(study_api.SEEDS[family]),
               screening_excluded=True,default_change=False,rule=study_api.RULE)
    for left,right in study_api.pairs(family):
        key=left+"/"+right
        values=[r["performance_assessment"][key] for r in rows] if complete else []
        gains=[v["image_p95_reduction_percent"] for v in values]
        safety=bool(complete and all(v["safety_passed"] for v in values))
        benefit=bool(complete and family!="quantum_unique" and statistics.median(gains)>=10 and sum(v>0 for v in gains)>=2)
        out["comparisons"][key]=dict(qualified=complete,safety_passed=safety,benefit_confirmed=benefit,
            promotable=safety and benefit,safety_only=family=="quantum_unique",
            image_p95_reduction_median_percent=statistics.median(gains) if gains else None,
            improving_seeds=sum(v>0 for v in gains),
            mean_service_reduction_median_percent=statistics.median(v["mean_service_reduction_percent"] for v in values) if values else None)
    out["paired_changes_median"] = encoding._median_tree([r["paired_changes"] for r in rows]) if complete else None
    out["absolute_arm_medians"] = encoding._median_tree([r["arms"] for r in rows]) if complete else None
    out["interactions_median"] = encoding._median_tree([r["interactions"] for r in rows]) if complete else None
    return out


def summarize(study):
    out=dict(schema=SCHEMA,per_seed=[],families={},default_change=False,rule=study_api.RULE,
             scope="Frozen independent confirmation groups; all safety failures retained; no automatic selection")
    for job in study["jobs"]:
        if job.get("superseded") or job.get("metadata",{}).get("role")!=ROLE:
            continue
        metadata=job["metadata"]
        ev=report.evidence(study,job)
        row=dict(job_id=job["id"],family=metadata["family"],seed=metadata["seed"],stage=metadata["stage"],
                 status=ev["launch"].get("status","not_started"),qualified=False,audit=None,arms={},
                 paired_changes={},interactions={},performance_assessment={},prerequisites={},sampled_capacity=None)
        if row["status"]=="completed":
            checked=audit_result(ev["result"],job,read(job["input"],{}))
            capacity=encoding.raw_capacity(ev["launch"],job)
            for logical in job["requires"]:
                try:
                    gate=report.evidence(study,report.current_job(study,logical))
                    row["prerequisites"][logical]=bool(gate["qualified"] and gate["capacity"])
                except (KeyError,ValueError,TypeError,OSError):
                    row["prerequisites"][logical]=False
            row.update(audit=checked,sampled_capacity=capacity,qualified=bool(ev["qualified"] and ev["capacity"]
                and checked["passed"] and capacity["passed"] and row["prerequisites"] and all(row["prerequisites"].values())))
            if checked["passed"]:
                row["arms"]={name:_metrics(cell,checked["arms"][name]) for name,cell in report.native_arms(ev["result"]).items()}
                row["paired_changes"]=paired_changes(row["arms"],metadata["family"])
                row["performance_assessment"]=performance_assessment(row["arms"],metadata["family"])
                row["interactions"]=interactions(row["arms"],metadata["family"])
        out["per_seed"].append(row)
    out["families"]={name:family_assessment(out["per_seed"],name) for name in study_api.SEEDS}
    out["qualified"]=len(out["per_seed"])==20 and all(r["qualified"] for r in out["per_seed"])
    out["all_confirmations_qualified"]=all(f["qualified"] for f in out["families"].values())
    return out
