"""Fail-closed qualification and scoped summaries for the remaining matrix.

Launch/source bindings and outer sampled capacity are checked by the suite and
workflow_next_report. This module replays raw results and dependency evidence.
"""

import copy
import math
from pathlib import Path

from . import workflow_next_study as api
from . import workflow_next_report as report
from . import workflow_remaining_online_report as online
from . import workflow_encoding_optimization_report as encoding
from . import workflow_dispatch_refresh_report as dispatch
from . import workflow_occupancy_report as occupancy
from .workflow_serving_study import digest

SCHEMA = "ditserve.workflow.remaining-report.v1"
CPU_SEEDS = (5301, 5302, 5303)
CPU_PAIRS = (("depth0", "depth1"), ("depth1", "depth2"))
_require, _number = encoding._require, encoding._number
ERRORS = (KeyError, TypeError, ValueError, AttributeError, IndexError, ArithmeticError, OSError)


def _hash(value):
    return (isinstance(value, dict) and value.get("finite") is True
            and isinstance(value.get("sha256"), str) and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"])
            and isinstance(value.get("shape"), list) and bool(value["shape"])
            and all(type(v) is int and v > 0 for v in value["shape"])
            and isinstance(value.get("dtype"), str) and bool(value["dtype"]))


def _cpu_config(frozen, metadata):
    seed, repeat, capacity = metadata["seed"], metadata["repeat"], metadata["frozen_video_capacity"]
    _require(seed in CPU_SEEDS and repeat == CPU_SEEDS.index(seed) and _number(capacity, positive=True),
             "CPU seed/repeat/capacity differs")
    profile = frozen["phases"][0]["profile"]
    online.contention._profile(profile)
    workload = api.trajectory(profile, seed, count=16, videos_only=True, pattern="bursty", load=.8, lambda0=capacity)
    arms = [dict(id=f"depth{d}", policy="block_elastic", quantum_ms=300, decode_frames=1,
                 chunked_decode=True, late_short_ms=2000, depth=d) for d in (0, 1, 2)]
    arms = arms[repeat:] + arms[:repeat]
    expected = api.cluster_config(profile, workload, arms, cores=8, depth=1)
    for phase in expected["phases"]:
        phase.update(controller_cpu_affinity=[59], record_dispatch_timeline=True,
                     dispatch_refresh=False, occupancy_backend="nvml")
    observed = copy.deepcopy(frozen)
    identity = observed.pop("experiment")
    _require(identity in ("run03",f"run03-remaining-cpu-depth-s{seed}"), "CPU experiment identity differs")
    envelope = observed.pop("budget",None)
    _require(envelope is None or envelope==dict(limit_gpu_seconds=8400,watchdog_seconds=4200,
                 repair_reserve_gpu_seconds=0,carry="none"), "CPU watchdog envelope differs")
    for phase in observed["phases"]:
        dependencies = phase.pop("artifact_dependencies",None)
        _require(dependencies is None or dependencies==[dict(closed=False,reason="run03 numerical diagnosis and review pending")],
                 "CPU artifact-review envelope differs")
    expected.pop("experiment")
    _require(observed == expected, "CPU frozen configuration differs from fresh depth-only matrix")


def _cpu_timing(phase, cell):
    if phase["text_prefetch_depth"]:
        return dispatch._telemetry(phase, cell)
    _require(cell.get("dispatch_refresh") is False and cell["elapsed_ms"] >=
             max(r["finish_ms"] for r in cell["requests"]), "invalid synchronous cell timing")
    admission, harvest = [], []
    for row in cell["requests"]:
        admitted = row["admission_ms"]
        _require(_number(admitted) and row["arrival_ms"] <= admitted <= row["start_ms"], "invalid synchronous admission")
        _require(not any(row["operation_counts"].get(k, 0) for k in ("encode_start", "encode_poll"))
                 and all(k not in row for k in ("encoder_rpc", "encode_submit_ms", "encode_ready_ms"))
                 and [e["op"] for e in row["encoding_events"]] == ["prepare"], "depth0 unexpectedly used prefetch")
        admission.append(admitted-row["arrival_ms"])
        for event in row["dispatch_timeline"]:
            complete, harvested, delay = event["rpc_completed_ms"], event["harvested_ms"], event["harvest_delay_ms"]
            _require(event.get("timing_scope") == "host_dispatch_task" and all(_number(v) for v in (complete, harvested, delay))
                     and event["start_ms"] <= complete <= harvested <= event["finish_ms"]
                     and math.isclose(harvested-complete, delay, abs_tol=1e-5), "invalid synchronous harvest timing")
            harvest.append(delay)
    return dict(admission_delay_p95_ms=report.percentile(admission,.95), admission_delay_max_ms=max(admission),
                harvest_delay_p95_ms=report.percentile(harvest,.95), harvest_delay_max_ms=max(harvest),
                encoding_rpc_max_ms=None, scope="synchronous prepare; host task times are not CUDA intervals")


def _cpu_audit(result, job, frozen):
    _cpu_config(frozen, job["metadata"])
    _require(job.get("logical_id", job["id"]) == f"remaining-cpu-depth-s{job['metadata']['seed']}", "CPU job identity differs")
    _require(result.get("schema") == "ditserve.run.v1" and result.get("status") == "completed"
             and result["config"]["phases"] == frozen["phases"], "CPU result/frozen phases differ")
    _require([x["id"] for x in result["phases"]] == [x["id"] for x in frozen["phases"]], "CPU arm order differs")
    mapping = {r["ordinal"]: r["uuid"] for r in job["gpu_mapping"]}
    _require(set(mapping) == {0,1} and len(job["gpu_mapping"]) == 2, "CPU GPU mapping differs")
    uuids = {name: mapping[w["gpu"]] for name,w in frozen["workers"].items()}
    _require(all(result["config"]["workers"][n]["gpu_uuid"] == u for n,u in uuids.items()), "CPU resolved GPU UUID differs")
    arms, cells, owners = {}, {}, []
    for phase, actual in zip(frozen["phases"], result["phases"]):
        _require(actual["cells"][0]["text_prefetch_depth"] == phase["text_prefetch_depth"], "actual prefetch depth differs")
        # Only the already-checked depth is projected for the shared generic
        # work validator. Its fixed-depth online role remains strict.
        projected = copy.deepcopy(actual)
        projected["cells"][0]["text_prefetch_depth"] = 1
        effective = copy.deepcopy(phase)
        effective["worker_overrides"] = {n: dict(w["config"], **phase["worker_overrides"].get(n, {}))
                                         for n,w in frozen["workers"].items()}
        _, work = online._cell(effective, projected)
        cell = actual["cells"][0]
        sana = encoding._sana_cache(dict(phase,id="disabled"),cell,"sana_prompt_cache")
        wan = encoding._wan_cache(dict(phase,id="disabled"),cell)
        for index,warm in enumerate(cell["warmup"]):
            name = phase["warmup"][index]["worker"]
            _require(warm.get("worker_pid") == cell["worker_runtime"][name]["pid"] and warm.get("state_id") == f"warmup-{index}"
                     and warm.get("spec") == dict(phase["warmup"][index]["spec"],request_id=f"warmup-{index}"), "CPU warmup provenance differs")
        for row in cell["requests"]:
            hashes = report.request_hashes(row)
            _require(len(hashes) == 2 and all(isinstance(v,str) and len(v)==64 and
                     all(c in "0123456789abcdef" for c in v) for v in hashes.values()), "CPU full output/latent hashes missing")
        arms[phase["id"]] = dict(sana=sana,wan=wan,work=work,timing=_cpu_timing(phase,cell),
                                  guard=occupancy.guard_telemetry(phase,cell,uuids))
        cells[phase["id"]] = cell
        owners.extend(v["adapter_id"] for v in sana.values())
    _require(len(owners)==len(set(owners)), "CPU worker adapter reused across arms")
    exact = {a+"/"+b:report.exact_pair(cells[a]["requests"],cells[b]["requests"]) for a,b in CPU_PAIRS}
    _require(all(x["passed"] and x["requests"]==16 for x in exact.values()), "CPU paired output/latent differs")
    return dict(passed=True,errors=[],arms=arms,exact=exact,performance_evidence=True)


def _interleave(result, frozen):
    from . import workflow_elastic_validation as module
    _require(module.normalize_config(frozen)==frozen and frozen["stage"]=="interleave"
             and frozen["seeds"]==[5401,5402], "interleave frozen matrix differs")
    checked = module.validate_repaired_result(result)
    _require(checked["passed"], "; ".join(checked["errors"]))
    count = 0
    for rank in result["ranks"]:
        for case in rank["cases"]:
            for gate in case["gates"]:
                if gate["gate"] != "sp2_sana_interleave":
                    continue
                rows = gate["sana_interruptions"]
                if rank["rank"] == 1:
                    _require(rows == [], "rank1 must not execute Sana interruptions")
                    continue
                reference = case["sana_reference"]
                _require(_hash(reference) and [r["step"] for r in rows]==[0,5,10,15], "interleave image matrix incomplete")
                for row in rows:
                    _require(row["after_blocks"]==23 and row["passed"] is True and row["wan_latent_unchanged"] is True
                             and row["output"]==reference and row["reference"]==reference, "Sana insertion output or Wan continuation differs")
                    count += 1
    _require(count==16, "expected sixteen rank0 Sana insertions")
    return dict(checked,interruptions=count,performance_evidence=False)


def _device_binding(frozen, job, role):
    mapping = {r["ordinal"]:r["uuid"] for r in job["gpu_mapping"]}
    _require(len(mapping)==len(job["gpu_mapping"]), "duplicate physical GPU binding")
    if role=="remaining_sp_comparison":
        _require(frozen["gpu_uuids"]==[mapping[i] for i in (0,1)], "SP physical UUID binding differs")
    elif role=="remaining_vae_lifecycle_bundle":
        _require(all(mapping[c["device_ordinal"]]==c["gpu_uuid"]==c["config"]["gpu_uuid"] for c in frozen["cases"]), "bundle physical UUID binding differs")
    elif role in ("remaining_vae_performance","remaining_vae_trace"):
        _require(len(job["devices"])==1 and frozen["gpu_uuid"]==mapping[job["devices"][0]], "VAE physical UUID binding differs")


def audit_result(result, job, frozen):
    checked = dict(passed=False, errors=[], performance_evidence=False)
    try:
        role = job["metadata"]["role"]
        _require(isinstance(frozen,dict) and digest(frozen)==job["configuration_sha256"], "frozen input digest differs")
        if role=="remaining_online":
            return online.audit_result(result,job,frozen)
        if role=="remaining_cpu_depth":
            return _cpu_audit(result,job,frozen)
        _require(result.get("configuration")==frozen, "diagnostic result/frozen configuration differs")
        _device_binding(frozen,job,role)
        if role=="remaining_interleave":
            return _interleave(result,frozen)
        from . import workflow_sp_comparison as sp
        from . import workflow_overlap_bundle as bundle
        from . import workflow_overlap_performance as performance
        from . import workflow_overlap_trace as trace
        modules = dict(remaining_sp_comparison=sp,remaining_vae_lifecycle_bundle=bundle,
                       remaining_vae_performance=performance,remaining_vae_trace=trace)
        _require(role in modules, "unknown remaining role")
        module = modules[role]
        _require(module.normalize_config(frozen)==frozen, "diagnostic normalized configuration differs")
        if role=="remaining_sp_comparison":
            _require(frozen["seeds"]==[job["metadata"]["seed"]] and job["metadata"]["seed"] in sp.SEEDS, "SP independent seed differs")
        checked = module.validate_result(result,frozen) if role=="remaining_vae_lifecycle_bundle" else module.validate_result(result)
        checked["performance_evidence"] = bool(checked["passed"] and role in ("remaining_sp_comparison","remaining_vae_performance"))
    except ERRORS as error:
        checked.update(passed=False)
        checked.setdefault("errors",[]).append(str(error))
    return checked


def prerequisite_gate(study, job):
    """Only inspect dependencies, never this job's evidence (no self recursion)."""
    checked = dict(passed=False, errors=[], reasons=[], dependencies={})
    try:
        dependencies = job["requires"]
        _require(isinstance(dependencies,list) and dependencies and len(set(dependencies))==len(dependencies), "missing/duplicate dependency list")
        for logical in dependencies:
            _require(logical not in (job["id"],job.get("logical_id")), "self dependency rejected")
            dependency = report.current_job(study,logical)
            ev = report.evidence(study,dependency)
            detail = dict(passed=False)
            if dependency.get("metadata",{}).get("role")=="remaining_vae_lifecycle_bundle":
                selected = job["metadata"]["lifecycle_case"]
                frozen = api.read(dependency["input"],{})
                audit = audit_result(ev.get("result",{}),dependency,frozen)
                case = audit.get("cases",{}).get(selected,{})
                target = api.read(job["input"],{})
                source = next(c for c in frozen["cases"] if c["id"]==selected)
                matching = (target.get("frame_pairs")==source["config"]["frame_pairs"]
                            and target.get("gpu_uuid")==source["gpu_uuid"]
                            and job["devices"]==[source["device_ordinal"]])
                dependency_gates = ev.get("remaining_prerequisites",{}).get("passed") is True
                observer = ev.get("remaining_audit",{}).get("observer_affinity_verified") is True
                detail.update(passed=bool(ev.get("guarded") and dependency_gates and observer
                                          and audit.get("structure_valid") and case.get("passed") and matching),
                              lifecycle_case=selected,structure_valid=audit.get("structure_valid",False),
                              observer_affinity_verified=observer,prerequisites_passed=dependency_gates,
                              required_scope=case.get("required_scope"),scope="guarded output + complete bundle structure + selected case; diagnostic third arm is independent")
            else:
                detail["passed"] = bool(ev.get("qualified") and ev.get("capacity"))
            checked["dependencies"][logical] = detail
            if not detail["passed"]:
                checked["errors"].append("prerequisite unqualified: "+logical)
        if job.get("metadata",{}).get("role")=="remaining_cpu_depth":
            exact = report.screening_exact(study,cross_cores=False)
            checked["cross_depth_exact"] = exact is True
            if not exact:
                checked["errors"].append("original eight-core cross-depth exact gate failed")
        checked["passed"] = not checked["errors"]
    except (*ERRORS, StopIteration) as error:
        checked["errors"].append(str(error))
    checked["reasons"] = list(checked["errors"])
    return checked


def _cpu_metrics(result, audit):
    arms = {name:encoding._metrics(cell,audit["arms"][name]) for name,cell in report.native_arms(result).items()}
    changes, safety = {}, {}
    for left,right in CPU_PAIRS:
        a,b = arms[left],arms[right]
        delta = {}
        for kind in ("total","video"):
            delta[kind] = {}
            for metric in ("latency_p50_ms","latency_p95_ms","wait_p50_ms","wait_p95_ms","throughput_per_second"):
                x,y = a[kind][metric],b[kind][metric]
                delta[kind][metric] = dict(reference=x,candidate=y,candidate_minus_reference=y-x,
                    improvement_percent=((y-x) if metric=="throughput_per_second" else (x-y))*100/x if x else None)
            delta[kind]["slo_change_percentage_points"] = 100*(b[kind]["slo_attainment"]-a[kind]["slo_attainment"])
        key = left+"/"+right
        changes[key] = delta
        safety[key] = dict(video_p95=b["video"]["latency_p95_ms"]<=1.05*a["video"]["latency_p95_ms"],
                           throughput=b["total"]["throughput_per_second"]>=.97*a["total"]["throughput_per_second"])
        safety[key]["passed"] = all(safety[key].values())
    return dict(arms=arms,paired_changes=changes,safety=safety)


def _trace_summary(study, job):
    value = dict(available=False,cuda_overlap_proven=False,scope="parsed launch-correlated CUDA kernel intersections only")
    try:
        from .workflow_overlap_trace import normalize_config, capture_range
        from .workflow_trace import parse_label
        root = Path(study["root"])
        frozen = api.read(job["input"], {})
        _require(digest(frozen) == job["configuration_sha256"] and normalize_config(frozen) == frozen,
                 "trace frozen configuration differs")
        expected = parse_label(capture_range(frozen))
        manifest = api.read(root/"remaining-traces.json",{})
        _require(manifest.get("schema")=="ditserve.remaining.trace-postprocess.v1", "trace postprocess absent")
        rows = [r for r in manifest["entries"] if r["job"]==job["id"]]
        _require(len(rows)==1 and rows[0]["status"]=="parsed", "trace not parsed")
        row = rows[0]
        capture = root/"traces"/(job["id"]+".nsys-rep")
        _require(row.get("capture") == str(capture), "capture belongs to a different job")
        bindings = {key: row[key] for key in ("capture_binding", "sqlite", "parsed")}
        for key, path in (("capture_binding", capture), ("sqlite", capture.with_suffix(".sqlite")),
                          ("parsed", capture.with_suffix(".parsed.json"))):
            _require(bindings[key].get("path") == str(path) and api.binding(path) == bindings[key],
                     key + " trace binding differs")
        binding = bindings["parsed"]
        parsed = api.read(binding["path"],{})
        _require(parsed.get("schema_version")=="ditserve.workflow_trace.v1" and
                 parsed.get("source") == bindings["sqlite"]["path"] and
                 parsed.get("availability",{}).get("cuda_kernels") is True and
                 parsed.get("availability",{}).get("workflow_nvtx") is True, "bound CUDA/NVTX tables unavailable")
        phases = parsed["phases"]
        stages = [p for p in phases if p.get("name") == "P4/stage_pair"]
        _require(len(stages) == 1 and stages[0].get("label") == capture_range(frozen) and
                 stages[0].get("metadata") == expected["metadata"], "actual captured stage differs")
        _require(len(parsed["arms"]) == 1 and parsed["arms"][0]["name"] == expected["arm"],
                 "parsed arm differs from the single captured pair")
        arm = parsed["arms"][0]
        members = [p for p in phases if p.get("included") is True and p.get("arm") == expected["arm"]]
        ids = [p["id"] for p in members]
        _require(len(ids) == len(set(ids)) and set(arm["phase_ids"]) == set(ids) and
                 len(arm["phase_ids"]) == len(ids), "captured phase ownership differs")
        _require(all(p.get("metadata_error") is None and p.get("cross_thread") is False and
                     all(p["metadata"].get(k) == v for k,v in expected["metadata"].items()) for p in members),
                 "phase metadata or launch thread differs")
        dit = [p for p in members if p["name"] == "P4/dit_step" and p["metadata"].get("role") == "A"]
        vae = [p for p in members if p["name"] == "P4/vae_decode" and p["metadata"].get("role") == "B"]
        _require(len(dit) == 20 and sorted(p["metadata"].get("step") for p in dit) == list(range(20)) and
                 len(vae) == 1, "captured DiT/VAE leaf coverage differs")
        stage = stages[0]
        _require(all(type(p.get("start_ns")) is int and type(p.get("end_ns")) is int and
                     stage["start_ns"] <= p["start_ns"] < p["end_ns"] <= stage["end_ns"] for p in members),
                 "captured leaves lie outside the stage")
        _require(parsed.get("counts",{}).get("cuda_operations_correlated_to_workflow",0) > 0 and
                 len(arm["devices"]) == 1, "same-device launch-correlated CUDA operations unavailable")
        intersections = []
        basis = "same-device launch-correlated compute kernel interval intersection; exact dit_step/A and vae_decode/B leaves"
        for device,data in arm["devices"].items():
            overlap, dit_ms, vae_ms = (data.get(key) for key in
                ("dit_vae_kernel_overlap_ms", "dit_a_kernel_union_ms", "vae_b_kernel_union_ms"))
            _require(data.get("dit_vae_kernel_overlap_available") is True and
                     data.get("dit_vae_kernel_overlap_basis") == basis and _number(overlap) and
                     _number(dit_ms,positive=True) and _number(vae_ms,positive=True) and
                     0 <= overlap <= min(dit_ms,vae_ms), "role-correlated kernel intersection unavailable or inconsistent")
            intersections.append(dict(arm=arm["name"],device=device,overlap_ms=overlap,basis=basis))
        value.update(available=True,parsed=binding,capture=bindings["capture_binding"],sqlite=bindings["sqlite"],
                     intersections=intersections,
                     cuda_overlap_proven=any(x["overlap_ms"]>0 for x in intersections))
    except ERRORS as error:
        value["reason"] = str(error)
    return value


def summarize(study):
    from . import workflow_sp_comparison as sp
    from . import workflow_overlap_performance as performance
    from .workflow_remaining_study import UNAVAILABLE
    out = dict(schema=SCHEMA,default_change=False,online=online.summarize(study),jobs=[],cpu_depth={},sp_comparison={},
        unavailable=copy.deepcopy(UNAVAILABLE),
        scope="Technical qualification, per-seed safety and measured benefit remain separate; no automatic promotion.",
        dispatch_refresh_bridge=dict(seeds=[4301,4302,4303],screen_seed=4300,screen_failure_preserved=True,
                                     original_screen=copy.deepcopy(study.get("selections",{}).get("dispatch_refresh_screening")),per_seed=[],
                                     promotion=False,scope="supplemental confirmations do not override the failed original screen"))
    for seed in (4301,4302,4303):
        row = dict(seed=seed,job_id=f"wan-dispatch-refresh-s{seed}",status="not_started",qualified=False,capacity=False)
        try:
            job = report.current_job(study,row["job_id"])
            ev = report.evidence(study,job)
            row.update(status=ev.get("launch",{}).get("status","not_started"),
                       qualified=bool(ev.get("qualified")),capacity=bool(ev.get("capacity")),guarded=bool(ev.get("guarded")))
        except ERRORS as error:
            row["errors"] = [str(error)]
        out["dispatch_refresh_bridge"]["per_seed"].append(row)
    for job in study["jobs"]:
        role = job.get("metadata",{}).get("role","")
        if job.get("superseded") or not role.startswith("remaining_") or role=="remaining_online":
            continue
        row = dict(job_id=job["id"],role=role,seed=job["metadata"].get("seed"),qualified=False,
                   status="not_started",capacity=False,performance_evidence=False)
        try:
            ev = report.evidence(study,job)
            row.update(status=ev.get("launch",{}).get("status","not_started"),capacity=bool(ev.get("capacity")))
            if row["status"]=="completed":
                audit = ev.get("remaining_audit") or audit_result(ev.get("result",{}),job,api.read(job["input"],{}))
                gate = ev.get("remaining_prerequisites") or prerequisite_gate(study,job)
                row.update(audit=audit,prerequisites=gate,qualified=bool(ev.get("qualified") and gate["passed"] and audit["passed"]
                    and (row["capacity"] or role=="remaining_vae_lifecycle_bundle")))
                row["performance_evidence"] = bool(row["qualified"] and audit.get("performance_evidence"))
                if audit["passed"] and role=="remaining_cpu_depth":
                    row.update(_cpu_metrics(ev["result"],audit))
                if row["performance_evidence"] and role=="remaining_sp_comparison":
                    row["timing"] = sp.summarize(ev["result"])
                if row["performance_evidence"] and role=="remaining_vae_performance":
                    row["timing"] = performance._statistics(ev["result"])
                if row["qualified"] and role=="remaining_vae_trace":
                    row["cuda_trace"] = _trace_summary(study,job)
        except ERRORS as error:
            row["errors"] = [str(error)]
        out["jobs"].append(row)
    cpu = [r for r in out["jobs"] if r["role"]=="remaining_cpu_depth"]
    complete = len(cpu)==3 and {r["seed"] for r in cpu}==set(CPU_SEEDS) and all(r["qualified"] for r in cpu)
    out["cpu_depth"] = dict(qualified=complete,seeds=list(CPU_SEEDS),default_change=False,benefit_threshold=None,
        scope="three fresh independent seeds; video-only, same eight CPU cores; descriptive paired changes",
        safety_passed=bool(complete and all(s["passed"] for r in cpu for s in r["safety"].values())),
        paired_changes_median=encoding._median_tree([r["paired_changes"] for r in cpu]) if complete else None,
        absolute_arm_medians=encoding._median_tree([r["arms"] for r in cpu]) if complete else None)
    rows = [r for r in out["jobs"] if r["role"]=="remaining_sp_comparison"]
    complete = len(rows)==3 and {r["seed"] for r in rows}==set(sp.SEEDS) and all(r["qualified"] for r in rows)
    out["sp_comparison"] = dict(qualified=complete,seeds=list(sp.SEEDS),timing_scopes=sp.TIMING_SCOPES,
        scope="single-request timings; three technical repeats within each of three independent seeds; no dual-SP1 throughput or online SLO claim",by_case=[])
    if complete:
        cases = [c for r in rows for c in r["timing"]["by_case"]]
        for frames in (17,33):
            for scope in sp.SCOPES:
                selected = [c for c in cases if c["frames"]==frames and c["scope"]==scope]
                out["sp_comparison"]["by_case"].append(dict(frames=frames,scope=scope,
                    absolute_arm_medians=encoding._median_tree([c["median_wall_ms"] for c in selected]),
                    paired_reduction_median_percent=encoding._median_tree([c["median_wall_reduction_percent"] for c in selected])))
    return out
