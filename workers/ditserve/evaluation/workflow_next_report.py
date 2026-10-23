"""Read-only evidence evaluation and compact, reproducible run03 reports."""

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import statistics

from .workflow_budget import atomic_json
from .workflow_next_study import read
from .workflow_serving_study import binding, digest


def current_job(study, logical):
    rows = [row for row in study["jobs"] if row.get("logical_id", row["id"]) == logical and not row.get("superseded")]
    if len(rows) != 1:
        raise ValueError("expected one current job: " + logical)
    return rows[0]


def evidence(study, job):
    launch = read(Path(study["root"]) / "launches" / job["id"] / "launch.json", {})
    path = Path(job["output"]) / "run.json"
    result = read(path, {})
    monitor = launch.get("gpu_monitor", {})
    guarded = (launch.get("status") == "completed" and path.exists()
               and hashlib.sha256(path.read_bytes()).hexdigest() == launch.get("output_sha256")
               and not monitor.get("errors") and not monitor.get("conflicts"))
    memory = monitor.get("memory", {})
    capacity = bool(job["devices"]) and all(
        str(device) in memory and memory[str(device)]["samples"] >= 2
        and memory[str(device)]["minimum_free_bytes"] >= max(1024 ** 3, memory[str(device)]["total_bytes"] * .05)
        for device in job["devices"])
    qualified = guarded and result.get("status") == "completed"
    expected_schema = {
        "sp2_scratch_cost_ablation": "ditserve.workflow.sp2_performance.v1",
        "isolated_shared_decode_performance": "ditserve.workflow.overlap-performance.v1",
    }.get(job.get("metadata", {}).get("role"))
    if expected_schema is not None:
        frozen = read(job["input"]) if isinstance(job.get("input"), str) else None
        qualified = (qualified and result.get("schema") == expected_schema
                     and isinstance(frozen, dict)
                     and digest(frozen) == job.get("configuration_sha256")
                     and result.get("configuration") == frozen)
    q = result.get("qualification") or {}
    if isinstance(q, dict) and any(q.get(k) is False for k in ("passed", "qualified", "finite_and_shape", "repeated_exact")):
        qualified = False
    for phase in result.get("phases", []):
        pq = phase.get("qualification") or {}
        no_configured_gates = pq.get("status") == "unverified" and not pq.get("checks")
        if phase.get("status") != "completed" or (pq.get("qualified") is False and not no_configured_gates):
            qualified = False
    if result.get("schema") in ("ditserve.workflow.elastic_validation.v1", "ditserve.workflow.serving.sp.v1"):
        qualified = qualified and q.get("passed") is True
    if result.get("schema") == "ditserve.workflow.elastic_validation.v1":
        from .workflow_elastic_validation import STAGE_GATES
        ranks = result.get("ranks", [])
        config = result.get("configuration", {})
        expected = {(seed, frame) for seed in config.get("seeds", []) for frame in config.get("frames", [])}
        gates = STAGE_GATES.get(config.get("stage", "all"), ())
        qualified = qualified and len(ranks) == 2 and all(
            rank.get("status") == "completed" and not rank.get("unrun_cases") and
            rank.get("qualification", {}).get("passed") is True and
            bool(expected) and len(rank.get("cases", [])) == len(expected) and
            {(case.get("seed"), case.get("frames")) for case in rank.get("cases", [])} == expected and
            bool(gates) and all(case.get("passed") is True and case.get("cross_card_sp1_exact") is True and
                [gate.get("gate") for gate in case.get("gates", [])] == list(gates) and
                all(gate.get("passed") is True for gate in case["gates"]) for case in rank["cases"])
            for rank in ranks)
        if config.get("sp_linear_policy", "native") != "native":
            from .workflow_elastic_validation import validate_repaired_result
            qualified = qualified and validate_repaired_result(result)["passed"]
    if result.get("schema") == "ditserve.workflow.serving.sp.v1":
        config = result.get("configuration", {})
        expected = {(seed, frame, mode) for seed in config.get("seeds", [config.get("seed")])
                    for frame in config.get("frames", []) for mode in config.get("variants", [])}
        ranks = result.get("ranks", [])
        qualified = qualified and len(ranks) == 2 and bool(expected) and all(
            rank.get("status") == "completed" and len(rank.get("variants", [])) == len(expected) and
            {(row.get("seed", config.get("seed")), row.get("frames"), row.get("mode"))
             for row in rank.get("variants", [])} == expected and
            all(row.get("status") == "qualified" and
                [prefix.get("prefix") for prefix in row.get("prefixes", [])] == [1, 4, 20] and
                all(prefix.get("passed") is True for prefix in row["prefixes"])
                for row in rank["variants"]) for rank in ranks)
    if result.get("schema") == "ditserve.workflow.sp2_performance.v1":
        from .workflow_sp2_performance import validate_result
        qualified = qualified and validate_result(result)["passed"]
    if result.get("schema") == "ditserve.workflow.overlap-performance.v1":
        from .workflow_overlap_performance import validate_result
        qualified = qualified and validate_result(result)["passed"]
    if job.get("metadata", {}).get("comparison") == "adaptive-quantum":
        from .workflow_adaptive_study import feedback_audit
        frozen = (read(job["input"], {}) if isinstance(job.get("input"), str) else {})
        expected_phases = {phase["id"]: phase for phase in frozen.get("phases", [])}
        observed_phases = {phase["id"]: phase for phase in result.get("config", {}).get("phases", [])}
        bound = (digest(frozen) == job.get("configuration_sha256") and
                 expected_phases.keys() == observed_phases.keys() and bool(expected_phases) and all(
                     all(expected_phases[name].get(key) == observed_phases[name].get(key)
                         for key in ("options", "profile", "workload", "policy", "record_dispatch_timeline"))
                     for name in expected_phases))
        qualified = qualified and bound and feedback_audit(result)["passed"]
    if job.get("metadata", {}).get("comparison") == "image-prompt-cache":
        from .workflow_image_cache_audit import audit
        frozen = read(job["input"], {}) if isinstance(job.get("input"), str) else {}
        expected_phases = {phase["id"]: phase for phase in frozen.get("phases", [])}
        observed_phases = {phase["id"]: phase for phase in result.get("config", {}).get("phases", [])}
        bound = (result.get("schema") == "ditserve.run.v1" and
                 digest(frozen) == job.get("configuration_sha256") and
                 expected_phases.keys() == observed_phases.keys() == {"reference", "candidate"} and
                 frozen.get("phases") == result.get("config", {}).get("phases"))
        qualified = qualified and bound and audit(result)["passed"]
    encoding_audit = None
    role = job.get("metadata", {}).get("role")
    if role in ("encoding_numeric", "encoding_online", "dispatch_refresh_online", "occupancy_online", "cache_combination_online"):
        from .workflow_encoding_optimization_report import audit_result, raw_capacity
        if role == "dispatch_refresh_online":
            from .workflow_dispatch_refresh_report import audit_result
        if role == "occupancy_online":
            from .workflow_occupancy_report import audit_result
        if role == "cache_combination_online":
            from .workflow_cache_combination_report import audit_result
        frozen = read(job["input"], {}) if isinstance(job.get("input"), str) else {}
        encoding_audit = audit_result(result, job, frozen)
        observed_capacity = raw_capacity(launch, job)
        encoding_audit = dict(encoding_audit, launcher_capacity=observed_capacity)
        if role in ("dispatch_refresh_online", "occupancy_online", "cache_combination_online"):
            affinity_ok = (launch.get("observer_cpu_affinity") == [58]
                           and job.get("metadata", {}).get("monitor_cpu_affinity") == [58])
            encoding_audit["observer_affinity_verified"] = affinity_ok
            qualified = qualified and affinity_ok
        qualified = (qualified and digest(frozen) == job.get("configuration_sha256")
                     and encoding_audit["passed"] and observed_capacity["passed"])
        if role in ("encoding_online", "dispatch_refresh_online", "occupancy_online", "cache_combination_online"):
            dependencies = {}
            for name in job.get("requires", []):
                try:
                    dependency = evidence(study, current_job(study, name))
                    dependencies[name] = dependency["qualified"] and dependency["capacity"]
                except ValueError:
                    dependencies[name] = False
            qualified = qualified and bool(dependencies) and all(dependencies.values())
            encoding_audit = dict(encoding_audit, dependencies=dependencies)
    value = dict(launch=launch, result=result, guarded=guarded, qualified=qualified,
                 capacity=capacity, path=path)
    if job.get("metadata", {}).get("remaining_validation"):
        from .workflow_remaining_report import audit_result, prerequisite_gate
        from .workflow_encoding_optimization_report import raw_capacity
        frozen = read(job["input"], {}) if isinstance(job.get("input"), str) else {}
        audit = audit_result(result, job, frozen)
        observed_capacity = raw_capacity(launch, job)
        prerequisites = prerequisite_gate(study, job)
        affinity_ok = (launch.get("observer_cpu_affinity") == [58]
                       and job.get("metadata", {}).get("monitor_cpu_affinity") == [58])
        value["qualified"] = bool(qualified and audit["passed"] and affinity_ok
            and digest(frozen) == job.get("configuration_sha256") and prerequisites["passed"])
        # Capacity remains independent: a lifecycle bundle may contain an
        # explicitly failed diagnostic arm and a qualified per-case subset.
        value["capacity"] = bool(capacity and observed_capacity["passed"])
        value["remaining_audit"] = dict(audit, observer_affinity_verified=affinity_ok,
                                        launcher_capacity=observed_capacity)
        value["remaining_prerequisites"] = prerequisites
    if encoding_audit is not None:
        value["encoding_audit"] = encoding_audit
    return value


def performance_qualified(ev, job):
    result = ev["result"]
    diagnostic = result.get("schema") in (
        "ditserve.workflow.elastic_validation.v1", "ditserve.workflow.serving.sp.v1")
    return (ev["qualified"] and ev["capacity"] and not diagnostic
            and result.get("performance_evidence") is not False
            and not job.get("profiling") and not result.get("memory_diagnostics"))


def numeric_status(ev):
    if not ev["guarded"]:
        return "unverified"
    if ev["result"].get("status") == "capacity_failed":
        # An unrun overlap arm cannot fail (or pass) the numerical comparison.
        return "unverified"
    qualifications = [ev["result"].get("qualification") or {}] + [
        phase.get("qualification") or {} for phase in ev["result"].get("phases", [])]
    states = [q["scopes"].get("numeric", {}).get("status", "unverified")
              for q in qualifications if "scopes" in q]
    unscoped = [q for q in qualifications if "scopes" not in q]
    if "failed" in states or any(any(q.get(name) is False for name in
               ("passed", "qualified", "finite_and_shape", "repeated_exact")) for q in unscoped):
        return "failed"
    if ev["qualified"] and ("passed" in states or any(q.get("passed") is True or q.get("qualified") is True
                                                    for q in unscoped)):
        return "passed"
    return "unverified"


def compact_diagnostic(value):
    """Keep numerical evidence, not full event/NVML sample streams, in reports."""
    omitted = {"samples", "nvml_samples", "phase_records", "phases", "events"}
    if isinstance(value, dict):
        result = {key: (compact_intervention_audit(child) if key == "intervention_audit"
                        and isinstance(child, dict) else compact_diagnostic(child))
                  for key, child in value.items() if key not in omitted}
        ranks = value.get("ranks")
        if (value.get("schema") == "ditserve.workflow.serving.sp.v1"
                and isinstance(value.get("variants"), list) and isinstance(ranks, list) and ranks
                and isinstance(ranks[0], dict) and value["variants"] == ranks[0].get("variants")):
            result.pop("variants")
            result["variants_reference"] = dict(json_path="$.ranks[0].variants",
                scope="this diagnostic evidence object", reason="Exact duplicate of rank0 variants.")
        return result
    if isinstance(value, list):
        return [compact_diagnostic(child) for child in value]
    return value


def compact_intervention_audit(audit):
    """Summarize repeated geometry; the bound source retains every shape row."""
    if not isinstance(audit.get("shapes"), list):
        return compact_diagnostic(audit)
    shapes = audit["shapes"]
    actions, modules, invalid = {}, {}, 0
    for row in shapes:
        if not isinstance(row, dict):
            invalid += 1
            continue
        action = str(row.get("action", "<missing>"))
        group = actions.setdefault(action, dict(action=action, records=0, calls=0))
        group["records"] += 1
        calls = row.get("calls")
        if type(calls) is not int or calls < 0:
            invalid += 1
            continue
        group["calls"] += calls
        name = str(row.get("module", "<missing>"))
        modules[name] = modules.get(name, 0) + calls
    canonical = json.dumps(shapes, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode("utf-8")
    result = {key: compact_diagnostic(child) for key, child in audit.items() if key != "shapes"}
    result["shapes_summary"] = dict(records=len(shapes),
        calls=sum(group["calls"] for group in actions.values()),
        actions=[actions[name] for name in sorted(actions)], module_calls=modules,
        invalid_call_records=invalid, canonical_sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_encoding="UTF-8 JSON; sorted keys; separators=(',', ':'); ensure_ascii=False",
        source_field="intervention_audit.shapes",
        details="All original shape records remain in the diagnostic source_result bound artifact.")
    return result


def request_hashes(row):
    outputs = row.get("outputs", [])
    if isinstance(outputs, dict):
        return {name: outputs.get(name, {}).get("sha256", outputs.get(name, {}).get("array_sha256"))
                if outputs.get(name, {}).get("finite") is True else None for name in ("video", "latent")}
    if not outputs:
        return {}
    final = outputs[-1]
    value = final.get("output", {})
    result = dict(video=value.get("sha256", value.get("array_sha256"))) if value.get("finite") is True else {}
    if row.get("kind") == "video":
        latent = final.get("latent", {})
        result["latent"] = latent.get("sha256", latent.get("array_sha256")) if latent.get("finite") is True else None
    return result


def exact_pair(left, right):
    lhs = {row["request_id"]: row for row in left}
    rhs = {row["request_id"]: row for row in right}
    if not lhs or len(lhs) != len(left) or len(rhs) != len(right) or lhs.keys() != rhs.keys():
        return dict(passed=False, reason="incomplete or duplicate request sets")
    mismatches = []
    for key, a in lhs.items():
        b = rhs[key]
        ah, bh = request_hashes(a), request_hashes(b)
        if a.get("status", "completed") != "completed" or b.get("status", "completed") != "completed":
            mismatches.append(key)
        elif not ah or any(not v for v in ah.values()) or ah != bh or a.get("spec") != b.get("spec"):
            mismatches.append(key)
    return dict(passed=not mismatches, requests=len(lhs), mismatches=mismatches)


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def summary(rows, elapsed):
    def one(selected):
        complete = [r for r in selected if r.get("status") == "completed"]
        latencies = [r["finish_ms"] - r["arrival_ms"] for r in complete]
        return dict(requests=len(selected), completed=len(complete),
                    unfinished=len(selected)-len(complete),
                    slo_met=sum(r["finish_ms"] <= r["deadline_ms"] for r in complete),
                    slo_attainment=sum(r["finish_ms"] <= r["deadline_ms"] for r in complete) / len(selected) if selected else None,
                    latency_p50_ms=percentile(latencies, .5), latency_p95_ms=percentile(latencies, .95),
                    maximum_latency_ms=max(latencies) if latencies else None,
                    throughput_per_second=len(complete)*1000/elapsed if elapsed else None)
    return dict(total=one(rows), image=one([r for r in rows if r["kind"] == "image"]),
                video=one([r for r in rows if r["kind"] == "video"]), elapsed_ms=elapsed)


def native_arms(result):
    arms = {}
    for phase in result.get("phases", []):
        cells = phase.get("cells", [])
        if len(cells) == 1:
            arms[phase["id"]] = cells[0]
    return arms


def frozen_request_map(rows):
    """Normalize the request ID added to specs by the native study runner."""
    result = {}
    for row in rows:
        identifier = row["request_id"]
        spec = dict(row["spec"])
        if identifier in result or spec.get("request_id", identifier) != identifier:
            return None
        spec["request_id"] = identifier
        result[identifier] = (row["arrival_ms"], row["deadline_ms"], spec)
    return result


def comparison(result, reference, candidate, *, capacity=True, guarded=True,
               elastic=False, numeric_gate_linkage=None):
    arms = native_arms(result)
    if reference not in arms or candidate not in arms:
        return dict(eligible=False, reason="missing complete comparison arm")
    left, right = arms[reference], arms[candidate]
    a, b = left.get("requests", []), right.get("requests", [])
    exact = exact_pair(a, b)
    left_frozen, right_frozen = frozen_request_map(a), frozen_request_map(b)
    frozen = bool(left_frozen) and left_frozen == right_frozen
    planned = {phase["id"]: phase.get("workload") for phase in result.get("config", {}).get("phases", [])}
    planned_complete = True
    for name, rows in ((reference, a), (candidate, b)):
        if isinstance(planned.get(name), list):
            expected = frozen_request_map(planned[name])
            observed = frozen_request_map(rows)
            planned_complete = planned_complete and bool(expected) and expected == observed
    technical = guarded and frozen and planned_complete and all(
        cell.get("status") == "completed" for cell in (left, right)) and all(
            row.get("status") == "completed" for row in a + b)
    gates_passed = not elastic or bool(numeric_gate_linkage and numeric_gate_linkage.get("passed"))
    eligible = technical and capacity and exact["passed"] and gates_passed
    numeric = dict(exact, status="passed" if exact["passed"] else ("unverified" if elastic else "failed"),
                   method="exact final video and latent hashes")
    if elastic:
        numeric.update(tolerance_status="unverified", tolerance=0.05,
                       tolerance_reason="Online requests retain hashes, not paired arrays; dedicated gates do not prove every online trajectory.",
                       dedicated_gates=numeric_gate_linkage or dict(passed=False, reason="missing gate linkage"))
    return dict(reference=reference, candidate=candidate, eligible=eligible, numeric=numeric,
                same_trajectory=frozen, complete_frozen_requests=planned_complete, capacity=capacity,
                technical_feasibility=technical,
                throughput_evidence="qualified" if eligible else ("observational" if technical else "unverified"),
                reference_summary=summary(a, left.get("elapsed_ms", 0)),
                candidate_summary=summary(b, right.get("elapsed_ms", 0)),
                semantic="unverified", tail_caveat="small-trace p95 is descriptive")


def native_comparison_pairs(result):
    arms = native_arms(result)
    if set(arms) == {"cache_off", "cache_on", "cache_on_refresh"}:
        return [("cache_off", "cache_on"), ("cache_on", "cache_on_refresh"),
                ("cache_off", "cache_on_refresh")]
    if "reference" in arms and "candidate" in arms:
        return [("reference", "candidate")]
    if "disabled" in arms:
        pairs = [("disabled", name) for name in ("staged", "fused", "enabled") if name in arms]
        if "staged" in arms and "fused" in arms:
            pairs.append(("staged", "fused"))
        return pairs
    pairs = [("step", name) for name in ("block", "step-reserved", "block100", "block300")
             if "step" in arms and name in arms]
    if "block100" in arms and "block300" in arms:
        pairs.append(("block300", "block100"))
    pairs.extend((name, "adaptive") for name in ("block300", "block100")
                 if name in arms and "adaptive" in arms)
    return pairs


def elastic_gate_linkage(study):
    rows = []
    for logical in ("sp-diagnostic", "elastic-transfer", "elastic-pause",
                    "elastic-resize", "elastic-interleave"):
        try:
            job = current_job(study, logical)
        except ValueError:
            rows.append(dict(job=logical, passed=False, reason="missing current dedicated gate"))
            continue
        ev = evidence(study, job)
        rows.append(dict(job=job["id"], passed=ev["qualified"], guarded=ev["guarded"],
                         result=str(ev["path"]), sha256=ev["launch"].get("output_sha256"),
                         status=ev["result"].get("status")))
    return dict(passed=all(row["passed"] for row in rows), gates=rows,
                scope="Dedicated two-seed, 17/33-frame gates; separate from online request tolerance.")


def select_quantum(study):
    ev = evidence(study, current_job(study, "quantum-screen"))
    row = comparison(ev["result"], "block300", "block100", capacity=ev["capacity"], guarded=ev["guarded"])
    chosen = 300
    if row["eligible"]:
        a, b = row["reference_summary"], row["candidate_summary"]
        image_gain = b["image"]["slo_attainment"] - a["image"]["slo_attainment"] >= .05
        tail_gain = b["image"]["latency_p95_ms"] <= a["image"]["latency_p95_ms"] * .8
        video_ok = b["video"]["latency_p50_ms"] <= a["video"]["latency_p50_ms"] * 1.05
        throughput_ok = b["total"]["throughput_per_second"] >= a["total"]["throughput_per_second"] * .95
        if (image_gain or tail_gain) and video_ok and throughput_ok:
            chosen = 100
    return dict(quantum_ms=chosen, evidence=row, rule="100 only if image SLO +5pp or p95 -20%, video/throughput regression <=5%; otherwise 300")


def screening_exact(study, *, cross_cores=False):
    pairs = []
    for logical in ("cpu-screen-r0", "cpu-screen-r1"):
        ev = evidence(study, current_job(study, logical))
        arms = {r["id"]: r.get("result", {}) for r in ev["result"].get("arms", [])}
        if not ev["qualified"]:
            return False
        for depth in (0, 1, 2):
            left = arms.get("c8-d1", {}) if not cross_cores else arms.get("c8-d" + str(depth), {})
            right = arms.get(("c16-d" if cross_cores else "c8-d") + str(depth), {})
            pairs.append(exact_pair(left.get("requests", []), right.get("requests", []))["passed"])
    return bool(pairs) and all(pairs)


def build_report(root, *, final=False):
    root = Path(root)
    study = read(root / "study.json")
    phases, failures, sources, requests, comparisons, screening, diagnostics = [], [], [], [], [], [], []
    inputs, contention_observations = {}, []
    for job in study["jobs"]:
        if job.get("superseded"):
            continue
        ev = evidence(study, job)
        result, launch = ev["result"], ev["launch"]
        if job.get("input"):
            configuration = read(job["input"])
            inputs[job["id"]] = dict(configuration=configuration,
                matches_frozen_digest=configuration is not None and digest(configuration) == job["configuration_sha256"],
                environment=read(job["environment_input"]) if job.get("environment_input") else None)
        status = job.get("decision", "pending")
        if launch:
            status = launch.get("status", status)
        phases.append(dict(id=job["id"], family=job["family"], status=status,
            result_status=result.get("status"), capacity="passed" if ev["capacity"] else "unverified_or_failed",
            qualification=dict(scopes=dict(numeric=dict(status=numeric_status(ev)),
                semantic=dict(status="unverified"), performance=dict(
                    status="measured" if performance_qualified(ev, job) else "unverified")))))
        source_result = None
        if ev["path"].exists():
            source_result = dict(binding(ev["path"]), job=job["id"], guarded=ev["guarded"])
            sources.append(source_result)
        if result and not ev["qualified"]:
            failures.append(dict(phase=job["id"], error=result.get("error") or
                                 ("incomplete_or_failed_qualification" if result.get("status") == "completed"
                                  else result.get("status"))))
        if job.get("command_type") == "native":
            for index, row in enumerate(result.get("requests", [])):
                if job.get("metadata", {}).get("remaining_validation"):
                    # Raw timelines/RPCs remain in the SHA-bound source; avoid
                    # duplicating them in the cumulative Excel/canonical JSON.
                    projected = {k: v for k, v in row.items() if k not in
                                 {"dispatch_timeline", "encoding_events", "encoder_rpc", "quanta"}}
                    projected["raw_evidence"] = dict(source=source_result, json_path=f"$.requests[{index}]")
                    requests.append(dict(projected, job=job["id"]))
                else:
                    requests.append(dict(row, job=job["id"]))
            elastic = job.get("metadata", {}).get("comparison") == "elastic"
            linkage = elastic_gate_linkage(study) if elastic else None
            paired = [dict(job=job["id"], metadata=job.get("metadata", {}),
                           **comparison(result, *pair, capacity=ev["capacity"],
                                        guarded=(ev["qualified"] if job.get("metadata", {}).get("comparison")
                                                 in ("adaptive-quantum", "image-prompt-cache")
                                                 or job.get("metadata", {}).get("role") in ("encoding_online", "dispatch_refresh_online", "occupancy_online", "cache_combination_online")
                                                 or job.get("metadata", {}).get("remaining_validation") else ev["guarded"]),
                                        elastic=elastic, numeric_gate_linkage=linkage))
                      for pair in (job["metadata"].get("comparisons", []) if job.get("metadata", {}).get("remaining_validation")
                                   else native_comparison_pairs(result))]
            comparisons.extend(paired)
            if job.get("metadata", {}).get("comparison") in ("contention", "contention-ablation"):
                from .workflow_contention_study import trigger_summary
                contention_observations.append(dict(
                    job=job["id"], guarded=ev["guarded"], capacity=ev["capacity"],
                    observation=trigger_summary(result)))
            if paired:
                scopes = phases[-1]["qualification"]["scopes"]
                scopes["performance"]["status"] = "measured" if all(
                    pair["eligible"] for pair in paired) and performance_qualified(ev, job) else "unverified"
                if elastic:
                    scopes["numeric"].update(
                        status="passed" if all(pair["numeric"]["passed"] for pair in paired)
                        and linkage["passed"] else "unverified",
                        method="exact hashes plus dedicated gates", dedicated_gates=linkage,
                        online_tolerance_status="unverified")
        elif result.get("schema") == "ditserve.workflow.next-block.v1":
            for arm in result.get("arms", []):
                child = arm.get("result") or {}
                screening.append(dict(job=job["id"], arm=arm["id"], metrics=child.get("metrics"),
                    qualification=child.get("qualification"), capacity=ev["capacity"]))
                for row in child.get("requests", []):
                    requests.append(dict(row, job=job["id"], arm=arm["id"],
                        observation_id=job["id"] + "/" + arm["id"],
                        status=row.get("status", "completed"), kind="video"))
        elif result:
            diagnostic = compact_diagnostic(result)
            if job.get("metadata", {}).get("remaining_validation"):
                diagnostic = dict(schema=result.get("schema"), status=result.get("status"),
                    scope="Audited projection; complete raw evidence retained at source_result.",
                    audit=compact_diagnostic(ev.get("remaining_audit", {})),
                    summary=compact_diagnostic(result.get("summary", {})))
            diagnostics.append(dict(job=job["id"], guarded=ev["guarded"], qualified=ev["qualified"],
                                    performance_evidence=performance_qualified(ev, job),
                                    source_result=source_result,
                                    evidence=diagnostic))
    complete = bool(phases) and all(row["status"] == "completed" and row["result_status"] == "completed" for row in phases)
    record = dict(schema="ditserve.run.v1", experiment="workflow", run_id=root.name,
        status="completed" if complete else "partial", created_utc=datetime.now(timezone.utc).isoformat(),
        config=dict(study=study, inputs=inputs, units=dict(time="ms", memory="bytes")),
        metrics=dict(comparisons=comparisons, screening=screening, diagnostics=diagnostics,
            contention_observations=contention_observations,
            selections=study["selections"], sources=sources,
            limitations=["No semantic quality qualification.", "SP1/2 on dual4090 does not establish A100 scaling.",
                         "NVML observed peaks can miss transients.", "Screening and confirmation are separate."]),
        requests=requests, phases=phases, failures=failures, budget=read(root / "budget.json", {}),
        retention=dict(pending_dependencies=["SP numerical diagnosis and downstream quality review"]),
        artifacts=[dict(path=str(Path(row["path"]).relative_to(root)), bytes=row["bytes"], sha256=row["sha256"],
                        kind="control", dependencies=[]) for row in sources])
    if any(job.get("metadata", {}).get("role") == "cache_combination_online" and not job.get("superseded") for job in study["jobs"]):
        from .workflow_cache_combination_report import summarize
        record["metrics"]["cache_combination"] = summarize(study)
    if any(job.get("metadata", {}).get("remaining_validation") for job in study["jobs"]):
        from .workflow_remaining_report import summarize
        record["metrics"]["remaining_validation"] = summarize(study)
    filename = "run.json" if final else "analysis-preview.json"
    atomic_json(root / filename, record)
    def table(headers, rows):
        return "<table><tr>" + "".join("<th>"+html.escape(str(v))+"</th>" for v in headers) + "</tr>" + "".join(
            "<tr>" + "".join("<td>"+html.escape(str(v))+"</td>" for v in row) + "</tr>" for row in rows) + "</table>"
    page = '<!doctype html><meta charset="utf-8"><title>Run03 experiments</title><style>body{font:15px system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px;text-align:left}pre{white-space:pre-wrap}</style>'
    page += "<h1>run03 · " + html.escape(record["status"]) + "</h1>"
    page += "<p>技术执行、数值、容量和语义资格分别评估；旧证据与失败保留。</p>"
    page += table(["任务", "阶段", "执行", "结果", "容量"], [[r[k] for k in ("id", "family", "status", "result_status", "capacity")] for r in phases])
    page += "<h2>完整配对对照</h2><pre>" + html.escape(json.dumps(comparisons, ensure_ascii=False, indent=2)) + "</pre>"
    if "cache_combination" in record["metrics"]:
        page += "<h2>组合缓存：技术资格、服务收益与图像尾延迟</h2><pre>" + html.escape(json.dumps(
            record["metrics"]["cache_combination"], ensure_ascii=False, indent=2)) + "</pre>"
    if contention_observations:
        page += "<h2>实际竞争窗口</h2><pre>" + html.escape(json.dumps(
            contention_observations, ensure_ascii=False, indent=2)) + "</pre>"
    if "remaining_validation" in record["metrics"]:
        page += "<h2>剩余验证：独立资格、性能和未执行范围</h2><pre>" + html.escape(json.dumps(
            record["metrics"]["remaining_validation"], ensure_ascii=False, indent=2)) + "</pre>"
    page += "<h2>数值与容量诊断</h2><pre>" + html.escape(json.dumps(diagnostics, ensure_ascii=False, indent=2)) + "</pre>"
    page += "<h2>预算与选择</h2><pre>" + html.escape(json.dumps(dict(budget=record["budget"], selections=study["selections"]), ensure_ascii=False, indent=2)) + "</pre>"
    (root / ("ANALYSIS.html" if final else "ANALYSIS-LIVE.html")).write_text(page)
    return record
