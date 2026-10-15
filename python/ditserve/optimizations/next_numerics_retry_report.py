"""One-phase numerical retry report: immutable provenance and explicit failures.

This is a metadata/Excel exporter, not a GPU runner or serving qualification.
No source arrays are loaded, no historical result is rewritten, and nothing is
deleted. A native status of ``passed`` does not imply completed comparisons.
"""
import json
import math
from pathlib import Path
import tempfile

from ..directions.sana import RECIPE
from ..research.common import atomic_json, digest, file_digest
from .next_audit import Reader, _excel, _number, _require
from .next_report import _write_excel


PHASE = "sana_numerics"
SCHEMA = "ditserve.optimizations.next_numerics_retry_binding.v1"
NO_CLAIMS = ("local_qualified", "online_qualified", "changes_serving_default", "performance_claims")


def _path(value):
    path = Path(value).absolute()
    _require(path == path.resolve() and not path.is_symlink(), "retry report paths cannot traverse symlinks")
    return path


def _bound(reader, binding, *, raw=False):
    _require(isinstance(binding, dict) and set(("path", "sha256")) <= set(binding), "missing provenance binding")
    value, source = (reader.raw(binding["path"]) if raw else reader.read(binding["path"]))
    _require(source["sha256"] == binding["sha256"], "bound provenance changed: " + binding["path"])
    return value


def _claims(data):
    _require(all(data.get(key) is False for key in NO_CLAIMS), "numerical diagnosis must explicitly reject serving/performance claims")


def _units(inputs):
    ids = [item["input_id"] for item in inputs]
    result = [("eager_serial", "eager", [key]) for key in ids]
    for seed in (101, 102):
        pair = ["prompt%d-seed%d" % (prompt, seed) for prompt in (0, 1)]
        result.extend((("eager_batch2_forward", "eager", pair),
                       ("eager_batch2_reverse", "eager", list(reversed(pair)))))
    result.extend(("compiled_single", "compiled", [key]) for key in ids)
    return result


def _boundaries():
    names = ["initial.before.generator_state", "initial.latents", "initial.after.generator_state"]
    for step in range(2):
        prefix = "step%d." % step
        names += [prefix + "transformer.input." + key for key in
                  ("hidden_states", "encoder_hidden_states", "encoder_attention_mask", "guidance", "timestep")]
        names += [prefix + "transformer.prediction", prefix + "scheduler.before.generator_state"]
        names += [prefix + "scheduler.input." + key for key in ("prediction", "timestep", "sample")]
        names += [prefix + "scheduler.output.prev_sample", prefix + "scheduler.output.pred_x0",
                  prefix + "scheduler.after.generator_state"]
    return names


def _lpips(metrics):
    # Match the producer's quality_pass flag without treating incomplete
    # per-image metadata as independently verified comparison evidence.
    measured = (metrics.get("status") == "ok" and metrics.get("finite") is True
        and metrics.get("range_valid", True) is True
        and all(metrics.get(key) is None or _number(metrics[key]) for key in ("psnr_db", "temporal_delta_rmse"))
        and all(_number(metrics.get(key)) and metrics[key] >= 0 for key in ("lpips_mean", "lpips_p95")))
    frames = metrics.get("lpips_per_frame")
    valid = (measured and metrics.get("range_valid") is True and metrics.get("lpips_status") == "ok"
        and metrics.get("shape") == [1, 1, 1024, 1024, 3] and metrics.get("frame_count") == 1
        and isinstance(frames, list) and len(frames) == 1 and _number(frames[0]) and frames[0] >= 0
        and all(math.isclose(frames[0], metrics[key], rel_tol=1e-10, abs_tol=1e-12)
                for key in ("lpips_mean", "lpips_p95")))
    return bool(valid), bool(measured and metrics["lpips_mean"] <= .01)


def _numeric(value):
    _require(value.get("rtol") == 1e-5 and value.get("atol") == 1e-6,
             "numeric comparison changed frozen tolerance")
    for flag in ("shape_equal", "dtype_equal", "exact", "passed", "finite"):
        _require(type(value.get(flag)) is bool, "numeric comparison lacks explicit flags")
    for key in ("max_abs", "mean_abs", "rmse", "max_relative"):
        _require(value.get(key) is None or _number(value[key]) and value[key] >= 0,
                 "numeric error summary is invalid")
    _require(not value["passed"] or value["shape_equal"] and value["finite"], "invalid strict comparison claim")


def _comparison(row, unit, arm, label):
    result = dict(attempt=label, unit_index=unit, arm=arm, input_id=row["input_id"],
                  status=row.get("status"), reason=row.get("reason"), compared=False,
                  comparison_valid=False, lpips_passed=False, strict_output_passed=False,
                  trajectory_all_exact=False, trajectory_all_strict=False)
    if row.get("status") == "unavailable":
        _require(bool(row.get("reason")), "unavailable comparison lacks reason")
        return result, []
    _require(row.get("status") == "completed", "comparison status is invalid")
    result["compared"] = True
    trajectory, output = row["trajectory"], row["output"]
    _require(output.get("epsilon") == .01, "output comparison changed LPIPS threshold")
    _numeric(output["numerical"])
    valid, passed = _lpips(output["metrics"])
    _require(output.get("passed") is passed, "LPIPS pass flag differs from recorded metrics")
    components = trajectory.get("components", [])
    names = [item["boundary"] for item in components]
    complete = trajectory.get("complete") is True
    if complete:
        _require(names == _boundaries(), "complete trajectory lacks fixed captured boundary sequence")
    else:
        _require(len(names) == len(set(names)) and set(names) <= set(_boundaries()), "invalid partial boundary list")
    boundaries = []
    for item in components:
        number = item["numerical"]
        _numeric(number)
        reference, candidate = item["reference"], item["candidate"]
        _require(all(isinstance(v.get("sha256"), str) and len(v["sha256"]) == 64 for v in (reference, candidate)),
                 "boundary tensor hash missing")
        boundaries.append(dict(attempt=label, unit_index=unit, arm=arm, input_id=row["input_id"],
            boundary=item["boundary"], **number, reference_sha256=reference["sha256"],
            candidate_sha256=candidate["sha256"], reference_shape=reference.get("shape"),
            candidate_shape=candidate.get("shape"), reference_dtype=reference.get("dtype"),
            candidate_dtype=candidate.get("dtype"), bitwise_sha_equal=reference["sha256"] == candidate["sha256"],
            equality_scope="exact is recorded torch.equal value equality; hashes preserve bytes including signed zero"))
    first_exact = next((x["boundary"] for x in components if not x["numerical"]["exact"]), None)
    first_strict = next((x["boundary"] for x in components if not x["numerical"]["passed"]), None)
    if complete:
        _require(trajectory.get("first_exact_drift") == first_exact and trajectory.get("first_strict_drift") == first_strict
                 and trajectory.get("all_exact") is (first_exact is None)
                 and trajectory.get("all_strict") is (first_strict is None), "trajectory drift summary differs")
    result.update(comparison_valid=bool(complete and valid), lpips_passed=passed,
        lpips_mean=output["metrics"].get("lpips_mean"), lpips_p95=output["metrics"].get("lpips_p95"),
        strict_output_passed=output["numerical"]["passed"], output_exact=output["numerical"]["exact"],
        output_max_abs=output["numerical"]["max_abs"], first_exact_drift=first_exact,
        first_strict_drift=first_strict, trajectory_all_exact=bool(complete and first_exact is None),
        trajectory_all_strict=bool(complete and first_strict is None),
        reference_sha256=output.get("reference_sha256"), candidate_sha256=output.get("candidate_sha256"))
    return result, boundaries


def _matrix(data, inputs, label):
    _claims(data)
    _require(data.get("inputs") == inputs and data.get("recipe") == RECIPE
             and data.get("quality") == dict(rtol=1e-5, atol=1e-6, image_lpips=.01), "numerics recipe/inputs changed")
    _require(data.get("required_units") == 12 and data.get("required_generated_images") == 16,
             "fixed numerical denominator changed")
    expected, rows = _units(inputs), data.get("rows", [])
    _require(isinstance(rows, list) and len(rows) <= 12, "invalid numerical unit count")
    units, comparisons, boundaries, failures = [], [], [], []
    generated, processed = 0, 0
    for index, row in enumerate(rows):
        arm, backend, members = expected[index]
        _require((row.get("unit_index"), row.get("arm"), row.get("backend"), row.get("input_ids")) ==
                 (index, arm, backend, members), "numerical unit identity/order changed")
        _require(row.get("status") in ("completed", "failed", "incomplete"), "unit is not a retained terminal observation")
        execution = row.get("execution", {})
        execution_done = execution.get("generation_completed") is True
        generated += len(members) if execution_done else 0
        processed += len(members) if row.get("generation_completed") is True else 0
        if execution_done:
            _require(execution.get("requested_backend") == backend and execution.get("backend") == backend
                and execution.get("transformer_calls") == 2 and execution.get("scheduler_calls") == 2
                and execution.get("same_cpu_generators_through_scm") is True,
                "completed generation changed backend/steps/RNG evidence")
        seen = set()
        for comparison in row.get("comparisons", []):
            key = comparison.get("input_id")
            _require(arm != "eager_serial" and key in members and key not in seen,
                     "unexpected or duplicate numerical comparison")
            seen.add(key)
            summary, components = _comparison(comparison, index, arm, label)
            comparisons.append(summary)
            boundaries.extend(components)
        units.append(dict(attempt=label, unit_index=index, arm=arm, backend=backend, input_ids=members,
            status=row["status"], pipeline_generation_completed=execution_done,
            boundary_processing_completed=row.get("generation_completed") is True,
            comparison_count=len(seen), error=row.get("error"), generated_images=len(members) if execution_done else 0))
        if row["status"] != "completed":
            failures.append(dict(attempt=label, unit_index=index, arm=arm, input_ids=members,
                                 status=row["status"], reason=row.get("error", "unit did not complete")))
    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    compared = sum(row["compared"] for row in comparisons)
    valid = sum(row["comparison_valid"] for row in comparisons)
    lpips = sum(row["lpips_passed"] for row in comparisons)
    if data.get("diagnostic_complete"):
        _require(len(rows) == 12 and data.get("completed_units") == completed and data.get("failed_units") == failed
                 and data.get("compared_images") == compared and data.get("lpips_passed_images") == lpips,
                 "native completion/counters differ from actual units/comparisons")
    full = (len(rows) == 12 and completed == 12 and generated == 16 and processed == 16
            and valid == 12 and data.get("backend_restored") is True and data.get("active_unit") is None)
    return dict(attempt=label, native_status=data.get("status"), full_comparison=full,
        counts=dict(required_units=12, attempted_units=len(rows), completed_units=completed,
            failed_units=failed, incomplete_units=len(rows)-completed-failed, missing_units=12-len(rows),
            required_generated_images=16, pipeline_generated_images=generated, processed_images=processed,
            required_comparisons=12, recorded_comparisons=len(comparisons), completed_comparisons=compared,
            verified_comparisons=valid, unavailable_comparisons=sum(not row["compared"] for row in comparisons),
            absent_comparisons=12-len(comparisons), unverified_comparisons=12-valid,
            lpips_passed=lpips, strict_output_passed=sum(r["strict_output_passed"] for r in comparisons),
            trajectory_all_strict=sum(r["trajectory_all_strict"] for r in comparisons)),
        units=units, comparisons=comparisons, boundaries=boundaries, failures=failures,
        reason=data.get("reason"), backend_restored=data.get("backend_restored"),
        source_only_metadata=True, serving_qualified=False, performance_qualified=False)


def _process_gone(identity):
    _require(type(identity.get("pid")) is int and identity["pid"] > 0
             and type(identity.get("start_ticks")) is int and isinstance(identity.get("boot_id"), str),
             "parent process identity missing")
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]:
        return True
    path = Path("/proc") / str(identity["pid"]) / "stat"
    if not path.exists():
        return True
    value = path.read_text()
    fields = value[value.rfind(")")+2:].split()
    return fields[0] == "Z" or int(fields[19]) != identity["start_ticks"]


def _admission(reader, output, study, binding):
    _require(study.get("phases") == [PHASE] and study.get("config_sha256") == binding.get("config_sha256"),
             "retry study is not the single frozen numerical phase")
    admission = study["cache_local_binding"]["bindings"]["numerics_retry_admission"]
    _require(_path(admission["path"]) == output/"retry-binding.json", "admission points outside retry output")
    _require(_bound(reader, admission) == binding, "study admission differs")
    _require(study.get("authorization") == binding["authorization"], "retry authorization differs")
    auth = _bound(reader, binding["authorization"])
    _require(auth.get("approved_by_user") is True and auth.get("budget_unlimited") is True
             and auth.get("authorized_gpu_seconds", "missing") is None, "retry is not bound to unlimited authorization")
    parents = binding.get("parents", [])
    _require(len(parents) == 4 and len({item["path"] for item in parents}) == 4, "retry needs all four immutable parent ledgers")
    _require({(x["path"], x["sha256"]) for x in study["historical_ledgers"]} ==
             {(x["path"], x["sha256"]) for x in parents}, "retry parent list differs from study")
    parent_values = []
    for parent in parents:
        value = _bound(reader, parent)
        parent_values.append(value)
        _require("active" in value and value["active"] is None and _number(value.get("used_gpu_seconds"))
                 and value["used_gpu_seconds"] >= 0, "parent ledger is active or invalid")
        _require(_path(parent["path"]).parent not in output.parents, "retry output lies within a parent result tree")
    old = binding["original"]
    native, terminal, supervisor = (_bound(reader, old[key]) for key in ("native", "terminal", "supervisor"))
    _require(terminal == supervisor and terminal.get("phase") == PHASE, "original terminal/supervisor differ")
    _require(sum(session == terminal.get("charge") for value in parent_values for session in value.get("sessions", [])) == 1,
             "original numerical charge does not match exactly one historical ledger session")
    inherited = terminal["charge"]["identity"]["cache_local_binding"]
    admitted = dict(study["cache_local_binding"])
    admitted["bindings"] = {key:value for key,value in admitted["bindings"].items() if not key.startswith("numerics_retry_")}
    _require(admitted == inherited, "numerical retry changed inherited cache binding instead of appending provenance")
    _require(_path(old["native"]["path"]) == _path(terminal["output"])/"sana-numerics.json", "original native path differs")
    inputs = native["inputs"]
    expected_ids = ["prompt%d-seed%d" % (prompt, seed) for prompt in (0, 1) for seed in (101, 102)]
    _require([row.get("input_id") for row in inputs] == expected_ids
             and [row.get("seed") for row in inputs] == [101,102,101,102]
             and all(isinstance(row.get("prompt"), str) for row in inputs), "original fixed input identities differ")
    original = _matrix(native, inputs, "original")
    _require(binding.get("original_failed_units") == original["counts"]["failed_units"] == 8
             and binding.get("original_compared_images") == original["counts"]["completed_comparisons"] == 0
             and binding.get("required_units") == 12 and binding.get("required_generated_images") == 16
             and binding.get("required_comparisons") == 12, "retry does not preserve original failure/denominators")
    parent_study = {key:_bound(reader, value) for key,value in binding["parent_study"].items()}
    chain, receipt, audit = (parent_study[key] for key in ("chain", "finalizer_receipt", "final_audit"))
    allowed_chain = (chain.get("status") == "completed" or chain.get("status") == "needs_attention"
        and chain.get("reason") == "RuntimeError: final independent integrity audit did not pass")
    _require(allowed_chain and receipt.get("status") == "completed" and audit.get("status") == "passed",
             "parent chain/finalizer/integrity audit has not completed")
    identities = binding["process_identities"]
    _require(receipt.get("pid") == identities["finalizer"]["pid"]
             and receipt.get("process_identity") == identities["chain"]
             and all(_process_gone(value) for value in identities.values()), "parent chain/finalizer process is not confirmed exited")
    _require(receipt.get("chain_identity", {}).get("started_unix") == binding.get("chain_started_unix"),
             "parent chain launch identity differs")
    _require(receipt.get("audit", {}).get("sha256") == binding["parent_study"]["final_audit"]["sha256"],
             "finalizer audit binding differs")
    _require(receipt.get("chain_final", {}).get("sha256") == binding["parent_study"]["chain"]["sha256"]
             and audit.get("report_sha256") == binding["parent_study"]["report"]["sha256"],
             "finalizer chain/report binding differs")
    for source in binding["implementation"].values():
        _bound(reader, source, raw=True)
    return inputs, original


def _session(reader, output, ledger, study, binding):
    budget, unused = reader.read(ledger)
    _require(budget.get("stages") == [PHASE] and budget.get("budget_unlimited") is True
             and budget.get("limit_gpu_seconds", "missing") is None, "retry ledger scope/budget differs")
    _require("active" in budget and budget["active"] is None, "retry allocation is still active")
    sessions = budget.get("sessions", [])
    _require(len(sessions) == 1, "retry report requires exactly one settled session")
    session = sessions[0]
    identity = session["identity"]
    _require(session.get("stage") == PHASE and session.get("gpu_count") == 1
             and identity.get("phase") == PHASE and identity.get("run_root") == str(output)
             and identity.get("config_sha256") == binding["config_sha256"], "settled retry identity differs")
    _require(identity.get("cache_local_binding") == study["cache_local_binding"], "settled retry admission differs")
    _require(identity.get("authorization_source") == study["authorization"] == budget.get("authorization_binding"),
             "charged authorization differs")
    parents = {(x["path"], x["sha256"]) for x in binding["parents"]}
    _require({(x["path"], x["sha256"]) for x in budget.get("parents", {}).values()} == parents,
             "charged parent bindings differ")
    start, end, elapsed, charge = (session.get(key) for key in
                                  ("started_monotonic", "ended_monotonic", "elapsed_seconds", "gpu_seconds"))
    _require(all(_number(v) for v in (start,end,elapsed,charge)) and elapsed >= 0
             and elapsed == end-start and charge == elapsed and budget.get("used_gpu_seconds") == charge
             and budget.get("stage_used_gpu_seconds") == {PHASE:charge}, "retry does not charge its whole GPU lifetime")
    attempt = _path(identity["output"])
    _require(attempt.parent == output/PHASE, "retry attempt leaves phase output")
    config, unused = reader.read(attempt/"config.json")
    attempt_study, unused = reader.read(attempt/"study.json")
    _require(digest(config) == binding["config_sha256"] and attempt_study == study, "retry attempt config/study changed")
    workloads, unused = reader.read(output/"workloads.json")
    _require(digest(workloads) == study["workloads_sha256"] == identity["workloads_sha256"], "frozen workload provenance changed")
    manifest_path = output/"sources"/(PHASE+"-"+attempt.name)/"manifest.json"
    manifest, unused = reader.read(manifest_path)
    hashes, root = manifest["source_files"], manifest_path.parent
    _require(bool(hashes) and digest(hashes) == identity["source_sha256"] == manifest["source_sha256"]
             and manifest.get("execution_root") == str(root) and manifest.get("config_sha256") == binding["config_sha256"],
             "retry source manifest changed")
    for name, sha in hashes.items():
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix in (".py", ".json"),
                 "invalid source manifest member")
        raw, source = reader.raw(root/relative, limit=32*1024*1024)
        _require(source["sha256"] == sha, "retry executed source changed: " + name)
    cache = study["cache_local_binding"]
    mathematics = cache["mathematical_source_files"]
    _require(digest(mathematics) == cache["mathematical_source_sha256"]
             and all(hashes.get(name) == sha for name,sha in mathematics.items()), "resident Wan/cache mathematical sources changed")
    for value in cache["bindings"].values():
        _bound(reader, value, raw=True)
    for key, value in binding["implementation"].items():
        relative = ("tests/python/" if key.endswith("_test") else "python/ditserve/optimizations/") + Path(value["path"]).name
        _require(hashes.get(relative) == value["sha256"], "retry implementation differs from executed snapshot: " + key)
    terminal, unused = reader.read(output/PHASE/"terminal.json")
    supervisor, unused = reader.read(attempt/"supervisor.json")
    _require(terminal == supervisor and terminal.get("charge") == session and terminal.get("output") == str(attempt)
             and terminal.get("phase") == PHASE and terminal.get("source_sha256") == identity["source_sha256"]
             and terminal.get("config_sha256") == binding["config_sha256"], "retry terminal differs from charged supervisor")
    _require(terminal.get("evidence_path") == str(attempt/"sana-numerics.json"), "retry native evidence path differs")
    _claims(terminal)
    final = terminal.get("final_occupancy", {})
    _require(final.get("status") == "exclusive_observed" and final.get("owned") == [] and final.get("foreign") == [],
             "retry GPU cleanup unconfirmed")
    devices = [config["gpu_uuid"]]
    _require(identity.get("gpu_uuids") == devices and budget.get("gpu_uuid") == devices[0], "charged GPU identity differs")
    raw, isolation = reader.raw(attempt/"isolation.jsonl")
    observations = [json.loads(line) for line in raw.splitlines() if line.strip()]
    _require(bool(observations) and all(row.get("status") == "exclusive_observed" and row.get("foreign") == []
        and row.get("devices") == devices and row.get("allowed_pgid") == session.get("child_pgid")
        and all(owner.get("pgid") == session["child_pgid"] and owner.get("uuid") in devices for owner in row.get("owned", []))
        for row in observations), "retry isolation observations missing or invalid")
    return dict(session=session, budget=budget, attempt=attempt, config=config, terminal=terminal,
        isolation=dict(source=isolation, observations=len(observations), status="exclusive_observed"),
        completed=bool(session.get("status") == "completed" and terminal.get("returncode") == 0
            and not terminal.get("supervisor_failure") and terminal.get("status") in ("passed", "completed", "stopped")))


def collect(output, ledger):
    output, ledger = _path(output), _path(ledger)
    _require(ledger.name == "gpu-budget.json" and ledger.parent in output.parents,
             "retry report requires its independent enclosing ledger")
    reader = Reader()
    study, unused = reader.read(output/"study.json")
    binding, unused = reader.read(output/"retry-binding.json")
    _require(binding.get("schema_version") == SCHEMA and study.get("phases") == [PHASE],
             "refusing to overwrite a non-retry report directory")
    report = dict(schema_version="ditserve.optimizations.next_numerics_retry_report.v1",
        output=str(output), ledger=str(ledger), status="unverified", decision="未验证", partial=True,
        original=None, retry=None, integrity_errors=[], source_bindings=[], budget=None,
        **{key:False for key in NO_CLAIMS}, originals_deleted=0,
        scope="fixed numerical diagnosis only; first observed drift is not a causal kernel attribution; no serving or speed qualification")
    inputs = None
    # Keep readable original failure counts even when some other admission
    # evidence is missing or no new session was launched. These observations
    # never override the integrity status below.
    try:
        original = _bound(reader, binding["original"]["native"])
        inputs = original["inputs"]
        report["original"] = _matrix(original, inputs, "original")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        report["integrity_errors"].append("original: " + type(error).__name__ + ": " + str(error))
    try:
        inputs, report["original"] = _admission(reader, output, study, binding)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        report["integrity_errors"].append("admission: " + type(error).__name__ + ": " + str(error))
    session = None
    try:
        session = _session(reader, output, ledger, study, binding)
        report["budget"] = dict(gpu_seconds=session["session"]["gpu_seconds"], gpu_count=1,
            session_id=session["session"]["session_id"], status=session["session"]["status"],
            parent_usage_modified=False, accounting_scope="complete loading/compilation/warmup/failure/cleanup lifetime")
        report["isolation"] = session["isolation"]
        report["configuration"] = dict(config_sha256=binding["config_sha256"], recipe=dict(RECIPE),
            inputs=inputs, quality=dict(rtol=1e-5,atol=1e-6,image_lpips=.01),
            models=session["config"].get("models"), GPU=session["config"]["gpu_uuid"],
            compiler_scope="original transformer-only Inductor fullgraph/static/no CUDA Graph; numerical diagnosis only")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        report["integrity_errors"].append("settlement: " + type(error).__name__ + ": " + str(error))
    attempt = session["attempt"] if session is not None else output/PHASE/"attempt00"
    if inputs is not None and (attempt/"sana-numerics.json").exists():
        try:
            native, source = reader.read(attempt/"sana-numerics.json")
            if session is not None:
                _require(native.get("inputs") == [dict(input_id="prompt%d-seed%d" % (index, seed), prompt=prompt, seed=seed)
                    for index,prompt in enumerate(session["config"]["direction1"]["prompts"][:2]) for seed in (101,102)],
                    "retry input prompts differ from frozen configuration")
            for value in native.get("source", {}).values():
                _bound(reader, value, raw=True)
            report["retry"] = dict(_matrix(native, inputs, "retry"), source=source)
            if session is not None and not report["integrity_errors"]:
                if session["completed"] and report["retry"]["full_comparison"]:
                    report.update(status="completed", decision="完整比较已完成；仅用于数值诊断", partial=False)
                elif report["retry"]["counts"]["verified_comparisons"]:
                    report.update(status="partial", decision="部分比较完成；未比较输入仍保留在分母")
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            report["integrity_errors"].append("numerics: " + type(error).__name__ + ": " + str(error))
    summaries = [value for value in (report["original"], report["retry"]) if value is not None]
    report["combined_denominators"] = dict(required_units=24, required_generated_images=32, required_comparisons=24,
        failed_units=sum(v["counts"]["failed_units"] for v in summaries),
        verified_comparisons=sum(v["counts"]["verified_comparisons"] for v in summaries),
        unverified_comparisons=24-sum(v["counts"]["verified_comparisons"] for v in summaries),
        scope="original failed attempt and new retry are distinct observations; retry never overwrites original failure")
    report["source_bindings"] = list(reader.sources.values())
    report["integrity_status"] = "passed" if not report["integrity_errors"] else "unverified"
    report["diagnostic_complete"] = report["status"] == "completed"
    return report


def _tables(report):
    summary = [["field", "value"], ["status", report["status"]], ["decision", report["decision"]],
               ["serving_qualified", False], ["performance_claims", False], ["scope", report["scope"]]]
    tables = dict(Summary=summary,
        Units=[["attempt", "unit_index", "arm", "backend", "input_ids", "status", "pipeline_generation_completed",
                "boundary_processing_completed", "comparison_count", "generated_images", "error"]],
        Comparisons=[["attempt", "unit_index", "arm", "input_id", "status", "comparison_valid", "lpips_mean",
                "lpips_passed", "strict_output_passed", "output_max_abs", "first_exact_drift", "first_strict_drift", "reason"]],
        Boundaries=[["attempt", "unit_index", "arm", "input_id", "boundary", "exact", "passed", "finite", "max_abs",
                "mean_abs", "rmse", "max_relative", "reference_sha256", "candidate_sha256", "bitwise_sha_equal"]],
        Failures=[["attempt", "unit_index", "arm", "input_ids", "status", "reason"]],
        Sources=[["path", "sha256", "bytes"]])
    for label in ("original", "retry"):
        value = report[label]
        if value is None:
            summary.append([label+".available", False])
            continue
        summary += [[label+"."+key, item] for key,item in value["counts"].items()]
        for key in ("units", "comparisons", "boundaries", "failures"):
            target = tables[key.title()]
            target.extend([[json.dumps(row.get(field), ensure_ascii=False, sort_keys=True) if
                            isinstance(row.get(field), (dict, list)) else row.get(field) for field in target[0]]
                           for row in value[key]])
    summary += [["combined."+key,value] for key,value in report["combined_denominators"].items()]
    summary += [["budget."+key,value] for key,value in (report["budget"] or {}).items()]
    summary += [["config."+key,json.dumps(value,ensure_ascii=False,sort_keys=True)]
                for key,value in report.get("configuration", {}).items()]
    summary += [["integrity_error", error] for error in report["integrity_errors"]]
    tables["Sources"] += [[row.get(key) for key in tables["Sources"][0]] for row in report["source_bindings"]]
    return tables


def write_retry_report(output, ledger):
    report = collect(output, ledger)
    output = Path(report["output"])
    tables = _tables(report)
    with tempfile.TemporaryDirectory(prefix=".numerics-retry-report-", dir=str(output)) as temporary:
        excel = Path(temporary)/"results.xlsx"
        verification = _write_excel(excel, tables)
        independent = Reader()
        _excel(excel, tables, independent)
        excel.replace(output/"results.xlsx")
    report["excel"] = dict(path=str(output/"results.xlsx"), sha256=file_digest(output/"results.xlsx"), **verification)
    # Sources must still match after Excel export; a live/changed ledger cannot
    # become a settled result merely because its earlier read was consistent.
    for source in report["source_bindings"]:
        _require(file_digest(source["path"]) == source["sha256"], "source changed during retry report export")
    atomic_json(output/"report.json", report)
    lines = ["数值诊断补跑：" + report["decision"], "", report["scope"], ""]
    for label, title in (("original", "原尝试"), ("retry", "补跑")):
        value = report[label]
        if value is None:
            lines.append(title+"：证据未验证。")
        else:
            count = value["counts"]
            lines.append("%s：尝试 %d/12 单元，失败 %d；有效比较 %d/12，LPIPS 通过 %d/12，未验证 %d/12。" %
                (title,count["attempted_units"],count["failed_units"],count["verified_comparisons"],count["lpips_passed"],count["unverified_comparisons"]))
    lines += ["", "原尝试与补跑合计保留 24 个单元、32 张计划输出、24 次计划比较；任何失败或未比较结果均未删除。",
              "质量阈值：图像 LPIPS≤0.01；严格对照 rtol=1e-5、atol=1e-6。完成诊断不代表通过 serving 验收。"]
    if report["integrity_errors"]:
        lines += ["", "完整性问题："] + ["- "+value for value in report["integrity_errors"]]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    receipt = dict(schema_version="ditserve.optimizations.next_numerics_retry_audit.v1",
        status=report["integrity_status"], diagnostic_status=report["status"],
        report=dict(path=str(output/"report.json"), sha256=file_digest(output/"report.json")),
        excel=report["excel"], sources_verified=len(report["source_bindings"]),
        errors=report["integrity_errors"], serving_qualified=False, performance_qualified=False,
        originals_deleted=0, raw_cleanup_performed=False)
    atomic_json(output/"audit.json", receipt)
    return dict(status=report["status"], decision=report["decision"], partial=report["partial"],
        report=str(output/"report.json"), markdown=str(output/"REPORT.md"), excel=report["excel"],
        audit=str(output/"audit.json"), originals_deleted=0, serving_qualified=False,
        integrity_status=report["integrity_status"], integrity_valid=report["integrity_status"] == "passed",
        diagnostic_complete=report["diagnostic_complete"],
        failed_units=report["retry"]["counts"]["failed_units"] if report["retry"] else None,
        completed_comparisons=report["retry"]["counts"]["completed_comparisons"] if report["retry"] else 0,
        verified_comparisons=report["retry"]["counts"]["verified_comparisons"] if report["retry"] else 0)


write_report = write_retry_report
