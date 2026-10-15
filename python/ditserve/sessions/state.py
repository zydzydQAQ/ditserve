"""Conditional state capacity experiments over the pilot's real session inputs.

This is a capacity diagnostic, not an arrival replay or a throughput claim.
Every held object comes from an actual LLM turn and an actual half-complete
Sana image.  No padding allocations, alternative inputs or model offload are
used to manufacture pressure.  Raw baseline images remain a live dependency.
"""

import gc
from pathlib import Path
import time

import numpy as np

from ..research.common import file_digest, read_json
from ..scheduling.execution import _arrays_equal
from . import workload
from .runtime import tensor_comparison


def snapshot_comparison(before, after):
    """Compare scalar structure and every CPU snapshot tensor exactly."""
    checks = []

    def visit(a, b, name):
        if hasattr(a, "shape") or hasattr(b, "shape"):
            check = tensor_comparison(a, b)
            checks.append(dict(name=name, passed=check["passed"] and check["exact"],
                               exact=check["exact"], max_abs=check["max_abs"]))
        elif isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in a:
                visit(a[key], b[key], name + "." + key)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
            for index, (left, right) in enumerate(zip(a, b)):
                visit(left, right, name + ".%d" % index)
        else:
            checks.append(dict(name=name, passed=type(a) is type(b) and a == b,
                               exact=type(a) is type(b) and a == b, max_abs=None))

    visit(before, after, "state")
    return dict(passed=all(row["passed"] for row in checks), exact=all(row["exact"] for row in checks),
                checked=len(checks), mismatches=[row for row in checks if not row["passed"]])


def _references(config, root, result, costs):
    trace = workload.make_trace(config, costs, config["loads"][0], config["seed"])
    path = root / "pilot/result.json"
    pilot, supervised = read_json(path), read_json(root / "pilot/supervisor.json")
    if (pilot.get("status") != "passed" or supervised.get("status") != "passed"
            or supervised.get("returncode", 0) != 0 or supervised.get("supervisor_failure")):
        raise ValueError("state experiment requires a fully qualified pilot")
    records = [r for r in pilot["records"] if r["arm"] == "session_serial"
               and r["load"] == trace["load"] and r["seed"] == trace["seed"]]
    if len(records) != 1:
        raise ValueError("state experiment requires one matching serial reference")
    reference = records[0]
    if (not reference.get("quality", {}).get("passed")
            or reference.get("trace_sha256") != trace["trace_sha256"]
            or reference.get("device_identity") != result.get("device_identity")
            or reference.get("cost_model_sha256") != result.get("cost_model_sha256")):
        raise ValueError("pilot reference does not match the qualified trace/device/costs")
    rows = {row["request_id"]: row for row in reference["rows"]}
    expected = {turn["request_id"] for session in trace["sessions"] for turn in session["turns"]}
    if len(reference["rows"]) != len(expected) or set(rows) != expected:
        raise ValueError("pilot reference does not contain every declared session turn")
    for row in rows.values():
        source = Path(row["outputs"]["output"])
        if (row.get("status") != "completed" or not row.get("quality", {}).get("passed")
                or not source.is_file() or file_digest(source) != row["quality"].get("output_sha256")):
            raise ValueError("baseline output is absent, changed or unqualified: " + row["request_id"])
    result["reference"] = dict(path=str(path), sha256=file_digest(path), arm="session_serial",
                               load=trace["load"], trace_sha256=trace["trace_sha256"],
                               raw_outputs_required_until_state_phase_complete=True)
    return trace["sessions"], rows


def run_state(config, root, llm, sana, result, save, check, costs):
    """Measure real retained states; try state offload only after observed OOM."""
    from .execution import messages_for, memory

    root = Path(root)
    sessions, baseline = _references(config, root, result, costs)
    counts = config.get("state_capacity_counts", [1, 2, 4, 8, 12])
    if (not counts or any(type(value) is not int or value < 1 or value > len(sessions) for value in counts)
            or counts != sorted(set(counts)) or counts[-1] != len(sessions)):
        raise ValueError("capacity counts must increase through the complete real session population")
    result.update(capacity_trials=[], state_policies=[], capacity_pressure_observed=False,
        scope="real pilot inputs and two-turn dependencies; simultaneous retention diagnostic; arrival/think delays omitted; no online latency or throughput claim")
    held = []
    current_stage = {}
    oom_class = getattr(llm.torch, "OutOfMemoryError", llm.torch.cuda.OutOfMemoryError)

    def mark(session, round_index, action):
        check()
        current_stage.clear()
        current_stage.update(session_id=session["session_id"], round=round_index + 1, action=action)

    def cleanup(item):
        try:
            if item.get("image") is not None:
                sana.release(item["image"])
                item["image"] = None
        finally:
            if item.get("llm") is not None:
                llm.release(item["llm"])
                item["llm"] = None

    def cleanup_all(items):
        errors = []
        for item in list(items):
            try:
                cleanup(item)
            except BaseException as error:
                errors.append(error)
        items.clear()
        if errors:
            raise errors[0]

    def recover_allocator():
        gc.collect()
        llm.torch.cuda.empty_cache()

    def failure(error):
        return dict(error_type=type(error).__name__, reason=str(error),
                    action=dict(current_stage), memory=memory(llm))

    def check_tokens(item, round_index):
        text = llm.finish(item["llm"])
        expected = baseline[item["session"]["turns"][round_index]["request_id"]]
        if item["llm"].generated_ids != expected["llm_token_ids"] or text != expected["llm_text"]:
            result["failed_comparison"] = dict(request_id=expected["request_id"],
                tokens_exact=item["llm"].generated_ids == expected["llm_token_ids"], text_exact=text == expected["llm_text"])
            save()
            raise RuntimeError("capacity experiment changed the real LLM image prompt")
        return text

    def build(index):
        session = sessions[index]
        item = dict(index=index, session=session, llm=None, image=None, first_text=None)
        try:
            mark(session, 0, "llm_prepare")
            item["llm"] = llm.prepare(messages_for(session, 0), max_new_tokens=config["max_new_tokens"],
                                       request_id=session["turns"][0]["request_id"])
            while not item["llm"].done:
                mark(session, 0, item["llm"].action)
                llm.advance(item["llm"])
            item["first_text"] = check_tokens(item, 0)
            mark(session, 0, "image_prepare")
            item["image"] = sana.prepare(item["first_text"], seed=session["turns"][0]["image_seed"],
                                         request_id=session["turns"][0]["request_id"])
            mark(session, 0, "image_step_1")
            sana.advance(item["image"])
            if item["image"].step_index != 1 or item["image"].done:
                raise RuntimeError("capacity state must contain one real unfinished Sana step")
            return item
        except BaseException:
            cleanup(item)
            raise

    def finish_image(item, round_index, rows):
        session, image = item["session"], item["image"]
        while not image.done:
            mark(session, round_index, "image_step_%d" % (image.step_index + 1))
            sana.advance(image)
        mark(session, round_index, "image_decode")
        output = sana.finish(image)
        expected = baseline[session["turns"][round_index]["request_id"]]
        comparison = _arrays_equal(output, np.load(expected["outputs"]["output"], mmap_mode="r", allow_pickle=False))
        row = dict(request_id=expected["request_id"], round=round_index + 1, tokens_exact=True,
                   prompt_exact=True, image=comparison, passed=comparison["passed"], memory=memory(llm))
        rows.append(row)
        if not comparison["passed"]:
            directory = root / "state/failed-arrays"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (expected["request_id"] + "-" + str(time.time_ns()) + ".npy")
            np.save(path, output, allow_pickle=False)
            result["failed_comparison"] = dict(row, output_path=str(path), output_sha256=file_digest(path))
            save()
            raise RuntimeError("capacity experiment changed the Sana output")
        sana.release(image)
        item["image"] = None

    def complete(item, rows):
        finish_image(item, 0, rows)
        session = item["session"]
        mark(session, 1, "llm_prepare")
        item["llm"] = llm.prepare(messages_for(session, 1, item["first_text"]),
            previous=item["llm"], max_new_tokens=config["max_new_tokens"],
            request_id=session["turns"][1]["request_id"])
        while not item["llm"].done:
            mark(session, 1, item["llm"].action)
            llm.advance(item["llm"])
        text = check_tokens(item, 1)
        mark(session, 1, "image_prepare")
        item["image"] = sana.prepare(text, seed=session["turns"][1]["image_seed"],
                                     request_id=session["turns"][1]["request_id"])
        finish_image(item, 1, rows)
        cleanup(item)

    def inventory(items):
        def compact(value):
            return {key: item for key, item in value.items() if key != "tensors"}
        return [dict(session_id=item["session"]["session_id"],
                     llm=compact(llm.inventory(item["llm"])), image=compact(sana.inventory(item["image"]))) for item in items]

    boundary = None
    try:
        save()
        for count in counts:
            trial = dict(count=count, status="running", rows=[], retained=[], started_at=time.time())
            result["capacity_trials"].append(trial)
            started = time.perf_counter()
            try:
                for index in range(count):
                    held.append(build(index))
                    trial["retained"].append(dict(count=len(held), memory=memory(llm)))
                trial["inventory"] = inventory(held)
                trial["held_memory"] = memory(llm)
                save()
                for item in list(held):
                    try:
                        complete(item, trial["rows"])
                    except BaseException:
                        cleanup(item)
                        held.remove(item)
                        raise
                    held.remove(item)
                trial.update(status="passed", passed=True, completed_turns=len(trial["rows"]))
            except oom_class as error:
                boundary = dict(count=count, failure=failure(error),
                    retained_indices=[item["index"] for item in held],
                    target_index=next(i for i, session in enumerate(sessions)
                                      if session["session_id"] == current_stage["session_id"]))
                error.__traceback__ = None
                trial.update(status="capacity_boundary", passed=False, boundary=boundary)
                result.update(capacity_pressure_observed=True, capacity_boundary=boundary)
            except BaseException as error:
                trial.update(status="failed", passed=False, failure=failure(error))
                raise
            finally:
                trial["elapsed_ms"] = (time.perf_counter() - started) * 1000
            save()
            if boundary is not None:
                break
        if boundary is None:
            result.update(status="passed", eligible=True, technical_complete=True,
                          conclusion="not_applicable", reason="no_observed_capacity_pressure",
                          state_policy_status="not_applicable", qualified_retained_sessions=len(sessions))
            save()
            return result

        healthy = boundary["retained_indices"]
        target = boundary["target_index"]
        if not healthy:
            result.update(status="stopped", eligible=False, technical_complete=True,
                conclusion="observed_no_viable_state_policy", reason="no healthy retained states exist to offload")
            save()
            return result
        recover_allocator()
        keep = dict(policy="keep", status="running", initial_boundary=boundary, rows=[],
                    retained_indices=list(healthy), target_index=target,
                    scope="retry the failed real session after failed-state cleanup; no retained-state offload")
        result["state_policies"].append(keep)
        extra, started = None, time.perf_counter()
        try:
            extra = build(target)
            complete(extra, keep["rows"])
            extra = None
            for item in list(held):
                complete(item, keep["rows"])
                held.remove(item)
            keep.update(status="passed", passed=True, completed_turns=len(keep["rows"]))
        except oom_class as error:
            keep.update(status="capacity_failure", passed=False, failure=failure(error))
            error.__traceback__ = None
        except BaseException as error:
            keep.update(status="failed", passed=False, failure=failure(error))
            raise
        finally:
            try:
                if extra is not None:
                    cleanup(extra)
            finally:
                if not keep.get("passed"):
                    cleanup_all(held)
            keep["elapsed_ms"] = (time.perf_counter() - started) * 1000
            recover_allocator()
            save()
        if keep.get("passed"):
            result.update(status="stopped", eligible=False, technical_complete=True,
                conclusion="observed_transient_capacity_failure", state_policy_status="not_applicable",
                reason="keep completed after cleanup; no offload benefit established; larger capacity cohorts remain unmeasured",
                unmeasured_capacity_counts=[count for count in counts if count > boundary["count"]])
            save()
            return result
        policies = ["offload_kv", "offload_image"]
        successes = []
        for policy_index in range(3):
            if policy_index == 2:
                if successes:
                    break
                policies.append("offload_both")
            arm = policies[policy_index]
            trial = dict(policy=arm, status="running", rows=[], transfers=[],
                         retained_indices=list(healthy), target_index=target,
                         scope="capacity rescue of the same failed real session; rebuild/setup and completion time included; not a serving speedup")
            result["state_policies"].append(trial)
            started, extra = time.perf_counter(), None
            try:
                if not held:
                    for index in healthy:
                        held.append(build(index))
                snapshots = []
                for item in held:
                    for component, runtime in (("llm", llm), ("image", sana)):
                        if (component == "llm" and arm == "offload_image") or (component == "image" and arm == "offload_kv"):
                            continue
                        state = item[component]
                        before = runtime.snapshot(state)
                        mark(item["session"], 0, "offload_" + component)
                        cost = runtime.offload_to_cpu(state)
                        comparison = snapshot_comparison(before, runtime.snapshot(state))
                        trial["transfers"].append(dict(session_id=item["session"]["session_id"],
                            component=component, direction="to_cpu", cost=cost, comparison=comparison))
                        if not comparison["passed"]:
                            raise RuntimeError("offload changed a retained session state")
                        snapshots.append((item, component, runtime, before))
                trial["offloaded_memory"] = memory(llm)
                save()
                extra = build(target)
                complete(extra, trial["rows"])
                extra = None
                for item, component, runtime, before in snapshots:
                    mark(item["session"], 0, "restore_" + component)
                    cost = runtime.restore(item[component])
                    comparison = snapshot_comparison(before, runtime.snapshot(item[component]))
                    trial["transfers"].append(dict(session_id=item["session"]["session_id"],
                        component=component, direction="to_gpu", cost=cost, comparison=comparison))
                    if not comparison["passed"]:
                        raise RuntimeError("restore changed a retained session state")
                trial["restored_memory"] = memory(llm)
                for item in list(held):
                    complete(item, trial["rows"])
                    held.remove(item)
                trial.update(status="passed", passed=True, completed_turns=len(trial["rows"]))
                successes.append(arm)
            except oom_class as error:
                trial.update(status="capacity_failure", passed=False, failure=failure(error))
                error.__traceback__ = None
            except BaseException as error:
                trial.update(status="failed", passed=False, failure=failure(error))
                raise
            finally:
                try:
                    if extra is not None:
                        cleanup(extra)
                finally:
                    cleanup_all(held)
                trial["elapsed_ms"] = (time.perf_counter() - started) * 1000
                recover_allocator()
                save()
        result.update(status="passed" if successes else "stopped", eligible=bool(successes),
            technical_complete=True, conclusion="observed_state_capacity_relief" if successes else "observed_no_viable_state_policy",
            successful_state_policies=successes,
            reason="only the declared actual session states and model recipes were tested")
        save()
        return result
    finally:
        cleanup_all(held)
