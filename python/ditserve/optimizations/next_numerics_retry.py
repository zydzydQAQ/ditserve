"""One separately charged numerical replay after the primary study closes.

The unchanged supervisor/worker run only sana_numerics through a private copy
of the supervisor function's globals. No shared globals, old results, sequence,
scientific configuration or ledger are changed; there is no automatic retry.
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
from types import FunctionType

from ..research.common import atomic_json, digest, file_digest
from .next_chain import _success
from .next_finalize import _read, _require, _stable, _process, _process_identity, _identity, _settled
from .next_protocol import CONFIG, AUTHORIZATION, OUTPUT as PARENT_OUTPUT, LEDGER as PARENT_LEDGER, PARENTS, safe_paths
from .protocol import ROOT

OUTPUT = ROOT / "results/optimization-next-numerics-retry/run01"
LEDGER = ROOT / "results/optimization-next-numerics-retry/gpu-budget.json"
FIXED_NUMERICS_SHA = "bc9ab96997c865be087be69745dc9a369dfd7eea236c91c5a8d829461bd6a958"
PROVENANCE_SCOPE = "numerics retry provenance dependencies only; no additional cache-quality evidence"


def _source(path):
    path = Path(path).absolute()
    _require(path.resolve() == path and path.is_file(), "source must be an ordinary nonsymlink file")
    return dict(path=str(path), sha256=file_digest(path))


def _isolated_executor(prepared, parents, implementation):
    from . import next_study
    original = next_study.execute
    def inspect(config, *, observe=True):
        _require(not observe and digest(config) == prepared["config_sha256"], "isolated inspection configuration differs")
        return copy.deepcopy(prepared)
    def freeze(config, output, phase, attempt):
        result = next_study.freeze(config, output, phase, attempt)
        for binding in implementation.values():
            relative = str(Path(binding["path"]).relative_to(ROOT))
            _require(result["source_files"].get(relative) == binding["sha256"], "retry source changed before freezing")
        return result
    namespace = dict(original.__globals__, PHASES=("sana_numerics",), PARENTS=tuple(parents),
                     inspect_study=inspect, freeze=freeze)
    isolated = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    isolated.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    return isolated


def _original_evidence(output, ledger):
    terminal, terminal_binding, unused = _read(output/"sana_numerics/terminal.json")
    native_path = Path(terminal["output"])/"sana-numerics.json"
    _require(terminal.get("evidence_path") == str(native_path), "original numerical evidence path differs")
    native, native_binding, unused = _read(native_path)
    supervisor, supervisor_binding, unused = _read(Path(terminal["output"])/"supervisor.json")
    rows = native.get("rows", [])
    compared = sum(c.get("status") == "completed" for row in rows for c in row.get("comparisons", []))
    failed = [row for row in rows if row.get("status") == "failed"]
    _require(terminal == supervisor and _success(terminal) and len(rows) == 12
        and [row.get("unit_index") for row in rows] == list(range(12))
        and len(failed) == native.get("failed_units") == 8 and compared == native.get("compared_images") == 0
        and all("stride(-1) must be 1" in row.get("error", "") and "got 0" in row["error"] for row in failed),
        "original failure does not match this bounded diagnostic repair")
    return dict(native={k:v for k,v in native_binding.items() if k != "bytes"},
        terminal={k:v for k,v in terminal_binding.items() if k != "bytes"},
        supervisor={k:v for k,v in supervisor_binding.items() if k != "bytes"})


def run_retry(*, chain_pid, chain_started_unix, chain_start_ticks, finalizer_pid, finalizer_start_ticks,
              boot_id, parent_output=PARENT_OUTPUT, parent_ledger=PARENT_LEDGER,
              finalizer_receipt=None, output=OUTPUT, ledger=LEDGER, config_path=CONFIG,
              authorization_path=AUTHORIZATION, wait_seconds=None, wait_idle=3600.,
              process_fn=_process, sleep=time.sleep, clock=time.monotonic,
              config_loader=None, prepare_fn=None, executor_factory=None, report_fn=None):
    """Wait without a GPU lease, bind completed parents, and execute once."""
    from .next_finalize import _ordinary
    from .next_protocol import load_config, authorization
    _require(wait_seconds is None or type(wait_seconds) in (int,float) and math.isfinite(wait_seconds) and wait_seconds >= 0,
             "wait timeout must be finite and nonnegative")
    output, ledger = safe_paths(_ordinary(output), _ordinary(ledger))
    parent_output, parent_ledger = safe_paths(_ordinary(parent_output), _ordinary(parent_ledger))
    _require(output != parent_output and parent_output not in output.parents and output not in parent_output.parents
        and ledger != parent_ledger and ledger.parent != parent_ledger.parent, "retry must use a separate output and ledger tree")
    config_path, authorization_path = _ordinary(config_path), _ordinary(authorization_path)
    config = (config_loader or load_config)(config_path)
    unused, auth = authorization(authorization_path)
    config_source, auth_source = _source(config_path), _source(authorization_path)
    _require(type(chain_pid) is int and type(finalizer_pid) is int and chain_pid > 0 and finalizer_pid > 0
        and len({chain_pid,finalizer_pid,os.getpid()}) == 3 and type(chain_start_ticks) is int and chain_start_ticks > 0
        and type(finalizer_start_ticks) is int and finalizer_start_ticks > 0 and isinstance(boot_id,str) and boot_id,
        "exact separate parent process identities are required")
    finalizer_receipt = _ordinary(finalizer_receipt or parent_output/"audits/finalize-attempt000/receipt.json")
    _require(parent_output/"audits" in finalizer_receipt.parents, "finalizer receipt leaves the parent study")
    expected = dict(chain=dict(pid=chain_pid,start_ticks=chain_start_ticks,boot_id=boot_id),
                    finalizer=dict(pid=finalizer_pid,start_ticks=finalizer_start_ticks,boot_id=boot_id))
    chain, initial_chain_source, unused = _read(parent_output/"chain.json")
    _require(chain.get("pid") == chain_pid and chain.get("started_unix") == chain_started_unix
        and chain.get("output") == str(parent_output) and chain.get("ledger") == str(parent_ledger)
        and chain.get("config") == config_source and chain.get("config_sha256") == digest(config)
        and chain.get("authorization") == auth, "parent chain/config/authorization identity differs")
    chain_identity = _identity(chain)
    implementation = dict(numerics_source=_source(ROOT/"python/ditserve/optimizations/next_numerics.py"),
        numerics_test=_source(ROOT/"tests/python/test_optimizations_next_numerics.py"),
        supervisor_source=_source(ROOT/"python/ditserve/optimizations/next_study.py"), runner_source=_source(Path(__file__)),
        retry_report_source=_source(ROOT/"python/ditserve/optimizations/next_numerics_retry_report.py"),
        retry_report_test=_source(ROOT/"tests/python/test_optimizations_next_numerics_retry_report.py"))
    _require(implementation["numerics_source"]["sha256"] == FIXED_NUMERICS_SHA, "the CPU-validated numerical repair changed")
    output.mkdir(parents=True,exist_ok=True)
    new_lock = (output/"retry.lock").open("a+")
    try:
        fcntl.flock(new_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        new_lock.close()
        raise
    driver_path = output/"retry-driver.json"
    if driver_path.exists() or ledger.exists() or (output/"sana_numerics/terminal.json").exists():
        new_lock.close()
        raise ValueError("retry output was already used; no automatic rerun")
    record = dict(schema_version="ditserve.optimizations.next_numerics_retry_driver.v1",status="waiting",
        pid=os.getpid(),output=str(output),ledger=str(ledger),parent_output=str(parent_output),
        process_identities=expected,chain_started_unix=chain_started_unix,initial_chain=initial_chain_source,
        source=implementation,scope="one diagnostic replay; no performance claim or serving default change",
        latest_user_instruction="继续运行",authorization_source=auth,
        gpu_work_started=False,automatic_retry=False,originals_modified=False)
    atomic_json(driver_path,record)
    parent_lock, started = None, clock()
    try:
        while True:
            for source in (config_source,auth_source,*implementation.values()):
                _require(file_digest(source["path"]) == source["sha256"], "bound configuration/authorization/source changed while waiting")
            chain, chain_source, unused = _read(parent_output/"chain.json")
            _require(_identity(chain) == chain_identity, "parent chain identity changed")
            states = {name:process_fn(identity["pid"]) for name,identity in expected.items()}
            _require(all(state is None or _process_identity(state) == expected[name] for name,state in states.items()),
                     "parent process PID was reused")
            if all(state is None or state.get("alive") is False for state in states.values()):
                break
            if wait_seconds is not None and clock()-started >= wait_seconds:
                raise RuntimeError("parent chain/finalizer wait timed out")
            sleep(5. if wait_seconds is None else min(5.,wait_seconds-(clock()-started)))
        parent_lock = _ordinary(parent_output/"chain.lock").open("a+")
        try:
            fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("parent chain lock remains occupied")
        chain, chain_source, unused = _read(parent_output/"chain.json")
        _require(_identity(chain) == chain_identity, "parent chain changed before admission")
        for name,identity in expected.items():
            state = process_fn(identity["pid"])
            _require(state is None or _process_identity(state) == identity and state.get("alive") is False,
                     "parent process identity/exit changed before admission")
        protected = _settled(parent_output,parent_ledger,chain,digest(config),auth)
        receipt, receipt_source, unused = _read(finalizer_receipt)
        audit, audit_source, unused = _read(parent_output/"audits/final-integrity.json")
        parent_report, parent_report_source, unused = _read(parent_output/"report.json")
        _require(receipt.get("status") == "completed" and receipt.get("pid") == finalizer_pid
            and receipt.get("process_identity") == expected["chain"] and receipt.get("chain_identity") == chain_identity
            and receipt.get("chain_final") == chain_source and receipt.get("audit") == audit_source
            and audit.get("status") == "passed" and audit.get("all_phases_terminal") is True
            and audit.get("output") == str(parent_output) and audit.get("ledger") == str(parent_ledger)
            and audit.get("ledger_source") == protected[0] and audit.get("report_sha256") == parent_report_source["sha256"],
            "parent lacks its exact fresh successful finalizer/audit")
        original = _original_evidence(parent_output,parent_ledger)
        protected += [chain_source,receipt_source,audit_source,parent_report_source]
        parent_binding = _source(parent_ledger)
        parents = [_source(path) for path in PARENTS] + [parent_binding]
        parent_study = dict(chain={k:v for k,v in chain_source.items() if k != "bytes"},
            finalizer_receipt={k:v for k,v in receipt_source.items() if k != "bytes"},
            final_audit={k:v for k,v in audit_source.items() if k != "bytes"},
            report={k:v for k,v in parent_report_source.items() if k != "bytes"})
        binding = dict(schema_version="ditserve.optimizations.next_numerics_retry_binding.v1",scope=PROVENANCE_SCOPE,
            config_sha256=digest(config),authorization=auth,parents=parents,original=original,
            original_failed_units=8,original_compared_images=0,required_units=12,required_generated_images=16,
            required_comparisons=12,parent_study=parent_study,process_identities=expected,
            chain_started_unix=chain_started_unix,implementation=implementation,
            latest_user_instruction="继续运行",
            supervisor_reuse=dict(private_globals_copy=True,shared_globals_modified=False,phase_sequence=["sana_numerics"],
                original_worker="ditserve.optimizations.next_study worker",new_cache_quality_evidence=False))
        binding_path = output/"retry-binding.json"
        atomic_json(binding_path,binding)
        admission = _source(binding_path)
        if prepare_fn is None:
            from .next_study import inspect_study
            prepare_fn = inspect_study
        prepared = copy.deepcopy(prepare_fn(config,observe=False))
        _require(prepared["historical_ledgers"] == parents[:-1], "original three parent bindings differ")
        prepared["historical_ledgers"].append(parent_binding)
        additional = dict(numerics_retry_admission=admission)
        for group,values in (("original",original),("parent",parent_study),("implementation",implementation)):
            additional.update({"numerics_retry_"+group+"_"+name:source for name,source in values.items()})
        current = prepared["cache_local_binding"]["bindings"]
        _require(not set(current).intersection(additional), "retry provenance collides with inherited cache bindings")
        current.update(additional)
        executor = (executor_factory or _isolated_executor)(prepared,[Path(p["path"]) for p in parents],implementation)
        _stable(protected)
        record.update(status="running",retry_binding=admission,protected_parent_sources=protected,
            reason="parents closed; waiting for idle A100 then executing the fixed twelve units")
        atomic_json(driver_path,record)
        result = executor(config,output,ledger,"sana_numerics",authorization_path,wait_idle=wait_idle)
        record["result"] = {key:result.get(key) for key in ("status","reason","output","charge","returncode")}
        record["gpu_work_started"] = bool(result.get("charge"))
        _stable(protected)
        _require(file_digest(parent_ledger) == parent_binding["sha256"], "final parent ledger changed after replay")
        if report_fn is None:
            from .next_numerics_retry_report import write_report
            report_fn = write_report
        record["report"] = report_fn(output,ledger)
        _stable(protected)
        complete = (_success(result) and record["report"].get("integrity_valid") is True
                    and record["report"].get("diagnostic_complete") is True
                    and record["report"].get("verified_comparisons") == 12)
        record.update(status="completed" if complete else "unverified",
            execution_finished=_success(result),diagnostic_complete=bool(complete),
            reason="all twelve comparisons independently verified; diagnosis grants no serving qualification" if complete else
                   "one attempt finished with failed supervision, incomplete comparisons or unverified report integrity",
            local_qualified=False,online_qualified=False,changes_serving_default=False)
    except BaseException as error:
        record.update(status="failed",reason=type(error).__name__+": "+str(error))
        if ledger.exists():
            try:
                budget,unused,unused = _read(ledger)
                record["gpu_work_started"] = bool(budget.get("active") or budget.get("sessions"))
            except Exception:
                record["gpu_work_started"] = None
        # Native failed/partial rows remain intact even if admission/supervision
        # fails. Only a separate reporter may summarize an existing attempt.
        if (output/"retry-binding.json").exists():
            try:
                if report_fn is None:
                    from .next_numerics_retry_report import write_report
                    report_fn = write_report
                record["report"] = report_fn(output,ledger)
            except Exception as reporting_error:
                record["report_error"] = type(reporting_error).__name__+": "+str(reporting_error)
        if isinstance(error,(KeyboardInterrupt,SystemExit)):
            atomic_json(driver_path,record)
            raise
    finally:
        if parent_lock is not None:
            parent_lock.close()
        atomic_json(driver_path,record)
        new_lock.close()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay only fixed Sana numerical diagnosis after exact parent closure")
    for name in ("chain-pid","chain-start-ticks","finalizer-pid","finalizer-start-ticks"):
        parser.add_argument("--"+name,type=int,required=True)
    parser.add_argument("--chain-started-unix",type=float,required=True)
    parser.add_argument("--boot-id",required=True)
    parser.add_argument("--wait-seconds",type=float)
    args = parser.parse_args(argv)
    value = run_retry(**vars(args))
    print(json.dumps({key:value.get(key) for key in ("status","reason","output","report")},ensure_ascii=False),flush=True)
    return 0 if value["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
