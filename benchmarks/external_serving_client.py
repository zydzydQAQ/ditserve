"""Open-loop replay of a sealed external workload, without a system scheduler.

Adapters are initialized before replay. Their async execute(request, context)
MUST yield during I/O/polling and return only after the client has a verified,
fully decoded output (or a terminal failure). A blocking model/HTTP call belongs
in the adapter's independently managed thread/process, never on this event loop.
No client semaphore or previous-completion dependency is introduced.

A completion receipt echoes request_id/source_request_sha256 and supplies status,
executed_spec_sha256, model_manifest_sha256 and the external_comparison output
schema. The client owns every timestamp: no server clock or latency subtraction.
For HTTP this includes receiving/decoding the response; for a runner it includes
observing/retrieving the decoded result. Hash/finite checks precede the receipt.
This boundary is conservative and includes transport/validation overhead.

GF's original EDF reanchors (deadline-arrival) at submission. An adapter must
use context.remaining_budget_ms() immediately at that native submit boundary,
with native arrival=0 and deadline=remaining budget, while preserving the frozen
request and context unchanged. Zero budget is a rejected/expired offered request,
never a reset SLO. Actual native absolute-deadline mapping still needs validation.

This module has no model, HTTP implementation, GPU launcher or qualification
promotion. The optional injected clock is explicitly labelled test-only. Real
runs require the caller's process/storage watchdog; cancellation of a client
coroutine never proves cancellation of server work.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import copy
import math
from pathlib import Path
import time
from typing import Callable, Protocol
import uuid

import external_comparison as common

ADAPTER_TERMINAL = {'completed', 'failed', 'rejected', 'cancelled', 'timed_out'}
RECEIPT_FIELDS = {'request_id', 'source_request_sha256', 'status',
                  'executed_spec_sha256', 'model_manifest_sha256', 'output',
                  'error', 'native_request_id'}


def offset_ns(milliseconds):
    common.finite(milliseconds, 'clock offset')
    return math.ceil(milliseconds * 1_000_000)


class MonotonicClock:
    def now_ns(self):
        return time.monotonic_ns()

    async def sleep_until_ns(self, target):
        while (remaining := target - self.now_ns()) > 0:
            await asyncio.sleep(remaining / 1_000_000_000)


@dataclass(frozen=True)
class DispatchContext:
    clock_id: str
    origin_monotonic_ns: int
    arrival_ms: float
    deadline_ms: float
    arrival_monotonic_ns: int
    deadline_monotonic_ns: int
    cutoff_monotonic_ns: int
    _now_ns: Callable[[], int] = field(repr=False, compare=False)

    def remaining_budget_ms(self, now_ns=None):
        """Only the remaining absolute budget, including all dispatch/queue lag."""
        now = self._now_ns() if now_ns is None else now_ns
        if type(now) is not int or now < self.origin_monotonic_ns:
            raise ValueError('native submit clock must use the common monotonic domain')
        return max(0., (self.deadline_monotonic_ns - now) / 1_000_000)


class Adapter(Protocol):
    async def execute(self, request: dict, context: DispatchContext) -> dict:
        """Return an identity-bound decoded receipt; never return latent-only success."""
        ...

    async def aclose(self) -> dict:
        """Close owned client resources, returning a small JSON cleanup receipt."""
        ...


def describe(export, drain_ms):
    common.verify_export(export)
    common.finite(drain_ms, 'drain_ms')
    horizon = export['clock_contract']['measurement_horizon_ms']
    return dict(workload_sha256=export['sha256'], offered_requests=len(export['requests']),
                measurement_horizon_ms=horizon, drain_ms=drain_ms,
                observation_cutoff_ms=horizon + drain_ms,
                completion_boundary=common.BOUNDARY,
                deadline_rule='original absolute deadline; late output may complete but never resets SLO',
                drain_rule='explicit extra observation only; throughput denominator stays frozen',
                qualification_eligible=False, allows=[])


def validate_receipt(receipt, request, base, observed_ms):
    if not isinstance(receipt, dict) or set(receipt) - RECEIPT_FIELDS:
        raise ValueError('adapter receipt contains unknown fields or client timestamps')
    if (receipt.get('request_id') != request['request_id']
            or receipt.get('source_request_sha256') != request['source_request_sha256']):
        raise ValueError('adapter receipt belongs to another frozen request')
    if receipt.get('status') not in ADAPTER_TERMINAL:
        raise ValueError('adapter did not return a terminal status')
    row = {**base, **copy.deepcopy(receipt), 'terminal_observed_ms': observed_ms}
    if row['status'] == 'completed':
        row.update(output_ready_ms=observed_ms)
    common.checked_result(row, request, base['clock_id'], observed_ms)
    # Reject unserializable/NaN evidence before it can poison the entire run.
    common.canonical(row)
    return row


def _error(exc):
    return type(exc).__name__ + ': ' + str(exc)[:2000]


async def _bounded_cleanup(tasks, adapter, timeout_seconds):
    for task in tasks:
        if not task.done():
            task.cancel()
    pending = set()
    if tasks:
        done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
        for task in done:
            if not task.cancelled():
                task.exception()  # Consume failure without losing the per-request receipt.
    result = dict(pending_client_tasks=len(pending), server_cancellation_confirmed=False)
    async def close():
        return await adapter.aclose()
    close_task = asyncio.create_task(close())
    done, closing = await asyncio.wait({close_task}, timeout=timeout_seconds)
    if closing:
        close_task.cancel()
        result.update(status='incomplete', close_error='adapter aclose exceeded cleanup bound')
    else:
        try:
            receipt = close_task.result()
            common.canonical(receipt)
            result.update(status='closed' if not pending else 'incomplete', adapter_receipt=receipt)
        except BaseException as exc:
            result.update(status='failed', close_error=_error(exc))
    # A cancellation-resistant adapter requires the caller's process watchdog.
    return result


async def replay(export, adapter: Adapter, *, system, drain_ms, cleanup_timeout_seconds=5., _clock=None):
    """Return one result envelope compatible with external_comparison.account_results.

The entire offered trace is registered before any execute call. Every request
has its own arrival coroutine, so service completions cannot gate future arrivals.
The observation lasts through the fixed horizon even when every request finishes
early. Extra drain is explicit and does not change the metric denominator.

Private _clock injection is for deterministic CPU contracts only. Adapters must
cooperate with cancellation; this library cannot kill service-owned GPU work.
"""
    export = copy.deepcopy(export)
    plan = describe(export, drain_ms)
    if (not isinstance(system, dict) or not isinstance(system.get('id'), str) or not system['id']
            or not common.hex64(system.get('source_manifest_sha256'))):
        raise ValueError('a pinned system source identity is required')
    common.canonical(system)
    common.finite(cleanup_timeout_seconds, 'cleanup timeout', positive=True)
    if cleanup_timeout_seconds > 60:
        raise ValueError('cleanup timeout must be at most 60 seconds')
    if not callable(getattr(adapter, 'execute', None)) or not callable(getattr(adapter, 'aclose', None)):
        raise ValueError('adapter must implement async execute and aclose')
    clock = MonotonicClock() if _clock is None else _clock
    origin = clock.now_ns()
    if type(origin) is not int or origin < 0:
        raise ValueError('invalid monotonic origin')
    clock_id = 'external-client-' + uuid.uuid4().hex
    cutoff = origin + offset_ns(plan['observation_cutoff_ms'])
    rows = [dict(request_id=request['request_id'], source_request_sha256=request['source_request_sha256'],
                 arrival_ms=request['arrival_ms'], deadline_ms=request['deadline_ms'],
                 clock_id=clock_id, time_reference='client_origin_ms', status='not_executed')
            for request in export['requests']]
    accepting = True
    controller_error = None
    tasks = []

    def elapsed():
        now = clock.now_ns()
        if type(now) is not int or now < origin:
            raise ValueError('common monotonic clock moved before origin')
        return (now - origin) / 1_000_000

    async def one(index, frozen):
        row = rows[index]
        context = DispatchContext(clock_id, origin, frozen['arrival_ms'], frozen['deadline_ms'],
            origin + offset_ns(frozen['arrival_ms']), origin + offset_ns(frozen['deadline_ms']), cutoff, clock.now_ns)
        try:
            await clock.sleep_until_ns(context.arrival_monotonic_ns)
            if not accepting or clock.now_ns() >= cutoff:
                return
            row.update(status='incomplete', dispatch_ms=elapsed())
            if row['dispatch_ms'] < frozen['arrival_ms']:
                raise ValueError('client dispatched before the frozen arrival')
            request = copy.deepcopy(frozen)
            receipt = await adapter.execute(request, context)
            observed = elapsed()
            if request != frozen:
                raise ValueError('adapter mutated the frozen request')
            if not accepting or observed > plan['observation_cutoff_ms']:
                return
            validated = validate_receipt(receipt, frozen, row, observed)
            row.clear(); row.update(validated)
        except asyncio.CancelledError:
            raise  # Controller records timeout; never infer server cancellation.
        except Exception as exc:
            if accepting:
                row.update(status='failed', error=_error(exc), terminal_observed_ms=elapsed())

    try:
        tasks = [asyncio.create_task(one(i, request), name='external-offer-' + str(i))
                 for i, request in enumerate(export['requests'])]
        await clock.sleep_until_ns(cutoff)
        # Let already-due arrival/completion callbacks be observed once, without
        # waiting on a service or extending the declared observation cutoff.
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        controller_error = 'client replay externally cancelled'
    except Exception as exc:
        controller_error = _error(exc)
    finally:
        accepting = False
        observed_until = elapsed()
        for row in rows:
            if row['status'] == 'incomplete':
                row.update(status='cancelled' if controller_error else 'timed_out',
                           error=controller_error or 'common observation cutoff reached',
                           terminal_observed_ms=observed_until)
            elif row['status'] == 'not_executed':
                row['error'] = controller_error or 'arrival coroutine was not dispatched before common cutoff'
        cleanup = await _bounded_cleanup(tasks, adapter, cleanup_timeout_seconds)
    result = dict(schema=common.RESULT_SCHEMA, workload_sha256=export['sha256'],
        completion_boundary=common.BOUNDARY, system=copy.deepcopy(system),
        client_clock=dict(id=clock_id, origin_monotonic_ns=origin, time_reference='client_origin_ms'),
        observed_until_ms=observed_until, requests=rows,
        replay_plan=plan, cleanup=cleanup, controller_error=controller_error,
        adapter_executions_started=sum('dispatch_ms' in row for row in rows),
        execution_performed=_clock is None and any('dispatch_ms' in row for row in rows),
        timing_evidence='client_monotonic_observation' if _clock is None else 'test_clock_no_performance_evidence',
        qualification_eligible=False, allows=[], quality_validated=False,
        transport_breakdown_measured=False,
        scope='open-loop client observations only; source/model correctness, calibrated load and full-system qualification require independent validation')
    return common.seal(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit'])
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--workload-sha256', required=True)
    parser.add_argument('--drain-ms', type=float, required=True)
    args = parser.parse_args(argv)
    export, binding = common.load_bound(args.workload, args.workload_sha256)
    result = describe(export, args.drain_ms)
    result.update(raw_input_binding=binding, execution_performed=False)
    import json
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
