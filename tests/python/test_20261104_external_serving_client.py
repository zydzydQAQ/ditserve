"""Deterministic CPU contracts only: no model, HTTP service or performance result.

The manual clock and output descriptors below are synthetic test fixtures kept
in memory. They must never be published as external-system execution evidence.
"""
import asyncio
import copy
import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import external_serving_client as client
import external_comparison as comparison
fixtures = importlib.import_module('test_20261103_external_comparison')


class ManualClock:
    def __init__(self):
        self.origin = self.value = 10**12
        self.waiters = []
    def now_ns(self):
        return self.value
    async def sleep_until_ns(self, target):
        if self.value < target:
            future = asyncio.get_running_loop().create_future()
            self.waiters.append((target, future))
            await future
    async def advance(self, milliseconds):
        target = self.origin + client.offset_ns(milliseconds)
        if target < self.value:
            raise ValueError('test clock cannot move backwards')
        self.value = target
        for boundary, future in self.waiters:
            if boundary <= target and not future.done():
                future.set_result(None)
        for _ in range(12):
            await asyncio.sleep(0)


def workload(count=4):
    return fixtures.ExternalComparisonTests().exported(count=count)


def completed(request):
    return dict(request_id=request['request_id'], source_request_sha256=request['source_request_sha256'],
        status='completed', executed_spec_sha256=request['spec_sha256'],
        model_manifest_sha256=request['model_manifest']['sha256'],
        output=dict(decoded=True, finite=True, count=1, sha256='a'*64,
                    **{key:request['spec'][key] for key in ('height', 'width', 'frames')}))


class MemoryAdapter:
    def __init__(self, clock, behavior=None):
        self.clock, self.behavior = clock, behavior
        self.calls = []
        self.closed = False
    async def execute(self, request, context):
        self.calls.append((copy.deepcopy(request), context, self.clock.now_ns()))
        if self.behavior is not None:
            return await self.behavior(request, context)
        return completed(request)
    async def aclose(self):
        self.closed = True
        return dict(status='test_client_closed', server_cancellation_confirmed=False)


SYSTEM = dict(id='synthetic-contract-test-only', source_manifest_sha256='f'*64)


class ServingClientTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, export, adapter, clock, drain=0., **kwargs):
        task = asyncio.create_task(client.replay(export, adapter, system=SYSTEM, drain_ms=drain,
            cleanup_timeout_seconds=.05, _clock=clock, **kwargs))
        await clock.advance(0)
        return task

    async def test_open_loop_offers_all_arrivals_while_first_is_pending_and_preserves_inputs(self):
        export = workload(); original = copy.deepcopy(export); clock = ManualClock()
        async def behavior(request, context):
            end_ms = 200 if request['request_id'] == 'r0' else request['arrival_ms'] + 5
            await clock.sleep_until_ns(clock.origin + client.offset_ns(end_ms))
            return completed(request)
        adapter = MemoryAdapter(clock, behavior)
        task = await self.start(export, adapter, clock, drain=80)
        for moment in (10, 15, 20, 25, 30, 35):
            await clock.advance(moment)
        self.assertEqual(len(adapter.calls), 4)
        self.assertFalse(task.done())
        for request, context, sent in adapter.calls:
            self.assertEqual(request, export['requests'][request['source_index']])
            self.assertEqual(context.origin_monotonic_ns, clock.origin)
            self.assertEqual(context.deadline_monotonic_ns, clock.origin + client.offset_ns(request['deadline_ms']))
            self.assertEqual(sent, clock.origin + client.offset_ns(request['arrival_ms']))
        await clock.advance(200)
        self.assertFalse(task.done(), 'drain/cutoff must not depend on earlier completions')
        await clock.advance(210)
        result = await task
        self.assertEqual(export, original)
        self.assertEqual(result['observed_until_ms'], 210)
        self.assertEqual(result['adapter_executions_started'], 4)
        self.assertFalse(result['execution_performed'])
        self.assertEqual(result['timing_evidence'], 'test_clock_no_performance_evidence')
        self.assertTrue(adapter.closed)
        self.assertFalse(result['transport_breakdown_measured'])
        self.assertTrue(all('client_received_ms' not in row for row in result['requests']))
        report = comparison.account_results(export, result)
        self.assertEqual(report['status'], 'accounted')
        self.assertEqual(report['overall']['offered'], 4)
        self.assertEqual(report['overall']['completed'], 4)
        self.assertEqual(report['overall']['slo_met'], 3)
        self.assertEqual(report['overall']['completed_in_window'], 3)
        self.assertEqual(report['overall']['measurement_horizon_ms'], 130)
        self.assertAlmostEqual(report['overall']['throughput_rps'], 3*1000/130)
        self.assertFalse(report['qualification_eligible'])
        self.assertEqual(report['overall']['wire_after_ready_ms'], {'count':0, 'p50':None})

    async def test_dispatch_lag_and_native_gf_budget_use_original_absolute_deadline(self):
        class LagClock(ManualClock):
            async def sleep_until_ns(self, target):
                if target == self.origin and self.value == self.origin:
                    self.value += client.offset_ns(35)
                await super().sleep_until_ns(target)
        export = workload(1); clock = LagClock()
        budgets = []
        async def behavior(request, context):
            # Native submit can occur later than the common client's execute entry.
            await clock.sleep_until_ns(clock.origin + client.offset_ns(40))
            budgets.append(context.remaining_budget_ms())
            await clock.sleep_until_ns(clock.origin + client.offset_ns(110))
            budgets.append(context.remaining_budget_ms())
            return completed(request)
        adapter = MemoryAdapter(clock, behavior)
        task = await self.start(export, adapter, clock, drain=20)
        for value in (40, 110, 120):
            await clock.advance(value)
        result = await task
        self.assertEqual(budgets, [60., 0.])
        row = result['requests'][0]
        self.assertEqual((row['arrival_ms'], row['deadline_ms'], row['output_ready_ms']), (0.,100.,110.))
        self.assertEqual(row['dispatch_ms'],35.)
        report = comparison.account_results(export, result)
        self.assertEqual(report['overall']['slo_met'], 0)
        self.assertEqual(report['overall']['completed'], 1)
        self.assertEqual(report['overall']['dispatch_lag_ms']['p50'],35.)
        context = adapter.calls[0][1]
        self.assertEqual(context.remaining_budget_ms(clock.origin + client.offset_ns(35)), 65.)
        self.assertEqual(context.remaining_budget_ms(clock.origin + client.offset_ns(150)), 0.)
        with self.assertRaises(ValueError): context.remaining_budget_ms(clock.origin-1)

    async def test_all_offered_failed_rejected_timeout_and_late_completion_keep_denominator(self):
        export = workload(); clock = ManualClock()
        async def behavior(request, context):
            rid = request['request_id']
            if rid == 'r0':
                await clock.sleep_until_ns(clock.origin + client.offset_ns(150))
                return completed(request)
            if rid == 'r1': raise RuntimeError('test transport failure')
            if rid == 'r2':
                return dict(request_id=rid, source_request_sha256=request['source_request_sha256'], status='rejected')
            await asyncio.Future()
        adapter = MemoryAdapter(clock, behavior)
        task = await self.start(export, adapter, clock, drain=50)
        for moment in (10,20,30,150,180): await clock.advance(moment)
        result = await task
        report = comparison.account_results(export, result)
        self.assertEqual(report['status'], 'accounted')
        self.assertEqual(report['overall']['status_counts'], dict(completed=1, failed=1, rejected=1, timed_out=1))
        self.assertEqual(report['overall']['offered'], 4)
        self.assertEqual(report['overall']['slo_attainment'], 0)
        self.assertEqual(result['cleanup']['pending_client_tasks'], 0)
        self.assertFalse(result['cleanup']['server_cancellation_confirmed'])

    async def test_expired_request_is_offered_with_zero_budget_and_never_gets_new_slo(self):
        export = workload(1); clock = ManualClock()
        async def behavior(request, context):
            await clock.sleep_until_ns(clock.origin + client.offset_ns(105))
            self.assertEqual(context.remaining_budget_ms(), 0.)
            return dict(request_id=request['request_id'], source_request_sha256=request['source_request_sha256'],
                        status='rejected', error='native admission: absolute deadline expired')
        task = await self.start(export, MemoryAdapter(clock, behavior), clock, drain=10)
        await clock.advance(105); await clock.advance(110)
        result = await task
        self.assertEqual(result['requests'][0]['deadline_ms'], 100.)
        report = comparison.account_results(export, result)
        self.assertEqual(report['overall']['offered'], 1)
        self.assertEqual(report['overall']['status_counts'], {'rejected':1})

    async def test_malformed_decoded_receipts_and_server_clock_overrides_fail_closed(self):
        mutations = ('request', 'source', 'spec', 'model', 'latent', 'shape', 'nonfinite', 'count', 'clock')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                export = workload(1); clock = ManualClock()
                async def behavior(request, context):
                    row = completed(request)
                    if mutation == 'request': row['request_id'] = 'other'
                    if mutation == 'source': row['source_request_sha256'] = '0'*64
                    if mutation == 'spec': row['executed_spec_sha256'] = '0'*64
                    if mutation == 'model': row['model_manifest_sha256'] = '0'*64
                    if mutation == 'latent': row['output']['decoded'] = False
                    if mutation == 'shape': row['output']['frames'] += 1
                    if mutation == 'nonfinite': row['output']['finite'] = False
                    if mutation == 'count': row['output']['count'] = 2
                    if mutation == 'clock': row['output_ready_ms'] = 0.
                    return row
                task = await self.start(export, MemoryAdapter(clock, behavior), clock)
                await clock.advance(100); result = await task
                self.assertEqual(result['requests'][0]['status'], 'failed')
                self.assertEqual(comparison.account_results(export,result)['overall']['offered'],1)

    async def test_adapter_cannot_mutate_frozen_inputs(self):
        export = workload(1); original = copy.deepcopy(export); clock = ManualClock()
        async def behavior(request, context):
            request['spec']['seed'] += 1
            return completed(request)
        task = await self.start(export, MemoryAdapter(clock, behavior), clock)
        await clock.advance(100); result = await task
        self.assertEqual(export, original)
        self.assertEqual(result['requests'][0]['status'], 'failed')
        self.assertIn('mutated', result['requests'][0]['error'])

    async def test_external_abort_keeps_unoffered_and_inflight_requests_without_server_cancel_claim(self):
        export = workload(); clock = ManualClock()
        async def behavior(request, context): await asyncio.Future()
        adapter = MemoryAdapter(clock, behavior)
        task = await self.start(export, adapter, clock)
        await clock.advance(5)
        task.cancel()
        result = await task
        self.assertEqual(len(result['requests']), 4)
        self.assertEqual([r['status'] for r in result['requests']], ['cancelled']+['not_executed']*3)
        report = comparison.account_results(export,result)
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['overall']['offered'],4)
        self.assertEqual(result['adapter_executions_started'],1)
        self.assertFalse(result['cleanup']['server_cancellation_confirmed'])
        self.assertTrue(adapter.closed)

    async def test_failed_cleanup_is_visible_and_never_implies_source_or_quality_qualification(self):
        export = workload(1); clock = ManualClock()
        class FailingClose(MemoryAdapter):
            async def aclose(self): raise RuntimeError('test cleanup failed')
        task = await self.start(export, FailingClose(clock), clock)
        await clock.advance(100); result = await task
        self.assertEqual(result['cleanup']['status'],'failed')
        self.assertIn('test cleanup failed', result['cleanup']['close_error'])
        self.assertFalse(result['qualification_eligible'])
        self.assertFalse(result['quality_validated'])
        self.assertEqual(result['allows'], [])

    async def test_invalid_workload_and_unpinned_system_rejected_before_any_offer(self):
        clock = ManualClock(); adapter = MemoryAdapter(clock)
        export = workload(1); export['requests'][0]['spec']['seed'] += 1
        with self.assertRaises(ValueError):
            await client.replay(export, adapter, system=SYSTEM, drain_ms=0, _clock=clock)
        with self.assertRaises(ValueError):
            await client.replay(workload(1), adapter, system={'id':'unknown'}, drain_ms=0, _clock=clock)
        for drain in (-1, float('nan'), True):
            with self.assertRaises(ValueError): client.describe(workload(1), drain)
        self.assertEqual(adapter.calls, [])


if __name__ == '__main__':
    unittest.main()
