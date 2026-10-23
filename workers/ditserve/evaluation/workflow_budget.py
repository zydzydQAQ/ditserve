"""Crash-conservative wall-clock budget for resumable, exclusive GPU studies."""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


class BudgetExhausted(RuntimeError):
    pass


class StudyBudget:
    """One launcher holds the lock; retries and restarted launchers share charges.

    Charge model loading, warmup, measurement, failures and shutdown. A stale
    active attempt is charged conservatively up to recovery (bounded by a
    finite remaining allowance), and explicitly marked uncertain rather than
    free. A None limit removes only the cumulative cap; each attempt retains
    a finite timeout and independent launcher watchdog.
    """

    def __init__(self, root, limit_seconds=28800, *, clock=time.time, monotonic=time.monotonic,
                 recovery_check=None):
        self.root = Path(root)
        self.limit = None if limit_seconds is None else float(limit_seconds)
        if self.limit is not None and (not math.isfinite(self.limit) or self.limit <= 90):
            raise ValueError('study budget must exceed the 90-second cleanup reserve')
        self.clock, self.monotonic = clock, monotonic
        self.recovery_check = recovery_check
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.handle = None
        self.started = None
        self.record = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root / 'budget.lock').open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            raise RuntimeError('another launcher owns this study budget') from None
        try:
            return self._enter_locked()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _enter_locked(self):
        path = self.root / 'budget.json'
        self.record = json.loads(path.read_text()) if path.exists() else dict(
            schema='ditserve.workflow.budget.v1', limit_seconds=self.limit,
            charged_seconds=0., attempts=[], active=None)
        if self.record['limit_seconds'] != self.limit:
            raise ValueError('resuming cannot change the original study budget')
        if self.record.get('active'):
            active = self.record['active']
            if self.recovery_check is None:
                raise RuntimeError('active budget attempt requires verified recovery before resuming')
            self.recovery_check(dict(active))
            accounting = 'conservative wall time to recovery'
            if active.get('boot_id') == self.boot_id and 'started_monotonic' in active:
                elapsed = max(0., self.monotonic() - active['started_monotonic'])
            else:
                elapsed = self.clock() - active['started_unix']
                if elapsed < 0:
                    # After a reboot monotonic time is incomparable. Preserve
                    # the finite policy; unlimited attempts still have a finite
                    # authorized runtime plus cleanup allowance to charge.
                    elapsed = (self.remaining_seconds if self.limit is not None else
                               active['timeout_seconds'] + active['cleanup_seconds'])
                    if self.limit is None:
                        accounting = 'clock rollback after reboot; authorized task timeout plus cleanup reserve'
            charge = elapsed if self.limit is None else min(elapsed, self.remaining_seconds)
            active.update(status='recovered_uncertain', charged_seconds=charge,
                          finished_unix=self.clock(), accounting=accounting)
            self.record['attempts'].append(active)
            self.record['charged_seconds'] += charge
            self.record['active'] = None
        self.save()
        return self

    @property
    def remaining_seconds(self):
        return None if self.limit is None else max(0., self.limit - self.record['charged_seconds'])

    def authorize_unlimited(self, authorization, *, authorized_unix=None):
        """Explicitly replace a finite cap while owning this instance's lock.

        The caller owns the study driver lock, if there is one. Recover any
        interrupted attempt before calling. Normal resume never changes a cap.
        The old ledger is retained byte for byte, including every attempt.
        """
        if self.handle is None or self.record is None:
            raise RuntimeError('budget authorization requires the held budget lock')
        if self.record.get('active'):
            raise RuntimeError('finish or verify recovery of the active attempt before authorization')
        if self.limit is None:
            raise ValueError('study budget is already unlimited')
        if not isinstance(authorization, str) or not authorization.strip():
            raise ValueError('unlimited budget requires the explicit authorization text')
        when = self.clock() if authorized_unix is None else authorized_unix
        if type(when) not in (int, float) or not math.isfinite(when) or when < 0:
            raise ValueError('authorization timestamp must be finite and nonnegative')
        previous = (self.root / 'budget.json').read_bytes()
        if json.loads(previous) != self.record:
            raise RuntimeError('budget ledger changed while its lock was held')
        name = 'budget-history/before-unlimited-' + uuid.uuid4().hex + '.json'
        snapshot = self.root / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open('xb') as handle:
            handle.write(previous)
        event = dict(old_limit_seconds=self.limit, new_limit_seconds=None,
                     authorization=authorization, authorized_unix=float(when),
                     snapshot=name, snapshot_sha256=hashlib.sha256(previous).hexdigest())
        record = json.loads(previous)
        record.setdefault('authorization_history', []).append(event)
        record['limit_seconds'] = None
        record['overrun_seconds'] = 0.
        atomic_json(self.root / 'budget.json', record)
        self.limit, self.record = None, record
        return dict(event)

    def save(self):
        atomic_json(self.root / 'budget.json', self.record)

    def start(self, job_id, *, minimum_seconds=0., cleanup_seconds=90., timeout_seconds=3600.):
        if self.record.get('active'):
            raise RuntimeError('budget already has an active attempt')
        for name, value in (('minimum_seconds', minimum_seconds), ('cleanup_seconds', cleanup_seconds),
                            ('timeout_seconds', timeout_seconds)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive')
        remaining = self.remaining_seconds
        if remaining is not None and remaining < max(float(minimum_seconds), 1.) + cleanup_seconds:
            raise BudgetExhausted('insufficient budget for next complete attempt')
        self.started = self.monotonic()
        run_seconds = min(timeout_seconds, remaining - cleanup_seconds) if remaining is not None else timeout_seconds
        reason = 'task_timeout' if run_seconds == timeout_seconds else 'budget_deadline'
        self.record['active'] = dict(id=uuid.uuid4().hex, job_id=job_id,
            started_unix=self.clock(), remaining_at_start=remaining,
            started_monotonic=self.started, boot_id=self.boot_id,
            timeout_seconds=timeout_seconds, cleanup_seconds=cleanup_seconds, deadline_reason=reason,
            stop_at_monotonic=self.started + run_seconds,
            stop_at_unix=self.clock() + run_seconds)
        self.save()
        return dict(self.record['active'])

    def finish(self, status, **metadata):
        active = self.record.get('active')
        if active is None:
            return
        elapsed = max(0., self.monotonic() - self.started)
        active.update(metadata, status=status, finished_unix=self.clock(), charged_seconds=elapsed)
        self.record['charged_seconds'] += elapsed
        self.record['attempts'].append(active)
        self.record['active'] = None
        self.started = None
        self.record['overrun_seconds'] = (0. if self.limit is None else
                                          max(0., self.record['charged_seconds'] - self.limit))
        self.save()

    def __exit__(self, exc_type, exc, tb):
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
