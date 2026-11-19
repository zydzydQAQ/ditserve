#!/usr/bin/env python3
"""Item 6 GPU validation: cross-worker /dev/shm encoder-pool race on real SD3.5 workers.

N processes (one per allocated GPU, CUDA_VISIBLE_DEVICES=<uuid> like the controller) each hold a
real SD35Backend and share one encoder-pool shm_dir with write_shm=True. Worker backends are called
directly (no native study config): atomic, step, async-encode (encode_start/poll -> encode_ticket)
and continuous-batch paths, real T5/CLIP encode, a few denoise steps and VAE decode.

Phases: ``stress`` (tiny max_entries + per-request churn slabs, so writers evict and unlink their
own slabs continuously) and ``s4`` (the build's S4 pool descriptor, reusable for a later S4 run).
Instrumentation wraps the live objects in this process only; the repository is not modified.
A legacy-API control thread (has_key, gap, fetch) shows whether the pre-fix race window occurred.

Writes <output-dir>/summary.json and <output-dir>/<report-name>; no arrays or images.
Exit: 0 pass, 1 fail, 2 blocked (platform precheck), 3 inconclusive.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import traceback

SCHEMA = "ditserve.val-i6-shm-pool.v1"
GiB = 1 << 30
SD35_PATH = os.environ.get("DITSERVE_SD35_MODEL_PATH", "/shared/data/DITSERVE/models/sd3.5-medium")
SD35_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"  # benchmarks/cluster_config.py
NEG = "blurry, low quality, distorted, watermark"
PROMPTS = [
    "A cinematic photograph of an alpine lake reflecting snow-capped peaks at sunrise.",
    "A futuristic glass conservatory glowing at dusk under a starry indigo sky.",
    "An artisan ceramic teapot pouring green tea onto a weathered oak table, macro shot.",
    "A red fox trotting through fresh powder snow in a quiet birch forest.",
    "A coastal lighthouse standing against dramatic ocean waves at stormy twilight.",
    "A high-speed train crossing a steel arch bridge above a misty canyon at dawn.",
]
PATHS = ("atomic", "step", "async", "batch")
# Host-memory model used to decide how many workers may park T5-XXL on the CPU.
T5_HOST_GIB, BASE_HOST_GIB, RESERVE_GIB = 10.0, 3.0, 18.0


def platform_facts(shm_root="/dev/shm"):
    st = os.statvfs(shm_root)
    facts = dict(shm_root=shm_root, shm_total_bytes=st.f_blocks * st.f_frsize,
                 shm_free_bytes=st.f_bavail * st.f_frsize, nproc=os.cpu_count())
    try:
        raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        facts["cgroup_memory_max"] = None if raw == "max" else int(raw)
    except (OSError, ValueError):
        facts["cgroup_memory_max"] = None
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        info[name] = int(value.split()[0]) * 1024
    facts["mem_total_bytes"], facts["mem_available_bytes"] = info.get("MemTotal"), info.get("MemAvailable")
    limit = facts["cgroup_memory_max"] or facts["mem_total_bytes"]
    facts["memory_limit_bytes"] = min(limit, facts["mem_total_bytes"] or limit)
    return facts


def gpu_list():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [tuple(part.strip() for part in line.split(",", 1)) for line in out.splitlines() if line.strip()]


def pct(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    pos = (len(values) - 1) * q / 100.0
    lo, hi = int(pos), min(len(values) - 1, int(pos) + 1)
    return round(values[lo] + (values[hi] - values[lo]) * (pos - lo), 3)


def s4_pool_descriptor():
    try:
        from cluster_config import memory_strategy_descriptor
        return dict(memory_strategy_descriptor("s4_disaggregated_pool")["encoder_pool"]), "build benchmarks/cluster_config.py"
    except Exception as error:  # recorded; the fallback mirrors cluster_config at a7871f0
        return ({"enabled": True, "write_shm": True, "pin_memory": True, "shm_dir": "/dev/shm/ditserve-hetero-embeds",
                 "max_entries": 64, "max_bytes": 512 * (1 << 20)}, "fallback (%s)" % error)


class Instrument:
    """Counters on one worker's live pool/backend; restores backend attributes on close."""

    def __init__(self, backend, pool, pool_cls):
        self.tls, self.lock = threading.local(), threading.Lock()
        self.c = dict(ensure_hits=0, ensure_misses=0, restore_calls=0, restore_on_hit=0, encode_calls=0,
                      encode_on_hit=0, fetch_keyerrors_absorbed=0, unlink_attempts=0,
                      unlink_skipped_foreign=0, unlink_removed_own=0)
        self.t = dict(ensure_hit_ms=[], ensure_miss_ms=[], restore_ms=[], encode_ms=[], fetch_ms=[])
        ensure, fetch, unlink = pool.ensure_resident, pool.fetch_to_device, pool._unlink_own_slab

        def add(name, value=1):
            with self.lock:
                self.c[name] += value

        def ensure_wrap(key):
            t0 = time.perf_counter()
            hit = ensure(key)
            self.tls.last = bool(hit)
            add("ensure_hits" if hit else "ensure_misses")
            self.t["ensure_hit_ms" if hit else "ensure_miss_ms"].append((time.perf_counter() - t0) * 1e3)
            return hit

        def fetch_wrap(key, device, stream=None):
            t0 = time.perf_counter()
            try:
                result = fetch(key, device, stream=stream)
            except KeyError:
                add("fetch_keyerrors_absorbed")
                raise
            self.t["fetch_ms"].append((time.perf_counter() - t0) * 1e3)
            return result

        def unlink_wrap(record):
            path, slab_id = record.get("slab_path"), record.get("slab_id")
            if path and slab_id is not None:
                try:
                    ident = pool_cls._file_identity(os.stat(path))
                except OSError:
                    ident = None
                add("unlink_attempts")
                if ident is not None and ident != slab_id:
                    add("unlink_skipped_foreign")
                unlink(record)
                if ident == slab_id and not os.path.exists(path):
                    add("unlink_removed_own")
                return
            unlink(record)

        pool.ensure_resident, pool.fetch_to_device, pool._unlink_own_slab = ensure_wrap, fetch_wrap, unlink_wrap
        restore = backend.restore_encoders
        module = getattr(backend.pipe, "module", backend.pipe)
        encode = module.encode_prompt

        def restore_wrap(*args, **kwargs):
            if getattr(self.tls, "last", None) is True:
                add("restore_on_hit")
            add("restore_calls")
            t0 = time.perf_counter()
            result = restore(*args, **kwargs)
            self.t["restore_ms"].append((time.perf_counter() - t0) * 1e3)
            return result

        def encode_wrap(*args, **kwargs):
            real = kwargs.get("prompt") is not None and kwargs.get("prompt_embeds") is None
            if real:
                add("encode_calls")
                if getattr(self.tls, "last", None) is True:
                    add("encode_on_hit")
            t0 = time.perf_counter()
            result = encode(*args, **kwargs)
            if real:
                self.t["encode_ms"].append((time.perf_counter() - t0) * 1e3)
            return result

        backend.restore_encoders, module.encode_prompt = restore_wrap, encode_wrap
        self._undo = [(backend, "restore_encoders"), (module, "encode_prompt")]

    def close(self):
        for owner, name in self._undo:
            try:
                delattr(owner, name)
            except AttributeError:
                pass

    def summary(self):
        timing = {name: dict(n=len(v), p50=pct(v, 50), p95=pct(v, 95)) for name, v in self.t.items()}
        return dict(counters=dict(self.c), timing_ms=timing)


class LegacyControl(threading.Thread):
    """Pre-fix API on a separate, never-staging pool: has_key, a gap, then fetch from the slab."""

    def __init__(self, pool_cls, shm_dir, keys, gap_s, stop):
        super().__init__(daemon=True)
        self.reader = pool_cls({"shm_dir": shm_dir, "write_shm": False, "pin_memory": False})
        self.keys, self.gap_s, self.stop = keys, gap_s, stop
        self.checks = self.present = self.race_hits = self.ok = 0

    def run(self):
        while not self.stop.is_set():
            for key in self.keys:
                if self.stop.is_set():
                    return
                self.checks += 1
                if self.reader.has_key(key):
                    self.present += 1
                    time.sleep(self.gap_s)
                    try:
                        self.reader.fetch_to_device(key, "cpu")
                        self.ok += 1
                    except KeyError:
                        self.race_hits += 1
                time.sleep(0.01)


def spec(pidx, seed, cfg):
    return dict(prompt=PROMPTS[pidx], negative_prompt=NEG, seed=seed, height=cfg["size"], width=cfg["size"],
                steps=cfg["steps"], guidance_scale=4.5, max_sequence_length=256, batch_size=1, frames=1)


def finish(backend, sid, rid):
    result = backend.finish({"id": rid, "state_id": sid, "output_mode": "digest", "bubble_free": True})
    backend.release({"state_id": sid})
    return result["output"].get("array_sha256")


def run_path(backend, kind, prefix, pidx, seed, cfg):
    """One path; returns per-image records. Failures are recorded, never raised."""
    n_prompts, steps = len(PROMPTS), cfg["steps"]
    members = [(pidx, seed)] if kind in ("atomic", "step") else [(pidx, seed), ((pidx + 1) % n_prompts, seed + 1)]
    sids = ["%s-%s-%d" % (prefix, kind, k) for k in range(len(members))]
    specs = [spec(p, s, cfg) for p, s in members]
    t0 = time.perf_counter()
    try:
        digests = []
        if kind == "atomic":
            backend.prepare({"state_id": sids[0], "spec": specs[0]})
            backend.advance({"state_id": sids[0], "bubble_free": True})
        elif kind == "step":
            backend.prepare({"state_id": sids[0], "spec": specs[0]})
            backend.advance({"state_id": sids[0], "step_mode": True, "steps": steps, "bubble_free": True})
        elif kind == "async":
            # Ticket B encodes on the encode thread while request A denoises on the request thread.
            tickets = [backend.encode_start({"spec": s})["ticket"] for s in specs]
            for sid, s, ticket in zip(sids, specs, tickets):
                backend.encode_poll({"ticket": ticket, "wait": True})
                backend.prepare({"state_id": sid, "spec": s, "encode_ticket": ticket})
                backend.advance_batch({"state_ids": [sid], "steps": steps, "bubble_free": True})
        else:
            for sid, s in zip(sids, specs):
                backend.prepare({"state_id": sid, "spec": s})
            backend.advance_batch({"state_ids": sids, "steps": steps, "bubble_free": True})
        for k, sid in enumerate(sids):
            digests.append(finish(backend, sid, k + 1))
        ms = (time.perf_counter() - t0) * 1e3
        return [dict(path=kind, prompt=p, seed=s, ok=True, ms=round(ms / len(members), 3), digest=d)
                for (p, s), d in zip(members, digests)]
    except Exception as error:
        for sid in sids:
            if sid in backend.states:
                try:
                    backend.release({"state_id": sid})
                except Exception:
                    pass
        keyerror = isinstance(error, KeyError) or "Prompt embedding key not found" in str(error)
        return [dict(path=kind, prompt=p, seed=s, ok=False, keyerror=keyerror, error_type=type(error).__name__,
                     error=str(error)[:500]) for p, s in members]


def run_phase(backend, index, phase, cfg, barrier, pool_cls, torch):
    pool_cfg = dict(phase["pool"])
    pool = pool_cls(pool_cfg)
    backend.encoder_pool = pool
    backend.config.setdefault("options", {})["encoder_pool"] = pool_cfg
    probe = Instrument(backend, pool, pool_cls)
    keys = [backend._prompt_key(p, NEG, 256, True) for p in PROMPTS]
    stop = threading.Event()
    control = LegacyControl(pool_cls, pool_cfg["shm_dir"], keys, cfg["legacy_gap_ms"] / 1e3, stop)
    records, churn = [], 0
    barrier.wait(timeout=cfg["barrier_timeout"])
    started = time.perf_counter()
    time.sleep(index * cfg["stagger_s"])
    control.start()
    try:
        j = 0
        for iteration in range(phase["iterations"]):
            for kind in PATHS:
                seed = 100000 * phase["number"] + 1000 * index + j
                prefix = "%s-w%d-i%d" % (phase["name"], index, iteration)
                records.extend(run_path(backend, kind, prefix, (index + j) % len(PROMPTS), seed, cfg))
                j += 1
                if phase["churn"]:
                    churn += 1
                    key = hashlib.sha256(("churn-%d-%d" % (index, churn)).encode()).hexdigest()
                    pool.stage_embeddings(key, {"prompt_embeds": torch.zeros(1, 8, dtype=torch.bfloat16)},
                                          pin_memory=False, write_shm=True)
    finally:
        stop.set()
        control.join(30)
    wall_s = time.perf_counter() - started
    stats = pool.stats()
    pool.clear()
    probe.close()
    barrier.wait(timeout=cfg["barrier_timeout"])
    leftover = sorted(p.name for p in Path(pool_cfg["shm_dir"]).glob("*.pt")) if index == 0 else None
    barrier.wait(timeout=cfg["barrier_timeout"])
    return dict(pool_config=pool_cfg, pool_stats=stats, instrument=probe.summary(), records=records,
                wall_s=round(wall_s, 3), churn_slabs=churn,
                legacy_control=dict(checks=control.checks, present=control.present, fetch_ok=control.ok,
                                    race_hits=control.race_hits, gap_ms=cfg["legacy_gap_ms"]),
                leftover_slabs_after_clear=None if leftover is None else len(leftover))


def worker_main(index, uuid, offload, cfg, load_lock, barrier, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    out = dict(index=index, gpu_uuid=uuid, offload_text_encoders=offload, phases={}, errors=[])
    try:
        import torch
        from ditserve.models.encoder_pool import ProactiveEncoderPool
        from ditserve.models.sd35 import SD35Backend

        work = Path(cfg["work_root"]) / ("w%d" % index)
        work.mkdir(parents=True, exist_ok=True)
        with load_lock:  # one model load at a time bounds the transient host-memory peak
            t0 = time.perf_counter()
            backend = SD35Backend({
                "model_path": cfg["model_path"], "device": "cuda:0", "revision": SD35_REVISION,
                "dtype": "bfloat16", "text_encoder_dtype": "bfloat16", "attention": "flash", "work_dir": str(work),
                "options": {"profile": False, "cache": False, "continuous_batch": True, "sequence_packing": False,
                            "bubble_free": True, "offload_text_encoders_after_warmup": offload,
                            "encoder_pool": dict(cfg["phases"][0]["pool"])},
            })
            out["load_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["encoders_offloaded_after_load"] = bool(backend.encoders_offloaded)
        barrier.wait(timeout=cfg["barrier_timeout"])
        for phase in cfg["phases"]:
            out["phases"][phase["name"]] = run_phase(backend, index, phase, cfg, barrier, ProactiveEncoderPool, torch)
        backend.close()
    except BaseException as error:
        out["errors"].append(dict(type=type(error).__name__, message=str(error)[:2000],
                                  traceback=traceback.format_exc()[-4000:]))
        try:
            barrier.abort()
        except Exception:
            pass
    results.put(out)


def evaluate(phase, workers, n):
    rows = [w["phases"].get(phase["name"]) for w in workers]
    present = [r for r in rows if r is not None]
    records = [rec for r in present for rec in r["records"]]
    counters = {}
    for r in present:
        for name, value in r["instrument"]["counters"].items():
            counters[name] = counters.get(name, 0) + value
    adoptions = [(w["index"], w["offload_text_encoders"], r["pool_stats"]["shm_adoptions"])
                 for w, r in zip(workers, rows) if r is not None]
    failures = [rec for rec in records if not rec["ok"]]
    legacy = sum(r["legacy_control"]["race_hits"] for r in present)
    offload_adopt = sum(a for _, off, a in adoptions if off)
    with_adopt = sum(1 for _, _, a in adoptions if a > 0)
    criteria = dict(
        all_workers_reported=len(present) == n,
        zero_request_failures=not failures,
        zero_escaped_keyerrors=not any(rec.get("keyerror") for rec in failures),
        shm_adoptions_positive=sum(a for _, _, a in adoptions) > 0,
        adoption_on_half_the_workers=with_adopt >= math.ceil(n / 2),
        zero_restore_on_shm_hit=counters.get("restore_on_hit", 0) == 0,
        zero_encode_on_shm_hit=counters.get("encode_on_hit", 0) == 0,
        restore_gate_exercised=offload_adopt > 0,
        legacy_race_window_hit=legacy > 0,
    )
    hard = ("all_workers_reported", "zero_request_failures", "zero_escaped_keyerrors",
            "zero_restore_on_shm_hit", "zero_encode_on_shm_hit")
    soft = ["shm_adoptions_positive", "adoption_on_half_the_workers", "restore_gate_exercised"]
    if phase["churn"] and n >= 8:
        soft.append("legacy_race_window_hit")
    verdict = ("FAIL" if not all(criteria[k] for k in hard) else
               "INCONCLUSIVE" if not all(criteria[k] for k in soft) else "PASS")
    by_path = {}
    for kind in PATHS:
        ms = [rec["ms"] for rec in records if rec["path"] == kind and rec["ok"]]
        by_path[kind] = dict(images=sum(1 for rec in records if rec["path"] == kind), ok=len(ms),
                             p50_ms=pct(ms, 50), p95_ms=pct(ms, 95))
    timing = {}
    for name in ("ensure_hit_ms", "ensure_miss_ms", "restore_ms", "encode_ms", "fetch_ms"):
        p50s = [r["instrument"]["timing_ms"][name]["p50"] for r in present if r["instrument"]["timing_ms"][name]["p50"] is not None]
        timing[name + "_worker_p50_median"] = round(statistics.median(p50s), 3) if p50s else None
    digests = {}
    for rec in records:
        if rec["ok"]:
            digests.setdefault("%d/%d" % (rec["prompt"], rec["seed"]), set()).add(rec["digest"])
    return dict(verdict=verdict, criteria=criteria, hard=list(hard), soft=soft, images=len(records),
                failures=len(failures), failure_samples=failures[:5], counters=counters,
                shm_adoptions_per_worker=[dict(worker=i, offloaded=o, adoptions=a) for i, o, a in adoptions],
                offloaded_worker_adoptions=offload_adopt, legacy_race_hits=legacy, latency_by_path=by_path,
                timing=timing, wall_s_max=max((r["wall_s"] for r in present), default=None),
                leftover_slabs_after_clear=next((r["leftover_slabs_after_clear"] for r in present
                                                 if r["leftover_slabs_after_clear"] is not None), None),
                unique_seed_prompt_pairs=len(digests))


def write_report(path, summary):
    lines = ["# %s item 6: shared /dev/shm encoder-pool race (SD3.5, %d GPU)" % (
        summary["date"], summary["gpus"]), "",
        "Overall verdict: **%s**. Build `%s`, run `%s`. Worker backends were called directly; this is not a serving or S4 run." % (
            summary["verdict"], summary["build"], summary["run_id"]), "",
        "Platform: /dev/shm %.1f GiB free, memory limit %.1f GiB, nproc %s; %d of %d workers offload T5-XXL." % (
            summary["platform"]["shm_free_bytes"] / GiB, summary["platform"]["memory_limit_bytes"] / GiB,
            summary["platform"]["nproc"], summary["offloaded_workers"], summary["gpus"]), ""]
    for name, result in summary["phases"].items():
        c = result["counters"]
        lines += ["## Phase %s: %s" % (name, result["verdict"]), "",
                  "| Criterion | Result |", "| --- | --- |"]
        lines += ["| %s%s | %s |" % (k, "" if k in result["hard"] else " (soft)", v) for k, v in result["criteria"].items()]
        lines += ["", "Images %d, failures %d; shm adoptions %s; offloaded-worker adoptions %d; legacy race hits %d;"
                  " absorbed fetch KeyErrors %d; restore calls %d; foreign-slab unlinks skipped %d; leftover slabs %s." % (
                      result["images"], result["failures"],
                      sum(x["adoptions"] for x in result["shm_adoptions_per_worker"]),
                      result["offloaded_worker_adoptions"], result["legacy_race_hits"],
                      c.get("fetch_keyerrors_absorbed", 0), c.get("restore_calls", 0),
                      c.get("unlink_skipped_foreign", 0), result["leftover_slabs_after_clear"]), "",
                  "| Path | Images | OK | p50 ms/image | p95 ms/image |", "| --- | --- | --- | --- | --- |"]
        lines += ["| %s | %d | %d | %s | %s |" % (k, v["images"], v["ok"], v["p50_ms"], v["p95_ms"])
                  for k, v in result["latency_by_path"].items()]
        lines += ["", "Timing (median of worker p50, ms): " + ", ".join(
            "%s=%s" % (k, v) for k, v in result["timing"].items()), ""]
    if summary["worker_errors"]:
        lines += ["## Worker errors", ""] + ["- w%d %s: %s" % (e["worker"], e["type"], e["message"][:300])
                                             for e in summary["worker_errors"]] + [""]
    lines += ["Scope: GPU worker-level evidence for the race fix only. A finite run cannot prove absence of the race;"
              " the legacy control counts how often the pre-fix window was actually hit. Cross-GPU digest equality is"
              " not asserted.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--report-name", default="20261105-encoder-shm-race-validation.md")
    ap.add_argument("--gpus", type=int, default=0, help="0 = every visible GPU")
    ap.add_argument("--phases", default="stress,s4")
    ap.add_argument("--iterations-stress", type=int, default=3)
    ap.add_argument("--iterations-s4", type=int, default=2)
    ap.add_argument("--stress-max-entries", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--legacy-gap-ms", type=float, default=50.0)
    ap.add_argument("--stagger-s", type=float, default=1.5)
    ap.add_argument("--offload", default="auto", help="auto | none | all | <count>")
    ap.add_argument("--model-path", default=SD35_PATH)
    ap.add_argument("--timeout-s", type=int, default=3000)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("DITSERVE_RUN_ID", "local")
    facts = platform_facts()
    gpus = gpu_list()
    n = args.gpus or len(gpus)
    summary = dict(schema=SCHEMA, date=time.strftime("%Y%m%d"), run_id=run_id, build=os.environ.get("DITSERVE_BUILD"),
                   gpus=n, gpu_names=[name for _, name in gpus[:n]], platform=facts, args=vars(args) | {"output_dir": str(args.output_dir)})
    budget = (facts["memory_limit_bytes"] / GiB - RESERVE_GIB - n * BASE_HOST_GIB) / T5_HOST_GIB
    blocked = []
    if n < 1 or n > len(gpus):
        blocked.append("requested %d GPUs, %d visible" % (n, len(gpus)))
    if facts["shm_free_bytes"] < GiB:
        blocked.append("/dev/shm free below 1 GiB")
    if budget < 0:
        blocked.append("memory limit %.1f GiB cannot hold %d SD3.5 workers" % (facts["memory_limit_bytes"] / GiB, n))
    if blocked:
        summary.update(verdict="BLOCKED", blocked=blocked)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(dict(verdict="BLOCKED", blocked=blocked)), flush=True)
        return 2
    auto = max(0, min(n, int(budget)))
    offload_n = {"auto": auto, "none": 0, "all": n}.get(args.offload)
    offload_n = min(n, int(args.offload)) if offload_n is None else offload_n
    if offload_n > auto:
        print("warning: offload %d exceeds the host-memory model (%d)" % (offload_n, auto), flush=True)
    s4_pool, s4_source = s4_pool_descriptor()
    stress_dir = "/dev/shm/ditserve-val-i6-%s/stress" % run_id
    phases = []
    for number, name in enumerate(p for p in args.phases.split(",") if p):
        if name == "stress":
            pool = {"enabled": True, "write_shm": True, "pin_memory": True, "shm_dir": stress_dir,
                    "max_entries": args.stress_max_entries, "max_bytes": 512 * (1 << 20)}
            phases.append(dict(name=name, number=number + 1, pool=pool, churn=True, iterations=args.iterations_stress))
        elif name == "s4":
            phases.append(dict(name=name, number=number + 1, pool=s4_pool, churn=False, iterations=args.iterations_s4))
        else:
            raise SystemExit("unknown phase " + name)
    for phase in phases:
        Path(phase["pool"]["shm_dir"]).mkdir(parents=True, exist_ok=True)
        phase["preexisting_slabs"] = len(list(Path(phase["pool"]["shm_dir"]).glob("*.pt")))
    cfg = dict(model_path=args.model_path, steps=args.steps, size=args.size, legacy_gap_ms=args.legacy_gap_ms,
               stagger_s=args.stagger_s, barrier_timeout=1800, phases=phases,
               work_root="/dev/shm/ditserve-val-i6-%s/work" % run_id)
    summary.update(offloaded_workers=offload_n, offload_budget_model=dict(
        t5_host_gib=T5_HOST_GIB, base_host_gib=BASE_HOST_GIB, reserve_gib=RESERVE_GIB, allowed=auto),
        s4_pool_source=s4_source, phase_configs=phases)
    ctx = mp.get_context("spawn")
    barrier, lock, queue = ctx.Barrier(n), ctx.Lock(), ctx.Queue()
    procs = [ctx.Process(target=worker_main, args=(i, gpus[i][0], i < offload_n, cfg, lock, barrier, queue))
             for i in range(n)]
    for proc in procs:
        proc.start()
    results, deadline = [], time.monotonic() + args.timeout_s
    while len(results) < n and time.monotonic() < deadline:
        try:
            results.append(queue.get(timeout=5))
        except Exception:
            if not any(p.is_alive() for p in procs) and queue.empty():
                break
    for proc in procs:
        proc.join(60)
        if proc.is_alive():
            proc.terminate()
    results.sort(key=lambda w: w["index"])
    summary["worker_exitcodes"] = [p.exitcode for p in procs]
    summary["workers"] = [dict(index=w["index"], gpu_uuid=w["gpu_uuid"], gpu_name=w.get("gpu_name"),
                               load_ms=w.get("load_ms"), offloaded=w["offload_text_encoders"],
                               encoders_offloaded_after_load=w.get("encoders_offloaded_after_load"),
                               pool_stats={k: v["pool_stats"] for k, v in w["phases"].items()},
                               counters={k: v["instrument"]["counters"] for k, v in w["phases"].items()},
                               legacy_control={k: v["legacy_control"] for k, v in w["phases"].items()})
                          for w in results]
    summary["worker_errors"] = [dict(worker=w["index"], **{k: e[k] for k in ("type", "message", "traceback")})
                                for w in results for e in w["errors"]]
    summary["phases"] = {p["name"]: evaluate(p, results, n) for p in phases}
    verdicts = [r["verdict"] for r in summary["phases"].values()]
    summary["verdict"] = ("FAIL" if "FAIL" in verdicts or summary["worker_errors"] or len(results) < n else
                          "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts else "PASS")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=sorted) + "\n")
    write_report(args.output_dir / args.report_name, summary)
    print(json.dumps(dict(verdict=summary["verdict"], phases={k: v["verdict"] for k, v in summary["phases"].items()})), flush=True)
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 3}[summary["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
