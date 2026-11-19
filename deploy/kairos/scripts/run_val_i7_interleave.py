#!/usr/bin/env python3
"""Item 7 GPU validation: interleaved Ulysses all-to-all routing and exactness (Wan2.1, SP2, cfg=1).

For each collective backend (nccl, gfc) the parent spawns two ranks the way the C++ controller does
(CUDA_VISIBLE_DEVICES=<gpu uuid>, device cuda:0, tcp://127.0.0.1 init) running real WanBackend
workers (execution=block_interleaved, flash attention) and calls distributed(action=init,
collective_backend=...). Rank 0 prepares each shape once and block_exports it; every arm on both
ranks block_imports that same file, so all arms start from bitwise-identical state. Arms differ only
in WanBlockExecutor.interleave (what execution=block vs block_interleaved selects):

    warm (non-interleaved, 1 step), N1, I1, N2, I2 (3 solver steps each)

Counted in-process (no repository change): functional NCCL all_to_all_single calls and their group
(ULYSSES_PG or other), GFC all2all/all_gather calls (collectives.info), the interleaved Q/K norm+RoPE
helper and the processor reference. Latents are hashed after every step and compared bitwise.

If GFC init fails under controller visibility, that backend is retried once with both GPUs visible
(device cuda:<rank>) and the failure is recorded. Exit 0 pass, 1 fail, 2 blocked.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

SCHEMA = "ditserve.val-i7-interleaved-ulysses.v1"
WAN21_PATH = os.environ.get("DITSERVE_WAN21_MODEL_PATH", "/shared/data/DITSERVE/models/wan")
PROMPT = "A red fox trotting through fresh powder snow in a quiet birch forest, golden hour, cinematic."
NEG = "blurry, low quality, distorted, watermark, static"
ARMS = (("warm", False, 1), ("N1", False, 3), ("I1", True, 3), ("N2", False, 3), ("I2", True, 3))
GFC_DEFAULT_BYTES = 128 * 1024 * 1024


def gpu_uuids():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def parse_shape(text):
    h, w, f, steps = (int(x) for x in text.split("x"))
    return dict(name="%dx%dx%d" % (h, w, f), height=h, width=w, frames=f, steps=steps)


def stacked_qkv_bytes(shape, sp, heads=12, head_dim=128, itemsize=2):
    tokens = ((shape["frames"] - 1) // 4 + 1) * (shape["height"] // 16) * (shape["width"] // 16)
    local = -(-tokens // sp)
    return dict(tokens=tokens, stacked_qkv_bytes=3 * heads * local * head_dim * itemsize,
                single_tensor_bytes=heads * local * head_dim * itemsize)


class Counters:
    def __init__(self, torch, block_mod, processor_cls, collectives, process_group):
        import torch.distributed._functional_collectives as ft_c
        self.c = dict(nccl_a2a_ulysses=0, nccl_a2a_other=0, dist_a2a=0, helper_qk=0, reference_qk=0)
        self.collectives = collectives
        a2a, dist_a2a = ft_c.all_to_all_single, torch.distributed.all_to_all_single
        helper, reference = block_mod._processor_qk_norm_rope, processor_cls._qk_norm_rope_reference

        def a2a_wrap(tensor, output_split_sizes=None, input_split_sizes=None, group=None, *args, **kwargs):
            self.c["nccl_a2a_ulysses" if group is process_group.ULYSSES_PG else "nccl_a2a_other"] += 1
            return a2a(tensor, output_split_sizes, input_split_sizes, group, *args, **kwargs)

        def dist_wrap(*args, **kwargs):
            self.c["dist_a2a"] += 1
            return dist_a2a(*args, **kwargs)

        def helper_wrap(*args, **kwargs):
            self.c["helper_qk"] += 1
            return helper(*args, **kwargs)

        def reference_wrap(proc, *args, **kwargs):
            self.c["reference_qk"] += 1
            return reference(proc, *args, **kwargs)

        ft_c.all_to_all_single, torch.distributed.all_to_all_single = a2a_wrap, dist_wrap
        block_mod._processor_qk_norm_rope = helper_wrap
        processor_cls._qk_norm_rope_reference = reference_wrap

    def snapshot(self):
        calls = self.collectives.info().get("calls", {})
        return dict(self.c, gfc_all2all=calls.get("all2all", 0), gfc_all_gather=calls.get("all_gather", 0))


def rank_main(rank, uuids, name, visibility, port, cfg, barrier, results):
    if visibility == "controller":
        os.environ["CUDA_VISIBLE_DEVICES"], device = uuids[rank], "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"], device = ",".join(uuids[:2]), "cuda:%d" % rank
    out = dict(rank=rank, backend=name, visibility=visibility, shapes={}, errors=[])
    try:
        import torch
        from ditserve.models.wan.backend import WanBackend
        from ditserve.models.wan import block as block_mod
        from ditserve.models.xdit import collectives
        from ditserve.models.xdit.process_group import PROCESS_GROUP

        t0 = time.perf_counter()
        backend = WanBackend({"model_path": cfg["model_path"], "device": device, "attention": "flash",
                              "vae_dtype": "bfloat16", "text_encoder_dtype": "bfloat16",
                              "options": {"execution": "block_interleaved"}})
        out["load_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        out["gpu_name"] = torch.cuda.get_device_name(torch.device(device))
        model = backend.runtime.pipe.transformer
        processor = model.blocks[0].attn1.processor
        namespace = getattr(type(processor).__call__, "__globals__", {})
        out["processor"] = dict(cls=type(processor).__name__, has_flydsl=bool(namespace.get("_HAS_FLYDSL", False)),
                                blocks=len(model.blocks), heads=int(model.blocks[0].attn1.heads))
        counters = Counters(torch, block_mod, type(processor), collectives, PROCESS_GROUP)
        init = dict(action="init", init_method="tcp://127.0.0.1:%d" % port, rank=rank, world_size=2,
                    ranks=[rank], collective_backend=name, timeout_seconds=60)
        if name == "gfc":
            init["gfc_max_collective_bytes"] = cfg["gfc_max_bytes"]
        try:
            out["init"] = backend.distributed(init)
        except BaseException as error:
            out["init_error"] = "%s: %s" % (type(error).__name__, str(error)[:1500])
            raise
        out["execution"] = backend.execution
        transfer = Path(cfg["transfer_dir"])
        for shape in cfg["shapes"]:
            path = transfer / ("%s-%s-%s.pt" % (cfg["run_id"], name, shape["name"]))
            spec = dict(prompt=PROMPT, negative_prompt=NEG, seed=cfg["seed"], height=shape["height"],
                        width=shape["width"], frames=shape["frames"], steps=shape["steps"], guidance_scale=5.0)
            if rank == 0:
                prepared = backend.prepare({"state_id": "src-" + shape["name"],
                                            "spec": dict(spec, request_id="val-i7-" + shape["name"])})
                exported = backend.block_export({"state_id": "src-" + shape["name"], "path": str(path), "release": True})
                Path(str(path) + ".sha256").write_text(exported["sha256"])
                prepare_ms = prepared.get("model_ms")
            barrier.wait(timeout=1800)
            sha = Path(str(path) + ".sha256").read_text().strip()
            arms, ref = {}, None
            for arm, interleave, steps in ARMS:
                backend.executor.interleave = interleave
                sid = "%s-%s" % (shape["name"], arm)
                backend.block_import({"state_id": sid, "path": str(path), "sha256": sha})
                before = counters.snapshot()
                digests, walls, model_ms, lat = [], [], [], []
                for _ in range(steps):
                    torch.cuda.synchronize(torch.device(device))
                    start = time.perf_counter()
                    result = backend.advance({"state_id": sid, "group": [0, 1], "epoch": 1, "cfg": 1, "ring": 1,
                                              "quantum": "step"})
                    torch.cuda.synchronize(torch.device(device))
                    walls.append(round((time.perf_counter() - start) * 1e3, 3))
                    model_ms.append(round(float(result.get("model_ms", 0.0)), 3))
                    latents = backend.states[sid].latents.detach()
                    lat.append(latents.float().cpu())
                    digests.append(hashlib.sha256(latents.contiguous().cpu().view(-1).view(torch.uint8).numpy()
                                                  .tobytes()).hexdigest())
                after = counters.snapshot()
                record = dict(interleave=interleave, steps=steps, step_wall_ms=walls, step_model_ms=model_ms,
                              digests=digests, latent_dtype=str(latents.dtype),
                              counts={k: after[k] - before[k] for k in after},
                              step_index=backend.states[sid].step_index)
                if arm == "N1":
                    ref = lat
                elif ref is not None and arm != "warm":
                    diffs = []
                    for a, b in zip(lat, ref):
                        delta = (a - b).abs()
                        diffs.append(dict(equal=bool(torch.equal(a, b)), max_abs=float(delta.max()),
                                          max_rel=float((delta / b.abs().clamp_min(1e-6)).max())))
                    record["vs_N1"] = diffs
                arms[arm] = record
                backend.release({"state_id": sid})
                barrier.wait(timeout=1800)
            if rank == 0:
                for leftover in (path, Path(str(path) + ".sha256")):
                    leftover.unlink(missing_ok=True)
            out["shapes"][shape["name"]] = dict(arms=arms, prepare_ms=prepare_ms if rank == 0 else None)
        out["collectives_info"] = collectives.info()
        backend.close()
    except BaseException as error:
        out["errors"].append(dict(type=type(error).__name__, message=str(error)[:2000],
                                  traceback=traceback.format_exc()[-4000:]))
        try:
            barrier.abort()
        except Exception:
            pass
    results.put(out)


def run_pair(name, visibility, port, uuids, cfg, timeout_s):
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    procs = [ctx.Process(target=rank_main, args=(r, uuids, name, visibility, port, cfg, barrier, queue)) for r in (0, 1)]
    for proc in procs:
        proc.start()
    results, deadline = [], time.monotonic() + timeout_s
    while len(results) < 2 and time.monotonic() < deadline:
        try:
            results.append(queue.get(timeout=5))
        except Exception:
            if not any(p.is_alive() for p in procs) and queue.empty():
                break
    for proc in procs:
        proc.join(60)
        if proc.is_alive():
            proc.terminate()
    results.sort(key=lambda r: r["rank"])
    return dict(backend=name, visibility=visibility, ranks=results, exitcodes=[p.exitcode for p in procs],
                complete=len(results) == 2 and not any(r["errors"] for r in results))


def evaluate(pair, cfg):
    name, crit, notes, timing = pair["backend"], {}, [], {}
    if not pair["complete"]:
        return dict(verdict="FAIL", criteria=dict(completed=False), notes=[
            "%s rank %d: %s" % (name, r["rank"], (r["errors"] or [{}])[0].get("message", "")[:300]) for r in pair["ranks"]])
    blocks = pair["ranks"][0]["processor"]["blocks"]
    per_step = dict(I=4 * blocks, N=8 * blocks)
    crit["completed"] = True
    crit["reference_qk_branch_only"] = all(not r["processor"]["has_flydsl"] for r in pair["ranks"])
    for shape in cfg["shapes"]:
        sname = shape["name"]
        for r in pair["ranks"]:
            arms = r["shapes"][sname]["arms"]
            for arm, rec in arms.items():
                kind = "I" if rec["interleave"] else "N"
                expected = per_step[kind] * rec["steps"]
                c = rec["counts"]
                tag = "%s/r%d/%s" % (sname, r["rank"], arm)
                if name == "gfc":
                    ok = c["nccl_a2a_ulysses"] == 0 and c["dist_a2a"] == 0 and c["gfc_all2all"] == expected
                else:
                    ok = c["gfc_all2all"] == 0 and c["nccl_a2a_ulysses"] == expected
                ok = ok and c["nccl_a2a_other"] == 0
                crit["routing:" + tag] = ok
                if not ok:
                    notes.append("%s counts %s, expected %d %s all2all" % (tag, c, expected, name))
                # Per block and branch: attn1 (helper when interleaved, else processor __call__) and the
                # cross-attention attn2 (always __call__, rotary None -> reference). Without FlyDSL every
                # attn1 also lands in the reference; with it only attn2 does.
                helper_expected = 2 * blocks * rec["steps"] if rec["interleave"] else 0
                flydsl = r["processor"]["has_flydsl"]
                reference_expected = (2 if flydsl else 4) * blocks * rec["steps"]
                crit["fast_path:" + tag] = (c["helper_qk"] == helper_expected
                                            and c["reference_qk"] == reference_expected)
                if "vs_N1" in rec:
                    equal = all(d["equal"] for d in rec["vs_N1"])
                    crit["bitwise_vs_N1:" + tag] = equal
                    if not equal:
                        notes.append("%s differs from N1: max_abs=%.3e max_rel=%.3e" % (
                            tag, max(d["max_abs"] for d in rec["vs_N1"]), max(d["max_rel"] for d in rec["vs_N1"])))
        r0, r1 = (p["shapes"][sname]["arms"] for p in pair["ranks"])
        crit["ranks_replicated:" + sname] = all(r0[a]["digests"] == r1[a]["digests"] for a in r0)
        inter = [ms for a in ("I1", "I2") for ms in r0[a]["step_wall_ms"]]
        plain = [ms for a in ("N1", "N2") for ms in r0[a]["step_wall_ms"]]
        timing[sname] = dict(interleaved_median_ms=round(statistics.median(inter), 3),
                             non_interleaved_median_ms=round(statistics.median(plain), 3),
                             ratio_i_over_n=round(statistics.median(inter) / statistics.median(plain), 4),
                             n1_digests=r0["N1"]["digests"])
    hard = [k for k in crit if not k.startswith("reference_qk")]
    verdict = "PASS" if all(crit[k] for k in hard) else "FAIL"
    return dict(verdict=verdict, criteria=crit, notes=notes[:40], timing=timing, expected_per_step=per_step)


def write_report(path, summary):
    lines = ["# %s item 7: interleaved Ulysses all-to-all routing and exactness (Wan2.1, SP2)" % summary["date"], "",
             "Overall verdict: **%s**. Build `%s`, run `%s`; worker backends were called directly with real weights." % (
                 summary["verdict"], summary["build"], summary["run_id"]), ""]
    for name, res in summary["backends"].items():
        failed = [k for k, v in res["criteria"].items() if not v]
        lines += ["## collective_backend=%s: %s" % (name, res["verdict"]), "",
                  "Visibility: %s. Criteria passed %d/%d%s." % (
                      summary["runs"][name]["visibility"], sum(bool(v) for v in res["criteria"].values()),
                      len(res["criteria"]), "; failed: " + ", ".join(failed[:12]) if failed else ""), ""]
        if summary["runs"][name].get("controller_visibility_init_error"):
            lines += ["Controller-visibility init failed first: `%s`" % summary["runs"][name]["controller_visibility_init_error"][:300], ""]
        lines += ["| Shape | interleaved median ms/step | non-interleaved median ms/step | ratio |", "| --- | --- | --- | --- |"]
        lines += ["| %s | %s | %s | %s |" % (s, t["interleaved_median_ms"], t["non_interleaved_median_ms"], t["ratio_i_over_n"])
                  for s, t in res.get("timing", {}).items()]
        lines += [""] + ["- " + n for n in res["notes"]] + [""]
    lines += ["Cross-backend N1 digests equal: %s." % summary.get("cross_backend_n1_equal"), "",
              "GFC payload check (default limit %d MiB): %s" % (GFC_DEFAULT_BYTES >> 20, "; ".join(
                  "%s stacked QKV %.1f MiB" % (s, v["stacked_qkv_bytes"] / 2**20) for s, v in summary["payloads"].items())), "",
              "Scope: routing and bitwise exactness on two GPUs at SP2, cfg=1. `_HAS_FLYDSL` is %s in the vendored"
              " processor, so only the reference Q/K norm+RoPE branch ran. Timings are a short diagnostic, not a"
              " performance claim." % summary["has_flydsl"], ""]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--report-name", default="20261105-interleaved-ulysses-validation.md")
    ap.add_argument("--backends", default="nccl,gfc")
    ap.add_argument("--shapes", default="480x832x33x20,480x832x81x50", help="HxWxFRAMESxSTEPS,...")
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--gfc-max-bytes", type=int, default=256 * 1024 * 1024)
    ap.add_argument("--visibility", choices=["controller", "all"], default="controller")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--model-path", default=WAN21_PATH)
    ap.add_argument("--port", type=int, default=29731)
    ap.add_argument("--timeout-s", type=int, default=1500, help="per backend")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("DITSERVE_RUN_ID", "local")
    uuids = gpu_uuids()
    shapes = [parse_shape(s) for s in args.shapes.split(",") if s]
    payloads = {s["name"]: stacked_qkv_bytes(s, 2) for s in shapes}
    summary = dict(schema=SCHEMA, date=time.strftime("%Y%m%d"), run_id=run_id, build=os.environ.get("DITSERVE_BUILD"),
                   gpus=len(uuids), shapes=shapes, payloads=payloads, arms=[list(a) for a in ARMS],
                   args=vars(args) | {"output_dir": str(args.output_dir)})
    if len(uuids) < 2:
        summary.update(verdict="BLOCKED", blocked=["needs 2 visible GPUs"])
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        return 2
    transfer = Path(os.environ.get("DITSERVE_TRANSFER_DIR") or "/dev/shm/ditserve-val-i7-" + run_id)
    transfer.mkdir(parents=True, exist_ok=True)
    cfg = dict(model_path=args.model_path, seed=args.seed, shapes=shapes, gfc_max_bytes=args.gfc_max_bytes,
               transfer_dir=str(transfer), run_id=run_id)
    summary["runs"], summary["backends"] = {}, {}
    for number, name in enumerate(b for b in args.backends.split(",") if b):
        port = args.port + 10 * number
        pair = run_pair(name, args.visibility, port, uuids, cfg, args.timeout_s)
        record = dict(visibility=pair["visibility"], exitcodes=pair["exitcodes"])
        init_errors = [r.get("init_error") for r in pair["ranks"] if r.get("init_error")]
        if (not pair["complete"] and init_errors and args.visibility == "controller" and not args.no_fallback):
            record["controller_visibility_init_error"] = init_errors[0]
            pair = run_pair(name, "all", port + 5, uuids, cfg, args.timeout_s)
            record.update(visibility="all (fallback)", exitcodes=pair["exitcodes"])
        record["ranks"] = [dict(rank=r["rank"], gpu_name=r.get("gpu_name"), load_ms=r.get("load_ms"),
                                processor=r.get("processor"), init=r.get("init"), collectives_info=r.get("collectives_info"),
                                errors=r["errors"], shapes=r["shapes"]) for r in pair["ranks"]]
        summary["runs"][name] = record
        summary["backends"][name] = evaluate(pair, cfg)
    procs = [r.get("processor") or {} for run in summary["runs"].values() for r in run["ranks"]]
    summary["has_flydsl"] = any(p.get("has_flydsl") for p in procs)
    n1 = [res.get("timing", {}).get(s["name"], {}).get("n1_digests") for res in summary["backends"].values() for s in shapes]
    summary["cross_backend_n1_equal"] = (len(summary["backends"]) > 1 and None not in n1
                                         and all(n1[i] == n1[i + len(shapes)] for i in range(len(shapes))))
    verdicts = [res["verdict"] for res in summary["backends"].values()]
    summary["verdict"] = "PASS" if verdicts and all(v == "PASS" for v in verdicts) else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_report(args.output_dir / args.report_name, summary)
    print(json.dumps(dict(verdict=summary["verdict"], backends={k: v["verdict"] for k, v in summary["backends"].items()})),
          flush=True)
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
