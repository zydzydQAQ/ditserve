"""Tests for S4 C++ controller image_text_prefetch, prepare telemetry, and per-cell /dev/shm isolation."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from ditserve import api
from ditserve.runtime import PACKAGE_ROOT, ROOT
from ditserve.runtime.client import CoreBridge


class FixtureBridge(CoreBridge):
    def __init__(self, package_root):
        super().__init__()
        self.package_root = package_root

    def call(self, command, payload):
        return super().call(command, dict(payload, package_root=str(self.package_root)))


@unittest.skipUnless(CoreBridge().available(), "build the C++ core first")
class ImagePrefetchAndIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.package_directory.cleanup)
        cls.package_root = Path(cls.package_directory.name) / "python"
        for package in ("ditserve", "xfuser"):
            source = PACKAGE_ROOT / package
            for path in source.rglob("*.py"):
                target = cls.package_root / package / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.bridge = FixtureBridge(self.package_root)
        paths = [str(ROOT / "workers")]
        if importlib.util.find_spec("openpyxl") is None:
            paths.append(str(ROOT / ".cache/tools"))
        if os.environ.get("PYTHONPATH"):
            paths.append(os.environ["PYTHONPATH"])
        environment = mock.patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join(paths)})
        environment.start()
        self.addCleanup(environment.stop)

    def config(self, phases, workers):
        return dict(
            schema="ditserve.study.v1",
            experiment="s4-prefetch-isolation",
            workers=workers,
            budget=dict(limit_gpu_seconds=None, watchdog_seconds=15),
            phases=phases,
        )

    def run_study(self, config, name="run"):
        return api.run(
            config,
            run_id=name,
            work_dir=self.root / name,
            results_root=self.root / "results",
            ledger=self.root / (name + "-budget.json"),
            core=self.bridge,
        )

    @staticmethod
    def base_profile():
        return dict(
            provenance="synthetic",
            links={"2": dict(gbps=10, latency_ms=0)},
            classes=dict(
                video=dict(
                    kind="video",
                    steps=2,
                    blocks_per_step=4,
                    prepare_ms=5,
                    encode_ms=10,
                    step_compute_ms=80,
                    comm_mb_per_step=0,
                    collectives_per_step=0,
                    finish_ms=5,
                ),
                image=dict(
                    kind="image",
                    service_ms=20,
                    steps=2,
                    prepare_ms=4,
                    finish_ms=4,
                    batch_step_ms={"1": 6, "2": 9, "4": 15, "8": 25},
                ),
            ),
        )

    def test_continuous_batch_image_text_prefetch_and_telemetry(self):
        """When image_text_prefetch is true, arriving images run encode_start -> encode_poll -> prepare(encode_ticket) and record telemetry."""
        shm_base = self.root / "shm-pool"
        workers = {
            "v0": {"backend": "mock", "config": {"options": {"costs": {"step_ms": 20, "finish_ms": 2}}}},
            "i0": {
                "backend": "mock",
                "config": {
                    "options": {
                        "costs": {"encode_ms": 5, "prepare_ms": 4, "step_ms": 3, "finish_ms": 2},
                        "encoder_pool": {"enabled": True, "write_shm": True, "shm_dir": str(shm_base)},
                    }
                },
            },
        }
        phase = dict(
            id="cb-prefetch",
            kind="cluster",
            policy="ditserve",
            options=dict(continuous_batch=4, bubble_free=True, image_text_prefetch=True),
            gpus=[dict(video="v0", image="i0")],
            profile=self.base_profile(),
            workload=[
                dict(
                    request_id="img-0",
                    kind="image",
                    worker="i0",
                    **{"class": "image"},
                    arrival_ms=0,
                    slo_ms=1000,
                    spec=dict(prompt="a scenic mountain lake", steps=2, seed=11),
                ),
                dict(
                    request_id="img-1",
                    kind="image",
                    worker="i0",
                    **{"class": "image"},
                    arrival_ms=0,
                    slo_ms=1000,
                    spec=dict(prompt="a futuristic city skyline", steps=2, seed=22),
                ),
            ],
        )
        result = self.run_study(self.config([phase], workers=workers), name="cb-prefetch")
        self.assertEqual(result["status"], "completed", json.dumps(result.get("failures", result), indent=2))
        cell = result["phases"][0]["cells"][0]
        rows = {r["request_id"]: r for r in cell["requests"]}
        self.assertEqual(set(rows.keys()), {"img-0", "img-1"})
        for rid, row in rows.items():
            self.assertEqual(row["status"], "completed", rid)
            self.assertIn("encode_submit_ms", row, rid)
            self.assertIn("encode_ready_ms", row, rid)
            self.assertLessEqual(row["encode_submit_ms"], row["encode_ready_ms"], rid)
            self.assertLessEqual(row["encode_ready_ms"], row["start_ms"], rid)
            self.assertEqual(row.get("encode_worker"), "i0", rid)
            self.assertEqual(row.get("encode_gpu"), 0, rid)
            self.assertTrue(row.get("encode_ticket_used"), rid)
            self.assertFalse(row.get("encode_cross_worker"), rid)
            self.assertIn("prepare_cb_ms", row, rid)
            self.assertGreater(row["prepare_cb_ms"], 0.0, rid)
            self.assertIn("prepare_cb_wall_ms", row, rid)
            self.assertGreater(row["prepare_cb_wall_ms"], 0.0, rid)
            self.assertIn("encoder_pool_stage", row, rid)
            stage = row["encoder_pool_stage"]
            self.assertIsInstance(stage, dict)
            self.assertTrue(stage.get("encode_ticket_used"), rid)
            self.assertTrue(row.get("encoder_pool_hit"), rid)
            self.assertIn("stage_ms", row, rid)

    def test_image_text_prefetch_default_false_skips_encode_rpc(self):
        """By default (image_text_prefetch=false), image requests go directly to waiting without encode_start/encode_poll."""
        workers = {
            "v0": {"backend": "mock", "config": {"options": {"costs": {"step_ms": 20, "finish_ms": 2}}}},
            "i0": {"backend": "mock", "config": {"options": {"costs": {"encode_ms": 5, "prepare_ms": 3, "step_ms": 2, "finish_ms": 1}}}},
        }
        phase = dict(
            id="cb-default",
            kind="cluster",
            policy="ditserve",
            options=dict(continuous_batch=2),
            gpus=[dict(video="v0", image="i0")],
            profile=self.base_profile(),
            workload=[
                dict(
                    request_id="img-no-pref",
                    kind="image",
                    worker="i0",
                    **{"class": "image"},
                    arrival_ms=0,
                    slo_ms=1000,
                    spec=dict(prompt="direct prepare prompt", steps=2, seed=3),
                )
            ],
        )
        result = self.run_study(self.config([phase], workers=workers), name="cb-no-prefetch")
        self.assertEqual(result["status"], "completed")
        row = result["phases"][0]["cells"][0]["requests"][0]
        self.assertEqual(row["status"], "completed")
        self.assertNotIn("encode_submit_ms", row)
        self.assertNotIn("encode_ready_ms", row)
        self.assertIn("prepare_cb_ms", row)
        self.assertGreater(row["prepare_cb_ms"], 0.0)

    def test_atomic_image_prefetch_and_overlap_with_active_video(self):
        """Image encode_start/encode_poll runs on image0 concurrently while video0 is busy executing a video quantum on GPU 0."""
        workers = {
            "v0": {"backend": "mock", "config": {"options": {"costs": {"prepare_ms": 2, "block_ms": 15, "finish_ms": 2}}}},
            "i0": {"backend": "mock", "config": {"options": {"costs": {"encode_ms": 5, "prepare_ms": 2, "step_ms": 2, "finish_ms": 1}}}},
        }
        phase = dict(
            id="atomic-overlap",
            kind="cluster",
            policy="ditserve",
            text_prefetch=0,
            options=dict(image_text_prefetch=True, quantum_blocks=4),
            gpus=[dict(video="v0", image="i0")],
            profile=self.base_profile(),
            workload=[
                dict(
                    request_id="vid-0",
                    kind="video",
                    worker="v0",
                    **{"class": "video"},
                    arrival_ms=0,
                    slo_ms=2000,
                    spec=dict(prompt="video prompt", steps=2, blocks_per_step=4, seed=1),
                ),
                dict(
                    request_id="img-overlap",
                    kind="image",
                    worker="i0",
                    **{"class": "image"},
                    arrival_ms=5,
                    slo_ms=500,
                    spec=dict(prompt="overlapping image prompt", steps=2, seed=2),
                ),
            ],
        )
        result = self.run_study(self.config([phase], workers=workers), name="atomic-overlap")
        self.assertEqual(result["status"], "completed")
        rows = {r["request_id"]: r for r in result["phases"][0]["cells"][0]["requests"]}
        img = rows["img-overlap"]
        self.assertEqual(img["status"], "completed")
        self.assertIn("encode_submit_ms", img)
        self.assertIn("encode_ready_ms", img)
        # Image arrives at 5ms while vid-0 is inside its first 60ms quantum on GPU 0;
        # encode_start and encode_poll complete on i0 BEFORE GPU 0 finishes vid-0's quantum!
        self.assertLess(img["encode_ready_ms"], img["start_ms"])
        self.assertTrue(img.get("encode_ticket_used"))
        self.assertIn("encoder_pool_stage", img)
        self.assertTrue(img["encoder_pool_stage"].get("encode_ticket_used"))

    def test_cross_worker_image_prefetch_falls_back_cleanly(self):
        """If an image is encoded on GPU 0 (i0) and prepared on GPU 1 (i1), controller omits worker-local ticket and records shm adoption."""
        workers = {
            "v0": {"backend": "mock", "config": {"options": {"costs": {"step_ms": 20, "finish_ms": 2}}}},
            "v1": {"backend": "mock", "config": {"options": {"costs": {"step_ms": 20, "finish_ms": 2}}}},
            "i0": {"backend": "mock", "config": {"options": {"costs": {"encode_ms": 2, "prepare_ms": 2, "step_ms": 10, "finish_ms": 2}}}},
            "i1": {"backend": "mock", "config": {"options": {"costs": {"encode_ms": 2, "prepare_ms": 2, "step_ms": 10, "finish_ms": 2}}}},
        }
        phase = dict(
            id="cross-worker",
            kind="cluster",
            policy="static_sp1",
            options=dict(
                sp_transport="resume",
                reserve_image_gpus=1,
                strict_partition=True,
                image_text_prefetch=True,
                continuous_batch=1,
            ),
            gpus=[
                dict(video="v0", image="i0"),
                dict(video="v1", image="i1"),
            ],
            profile=self.base_profile(),
            workload=[
                dict(
                    request_id="img-first",
                    kind="image",
                    worker="i0",
                    **{"class": "image"},
                    arrival_ms=0,
                    slo_ms=1000,
                    spec=dict(prompt="first image", steps=2, seed=10),
                ),
                dict(
                    request_id="img-second",
                    kind="image",
                    worker="i0",
                    **{"class": "image"},
                    arrival_ms=2,
                    slo_ms=1000,
                    spec=dict(prompt="second image", steps=2, seed=20),
                ),
            ],
        )
        result = self.run_study(self.config([phase], workers=workers), name="cross-worker")
        self.assertEqual(result["status"], "completed")
        rows = {r["request_id"]: r for r in result["phases"][0]["cells"][0]["requests"]}
        # Under strict_partition with reserve_image_gpus=1 on 2 GPUs, only GPU 1 (i1) serves image denoising,
        # while pick_image_encode_gpu starts at GPU 0 (i0) for img-first!
        first = rows["img-first"]
        self.assertEqual(first["encode_gpu"], 0)
        self.assertEqual(first["gpu_group"], [1])
        self.assertFalse(first["encode_ticket_used"])
        self.assertTrue(first["encode_cross_worker"])
        self.assertTrue(first["shm_adopted"])
        self.assertTrue(first["encoder_pool_stage"]["shm_adopted"])

    def test_per_cell_shm_pool_isolation_and_cleanup(self):
        """Decision D4: each matrix cell gets an isolated per-cell shm_dir and cleans it up after the cell completes."""
        shm_base = self.root / "shared-shm-root"
        shm_base.mkdir(parents=True, exist_ok=True)
        workers = {
            "v0": {
                "backend": "mock",
                "config": {
                    "options": {
                        "costs": {"step_ms": 10, "finish_ms": 2},
                        "encoder_pool": {"enabled": True, "write_shm": True, "shm_dir": str(shm_base)},
                    }
                },
            },
            "i0": {
                "backend": "mock",
                "config": {
                    "options": {
                        "costs": {"encode_ms": 1, "prepare_ms": 1, "step_ms": 1, "finish_ms": 1},
                        "encoder_pool": {"enabled": True, "write_shm": True, "shm_dir": str(shm_base)},
                    }
                },
            },
        }
        phase = dict(
            id="policy-matrix",
            kind="cluster",
            policy="ditserve",
            options=dict(continuous_batch=2, image_text_prefetch=True),
            gpus=[dict(video="v0", image="i0")],
            profile=self.base_profile(),
            matrix={"/policy": ["ditserve", "static_sp1"]},
            workload=[
                dict(
                    request_id="img-cell",
                    kind="image",
                    worker="i0",
                    **{"class": "image"},
                    arrival_ms=0,
                    slo_ms=1000,
                    spec=dict(prompt="shared prompt across cells", steps=2, seed=7),
                )
            ],
        )
        run_id = "shm-isolation"
        result = self.run_study(self.config([phase], workers=workers), name=run_id)
        self.assertEqual(result["status"], "completed")
        cells = result["phases"][0]["cells"]
        self.assertEqual(len(cells), 2)
        shm_dirs = []
        for idx, cell in enumerate(cells):
            self.assertIn("encoder_pool_isolation", cell)
            iso = cell["encoder_pool_isolation"]
            self.assertTrue(iso["isolated"])
            self.assertTrue(iso["cleaned_after_cell"])
            self.assertEqual(len(iso["shm_dirs"]), 1)
            cell_shm = Path(iso["shm_dirs"][0])
            shm_dirs.append(cell_shm)
            self.assertIn(f"cell-{idx}", str(cell_shm))
            # Verify the directory was cleaned up after the cell stopped
            self.assertFalse(cell_shm.exists(), f"{cell_shm} should be removed after cell {idx}")
            # Verify the materialized worker config in cell-N/workers/i0.json had the isolated shm_dir
            worker_cfg_path = self.root / run_id / "attempts" / "policy-matrix" / "attempt-1" / f"cell-{idx}" / "workers" / "i0.json"
            worker_cfg = json.loads(worker_cfg_path.read_text(encoding="utf-8"))
            pool_cfg = worker_cfg["options"]["encoder_pool"]
            self.assertEqual(pool_cfg["shm_dir"], str(cell_shm))
            self.assertEqual(pool_cfg["base_shm_dir"], str(shm_base))
            self.assertFalse(pool_cfg["per_cell_isolation"])
            self.assertTrue(pool_cfg["cell_scoped"])
        self.assertNotEqual(shm_dirs[0], shm_dirs[1])


if __name__ == "__main__":
    unittest.main()
