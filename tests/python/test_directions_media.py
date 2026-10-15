import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ditserve.directions.media import export_media


class MediaExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.encoded = []
        def encode(path, frames):
            self.encoded.append(frames.copy())
            path.write_bytes(b"CPU-test-video" + frames.tobytes())
        self.encoder = patch("ditserve.directions.media._write_mp4", side_effect=encode)
        self.encoder.start()
        self.addCleanup(self.encoder.stop)

    def report(self, direction, value, name="supervisor-phase.json"):
        folder = self.root / ("direction%d" % direction)
        folder.mkdir(exist_ok=True)
        (folder / name).write_text(json.dumps(value))

    def array(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, value, allow_pickle=False)
        return path

    def test_all_three_artifact_schemas_and_originals_remain_unchanged(self):
        pixels = np.linspace(0, 1, 7 * 4 * 6 * 3, dtype=np.float32).reshape(1, 7, 4, 6, 3)
        archive = self.root / "d1.npz"
        np.savez(archive, output=pixels, reference_output=pixels, latent=np.zeros((1, 2)))
        second = self.array("d2.npy", pixels)
        third = self.array("d3.npy", np.rint(pixels[0] * 255).astype(np.uint8))
        self.report(1, dict(rows=[dict(passed=True, artifact=str(archive))]))
        self.report(2, dict(units=[dict(retention=dict(files=[dict(path=str(second))]))]))
        self.report(3, dict(rows=[dict(artifact=dict(path=str(third)))]))
        before = {path: path.read_bytes() for path in (archive, second, third)}
        result = export_media(self.root)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["counts"], dict(direction1=2, direction2=1, direction3=1))
        self.assertEqual({row["member"] for row in result["videos"] if row["direction"] == 1},
                         {"output", "reference_output"})
        for record in result["videos"]:
            self.assertEqual(record["frame_count"], 7)
            self.assertEqual(record["fps"], 16)
            self.assertEqual(len(record["contact_sheet"]["frame_indices"]), 6)
            self.assertEqual(record["source_sha256"], hashlib.sha256(before[Path(record["source"])]).hexdigest())
            self.assertTrue(Path(record["video"]["path"]).is_file())
            self.assertTrue(Path(record["contact_sheet"]["path"]).is_file())
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertTrue(all(array.dtype == np.uint8 for array in self.encoded))
        self.assertEqual(self.encoded[0].shape, (7, 4, 6, 3))
        self.assertEqual(int(self.encoded[0].min()), 0)
        self.assertEqual(int(self.encoded[0].max()), 255)
        self.assertIn("not_blinded", result["quality_status"])

    def test_failure_priority_limit_dedup_and_sources_exclusion(self):
        pixels = np.zeros((1, 3, 4, 4, 3), dtype=np.float32)
        good, bad = self.array("a-good.npy", pixels), self.array("z-bad.npy", pixels)
        self.report(2, dict(status="failed", units=[dict(quality_passed=True, retention=dict(files=[dict(path=str(good))])),
                                       dict(quality_passed=False, retention=dict(files=[dict(path=str(bad))]))]))
        self.report(2, dict(artifacts=[dict(path=str(bad))]), "screen.json")
        history = self.root / "sources/old/direction1"
        history.mkdir(parents=True)
        (history / "report.json").write_text(json.dumps(dict(artifact=str(good))))
        result = export_media(self.root, limit=1)
        self.assertEqual(len(result["videos"]), 1)
        self.assertEqual(result["videos"][0]["source"], str(bad))
        self.assertTrue(result["videos"][0]["failure_priority"])
        self.assertEqual(result["referenced_arrays"], 2)
        repeated = export_media(self.root, limit=1)
        self.assertEqual(repeated["referenced_arrays"], 2)
        self.assertEqual(len(repeated["videos"]), 1)

    def test_actual_cpu_mp4_has_all_frames_and_manifest_digest(self):
        self.encoder.stop()
        import imageio.v2 as imageio
        pixels = np.zeros((1, 7, 8, 8, 3), dtype=np.uint8)
        pixels[0, :, :, :, 0] = np.arange(7, dtype=np.uint8)[:, None, None] * 30
        artifact = self.array("preview.npy", pixels)
        self.report(2, dict(retention=dict(files=[dict(path=str(artifact))])))
        result = export_media(self.root, limit=1)
        self.assertEqual(result["status"], "passed", result["errors"])
        record = result["videos"][0]
        with imageio.get_reader(record["video"]["path"]) as video:
            self.assertEqual(video.count_frames(), 7)
            self.assertEqual(video.get_meta_data()["fps"], 16.)
        unsigned = {key: value for key, value in result.items() if key != "manifest_sha256"}
        digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(result["manifest_sha256"], digest)

    def test_invalid_pixels_dtype_or_layout_are_not_clipped_or_encoded(self):
        for index, value in enumerate((np.full((1, 2, 4, 4, 3), 1.01, dtype=np.float32),
                                       np.full((1, 2, 4, 4, 3), np.nan, dtype=np.float32),
                                       np.ones((1, 2, 4, 4, 3), dtype=np.int16),
                                       np.zeros((2, 4, 4, 3), dtype=np.float32))):
            self.array("invalid%d.npy" % index, value)
        self.report(2, dict(artifacts=[dict(path=str(self.root / ("invalid%d.npy" % index))) for index in range(4)]))
        result = export_media(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["errors"]), 4)
        self.assertEqual(result["videos"], [])
        self.assertEqual(self.encoded, [])

    def test_unreferenced_arrays_and_invalid_limits_are_not_exported(self):
        self.array("unreferenced.npy", np.zeros((1, 1, 4, 4, 3), dtype=np.uint8))
        self.assertEqual(export_media(self.root)["status"], "blocked")
        for limit in (0, -1, 9, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                export_media(self.root, limit=limit)

    def test_complete_pairs_cover_stages_before_extra_samples(self):
        pixels = np.zeros((1, 7, 4, 4, 3), dtype=np.float32)
        for stage, cases in (("screen", ("a", "b")), ("qualify", ("c",)), ("actions", ("d",))):
            rows = []
            for case in cases:
                files = [dict(path=str(self.array("raw/%s/%s/%s.npy" % (stage, case, policy), pixels)))
                         for policy in ("off", "paired_008")]
                rows.append(dict(quality_passed=stage != "actions", retention=dict(files=files)))
            if stage == "qualify":
                rows.append(dict(item=dict(split="holdout"), retention=dict(raw_available=False,
                    reason="evaluated pair retained as metrics and hashes only")))
            self.report(2, dict(phase=stage, units=rows), stage + ".json")
        result = export_media(self.root, limit=6)
        self.assertEqual(result["counts"]["direction2"], 6)
        self.assertEqual(result["videos"][0]["stage"], "actions")
        self.assertEqual({row["stage"] for row in result["videos"]}, {"screen", "qualify", "actions"})
        groups = [g for g in result["selection_groups"] if g["selected"]]
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(g["kind"] == "full_pair" and g["video_count"] == 2 for g in groups))
        self.assertEqual(result["selection_summary"]["direction2"]["unselected_videos"], 2)
        self.assertEqual(len(result["raw_unavailable_records"]), 1)
        self.assertEqual(result["raw_unavailable_records"][0]["stage"], "qualify")
        self.assertIn("not exhaustive heldout/action", result["coverage_scope"])

    def test_d3_full_comparison_precedes_partial_chunks(self):
        pixels = np.zeros((7, 4, 4, 3), dtype=np.uint8)
        full = self.array("raw/correctness/sample.npy", pixels)
        reference = self.array("raw/correctness/sample-reference.npy", pixels)
        chunks = [self.array("raw/correctness/sample-chunk-%02d.npy" % i, pixels) for i in range(8)]
        self.report(3, dict(phase="correctness", passed=True, artifact=dict(path=str(full)),
                            reference_artifact=dict(path=str(reference))), "supervisor-correctness.json")
        self.report(3, dict(phase="correctness", status="incomplete",
                            artifacts=[dict(path=str(path)) for path in chunks]), "partial.json")
        result = export_media(self.root, limit=2)
        self.assertEqual({row["source"] for row in result["videos"]}, {str(full), str(reference)})
        self.assertEqual(result["selection_summary"]["direction3"]["unselected_videos"], 8)
        self.assertTrue(all(not g["selected"] for g in result["selection_groups"] if g["kind"] == "chunk"))

    def test_pair_does_not_split_to_fill_odd_quota(self):
        pixels = np.zeros((1, 7, 4, 4, 3), dtype=np.float32)
        paths = []
        for index in range(2):
            path = self.root / ("d1-%d.npz" % index)
            np.savez(path, output=pixels, reference_output=pixels)
            paths.append(str(path))
        self.report(1, dict(phase="correctness", artifacts=paths))
        result = export_media(self.root, limit=3)
        self.assertEqual(result["counts"]["direction1"], 2)
        self.assertEqual(result["selection_summary"]["direction1"]["unselected_videos"], 2)
        too_small = export_media(self.root, limit=1)
        self.assertEqual(too_small["status"], "blocked")
        self.assertEqual(too_small["videos"], [])

    def test_d1_real_correctness_names_keep_main_shape_and_tiny_pairs(self):
        shapes = [dict(height=256, width=256, frames=9), dict(height=480, width=832, frames=81)]
        # Identical small tensors keep this CPU fixture cheap and establish
        # that selection uses report metadata, not the decoded pixel sizes.
        pixels = np.zeros((1, 3, 4, 4, 3), dtype=np.float32)
        rows = []
        for shape_index, shape in enumerate(shapes):
            for sample in range(4):
                path = self.root / ("d1-correctness-%d-%d.npz" % (shape_index, sample))
                np.savez(path, output=pixels, reference_output=pixels)
                rows.append(dict(shape=shape, passed=True, artifact=str(path)))
        self.report(1, dict(phase="correctness", rows=rows), "direction1-correctness.json")

        smallest_preview = export_media(self.root, limit=2)
        self.assertEqual({Path(row["source"]).name for row in smallest_preview["videos"]},
                         {"d1-correctness-1-0.npz"})
        self.assertTrue(all(row["reported_shape"] == shapes[1] for row in smallest_preview["videos"]))
        self.assertTrue(all(row["shape"] == list(pixels.shape) for row in smallest_preview["videos"]))

        result = export_media(self.root, limit=8)
        self.assertEqual(result["counts"]["direction1"], 8)
        self.assertEqual([row["reported_shape"]["frames"] for row in result["videos"]],
                         [81, 81, 9, 9, 81, 81, 9, 9])
        groups = [group for group in result["selection_groups"] if group["selected"]]
        self.assertEqual(len(groups), 4)
        self.assertEqual({group["reported_shape"]["frames"] for group in groups}, {9, 81})
        self.assertTrue(all({row["member"] for row in group["sources"]} ==
                            {"output", "reference_output"} for group in groups))
        self.assertEqual(result["selection_summary"]["direction1"]["unselected_videos"], 8)

    def test_reported_shape_order_preserves_failure_priority_and_ignores_name_index(self):
        pixels = np.zeros((1, 3, 4, 4, 3), dtype=np.float32)
        rows = []
        # Reverse the real filename indices: neither index is a shape label.
        for index, frames in enumerate((81, 9)):
            path = self.root / ("d1-correctness-%d-0.npz" % index)
            np.savez(path, output=pixels, reference_output=pixels)
            rows.append(dict(shape=dict(height=480, width=832, frames=frames), passed=index == 0,
                             retained=dict(artifact=str(path))))
        self.report(1, dict(phase="correctness", rows=rows), "direction1-correctness.json")
        failed_first = export_media(self.root, limit=2)
        self.assertEqual([row["reported_shape"]["frames"] for row in failed_first["videos"]], [9, 9])
        self.assertTrue(all(row["failure_priority"] for row in failed_first["videos"]))

        rows[1]["passed"] = True
        self.report(1, dict(phase="correctness", rows=rows), "direction1-correctness.json")
        larger_first = export_media(self.root, limit=2)
        self.assertEqual([row["reported_shape"]["frames"] for row in larger_first["videos"]], [81, 81])
        self.assertEqual({Path(row["source"]).name for row in larger_first["videos"]},
                         {"d1-correctness-0-0.npz"})

    def test_bad_pair_member_exports_no_successful_half_pair(self):
        good = self.array("raw/screen/a/off.npy", np.zeros((1, 7, 4, 4, 3), dtype=np.float32))
        bad = self.array("raw/screen/a/paired_008.npy", np.full((1, 7, 4, 4, 3), 1.1, dtype=np.float32))
        self.report(2, dict(phase="screen", retention=dict(files=[dict(path=str(good)), dict(path=str(bad))])))
        result = export_media(self.root, limit=2)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["videos"], [])
        self.assertEqual(self.encoded, [])
        self.assertTrue(result["errors"][0]["whole_group_unqualified"])
        self.assertEqual(result["selection_summary"]["direction2"]["selected_but_not_exported"], 2)

    def test_missing_counterpart_and_missing_referenced_file_are_explicit(self):
        off = self.array("raw/qualify/partial/off.npy", np.zeros((1, 7, 4, 4, 3), dtype=np.float32))
        missing = self.root / "raw/actions/deleted.npy"
        self.report(2, dict(artifacts=[dict(path=str(off)), dict(path=str(missing))]))
        result = export_media(self.root)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_referenced_sources"], [str(missing)])
        self.assertEqual(len(result["missing_pair_counterparts"]), 1)
        self.assertTrue(result["missing_pair_counterparts"][0]["source"].endswith("paired_008.npy"))

    def state_cost_artifacts(self, failed=None, base="raw/state_cost", report_phase=True):
        pixels = np.zeros((1, 3, 4, 4, 3), dtype=np.float32)
        off = self.array(base + "/reference-off/off.npy", pixels)
        paths, units = {}, []
        for index in range(2):
            for action in ("keep", "full_cpu", "flush"):
                key = "round%d-%s" % (index, action)
                path = self.array(base + "/" + key + "/" + action + ".npy", pixels)
                paths[key] = path
                units.append(dict(unit_id=key, round=index, action=action, quality_passed=key != failed,
                                  retention=dict(files=[dict(path=str(path), policy=action)])))
        value = dict(reference_off=dict(retention=dict(files=[dict(path=str(off), policy="off")])), units=units)
        if report_phase:
            value["phase"] = "state_cost"
        self.report(2, value, "worker-state_cost.json")
        return off, paths

    def test_state_cost_groups_use_same_round_keep_and_shared_fresh_off(self):
        off, paths = self.state_cost_artifacts()
        result = export_media(self.root, limit=8)
        self.assertEqual(result["status"], "passed", result["errors"])
        self.assertEqual(result["counts"]["direction2"], 7)
        self.assertEqual(result["missing_pair_counterparts"], [])
        self.assertEqual(len(result["selection_groups"]), 6)
        self.assertTrue(all(group["selected"] and group["kind"] == "full_pair"
                            and group["stage"] == "state_cost" for group in result["selection_groups"]))
        exported = [row["source"] for row in result["videos"]]
        self.assertEqual(exported.count(str(off)), 1)
        for index in range(2):
            keep = paths["round%d-keep" % index]
            self.assertEqual(exported.count(str(keep)), 1)
            for action in ("full_cpu", "flush"):
                candidate = paths["round%d-%s" % (index, action)]
                matching = [group for group in result["selection_groups"]
                            if str(candidate) in {row["source"] for row in group["sources"]}]
                self.assertEqual(len(matching), 1)
                self.assertEqual({row["source"] for row in matching[0]["sources"]},
                                 {str(candidate), str(keep), str(off)})
        self.assertEqual(result["attempted_per_direction"]["direction2"], 7)

    def test_state_cost_underscore_stage_recognized_from_report_filename(self):
        self.state_cost_artifacts(base="external/cost-outputs", report_phase=False)
        result = export_media(self.root)
        self.assertEqual(result["counts"]["direction2"], 7)
        self.assertEqual(result["selection_summary"]["direction2"]["referenced_stages"], ["state_cost"])
        self.assertEqual(result["missing_pair_counterparts"], [])

    def test_state_cost_failure_first_triplet_stays_whole_and_shared_refs_do_not_use_extra_quota(self):
        off, paths = self.state_cost_artifacts(failed="round1-flush")
        result = export_media(self.root, limit=3)
        expected = {str(paths["round1-flush"]), str(paths["round1-keep"]), str(off)}
        self.assertEqual({row["source"] for row in result["videos"]}, expected)
        self.assertEqual(result["videos"][0]["source"], str(paths["round1-flush"]))
        self.assertTrue(result["videos"][0]["failure_priority"])
        self.assertTrue(all({row["source"] for row in group["sources"]} <= expected
                            for group in result["selection_groups"] if group["selected"]))
        self.assertEqual(result["attempted_per_direction"]["direction2"], 3)
        too_small = export_media(self.root, limit=1)
        self.assertEqual(too_small["videos"], [])
        self.assertEqual(too_small["status"], "blocked")

    def test_state_cost_unreferenced_keep_is_missing_even_when_file_exists(self):
        off, paths = self.state_cost_artifacts()
        candidate, keep = paths["round0-full_cpu"], paths["round0-keep"]
        self.report(2, dict(phase="state_cost", reference_off=dict(retention=dict(files=[dict(path=str(off))])),
                            units=[dict(action="full_cpu", round=0,
                                        retention=dict(files=[dict(path=str(candidate))]))]), "worker-state_cost.json")
        result = export_media(self.root)
        self.assertTrue(keep.is_file())
        self.assertEqual(result["referenced_arrays"], 2)
        self.assertEqual({row["source"] for row in result["videos"]}, {str(candidate), str(off)})
        self.assertEqual([row["source"] for row in result["missing_pair_counterparts"]], [str(keep)])
        self.assertEqual(len(result["selection_groups"]), 1)
        self.assertEqual(result["selection_groups"][0]["kind"], "incomplete_pair")
        self.assertNotIn("paired_008", json.dumps(result["missing_pair_counterparts"]))

    def test_state_cost_missing_referenced_keep_exports_no_successful_half_triplet(self):
        off, paths = self.state_cost_artifacts()
        candidate, keep = paths["round0-full_cpu"], paths["round0-keep"]
        self.report(2, dict(phase="state_cost", artifacts=[str(candidate), str(keep), str(off)]), "worker-state_cost.json")
        keep.unlink()
        result = export_media(self.root, limit=3)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["videos"], [])
        self.assertEqual(self.encoded, [])
        self.assertEqual(result["missing_referenced_sources"], [str(keep)])
        self.assertTrue(all(row["whole_group_unqualified"] for row in result["errors"]))

    def test_state_cost_lone_off_reports_missing_round_instead_of_inventing_paired_008(self):
        off = self.array("raw/state_cost/reference-off/off.npy", np.zeros((1, 3, 4, 4, 3), dtype=np.float32))
        # No explicit phase: the actual state_cost directory also identifies it.
        self.report(2, dict(reference_off=dict(retention=dict(files=[dict(path=str(off))]))), "partial.json")
        result = export_media(self.root)
        self.assertEqual(len(result["missing_pair_counterparts"]), 1)
        missing = result["missing_pair_counterparts"][0]
        self.assertEqual(missing["stage"], "state_cost")
        self.assertEqual(missing["role"], "state_cost_round_comparison")
        self.assertNotIn("paired_008", json.dumps(result["missing_pair_counterparts"]))


if __name__ == "__main__":
    unittest.main()
