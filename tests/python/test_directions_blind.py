"""CPU reconstruction and blinded package checks; never load a GPU model."""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve.directions import blind


class BlindTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy required")
        self.np = np
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report_path = self.root / "direction3/update/report.json"
        self.report_path.parent.mkdir(parents=True)
        self.path = self.report_path.parent / "source.npy"
        frames = np.zeros((3, 8, 16, 3), dtype=np.uint8)
        frames[0, :, :, 0] = 90
        frames[1, :, :, 1] = 160
        frames[2, :, :, 2] = 240  # Generated, but deliberately never presented.
        np.save(self.path, frames)
        self.original_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.sample = dict(policy="latest_only", buffer_chunks=2, seed=101, pattern="burst",
            observation_end_s=2., chunks=[dict(session_id="s0", block=0,
                artifact=dict(path=str(self.path), sha256=self.original_hash, frames=3))],
            playback={"s0": dict(fps=16, presentations=[
                dict(frame_id="s0-chunk-000-00", chunk=0, presented_s=.25, prompt_version=0),
                dict(frame_id="s0-chunk-000-01", chunk=0, presented_s=.75, prompt_version=2)])},
            updates={"s0": [dict(version=0, arrival_s=0., prompt="red cube", status="applied"),
                dict(version=1, arrival_s=.5, prompt="green cube", status="superseded_before_encoding"),
                dict(version=2, arrival_s=1.5, prompt="blue cube", status="pending_at_stop")]})
        self.report = dict(effective_config=dict(height=8, width=16), samples=[self.sample])
        self.report_path.write_text(json.dumps(self.report))

    @staticmethod
    def dummy_media(public, anonymous_id, window, sources):
        # The separate real-codec test covers writing; these tests inspect the
        # exact uncompressed reconstruction and package metadata.
        list(blind.window_frames(window, sources))
        return dict(video=anonymous_id + ".mp4", contact_sheet=anonymous_id + ".png"), {}

    def test_actual_presentations_black_start_and_hold_stalls(self):
        sources = blind.FrameSources(self.report_path)
        window = blind.reconstruct_window(self.sample, "s0", .5, 1., .5, sources, (8, 16))
        frames = list(blind.window_frames(window, sources))
        self.assertEqual(len(frames), 24)
        self.assertTrue(self.np.all(frames[3] == 0))
        self.assertEqual(frames[4][0, 0].tolist(), [90, 0, 0])
        self.assertEqual(frames[11][0, 0].tolist(), [90, 0, 0])
        self.assertEqual(frames[12][0, 0].tolist(), [0, 160, 0])
        self.assertEqual(frames[-1][0, 0].tolist(), [0, 160, 0])
        self.assertTrue(all(frame[0, 0, 2] == 0 for frame in frames))
        self.assertEqual(window["black_frames"], 4)
        self.assertEqual(window["sources"][0]["sha256"], self.original_hash)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.original_hash)

    def test_window_censors_at_observation_end_and_retains_prearrival_frame(self):
        window = blind.reconstruct_window(self.sample, "s0", 1.5, 8., .25,
                                           blind.FrameSources(self.report_path), (8, 16))
        self.assertEqual((window["start_s"], window["end_s"]), (1.25, 2.))
        self.assertTrue(window["censored"])
        self.assertTrue(all(index == 1 for index in window["selected"]))

    def test_package_keeps_all_events_and_public_contains_no_arm_labels(self):
        with patch.object(blind, "_write_media", side_effect=self.dummy_media):
            result = blind.export_blind_windows(self.root)
        public = Path(result["public_directory"])
        manifest = json.loads((public / "manifest.json").read_text())
        private = json.loads(Path(result["private_key"]).read_text())
        self.assertEqual(result["event_denominator"], 3)
        self.assertEqual(private["shuffle_seed"], 13013)
        self.assertEqual({row["original_event"]["status"] for row in private["items"]},
                         {"applied", "superseded_before_encoding", "pending_at_stop"})
        self.assertEqual({row["assessment_status"] for row in manifest["items"]}, {"unrated"})
        public_text = (public / "manifest.json").read_text()
        for forbidden in ("policy", "buffer", "seed", '"version":', "superseded", "conditioned", "source.npy"):
            self.assertNotIn(forbidden, public_text)
        with (public / "ratings.csv").open(newline="") as handle:
            ratings = list(csv.DictReader(handle))
        self.assertEqual(len(ratings), 3)
        self.assertTrue(all(value == "" for row in ratings for name, value in row.items() if name != "id"))
        mapped = {row["id"]: row["original_event"]["version"] for row in private["items"]}
        row = next(item for item in manifest["items"] if mapped[item["id"]] == 1)
        self.assertEqual(row["previous_prompt"], "red cube")
        self.assertEqual(row["desired_prompt"], "green cube")
        with self.assertRaises(FileExistsError):
            blind.export_blind_windows(self.root)

    def test_shuffle_is_reproducible_and_each_session_keeps_its_events(self):
        sample = self.report["samples"][0]
        sample["updates"]["s1"] = [dict(version=0, arrival_s=0., prompt="second session", status="pending_at_stop")]
        sample["playback"]["s1"] = dict(fps=16, presentations=[])
        self.report_path.write_text(json.dumps(self.report))
        with patch.object(blind, "_write_media", side_effect=self.dummy_media):
            first = blind.export_blind_windows(self.root)
        first_key = json.loads(Path(first["private_key"]).read_text())
        other = self.root / "copy"
        other_report = other / "direction3/update/report.json"
        other_report.parent.mkdir(parents=True)
        other_report.write_text(json.dumps(self.report))
        with patch.object(blind, "_write_media", side_effect=self.dummy_media):
            second = blind.export_blind_windows(other)
        second_key = json.loads(Path(second["private_key"]).read_text())
        order = lambda key: [(row["id"], row["session_id"], row["event_ordinal"]) for row in key["items"]]
        self.assertEqual(order(first_key), order(second_key))
        self.assertEqual(first["event_denominator"], 4)

    def test_missing_source_preserves_failed_unrated_denominator(self):
        self.path.unlink()
        with patch.object(blind, "_write_media", side_effect=self.dummy_media):
            result = blind.export_blind_windows(self.root)
        self.assertEqual(result["event_denominator"], 3)
        self.assertEqual(result["media_failures"], 3)
        self.assertEqual(result["status"], "partial_failure")
        manifest = json.loads((Path(result["public_directory"]) / "manifest.json").read_text())
        self.assertTrue(all(row["assessment_status"] == "unrated" and row["video"] is None for row in manifest["items"]))

    def test_no_presentation_exports_black_without_semantic_success(self):
        self.report["samples"][0]["playback"]["s0"]["presentations"] = []
        self.report_path.write_text(json.dumps(self.report))
        with patch.object(blind, "_write_media", side_effect=self.dummy_media):
            result = blind.export_blind_windows(self.root)
        private = json.loads(Path(result["private_key"]).read_text())
        self.assertEqual(result["event_denominator"], 3)
        self.assertEqual(result["semantic_status"], "unrated")
        self.assertTrue(all(row["no_presented_frames"] for row in private["items"]))

    def test_bad_hash_nonfinite_and_wrong_fps_fail_validation(self):
        sample = copy.deepcopy(self.sample)
        sample["chunks"][0]["artifact"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            blind.reconstruct_window(sample, "s0", .5, 1., .5, blind.FrameSources(self.report_path), (8, 16))
        sample = copy.deepcopy(self.sample)
        sample["playback"]["s0"]["fps"] = 8
        with self.assertRaisesRegex(ValueError, "16fps"):
            blind.reconstruct_window(sample, "s0", .5, 1., .5, blind.FrameSources(self.report_path), (8, 16))
        bad = self.np.ones((3, 8, 16, 3), dtype=self.np.float32)
        bad[0, 0, 0, 0] = self.np.nan
        self.np.save(self.path, bad)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            blind.FrameSources(self.report_path).load(dict(path=str(self.path)))

    def test_real_cpu_video_and_contact_sheet_export(self):
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg
            from PIL import Image
        except ImportError:
            self.skipTest("imageio/ffmpeg/Pillow required")
        window = blind.reconstruct_window(self.sample, "s0", .25, .25, 0.,
                                           blind.FrameSources(self.report_path), (8, 16))
        media, hashes = blind._write_media(self.report_path.parent, "W00001", window,
                                            blind.FrameSources(self.report_path))
        path = self.report_path.parent / media["video"]
        with imageio.get_reader(path) as reader:
            self.assertEqual(reader.get_meta_data()["fps"], 16.)
            self.assertEqual(reader.count_frames(), 4)
        with Image.open(self.report_path.parent / media["contact_sheet"]) as sheet:
            self.assertGreater(sheet.width, 0)
        self.assertEqual(hashes["video_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
