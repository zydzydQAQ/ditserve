import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ditserve.benchmark.common import verify_seal
from ditserve.benchmark.datasets import OFFICIAL, prepare_dataset


class DownloadResponse(io.BytesIO):
    def __init__(self, blob, content_length=None):
        super().__init__(blob)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}


def github_metadata(blob):
    return {"type": "file", "size": len(blob),
            "sha": hashlib.sha1(("blob %d\0" % len(blob)).encode() + blob).hexdigest()}


def hf_metadata(blob):
    return [{"type": "file", "path": "VidProM_unique.csv", "size": len(blob),
             "oid": "b" * 40, "lfs": {"oid": hashlib.sha256(blob).hexdigest(), "size": len(blob)}}]


class BenchmarkDatasetTests(unittest.TestCase):
    def test_prompt_whitespace_is_preserved_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vbench.json"
            path.write_text(json.dumps([{"prompt_en": "  cat\n"}, {"prompt_en": "dog"}]))
            dataset = prepare_dataset("vbench", source=path, dev_count=1)
            self.assertEqual(dataset["samples"][0]["prompt"], "  cat\n")
            self.assertIn("original whitespace preserved", dataset["selection"]["prompt_transform"])

    def test_vbench_preserves_every_evaluation_row_and_separates_service_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vbench.json"
            rows = [{"prompt_en": "prompt %d" % n, "dimension": ["subject_consistency"],
                     "auxiliary_info": {"object": str(n)}} for n in range(8)]
            rows.append(dict(rows[0]))
            path.write_text(json.dumps(rows))
            dataset = prepare_dataset("vbench", source=path, dev_count=2, seed=17)
            self.assertEqual(dataset, prepare_dataset("vbench", source=path, dev_count=2, seed=17))
            verify_seal(dataset)
            self.assertEqual(len(dataset["samples"]), len(rows))
            self.assertTrue(all(row["split"] == "evaluation" for row in dataset["samples"]))
            self.assertEqual(dataset["samples"][0]["metadata"], rows[0])
            self.assertEqual(dataset["samples"][0]["service_split"], dataset["samples"][-1]["service_split"])
            dev = {row["prompt"] for row in dataset["samples"] if row["service_split"] == "dev"}
            test = {row["prompt"] for row in dataset["samples"] if row["service_split"] == "test"}
            self.assertEqual(len(dev), 2)
            self.assertFalse(dev & test)
            self.assertEqual(dataset["selection"]["method"], "full_official_list")
            self.assertIsNone(dataset["source"]["revision"])

    def test_vidprom_bounded_sample_keeps_timestamps_and_ignores_video_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompts.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["uuid", "prompt", "time", "video"])
                writer.writeheader()
                for index in range(250):
                    writer.writerow({"uuid": str(index), "prompt": "scene,%d\nnew line" % index,
                                     "time": "2024-01-01T00:00:00Z", "video": "/missing/video.mp4"})
            with patch("urllib.request.urlopen", side_effect=AssertionError("local import must not network")):
                dataset = prepare_dataset("vidprom", source=path, dev_count=3, seed=7, sample_count=11)
                repeated = prepare_dataset("vidprom", source=path, dev_count=3, seed=7, sample_count=11)
            self.assertEqual(dataset, repeated)
            self.assertEqual(dataset["selection"]["rows_seen"], 250)
            self.assertEqual(len(dataset["samples"]), 11)
            self.assertEqual(sum(sample["split"] == "dev" for sample in dataset["samples"]), 3)
            self.assertTrue(all(sample["timestamp"] == "2024-01-01T00:00:00Z" for sample in dataset["samples"]))
            self.assertTrue(all("video" not in sample for sample in dataset["samples"]))
            self.assertEqual(len(dataset["source"]["sha256"]), 64)

    def test_download_is_opt_in_and_pins_metadata_url(self):
        with self.assertRaises(ValueError):
            prepare_dataset("vbench")
        with tempfile.TemporaryDirectory() as folder:
            blob = json.dumps([{"prompt_en": "a"}, {"prompt_en": "b"}]).encode()
            with patch("ditserve.benchmark.datasets._json_url", return_value=github_metadata(blob)) as metadata, \
                    patch("urllib.request.urlopen", return_value=DownloadResponse(blob, len(blob))) as fetch:
                result = prepare_dataset("vbench", download=True, cache_dir=folder, dev_count=1)
            url = fetch.call_args[0][0].full_url
            self.assertIn(OFFICIAL["vbench"]["default_ref"], url)
            self.assertTrue(url.endswith("/vbench/VBench_full_info.json"))
            self.assertEqual(result["source"]["revision"], OFFICIAL["vbench"]["default_ref"])
            self.assertTrue(result["source"]["downloaded_metadata_only"])
            self.assertIn("/contents/vbench/VBench_full_info.json?ref=", metadata.call_args[0][0])
            self.assertEqual(result["source"]["integrity"]["hash_algorithm"], "git_blob_sha1")
            self.assertEqual(result["source"]["integrity"]["status"], "verified")

    def test_vidprom_named_ref_resolves_to_immutable_commit_before_fetch(self):
        with tempfile.TemporaryDirectory() as folder:
            sha = "a" * 40
            blob = b"uuid,prompt,time\na,cat,1\nb,dog,2\n"
            with patch("ditserve.benchmark.datasets._json_url", side_effect=[{"sha": sha}, hf_metadata(blob)]) as resolve, \
                    patch("urllib.request.urlopen", return_value=DownloadResponse(blob, len(blob))) as fetch:
                result = prepare_dataset("vidprom", download=True, cache_dir=folder, dev_count=1)
            self.assertIn("/revision/main", resolve.call_args_list[0][0][0])
            self.assertIn("/tree/" + sha, resolve.call_args_list[1][0][0])
            self.assertIn("/resolve/" + sha + "/VidProM_unique.csv", fetch.call_args[0][0].full_url)
            self.assertEqual(result["source"]["revision"], sha)
            self.assertEqual(result["source"]["integrity"]["expected_hash"], hashlib.sha256(blob).hexdigest())

    def test_truncated_csv_is_never_parsed_and_partial_evidence_is_preserved(self):
        blob = b"uuid,prompt,time\na,cat,1\nb,dog,222\n"
        truncated = blob[:-2]
        for declared_length in (None, len(blob)):
            with self.subTest(declared_length=declared_length), tempfile.TemporaryDirectory() as folder:
                with patch("ditserve.benchmark.datasets._json_url", return_value=hf_metadata(blob)), \
                        patch("urllib.request.urlopen", return_value=DownloadResponse(truncated, declared_length)), \
                        patch("ditserve.benchmark.datasets._vidprom", side_effect=AssertionError("do not parse incomplete CSV")):
                    with self.assertRaisesRegex(ValueError, "size mismatch|Content-Length mismatch"):
                        prepare_dataset("vidprom", revision="a" * 40, download=True, cache_dir=folder, dev_count=1)
                cache = Path(folder) / "vidprom" / ("a" * 40)
                self.assertFalse((cache / "VidProM_unique.csv").exists())
                partials = list(cache.glob("*.partial"))
                self.assertEqual(len(partials), 1)
                self.assertEqual(partials[0].read_bytes(), truncated)
                receipt = json.loads(Path(str(partials[0]) + ".error.json").read_text())
                self.assertEqual(receipt["status"], "untrusted")
                self.assertEqual(receipt["actual"]["bytes"], len(truncated))
                self.assertEqual(receipt["expected"]["expected_bytes"], len(blob))

    def test_same_size_hash_corruption_and_http_length_mismatch_are_rejected(self):
        blob = b"uuid,prompt,time\na,cat,1\nb,dog,2\n"
        cases = ((blob.replace(b"cat", b"bat"), len(blob), "sha256 mismatch"),
                 (blob, len(blob) + 7, "Content-Length mismatch"))
        for actual, declared, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                with patch("ditserve.benchmark.datasets._json_url", return_value=hf_metadata(blob)), \
                        patch("urllib.request.urlopen", return_value=DownloadResponse(actual, declared)):
                    with self.assertRaisesRegex(ValueError, reason):
                        prepare_dataset("vidprom", revision="a" * 40, download=True, cache_dir=folder, dev_count=1)
                self.assertEqual(len(list(Path(folder).rglob("*.partial"))), 1)

    def test_cache_hits_are_verified_and_corrupt_cache_is_preserved_before_retry(self):
        blob = b"uuid,prompt,time\na,cat,1\nb,dog,2\n"
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "vidprom" / ("a" * 40) / "VidProM_unique.csv"
            target.parent.mkdir(parents=True)
            bad_cache = blob.replace(b"cat", b"bat")
            target.write_bytes(bad_cache)
            with patch("ditserve.benchmark.datasets._json_url", return_value=hf_metadata(blob)) as metadata, \
                    patch("urllib.request.urlopen", return_value=DownloadResponse(blob, len(blob))) as fetch:
                result = prepare_dataset("vidprom", revision="a" * 40, download=True, cache_dir=folder, dev_count=1)
                repeated = prepare_dataset("vidprom", revision="a" * 40, download=True, cache_dir=folder, dev_count=1)
            self.assertEqual(result, repeated)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(metadata.call_count, 2)
            self.assertEqual(target.read_bytes(), blob)
            evidence = list(target.parent.glob("*.corrupt"))
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0].read_bytes(), bad_cache)
            receipt = json.loads(Path(str(evidence[0]) + ".error.json").read_text())
            self.assertIn("sha256 mismatch", receipt["reason"])

    def test_failed_retry_preserves_every_attempt_without_overwriting_evidence(self):
        blob = b"uuid,prompt,time\na,cat,1\nb,dog,2\n"
        with tempfile.TemporaryDirectory() as folder:
            with patch("ditserve.benchmark.datasets._json_url", return_value=hf_metadata(blob)), \
                    patch("urllib.request.urlopen", side_effect=[DownloadResponse(blob[:12]), DownloadResponse(blob[:14])]):
                for unused in range(2):
                    with self.assertRaisesRegex(ValueError, "size mismatch"):
                        prepare_dataset("vidprom", revision="a" * 40, download=True, cache_dir=folder, dev_count=1)
            partials = list(Path(folder).rglob("*.partial"))
            self.assertEqual({path.read_bytes() for path in partials}, {blob[:12], blob[:14]})
            self.assertEqual(len(list(Path(folder).rglob("*.error.json"))), 2)

    def test_missing_lfs_identity_cannot_fall_back_to_trusting_content_length(self):
        blob = b"uuid,prompt,time\na,cat,1\nb,dog,2\n"
        info = hf_metadata(blob)
        del info[0]["lfs"]
        with tempfile.TemporaryDirectory() as folder:
            with patch("ditserve.benchmark.datasets._json_url", return_value=info), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("missing LFS identity must stop before transfer")):
                with self.assertRaisesRegex(ValueError, "LFS"):
                    prepare_dataset("vidprom", revision="a" * 40, download=True, cache_dir=folder, dev_count=1)

    def test_invalid_metadata_does_not_silently_drop_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for rows in ([{"prompt_en": ""}], [{"prompt_en": "a"}], {"unexpected": []}):
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    prepare_dataset("vbench", source=path, dev_count=1)


if __name__ == "__main__":
    unittest.main()
