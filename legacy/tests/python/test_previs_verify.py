"""Small CPU-only artifact fixtures; no real experiment arrays are scanned."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ditserve.experiments import previs_verify as verifier


class ArtifactVerificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="previs-verify-fixture-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.suite = self.directory / "suite"
        self.suite.mkdir()

    def artifact(self, name, array=None, attempt="attempt"):
        array = np.arange(12, dtype=np.float32).reshape(3, 4) if array is None else array
        path = self.suite / attempt / "arrays" / (name + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
        return {"path": str(path), "shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(), "finite": bool(np.isfinite(array).all())}

    def records(self, rows, attempt="attempt"):
        path = self.suite / attempt / "records.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        return path

    def correctness(self, *, latents=True, altered=False, attempt="attempt"):
        official = {"type": "correctness", "case_id": "official-none", "cache": False, "status": "completed",
                    "output_exact_match": True, "latent_exact_match": True,
                    "output": self.artifact("official", attempt=attempt),
                    "candidate": self.artifact("baseline", attempt=attempt)}
        resumed = {"type": "correctness", "case_id": "none-hot-pause-50", "cache": False, "status": "completed",
                   "output_exact_match": True, "latent_exact_match": True,
                   "output": self.artifact("resumed", np.full((3, 4), 9, dtype=np.float32) if altered else None, attempt)}
        if latents:
            latent = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
            official["latent_reference"] = self.artifact("official-latent", latent, attempt)
            official["latent_candidate"] = self.artifact("baseline-latent", latent, attempt)
            resumed["latent"] = self.artifact("resumed-latent", latent, attempt)
        self.records([official, resumed], attempt)
        return official, resumed

    def test_recursive_references_deduplicate_physical_file_and_preserve_raw(self):
        artifact = self.artifact("same")
        alias = Path(artifact["path"]).with_name("alias.npy")
        os.link(artifact["path"], alias)
        symlink = alias.with_name("symlink.npy")
        symlink.symlink_to(alias)
        rows = [{"outer": [artifact, {"nested": dict(artifact, path=str(alias))}, dict(artifact, path=str(symlink))]}]
        record_path = self.records(rows)
        raw = record_path.read_bytes()
        npy_bytes = Path(artifact["path"]).read_bytes()
        with mock.patch.object(verifier, "_inspect", wraps=verifier._inspect) as inspect:
            report = verifier.verify_suite(self.suite)
        self.assertEqual(inspect.call_count, 1)
        self.assertEqual(report["summary"]["physical_arrays"], 1)
        self.assertEqual(report["summary"]["artifact_references"], 3)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(record_path.read_bytes(), raw)
        self.assertEqual(Path(artifact["path"]).read_bytes(), npy_bytes)
        self.assertTrue((self.suite / "verification.json").is_file())

    def test_c_order_array_hash_excludes_file_header_and_handles_fortran_arrays(self):
        array = np.asfortranarray(np.arange(24, dtype=np.float32).reshape(4, 6))
        artifact = self.artifact("fortran", array)
        self.assertNotEqual(artifact["sha256"], hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest())
        self.records([{"output": artifact}])
        with mock.patch.object(verifier, "CHUNK_BYTES", 20):
            report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["arrays"][0]["actual"]["sha256"], artifact["sha256"])

    def test_conflicting_metadata_for_same_file_fails_even_if_one_claim_is_correct(self):
        artifact = self.artifact("conflict")
        self.records([{"first": artifact, "second": dict(artifact, sha256="0" * 64)}])
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        self.assertIn("conflicting_metadata_for_same_physical_file", report["arrays"][0]["errors"])
        self.assertEqual(report["summary"]["physical_arrays"], 1)

    def test_shape_dtype_finite_and_hash_mismatches_each_fail(self):
        descriptors = []
        for name, field, value in (("shape", "shape", [12]), ("dtype", "dtype", "float64"),
                                   ("finite", "finite", False), ("hash", "sha256", "0" * 64)):
            descriptors.append(dict(self.artifact(name), **{field: value}))
        self.records([{"all": descriptors}])
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["summary"]["array_statuses"]["failed"], 4)
        reasons = {issue["reason"] for issue in report["issues"]}
        self.assertTrue({"metadata_mismatch_" + key for key in ("shape", "dtype", "finite", "sha256")}.issubset(reasons))

    def test_external_file_and_symlink_escape_are_rejected_without_loading(self):
        artifact = self.artifact("inside")
        outside = self.directory / "outside.npy"
        np.save(outside, np.arange(12, dtype=np.float32).reshape(3, 4))
        link = self.suite / "escape.npy"
        link.symlink_to(outside)
        self.records([{"outside": dict(artifact, path=str(outside)), "link": dict(artifact, path=str(link))}])
        with mock.patch.object(verifier, "_inspect", wraps=verifier._inspect) as inspect:
            report = verifier.verify_suite(self.suite)
        self.assertEqual(inspect.call_count, 0)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(all(issue.get("error") == "artifact_path_outside_suite" for issue in report["issues"]))

    def test_previous_attempt_artifact_inside_same_suite_is_allowed(self):
        artifact = self.artifact("old", attempt="old-round")
        self.records([{"reference": artifact}], attempt="new-round")
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "verified")

    def test_correctness_pairs_are_compared_independently(self):
        self.correctness()
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["summary"]["correctness_pair_statuses"], {"passed": 4})
        self.assertTrue(report["summary"]["all_historical_latents_independently_verified"])
        self.assertTrue(all(pair["exact_match"] for pair in report["correctness_pairs"]))

    def test_claimed_exact_match_does_not_override_different_saved_output(self):
        self.correctness(altered=True)
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        failed = [pair for pair in report["correctness_pairs"] if pair["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["recorded_exact_match"])
        self.assertFalse(failed[0]["exact_match"])

    def test_early_round_missing_latents_stays_unavailable(self):
        self.correctness(latents=False, attempt="early")
        self.correctness(latents=True, attempt="later")
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "verified_with_unavailable_pairs")
        self.assertEqual(report["summary"]["latent_pair_statuses"], {"unavailable": 2, "passed": 2})
        self.assertFalse(report["summary"]["all_historical_latents_independently_verified"])

    def test_baseline_is_not_guessed_from_a_different_attempt(self):
        _, resumed = self.correctness(attempt="first")
        self.records([copy.deepcopy(resumed)], attempt="second")
        report = verifier.verify_suite(self.suite)
        second_pairs = [pair for pair in report["correctness_pairs"] if "/second/" in pair["source"]]
        self.assertEqual([pair["status"] for pair in second_pairs], ["unavailable", "unavailable"])

    def test_missing_artifact_incomplete_metadata_and_invalid_json_are_visible(self):
        artifact = self.artifact("missing")
        Path(artifact["path"]).unlink()
        path = self.records([{"output": artifact}, {"latent": {"path": "lost.npy"}}])
        with path.open("a") as handle:
            handle.write('{"partial":\n')
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["issues"], 3)

    def test_nonfinite_arrays_never_report_verified(self):
        self.records([{"output": self.artifact("nan", np.array([np.nan], dtype=np.float32))}])
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        self.assertIn("nonfinite_array", report["arrays"][0]["errors"])

    def test_empty_suite_and_cli_exit_codes_do_not_claim_evidence(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verifier.main([str(self.suite)]), 2)
        self.records([{"output": self.artifact("good")}])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verifier.main([str(self.suite)]), 0)

    def test_execution_path_and_finite_step_fields_are_not_array_references(self):
        artifact = self.artifact("actual-video")
        events = [dict(kind="step", path=path, finite=True, step_index=index, completed_steps=index + 1,
            ms=10, forced_cold_anchor=False, max_continuous=3, threshold=.08, reused_forwards=0,
            cache_contexts=[], path_scope="DBCache middle blocks")
            for index, path in enumerate(("compute", "reuse", "mixed"))]
        self.records([dict(type="service_profile", output=artifact, events=events,
                           profile=dict(path="profile", shape={"height": 256}, dtype="float16", finite=True))])
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["summary"]["artifact_references"], 1)
        self.assertEqual(report["summary"]["physical_arrays"], 1)
        self.assertEqual(report["issues"], [])

    def test_ignored_execution_event_still_recurses_to_real_nested_array(self):
        artifact = self.artifact("nested-actual")
        self.records([dict(events=[dict(kind="step", path="compute", finite=True,
                                       diagnostic={"array": artifact})])])
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["summary"]["artifact_references"], 1)
        self.assertEqual(report["arrays"][0]["references"][0]["location"], "$.events[0].diagnostic.array")

    def test_npy_path_alone_remains_an_invalid_reference(self):
        self.records([dict(output={"path": "missing.npy"}, events=[dict(path="compute", finite=True)])])
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["artifact_references"], 1)
        self.assertEqual(report["issues"][0]["error"], "incomplete_array_metadata")

    def test_array_signature_with_missing_or_invalid_path_still_fails(self):
        artifact = self.artifact("valid-shape-signature")
        missing = {key: value for key, value in artifact.items() if key != "path"}
        self.records([dict(missing=missing, null=dict(artifact, path=None),
                           wrong_extension=dict(artifact, path="compute"))])
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["artifact_references"], 3)
        self.assertEqual(report["summary"]["issues"], 3)
        self.assertEqual({issue["error"] for issue in report["issues"]},
                         {"incomplete_array_metadata", "artifact_path_must_be_nonempty_string", "missing_or_non_npy_artifact"})


if __name__ == "__main__":
    unittest.main()
