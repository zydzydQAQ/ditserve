"""External artifact registration uses tiny synthetic files only, never CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ditserve.experiments import previs_storage as storage
from ditserve.experiments import previs_verify as verifier


class StorageIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="previs-storage-fixture-")
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.suite, self.external = self.parent / "suite", self.parent / "external"
        self.suite.mkdir()
        self.external.mkdir()
        self.logical = self.suite / "stage/arrays/example.npy"
        self.logical.parent.mkdir(parents=True)
        self.records = self.suite / "stage/records.jsonl"
        self.records.write_text("")
        self.target = self.external / "retained/example.npy"
        self.target.parent.mkdir()
        self.array = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.save(self.target, self.array, allow_pickle=False)
        self.logical.symlink_to(self.target)
        self.entry = dict(logical_path="stage/arrays/example.npy", target_path="retained/example.npy",
                          size_bytes=self.target.stat().st_size,
                          file_sha256=hashlib.sha256(self.target.read_bytes()).hexdigest())
        self.index = dict(schema_version=storage.SCHEMA, suite_root=str(self.suite),
                          storage_root=str(self.external), entries=[self.entry])

    def write_index(self):
        (self.suite / storage.INDEX_NAME).write_text(json.dumps(self.index))

    def resolve(self, path=None):
        return verifier._resolve(str(path or self.logical), self.suite, self.records)

    def test_absent_index_preserves_external_rejection_and_internal_links(self):
        with self.assertRaisesRegex(ValueError, "artifact_path_outside_suite"):
            self.resolve()
        self.logical.unlink()
        np.save(self.logical, self.array)
        alias = self.logical.with_name("alias.npy")
        alias.symlink_to(self.logical)
        self.assertEqual(self.resolve(alias), self.logical)
        self.assertEqual(storage.index_metadata(self.suite), {"enabled": False})

    def test_registered_file_passes_both_file_and_array_payload_hashes_without_rewriting(self):
        self.write_index()
        descriptor = dict(path=str(self.logical), shape=[3, 4], dtype="float32", finite=True,
                          sha256=hashlib.sha256(self.array.tobytes()).hexdigest())
        self.assertNotEqual(self.entry["file_sha256"], descriptor["sha256"])
        self.records.write_text(json.dumps({"output": descriptor}) + "\n")
        before = {p: p.read_bytes() for p in (self.records, self.target, self.suite / storage.INDEX_NAME)}
        result = verifier.verify_suite(self.suite)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["summary"]["physical_arrays"], 1)
        self.assertEqual(result["artifact_storage"]["registered_files"], 1)
        self.assertEqual(self.resolve(), self.target)
        resolution = storage.resolve_candidates([self.logical], self.suite)
        self.assertEqual(resolution.logical_path, self.logical)
        self.assertEqual(resolution.storage["file_sha256"], self.entry["file_sha256"])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_direct_external_path_and_unregistered_alias_remain_forbidden(self):
        self.write_index()
        with self.assertRaisesRegex(ValueError, "artifact_path_outside_suite"):
            self.resolve(self.target)
        alias = self.logical.with_name("alias.npy")
        alias.symlink_to(self.target)
        with self.assertRaisesRegex(ValueError, "unregistered"):
            self.resolve(alias)

    def test_records_calibration_and_directory_entries_cannot_be_registered(self):
        for name in ("stage/records.jsonl", "calibration.json", "development-records.json", "stage/arrays"):
            with self.subTest(name=name):
                self.entry["logical_path"] = name
                self.write_index()
                with self.assertRaisesRegex(ValueError, "only_stage_raw_npy"):
                    self.resolve()

    def test_parent_traversal_absolute_target_and_noncanonical_paths_are_rejected(self):
        base = copy.deepcopy(self.index)
        for field, path in (("logical_path", "stage/arrays/../arrays/example.npy"),
                            ("logical_path", "../stage/arrays/example.npy"),
                            ("target_path", "../outside.npy"), ("target_path", str(self.target)),
                            ("target_path", "retained//example.npy")):
            with self.subTest(field=field, path=path):
                self.index = copy.deepcopy(base)
                self.index["entries"][0][field] = path
                self.write_index()
                with self.assertRaisesRegex(ValueError, "noncanonical_path"):
                    self.resolve()

    def test_symlink_target_must_exactly_match_registration(self):
        self.write_index()
        wrong = self.target.with_name("same-bytes.npy")
        wrong.write_bytes(self.target.read_bytes())
        self.logical.unlink()
        self.logical.symlink_to(wrong)
        with self.assertRaisesRegex(ValueError, "target_mismatch"):
            self.resolve()

    def test_target_parent_symlink_and_individual_target_symlink_are_rejected(self):
        self.write_index()
        real = self.external / "physical"
        self.target.parent.rename(real)
        self.target.parent.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "target_symlink_or_escape"):
            self.resolve()
        self.target.parent.unlink()
        real.rename(self.target.parent)
        actual = self.target.with_name("actual.npy")
        self.target.rename(actual)
        self.target.symlink_to(actual)
        with self.assertRaisesRegex(ValueError, "target_symlink_or_escape"):
            self.resolve()

    def test_array_directory_symlink_is_not_authorized(self):
        self.write_index()
        actual = self.suite / "stage/old-arrays"
        self.logical.parent.rename(actual)
        self.logical.parent.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "individual_npy_symlink"):
            self.resolve()

    def test_stage_record_stream_must_remain_local(self):
        self.write_index()
        external_records = self.external / "records.jsonl"
        self.records.rename(external_records)
        self.records.symlink_to(external_records)
        with self.assertRaisesRegex(ValueError, "records_must_remain_inside"):
            self.resolve()
        with self.assertRaisesRegex(ValueError, "unregistered"):
            self.resolve(self.records)

    def test_file_size_digest_and_same_size_mutation_after_first_read_are_rejected(self):
        self.write_index()
        self.assertEqual(self.resolve(), self.target)
        content = self.target.read_bytes()
        self.target.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "file_sha256_mismatch"):
            self.resolve()
        self.target.write_bytes(content + b"extra")
        with self.assertRaisesRegex(ValueError, "size_or_type_mismatch"):
            self.resolve()

    def test_wrong_suite_duplicate_entry_and_unknown_schema_fields_fail_closed(self):
        base = copy.deepcopy(self.index)
        for change in (lambda i: i.update(suite_root=str(self.parent)),
                       lambda i: i["entries"].append(copy.deepcopy(i["entries"][0])),
                       lambda i: i.update(allow_all_external=True)):
            self.index = copy.deepcopy(base)
            change(self.index)
            self.write_index()
            with self.assertRaises(ValueError):
                self.resolve()

    def test_index_symlink_and_duplicate_json_keys_are_rejected(self):
        external_index = self.external / "index.json"
        external_index.write_text(json.dumps(self.index))
        path = self.suite / storage.INDEX_NAME
        path.symlink_to(external_index)
        with self.assertRaisesRegex(ValueError, "regular_bounded_file"):
            self.resolve()
        path.unlink()
        path.write_text('{"schema_version":"a","schema_version":"b"}')
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            self.resolve()

    def test_registered_link_cannot_be_silently_replaced_with_internal_file(self):
        self.write_index()
        self.logical.unlink()
        self.logical.write_bytes(self.target.read_bytes())
        with self.assertRaisesRegex(ValueError, "individual_npy_symlink"):
            self.resolve()

    def test_invalid_index_is_a_failed_verification_not_success_without_arrays(self):
        self.index["entries"][0]["size_bytes"] = True
        self.write_index()
        report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["artifact_storage"]["invalid"])

    def test_index_change_during_verification_is_not_reported_verified(self):
        self.write_index()
        self.records.write_text(json.dumps({"output": dict(path=str(self.logical), shape=[3, 4], dtype="float32",
                               finite=True, sha256=hashlib.sha256(self.array.tobytes()).hexdigest())}) + "\n")
        real_inspect = verifier._inspect
        def mutate(path, np_module):
            result = real_inspect(path, np_module)
            (self.suite / storage.INDEX_NAME).unlink()
            return result
        with mock.patch.object(verifier, "_inspect", side_effect=mutate):
            report = verifier.verify_suite(self.suite)
        self.assertEqual(report["status"], "failed")
        self.assertIn("artifact_storage_index_changed_during_verification", [x["reason"] for x in report["issues"]])


if __name__ == "__main__":
    unittest.main()
