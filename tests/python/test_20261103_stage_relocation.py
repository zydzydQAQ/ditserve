"""Downloaded receipts retain scientific identity across explicit path maps."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
import relocate_stage_evidence
import stage_gates
from test_20261102_stage_gates import fixture, write


class StageRelocationTests(unittest.TestCase):
    remote = "/shared/output/DITSERVE"

    def prepare(self, root, *, contract=False, relative=False):
        mirror = root / "mirror"
        mirror.mkdir()
        manifest = fixture(mirror)
        if contract:
            manifest["runs"] = []
        result = stage_gates.evaluate(manifest, mirror)
        for reference in [manifest["protocol"], *manifest["contract"].values(), *manifest["runs"]]:
            reference["path"] = ("../" if relative else self.remote + "/") + reference["path"]
        (mirror / "evidence").mkdir()
        source = write(mirror, "evidence/original-evidence.json", manifest)
        source["path"] = ("../evidence/original-evidence.json" if relative
                          else self.remote + "/evidence/original-evidence.json")
        result["source_manifest"] = source
        (mirror / "acceptance").mkdir()
        write(mirror, "acceptance/original-receipt.json", result)
        receipt = mirror / "acceptance/original-receipt.json"
        location = self.remote + "/acceptance/original-receipt.json"
        return receipt, location, [self.remote + "=" + str(mirror)], root / "relocated"

    def test_contract_and_p0_relocate_without_changing_sources(self):
        for contract in (True, False):
            for relative in (True, False):
                with self.subTest(contract=contract, relative=relative), TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    args = self.prepare(root, contract=contract, relative=relative)
                    originals = {p: p.read_bytes() for p in (root / "mirror").rglob("*") if p.is_file()}
                    path, result = relocate_stage_evidence.relocate(*args)
                    self.assertEqual(result["relocated_from"]["sha256"], hashlib.sha256(originals[args[0]]).hexdigest())
                    verified = stage_gates.verify_receipt(path, required_stage="CONTRACT" if contract else "P0")
                    self.assertEqual(verified["stages"]["P0"]["status"], "blocked" if contract else "passed")
                    self.assertTrue(all(p.read_bytes() == raw for p, raw in originals.items()))
                    with self.assertRaises(FileExistsError):
                        relocate_stage_evidence.relocate(*args)

    def test_corrupt_original_manifest_and_sources_never_write_outputs(self):
        for name in ("evidence/original-evidence.json", "protocol.json", "contract.log", "job.json",
                     "smoke.json", "preemption.json", "no_preemption.json"):
            with self.subTest(name=name), TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = self.prepare(root)
                (root / "mirror" / name).write_text("{}")
                with self.assertRaisesRegex(ValueError, "original source SHA256"):
                    relocate_stage_evidence.relocate(*args)
                self.assertFalse(args[3].exists())

    def test_changed_stage_claim_is_rejected_after_recomputation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.prepare(root)
            original = json.loads(args[0].read_bytes())
            original["stages"]["P1"]["status"] = "passed"
            args[0].write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "gate results differ"):
                relocate_stage_evidence.relocate(*args)
            self.assertFalse(args[3].exists())

    def test_missing_mapping_root_and_source_fail_closed(self):
        for failure in ("unmapped", "missing-root", "missing-source"):
            with self.subTest(failure=failure), TemporaryDirectory() as temporary:
                root = Path(temporary)
                receipt, location, mappings, output = self.prepare(root)
                if failure == "unmapped":
                    mappings = ["/shared/data/DITSERVE=" + str(root / "mirror")]
                elif failure == "missing-root":
                    mappings = [self.remote + "=" + str(root / "absent")]
                else:
                    (root / "mirror" / "contract.log").unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    relocate_stage_evidence.relocate(receipt, location, mappings, output)
                self.assertFalse(output.exists())

    def test_specific_mapping_is_explicit_and_cannot_escape_local_root(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "broad").mkdir()
            (root / "narrow").mkdir()
            (root / "narrow" / "source").write_text("data")
            mapping = relocate_stage_evidence.PrefixMap([
                "/shared=" + str(root / "broad"), "/shared/output=" + str(root / "narrow")])
            from pathlib import PurePosixPath
            self.assertEqual(mapping.local(PurePosixPath("/shared/output/source")), root / "narrow" / "source")
            (root / "outside").write_text("data")
            (root / "narrow" / "escape").symlink_to(root / "outside")
            with self.assertRaisesRegex(ValueError, "escapes"):
                mapping.local(PurePosixPath("/shared/output/escape"))


if __name__ == "__main__":
    unittest.main()
