"""The driver must reject deferred finalization before loading an old workbook."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ditserve import results


REPO = Path(__file__).resolve().parents[2]


class WorkflowArchiveGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "workflow/run03"
        self.root.mkdir(parents=True)
        self.expected = dict(schema="ditserve.run.v1", metrics=dict(nanoseconds=1792555200123456789))
        (self.root / "run.json").write_text(json.dumps(self.expected))
        spec = importlib.util.spec_from_file_location("archive_guard_driver", REPO / "docker/run-next-study.py")
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)
        with patch.object(self.driver, "docker", return_value=subprocess.CompletedProcess([], 0)) as docker, \
                redirect_stdout(io.StringIO()):
            self.driver.archive(dict(root=str(self.root), source="/frozen/source", image="sha256:fixture"))
        code = docker.call_args.args[0][-1]
        tree = ast.parse(code)
        # Execute the emitted archive logic unchanged, excluding only container
        # chmod/setuid setup, which is outside this CPU contract's scope.
        begin = next(index for index, node in enumerate(tree.body)
                     if isinstance(node, ast.ImportFrom) and node.module == "ditserve.results")
        self.program = compile(ast.Module(body=tree.body[begin:], type_ignores=[]), "archive-container-payload", "exec")

    def execute(self):
        with redirect_stdout(io.StringIO()):
            exec(self.program, dict(json=json, pathlib=pathlib))

    def test_noncompleted_outcomes_fail_before_load_or_verify(self):
        for status in ("deferred", "protected", "failed", None):
            with self.subTest(status=status), \
                    patch.object(results, "finalize", return_value=dict(status=status, reason="metadata exceeds bound")), \
                    patch.object(results, "load") as load, patch.object(results, "verify") as verify:
                with self.assertRaisesRegex(RuntimeError, "archive finalization did not complete:.*metadata exceeds bound"):
                    self.execute()
                load.assert_not_called()
                verify.assert_not_called()

    def test_completed_outcome_keeps_exact_roundtrip_and_verification(self):
        calls = []
        with patch.object(results, "finalize", side_effect=lambda *a, **k: calls.append("finalize") or dict(status="completed")), \
                patch.object(results, "load", side_effect=lambda *a: calls.append("load") or dict(run=self.expected)), \
                patch.object(results, "verify", side_effect=lambda *a: calls.append("verify") or dict(status="verified")):
            self.execute()
        self.assertEqual(calls, ["finalize", "load", "verify"])

    def test_completed_but_different_old_workbook_still_fails(self):
        with patch.object(results, "finalize", return_value=dict(status="completed")), \
                patch.object(results, "load", return_value=dict(run={"old": True})), \
                patch.object(results, "verify") as verify:
            with self.assertRaisesRegex(AssertionError, "canonical workbook roundtrip mismatch"):
                self.execute()
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
