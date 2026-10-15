from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ditserve.optimizations import next_followups
from ditserve.research.common import atomic_json, file_digest, read_json


class FollowupTests(unittest.TestCase):
    def context(self, root, *, video=False, encoder=False, auxiliary=False):
        return SimpleNamespace(output=root, check_budget=lambda:None,
            conditional_admission=dict(devices=["GPU-a", "GPU-v"] if auxiliary else ["GPU-a"],
                auxiliary_encoder=auxiliary, trigger_evidence=dict(admitted_sources=[{}],
                    vae_trigger=video, encoder_trigger=encoder)))

    def test_absent_triggers_load_no_models_or_candidate_work(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ditserve.optimizations.next_conditional.run_video_vae") as vae, \
             patch("ditserve.optimizations.next_encoder.run_encoder") as encoder:
            context = self.context(Path(tmp))
            result = next_followups.run_followups(context)
        vae.assert_not_called(); encoder.assert_not_called()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["decision"], "简单策略已足够")
        self.assertFalse(result["online_qualified"])

    def test_busy_v100_does_not_skip_independent_video_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(context):
                value = dict(status="passed", online_qualified=True, decision="值得继续")
                atomic_json(root/"conditional-video-vae.json", value)
                return value
            with patch("ditserve.optimizations.next_conditional.run_video_vae", side_effect=run) as vae, \
                 patch("ditserve.optimizations.next_encoder.run_encoder") as encoder:
                result = next_followups.run_followups(self.context(root, video=True, encoder=True))
            vae.assert_called_once(); encoder.assert_not_called()
            self.assertEqual(result["text_encoder"]["decision"], "未验证")
            self.assertEqual(result["independently_qualified"], ["video_vae"])
            self.assertFalse(result["online_qualified"])
            self.assertEqual(read_json(root/"conditional.json"), result)

    def test_unexpected_failure_preserves_active_component_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch("ditserve.optimizations.next_conditional.run_video_vae", side_effect=RuntimeError("state unsafe")), \
             patch("ditserve.optimizations.next_encoder.run_encoder") as encoder:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "state unsafe"):
                next_followups.run_followups(self.context(root, video=True, encoder=True, auxiliary=True))
            saved = read_json(root/"conditional.json")
        encoder.assert_not_called()
        self.assertEqual(saved["status"], "incomplete")
        self.assertEqual(saved["active_component"], "video_vae")
        self.assertFalse(saved["online_qualified"])

    def test_admission_rejects_unbound_or_changed_supporting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/"admission.json"; source=root/"source.json"
            atomic_json(source, {"fixed":True})
            value=dict(trigger_evidence=dict(vae_trigger=True, immutable_bindings=[]))
            atomic_json(path,value)
            binding=dict(path=str(path),sha256=file_digest(path))
            with self.assertRaisesRegex(ValueError,"immutable"):
                next_followups.read_admission(binding)
            value["trigger_evidence"]["immutable_bindings"]=[dict(path=str(source),sha256=file_digest(source))]
            atomic_json(path,value);binding["sha256"]=file_digest(path)
            self.assertEqual(next_followups.read_admission(binding), value)
            atomic_json(source,{"fixed":False})
            with self.assertRaisesRegex(ValueError,"source changed"):
                next_followups.read_admission(binding)

    def test_component_failure_is_unverified_not_negative_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def run(context):
                value=dict(status="failed", reason="compiler failed", online_qualified=False)
                atomic_json(root/"conditional-video-vae.json",value)
                return value
            with patch("ditserve.optimizations.next_conditional.run_video_vae",side_effect=run):
                result=next_followups.run_followups(self.context(root,video=True))
            self.assertEqual(result["decision"], "未验证")
            self.assertEqual(result["status"], "completed")

    def test_explicit_unclean_component_stops_following_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def run(context):
                value=dict(status="failed", decoder_cleanup_passed=False, online_qualified=False)
                atomic_json(root/"conditional-video-vae.json",value)
                return value
            with patch("ditserve.optimizations.next_conditional.run_video_vae",side_effect=run), \
                 patch("ditserve.optimizations.next_encoder.run_encoder") as encoder:
                with self.assertRaisesRegex(RuntimeError,"cleanup unconfirmed"):
                    next_followups.run_followups(self.context(root,video=True,encoder=True,auxiliary=True))
                encoder.assert_not_called()
            self.assertFalse(read_json(root/"conditional.json")["video_vae"]["decoder_cleanup_passed"])


if __name__ == "__main__":
    unittest.main()
