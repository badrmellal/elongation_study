import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_confirmation as runner
from analyze_confirmation import holm, paired
import numpy as np


class RunnerTests(unittest.TestCase):
    def setUp(self):
        
        real_digest = runner.digest
        expected = json.loads((runner.HERE / "manifest_v2.json").read_text())["index_sha256"]
        patcher = patch.object(runner, "digest", side_effect=lambda path:
                               expected if path == runner.INDEX else real_digest(path))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_log_newline_cannot_corrupt_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "metadata.json").write_text('{}')
            (output / "observations.jsonl").write_text('{"kind":"excluded"}')
            args = argparse.Namespace(manifest=runner.HERE / "manifest_v2.json", output=output, resume=True)
            with patch.object(runner, "Models", side_effect=AssertionError("Must reject before model loading")):
                with self.assertRaisesRegex(ValueError, "final newline"):
                    runner.execute(args)

    def test_silence_reference_excludes_inserted_gaps(self):
        from inference import reference_frame_mask
        condition = {"intervals": [(0.2, 0.4), (1.2, 1.4)],
                     "metadata": {"interval_added_samples": [8000, 8000]}}
        exclusions = runner.reference_regions("silence", condition, 16000)
        self.assertEqual(exclusions, [(0.2, 0.9), (1.2, 1.9)])
        mask = reference_frame_mask(110, exclusions)
        self.assertFalse(mask[20:45].any())
        self.assertFalse(mask[70:95].any())
        self.assertTrue(mask[50])

    def test_nonempty_log_without_metadata_cannot_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "observations.jsonl").write_text('{"kind":"excluded"}\n')
            args = argparse.Namespace(manifest=runner.HERE / "manifest_v2.json", output=output, resume=True)
            with patch.object(runner, "Models", side_effect=AssertionError("Must reject before model loading")):
                with self.assertRaisesRegex(ValueError, "original metadata"):
                    runner.execute(args)

    def test_transformation_failure_is_logged_then_raised(self):
        with tempfile.TemporaryFile(mode="w+", encoding="utf8") as log:
            def broken():
                raise RuntimeError("Rubber Band failed")
            with self.assertRaisesRegex(RuntimeError, "Rubber Band"):
                runner.logged_transform(log, {"reciter": "test", "verse": "1:1"}, 4, "rubberband", broken)
            log.seek(0)
            value = json.loads(log.read())
            self.assertEqual(value["kind"], "transform_failure")
            self.assertEqual(value["k"], 4)
            self.assertEqual(value["exception_type"], "RuntimeError")

    def test_constant_paired_data_and_holm(self):
        zero = paired([0., 0., 0.], np.random.default_rng(4))
        self.assertEqual(zero["ci95"], [0., 0.])
        self.assertEqual(zero["p_two_sided"], 1)
        cells = [{"p_two_sided": p} for p in [.01, .04, .03]]
        holm(cells)
        self.assertEqual([c["p_holm"] for c in cells], [.03, .06, .06])


if __name__ == "__main__":
    unittest.main()
