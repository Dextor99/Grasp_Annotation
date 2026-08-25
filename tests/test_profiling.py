import io
import time
import unittest
import tempfile
from pathlib import Path
from contextlib import redirect_stdout

from profiling import ProfileRecorder


class ProfileRecorderTests(unittest.TestCase):
    def test_disabled_recorder_is_silent(self):
        recorder = ProfileRecorder(enabled=False)
        output = io.StringIO()
        with redirect_stdout(output):
            with recorder.stage("disabled"):
                pass
            recorder.print_report()
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(recorder.records, [])

    def test_enabled_recorder_collects_stage_and_reports_percentage(self):
        recorder = ProfileRecorder(enabled=True)
        recorder.count("candidates.input", 2112)
        recorder.group_count("depth", "0", "candidate")
        recorder.group_count("depth", "0", "collision_free")
        with recorder:
            with recorder.stage("load"):
                time.sleep(0.001)
        self.assertEqual(len(recorder.records), 1)
        self.assertEqual(recorder.records[0].name, "load")
        self.assertGreaterEqual(recorder.records[0].seconds, 0.0)
        output = io.StringIO()
        with redirect_stdout(output):
            recorder.print_report()
        self.assertIn("load", output.getvalue())
        self.assertIn("100.0%", output.getvalue())
        self.assertIn("candidates.input", output.getvalue())
        self.assertIn("2112", output.getvalue())
        self.assertIn("depth", output.getvalue())
        self.assertIn("collision_free", output.getvalue())

    def test_matrix_counts_write_variant_depth_csv(self):
        recorder = ProfileRecorder(enabled=True)
        recorder.matrix_count("variant_depth", 2, 30, "candidate")
        recorder.matrix_count("variant_depth", 2, 30, "collision_free")
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "matrix.csv"
            recorder.write_matrix_csv(output_path)
            rows = output_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows[0], "variant_id,depth,candidate,collision_free,final")
        self.assertEqual(rows[1], "2,30,1,1,0")


if __name__ == "__main__":
    unittest.main()
