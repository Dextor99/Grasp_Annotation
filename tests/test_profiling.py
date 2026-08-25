import io
import time
import unittest
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


if __name__ == "__main__":
    unittest.main()
