import csv
import tempfile
import unittest
from pathlib import Path

from opening_profile import OpeningProfiler


class OpeningProfilerTests(unittest.TestCase):
    def test_records_opening_funnel_and_writes_csv(self):
        profiler = OpeningProfiler()
        profiler.add(
            candidate_id=7,
            depth_id=2,
            depth_value=30.0,
            angle_deg=45.0,
            opening=60.0,
            collision_free=True,
            opening_selected=True,
            final_valid=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "opening_profile.csv"
            profiler.save(output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["depth_id"], "2")
        self.assertEqual(rows[0]["angle_deg"], "45.0")
        self.assertEqual(rows[0]["opening"], "60.0")
        self.assertEqual(rows[0]["collision_free"], "1")
        self.assertEqual(rows[0]["opening_selected"], "1")
        self.assertEqual(rows[0]["final_valid"], "1")

    def test_empty_profiler_writes_header(self):
        profiler = OpeningProfiler()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "opening_profile.csv"
            profiler.save(output)
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0], profiler.fieldnames_line)


if __name__ == "__main__":
    unittest.main()
