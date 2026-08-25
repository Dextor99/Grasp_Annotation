import csv
import tempfile
import unittest
from pathlib import Path

from depth_profile import DepthProfiler


class DepthProfilerTests(unittest.TestCase):
    def test_records_quality_fields_and_writes_csv(self):
        profiler = DepthProfiler()
        profiler.add(
            variant_id=3,
            depth_id=5,
            depth_value=42.5,
            collision_free=True,
            opening_valid=True,
            intersection_valid=True,
            contact_points=7,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "depth_profile.csv"
            profiler.save(output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant_id"], "3")
        self.assertEqual(rows[0]["depth_id"], "5")
        self.assertEqual(rows[0]["collision_free"], "1")
        self.assertEqual(rows[0]["final_valid"], "1")
        self.assertEqual(rows[0]["contact_points"], "7")

    def test_empty_profiler_still_writes_header(self):
        profiler = DepthProfiler()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "depth_profile.csv"
            profiler.save(output)
            self.assertTrue(output.exists())
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0], profiler.fieldnames_line)


if __name__ == "__main__":
    unittest.main()
