import json
import tempfile
import unittest
from pathlib import Path


class PlyGeometryAggregationTests(unittest.TestCase):
    def test_aggregate_reads_gn_and_ours_geometry_summaries(self):
        from scripts.common_eval.aggregate_ply_geometry import aggregate_ply_geometry

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gn = root / "gn"; ours = root / "ours"
            gn.mkdir(); ours.mkdir()
            (gn / "summary.json").write_text(json.dumps({
                "n_candidates": 100, "n_grasp_points": 2,
                "n_geometry_valid": 25, "geometry_valid_rate": 0.25,
                "geometry_runtime_s": 3.5, "input_source": "surface_ply",
            }), encoding="utf-8")
            (ours / "summary.json").write_text(json.dumps({
                "n_raw_candidates": 40, "n_unique_outputs": 30,
                "n_common_geometry_valid": 12,
                "common_geometry_valid_rate_output": 0.4,
                "common_geometry_yield_raw": 0.3,
                "common_geometry_eval_time_s": 1.2,
                "native_generation_runtime_s": 9.5,
            }), encoding="utf-8")
            rows, summary = aggregate_ply_geometry({"cube": {"gn": gn, "ours": ours}})

        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["GN-geometry"]["n_objects"], 1)
        self.assertEqual(summary["Ours-common-geometry"]["n_objects"], 1)
        gn_row = next(row for row in rows if row["method"] == "GN-geometry")
        self.assertEqual(gn_row["n_geometry_valid"], 25)
        self.assertAlmostEqual(gn_row["geometry_valid_rate"], 0.25)
        ours_row = next(row for row in rows if row["method"] == "Ours-common-geometry")
        self.assertAlmostEqual(ours_row["native_generation_runtime_s"], 9.5)

    def test_aggregate_rejects_missing_summary(self):
        from scripts.common_eval.aggregate_ply_geometry import aggregate_ply_geometry

        with self.assertRaises(FileNotFoundError):
            aggregate_ply_geometry({"cube": {"gn": Path("missing"), "ours": Path("missing2")}})


if __name__ == "__main__":
    unittest.main()
