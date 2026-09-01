import unittest

import numpy as np
from types import SimpleNamespace


class ExperimentVisualizationTests(unittest.TestCase):
    def test_wireframe_segments_are_world_space_line_pairs(self):
        from scripts.export_experiment_visualizations import wireframe_segments

        record = {
            "translation": [1.0, 2.0, 3.0],
            "rotation_matrix": np.eye(3).tolist(),
            "opening_mm": 30.0,
            "grasp_width_mm": 20.0,
        }
        object_data = SimpleNamespace(T_object_world=np.eye(4))
        segments = wireframe_segments(record, object_data)

        self.assertEqual(segments.ndim, 3)
        self.assertEqual(segments.shape[1:], (2, 3))
        self.assertGreaterEqual(len(segments), 3)
        self.assertTrue(np.all(np.isfinite(segments)))

    def test_parser_accepts_topk_and_output_directory(self):
        from scripts.export_experiment_visualizations import build_parser

        args = build_parser().parse_args(
            [
                "--object", "model/cat.ply",
                "--results", "results/cat",
                "--output-dir", "results/plots",
                "--topk", "20",
            ]
        )

        self.assertEqual(args.topk, 20)
        self.assertEqual(args.output_dir, "results/plots")

    def test_parser_accepts_score_threshold(self):
        from scripts.export_experiment_visualizations import build_parser

        args = build_parser().parse_args(
            [
                "--object", "model/cat.ply",
                "--results", "results/cat",
                "--output-dir", "results/plots",
                "--score-threshold", "0.8",
            ]
        )

        self.assertAlmostEqual(args.score_threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
