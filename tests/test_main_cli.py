import unittest
from types import SimpleNamespace
from unittest.mock import patch


class MainCliTests(unittest.TestCase):
    def test_cli_builds_config_runs_pipeline_and_exports(self):
        import main as grasp_main

        fake_result = SimpleNamespace(meta={"raw_grasp_count": 2, "unique_grasp_count": 1})
        with patch("main.run_grasp_annotation", return_value=fake_result) as run, patch(
            "main.export_grasp_annotations",
            return_value={"grasps_json": "out/grasps.json"},
        ) as export:
            exit_code = grasp_main.main(
                [
                    "--object", "model/shuilongtou.ply",
                    "--views", "5",
                    "--anchors", "3",
                    "--mode", "cone",
                    "--output", "results/shuilongtou",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = run.call_args.kwargs["config"]
        self.assertEqual(config.num_views, 5)
        self.assertEqual(config.anchors_per_view, 3)
        self.assertEqual(config.mode, "cone")
        export.assert_called_once_with(fake_result, "results/shuilongtou")


if __name__ == "__main__":
    unittest.main()
