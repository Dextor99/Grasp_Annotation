import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from grasp_database import save_grasp_dataset
import main


class GraspDatabaseTests(unittest.TestCase):
    def test_nonempty_dataset_has_json_npz_and_metadata_schemas(self):
        pose = np.eye(4)
        pose[:3, 3] = [1.0, 2.0, 3.0]
        grasp = {
            "T_gripper_object": pose,
            "opening": np.float64(50.0),
            "score_total": np.float32(0.7),
            "score_force_closure": np.float64(0.4),
            "score_label": 2,
            "score_bad": np.nan,
            "view_id": np.int64(1),
            "view_direction": np.array([0.0, 0.0, 4.0]),
            "source": "synthetic",
            "nested": {"sample": np.int64(3)},
            "not_serializable": object(),
        }
        with tempfile.TemporaryDirectory() as directory:
            result = save_grasp_dataset([grasp], directory, {"object": "demo", "units": "mm"})
            self.assertTrue(result.json_path.exists())
            self.assertTrue(result.npz_path.exists())
            self.assertTrue(result.meta_path.exists())

            records = json.loads(result.json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(result.saved_count, 1)
            self.assertEqual(record["id"], 0)
            self.assertEqual(record["translation"], [1.0, 2.0, 3.0])
            self.assertEqual(record["rotation"], np.eye(3).reshape(-1).tolist())
            self.assertEqual(record["quaternion_xyzw"], [0.0, 0.0, 0.0, 1.0])
            self.assertEqual(record["opening"], 50.0)
            self.assertAlmostEqual(record["score_total"], 0.7)
            self.assertEqual(record["score_force_closure"], 0.4)
            self.assertEqual(record["score_label"], 2)
            self.assertNotIn("score_bad", record)
            self.assertEqual(record["view_id"], 1)
            self.assertEqual(record["view_direction"], [0.0, 0.0, 1.0])
            self.assertEqual(record["source"], "synthetic")
            self.assertEqual(record["nested"], {"sample": 3})
            self.assertNotIn("not_serializable", record)

            with np.load(result.npz_path) as dataset:
                self.assertEqual(dataset["poses"].shape, (1, 4, 4))
                self.assertEqual(dataset["translations"].shape, (1, 3))
                self.assertEqual(dataset["rotations"].shape, (1, 3, 3))
                self.assertEqual(dataset["quaternions"].shape, (1, 4))
                self.assertEqual(dataset["openings"].shape, (1,))
                self.assertEqual(dataset["scores"].shape, (1,))
                self.assertEqual(dataset["view_ids"].shape, (1,))
                self.assertTrue(np.issubdtype(dataset["poses"].dtype, np.floating))
                np.testing.assert_allclose(dataset["translations"], [[1.0, 2.0, 3.0]])

            metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["object"], "demo")
            self.assertEqual(metadata["units"], "mm")
            self.assertEqual(metadata["visibility_strategy"], "normal_based_front_facing")

    def test_empty_dataset_uses_fixed_npz_shapes_and_default_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            result = save_grasp_dataset([], Path(directory) / "new-output", {})
            self.assertEqual(result.saved_count, 0)
            self.assertEqual(json.loads(result.json_path.read_text(encoding="utf-8")), [])
            with np.load(result.npz_path) as dataset:
                self.assertEqual(dataset["poses"].shape, (0, 4, 4))
                self.assertEqual(dataset["translations"].shape, (0, 3))
                self.assertEqual(dataset["rotations"].shape, (0, 3, 3))
                self.assertEqual(dataset["quaternions"].shape, (0, 4))
                self.assertEqual(dataset["openings"].shape, (0,))
                self.assertEqual(dataset["scores"].shape, (0,))
                self.assertEqual(dataset["view_ids"].shape, (0,))
            metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["units"], "mm")
            self.assertEqual(metadata["visibility_strategy"], "normal_based_front_facing")

    def test_invalid_or_nonfinite_poses_are_dropped(self):
        invalid_shape = {"T_gripper_object": np.eye(3)}
        invalid_nan = {"T_gripper_object": np.full((4, 4), np.nan)}
        invalid_rotation = {"T_gripper_object": np.diag([2.0, 1.0, 1.0, 1.0])}
        invalid_homogeneous = {"T_gripper_object": np.array([
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0],
        ])}
        valid = {"T_gripper_object": np.eye(4), "score_total": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            result = save_grasp_dataset([invalid_shape, invalid_nan, invalid_rotation, invalid_homogeneous, valid], directory, {})
            self.assertEqual(result.saved_count, 1)
            self.assertEqual(len(json.loads(result.json_path.read_text(encoding="utf-8"))), 1)
            with np.load(result.npz_path) as dataset:
                self.assertEqual(dataset["poses"].shape, (1, 4, 4))

    def test_rejects_rotation_perturbations_that_cannot_roundtrip_rigidly(self):
        perturbed = np.eye(4)
        perturbed[0, 0] = 1.000005
        with tempfile.TemporaryDirectory() as directory:
            result = save_grasp_dataset([
                {"T_gripper_object": perturbed},
                {"T_gripper_object": np.eye(4)},
            ], directory, {})
            self.assertEqual(result.saved_count, 1)
            record = json.loads(result.json_path.read_text(encoding="utf-8"))[0]
            with np.load(result.npz_path) as dataset:
                rotation = dataset["rotations"][0]
                np.testing.assert_allclose(dataset["poses"][0, :3, :3], rotation)
            np.testing.assert_allclose(rotation.reshape(-1), record["rotation"])
            np.testing.assert_allclose(
                rotation,
                __import__("scipy").spatial.transform.Rotation.from_quat(record["quaternion_xyzw"]).as_matrix(),
            )

    def test_invalid_view_ids_use_negative_one_in_json_and_npz(self):
        grasps = [
            {"T_gripper_object": np.eye(4), "view_id": 2.5},
            {"T_gripper_object": np.eye(4), "view_id": "2"},
            {"T_gripper_object": np.eye(4), "view_id": True},
            {"T_gripper_object": np.eye(4), "view_id": np.int64(4)},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = save_grasp_dataset(grasps, directory, {})
            records = json.loads(result.json_path.read_text(encoding="utf-8"))
            self.assertEqual([record["view_id"] for record in records], [-1, -1, -1, 4])
            with np.load(result.npz_path) as dataset:
                np.testing.assert_array_equal(dataset["view_ids"], [-1, -1, -1, 4])


class MainTests(unittest.TestCase):
    def test_parser_only_accepts_supported_view_counts(self):
        parser = main.build_parser()
        for views in (20, 40, 60, 100):
            with self.subTest(views=views):
                self.assertEqual(parser.parse_args(["--object", "a.ply", "--output", "out", "--views", str(views)]).views, views)
        for views in (0, 37):
            with self.subTest(views=views):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--object", "a.ply", "--output", "out", "--views", str(views)])

    def test_pipeline_reports_saved_count_when_serializer_drops_invalid_pose(self):
        result = type("Result", (), {
            "grasps": [
                {"T_gripper_object": np.eye(4), "score_total": 0.8},
                {"T_gripper_object": np.eye(3), "score_total": 0.1},
            ],
            "skipped_views": [{"view_id": 1, "reason": "no_candidates"}],
            "view_candidate_counts": {0: 2, 1: 0},
        })()
        with tempfile.TemporaryDirectory() as directory, patch.object(main, "generate_multi_view_grasps", return_value=result) as generate, patch("builtins.print") as printed:
            exit_code = main.run(["--object", "object.ply", "--views", "20", "--output", directory])
        self.assertEqual(exit_code, 0)
        generate.assert_called_once_with("object.ply", 20, 5.0, 10.0)
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("processed views: 2", output)
        self.assertIn("skipped views: 1", output)
        self.assertIn("raw grasps: 2", output)
        self.assertIn("deduplicated grasps: 2", output)
        self.assertIn("saved grasps: 1", output)
        self.assertIn("per-view candidate counts: {0: 2, 1: 0}", output)
        self.assertIn("grasps.json", output)


if __name__ == "__main__":
    unittest.main()
