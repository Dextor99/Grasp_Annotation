import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from grasp_detect import filter_collision_free_grippers_first_opening


class OpeningEarlyStopTests(unittest.TestCase):
    def test_checks_openings_in_order_and_stops_per_depth_angle_group(self):
        point_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
        )
        candidates = [
            {"id": 3, "depth": 10, "angle_deg": 0, "opening": 45, "meshes": []},
            {"id": 1, "depth": 10, "angle_deg": 0, "opening": 15, "meshes": []},
            {"id": 2, "depth": 10, "angle_deg": 0, "opening": 30, "meshes": []},
            {"id": 5, "depth": 10, "angle_deg": 15, "opening": 15, "meshes": []},
            {"id": 6, "depth": 10, "angle_deg": 15, "opening": 30, "meshes": []},
        ]
        # Group (10, 0): 15 collides, 30 is first free; group (10, 15): 15 is free.
        with patch("grasp_detect.check_collision", side_effect=[True, False, False]) as check:
            result = filter_collision_free_grippers_first_opening(candidates, point_cloud)

        self.assertEqual([candidate["id"] for candidate in result], [2, 5])
        self.assertEqual(check.call_count, 3)


if __name__ == "__main__":
    unittest.main()
