import unittest

import numpy as np
import open3d as o3d

from grasp_detect import CollisionIndex, aabb_overlaps, check_collision


class CollisionFilterTests(unittest.TestCase):
    def test_aabb_overlap_respects_margin(self):
        self.assertFalse(aabb_overlaps([0, 0, 0], [1, 1, 1], [1.1, 0, 0], [2, 1, 1]))
        self.assertTrue(aabb_overlaps([0, 0, 0], [1, 1, 1], [1.1, 0, 0], [2, 1, 1], margin=0.2))

    def test_collision_index_queries_only_points_in_local_aabb(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [5.0, 5.0, 5.0],
        ])
        index = CollisionIndex(points)
        result = index.query_aabb(np.array([-0.1, -0.1, -0.1]), np.array([0.6, 0.6, 0.6]))
        np.testing.assert_array_equal(result, [0, 1])

    def test_collision_index_rejects_empty_points(self):
        with self.assertRaises(ValueError):
            CollisionIndex(np.empty((0, 3)))

    def test_collision_check_keeps_true_and_false_results(self):
        box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        box.compute_vertex_normals()
        meshes = [box, o3d.geometry.TriangleMesh(), o3d.geometry.TriangleMesh()]
        near_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]]))
        far_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[10.0, 10.0, 10.0]]))
        self.assertTrue(check_collision(meshes, near_cloud, threshold=0.6))
        self.assertFalse(check_collision(meshes, far_cloud, threshold=0.6))


if __name__ == "__main__":
    unittest.main()
