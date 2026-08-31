import unittest

from cloud_process import normal_search_radius


class CloudProcessTests(unittest.TestCase):
    def test_normal_radius_scales_with_voxel_size(self):
        self.assertAlmostEqual(normal_search_radius(2.06166), 5.15415, places=5)
        self.assertAlmostEqual(normal_search_radius(0.91, factor=3.0), 2.73, places=6)

    def test_normal_radius_rejects_non_positive_inputs(self):
        for voxel_size, factor in ((0, 2.5), (-1, 2.5), (1, 0), (1, -1)):
            with self.assertRaises(ValueError):
                normal_search_radius(voxel_size, factor=factor)


if __name__ == "__main__":
    unittest.main()
