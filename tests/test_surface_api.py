import unittest

from grasp_detect import grasp_detect_from_surface


class SurfaceApiTests(unittest.TestCase):
    def test_validates_surface_shapes_before_generation(self):
        with self.assertRaises(ValueError):
            grasp_detect_from_surface(None, [[0, 0, 0]], [], [0, 0, 1])
        with self.assertRaises(ValueError):
            grasp_detect_from_surface(None, [[0, 0, 0]], [[0, 0, 1]], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
