import unittest

import numpy as np

from object_preprocess import orient_normals_outward


class ObjectPreprocessTests(unittest.TestCase):
    def test_orients_normals_away_from_center(self):
        points = np.array([[1, 0, 0], [-1, 0, 0]], dtype=float)
        normals = np.array([[-1, 0, 0], [-1, 0, 0]], dtype=float)
        oriented = orient_normals_outward(points, normals, np.zeros(3))
        np.testing.assert_array_equal(oriented, [[1, 0, 0], [-1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
