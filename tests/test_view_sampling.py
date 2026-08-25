import unittest

import numpy as np

from view_sampling import generate_viewpoints


class GenerateViewpointsTests(unittest.TestCase):
    def test_returns_requested_number_of_unit_vectors(self):
        viewpoints = generate_viewpoints(60)

        self.assertEqual(viewpoints.shape, (60, 3))
        np.testing.assert_allclose(np.linalg.norm(viewpoints, axis=1), 1.0)

    def test_rejects_non_positive_or_non_integer_counts(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    generate_viewpoints(value)


if __name__ == "__main__":
    unittest.main()
