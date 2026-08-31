import unittest

import numpy as np

from approach_sampling import sample_normal_guided_approaches


class ApproachSamplingTests(unittest.TestCase):
    def test_returns_nominal_and_four_cone_directions(self):
        samples = sample_normal_guided_approaches([0, 0, 1], cone_angle_deg=15, num_azimuth=4)
        self.assertEqual(len(samples), 5)
        np.testing.assert_allclose(samples[0].direction, [0, 0, -1], atol=1e-12)
        self.assertEqual(samples[0].offset_deg, 0.0)
        for sample in samples[1:]:
            self.assertAlmostEqual(float(np.linalg.norm(sample.direction)), 1.0)
            angle = np.degrees(np.arccos(np.clip(sample.direction @ samples[0].direction, -1.0, 1.0)))
            self.assertAlmostEqual(float(angle), 15.0, places=7)
            self.assertEqual(sample.offset_deg, 15.0)

    def test_azimuth_directions_are_distinct(self):
        samples = sample_normal_guided_approaches([1, 0, 0], cone_angle_deg=15, num_azimuth=4)
        rounded = {tuple(np.round(sample.direction, 8)) for sample in samples[1:]}
        self.assertEqual(len(rounded), 4)

    def test_rejects_invalid_normal(self):
        with self.assertRaises(ValueError):
            sample_normal_guided_approaches([0, 0, 0])

    def test_rejects_zero_cone_angle_to_avoid_duplicate_directions(self):
        with self.assertRaises(ValueError):
            sample_normal_guided_approaches([0, 0, 1], cone_angle_deg=0)


if __name__ == "__main__":
    unittest.main()
