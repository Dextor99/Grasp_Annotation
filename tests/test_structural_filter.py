import unittest

from grasp_detect import filter_structurally_valid_grippers


class StructuralFilterTests(unittest.TestCase):
    def test_rejects_zero_depth_or_zero_opening(self):
        candidates = [
            {"id": 1, "depth": 0.0, "opening": 15.0},
            {"id": 2, "depth": 10.0, "opening": 0.0},
            {"id": 3, "depth": 0.0, "opening": 0.0},
            {"id": 4, "depth": 10.0, "opening": 15.0},
        ]
        self.assertEqual(
            [candidate["id"] for candidate in filter_structurally_valid_grippers(candidates)],
            [4],
        )

    def test_does_not_mutate_candidate_records(self):
        candidate = {"id": 1, "depth": 10.0, "opening": 15.0}
        result = filter_structurally_valid_grippers([candidate])
        self.assertIs(result[0], candidate)


if __name__ == "__main__":
    unittest.main()
