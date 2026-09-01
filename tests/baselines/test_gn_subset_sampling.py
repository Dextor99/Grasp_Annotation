import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class GnSubsetSamplingTests(unittest.TestCase):
    def test_core_api_validates_candidates_per_point_and_builds_manifest(self):
        from scripts.common_eval.stratified_subset import build_sampling_manifest, sample_stratified_ids

        collision = np.zeros((2, 3, 2), dtype=bool)
        selected, details = sample_stratified_ids(
            collision, target_count=4, candidates_per_point=6, seed=0
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(details["candidates_per_point"], 6)
        manifest = build_sampling_manifest(details, source_geometry_run="run")
        self.assertNotIn("candidate_ids", manifest)
        self.assertEqual(manifest["source_geometry_run"], "run")
        with self.assertRaisesRegex(ValueError, "candidates_per_point"):
            sample_stratified_ids(collision, candidates_per_point=5)
        with self.assertRaisesRegex(ValueError, "at least one grasp point"):
            sample_stratified_ids(np.zeros((0, 2), dtype=bool))

    def test_stratified_sample_is_exact_and_balanced_per_grasp_point(self):
        from scripts.common_eval.sample_gn_subset import stratified_sample_ids

        # Four grasp points, with eight candidates per point.  All candidates
        # are geometry-valid so the expected allocation is unambiguous.
        collision = np.zeros((4, 2, 2, 2), dtype=bool)
        selected, details = stratified_sample_ids(collision, target_size=10, seed=0)

        self.assertEqual(selected.dtype, np.int64)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(np.unique(selected)), 10)
        self.assertEqual(details["n_geometry_valid"], 32)
        self.assertEqual(details["selected_per_point"], [3, 3, 2, 2])
        point_ids = np.unravel_index(selected, collision.shape)[0]
        self.assertEqual(np.bincount(point_ids, minlength=4).tolist(), [3, 3, 2, 2])
        self.assertTrue(np.all(~collision.reshape(-1)[selected]))

    def test_same_seed_produces_same_sorted_ids_and_different_seed_can_change_them(self):
        from scripts.common_eval.sample_gn_subset import stratified_sample_ids

        collision = np.zeros((5, 3, 2), dtype=bool)
        first, _ = stratified_sample_ids(collision, target_size=15, seed=7)
        second, _ = stratified_sample_ids(collision, target_size=15, seed=7)
        other, _ = stratified_sample_ids(collision, target_size=15, seed=8)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.all(first[:-1] <= first[1:]))

    def test_short_strata_are_capped_and_deficit_is_redistributed(self):
        from scripts.common_eval.sample_gn_subset import stratified_sample_ids

        # Point 0 has only one valid candidate; the remaining quota must be
        # redistributed to the other points rather than reducing the sample.
        collision = np.ones((3, 2, 2), dtype=bool)
        collision[0, 0, 0] = False
        collision[1, :, :] = False
        collision[2, :, :] = False
        selected, details = stratified_sample_ids(collision, target_size=7, seed=0)

        self.assertEqual(len(selected), 7)
        self.assertEqual(details["selected_per_point"], [1, 3, 3])
        point_ids = np.unravel_index(selected, collision.shape)[0]
        self.assertEqual(np.bincount(point_ids, minlength=3).tolist(), [1, 3, 3])

    def test_write_artifacts_contains_reproducibility_manifest(self):
        from scripts.common_eval.sample_gn_subset import write_sampling_artifacts

        collision = np.zeros((2, 2), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "subset"
            details = write_sampling_artifacts(
                collision, output, target_size=3, seed=11,
                source_geometry_run=Path("geometry-run"),
            )
            ids_path = output / "sampled_candidate_ids.npy"
            manifest_path = output / "sampling_manifest.json"
            self.assertTrue(ids_path.is_file())
            self.assertTrue(manifest_path.is_file())
            np.testing.assert_array_equal(np.load(ids_path), details["candidate_ids"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 11)
            self.assertEqual(manifest["target_size"], 3)
            self.assertEqual(manifest["source_geometry_run"], "geometry-run")
            self.assertEqual(manifest["selected_size"], 3)


if __name__ == "__main__":
    unittest.main()
