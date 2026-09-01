import tempfile
import unittest
from pathlib import Path

import numpy as np


class ForceClosureSubsetMergeTests(unittest.TestCase):
    def test_merge_requires_exact_manifest_union(self):
        from scripts.common_eval.merge_force_closure_subset import merge_subset

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            geometry = root / "geometry"
            shards = root / "shards"
            geometry.mkdir()
            shards.mkdir()
            collision = np.zeros((1, 1, 1, 3), dtype=bool)
            np.savez_compressed(
                geometry / "grasp_labels.npz",
                points=np.zeros((1, 3), dtype=np.float32),
                offsets=np.zeros((1, 1, 1, 3, 3), dtype=np.float32),
                collision=collision,
                scores=np.full(collision.shape, -1, dtype=np.float32),
            )
            ids = np.array([0, 2], dtype=np.int64)
            np.save(root / "ids.npy", ids, allow_pickle=False)
            np.savez_compressed(
                shards / "subset_fc_shard_000000.npz",
                candidate_ids=np.array([0], dtype=np.int64),
                mu_min=np.array([0.2], dtype=np.float32),
                scored_mask=np.array([True]),
                error_mask=np.array([False]),
            )
            with self.assertRaisesRegex(RuntimeError, "missing"):
                merge_subset(geometry, shards, root / "ids.npy", root / "merged")

    def test_merge_writes_subset_labels_and_summary(self):
        from scripts.common_eval.merge_force_closure_subset import merge_subset

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            geometry = root / "geometry"
            shards = root / "shards"
            geometry.mkdir()
            shards.mkdir()
            collision = np.zeros((1, 1, 1, 3), dtype=bool)
            np.savez_compressed(
                geometry / "grasp_labels.npz",
                points=np.zeros((1, 3), dtype=np.float32),
                offsets=np.zeros((1, 1, 1, 3, 3), dtype=np.float32),
                collision=collision,
                scores=np.full(collision.shape, -1, dtype=np.float32),
            )
            ids = np.array([0, 2], dtype=np.int64)
            np.save(root / "ids.npy", ids, allow_pickle=False)
            np.savez_compressed(
                shards / "subset_fc_shard_000000.npz",
                candidate_ids=ids,
                mu_min=np.array([0.2, 0.7], dtype=np.float32),
                scored_mask=np.array([True, True]),
                error_mask=np.array([False, False]),
            )
            summary = merge_subset(geometry, shards, root / "ids.npy", root / "merged")
            self.assertEqual(summary["subset_count"], 2)
            self.assertEqual(summary["n_fc_valid"], 2)
            self.assertTrue((root / "merged" / "labels.npz").is_file())


if __name__ == "__main__":
    unittest.main()
