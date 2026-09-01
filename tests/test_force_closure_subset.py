import tempfile
import unittest
from pathlib import Path

import numpy as np


class ForceClosureSubsetTests(unittest.TestCase):
    def test_partition_ids_uses_stable_shard_ranges(self):
        from scripts.common_eval.run_force_closure_subset import partition_ids

        ids = np.array([9, 2, 15, 20, 21], dtype=np.int64)
        chunks = partition_ids(ids, shard_size=2)
        self.assertEqual([chunk.tolist() for chunk in chunks], [[9, 2], [15, 20], [21]])

    def test_partition_rejects_invalid_ids(self):
        from scripts.common_eval.run_force_closure_subset import partition_ids

        with self.assertRaisesRegex(ValueError, "shard_size"):
            partition_ids(np.array([1], dtype=np.int64), shard_size=0)


if __name__ == "__main__":
    unittest.main()
