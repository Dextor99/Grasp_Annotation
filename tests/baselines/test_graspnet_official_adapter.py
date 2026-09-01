import importlib
import unittest
from unittest.mock import patch

import numpy as np


class GraspNetOfficialAdapterTests(unittest.TestCase):
    def test_missing_backend_gives_actionable_install_message(self):
        from baselines.graspnet_annotation.official_adapter import OfficialBackendUnavailable, require_official_backend

        with patch("baselines.graspnet_annotation.official_adapter.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(OfficialBackendUnavailable, "graspnetAPI"):
                require_official_backend()

    def test_backend_gate_returns_imported_symbols_when_available(self):
        from baselines.graspnet_annotation.official_adapter import require_official_backend

        sentinel = object()
        with patch("baselines.graspnet_annotation.official_adapter.importlib.util.find_spec", return_value=object()), patch(
            "baselines.graspnet_annotation.official_adapter.importlib.import_module", return_value=sentinel
        ):
            modules = require_official_backend()
        self.assertEqual(len(modules), 2)
        self.assertIs(modules[0], sentinel)
        self.assertIs(modules[1], sentinel)

    def test_point_evaluation_validates_official_tensor_shape(self):
        from baselines.graspnet_annotation.official_adapter import PointEvaluation

        shape = (3, 3, 2)
        result = PointEvaluation(
            widths_m=np.full(shape, 0.04),
            collision=np.zeros(shape, dtype=bool),
            mu_min=np.full(shape, 0.4),
        )
        self.assertEqual(result.widths_m.shape, shape)
        with self.assertRaisesRegex(ValueError, "shape"):
            PointEvaluation(np.zeros(shape), np.zeros((3, 3), dtype=bool), np.zeros(shape))

    @unittest.skipUnless(importlib.util.find_spec("graspnetAPI"), "requires the dedicated GN-Full environment")
    def test_force_closure_sweep_has_annotation_order_and_ten_levels(self):
        from baselines.graspnet_annotation.official_adapter import build_force_closure_configs

        values, configs = build_force_closure_configs()
        self.assertEqual(len(values), 10)
        self.assertEqual([round(float(value), 1) for value in values], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        self.assertEqual(len(configs), 10)


if __name__ == "__main__":
    unittest.main()
