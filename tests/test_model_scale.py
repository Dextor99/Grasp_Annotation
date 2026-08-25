import unittest

from model_scale import get_model_scale


class ModelScaleTests(unittest.TestCase):
    def test_known_millimeter_models_are_not_rescaled(self):
        self.assertEqual(get_model_scale("model/huixing.ply"), 1.0)
        self.assertEqual(get_model_scale("model/shuilongtou.ply"), 1.0)

    def test_meter_based_colmap_inputs_are_converted_to_mm(self):
        self.assertEqual(get_model_scale("model/colmap/cat.ply"), 1000.0)
        self.assertEqual(get_model_scale("model/0623/cat.ply"), 1000.0)

    def test_unknown_paths_keep_legacy_meter_default(self):
        self.assertEqual(get_model_scale("custom/object.ply"), 1000.0)


if __name__ == "__main__":
    unittest.main()
