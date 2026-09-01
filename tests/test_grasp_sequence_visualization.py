import unittest


class GraspSequenceVisualizationTests(unittest.TestCase):
    def test_sequence_advances_states_then_models(self):
        from scripts.visualize_grasp_sequence import advance_sequence_index

        self.assertEqual(advance_sequence_index(0, 0, 2, 3), (0, 1))
        self.assertEqual(advance_sequence_index(0, 2, 2, 3), (1, 0))
        self.assertEqual(advance_sequence_index(1, 2, 2, 3), (0, 0))

    def test_model_spec_parser_preserves_paths_with_equals(self):
        from scripts.visualize_grasp_sequence import parse_model_spec

        parsed = parse_model_spec("cat=model=colmap/cat.ply=results/ours-main/cat")

        self.assertEqual(parsed["name"], "cat")
        self.assertEqual(parsed["object"], "model=colmap/cat.ply")
        self.assertEqual(parsed["results"], "results/ours-main/cat")


if __name__ == "__main__":
    unittest.main()
