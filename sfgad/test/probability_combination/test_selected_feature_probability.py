from unittest import TestCase

from sfgad.modules.probability_combination.selected_feature_probability import SelectedFeatureProbability


class TestSelectedFeatureProbability(TestCase):
    def setUp(self):
        self.combiner = SelectedFeatureProbability(feature_position=2)

    def test_combine_output(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]

        # test the right output
        self.assertEqual(self.combiner.combine(p_values), p_values[2])

    def test_combine_empty_list(self):
        p_values = []

        # expect a value error
        self.assertRaises(ValueError, self.combiner.combine, p_values)

    def test_combine_type_list(self):
        p_values = 42

        # expect an assertion error
        self.assertRaises(AssertionError, self.combiner.combine, p_values)

    def test_combine_type_elements(self):
        p_values = [0.21, 0.12, 'A', 0.15, 0.067]

        # expect an assertion error
        self.assertRaises(ValueError, self.combiner.combine, p_values)

    def test_invalid_position(self):
        combiner = SelectedFeatureProbability(feature_position=5)
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]

        # expect an assertion error
        self.assertRaises(ValueError, combiner.combine, p_values)
