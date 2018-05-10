from unittest import TestCase
from sfgad.modules.probability_combiner.max_probability import MaxProbability


class TestMaxProbability(TestCase):

    def setUp(self):
        self.combiner = MaxProbability()

    def test_combine_output(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]

        # test the right output for position 2
        self.assertEqual(self.combiner.combine(p_values), 0.21)

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
