import numpy as np

from unittest import TestCase
from sfgad.modules.probability_combiner.empirical import Empirical


class TestEmpirical(TestCase):

    def setUp(self):
        self.combiner = Empirical()

    def test_combine_output(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]
        ref_p_values = np.array([[0.21, 0.12, 0.001, 0.15, 0.067], [0.21, 0.12, 0.21, 0.15, 0.067]], dtype=float)

        # test the right output
        self.assertEqual(self.combiner.combine(p_values, ref_p_values), 0.5)

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

    def test_combine_no_reference(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]

        # expect an assertion error
        self.assertRaises(ValueError, self.combiner.combine, p_values)

    def test_combine_type_reference(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]
        ref_p_values = 42

        # expect an assertion error
        self.assertRaises(AssertionError, self.combiner.combine, p_values, ref_p_values)

    def test_combine_length_reference(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]
        ref_p_values = np.array([[0.21, 0.12, 0.001, 'A', 0.067], [0.21, 0.12, 0.21, 0.15, 0.067]])

        # expect an assertion error
        self.assertRaises(ValueError, self.combiner.combine, p_values, ref_p_values)

    def test_direction_change(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]
        ref_p_values = np.array([[0.21, 0.12, 0.1, 0.15, 0.067], [0.21, 0.12, 0.21, 0.15, 0.067]], dtype=float)

        self.combiner.set_direction('right')

        # test the right output
        self.assertEqual(self.combiner.combine(p_values, ref_p_values), 1)

    def test_wrong_direction(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]
        ref_p_values = np.array([[0.21, 0.12, 0.1, 0.15, 0.067], [0.21, 0.12, 0.21, 0.15, 0.067]], dtype=float)

        self.combiner.set_direction('up')

        # expect an assertion error
        self.assertRaises(ValueError, self.combiner.combine, p_values, ref_p_values)
