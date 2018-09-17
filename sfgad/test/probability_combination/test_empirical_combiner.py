from unittest import TestCase

import numpy as np

from sfgad.modules.probability_combination.empirical_combiner import EmpiricalCombiner


class TestEmpiricalCombiner(TestCase):
    def setUp(self):
        self.combiner = EmpiricalCombiner()

    def test_combine_output(self):
        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]
        ref_p_values = np.array([[0.21, 0.12, 0.001, 0.15, 0.067], [0.21, 0.12, 0.21, 0.15, 0.067]], dtype=float)

        # test the right output
        self.assertEqual(self.combiner.combine(p_values, ref_p_values), 0.5)

    def test_combine_empty_list(self):
        empty_list = []

        # expect a value error
        self.assertRaises(ValueError, self.combiner.combine, empty_list)

    def test_combine_non_convertible_type(self):
        invalid_input = 'string_that_is_not_an_array_or_list'

        # expect an assertion error
        self.assertRaises(ValueError, self.combiner.combine, invalid_input)

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
        self.combiner = EmpiricalCombiner(direction='right-tailed')

        p_values = [0.21, 0.12, 0.021, 0.15, 0.067]
        ref_p_values = np.array([[0.21, 0.12, 0.1, 0.15, 0.067], [0.21, 0.12, 0.21, 0.15, 0.067]], dtype=float)

        # test the right output
        self.assertEqual(self.combiner.combine(p_values, ref_p_values), 1)

    def test_wrong_direction(self):
        # expect a value error
        self.assertRaises(ValueError, EmpiricalCombiner, direction='up')
