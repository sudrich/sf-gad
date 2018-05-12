import numpy as np

from .probability_combiner import ProbabilityCombiner


class Empirical(ProbabilityCombiner):

    def __init__(self, direction='left'):

        self.direction = direction

    def combine(self, p_values, ref_p_values=None):
        """
        Takes a list of p_values and combines them into a single p_value by calculating the average.
        :param p_values: a list of p_values of features.
        :param ref_p_values: p-values of reference observations.
        :return: The combined p-value.
        """
        ### INPUT VALIDATION

        # assert p_values is a list
        assert type(p_values) == list

        # check that p_values is not empty
        l = len(p_values)
        if l == 0:
            raise ValueError('The given list of p_values is empty')

        # check that all elements in p_values are floats (or integers)
        if not all(isinstance(x, (int, float)) for x in p_values):
            raise ValueError('The elements in p_values should all be of the type \'float\'')

        # check that ref_p_values are given
        if ref_p_values is None:
            raise ValueError('The empirical combiner needs reference p values to calculate the combined p value!')

        # assert ref_p_values is a ndarray
        assert type(ref_p_values) == np.ndarray

        # check that the reference p_values of each reference observation are all floats (or integers)
        for ref in ref_p_values:
            if not all(isinstance(x, (int, float)) for x in ref):
                raise ValueError('The p_values of each reference observation should all be of the type \'float\'')

        ### FUNCTION CODE

        min_p_value = min(p_values)
        min_ref_p_values = ref_p_values.min(axis=1)

        return self.empirical(min_p_value, min_ref_p_values, weights=np.ones(len(min_ref_p_values)),
                              direction=self.direction)

    def empirical(self, value, references, weights, direction='right'):
        """
        Execute the empirical p-value combination.
        :param value: the given minimal p-value
        :param references: the minimal p_values of the reference observations
        :param weights: weights for the reference observations
        :param direction: direction for the empirical calculation.
        :return: the empirical p_value
        """
        isnan = np.isnan(references)

        if direction == 'right':
            conditions = references[~isnan] >= value
        else:
            conditions = references[~isnan] <= value

        sum_all_weights = weights[~isnan].sum()
        sum_conditional_weights = (conditions * weights[~isnan]).sum()

        if sum_all_weights == 0:
            return np.nan

        return sum_conditional_weights / sum_all_weights

    def set_direction(self, direction):
        """
        Sets a direction for the empirical calculation.
        :param direction: the direction for the empirical calculation.
        """

        self.direction = direction
