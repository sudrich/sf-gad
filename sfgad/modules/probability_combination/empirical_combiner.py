import numpy as np

from sfgad.utils import check_p_values
from .probability_combiner import ProbabilityCombiner


class EmpiricalCombiner(ProbabilityCombiner):
    def __init__(self, direction='left-tailed'):

        if direction not in ['right-tailed', 'left-tailed']:
            raise ValueError("The given direction for empirical probability calculation is not known! "
                             "Possible directions are: 'right-tailed' & 'left-tailed'.")

        self.direction = direction

    def combine(self, p_values, ref_p_values=None):
        """
        Takes a list of p_values and combines them into a single p_value by calculating the average.
        :param p_values: a list of p_values of features.
        :param ref_p_values: p-values of reference observations.
        :return: The combined p-value.
        """
        ### INPUT VALIDATION

        p_values = check_p_values(p_values)

        # # assert p_values is a list
        # assert type(p_values) == list
        #
        # # check that p_values is not empty
        # if len(p_values) == 0:
        #     raise ValueError('The given list of p_values is empty')
        #
        # # check that all elements in p_values are floats
        # if not all(isinstance(x, (int, float)) for x in p_values):
        #     raise ValueError('The elements in p_values should all be of the type \'float\'')

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

        min_ref_p_values = ref_p_values.min(axis=1)

        combined_p_values = np.apply_along_axis(
            lambda x: self.empirical(min(x), min_ref_p_values, weights=np.ones(len(min_ref_p_values)),
                                     direction=self.direction), axis=1, arr=p_values)

        return combined_p_values

    def empirical(self, value, references, weights, direction):
        """
        Execute the empirical p-value combination.
        :param value: the given minimal p-value
        :param references: the minimal p_values of the reference observations
        :param weights: weights for the reference observations
        :param direction: direction for the empirical calculation.
        :return: the empirical p_value
        """
        isnan = np.isnan(references)

        if direction == 'right-tailed':
            conditions = references[~isnan] >= value
        elif direction == 'left-tailed':
            conditions = references[~isnan] <= value
        else:
            raise ValueError("The given direction for empirical calculation is not known.")

        sum_all_weights = weights[~isnan].sum()
        sum_conditional_weights = (conditions * weights[~isnan]).sum()

        if sum_all_weights == 0:
            return np.nan

        return sum_conditional_weights / sum_all_weights
