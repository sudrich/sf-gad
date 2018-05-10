from .probability_combiner import ProbabilityCombiner
from scipy.stats import combine_pvalues


class FisherMethod(ProbabilityCombiner):

    def combine(self, p_values):
        """
        Takes a list of p_values and combines them into a single p_value using the Fisher’s method.
        :param p_values: a list of p_values of features.
        :return: The combined p-value.
        """

        _, p = combine_pvalues(p_values, method='fisher', weights=None)
        return p
