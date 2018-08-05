import abc


class ProbabilityCombiner(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def combine(self, p_values, ref_p_values=None):
        """
        Takes a list of p_values and combines them into a single p_value.
        :param p_values: a list of p_values of features.
        :param ref_p_values: p-values of reference observations (if needed).
        :return: The combined p-value.
        """