import abc


class ProbabilityCombiner(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def combine(self, p_values):
        """
        Takes a list of p_values and combines them into a single p_value.
        :param p_values: a list of p_values of features.
        :return: The combined p-value.
        """