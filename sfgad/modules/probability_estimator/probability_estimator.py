import abc


class ProbabilityEstimator(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def estimate(self, vertex_name, features_df, reference_feature_values, weights):
        """
        Takes a vertex and a reference to the database.
        Returns a list of p_values with a p_value for each given feature value in features_df. These p_values are
        calculated based on the given reference_feature_values.
        :param vertex_name: the name of the vertex.
        :param features_df: Dataframe of the feature values.
        :param reference_feature_values: List of tuples of windows and the corresponding feature values.
        :param weights: the weights for the different windows.
        :return: List of p_values.
        """