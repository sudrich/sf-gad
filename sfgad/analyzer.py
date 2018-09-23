import abc
import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .modules.observation_selection import ExternalSQLDatabase
from .modules.observation_selection import InMemoryDatabase


class SequentialAnalyzer(metaclass=abc.ABCMeta):
    # @abc.abstractmethod
    # def fit(self, df_edges):
    #     """
    #     Analyses the given edge_frame (i.e. the attributes).
    #     Updates the statistical model based on the current analysis.
    #     :param df_edges: The edge_frame to analyse.
    #     :return: None
    #     """
    #
    # @abc.abstractmethod
    # def transform(self, df_edges):
    #     """
    #     Analyses the given edge_frame (i.e. the attributes). Returns a Dataframe of the p_values of the affected nodes.
    #     Does not update the statistical model.
    #     :param df_edges: The edge_frame to analyse.
    #     :return: df_p_values
    #     """

    @abc.abstractmethod
    def fit_transform(self, df_edges):
        """
        Analyses the given edge_frame (i.e the attributes). Calculates the p_values of the affected nodes.
        Updates the statistical model based on the current analyses. Returns a Dataframe of the p_values.
        :param df_edges: The edge_frame to analyse.
        :return: df_p_values
        """


class Analyzer(SequentialAnalyzer):
    def __init__(self, features_list, observation_selection, weighting_function, probability_estimator,
                 probability_combiner, db_con=None, threshold=2, n_jobs=1):

        self.features_list = features_list
        self.observation_selection = observation_selection
        self.weighting_function = weighting_function
        self.probability_estimator = probability_estimator
        self.probability_combiner = probability_combiner

        self.threshold = threshold
        self.n_jobs = n_jobs

        if db_con:
            self.db = ExternalSQLDatabase(db_con, [name for f in features_list for name in f.names])
        else:
            self.db = InMemoryDatabase([name for f in features_list for name in f.names])
        self.db = InMemoryDatabase([name for f in features_list for name in f.names])

        self.time_window = int(0)

    def fit_transform(self, df_edges):
        feature_df_list = [f.process_vertices(df_edges, self.n_jobs) for f in self.features_list]

        time = df_edges['TIMESTAMP'].max()

        # Calculate every feature for every vertex and return a list of dataframes with columns (name, Feature1, ...)
        feature_df_list = [f.process_vertices(df_edges, self.n_jobs) for f in self.features_list]

        # List of all unique vertices in the current edge dataframe
        # complete_df = pd.DataFrame(unique_vertices(df_edges), columns=['name', 'type', 'age'])
        # complete_df['age'] = self.db.calculate_age(complete_df, self.time_window)

        complete_df = pd.DataFrame(unique_vertices(df_edges), columns=['name', 'type'])

        # Start p_value calculation for every vertex of the current dataframe
        # The p_value calculation might be split into separated processes
        if self.n_jobs > 1:
            vertices_split = np.array_split(complete_df, self.n_jobs)

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.transform_vertices_list)(vertices, feature_df_list)
                for vertices in vertices_split)

            p_values_list, p_f_values_list = zip(*list(results))
            p_values_list = list(itertools.chain.from_iterable(p_values_list))
            p_f_values_list = list(itertools.chain.from_iterable(p_f_values_list))
        else:
            p_values_list, p_f_values_list = self.transform_vertices_list(complete_df, feature_df_list)

        # Add the data frames from feature calculation to the complete_df
        for index, feature_df in enumerate(feature_df_list):
            complete_df = pd.merge(complete_df, feature_df, on='name')

        p_f_df = pd.DataFrame(p_f_values_list,
                              columns=['name'] + ['p_' + f_name for f in self.features_list for f_name in f.names])
        complete_df = pd.merge(complete_df, p_f_df, on='name')

        # Add Time and Time Window
        complete_df['time'] = time
        complete_df['time_window'] = self.time_window

        # Save to Database and increase the time window counter
        self.db.insert_records(complete_df)
        self.time_window += 1

        return pd.DataFrame(p_values_list, columns=['name', 'time_window', 'p_value'])

    def transform_vertices_list(self, vertices, feature_df_list):

        # A list to hold the row entries for the final p_value dataframe
        p_values_list = []
        p_f_values_list = []

        features = vertices[['name']]

        # Add the data frames from feature calculation to the feature dataframe
        for index, feature_df in enumerate(feature_df_list):
            features = pd.merge(features, feature_df, on='name')

        features_grouped = features.groupby('name')

        # Get dataframe with all relevant rows from database
        for v in vertices.itertuples(index=False):
            observations = self.observation_selection.gather(v.name, v.type, self.time_window, self.db)

            if observations.shape[0] < self.threshold:
                p_value = np.nan
                feature_probabilities = [np.nan for f in self.features_list for name in f.names]
            else:
                reference_features = observations[
                    [name for f in self.features_list for name in f.names]]

                # Get list of weights for every window
                weights = self.weighting_function.compute(reference_features,
                                                          pd.Series({'time_window': self.time_window, 'type': v.type}))

                # Get a list of calculated p_values for every feature
                feature_probabilities = self.probability_estimator.estimate(
                    features_grouped.get_group(v.name)[[name for f in self.features_list for name in f.names]],
                    reference_features, weights)

                reference_feature_probabilities = observations[
                    ['name', 'time_window'] + ['p_' + name for f in self.features_list for name in f.names]]

                # Combine the multiple p_values into a single p_value for the vertex
                p_value = self.probability_combiner.combine(feature_probabilities, reference_feature_probabilities)[0]

            p_f_df_row = dict({'name': v.name}, **{'p_' + f: p_f_value for f, p_f_value in
                                                   zip([f_name for f in self.features_list for f_name in f.names],
                                                       feature_probabilities)})
            p_f_values_list.append(p_f_df_row)

            # Generate a new row entry
            p_df_row = {'name': v.name, 'time_window': self.time_window, 'p_value': p_value}
            p_values_list.append(p_df_row)

        return p_values_list, p_f_values_list


## HELPER

def unique_vertices(df_edges):
    df_src = df_edges[['SRC_NAME', 'SRC_TYPE']]
    df_dst = df_edges[['DST_NAME', 'DST_TYPE']]

    df_src.columns = ['name', 'type']
    df_dst.columns = ['name', 'type']

    df_vertices = pd.concat([df_src, df_dst])
    df_vertices = df_vertices.drop_duplicates()

    return df_vertices
