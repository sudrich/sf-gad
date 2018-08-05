import pandas as pd
import numpy as np
import datetime

from .feature import Feature
from itertools import dropwhile


class ExternalFeature(Feature):
    """
    The external feature maps a given node to a measured value based on a given dictionary.
    The idea behind this feature is, that nodes could be monitoring stations and there exists a mapping function with
    nodes as keys and lists of measured data (time, value) as values.
    """

    def __init__(self, values_dict):
        self.names = ['ExternalFeature']

        # the values dictionary should map nodes to -> list of tuples (time, value) sorted by time
        # check the scheme of the given dictionary first:
        if not isinstance(values_dict, dict):
            raise ValueError("Error! The given argument 'values_dict' should be a dictionary with node names as keys "
                             "and lists of measured data (time, value) as values!")
        #  all keys should be strings, ...
        if not all(isinstance(k, str) for k in values_dict.keys()):
            raise ValueError("Error! Not all keys are strings in the given dictionary!")
        # ... all values should be lists
        if not all(isinstance(v, list) for v in values_dict.values()):
            raise ValueError("Error! Not all values are lists in the given dictionary!")
        # ..., and all list elements should be tuples of the kind (time, value)
        for k in values_dict:
            if not all(isinstance(e, tuple) for e in values_dict[k]):
                raise ValueError("Error! Not all list elements of the nodes are tuples in the given dictionary!")
            list_data = list(zip(*values_dict[k]))
            # check the type of the timestamps
            if not all(isinstance(d, datetime.datetime) for d in list_data[0]):
                raise ValueError("Error! The list elements of the nodes in the given dictionary should be tuples of "
                                 "the kind (datetime, value)!")
            # no check for the value, because the type is domain specific
            if not self.is_sorted(list_data[0]):
                raise ValueError("Error! The list elements of the nodes in the given dictionary should be sorted by "
                                 "time!")

        self.values_dict = values_dict

    def reset(self):
        pass

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and maps each node to its measured value at the current timestamp, if there
        is an entry in the given dictionary.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns ['name', 'ExternalFeature'] and the measured values at the current
            timestamp for all vertices in the given df_edges.
        """

        # read out the active nodes and the current time
        active_nodes = list(pd.unique(df_edges[['SRC_NAME', 'DST_NAME']].values.ravel()))
        current_time = max(df_edges["TIMESTAMP"])

        # map each node to its measured value
        external_values = []

        for node in active_nodes:
            # check if there is an entry in the given dictionary for the current node
            if node in self.values_dict:
                node_values = self.values_dict[node]
                node_values = list(dropwhile(lambda record: record[0] < current_time, node_values))

                if len(node_values) > 0:
                    # read out the latest measured value after the current timestamp and add it to the results
                    _, latest_value = node_values[0]
                    external_values.append(latest_value)
                else:
                    # append NaN if there are no more measured values at the current time
                    external_values.append(np.nan)

                # update the dictionary
                self.values_dict[node] = node_values
            else:
                # append NaN if the node is not known
                external_values.append(np.nan)

        # create the result dataframe
        result_df = pd.DataFrame(data={'name': active_nodes, 'ExternalFeature': external_values},
                                 columns=['name', 'ExternalFeature'])

        return result_df

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass

    def is_sorted(self, lst, key=lambda x: x):
        """
        A helper function which checks whether the given list is sorted, or not.
        :param lst: The given list to check.
        :param key: key by which the list should be sorted.
        :return True, if the list is sorted, false otherwise.
        """

        for i, el in enumerate(lst[1:]):
            # i is the index of the previous element
            if key(el) < key(lst[i]):
                return False

        return True
