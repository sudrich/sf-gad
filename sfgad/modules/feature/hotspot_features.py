import pandas as pd
import numpy as np

from collections import defaultdict, deque
from functools import partial
from joblib import Parallel, delayed

from .feature import Feature
from .helper.hotspot_interpreter import HotSpotInterpreter


class HotSpotFeatures(Feature):
    """
    The HotSpotFeatures class contains two features: the CorrelationChange and the MagnitudeChange.
    The feature CorrelationChange of a single vertex is defined as the greatest eigenvector of the decay-based product
    matrix of this vertex, which represents the correlation structure of the locality.
    The feature MagnitudeChange of a single vertex is defined as the greatest eigenvalue of the decay-based product
    matrix.
    """

    def __init__(self, half_life=3, window_size=10):
        # the names of the features
        self.names = ["CorrelationChange", "MagnitudeChange"]

        # indicates whether the activity should be recorded and considered for future computations, or not
        self.update_activity = True
        self.half_life = half_life
        self.decay_lambda = 1 / (self.half_life * window_size)
        self.interpreter = HotSpotInterpreter(self.decay_lambda, self.half_life)

        # the age of the nodes
        self.node_age = defaultdict(int)
        # the product matrices of the last time step (t; product_matrix)
        self.prev_product_matrices = {}
        # contains for each node a deque of tuples of the form: (correlation; magnitude)
        self.activity_buffer = defaultdict(partial(deque, maxlen=self.half_life))

        self.prev_timestamps = deque(maxlen=self.half_life)

    def reset(self):
        self.interpreter = HotSpotInterpreter(self.decay_lambda, self.half_life)

        # the age of the nodes
        self.node_age = defaultdict(int)
        # the product matrices of the last time step (t; product_matrix)
        self.prev_product_matrices = {}
        # contains for each node a deque of tuples of the form: (correlation; magnitude)
        self.activity_buffer = defaultdict(partial(deque, maxlen=self.half_life))

        self.prev_timestamps = deque(maxlen=self.half_life)

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the features CorrelationChange and
        MagnitudeChange.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns ['name', 'CorrelationChange', 'MagnitudeChange'] and the calculated
            feature values for all vertices in the given df_edges.
        """

        self.update_activity = update_activity

        # interpret the edges and return the names of the active nodes in df_edges and the current time
        t, node_names = self.interpreter.interpret(df_edges)

        # TODO: parallel computation
        # compute the new feature values for each unique node in df_edges
        # if n_jobs > 1:
        #     node_names_split = np.array_split(node_names, n_jobs)
        #
        #     results = Parallel(n_jobs=n_jobs)(  #, backend="threading")(
        #         delayed(self.compute_multiple_nodes)(t, nodes)
        #         for nodes in node_names_split)
        #
        #     results = list(itertools.chain.from_iterable(results))
        # else:
        #     results = self.compute_multiple_nodes(t, node_names)

        results = self.compute_multiple_nodes(t, node_names)


        # update all activity
        if self.update_activity:
            # update the age of all nodes
            for node_id in self.interpreter.inv_ids:
                self.node_age[node_id] += 1

            # update the product_matrix, correlation and magnitude of the active nodes
            for node_result in results:
                node_id = self.interpreter.get_node_id(node_result[0])
                self.prev_product_matrices[node_id] = (t, node_result[1])
                self.activity_buffer[node_id].append((node_result[2], node_result[3]))

            # update the correlation and magnitude of all inactive nodes
            if len(self.prev_timestamps) > 0:
                inactive_nodes = set(self.interpreter.ids) - set(node_names)

                t_dif = (t - self.prev_timestamps[-1]).total_seconds()
                decay = 2 ** (-2 * self.decay_lambda * t_dif)

                for node_name in inactive_nodes:
                    node_id = self.interpreter.get_node_id(node_name)
                    cor_old, mag_old = self.activity_buffer[node_id][-1]
                    self.activity_buffer[node_id].append((cor_old, mag_old * decay))

            # update the timestamps
            self.prev_timestamps.append(t)
        else:
            # or set the interpreter back, if the activity shouldn't be recorded.
            if not df_edges.empty:
                self.interpreter.turn_back_time()

                # delete all new keys in node_age
                _, n_nodes, _, _ = self.interpreter.fit_buffer[-1]
                for k in list(self.node_age.keys()):
                    if k >= n_nodes:
                        self.node_age.pop(k)

        rel_results = []
        for node_result in results:
            #     if np.isnan(node_result[4]):
            #         continue
            rel_results.append((node_result[0], node_result[4], node_result[5]))

        # transform the relevant results into a dataframe
        results_df = pd.DataFrame(rel_results, columns=['name'] + self.names)

        return results_df

    def compute_multiple_nodes(self, t, node_names):
        """
        Computes for all given nodes the new feature values for CorrelationChange and
        MagnitudeChange. In addition it also computes and returns the data, which is needed for future computation of
        the feature values.
        :param t: The current time stamp.
        :param node_names: The names of the active nodes.
        :return a list with an entry for each active node. An entry contains the node_name, the newly calculated
        product_matrix, the calculated correlation and magnitude, and the calculated feature values CorrelationChange
        and MagnitudeChange.
        """

        results = []

        for node_name in node_names:
            product_matrix, cor, mag, cor_change, mag_change = self.compute(node_name, t)
            results.append((node_name, product_matrix, cor, mag, cor_change, mag_change))

        return results

    def compute(self, node_name, t):
        """
        Computes for a given single node the new feature values for CorrelationChange and
        MagnitudeChange. In addition it also computes and returns the data, which is needed for future computation of
        the feature values.
        :param node_name: The name of the given nodes.
        :param t: The current time stamp.
        :return a tuple, which contains the newly calculated product_matrix, the calculated correlation and magnitude,
        and the calculated feature values CorrelationChange and MagnitudeChange.
        """

        # map the node_name to its id
        node_id = self.interpreter.get_node_id(node_name)

        # calculate or update the product matrix
        product_matrix = self.calculate_product_matrix(node_id, t)

        # calculate correlation and magnitude
        cor, mag = self.cal_matrix_features(product_matrix)

        # calculate changes in the correlation and magnitude of the node
        cor_change, mag_change = self.cal_activity_changes(node_id, cor, mag)

        return product_matrix, cor, mag, cor_change, mag_change

    ### HELPER METHODS

    def calculate_product_matrix(self, node_id, t):
        """
        Calculates the product matrix for a given node at time t.
        :param node_id: The id of the given nodes.
        :param t: The current time stamp.
        :return the new product matrix
        """

        n_neighbors = self.interpreter.get_n_neighbors(node_id)

        if self.node_age[node_id] > 0:
            product_matrix = self.consider_prev_product_matrix(node_id, t, n_neighbors)
        else:
            product_matrix = np.zeros((n_neighbors, n_neighbors))

        # it should be ensured, that the reconstructed edges are ordered by the order in neighbors
        edges = self.reconstruct_edges(node_id)
        fs = [self.interpreter.get_freq(edge) for edge in edges]
        fp = np.outer(fs, fs)

        return product_matrix + fp

    def consider_prev_product_matrix(self, node_id, t, n_neighbors):
        """
        Helper method for the product matrix calculation. If a product matrix was calculated in a previous time step, it
        is more efficient to consider the previous matrix and don't calculate the new matrix from scratch.
        :param node_id: The id of the given nodes.
        :param t: The current time stamp.
        :param n_neighbors: The count of neighbors of the given node.
        :return the previous product matrix updated to time t
        """

        t_old, product_matrix_old = self.prev_product_matrices[node_id]
        decay_factor = 2 ** (-2 * self.decay_lambda * (t - t_old).total_seconds())

        product_matrix = np.copy(product_matrix_old)
        product_matrix *= decay_factor

        if n_neighbors > product_matrix.shape[0]:
            prev_size = product_matrix.shape[0]
            size_dif = n_neighbors - prev_size

            # add columns and rows
            cols = np.zeros((prev_size, size_dif))
            rows = np.zeros((size_dif, n_neighbors))
            product_matrix = np.append(product_matrix, cols, axis=1)
            product_matrix = np.append(product_matrix, rows, axis=0)

        return product_matrix

    def reconstruct_edges(self, node_id):
        """
        Reconstructs the edges of the given node to all it's neighbors.
        The reconstructed edges are ordered by the order of the neighbors.
        :param node_id: The id of the given node.
        :return the edges to all neighbors of the given node.
        """

        # translate neighbor_index to its id
        # get ids of the neighbors
        neighbor_ids = self.interpreter.get_neighbors(node_id)

        # reconstruct the edges
        edges = [(node_id, neighbor_id) if node_id < neighbor_id else (neighbor_id, node_id) for neighbor_id in
                 neighbor_ids]

        return edges

    def cal_matrix_features(self, product_matrix):
        """
        Calculates the correlation and the magnitude from the given matrix.
        :param product_matrix: The given product matrix.
        :return the correlation and the magnitude.
        """

        # get the greatest eigenvector and eigenvalue
        w, v = np.linalg.eigh(product_matrix)

        # derive and return the correlation and magnitude
        return list(v[:, -1]), w[-1]

    def cal_activity_changes(self, node_id, cor, mag):
        """
        Calculates the correlation and magnitude changes for the given node based on the given activity values.
        :param node_id: The given node.
        :param cor: The given new correlation value.
        :param mag: The given new magnitude value.
        :return the correlation change and the magnitude change.
        """

        cor_change = mag_change = np.nan

        if self.node_age[node_id] > 0:
            cor_old, mag_old = self.activity_buffer[node_id][0]

            # the correlation vectors should have the same length, so the vector is scaled-up if necessary
            cor_change = 1 - abs(np.dot(cor_old + [0] * (len(cor) - len(cor_old)), cor))

            mag_change = mag_old - mag

        return cor_change, mag_change