import pandas as pd
import numpy as np
from collections import defaultdict, deque
import copy


class HotSpotInterpreter:
    """
    The HotSpot-interpreter is a helper class, which interprets a given edge_frame and stores all relevant information
    about the available vertices and edges.
    """

    def __init__(self, decay_lambda, half_life):

        # decay constant
        self.decay_lambda = decay_lambda
        # half life of an edge
        self.half_life = half_life

        # mapping of nodes to numbers
        self.ids = {}
        # reverse mapping of numbers to nodes names
        self.inv_ids = {}

        # a dictionary, which contains a tuple (t; f) for each edge: the time point t when the edge frequency was
        # updated, and the frequency f
        self.frequencies = {}
        # a boolean dictionary, which memorizes whether an edge was already updated in the current time step
        self.edge_updated = defaultdict(bool)

        # a dictionary, which contains a list for each node: the ids of the neighbors
        self.neighbors = defaultdict(list)

        # a dictionary, which contains the information about the 2 last interpret()-calls (t; n_nodes; f; neighbors):
        # the timestamp t, the number of nodes (n_nodes), the frequencies f, and the neighbors (neighbors)
        self.fit_buffer = deque(maxlen=2)
        self.fit_buffer.append((pd.Timestamp.min, 0, {}, defaultdict(list)))

    def interpret(self, df_edges):
        """
        Interprets the given edge_frame by: registering new nodes; updating the node neighbors; updating the edge
        frequencies; and updating the fit buffer.
        :param df_edges: The edge_frame to interpret.
        :return: the unique node names in df_edges, and the current time stamp
        """

        current_time, n_nodes, _, _ = self.fit_buffer[-1]

        # registration phase of nodes
        unique_nodes = list(pd.unique(df_edges[['SRC_NAME', 'DST_NAME']].values.ravel()))
        for node_name in unique_nodes:
            if node_name not in self.ids:
                n_nodes = self.register_node(node_name, n_nodes)

        # update the current time
        if not df_edges.empty:
            assert (min(df_edges["TIMESTAMP"]) - current_time).total_seconds() >= 0
            current_time = max(df_edges["TIMESTAMP"])

        # interpret the new edges
        self.interpret_df(df_edges, current_time)

        # update the fit buffer
        self.fit_buffer.append((current_time, n_nodes, copy.deepcopy(self.frequencies), copy.deepcopy(self.neighbors)))
        # reset the edge occurrences of the current time step
        self.edge_updated = defaultdict(bool)

        return current_time, unique_nodes

    ### HELPER METHODS

    def interpret_df(self, df_edges, current_time):
        """
        Interprets the given edge_frame by interpreting each edge.
        :param df_edges: The edge_frame to interpret.
        :param current_time: The current time stamp.
        """

        iterator = zip(df_edges["SRC_NAME"], df_edges["DST_NAME"])
        for s, d in iterator:
            self.interpret_edge(s, d, current_time)

    def interpret_edge(self, s, d, t):
        """
        Interprets the given edge by updating the node neighbors and the edge frequency.
        :param s: The source node (name) of the edge.
        :param d: The destination node (name) of the edge.
        :param t: The current time stamp.
        """

        # map source and destination nodes to their ids
        s, d = self.ids[s], self.ids[d]
        # make sure that source id is smaller than destination id
        if s > d:
            s, d = d, s
        edge = (s, d)

        # update the neighbors
        self.update_neighbor(s, d)
        self.update_neighbor(d, s)

        # update the occurrences with new timestamp and new frequency
        # if the edge has already occurred in current time window, then overwrite the last entry
        if not self.edge_updated[edge]:
            self.frequencies[edge] = (t, self.update_weighted_frequency(edge, t))
            self.edge_updated[edge] = True
        else:
            t, freq = self.frequencies[edge]
            self.frequencies[edge] = (t, freq + 1)

    def update_weighted_frequency(self, edge, t):
        """
        Updates the frequency of the given edge by considering the occurrence at the time t.
        :param edge: The edge, which frequency to update.
        :param t: The time stamp at which the edge arrived (again).
        """

        freq_new = 0

        # consider the previous frequency of this edge
        if edge in self.frequencies:
            t_old, freq_old = self.frequencies[edge]
            freq_new = freq_old * 2 ** (-self.decay_lambda * (t - t_old).total_seconds())

        # consider the new arrival of the edge
        freq_new += 1

        return freq_new

    def register_node(self, node_name, n_nodes):
        """
        Records the new node by assigning an ID to it.
        :param node_name: The new node to record.
        :param n_nodes: The current amount of nodes.
        :return: the updated amount of nodes.
        """

        self.ids[node_name] = n_nodes
        self.inv_ids[n_nodes] = node_name
        n_nodes += 1

        return n_nodes

    def update_neighbor(self, node_id, neighbor_id):
        """
        Updates the neighbor of the given node.
        :param node_id: The given node.
        :param neighbor_id: The neighbor to update.
        """

        if neighbor_id not in self.neighbors[node_id]:
            self.neighbors[node_id].append(neighbor_id)

    def turn_back_time(self):
        """
        Resets all the updates of the interpreter of the last time step by going one interpret_df()-call back
        """

        t_now, n_nodes_now, _, _ = self.fit_buffer[-1]
        t_prev_fit, n_nodes_prev_fit, frequencies_prev_fit, neighbors_prev_fit = self.fit_buffer[-2]

        # set back the ids and inv_ids (by deleting the newly arrived nodes)
        nodes_to_delete = range(n_nodes_prev_fit, n_nodes_now)
        for node_id in nodes_to_delete:
            node_name = self.inv_ids[node_id]
            del self.inv_ids[node_id]
            del self.ids[node_name]

        # set back the neighbors
        self.neighbors = neighbors_prev_fit

        # set back the frequencies
        self.frequencies = frequencies_prev_fit

        # set back the fit-buffer
        del self.fit_buffer[-1]

    ### GETTER METHODS

    def get_freq(self, edge):
        """
        Returns the edge frequency of the given edge at the current time.
        :param edge: The given edge.
        :return: The edge frequency at the current time.
        """

        if edge not in self.frequencies:
            raise ValueError("Error! The given edge is unknown!")

        t, freq = self.frequencies[edge]
        cur_time = self.get_current_time()

        # update the frequency to the current time
        if cur_time > t:
            freq *= 2 ** (-self.decay_lambda * (cur_time - t).total_seconds())
            self.frequencies[edge] = (cur_time, freq)

        return freq

    def get_node_id(self, node_name):
        """
        Returns the id of the given node characterized by its name.
        :param node_name: The given node.
        :return: The id of the given.
        """

        # Exception, if node_id is unknown
        if node_name not in self.ids:
            raise ValueError("Error! The given node_name is unknown!")

        return self.ids[node_name]

    def get_neighbor_id(self, node_id, neighbor_idx):
        """
        Returns the id of the neighbor at the given index.
        :param node_id: The given node.
        :param neighbor_idx: The index of the neighbor.
        :return: The id of the neighbor.
        """
        if neighbor_idx >= len(self.neighbors[node_id]):
            raise ValueError("Error! The given neighbor index is invalid!")

        return self.neighbors[node_id][neighbor_idx]

    def get_current_time(self):
        """
        Returns the current time of the interpreter.
        :return: The current time.
        """

        t, _, _, _ = self.fit_buffer[-1]

        return t

    def get_n_neighbors(self, node_id):
        """
        Returns the number of neighbors of the given node.
        :param node_id: The given node.
        :return: The number of neighbors.
        """

        # Exception, if node_id is unknown
        if node_id not in self.neighbors:
            raise ValueError("Error! The given node_id is unknown!")

        return len(self.neighbors[node_id])

    def get_neighbors(self, node_id):
        """
        Returns the neighbors of the given node.
        :param node_id: The given node.
        :return: A list with the neighbor ids.
        """

        # Exception, if node_id is unknown
        if node_id not in self.neighbors:
            raise ValueError("Error! The given node_id is unknown!")

        return self.neighbors[node_id]


    #def del_neighbor(self, node_id, neighbor_id):
    #    neighbor_index = self.get_neighbor_idx(node_id, neighbor_id)

    #    if not np.isnan(neighbor_index):
    #        del self.neighbors[node_id][neighbor_index]
    #        del self.neighbors_time[node_id][neighbor_index]

    #def get_neighbor_idx(self, node_id, neighbor_id):

    #    neighbor_index = np.nan

    #    if len(self.neighbors[node_id]) > 0:
    #        neighbors = self.neighbors[node_id]

    #        if neighbor_id in neighbors:
    #            neighbor_index = neighbors.index(neighbor_id)

    #    return neighbor_index

    #def get_node_name(self, node_id):

        # Exception, if node_id is unknown
    #    if node_id not in self.inv_ids:
    #        raise ValueError("Error! The given node_id is unknown!")

    #    return self.inv_ids[node_id]
