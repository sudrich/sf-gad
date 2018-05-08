import pandas as pd

from .feature import Feature
from collections import defaultdict, Counter


class TwoHopReachByType(Feature):
    """
    The feature TwoHopReachByType of a single vertex is defined as the count of vertices in the 2-hop-neighborhood of a
    vertex grouped by vertex type.

    All vertex types should appear in the result data frame as columns, even if there are no occurrences of this vertex
    type in the current time step.
    """

    def __init__(self, vertex_types):
        self.names = ['TwoHopReachBy' + str(vertex_type) for vertex_type in vertex_types]
        self.vertex_types = vertex_types

        # mapping of nodes to feature names based on vertex types
        self.feature_names = {}

        # a dictionary, which contains the ids of the neighbors for each node
        self.neighbors = defaultdict(list)

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the two-hop reach.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns 'name' and 'TwoHopReachByTYPE' for each existing vertex type,
            and the calculated two_hop reach for all vertices and vertex types in the given df_edges.
        """

        # iterate over all edges, extract the neighbors and the vertex types
        iterator = zip(df_edges['SRC_NAME'], df_edges['SRC_TYPE'], df_edges['DST_NAME'], df_edges['DST_TYPE'])
        for s, s_type, d, d_type in iterator:
            self.interpret_edge(s, s_type, d, d_type)

        # count all vertices types in the 2-hop-neighborhood for each vertex
        two_hop_neighborhood = defaultdict(list)
        for v in self.neighbors:
            neighborhood = set(self.neighbors[v])

            # add all elements in the 2-hop reach
            for u in self.neighbors[v]:
                neighborhood.update(set(self.neighbors[u]))

            neighborhood = list(neighborhood)
            neighborhood.remove(v)

            two_hop_neighborhood[v] = Counter([self.feature_names[neighbor] for neighbor in neighborhood])

        # create the result data frame
        result_df = pd.DataFrame(columns=['name'] + self.names)
        for v in two_hop_neighborhood:
            data = two_hop_neighborhood[v]
            data['name'] = v
            result_df = result_df.append(data, ignore_index=True)

        result_df = result_df.fillna(0)

        # reset all dictionaries
        self.feature_names = {}
        self.neighbors = defaultdict(list)

        return result_df

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass

    def interpret_edge(self, s, s_type, d, d_type):
        """
        Interprets the given edge by updating the node neighbors and the mapping of nodes to feature_names.
        :param s: The source node (name) of the edge.
        :param s_type: The vertex type of the source node.
        :param d: The destination node (name) of the edge.
        :param d_type: The vertex type of the destination node.
        """

        # update the neighbors
        self.update_neighbor(s, d)
        self.update_neighbor(d, s)

        # map ids to types
        if s not in self.feature_names:
            self.feature_names[s] = 'TwoHopReachBy' + str(s_type)
        if d not in self.feature_names:
            self.feature_names[d] = 'TwoHopReachBy' + str(d_type)

    def update_neighbor(self, node, neighbor):
        """
        Updates the neighbor of the given node.
        :param node: The given node.
        :param neighbor: The neighbor to update.
        """

        if neighbor not in self.neighbors[node]:
            self.neighbors[node].append(neighbor)