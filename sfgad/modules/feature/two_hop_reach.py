import pandas as pd

from .feature import Feature
from collections import defaultdict


class TwoHopReach(Feature):
    """
    The feature TwoHopReach of a single vertex is defined as the count of vertices in the 2-hop-neighborhood of a
    vertex.
    """

    def __init__(self):
        self.names = ['TwoHopReach']

        # a dictionary, which contains the neighbors for each node
        self.neighbors = defaultdict(list)

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the two-hop reach.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns ['name', 'TwoHopReach'] and the calculated two-hop reach for all vertices
            in the given df_edges.
        """

        # iterate over all edges and extract the neighbors
        iterator = zip(df_edges["SRC_NAME"], df_edges["DST_NAME"])
        for s, d in iterator:
            self.interpret_edge(s, d)

        # count all vertices in the 2-hop-neighborhood for each vertex
        two_hop_neighborhood = defaultdict(int)
        for v in self.neighbors:
            neighborhood = set(self.neighbors[v])

            # add all elements in the 2-hop reach
            for u in self.neighbors[v]:
                neighborhood.update(set(self.neighbors[u]))

            neighborhood = list(neighborhood)
            neighborhood.remove(v)

            two_hop_neighborhood[v] = len(neighborhood)

        # transform the dictionary to a data frame
        result_df = pd.DataFrame(list(two_hop_neighborhood.items()), columns=['name', 'TwoHopReach'])

        # reset neighbors dictionary
        self.neighbors = defaultdict(list)

        return result_df

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass

    def interpret_edge(self, s, d):
        """
        Interprets the given edge by updating the node neighbors.
        :param s: The source node (name) of the edge.
        :param d: The destination node (name) of the edge.
        """

        self.update_neighbor(s, d)
        self.update_neighbor(d, s)

    def update_neighbor(self, node, neighbor):
        """
        Updates the neighbor of the given node.
        :param node: The given node.
        :param neighbor: The neighbor to update.
        """

        if neighbor not in self.neighbors[node]:
            self.neighbors[node].append(neighbor)