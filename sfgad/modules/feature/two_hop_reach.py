import pandas as pd
import copy

from .feature import Feature
from collections import defaultdict, deque


class TwoHopReach(Feature):
    """
    The feature TwoHopReach of a single vertex is defined as the count of vertices in the 2-hop-neighborhood of a
    vertex.
    """

    def __init__(self):
        self.names = ['TwoHopReach']

        # mapping of nodes to numbers
        self.ids = {}
        self.node_count = 0
        # reverse mapping of numbers to nodes names
        self.inv_ids = {}

        # a dictionary, which contains a list for each node: the ids of the neighbors
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

        # register nodes
        unique_nodes = list(pd.unique(df_edges[['SRC_NAME', 'DST_NAME']].values.ravel()))
        for node_name in unique_nodes:
            if node_name not in self.ids:
                self.node_count = self.register_node(node_name, self.node_count)

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

            two_hop_neighborhood[self.inv_ids[v]] = len(neighborhood)

        # transform the dictionary to a data frame
        result_df = pd.DataFrame(list(two_hop_neighborhood.items()), columns=['name', 'TwoHopReach'])

        # reset all dictionaries
        self.ids = self.inv_ids = {}
        self.node_count = 0
        self.neighbors = defaultdict(list)

        return result_df

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass

    def register_node(self, node_name, node_count):
        """
        Records the new node by assigning an ID to it.
        :param node_name: The new node to record.
        :param node_count: The current amount of nodes.
        :return: the updated amount of nodes.
        """

        self.ids[node_name] = node_count
        self.inv_ids[node_count] = node_name
        node_count += 1

        return node_count

    def interpret_edge(self, s, d):
        """
        Interprets the given edge by updating the node neighbors.
        :param s: The source node (name) of the edge.
        :param d: The distance node (name) of the edge.
        """

        # map source and destination nodes to their ids
        s, d = self.ids[s], self.ids[d]
        # make sure that source id is smaller than destination id
        if s > d:
            s, d = d, s

        # update the neighbors
        self.update_neighbor(s, d)
        self.update_neighbor(d, s)

    def update_neighbor(self, node_id, neighbor_id):
        """
        Updates the neighbor of the given node.
        :param node_id: The given node.
        :param neighbor_id: The neighbor to update.
        """

        if neighbor_id not in self.neighbors[node_id]:
            self.neighbors[node_id].append(neighbor_id)