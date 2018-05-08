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

        # mapping of nodes to numbers
        self.ids = {}
        self.node_count = 0
        # reverse mapping of numbers to nodes names
        self.inv_ids = {}

        # mapping of node_ids to feature names based on vertex types
        self.feature_names = {}

        # a dictionary, which contains a list for each node: the ids of the neighbors
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

        # register nodes
        unique_nodes = list(pd.unique(df_edges[['SRC_NAME', 'DST_NAME']].values.ravel()))
        for node_name in unique_nodes:
            if node_name not in self.ids:
                self.node_count = self.register_node(node_name, self.node_count)

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

            two_hop_neighborhood[self.inv_ids[v]] = Counter([self.feature_names[neighbor] for neighbor in neighborhood])

        # create the result data frame
        result_df = pd.DataFrame(columns=['name'] + self.names)
        for v in two_hop_neighborhood:
            data = two_hop_neighborhood[v]
            data['name'] = v
            result_df = result_df.append(data, ignore_index=True)

        result_df = result_df.fillna(0)

        # reset all dictionaries
        self.ids = self.inv_ids = self.types = {}
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

    def interpret_edge(self, s, s_type, d, d_type):
        """
        Interprets the given edge by updating the node neighbors.
        :param s: The source node (name) of the edge.
        :param s_type: The vertex type of the source node.
        :param d: The destination node (name) of the edge.
        :param d_type: The vertex type of the destination node.
        """

        # map source and destination nodes to their ids
        s, d = self.ids[s], self.ids[d]
        # make sure that source id is smaller than destination id
        if s > d:
            s, d = d, s
            s_type, d_type = d_type, s_type

        # update the neighbors
        self.update_neighbor(s, d)
        self.update_neighbor(d, s)

        # map ids to types
        if s not in self.feature_names:
            self.feature_names[s] = 'TwoHopReachBy' + str(s_type)
        if d not in self.feature_names:
            self.feature_names[d] = 'TwoHopReachBy' + str(d_type)

    def update_neighbor(self, node_id, neighbor_id):
        """
        Updates the neighbor of the given node.
        :param node_id: The given node.
        :param neighbor_id: The neighbor to update.
        """

        if neighbor_id not in self.neighbors[node_id]:
            self.neighbors[node_id].append(neighbor_id)