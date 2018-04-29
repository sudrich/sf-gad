import pandas as pd

from .feature import Feature


class VertexActivity(Feature):
    """
    The feature VertexActivity of a single vertex is defined as a binary indicator, which is 1 if the vertex has at
    least 1 incident edge, and 0 otherwise.
    """

    def __init__(self):
        self.names = ['VertexActivity']
        self.nodes = []

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the vertex activity.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns ['name', 'VertexActivity'] and the calculated vertex activity for all
            existing vertices.
        """

        active_nodes = list(pd.unique(df_edges[['SRC_NAME', 'DST_NAME']].values.ravel()))

        result_df = pd.DataFrame(data={'name': active_nodes, 'VertexActivity': 1},
                                 columns=['name', 'VertexActivity'])

        # add inactive vertices to result_df
        inactive_nodes = [name for name in self.nodes if name not in active_nodes]
        if len(inactive_nodes) > 0:
            inactive_nodes_df = pd.DataFrame(data={'name': inactive_nodes, 'VertexActivity': 0},
                                             columns=['name', 'VertexActivity'])
            result_df = result_df.append(inactive_nodes_df).reset_index(drop=True)

        # add new occurred vertices to self.nodes
        new_nodes = [name for name in active_nodes if name not in self.nodes]
        for name in new_nodes:
            self.nodes.append(name)

        return result_df

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass
