import pandas as pd

from .feature import Feature


class VertexActivityByType(Feature):
    """
    The feature VertexActivityByType of a single vertex is defined as a binary indicator, which is 1 if the vertex has
    at least 1 incident edge of a specific edge type, and 0 otherwise.

    All edge types should appear in the result data frame as columns, even if there are no occurrences of this edge type
    in the current time step.
    """

    def __init__(self, edge_types):
        self.names = ['VertexActivityBy' + str(edge_type) for edge_type in edge_types]
        self.nodes = []

    def reset(self):
        self.nodes = []

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the vertex activity by edge type.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns 'name' and 'VertexActivityByTYPE' for each existing edge type,
            and the calculated vertex activity for all vertices and edge types in the given df_edges.
        """

        # determine the active nodes
        active_nodes = list(pd.unique(df_edges[['SRC_NAME', 'DST_NAME']].values.ravel()))

        # calculate the activity by edge type
        src_counts = df_edges.groupby(['SRC_NAME', 'E_TYPE']).size()
        dst_counts = df_edges.groupby(['DST_NAME', 'E_TYPE']).size()

        src_counts[:] = dst_counts[:] = 1
        src_counts.index.names = dst_counts.index.names = ['name', 'type']

        counts = src_counts.add(dst_counts, fill_value=0).unstack(fill_value=0).astype('int64')

        count_df = counts.reset_index()
        count_df.columns = ['name'] + ['VertexActivityBy' + t for t in count_df.columns[1:]]

        # append missing columns
        missing_columns = [col for col in self.names if col not in count_df.columns]
        for col in missing_columns:
            count_df[col] = 0

        # add inactive vertices to result_df
        inactive_nodes = [name for name in self.nodes if name not in active_nodes]
        if len(inactive_nodes) > 0:
            inactive_nodes_df = pd.DataFrame(data={'name': inactive_nodes}, columns=['name'] + self.names).fillna(0)
            count_df = count_df.append(inactive_nodes_df).reset_index(drop=True)

        # add new occurred vertices to self.nodes
        new_nodes = [name for name in active_nodes if name not in self.nodes]
        for name in new_nodes:
            self.nodes.append(name)

        return count_df[['name'] + self.names]

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass
