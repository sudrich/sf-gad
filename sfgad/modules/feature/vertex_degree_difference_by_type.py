import pandas as pd

from .feature import Feature


class VertexDegreeDifferenceByType(Feature):
    """
    The feature VertexDegreeDifferenceByType of a single vertex is defined as the difference of the vertex degree
    between the current time step and the previous time step grouped by edge type.

    All edge types should appear in the result data frame as columns, even if there are no occurrences of this edge type
    in the current time step.
    """

    def __init__(self, edge_types, only_active_nodes=False):
        self.names = ['VertexDegreeDifferenceBy' + str(edge_type) for edge_type in edge_types]
        self.only_active_nodes = only_active_nodes

        self.previous_count_df = pd.DataFrame(columns=['name'] + self.names)

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the vertex degree difference by edge type.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns 'name' and 'VertexDegreeDifferenceByTYPE' for each existing edge type,
            and the calculated vertex degree difference for all vertices and edge types in the given df_edges.
        """

        src_counts = df_edges.groupby(['SRC_NAME', 'E_TYPE']).size()
        dst_counts = df_edges.groupby(['DST_NAME', 'E_TYPE']).size()

        src_counts.index.names = dst_counts.index.names = ['name', 'type']

        counts = src_counts.add(dst_counts, fill_value=0).unstack(fill_value=0).astype('int64')

        count_df = counts.reset_index()

        count_df.columns = ['name'] + ['VertexDegreeDifferenceBy' + t for t in count_df.columns[1:]]

        # append missing columns
        missing_columns = [col for col in self.names if col not in count_df.columns]
        for col in missing_columns:
            count_df[col] = 0

        # decide for which nodes the vertex degree difference should be calculated
        how = 'left' if self.only_active_nodes else 'outer'

        diff_df = pd.merge(count_df, self.previous_count_df, how=how, on=['name']).fillna(0)
        for col in self.names:
            diff_df[col] = (diff_df[col + '_x'] - diff_df[col + '_y']).astype('int64')

        # update previous_count_df for the next time step
        self.previous_count_df = count_df

        return diff_df[['name'] + self.names]

    def compute(self, node_name, t):
        # Not needed here, since feature to simple for multi processoring
        pass
