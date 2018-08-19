from .feature import Feature


class VertexDegreeByType(Feature):
    """
    The feature VertexDegreeByType of a single vertex is defined as the number of incident edges of the current time
    window by edge type.

    All edge types should appear in the result data frame as columns, even if there are no occurrences of this edge type
    in the current time step.
    """

    def __init__(self, edge_types):
        self.names = ['VertexDegreeBy' + str(edge_type) for edge_type in edge_types]

    def reset(self):
        pass

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the vertex degree by type.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns 'name' and 'VertexDegreeByTYPE' for each existing type, and the calculated
            vertex degree for all vertices and edge types in the given df_edges.
        """

        src_counts = df_edges.groupby(['SRC_NAME', 'E_TYPE']).size()
        dst_counts = df_edges.groupby(['DST_NAME', 'E_TYPE']).size()

        src_counts.index.names = dst_counts.index.names = ['name', 'type']

        counts = src_counts.add(dst_counts, fill_value=0).unstack(fill_value=0).astype('int64')

        count_df = counts.reset_index()

        count_df.columns = ['name'] + ['VertexDegreeBy' + t for t in count_df.columns[1:]]

        # append missing columns
        missing_columns = [col for col in self.names if col not in count_df.columns]
        for col in missing_columns:
            count_df[col] = 0

        return count_df[['name'] + self.names]

    def compute(self, node_name, t):
        # Not needed here, since feature to simple for multi processoring
        pass
