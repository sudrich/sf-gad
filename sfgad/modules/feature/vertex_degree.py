from .feature import Feature


class VertexDegree(Feature):
    """
    The feature VertexDegree of a single vertex is defined as the number of occurrences as either SRC or DST node in all
    the edges of the current time window
    """

    def __init__(self):
        self.names = ['VertexDegree']

    def reset(self):
        pass

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each vertex the vertex degree.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns ['name', 'VertexDegree'] and the calculated vertex degree for all vertices
            in the given df_edges.
        """

        src_counts = df_edges.SRC_NAME.value_counts()
        dst_counts = df_edges.DST_NAME.value_counts()

        counts = src_counts.add(dst_counts, fill_value=0).astype('int64')

        count_df = counts.to_frame().reset_index()
        count_df.columns = ['name'] + self.names

        return count_df

    def compute(self, node_name, t):
        # Not needed here, since this feature is to simple for multiprocessing
        pass
