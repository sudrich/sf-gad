from .feature import Feature
import pandas as pd

class VertexDegreeDifference(Feature):
    """
    The feature VertexDegreeDifference of a single vertex is defined as the difference of the vertex degree between the
    current time step and the previous time step.

    It calculates the vertex degree difference only for the active nodes.
    """

    def __init__(self):
        self.names = ['VertexDegreeDifference']
        self.previous_count_df = pd.DataFrame(columns=['name', 'VertexDegree'])

    def process_vertices(self, df_edges, n_jobs, update_activity=True):
        """
        Iterates over the current data frame and calculates for each active vertex the vertex degree difference.
        :param df_edges: The data frame to process.
        :param n_jobs: The number of cores that are supported for multiprocessing.
        :param update_activity: True, if the feature should consider the new edges for future computations (if needed),
            false otherwise.
        :return a data frame with the columns ['name', 'VertexDegreeDifference'] and the calculated vertex degree
            difference for all vertices in the given df_edges.
        """

        src_counts = df_edges.SRC_NAME.value_counts()
        dst_counts = df_edges.DST_NAME.value_counts()

        counts = src_counts.add(dst_counts, fill_value=0).astype('int64')

        count_df = counts.to_frame().reset_index()
        count_df.columns = ['name', 'VertexDegree']

        diff_df = pd.merge(count_df, self.previous_count_df, how='left', on=['name']).fillna(0)
        diff_df['VertexDegreeDifference'] = (diff_df['VertexDegree_x'] - diff_df['VertexDegree_y']).astype('int64')

        # update previous_count_df for the next time step
        self.previous_count_df = count_df

        return diff_df[['name', 'VertexDegreeDifference']]

    def compute(self, node_name, t):
        # Not needed here, since feature to simple for multi processoring
        pass