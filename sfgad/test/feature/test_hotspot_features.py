from unittest import TestCase
import pandas as pd
import numpy as np

from sfgad.modules.feature.hotspot_features import HotSpotFeatures
from collections import deque
from pandas.util.testing import assert_frame_equal, assert_series_equal


class TestHotSpotFeatures(TestCase):

    def setUp(self):

        self.df_1 = pd.DataFrame({
            'TIMESTAMP': ['2018-01-01 00:00:01', '2018-01-01 00:00:04', '2018-01-01 00:00:10'],
            'SRC_NAME': ['A', 'B', 'A'],
            'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
            'DST_NAME': ['B', 'A', 'C'],
            'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df_1['TIMESTAMP'] = pd.to_datetime(self.df_1['TIMESTAMP'])

        self.df_2 = pd.DataFrame({
            'TIMESTAMP': ['2018-01-01 00:00:12', '2018-01-01 00:00:13', '2018-01-01 00:00:20'],
            'SRC_NAME': ['A', 'B', 'A'],
            'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
            'DST_NAME': ['B', 'A', 'C'],
            'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df_2['TIMESTAMP'] = pd.to_datetime(self.df_2['TIMESTAMP'])

        self.df_3 = pd.DataFrame({
            'TIMESTAMP': ['2018-01-01 00:00:21', '2018-01-01 00:00:23', '2018-01-01 00:00:23',
                          '2018-01-01 00:00:25', '2018-01-01 00:00:27', '2018-01-01 00:00:30'],
            'SRC_NAME': ['A', 'B', 'C', 'C', 'C', 'A'],
            'SRC_TYPE': ['NODE', 'NODE', 'NODE', 'NODE', 'NODE', 'NODE'],
            'DST_NAME': ['B', 'A', 'D', 'E', 'F', 'C'],
            'DST_TYPE': ['NODE', 'NODE', 'NODE', 'NODE', 'NODE', 'NODE']})
        self.df_3['TIMESTAMP'] = pd.to_datetime(self.df_3['TIMESTAMP'])

        self.half_life = 2
        self.window_size = 10
        self.decay_lambda = 1 / (self.half_life * self.window_size)

        self.hotspot_features = HotSpotFeatures(half_life=self.half_life, window_size=self.window_size)

        # the target output of the feature after 1. (2.) time step
        self.target_df_1 = pd.DataFrame(data={'name': ['A', 'B', 'C'], 'CorrelationChange': [np.nan, np.nan, np.nan],
                                              'MagnitudeChange': [np.nan, np.nan, np.nan]},
                                        columns=['name', 'CorrelationChange', 'MagnitudeChange'])
        self.target_df_2 = pd.DataFrame(data={'name': ['A', 'B', 'C'], 'CorrelationChange': [1.110223e-16, 0, 0],
                                              'MagnitudeChange': [-12.071068, -9.656854, -2.414214]},
                                        columns=['name', 'CorrelationChange', 'MagnitudeChange'])
        self.target_df_3 = pd.DataFrame(data={'name': ['A', 'B', 'C', 'D', 'E', 'F'],
                                              'CorrelationChange': [1.110223e-16, 0, 1.562279e-01,
                                                                    np.nan, np.nan, np.nan],
                                              'MagnitudeChange': [-27.892136, -22.313708, -8.010025,
                                                                  np.nan, np.nan, np.nan]},
                                        columns=['name', 'CorrelationChange', 'MagnitudeChange'])


    def test_init(self):
        self.assertEqual(self.hotspot_features.names, ["CorrelationChange", "MagnitudeChange"])
        self.assertEqual(self.hotspot_features.update_activity, True)
        self.assertEqual(self.hotspot_features.half_life, self.half_life)
        self.assertEqual(self.hotspot_features.decay_lambda, self.decay_lambda)
        self.assertEqual(self.hotspot_features.node_age, {})
        self.assertEqual(self.hotspot_features.prev_product_matrices, {})
        self.assertEqual(self.hotspot_features.activity_buffer, {})
        self.assertEqual(self.hotspot_features.prev_timestamps, deque(maxlen=self.half_life))

        self.assertEqual(self.hotspot_features.interpreter.decay_lambda, self.decay_lambda)
        self.assertEqual(self.hotspot_features.interpreter.half_life, self.half_life)

    def test_activity_update(self):
        #
        # check the activity update after the first time step
        #
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)

        # check node age update
        self.assertEqual(self.hotspot_features.node_age, {0: 1, 1: 1, 2: 1})

        # check product matrices update
        prev_product_matrices = {
            0: (pd.Timestamp('2018-01-01 00:00:10'), np.array([[4., 2.], [2., 1.]])),
            1: (pd.Timestamp('2018-01-01 00:00:10'), np.array([[4.]])),
            2: (pd.Timestamp('2018-01-01 00:00:10'), np.array([[1.]]))
        }
        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.prev_product_matrices[node_id][0], prev_product_matrices[node_id][0])
            np.testing.assert_array_equal(self.hotspot_features.prev_product_matrices[node_id][1],
                                          prev_product_matrices[node_id][1])

        # check activity update
        activity_buffer = {
            0: deque([([-0.89442719099991586, -0.44721359549995793], 5.0)]),
            1: deque([([1.0], 4.0)]),
            2: deque([([1.0], 1.0)])
        }
        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.activity_buffer[node_id], activity_buffer[node_id])

        # check timestamp update
        self.assertEqual(self.hotspot_features.prev_timestamps, deque([pd.Timestamp('2018-01-01 00:00:10')]))

    def test_activity_update_2(self):
        #
        # check the activity update after the third time step
        #
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_3, n_jobs=1)

        # check node age update
        self.assertEqual(self.hotspot_features.node_age, {0: 3, 1: 3, 2: 3, 3: 1, 4: 1, 5: 1})

        # check product matrices update
        prev_product_matrices = {
            0: (pd.Timestamp('2018-01-01 00:00:30'), np.array([[26.3137085, 13.15685425], [13.15685425, 6.57842712]])),
            1: (pd.Timestamp('2018-01-01 00:00:30'), np.array([[26.3137085]])),
            2: (pd.Timestamp('2018-01-01 00:00:30'), np.array(
                [[6.57842712, 2.20710678, 2.20710678, 2.20710678],
                 [2.20710678, 1., 1., 1.],
                 [2.20710678, 1., 1., 1.],
                 [2.20710678, 1., 1., 1.]])),
            3: (pd.Timestamp('2018-01-01 00:00:30'), np.array([[1.]])),
            4: (pd.Timestamp('2018-01-01 00:00:30'), np.array([[1.]])),
            5: (pd.Timestamp('2018-01-01 00:00:30'), np.array([[1.]]))
        }
        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.prev_product_matrices[node_id][0],
                             prev_product_matrices[node_id][0])
            np.testing.assert_allclose(self.hotspot_features.prev_product_matrices[node_id][1],
                                       prev_product_matrices[node_id][1], rtol=1e-5, atol=0)

        # check activity update
        activity_buffer = {
            0: deque([([-0.89442719099991586, -0.44721359549995793], 17.071067811865476),
                      ([-0.89442719099991586, -0.44721359549995793], 32.892135623730951)]),
            1: deque([([1.0], 13.65685424949238),
                      ([1.0], 26.313708498984759)]),
            2: deque([([1.0], 3.4142135623730949),
                      ([-0.84377212042813954, -0.30986481395828158, -0.30986481395828164, -0.30986481395828164],
                       9.0100246458567845)]),
            3: deque([([1.0], 1.0)]),
            4: deque([([1.0], 1.0)]),
            5: deque([([1.0], 1.0)])
        }

        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.activity_buffer[node_id], activity_buffer[node_id])

        # check timestamp update
        self.assertEqual(self.hotspot_features.prev_timestamps, deque([pd.Timestamp('2018-01-01 00:00:20'),
                                                                       pd.Timestamp('2018-01-01 00:00:30')]))

    def test_activity_update_3(self):
        #
        # check the activity update, when no activity should be recorded (after the second time step)
        #
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1, update_activity=False)
        self.hotspot_features.process_vertices(df_edges=self.df_3, n_jobs=1, update_activity=False)

        # check node age update
        self.assertEqual(self.hotspot_features.node_age, {0: 1, 1: 1, 2: 1})

        # check product matrices update
        prev_product_matrices = {
            0: (pd.Timestamp('2018-01-01 00:00:10'), np.array([[4., 2.], [2., 1.]])),
            1: (pd.Timestamp('2018-01-01 00:00:10'), np.array([[4.]])),
            2: (pd.Timestamp('2018-01-01 00:00:10'), np.array([[1.]]))
        }
        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.prev_product_matrices[node_id][0], prev_product_matrices[node_id][0])
            np.testing.assert_array_equal(self.hotspot_features.prev_product_matrices[node_id][1],
                                          prev_product_matrices[node_id][1])

        # check activity update
        activity_buffer = {
            0: deque([([-0.89442719099991586, -0.44721359549995793], 5.0)]),
            1: deque([([1.0], 4.0)]),
            2: deque([([1.0], 1.0)])
        }
        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.activity_buffer[node_id], activity_buffer[node_id])

        # check timestamp update
        self.assertEqual(self.hotspot_features.prev_timestamps, deque([pd.Timestamp('2018-01-01 00:00:10')]))

    def test_activity_update_4(self):
        #
        # check the activity update, when there are inactive nodes (C)
        #
        df_2 = pd.DataFrame({
            'TIMESTAMP': ['2018-01-01 00:00:12', '2018-01-01 00:00:13'],
            'SRC_NAME': ['A', 'B'],
            'SRC_TYPE': ['NODE', 'NODE'],
            'DST_NAME': ['B', 'A'],
            'DST_TYPE': ['NODE', 'NODE']})
        df_2['TIMESTAMP'] = pd.to_datetime(df_2['TIMESTAMP'])

        # process the time steps
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=df_2, n_jobs=1)

        # check node age update
        self.assertEqual(self.hotspot_features.node_age, {0: 2, 1: 2, 2: 2})

        # check product matrices update
        prev_product_matrices = {
            0: (pd.Timestamp('2018-01-01 00:00:13'), np.array([[17.70802287, 5.05151051], [5.05151051, 1.62450479]])),
            1: (pd.Timestamp('2018-01-01 00:00:13'), np.array([[17.70802287]])),
            2: (pd.Timestamp('2018-01-01 00:00:10'), np.array([[1.]]))
        }
        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.prev_product_matrices[node_id][0], prev_product_matrices[node_id][0])
            np.testing.assert_allclose(self.hotspot_features.prev_product_matrices[node_id][1],
                                       prev_product_matrices[node_id][1], rtol=1e-5, atol=0)

        # check activity update
        activity_buffer = {
            0: deque([([-0.89442719099991586, -0.44721359549995793], 5.0),
                      ([-0.96093535787116113, -0.27677289967231888], 19.162981522497621)]),
            1: deque([([1.0], 4.0),
                      ([1.0], 17.708022871736524)]),
            2: deque([([1.0], 1.0),
                      ([1.0], 0.81225239635623547)])
        }
        for node_id in self.hotspot_features.interpreter.inv_ids:
            self.assertEqual(self.hotspot_features.activity_buffer[node_id], activity_buffer[node_id])

        # check timestamp update
        self.assertEqual(self.hotspot_features.prev_timestamps, deque([pd.Timestamp('2018-01-01 00:00:10'),
                                                                       pd.Timestamp('2018-01-01 00:00:13')]))

    def test_result_df_shape(self):
        result_df_1 = self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        result_df_2 = self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)
        result_df_3 = self.hotspot_features.process_vertices(df_edges=self.df_3, n_jobs=1)

        self.assertEqual(result_df_1.shape, self.target_df_1.shape)
        self.assertEqual(result_df_2.shape, self.target_df_2.shape)
        self.assertEqual(result_df_3.shape, self.target_df_3.shape)

    def test_result_df_columns(self):
        result_df_1 = self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)

        self.assertEqual(result_df_1.columns.tolist(), ['name', 'CorrelationChange', 'MagnitudeChange'])

    def test_result_df_dtypes(self):
        result_df_1 = self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        result_df_2 = self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)
        result_df_3 = self.hotspot_features.process_vertices(df_edges=self.df_3, n_jobs=1)

        assert_series_equal(result_df_1.dtypes, self.target_df_1.dtypes)
        assert_series_equal(result_df_2.dtypes, self.target_df_2.dtypes)
        assert_series_equal(result_df_3.dtypes, self.target_df_3.dtypes)

    def test_result_df_values(self):
        result_df_1 = self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        result_df_2 = self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)
        result_df_3 = self.hotspot_features.process_vertices(df_edges=self.df_3, n_jobs=1)

        assert_series_equal(result_df_1['name'], self.target_df_1['name'])
        assert_series_equal(result_df_2['name'], self.target_df_2['name'])
        assert_series_equal(result_df_3['name'], self.target_df_3['name'])
        assert_series_equal(result_df_1['CorrelationChange'], self.target_df_1['CorrelationChange'])
        assert_series_equal(result_df_2['CorrelationChange'], self.target_df_2['CorrelationChange'])
        assert_series_equal(result_df_3['CorrelationChange'], self.target_df_3['CorrelationChange'])
        assert_series_equal(result_df_1['MagnitudeChange'], self.target_df_1['MagnitudeChange'])
        assert_series_equal(result_df_2['MagnitudeChange'], self.target_df_2['MagnitudeChange'])
        assert_series_equal(result_df_3['MagnitudeChange'], self.target_df_3['MagnitudeChange'])

    def test_process_vertices(self):
        #
        # test_overall_processing_all_nodes
        #
        # test the calculation of hotspot features in the 1. time step ('df_1')
        assert_frame_equal(self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1), self.target_df_1)

        # test the calculation of hotspot features in the 2. time step ('df_2')
        assert_frame_equal(self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1), self.target_df_2)

        # test the calculation of hotspot features in the 3. time step ('df_3')
        assert_frame_equal(self.hotspot_features.process_vertices(df_edges=self.df_3, n_jobs=1), self.target_df_3)

    def test_process_vertices_wrong_input(self):
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)

        # assertion after getting old data
        self.assertRaises(AssertionError, self.hotspot_features.process_vertices, self.df_1, 1)

    def test_compute(self):
        # test compute method on the node 'A' in the 3. time step
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)

        # process 3. time step
        self.hotspot_features.interpreter.interpret(self.df_3)

        product_matrix, cor, mag, cor_change, mag_change = self.hotspot_features.compute(
            'A', pd.Timestamp('2018-01-01 00:00:30'))

        # check product matrix
        np.testing.assert_allclose(product_matrix, np.array([[26.3137085, 13.15685425], [13.15685425, 6.57842712]]),
                                   rtol=1e-5, atol=0)
        # check correlation
        self.assertEqual(cor, [-0.89442719099991586, -0.44721359549995793])

        # check magnitude
        self.assertEqual(mag, 32.892135623730951)

        # check correlation change
        self.assertEqual(cor_change, 1.1102230246251565e-16)

        # check magnitude change
        self.assertEqual(mag_change, -27.892135623730951)

    def test_calculate_product_matrix(self):
        # test compute method on the node 'A' in the 3. time step
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)

        # process 3. time step
        self.hotspot_features.interpreter.interpret(self.df_3)

        product_matrix = self.hotspot_features.calculate_product_matrix(0, pd.Timestamp('2018-01-01 00:00:30'))

        # check product matrix
        np.testing.assert_allclose(product_matrix, np.array([[26.3137085, 13.15685425], [13.15685425, 6.57842712]]),
                                   rtol=1e-5, atol=0)

    def test_consider_prev_product_matrix(self):
        # test the consideration of product matrix of 'A' in the 3. time step
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)

        # process 3. time step
        self.hotspot_features.interpreter.interpret(self.df_3)

        updated_matrix = self.hotspot_features.consider_prev_product_matrix(0, pd.Timestamp('2018-01-01 00:00:30'), 2)

        np.testing.assert_allclose(updated_matrix, np.array([[6.82842712, 3.41421356], [3.41421356, 1.70710678]]),
                                   rtol=1e-5, atol=0)

    def test_reconstruct_edges(self):
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)

        # reconstruct edges of 'A' after 2. time step
        self.assertEqual(self.hotspot_features.reconstruct_edges(0), [(0, 1), (0, 2)])

    def test_cal_matrix_features(self):
        # test computation of correlation and magnitude of 'A' in the 3. time step
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)

        # process 3. time step
        self.hotspot_features.interpreter.interpret(self.df_3)

        product_matrix = self.hotspot_features.calculate_product_matrix(0, pd.Timestamp('2018-01-01 00:00:30'))

        cor, mag = self.hotspot_features.cal_matrix_features(product_matrix)

        # check correlation
        self.assertEqual(cor, [-0.89442719099991586, -0.44721359549995793])

        # check magnitude
        self.assertEqual(mag, 32.892135623730951)

    def test_cal_activity_changes(self):
        # test computation of correlation-change and magnitude-change of 'A' in the 3. time step
        self.hotspot_features.process_vertices(df_edges=self.df_1, n_jobs=1)
        self.hotspot_features.process_vertices(df_edges=self.df_2, n_jobs=1)

        # process 3. time step
        self.hotspot_features.interpreter.interpret(self.df_3)

        product_matrix = self.hotspot_features.calculate_product_matrix(0, pd.Timestamp('2018-01-01 00:00:30'))

        cor, mag = self.hotspot_features.cal_matrix_features(product_matrix)

        cor_change, mag_change = self.hotspot_features.cal_activity_changes(0, cor, mag)

        # check correlation change
        self.assertEqual(cor_change, 1.1102230246251565e-16)

        # check magnitude change
        self.assertEqual(mag_change, -27.892135623730951)
