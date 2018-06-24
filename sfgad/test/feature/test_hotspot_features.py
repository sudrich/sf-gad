from unittest import TestCase
import pandas as pd
import numpy as np

from sfgad.modules.feature.hotspot_features import HotSpotFeatures
from collections import deque

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


    def test_process_vertices(self):
        self.fail()

    def test_compute_multiple_nodes(self):
        self.fail()

    def test_compute(self):
        self.fail()

    def test_calculate_product_matrix(self):
        self.fail()

    def test_consider_prev_product_matrix(self):
        self.fail()

    def test_reconstruct_edges(self):
        self.fail()

    def test_cal_matrix_features(self):
        self.fail()

    def test_cal_activity_changes(self):
        self.fail()
