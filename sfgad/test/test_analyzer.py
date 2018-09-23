import datetime as dt
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from sfgad.analyzer import Analyzer
from sfgad.modules.features import VertexDegree
from sfgad.modules.observation_selection import HistoricAllSelection
from sfgad.modules.probability_combination import AvgProbability
from sfgad.modules.probability_estimation import EmpiricalEstimator
from sfgad.modules.weighting import ConstantWeight


class TestAnalyzer(TestCase):
    def setUp(self):
        self.dfs = [
            pd.DataFrame({'TIMESTAMP': [dt.datetime.fromordinal(1).replace(year=2017) + dt.timedelta(days=i)] * 3,
                          'SRC_NAME': ['A', 'A', 'B'],
                          'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                          'DST_NAME': ['B', 'C', 'C'],
                          'DST_TYPE': ['NODE', 'NODE', 'NODE']}) for i in range(50)]

        features_list = [VertexDegree()]
        observation_gatherer = HistoricAllSelection()
        weighting_function = ConstantWeight(weight=1)
        probability_estimator = EmpiricalEstimator()
        probability_combiner = AvgProbability()
        n_jobs = 1

        self.analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                                 probability_combiner, n_jobs=n_jobs)

    def test_init(self):
        self.assertEqual(self.analyzer.time_window, 0)

    def test_single_timestep(self):
        # process a time step
        result = self.analyzer.fit_transform(self.dfs[0])

        # test resetting of the dictionaries
        assert_frame_equal(result,
                           pd.DataFrame({'name': ['A', 'B', 'C'], 'time_window': [0] * 3, 'p_value': [np.nan] * 3},
                                        columns=['name', 'time_window', 'p_value']))

    def test_two_timesteps(self):
        # process a time step
        self.analyzer.fit_transform(self.dfs[0])
        result = self.analyzer.fit_transform(self.dfs[1])

        # test resetting of the dictionaries
        assert_frame_equal(result,
                           pd.DataFrame({'name': ['A', 'B', 'C'], 'time_window': [1] * 3, 'p_value': [1.0] * 3},
                                        columns=['name', 'time_window', 'p_value']))

    def test_all_timesteps(self):
        # process a time step
        for df in self.dfs[:-1]:
            self.analyzer.fit_transform(df)
        result = self.analyzer.fit_transform(self.dfs[-1])

        # test resetting of the dictionaries
        assert_frame_equal(result,
                           pd.DataFrame({'name': ['A', 'B', 'C'], 'time_window': [49] * 3, 'p_value': [1.0] * 3},
                                        columns=['name', 'time_window', 'p_value']))

    def test_result_df_shape(self):
        result = self.analyzer.fit_transform(self.dfs[0])

        self.assertEqual(result.shape, (3, 3))

    def test_result_df_columns(self):
        result = self.analyzer.fit_transform(self.dfs[0])

        self.assertEqual(result.columns.tolist(), ['name', 'time_window', 'p_value'])

    def test_result_df_dtypes(self):
        result = self.analyzer.fit_transform(self.dfs[0])

        self.assertEqual(result.dtypes.tolist(), [np.object_, np.int64, np.float64])
