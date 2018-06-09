import unittest
import pandas as pd
import copy

from collections import deque
from sfgad.modules.feature.helper.hotspot_interpreter import HotSpotInterpreter


class TestHotSpotInterpreter(unittest.TestCase):

    def setUp(self):

        self.df_1 = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:00', '2018-01-01 00:00:01', '2018-01-01 00:00:05'],
                             'SRC_NAME': ['A', 'A', 'B'],
                             'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                             'DST_NAME': ['B', 'C', 'C'],
                             'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df_1['TIMESTAMP'] = pd.to_datetime(self.df_1['TIMESTAMP'])

        self.df_2 = pd.DataFrame({'TIMESTAMP': ['2018-01-01 00:00:10', '2018-01-01 00:00:12', '2018-01-01 00:00:14'],
                             'SRC_NAME': ['A', 'A', 'C'],
                             'SRC_TYPE': ['NODE', 'NODE', 'NODE'],
                             'DST_NAME': ['B', 'C', 'D'],
                             'DST_TYPE': ['NODE', 'NODE', 'NODE']})
        self.df_2['TIMESTAMP'] = pd.to_datetime(self.df_2['TIMESTAMP'])

        self.window_size = 10
        self.half_life = 2
        self.decay_lambda = 1 / (self.half_life * self.window_size)

        self.interpreter = HotSpotInterpreter(self.decay_lambda, self.half_life)

    def test_init(self):
        self.assertEqual(self.interpreter.decay_lambda, self.decay_lambda)
        self.assertEqual(self.interpreter.half_life, self.half_life)
        self.assertEqual(len(self.interpreter.fit_buffer), 1)
        self.assertEqual(self.interpreter.ids, {})
        self.assertEqual(self.interpreter.inv_ids, {})
        self.assertEqual(self.interpreter.frequencies, {})
        self.assertEqual(self.interpreter.edge_updated, {})
        self.assertEqual(self.interpreter.neighbors, {})
        self.assertEqual(self.interpreter.fit_buffer[-1], (pd.Timestamp.min, 0, {}, {}))

    def test_interpret(self):

        self.assertEqual(self.interpreter.interpret(self.df_1), (pd.Timestamp('2018-01-01 00:00:05'), ['A', 'B', 'C']))

        fit_buffer = deque([(pd.Timestamp('1677-09-21 00:12:43.145225'), 0, {}, {}),
               (pd.Timestamp('2018-01-01 00:00:05'),
                3,
                {(0, 1): (pd.Timestamp('2018-01-01 00:00:05'), 1),
                 (0, 2): (pd.Timestamp('2018-01-01 00:00:05'), 1),
                 (1, 2): (pd.Timestamp('2018-01-01 00:00:05'), 1)},
                {0: [1, 2], 1: [0, 2], 2: [0, 1]})])

        self.assertEqual(self.interpreter.fit_buffer, fit_buffer)

        self.assertEqual(self.interpreter.edge_updated, {})

        # verify the assertion
        self.assertRaises(AssertionError, self.interpreter.interpret, self.df_1)

    def test_interpret_edge(self):
        # 1. occurrence in a time window
        self.interpreter.register_node("A", 0)
        self.interpreter.register_node("B", 1)
        self.interpreter.interpret_edge("B", "A", pd.Timestamp('2018-01-01 00:00:00'))

        self.assertEqual(self.interpreter.get_neighbors(0), [1])
        self.assertEqual(self.interpreter.get_neighbors(1), [0])

        self.assertEqual(self.interpreter.frequencies[(0, 1)], (pd.Timestamp('2018-01-01 00:00:00'), 1))

        # 2. occurrence in a time window
        self.interpreter.interpret_edge("A", "B", pd.Timestamp('2018-01-01 00:00:00'))
        self.assertEqual(self.interpreter.frequencies[(0, 1)], (pd.Timestamp('2018-01-01 00:00:00'), 2))

    def test_update_weighted_frequency(self):
        # a new edge
        self.assertEqual(self.interpreter.update_weighted_frequency((0, 1), pd.Timestamp('2018-01-01 00:00:00')), 1)

        # a new arrival of an edge
        self.interpreter.frequencies[(0, 1)] = (pd.Timestamp('2018-01-01 00:00:00'), 1)
        self.assertEqual(self.interpreter.update_weighted_frequency((0, 1), pd.Timestamp('2018-01-01 00:00:05')),
                         1 * 2 ** (-self.decay_lambda * 5) + 1)

    def register_node(self):
        self.assertEqual(self.interpreter.register_node("A", 3), 4)
        self.assertEqual(self.interpreter.ids["A"], 3)
        self.assertEqual(self.interpreter.inv_ids[3], "A")

    def test_update_neighbor(self):
        self.interpreter.neighbors[0] = [1, 2]

        # try to add an existing neighbor
        self.interpreter.update_neighbor(0, 1)
        self.assertEqual(self.interpreter.get_neighbors(0), [1, 2])

        # add a new neighbor
        self.interpreter.update_neighbor(0, 3)
        self.assertEqual(self.interpreter.get_neighbors(0), [1, 2, 3])

    def test_turn_back_time(self):

        # let the interpreter intepret the 1. dataframe and make a copy
        self.interpreter.interpret(self.df_1)
        interpreter_copy = copy.deepcopy(self.interpreter)

        # let the interpreter intepret the 2. dataframe and turn back time
        self.interpreter.interpret(self.df_2)
        self.interpreter.turn_back_time()

        # assert that the interpreter returned to the state after the 1. interpret call
        self.assertEqual(self.interpreter.ids, interpreter_copy.ids)
        self.assertEqual(self.interpreter.inv_ids, interpreter_copy.inv_ids)
        self.assertEqual(self.interpreter.frequencies, interpreter_copy.frequencies)
        self.assertEqual(self.interpreter.neighbors, interpreter_copy.neighbors)
        self.assertEqual(self.interpreter.fit_buffer[-1], interpreter_copy.fit_buffer[-1])

    def test_get_freq(self):
        # edge (0, 1) is unknown
        self.assertRaises(ValueError, self.interpreter.get_freq, (0, 1))

        # edge (0, 1) was added recently
        self.interpreter.frequencies[(0, 1)] = (pd.Timestamp.min, 1.23)
        self.assertEqual(self.interpreter.get_freq((0, 1)), 1.23)

    def test_get_node_id(self):
        # node with the name "A" is unknown
        self.assertRaises(ValueError, self.interpreter.get_node_id, "A")

        # node with the name "A" was added recently and has the id 0
        self.interpreter.register_node("A", 0)
        self.assertEqual(self.interpreter.get_node_id("A"), 0)

    def test_get_current_time(self):
        # interpreter is empty
        self.assertEqual(self.interpreter.get_current_time(), pd.Timestamp.min)

        # interpreter had 1 interpret() call
        self.interpreter.interpret(self.df_1)
        self.assertEqual(self.interpreter.get_current_time(), pd.Timestamp('2018-01-01 00:00:05'))

    def test_get_n_neighbors(self):
        # node 1 has no neighbors
        self.assertRaises(ValueError, self.interpreter.get_n_neighbors, 1)

        # node 1 has 1 neighbor
        self.interpreter.neighbors[1].append(2)
        self.assertEqual(self.interpreter.get_n_neighbors(1), 1)

    def test_get_neighbors(self):
        # node 1 has no neighbors
        self.assertRaises(ValueError, self.interpreter.get_n_neighbors, 1)

        # node 1 has 1 neighbor
        self.interpreter.neighbors[1].append(2)
        self.assertEqual(self.interpreter.get_neighbors(1), [2])


if __name__ == '__main__':
    unittest.main()