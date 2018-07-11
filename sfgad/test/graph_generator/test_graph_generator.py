import numpy as np

from unittest import TestCase

from sfgad.modules.graph_generator.graph_generator import GraphGenerator


class TestGraphGenerator(TestCase):

    def setUp(self):
        self.n_vertices = 10

    def test_wrong_input_n_vertices(self):
        self.assertRaises(ValueError, GraphGenerator, "alice")

    def test_wrong_input_p_e(self):
        self.assertRaises(ValueError, GraphGenerator, 5, p_e=2);

    def test_wrong_input_loops(self):
        self.assertRaises(ValueError, GraphGenerator, 5, loops="true")

    def test_wrong_input_alpha_e(self):
        self.assertRaises(ValueError, GraphGenerator, 5, alpha_e=-1)

    def test_generate_random_matrix(self):
        graph_generator = GraphGenerator(5, p_e=0.9, seed=1)
        x = graph_generator.generate_random_matrix()
        y = [[1.,          0.09233859,  0.41919451,  0.67046751,  0.80074457],
            [0.09233859,  1.,          0.6852195,   0.4173048,   0.96826158],
            [0.41919451,  0.6852195,   1.,          0.55868983,  0.31342418],
            [0.67046751,  0.4173048,   0.55868983,  1.,          0.69232262],
            [0.80074457,  0.96826158,  0.31342418,  0.69232262,  1.]]
        xt = np.asarray(y)
        self.assertEqual(xt.shape, x.shape)
        self.assertTrue(np.allclose(xt, x, 0.000001))

    def test_generate_graph(self):
        graph_generator = GraphGenerator(5, p_e=0.9, seed=1)
        graph_generator.generate_graph()
        graph = graph_generator.graph
        y = [[1.,          0.09233859,  0.41919451,  0.67046751,  0.80074457],
            [0.09233859,  1.,          0.6852195,   0.4173048,   0.96826158],
            [0.41919451,  0.6852195,   1.,          0.55868983,  0.31342418],
            [0.67046751,  0.4173048,   0.55868983,  1.,          0.69232262],
            [0.80074457,  0.96826158,  0.31342418,  0.69232262,  1.]]
        graph_t = np.asarray(y) <= 0.9
        self.assertEqual(graph_t.shape, graph.shape)
        self.assertTrue(np.array_equal(graph_t, graph))