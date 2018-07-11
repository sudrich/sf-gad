import numpy as np
import pandas as pd
import datetime


class GraphGenerator:

    def __init__(self, n_vertices, p_e=1, alpha_e=1, loops=False, seed=None, vertex_degree=None):
        """
        Data generator for homogeneous graphs
        :param n_vertices: number of vertices
        :param p_e: probability of occurrence of an edge
        :param alpha_e: probability of adding a new edge. Using this probability and p_e the probability beta_e is computed
        according to the relation p = alpha / (alpha + beta), beta_e is then the probability of deleting an existing edge
        in order for the probabilities to be consistent the relation alpha <= p / (1 - p) must hold.
        P.S: this relation is guaranteed to hold for every probability alpha if p was chosen such that p >= 0.5
        :param loops: indicates if loops are allowed, defaults to False
        :param seed: the random seed, if not specified see numpy.random
        :param vertex_degree: if this parameter is not None alpha_e is implicitly computed from the vertex_degree
        """
        if not isinstance(n_vertices, int) or n_vertices <= 0:
            raise ValueError("Error! The given argument 'n_vertices' should be a positive integer !")
        if not (0.0 <= p_e <= 1.0):
            raise ValueError("Error! The given argument 'p_e' should be a number between 0 and 1 !")
        if not (0.0 <= alpha_e <= 1.0):
            raise ValueError("Error! The given argument 'alpha_e' should be a number between 0 and 1 !")
        if not isinstance(loops, bool):
            raise ValueError("Error! The given argument 'loops' should be a boolean !")

        self.random = np.random
        self.random.seed(seed)
        self.n_vertices = n_vertices
        s = -1
        if loops:
            s = 1
        if vertex_degree is not None:
            assert (0 < vertex_degree <= n_vertices)
            p_e = vertex_degree / (n_vertices + s)
        self.p_e = p_e
        if p_e <= 0.5:
            if not alpha_e <= (p_e / (1.0 - p_e)):
                raise ValueError("Error! The given argument 'alpha_e' should satisfy: alpha_e <= (p_e / (1.0 - p_e)) !")
        self.alpha_e = alpha_e
        self.beta_e = None
        self.graph = None
        self.set_betas()
        self.loops = loops
        self.time = 0

    def set_betas(self):
        """
        helper method to compute beta_e from p_e and alpha_e
        :return:
        """
        r_e = (1.0 - self.p_e) / self.p_e
        self.beta_e = self.alpha_e * r_e

    def generate_graph(self):
        """
        generates the initial graph and stores it in the matrix self.graph
        :return:
        """
        x = self.generate_random_matrix()
        edges = x <= self.p_e
        self.graph = edges

    def generate_random_matrix(self):
        """
        generates a random matrix that will be used in the evolution of the graph
        :return:
        """
        x = self.random.random((self.n_vertices, self.n_vertices))
        x = np.tril(x)
        x = x + x.T - np.diag(x.diagonal())
        if not self.loops:
            np.fill_diagonal(x, 1)
        return x

    def evolution(self):
        """
        evolves the graph by one step
        :return:
        """
        assert self.graph is not None
        y = np.where(self.graph, 1.0 - self.beta_e, self.alpha_e)
        x = self.generate_random_matrix()
        self.graph = x <= y
        self.time += 1

    def next_graph(self):
        """
        Makes a one step evolution and returns the graph as a data frame
        :return: a data frame representation of the graph
        """
        if self.graph is None:
            self.generate_graph()
            return self.graph_to_data_frame()
        self.evolution()
        return self.graph_to_data_frame()

    def generate_anomalous_graph(self, n, p_a, only_vertices=False):
        """
        generates an anomalous graph based on the given parameters
        :param n: number of anomalous vertices
        :param p_a: the new anomalous probability of the edges occurrence
        :param only_vertices: if true then the only effected edges by the anomaly are
        edges between the selected n anomalous nodes. If false then then the edges affected by the anomaly are
        all edges including a selected anomalous node
        :return: a list of the anomalous vertices and a data frame representation of the anomalous graph
        """
        if not isinstance(n, int) or not (0 <= n <= self.n_vertices):
            raise ValueError("Error! The given argument 'n' should be an integer between 0 and self.n_vertices !")
        if not  (0 <= p_a <= 1.0):
            raise ValueError("Error! The given argument 'p_a' should be a number between 0 and 1 !")
        self.evolution()
        #df1 = self.graph_to_data_frame()
        vertices = self.random.choice(self.n_vertices, n, replace=False)
        if only_vertices:
            size = n
        else:
            size = self.n_vertices
        x = self.random.random((n, size))
        x = x <= p_a
        for i in range(n):
            for j in range(size):
                if only_vertices:
                    k = vertices[j]
                else:
                    k = j
                if k != vertices[i] or self.loops:
                    self.graph[vertices[i]][k] = x[i][j]
                    self.graph[k][vertices[i]] = x[i][j]
        df2 = self.graph_to_data_frame()
        return vertices, df2

    def graph_to_data_frame(self):
        """
        convert the self.graph matrix into data frame representation of the graph
        :return: a data frame representation of the graph
        """
        if self.graph is None:
            raise ValueError("Error! the graph hasn't been initialized yet !")
        df = np.asarray([[str((i, j)), 'TYPE', datetime.datetime.fromordinal(1).replace(year=2017) + datetime.timedelta(days=self.time), str(i), 'U', str(j)] for i in range(self.n_vertices) for j in range(i, self.n_vertices) if self.graph[i][j]])
        names = df[:,0]
        src_names = df[:,3]
        dst_names = df[:,5]
        timestamp = df[:,2]
        e_type = df[:,1]
        src_type = df[:,4]
        dst_type = df[:,4]
        d = {'TIMESTAMP': timestamp, 'E_NAME': names, 'E_TYPE': e_type, 'SRC_NAME': src_names, 'SRC_TYPE': src_type, 'DST_NAME': dst_names, 'DST_TYPE': dst_type}
        edges_df = pd.DataFrame(data=d)
        return edges_df

    def generate_data(self, steps_number, anomaly_list, n, p_a):
        """
        this method generates graphs and inserts anomalies in some of them
        :param steps_number: the number of generated graphs
        :param anomaly_list: a list indicating the steps at which an anomaly should occur
        :param n: number of anomalous vertices
        :param p_a: the new anomalous probability of occurrence of an edge
        :return: a list of data frames representing the generated graphs
        """
        if not isinstance(steps_number, int) or steps_number <= 0:
            raise ValueError("Error! The given argument 'steps_number' should be a positive integer !")
        dfs = []
        for i in range(steps_number):
            if i not in anomaly_list:
                dfs.append(self.next_graph())
            else:
                dfs.append(self.generate_anomalous_graph(n, p_a, True))
        return dfs

random = np.random
random.seed(1)
x = random.random((5,5))
x = np.tril(x)
x = x + x.T - np.diag(x.diagonal())
np.fill_diagonal(x, 1)
print(x)
print(np.array_equal([[1,2],[3,4]], [[1,2],[3,4]]))