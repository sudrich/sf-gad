from math import log

import networkx as nx
import pandas as pd

COL_NAME = 'name'
COL_TYPE = 'type'
COL_P = 'p_value'

COL_SRC_NAME = 'SRC_NAME'
COL_SRC_TYPE = 'SRC_TYPE'
COL_DST_NAME = 'DST_NAME'
COL_DST_TYPE = 'DST_TYPE'
COL_E_NAME = 'E_NAME'
COL_E_TYPE = 'DST_TYPE'


def scan(df_edges, df_p, alpha_max, K, Z=None, C=None):
    """
    Extracts the most anomalous subgraph based on the Non-Parametric Heterogeneous Graph Scan (NPHGS).
    :param df_edges: DataFrame of edges in the graph.
    :param df_p: DataFrame of vertices and their p-values.
    :param alpha_max: the significance threshold.
    :param K: the number of seed vertices considered per type.
    :param Z: the number of iterations per seed vertex.
    :param C: the vertex types present in the graph.
    :return: tuples (subgraph, score) of the most anomalous subgraph with its score.
    """
    p_graph = create_p_graph(df_edges, df_p)
    if p_graph.number_of_nodes() == 0:
        return set(), 0

    if Z is None:
        Z = int(log(p_graph.number_of_nodes(), 2))
    if C is None:
        C = set(nx.get_node_attributes(p_graph, name='type').values())

    vertices = sort_by_ascending_p(p_graph.nodes(data=True))

    subgraphs = []

    for c in C:
        typed_vertices = [v for v in vertices if v[1]["type"] == c]
        for k in range(min(K, len(typed_vertices))):
            if typed_vertices[k][1]['p'] <= alpha_max:
                subgraphs.append(grow_max_subgraph(p_graph, typed_vertices[k], vertices, alpha_max, Z))

    best_subgraph, best_score = max(subgraphs, key=lambda s: s[1], default=([], 0))

    return [v[0] for v in best_subgraph], best_score


def grow_max_subgraph(p_graph, seed, vertices, alpha_max, Z):
    s = [seed]
    score = 0

    for z in range(Z):
        g = get_neighbors(p_graph, vertices, s)
        b, score = relaxed_problem(alpha_max, s, g)

        if len({v[0] for v in b} - {v[0] for v in s}) != 0:
            s = b
        else:
            return s, score

    return s, score


def kl_divergence(a, b):
    if b == 0 or b >= a:
        return 0
    elif a == 0:
        return log(1 / (1 - b), 2)
    elif a == 1:
        return log(1 / b, 2)
    else:
        return a * log(a / b, 2) + (1 - a) * log((1 - a) / (1 - b), 2)


def bj_statistic(alpha, n_alpha, n):
    if n == 0:
        return 0
    else:
        return n * kl_divergence(n_alpha / n, alpha)


def sort_by_ascending_p(vertices):
    return sorted(vertices, key=lambda v: v[1]["p"], reverse=False)


def get_neighbors(p_graph, vertices, seeds):
    seed_indices = {v[0] for v in seeds}
    return [v for v in vertices if v[0] not in seed_indices and len(set(p_graph.neighbors(v[0])) & seed_indices) != 0]


def relaxed_problem(alpha_max, vertices, neighbors):
    best_subgraph = []
    best_score = 0

    i = 0
    j = 0
    subgraph = []

    while i < len(vertices):
        if j == len(neighbors) or neighbors[j][1]['p'] >= alpha_max or vertices[i][1]['p'] < neighbors[j][1]['p']:
            p = vertices[i][1]['p']
            subgraph.append(vertices[i])
            i += 1
        else:
            p = neighbors[j][1]['p']
            subgraph.append(neighbors[j])
            j += 1

        score = bj_statistic(p, j + i, j + len(vertices))

        if score >= best_score:
            best_subgraph, best_score = (subgraph, score)

    return best_subgraph, best_score


def create_p_graph(df_edges, df_p):
    df_edges = df_edges.drop_duplicates()

    df_p[COL_NAME] = df_p[COL_NAME].apply(str)
    df_edges[COL_SRC_NAME] = df_edges[COL_SRC_NAME].apply(str)
    df_edges[COL_DST_NAME] = df_edges[COL_DST_NAME].apply(str)

    df_vertices = extract_vertices(df_edges)

    if not set(df_vertices.name.values).issubset(df_p.name.values):
        raise ValueError('Vertex has an edge, but not an associated p-value.')

    df_p = pd.merge(df_p, df_vertices, on=COL_NAME, suffixes=('', '_y'))

    graph = nx.Graph()

    for vertex in df_p.itertuples():
        graph.add_node(n=vertex.name, attr_dict={'type': vertex.type, 'p': vertex.p_value})

    for edge in df_edges.itertuples():
        graph.add_edge(u=edge.SRC_NAME, v=edge.DST_NAME, attr_dict={'name': edge.E_NAME, 'type': edge.E_TYPE})

    return graph


def extract_vertices(df_edges, col_name=COL_NAME, col_type=COL_TYPE):
    df_src_vertices = df_edges[[COL_SRC_NAME, COL_SRC_TYPE]]
    df_dst_vertices = df_edges[[COL_DST_NAME, COL_DST_TYPE]]

    df_src_vertices.columns = [col_name, col_type]
    df_dst_vertices.columns = [col_name, col_type]

    df_vertices = pd.concat([df_src_vertices, df_dst_vertices], axis=0)
    df_vertices = df_vertices.drop_duplicates()

    return df_vertices
