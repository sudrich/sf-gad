from time import time
from sfgad.modules.feature import *
import networkx as nx
import pandas as pd
import numpy as np
import datetime


def total_benchmark():
    # Small Sparse Graph
    benchmark_static_features(n_vertices=100, n_edges=500, n_vertex_types=3, n_runs=100)
    benchmark_dynamic_features(n_vertices=100, n_edges=500, n_vertex_types=3, n_timesteps=30, n_runs=100)
    # Small Dense Graph
    benchmark_static_features(n_vertices=100, n_edges=2500, n_vertex_types=3, n_runs=100)
    benchmark_dynamic_features(n_vertices=100, n_edges=2500, n_vertex_types=3, n_timesteps=30, n_runs=100)
    # Medium Sparse Graph
    benchmark_static_features(n_vertices=10000, n_edges=50000, n_vertex_types=3, n_runs=20)
    benchmark_dynamic_features(n_vertices=10000, n_edges=50000, n_vertex_types=3, n_timesteps=30, n_runs=20)
    # Medium Dense Graph
    benchmark_static_features(n_vertices=10000, n_edges=2500000, n_vertex_types=3, n_runs=20)
    benchmark_dynamic_features(n_vertices=10000, n_edges=2500000, n_vertex_types=3, n_timesteps=30, n_runs=20)
    # Large Sparse Graph
    benchmark_static_features(n_vertices=1000000, n_edges=5000000, n_vertex_types=3, n_runs=5)
    benchmark_dynamic_features(n_vertices=1000000, n_edges=5000000, n_vertex_types=3, n_timesteps=30, n_runs=5)
    # Large Dense Graph
    benchmark_static_features(n_vertices=1000000, n_edges=2500000000, n_vertex_types=3, n_runs=5)
    benchmark_dynamic_features(n_vertices=1000000, n_edges=2500000000, n_vertex_types=3, n_timesteps=30, n_runs=5)


def benchmark_dynamic_features(n_vertices, n_edges, n_vertex_types, n_timesteps, n_runs):
    dfs = []
    for i in range(n_timesteps):
        g = generate_graph(n_vertices, n_edges, n_vertex_types)
        dfs.append(from_nx(g, time=i))

    # vertex_types = list(dfs[0][['SRC_TYPE', 'DST_TYPE']].unstack().unique())
    edge_types = list(dfs[0].E_TYPE.unique())
    # timestamp = dfs[0].TIMESTAMP.unique()[0]

    features = [
        ('VertexDegreeDifference', VertexDegreeDifference()),
        ('VertexDegreeDifferenceByType', VertexDegreeDifferenceByType(edge_types)),
        ('HotSpotFeatures', HotSpotFeatures())
    ]

    print("DYNAMIC FEATURES")
    print("====================")

    print("")
    print_dataset_stats(n_vertices, n_edges, n_vertex_types, n_timesteps)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {3: >12}"
          "".format("Feature", "n_jobs=1", "n_jobs=2", "n_jobs=4", "n_jobs=8"))
    print("-" * 82)
    for name, f in features:
        time_1 = benchmark_dynamic_feature(f, dfs, n_jobs=1, n=n_runs)
        time_2 = benchmark_dynamic_feature(f, dfs, n_jobs=2, n=n_runs)
        time_4 = benchmark_dynamic_feature(f, dfs, n_jobs=4, n=n_runs)
        time_8 = benchmark_dynamic_feature(f, dfs, n_jobs=8, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {3: >11.4f}s"
              "".format(name, time_1, time_2, time_4, time_8))
    print()


def benchmark_static_features(n_vertices, n_edges, n_vertex_types, n_runs):
    g = generate_graph(n_vertices, n_edges, n_vertex_types)
    df = from_nx(g)

    vertex_types = list(df[['SRC_TYPE', 'DST_TYPE']].unstack().unique())
    edge_types = list(df.E_TYPE.unique())
    timestamp = df.TIMESTAMP.unique()[0]

    features = [
        ('VertexAcitivty', VertexActivity()),
        ('VertexAcitivtyByType', VertexActivityByType(edge_types)),
        ('VertexDegree', VertexDegree()),
        ('VertexDegreeByType', VertexDegreeByType(edge_types)),
        ('TwoHopReach', TwoHopReach()),
        ('TwoHopReachByType', TwoHopReachByType(vertex_types)),
        ('IncidentTriangles', IncidentTriangles()),
        ('IncidentTrianglesByType', IncidentTrianglesByType(edge_types)),
        ('ExternalFeature', ExternalFeature({str(v): [(timestamp, v % 10)] for v in g.nodes()}))
    ]

    print("STATIC FEATURES")
    print("====================")

    print("")
    print_dataset_stats(n_vertices, n_edges, n_vertex_types)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {3: >12}"
          "".format("Feature", "n_jobs=1", "n_jobs=2", "n_jobs=4", "n_jobs=8"))
    print("-" * 82)
    for name, f in features:
        time_1 = benchmark_static_feature(f, df, n_jobs=1, n=n_runs)
        time_2 = benchmark_static_feature(f, df, n_jobs=2, n=n_runs)
        time_4 = benchmark_static_feature(f, df, n_jobs=4, n=n_runs)
        time_8 = benchmark_static_feature(f, df, n_jobs=8, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {3: >11.4f}s"
              "".format(name, time_1, time_2, time_4, time_8))
    print()


def benchmark_static_feature(feature, df, n_jobs=1, n=100):
    total = 0
    for i in range(n):
        start = time()
        feature.process_vertices(df, n_jobs)
        total += time() - start
    return total / n


def benchmark_dynamic_feature(feature, dfs, n_jobs=1, n=100):
    total = 0
    for i in range(n):
        feature.reset()
        for df in dfs[:-1]:
            feature.process_vertices(df, n_jobs)

        start = time()
        feature.process_vertices(dfs[-1], n_jobs)
        total += time() - start
    return total / n


def print_dataset_stats(n_vertices, n_edges, n_vertex_types, n_timesteps=None):
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("Number of vertices:".ljust(25), n_vertices))
    print("%s %d" % ("Number of edges:".ljust(25), n_edges))
    print("%s %d" % ("Number of vertex types:".ljust(25), n_vertex_types))
    print("%s %d" % ("Number of edge types:".ljust(25), n_vertex_types * n_vertex_types))
    if n_timesteps:
        print("%s %d" % ("Number of timesteps:".ljust(25), n_timesteps))


def generate_graph(n_vertices, n_edges, n_vertex_types):
    graph = nx.gnm_random_graph(n_vertices, n_edges)
    for v, data in graph.nodes(data=True):
        data['type'] = v % n_vertex_types
    for u, v, data in graph.edges(data=True):
        data['type'] = str(graph.node[u]['type']) + '_' + str(graph.node[v]['type'])
    return graph


DF_COLUMNS = ['TIMESTAMP', 'E_NAME', 'E_TYPE', 'SRC_NAME', 'SRC_TYPE', 'DST_NAME', 'DST_TYPE']


def from_nx(network, timestep=0):
    df = np.asarray([[datetime.datetime.fromordinal(1).replace(year=2017) + datetime.timedelta(days=timestep),
                      data.get('name', str((i, j))),
                      data.get('type', 'E_TYPE'),
                      str(i),
                      network.node[i].get('type', 'V_TYPE'),
                      str(j),
                      network.node[j].get('type', 'V_TYPE')] for (i, j, data) in network.edges(data=True)])
    return pd.DataFrame(data=df, columns=DF_COLUMNS)
