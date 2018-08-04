from time import time
from math import ceil
from sfgad.modules.probability_combiner import *
import pandas as pd
import numpy as np


def total_benchmark():
    # Small Size
    benchmark_direct_combiners(100, 3, 100, 100)
    benchmark_ref_based_combiners(100, 3, 100, 100)
    # Medium Size
    benchmark_direct_combiners(10000, 3, 1000, 20)
    benchmark_ref_based_combiners(10000, 3, 1000, 20)
    # Large Size
    benchmark_direct_combiners(1000000, 3, 10000, 5)
    benchmark_ref_based_combiners(1000000, 3, 10000, 5)


def generate_dataset(n_samples, n_features, n_observations):
    p_values = pd.DataFrame(np.random.random((n_samples, n_features)),
                            columns=['p_' + str(i) for i in range(n_features)])
    ref_p_values = np.random.random((n_observations, n_features))
    return p_values, ref_p_values


def print_dataset_stats(n_samples, n_features, n_observations):
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("Number of samples:".ljust(25), n_samples))
    print("%s %d" % ("Number of features:".ljust(25), n_features))
    print("%s %d" % ("Number of observations:".ljust(25), n_observations))


def benchmark_direct_combiners(n_samples, n_features, n_observations, n_runs):
    p_values, ref_p_values = generate_dataset(n_samples, n_features, n_observations)

    combiners = [
        ('AvgProbability', AvgProbability()),
        ('MaxProbability', MaxProbability()),
        ('MinProbability', MinProbability()),
        ('FirstFeatureProbability', FirstFeatureProbability()),
        ('SelectedFeatureProbability', SelectedFeatureProbability()),
        ('FisherMethod', FisherMethod()),
        ('StoufferMethod', StoufferMethod())
    ]

    print("DIRECT COMBINERS")
    print("====================")

    print("")
    print_dataset_stats(n_samples, n_features, n_observations)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {4: >12}"
          "".format("Combiner", str(ceil(len(p_values) / 1000)) + " samples",
                    str(ceil(len(p_values) / 100)) + " samples", str(ceil(len(p_values) / 10)) + " samples",
                    str(len(p_values)) + " samples"))
    print("-" * 82)
    for name, c in combiners:
        time_1 = benchmark_combiner(c, p_values[:ceil(len(p_values) / 1000)], ref_p_values, n=n_runs)
        time_2 = benchmark_combiner(c, p_values[:ceil(len(p_values) / 100)], ref_p_values, n=n_runs)
        time_3 = benchmark_combiner(c, p_values[:ceil(len(p_values) / 10)], ref_p_values, n=n_runs)
        time_4 = benchmark_combiner(c, p_values, ref_p_values, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {4: >11.4f}s"
              "".format(name, time_1, time_2, time_3, time_4))
    print()


def benchmark_ref_based_combiners(n_samples, n_features, n_observations, n_runs):
    p_values, ref_p_values = generate_dataset(n_samples, n_features, n_observations)

    combiners = [
        ('EmpiricalCombiner', EmpiricalCombiner())
    ]

    print("REFERENCE-BASED COMBINERS")
    print("====================")

    print("")
    print_dataset_stats(n_samples, n_features, n_observations)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {4: >12}"
          "".format("Combiner", str(ceil(len(p_values) / 1000)) + " samples",
                    str(ceil(len(p_values) / 100)) + " samples", str(ceil(len(p_values) / 10)) + " samples",
                    str(len(p_values)) + " samples"))
    print("-" * 82)
    for name, c in combiners:
        time_1 = benchmark_combiner(c, p_values[:ceil(len(p_values) / 1000)], ref_p_values, n=n_runs)
        time_2 = benchmark_combiner(c, p_values[:ceil(len(p_values) / 100)], ref_p_values, n=n_runs)
        time_3 = benchmark_combiner(c, p_values[:ceil(len(p_values) / 10)], ref_p_values, n=n_runs)
        time_4 = benchmark_combiner(c, p_values, ref_p_values, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {4: >11.4f}s"
              "".format(name, time_1, time_2, time_3, time_4))
    print()


def benchmark_combiner(combiner, p_values, ref_p_values, n=100):
    total = 0
    for i in range(n):
        start = time()
        for sample in p_values.itertuples(index=False):
            combiner.combine(list(sample), ref_p_values)
        total += time() - start
    return total / n
