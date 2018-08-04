from time import time
from sfgad.modules.weighting import *
from math import ceil
import pandas as pd
import numpy as np


def total_benchmark():
    # Small Size
    benchmark_time_based_weightings(100, 100, 100)
    benchmark_meta_based_weightings(100, 100, 100)
    # Medium Size
    benchmark_time_based_weightings(1000, 10000, 50)
    benchmark_meta_based_weightings(1000, 10000, 50)
    # Large Size
    benchmark_time_based_weightings(10000, 1000000, 10)
    benchmark_meta_based_weightings(10000, 1000000, 10)


def generate_dataset(n_samples, n_observations):
    samples = pd.DataFrame(np.full(n_samples, 100, dtype=np.int64), columns=['time'])
    observations = pd.DataFrame(
        {'time_window': np.random.randint(100, size=n_observations), 'type': np.random.randint(3, size=n_observations)},
        columns=['time_window', 'type'])
    return samples, observations


def print_dataset_stats(n_samples, n_observations):
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("Number of samples:".ljust(25), n_samples))
    print("%s %d" % ("Number of observations:".ljust(25), n_observations))


def benchmark_time_based_weightings(n_samples, n_observations, n_runs):
    samples, observations = generate_dataset(n_samples, n_observations)

    weighting_functions = [
        ('ConstantWeight', ConstantWeight()),
        ('ExponentialDecayWeight', ExponentialDecayWeight(5)),
        ('LinearDecayWeight', LinearDecayWeight(0.5))
    ]

    print("TIME-BASED WEIGHTING FUNCTIONS")
    print("====================")

    print("")
    print_dataset_stats(n_samples, n_observations)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {4: >12}"
          "".format("Weighting", str(ceil(len(samples) / 1000)) + " samples",
                    str(ceil(len(samples) / 100)) + " samples", str(ceil(len(samples) / 10)) + " samples",
                    str(len(samples)) + " samples"))
    print("-" * 82)
    for name, wf in weighting_functions:
        time_1 = benchmark_weighting(wf, samples[:ceil(len(samples) / 1000)], observations, n=n_runs)
        time_2 = benchmark_weighting(wf, samples[:ceil(len(samples) / 100)], observations, n=n_runs)
        time_3 = benchmark_weighting(wf, samples[:ceil(len(samples) / 10)], observations, n=n_runs)
        time_4 = benchmark_weighting(wf, samples, observations, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {4: >11.4f}s"
              "".format(name, time_1, time_2, time_3, time_4))
    print()


def benchmark_meta_based_weightings(n_samples, n_observations, n_runs):
    samples, observations = generate_dataset(n_samples, n_observations)

    weighting_functions = [
        # ('SimilarityBasedWeight', SimilarityBasedWeight()),
        ('TypeSpecificWeight', TypeSpecificWeight({0: 1, 1: 5, 2: 10}))
    ]

    print("META-BASED WEIGHTING FUNCTIONS")
    print("====================")

    print("")
    print_dataset_stats(n_samples, n_observations)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {4: >12}"
          "".format("Weighting", str(ceil(len(samples) / 1000)) + " samples",
                    str(ceil(len(samples) / 100)) + " samples", str(ceil(len(samples) / 10)) + " samples",
                    str(len(samples)) + " samples"))
    print("-" * 82)
    for name, wf in weighting_functions:
        time_1 = benchmark_weighting(wf, samples[:ceil(len(samples) / 1000)], observations, n=n_runs)
        time_2 = benchmark_weighting(wf, samples[:ceil(len(samples) / 100)], observations, n=n_runs)
        time_3 = benchmark_weighting(wf, samples[:ceil(len(samples) / 10)], observations, n=n_runs)
        time_4 = benchmark_weighting(wf, samples, observations, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {4: >11.4f}s"
              "".format(name, time_1, time_2, time_3, time_4))
    print()


def benchmark_weighting(weighting, samples, observations, n=100):
    total = 0
    for i in range(n):
        start = time()
        for current_time in samples['time']:
            weighting.compute(observations, int(current_time))
        total += time() - start
    return total / n
