from time import time
from math import ceil
from sfgad.modules.probability_estimation import *
import pandas as pd
import numpy as np


def total_benchmark():
    # Small Size
    benchmark_parametric_estimators(10, 1, 10, 100)
    benchmark_non_parametric_estimators(10, 1, 10, 100)
    # Medium Size
    benchmark_parametric_estimators(100, 1, 100, 20)
    benchmark_non_parametric_estimators(100, 1, 100, 20)
    # Large Size
    benchmark_parametric_estimators(1000, 1, 1000, 5)
    benchmark_non_parametric_estimators(1000, 1, 1000, 5)


def generate_dataset(n_samples, n_features, n_observations):
    samples = pd.DataFrame(np.random.random((n_samples, n_features)),
                           columns=['f_' + str(i) for i in range(n_features)])
    observations = pd.DataFrame(np.random.random((n_observations, n_features)),
                                columns=['f_' + str(i) for i in range(n_features)])
    weights = pd.DataFrame(np.random.random(n_observations), columns=['weight'])

    time_windows = np.random.randint(0, 10, n_observations)
    observations['time_window'] = time_windows
    weights['time_window'] = time_windows
    return samples, observations, weights


def benchmark_parametric_estimators(n_samples, n_features, n_observations, n_runs):
    samples, observations, weights = generate_dataset(n_samples, n_features, n_observations)

    estimators = [
        ('Exponential', Exponential()),
        ('Gaussian', Gaussian()),
        ('Uniform', Uniform())
    ]

    print("PARAMETRIC ESTIMATORS")
    print("====================")

    print("")
    print_dataset_stats(n_samples, n_features, n_observations)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {4: >12}"
          "".format("Estimator", str(ceil(len(samples) / 1000)) + " samples",
                    str(ceil(len(samples) / 100)) + " samples", str(ceil(len(samples) / 10)) + " samples",
                    str(len(samples)) + " samples"))
    print("-" * 82)
    for name, e in estimators:
        time_1 = benchmark_estimator(e, samples[:ceil(len(samples) / 1000)], observations, weights, n=n_runs)
        time_2 = benchmark_estimator(e, samples[:ceil(len(samples) / 100)], observations, weights, n=n_runs)
        time_3 = benchmark_estimator(e, samples[:ceil(len(samples) / 10)], observations, weights, n=n_runs)
        time_4 = benchmark_estimator(e, samples, observations, weights, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {4: >11.4f}s"
              "".format(name, time_1, time_2, time_3, time_4))
    print()


def benchmark_non_parametric_estimators(n_samples, n_features, n_observations, n_runs):
    samples, observations, weights = generate_dataset(n_samples, n_features, n_observations)

    estimators = [
        ('EmpiricalEstimator', EmpiricalEstimator())
    ]

    print("NON-PARAMETRIC ESTIMATORS")
    print("====================")

    print("")
    print_dataset_stats(n_samples, n_features, n_observations)

    print()
    print("Computation time (average over %d runs):" % n_runs)
    print("====================")
    print("{0: <30} {1: >12} {2: >12} {3: >12} {4: >12}"
          "".format("Estimator", str(ceil(len(samples) / 1000)) + " samples",
                    str(ceil(len(samples) / 100)) + " samples", str(ceil(len(samples) / 10)) + " samples",
                    str(len(samples)) + " samples"))
    print("-" * 82)
    for name, e in estimators:
        time_1 = benchmark_estimator(e, samples[:ceil(len(samples) / 1000)], observations, weights, n=n_runs)
        time_2 = benchmark_estimator(e, samples[:ceil(len(samples) / 100)], observations, weights, n=n_runs)
        time_3 = benchmark_estimator(e, samples[:ceil(len(samples) / 10)], observations, weights, n=n_runs)
        time_4 = benchmark_estimator(e, samples, observations, weights, n=n_runs)
        print("{0: <30} {1: >11.4f}s {2: >11.4f}s {3: >11.4f}s {4: >11.4f}s"
              "".format(name, time_1, time_2, time_3, time_4))
    print()


def print_dataset_stats(n_samples, n_features, n_observations):
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("Number of samples:".ljust(25), n_samples))
    print("%s %d" % ("Number of features:".ljust(25), n_features))
    print("%s %d" % ("Number of observations:".ljust(25), n_observations))


def benchmark_estimator(estimator, samples, observations, weights, n=100):
    total = 0
    for i in range(n):
        start = time()
        for sample in samples.itertuples(index=False):
            estimator.estimate(pd.DataFrame(list(sample), columns=samples.columns), observations, weights)
        total += time() - start
    return total / n
