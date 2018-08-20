from sfgad.analyzer import Analyzer
from sfgad.modules.features import VertexDegree, HotSpotFeatures, VertexDegreeDifference, VertexDegreeByType
from sfgad.modules.observation_selection import HistoricSameSelection, FallbackSelection, HistoricAgeSimilarSelection
from sfgad.modules.probability_combination import SelectedFeatureProbability, FisherMethod, EmpiricalCombiner
from sfgad.modules.probability_estimation import Gaussian, EmpiricalEstimator
from sfgad.modules.weighting import ExponentialDecayWeight, ConstantWeight


def dapa(half_life=10, n_jobs=4):
    """
    Returns an analyzer configured according to the DAPA-V10 algorithm.
    :param half_life: Half-life period used for the exponential decay of weights. Default is 10.
    :param n_jobs: Number of jobs. Default is 4.
    :return: Configured analyzer.
    """
    features_list = [VertexDegree()]
    observation_gatherer = HistoricSameSelection()
    weighting_function = ExponentialDecayWeight(half_life=half_life)
    probability_estimator = Gaussian()
    probability_combiner = SelectedFeatureProbability()

    analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                        probability_combiner, n_jobs=n_jobs)

    return analyzer


def dapa_modified(half_life=10, n_jobs=4):
    """
    Returns an analyzer configured according to the DAPA-V10 algorithm with some improvements.
    :param half_life: Half-life period used for the exponential decay of weights. Default is 10.
    :param n_jobs: Number of jobs. Default is 4.
    :return: Configured analyzer.
    """
    features_list = [VertexDegree()]
    observation_gatherer = FallbackSelection(HistoricSameSelection(), HistoricAgeSimilarSelection(), threshold=20)
    weighting_function = ExponentialDecayWeight(half_life=half_life)
    probability_estimator = Gaussian()
    probability_combiner = SelectedFeatureProbability()

    analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                        probability_combiner, n_jobs=n_jobs)

    return analyzer


def hotspot(half_life=10, window_size=24 * 60 * 60, mode='HC', n_jobs=4):
    """
    Returns an analyzer configured according to the HotSpot algorithm.
    :param n_jobs: Number of jobs. Default is 4.
    :param half_life: Half-life period used for the calculation of the HotSpot features. Default is 10.
    :param window_size: Length of the time window used for the calculation of the HotSpot features. Default is 1 day.
    :param mode: Metric used from the HotSpot features. Either 'HC' for CorrelationChange, 'HA' for MagnitudeChange or
    'Fisher' for a combination of both using Fisher's method. Default is 'HC'.
    :param n_jobs: Number of jobs. Default is 4.
    :return: Configured analyzer.
    """
    features_list = [HotSpotFeatures(half_life=half_life, window_size=window_size)]
    observation_gatherer = HistoricSameSelection()
    weighting_function = ConstantWeight(default_weight=1)
    probability_estimator = Gaussian()
    if mode == 'HC':
        probability_combiner = SelectedFeatureProbability(feature_position=0)
    elif mode == 'HA':
        probability_combiner = SelectedFeatureProbability(feature_position=1)
    elif mode == 'Fisher':
        probability_combiner = FisherMethod()
    else:
        raise ValueError("Mode unsupported for HotSpot.")

    analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                        probability_combiner, n_jobs=n_jobs)

    return analyzer


def hotspot_modified(half_life=10, window_size=24 * 60 * 60, mode='HC', n_jobs=4):
    """
    Returns an analyzer configured according to the HotSpot algorithm with some improvements.
    :param half_life: Half-life period used for the calculation of the HotSpot features. Default is 10.
    :param window_size: Length of the time window used for the calculation of the HotSpot features. Default is 1 day.
    :param mode: Metric used from the HotSpot features. Either 'HC' for CorrelationChange, 'HA' for MagnitudeChange or
    'Fisher' for a combination of both using Fisher's method. Default is 'HC'.
    :param n_jobs: Number of jobs. Default is 4.
    :return: Configured analyzer.
    """
    features_list = [HotSpotFeatures(half_life=half_life, window_size=window_size)]
    observation_gatherer = FallbackSelection(HistoricSameSelection(), HistoricAgeSimilarSelection(),
                                             threshold=2 * half_life)
    weighting_function = ConstantWeight(default_weight=1)
    probability_estimator = Gaussian()
    if mode == 'HC':
        probability_combiner = SelectedFeatureProbability(feature_position=0)
    elif mode == 'HA':
        probability_combiner = SelectedFeatureProbability(feature_position=1)
    elif mode == 'Fisher':
        probability_combiner = FisherMethod()
    else:
        raise ValueError("Mode unsupported for HotSpot.")

    analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                        probability_combiner, n_jobs=n_jobs)

    return analyzer


# def dnd(n_jobs=4, mode='acc'):
#     """
#     Returns an analyzer configured according to the DND algorithm.
#     :param n_jobs: Number of jobs. Default is 4.
#     :param mode: Metric used from the DND features. Either 'fr' for firing-rate, 'acc' for acceleration, 'chi_comp' for
#     chi w.r.t. the comparison, 'chi_sys' for chi w.r.t. the whole system or 'Fisher' for a combination of all using
#     Fisher's method. Default is 'acc'.
#     :return: Configured analyzer.
#     """
#     features_list = [DNDFeatures()]
#     observation_gatherer = HistoricSameSelection()
#     weighting_function = ConstantWeight(default_weight=1)
#     probability_estimator = Gaussian()
#     if mode == 'fr':
#         probability_combiner = SelectedFeatureProbability(feature_position=0)
#     elif mode == 'acc':
#         probability_combiner = SelectedFeatureProbability(feature_position=1)
#     elif mode == 'chi_comp':
#         probability_combiner = SelectedFeatureProbability(feature_position=2)
#     elif mode == 'chi_sys':
#         probability_combiner = SelectedFeatureProbability(feature_position=3)
#     elif mode == 'Fisher':
#         probability_combiner = FisherMethod()
#     else:
#         raise ValueError("Mode unsupported for DND.")
#
#     analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
#                         probability_combiner, n_jobs=n_jobs)
#
#     return analyzer


def dnd_modified(n_jobs=4):
    """
    Returns an analyzer configured according to the DND algorithm with some improvements.
    :param n_jobs: Number of jobs. Default is 4.
    :return: Configured analyzer.
    """
    features_list = [VertexDegreeDifference()]
    observation_gatherer = FallbackSelection(HistoricSameSelection(), HistoricAgeSimilarSelection(), threshold=20)
    weighting_function = ConstantWeight(default_weight=1)
    probability_estimator = Gaussian()
    probability_combiner = SelectedFeatureProbability()

    analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                        probability_combiner, n_jobs=n_jobs)

    return analyzer


def nphgs(edge_types=tuple('TYPE'), n_jobs=4):
    """
    Returns an analyzer configured according to the NPHGS algorithm.
    :param edge_types: List of edge types present in the graph. Default is ['TYPE'].
    :param n_jobs: Number of jobs. Default is 4.
    :return: Configured analyzer.
    """
    features_list = [VertexDegreeByType(edge_types=edge_types)]
    observation_gatherer = FallbackSelection(HistoricSameSelection(), HistoricAgeSimilarSelection(), threshold=20)
    weighting_function = ConstantWeight(default_weight=1)
    probability_estimator = EmpiricalEstimator()
    probability_combiner = EmpiricalCombiner()

    analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                        probability_combiner, n_jobs=n_jobs)

    return analyzer


def nphgs_modified(edge_types=tuple('TYPE'), half_life=10, n_jobs=4):
    """
    Returns an analyzer configured according to the NPHGS algorithm with some improvements.
    :param edge_types: List of edge types present in the graph. Default is ['TYPE'].
    :param half_life: Half-life period used for the exponential decay of weights. Default is 10.
    :param n_jobs: Number of jobs. Default is 4.
    :return: Configured analyzer.
    """
    features_list = [VertexDegreeByType(edge_types=edge_types)]
    observation_gatherer = FallbackSelection(HistoricSameSelection(), HistoricAgeSimilarSelection(), threshold=20)
    weighting_function = ExponentialDecayWeight(half_life=half_life)
    probability_estimator = EmpiricalEstimator()
    probability_combiner = EmpiricalCombiner()

    analyzer = Analyzer(features_list, observation_gatherer, weighting_function, probability_estimator,
                        probability_combiner, n_jobs=n_jobs)

    return analyzer
