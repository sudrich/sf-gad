import numpy as np

from sfgad.utils.validation import check_meta_info_series, check_meta_info_dataframe
from .weighting import Weighting


class TypeSpecificWeight(Weighting):
    """
    This weighting function assigns a weight to every record so that for every vertex type a different weight is used.
    """

    def __init__(self, type_dict):
        self.type_dict = type_dict

    def compute(self, reference_meta_info, current_meta_info):
        check_meta_info_dataframe(reference_meta_info, required_columns=['type'])
        check_meta_info_series(current_meta_info, required_columns=[])

        if not set(list(reference_meta_info['type'])).issubset(set(self.type_dict.keys())):
            raise ValueError("Found DataFrame with types %s that have no specified weight." % set(
                list(reference_meta_info['type'])).difference(self.type_dict.keys()))

        if len(reference_meta_info['type']):
            return np.vectorize(self.type_dict.__getitem__)(reference_meta_info['type'])
        else:
            np.array([])
