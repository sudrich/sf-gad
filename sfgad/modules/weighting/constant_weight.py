import numpy as np

from sfgad.utils.validation import check_meta_info_dataframe, check_meta_info_series
from .weighting import Weighting


class ConstantWeight(Weighting):
    """
    This weighting function assigns a constant default weight of to every record
    """

    def __init__(self, weight=1):
        self.weight = weight

    def compute(self, reference_meta_info, current_meta_info):
        check_meta_info_dataframe(reference_meta_info, required_columns=[])
        check_meta_info_series(current_meta_info, required_columns=[])

        return np.full(len(reference_meta_info), self.weight)
