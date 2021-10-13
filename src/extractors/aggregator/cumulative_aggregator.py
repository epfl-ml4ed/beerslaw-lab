import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator

class CumulativeAverageAggregator(Aggregator):
    """This class averages on the feature representation dimension (such that no matter the length of the sequence, the output length is a constant) in a cumulative way.
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'cumulative average aggregator'
        self._notation = 'cumulaveagg'
        
    def aggregate(self, matrix: list) -> list:
        seq = [np.mean(matrix[:i+1], axis=0) for i in range(len(matrix))]
        return np.mean(seq, axis=0)