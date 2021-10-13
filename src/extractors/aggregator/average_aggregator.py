import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator

class AverageAggregator(Aggregator):
    """This class averages on the feature representation dimension (such that no matter the length of the sequence, the output length is a constant)
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'average aggregator'
        self._notation = 'aveagg'
        
    def aggregate(self, matrix: list) -> list:
        return np.mean(matrix, axis=0)