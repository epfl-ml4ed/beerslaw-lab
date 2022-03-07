import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator

class NormalisedAggregator(Aggregator):
    """This class leaves the sequence as is
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'normalised aggregator'
        self._notation = 'normagg'
        
    def aggregate(self, matrix: list) -> list:
        print(matrix)
        print(list(np.array(matrix) / np.sum(matrix)))
        print()
        return list(np.array(matrix) / np.sum(matrix))