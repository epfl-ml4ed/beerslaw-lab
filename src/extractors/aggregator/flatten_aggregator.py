import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator

class FlattenAggregator(Aggregator):
    """This class vectorises the feature matrices (from 2D vectors to 1D vectors)
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'flatten aggregator'
        self._notation = 'flatagg'
        
    def aggregate(self, matrix: list) -> list:
        return list(np.array(matrix).flatten())