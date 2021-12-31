
import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator

class TimestepNormaliser(Aggregator):
    """This class leaves the sequence as is
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'timestepnormalised aggregator'
        self._notation = 'tsnormagg'

    def normalise_ts(self, vector):
        if np.sum(vector) == 0:
            return vector
        vec = list(np.array(vector) / np.sum(vector))
        return vec
        
    def aggregate(self, matrix: list) -> list:
        return [self.normalise_ts(ts) for ts in matrix]