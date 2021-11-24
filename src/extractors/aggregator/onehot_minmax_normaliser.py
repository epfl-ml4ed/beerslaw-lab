import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator
from sklearn.preprocessing import MinMaxScaler

class OneHotMinMaxNormaliserAggregator(Aggregator):
    """This class leaves the sequence as is
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'onehotnormalised aggregator'
        self._notation = '1hotnormagg'
        
    def aggregate(self, matrix: list) -> list:
        scaler = MinMaxScaler()
        scaler.fit(matrix)
        normalised = scaler.transform(matrix)
        normalised = [list(n) for n in normalised]
        return normalised