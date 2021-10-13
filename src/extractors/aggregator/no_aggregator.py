import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator

class NoAggregator(Aggregator):
    """This class leaves the sequence as is
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'no aggregator'
        self._notation = 'noagg'
        
    def aggregate(self, matrix: list) -> list:
        return matrix