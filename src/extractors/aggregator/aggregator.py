import numpy as np
import pandas as pd

class Aggregator:
    """This class turns the 3D feature matrices into 2D matrices
    """
    
    def __init__(self):
        self._name = 'aggregator'
        self._notation = 'agg'
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
        
    def aggregate(self, matrix: list) -> list:
        raise NotImplementedError