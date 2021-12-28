import logging
import numpy as np 
import pandas as pd
from typing import Tuple

from ml.samplers.sampler import Sampler

class NoSampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'no resampling'
        self._notation = 'nosplr'
        
    def sample(self, x:list, y:list) -> Tuple[list, list]:
        self._indices = list(range(len(x)))
        return x, y

    def get_indices(self) -> np.array:
        return self._indices