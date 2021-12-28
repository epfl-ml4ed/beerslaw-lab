import logging
import numpy as np 
import pandas as pd
from typing import Tuple
from collections import Counter

from imblearn.over_sampling import RandomOverSampler as ros

from ml.samplers.sampler import Sampler

class RandomOversampler(Sampler):
    """This class oversamples the minority class to rebalance the distribution at 50/50. It takes all of the minority samples, and then randomly picks the other to fulfill the 50/50 criterion

    Args:
        Sampler (Sampler): Inherits from the Sampler class
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'random oversampling'
        self._notation = 'rdmos'
        
        self._ros = ros(random_state=0)
        
    def sample(self, x:list, y:list) -> Tuple[list, list]:
        logging.debug('distribution before the sampling: {}'.format(sorted(Counter(y).items())))

        xlen = len(x)
        x_resampled, y_resampled = self._ros.fit_resample([list(range(10)) for _ in range(xlen)], y)
        self._indices = self._ros.sample_indices_
        x_resampled = [x[idx] for idx in self._indices]

        logging.debug('distrbution after the sampling: {}'.format(sorted(Counter(y_resampled).items())))
        return x_resampled, y_resampled        

    def get_indices(self) -> np.array:
        return self._indices