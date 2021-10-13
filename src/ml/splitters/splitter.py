import logging
import numpy as np
import pandas as pd
from typing import Tuple

class Splitter:
    """This implements the superclass which creates the folds according to some criteria
    """
    
    def __init__(self, settings: dict):
        self._name = 'kfold splitter'
        self._notation = 'kfsplit'
        self._random_seed = settings['experiment']['random_seed']
        
        self._n_folds = settings['ML']['splitters']['n_folds']
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
        
    def split(self, x:list, y:list)-> Tuple[list, list]:
        """Splitting data into different indices

        Args:
            x (list): features
            y (list): labels
        Returns:
            train_indices: list
            test_indices: list
        """
        raise NotImplementedError
        
        