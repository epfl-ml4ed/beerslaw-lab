import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from ml.splitters.splitter import Splitter

class StratifiedKSplit(Splitter):
    """Stratifier that splits the data into stratified fold

    Args:
        Splitter (Splitter): Inherits from the class Splitter
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'stratified k folds'
        self._notation = 'stratkf'
        
        self._settings = dict(settings)
        self._splitter_settings = settings['ML']['splitters']['stratkf']
        self._random_seed = settings['experiment']['random_seed']
        self.__init_splitter()
        
    def set_n_folds(self, n_folds):
        self._n_folds = n_folds
        
    def __init_splitter(self):
        self._splitter = StratifiedKFold(
            n_splits=self._n_folds,
            random_state=self._random_seed,
            shuffle=self._splitter_settings['shuffle']
            )
        
    def split(self, x:list, y:list) -> Tuple[list, list]:
        if self._splitter_settings['stratifier_col'] == 'y':
            return self._splitter.split(x, y)
        else:
            fakey = [xx[self._splitter_settings['stratifier_col']] for xx in x]
            return self._splitter.split(x, fakey)
            
        
        