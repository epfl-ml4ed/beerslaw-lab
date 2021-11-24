import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.model_selection import StratifiedKFold

from ml.splitters.splitter import Splitter

class FlatStratified(Splitter):
    """Stratifier that splits the data into a train and a test fold (meant for flat cross valisation, ie non nested).
    This particular partition is for the chemlab (Fall 2021), where the students are stratified with regards to their 
    language, their year, their field, their prior knowledge and their score in the post test.

    Args:
        Splitter (Splitter): Inherits from the class Splitter
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'flat stratified folds'
        self._notation = 'flatstrat'
        
        self._settings = dict(settings)
        self._splitter_settings = settings['ML']['splitters']['flatstrat']
        self._random_seed = settings['experiment']['random_seed']
        
    def set_n_folds(self, n_folds):
        self._n_folds = n_folds
        
    def split(self, x:list, y:list) -> Tuple[list, list]:
        with open(self._splitter_settings['test_path'], 'rb') as fp:
            test_usernames = pickle.load(fp)

        with open(self._splitter_settings['train_path'], 'rb') as fp:
            train_usernames = pickle.load(fp)

        idd = self._settings['id_dictionary']
        test = [idx for idx in test_usernames if idx in idd['index']]
        train = [idx for idx in train_usernames if idx in idd['index']]
        test = [idd['index'][username] for username in test]
        train = [idd['index'][username] for username in train]

        test = [int(self._indices.index(idx)) for idx in test]
        train = [int(self._indices.index(idx)) for idx in train]
        print(train)
        print(test)

        return [[train, test]]
            
        
        