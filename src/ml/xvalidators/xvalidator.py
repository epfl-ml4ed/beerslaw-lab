import numpy as np
import pandas as pd
from typing import Tuple

from extractors.pipeline_maker import PipelineMaker

from ml.splitters.splitter import Splitter
from ml.models.model import Model
from ml.scorers.scorer import Scorer

class XValidator:
    """This implements the different cross validations that we may want to implement
    """
    
    def __init__(self, settings:dict, splitter:Splitter, model:Model, scorer:Scorer):
        self._name = 'cross validator'
        self._notation = 'xval'
        self._random_seed = settings['experiment']['random_seed']
        self._experiment_root = settings['experiment']['root_name']
        self._experiment_name = settings['experiment']['name']
        self._settings = dict(settings)
        self._xval_settings = settings['ML']['xvalidators']
        self._n_folds = settings['experiment']['n_folds']
        
        self._pipeline = PipelineMaker(settings)
        
        self._splitter = splitter
        self._scorer =scorer
        
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
    
    def _update_results(self, old_results: dict, new_results: dict) -> dict:
        for key in new_results:
            if key not in old_results:
                old_results[key] = []
            old_results[key].append(new_results[key])
        return old_results
        
    def xval(self, x:list, y:list) -> dict:
        """Performs the chosen cross validation on x and y

        Args:
            x (list): features
            y (list): labels

        Returns:
            results (dict): 
                Returns a dict where, per outer fold, we have:
                    - indices outer folds
                    - predictions 
                    - scores
                    - per inner folds:
                        - indices
        """
        raise NotImplementedError
