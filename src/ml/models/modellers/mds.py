import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model
from sklearn.manifold import MDS

class MDSClassifier(Model):
    """This class implements a MDS
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'MDS'
        self._notation = 'mds'
        self._model_settings = settings['ML']['models']['classifiers']['mds']
        self._fold = 0
        
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        return [xx for xx in x], [yy for yy in y]
    
    def _format_features(self, x:list) -> list:
        return [xx for xx in x]
    
    def _init_model(self):
        self._model = MDS(
            n_components=self._model_settings['n_components'],
            metric=self._model_settings['metric'],
            n_init=self._model_settings['n_init'],
            eps=self._model_settings['eps'],
            random_state=0, # debug,
            dissimilarity=self._model_settings['dissimilarity'],
        )
        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        self._init_model()
        x_train, y_train = self._format(x_train, y_train)
        self._model.fit(x_train, y_train)
        self._fold += 1
        
    def predict(self, x:list) -> list:
        self.predict_tensorflow(x)
    
    def predict_proba(self, x:list) -> list:
        self.predict_proba_tensorflow(x)
    
    def save(self) -> str:
        self.save_tensorflow()
    
    def get_path(self, fold: int) -> str:
        self.get_path(fold)
            
    def save_fold(self, fold: int) -> str:
        self.save_fold_tensorflow(fold)

    def save_fold_early(self, fold: int) -> str:
        return self.save_fold_early_tensorflow(fold)