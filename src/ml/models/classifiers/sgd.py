import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model
from sklearn.linear_model import SGDClassifier

class SGDModel(Model):
    """This class implements a sgd classifier
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'stochastic gradient descent'
        self._notation = 'sgd'
        self._model_settings = settings['ML']['models']['classifiers']['sgd']
        self._fold = 0
        
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        return [xx for xx in x], [yy for yy in y]
    
    def _format_features(self, x:list) -> list:
        return [xx for xx in x]
    
    def _init_model(self):
        self._model = SGDClassifier(
            loss=self._model_settings['loss'],
            penalty=self._model_settings['penalty'],
            alpha=self._model_settings['alpha'],
            random_state=0, # debug,
        )
        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        self._init_model()
        x_train, y_train = self._format(x_train, y_train)
        self._model.fit(x_train, y_train)
        self._fold += 1
        
    def predict(self, x:list) -> list:
        return self.predict_sklearn(x)
    
    def predict_proba(self, x:list) -> list:
        return self.predict_proba_sklearn(x)
    
    def save(self):
        return self.save_sklearn()
    
    def get_path(self, fold:int) -> str:
        return self.get_path_sklearn(fold)
            
    def save_fold(self, fold: int) -> str:
        return self.save_fold_sklearn(fold)