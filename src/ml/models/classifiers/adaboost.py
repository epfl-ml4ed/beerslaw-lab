import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

class ADABoostModel(Model):
    """This class implements an ada boost model with multiple basic classifiers
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'adaboost'
        self._notation = 'adaboost'
        self._model_settings = settings['ML']['models']['classifiers']['adaboost']
        self._fold = 0
        
        self._algo_map = {
            'decision_tree': DecisionTreeClassifier(),
        }
        
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        return [xx for xx in x], [yy for yy in y]
    
    def _format_features(self, x:list) -> list:
        return [xx for xx in x]
    
    def _init_model(self):
        self._model = AdaBoostClassifier(
            base_estimator=self._algo_map[self._model_settings['base_estimator']],
            n_estimators=self._model_settings['n_estimators'],
            learning_rate=self._model_settings['learning_rate'],
            algorithm=self._model_settings['algorithm'], 
            random_state=0
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

    