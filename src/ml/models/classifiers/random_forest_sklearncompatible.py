import os
import pickle
import logging 
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

class RandomForest(BaseEstimator, ClassifierMixin, Model):
    """This class implements a random forest classifier

    Args:
        Model (Model): inherits from the model class
        BaseEstimator (BaseEstimator): inherits from the BaseEstimator to use the gridsearch object from sklearn
        ClassifierMixin: inherits from the ClassifierMixin class from sklearn 
    """
    
    def __init__(self, settings={}, n_estimators=-1, criterion=-1, max_depth=-1, min_samples_split=-1, max_leaf_nodes=-1, max_features=-1):
        if settings != {}:
            super().__init__(settings)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.settings = settings
        self._fold = 0
        self._name = 'random forest'
        self._notation = 'rf'
        
    def _set_settings(self):
        """Can't initiate the settings in init, so we're calling the super method here, from the fit function
        """
        super().__init__(self.settings)
        self._name = 'random forest'
        self._notation = 'rf'
        
    def _format(self, x:list, y:list, x_val=[], y_val=[]) -> Tuple[list, list]:
        return [xx for xx in x], [yy for yy in y]
    
    def _format_features(self, x:list) -> list:
        return [xx for xx in x]
    
    def fit(self, x_train:list, y_train:list, x_val=[], y_val=[]):
        self._set_settings()
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_leaf_nodes=self.max_leaf_nodes,
            max_features=self.max_features
        )
        
        x_train, y_train = self.__format(x_train, y_train)
        self._model.fit(x_train, y_train)
        self.params = self._model.get_params()
        self._fold += 1
        return self
    
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
    
    
            