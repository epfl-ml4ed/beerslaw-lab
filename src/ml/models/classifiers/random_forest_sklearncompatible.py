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
        
    def __format(self, x:list, y:list, x_val=[], y_val=[]) -> Tuple[list, list]:
        return [xx for xx in x], [yy for yy in y]
    
    def __format_features(self, x:list) -> list:
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
        x_predict = self.__format_features(x)
        return self._model.predict(x_predict)
    
    def predict_proba(self, x):
        x_predict = self.__format_features(x)
        return self._model.predict_proba(x_predict)
    
    def decision_function(self, x):
        return self.predict_proba(x)
    
    def save(self):
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' 
        os.makedirs(path, exist_ok=True)
        path += self._name + '_f' + str(self._fold) + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path
            
    def save_fold(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' 
        os.makedirs(path, exist_ok=True)
        path += self._name + '_f' + str(fold) + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path
        
    
    
            