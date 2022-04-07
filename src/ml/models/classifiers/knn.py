import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model
from sklearn.neighbors import KNeighborsClassifier

class KNNModel(Model):
    """This class implements a K nearest neighbour
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'k nearest neighbour'
        self._notation = 'knn'
        self._model_settings = settings['ML']['models']['classifiers']['knn']
        self._fold = 0
        
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        return [xx for xx in x], [yy for yy in y]
    
    def _format_features(self, x:list) -> list:
        return [xx for xx in x]
    
    def _init_model(self):
        self._model = KNeighborsClassifier(
            n_neighbors=self._model_settings['n_neighbours'],
            weights=self._model_settings['weights'],
            algorithm=self._model_settings['algorithm'],
            leaf_size=self._model_settings['leaf_size'], 
            metric=self._model_settings['metric'],
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

    def save_fold_early(self, fold: int) -> str:
        return self.save_fold_early_sklearn(fold)