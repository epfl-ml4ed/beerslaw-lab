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
        x_predict = self._format_features(x)
        return self._model.predict(x_predict)
    
    def predict_proba(self, x:list) -> list:
        x_predict = self._format_features(x)
        probs = self._model.predict_proba(x_predict)
        if len(probs[0]) != self._n_classes:
            preds = self._model.predict(x_predict)
            probs = self._inpute_full_prob_vector(preds, probs)
        return probs
    
    def save(self):
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/'
        os.makedirs(path, exist_ok=True)
        path += self._name + '_l' + self._settings['data']['adjuster']['limit'] + '_f' + str(self._fold) + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path
    
    def get_path(self, fold:int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/'
        path += self._name + '_l' + str(self._settings['data']['adjuster']['limit']) + '_f' + str(fold) + '.pkl'
        return path
            
    def save_fold(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/'
        os.makedirs(path, exist_ok=True)
        path += self._name + '_l' + str(self._settings['data']['adjuster']['limit']) + '_f' + str(fold) + '.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path