import os
import shutil
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.neural_network import MLPClassifier

import logging

from ml.models.model import Model



class ScikitNN(Model):
    """This class implements a random forest classifier
    Args:
        Model (Model): inherits from the model class
    TODO: Check format (np.asaray)
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'scikit neural network'
        self._notation = 'sknn'
        self._model_settings = settings['ML']['models']['classifiers']['sknn']
        
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        return [xx for xx in x], [yy for yy in y]
    
    def _format_features(self, x:list) -> list:
        return [xx for xx in x]
    
    def _init_model(self, len_xtrain:int):
        self._model = MLPClassifier(
            random_state=0,
            max_iter=self._model_settings['max_iter'],
            solver=self._model_settings['solver'],
            learning_rate=self._model_settings['learning_rate'],
            learning_rate_init=self._model_settings['learning_rate_init'],
            batch_size=int(len_xtrain*0.9),
            hidden_layer_sizes=self._model_settings['hidden_layer_sizes'],
            activation=self._model_settings['activation']
        )
        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)
        
        logging.debug('x_train shape: {}, y_train shape: {}'.format(np.array(x_train).shape, np.array(y_train).shape))
        logging.debug('x_val shape: {}, y_val shape: {}'.format(np.array(x_val).shape, np.array(y_val).shape))
        logging.debug('n_ classes: {}'.format(self._n_classes))
        
        self._init_model(len(x_train))
        
        self._model.fit(
            x_train, y_train
        )
        
    def predict(self, x:list) -> list:
        x_predict = self._format_features(x)
        predictions = self._model.predict(x_predict)
        return predictions
    
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
        with open(self, 'wb') as fp:
            pickle.dump(self, fp)
        return path
    
    def get_path(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '_l' + str(self._settings['data']['adjuster']['limit']) + '_f' + str(fold) + '.pkl'
        return path 
    
    def save_fold(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' 
        os.makedirs(path, exist_ok=True)
        path += self._name + '_l' + str(self._settings['data']['adjuster']['limit']) + '_f' + str(fold) + '.pkl'
        
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)
        return path 