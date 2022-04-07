import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Tuple

import tensorflow as tf
from ml.models.model import Model
from sklearn.base import BaseEstimator, ClassifierMixin


class SimpleNN(Model, BaseEstimator, ClassifierMixin):
    """This class implements a simple neural network
    Args:
        Model (Model): inherits from the model class
        BaseEstimator (BaseEstimator): inherits from the BaseEstimator to use the gridsearch object from sklearn
        ClassifierMixin: inherits from the ClassifierMixin class from sklearn 
    """
    
    def __init__(self, settings={}, dense_units=-1, dropout=-1, batch_size=-1, shuffle=-1, epochs=-1, verbose=0, early_stopping=0):
        if settings != {}:
            super().__init__(settings)
        self._name = '1-layer neural network'
        self._notation = '1nn'
        
        self.settings = settings
        self.dense_units = dense_units
        self.dropout = dropout
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        
    def _set_settings(self, settings):
        """Can't initiate the settings in init, so we're calling the super method here, from the fit function
        """
        super().__init__(settings)
        self._metric_reporting = settings['ML']['models']['classifiers']['1nn']['metric_reporting']
        self._name = '1-layer neural network'
        self._notation = '1nn'
        
    def __format(self, x:list, y:list) -> Tuple[list, list]:
        return np.asarray([xx for xx in x], dtype=np.float), np.asarray([yy for yy in y], dtype=np.float)   
    
    def __format_features(self, x:list) -> list:
        return np.asarray([xx for xx in x], dtype=np.float)
    
        
    def fit(self, x_train:list, y_train:list, x_val=[], y_val=[]):
        self._set_settings(self.settings)
        # Init model
        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Dense(self.dense_units))
        self._model.add(tf.keras.layers.Dropout(self.dropout))
        self._model.add(tf.keras.layers.Dense(self._n_classes, activation='softmax'))
        
        self._model.compile(
            loss=['sparse_categorical_crossentropy'], optimizer='adam', metrics=self._metric_reporting
        )
        self._callbacks = []
        if self.early_stopping:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, min_delta=0.001, 
                restore_best_weights=True
            )
            self._callbacks.append(early_stopping)
            
        # Training
        x_train, y_train = self.__format(x_train, y_train)
        x_val, y_val = self.__format(x_val, y_val)
        self._model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=int(self.batch_size),
            shuffle=int(self.shuffle),
            epochs=int(self.epochs),
            verbose=self.verbose,
            callbacks=self._callbacks
        )
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