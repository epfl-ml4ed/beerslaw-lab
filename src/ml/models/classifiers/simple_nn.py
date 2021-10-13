import os
import shutil
import logging
import numpy as np
import pandas as pd
from typing import Tuple

import tensorflow as tf

import logging

from ml.models.model import Model



class SimpleNN(Model):
    """This class implements a random forest classifier
    Args:
        Model (Model): inherits from the model class
    TODO: Check format (np.asaray)
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = '1-layer neural network'
        self._notation = '1nn'
        self._model_settings = settings['ML']['models']['classifiers']['1nn']
        
    def _one_hot(self, target:int) -> list:
        zeros = np.zeros(self._n_classes)       
        zeros[target] = 1
        return list(zeros)
    
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        # y = [self._one_hot(yy) for yy in y]
        return np.array(x, dtype=np.float), np.array(y, dtype=np.float)
    
    def _format_features(self, x:list) -> list:
        return np.array(x, dtype=np.float)
    
    def _init_model(self):
        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Dense(self._model_settings['dense_units']))
        self._model.add(tf.keras.layers.Dropout(self._model_settings['dropout']))
        self._model.add(tf.keras.layers.Dense(self._n_classes, activation='softmax'))
        
        self._model.compile(
            loss=['sparse_categorical_crossentropy'], optimizer='adam', metrics=self._model_settings['metric_reporting']
        )
        
        self._callbacks = []
        if self._model_settings['early_stopping']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, min_delta=0.001, 
                restore_best_weights=True
            )
            self._callbacks.append(early_stopping)
        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)
        
        logging.debug('x_train shape: {}, y_train shape: {}'.format(x_train.shape, y_train.shape))
        logging.debug('x_val shape: {}, y_val shape: {}'.format(x_val.shape, y_val.shape))
        logging.debug('n_ classes: {}'.format(self._n_classes))
        
        self._init_model()
        
        self._model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=int(self._model_settings['batch_size']),
            shuffle=int(self._model_settings['batch_size']),
            epochs=int(self._model_settings['epochs']),
            verbose=self._model_settings['verbose'],
            callbacks=self._callbacks
        )
        
    def predict(self, x:list) -> list:
        x_predict = self._format_features(x)
        predictions = self._model.predict(x_predict)
        predictions = [np.argmax(x) for x in predictions]
        return predictions
    
    def predict_proba(self, x:list) -> list:
        x_predict = self._format_features(x)
        probs = self._model.predict(x_predict)
        if len(probs[0]) != self._n_classes:
            preds = self._model.predict(x_predict)
            probs = self._inpute_full_prob_vector(preds, probs)
        return probs
    
    def save(self):
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '_l' + self._settings['data']['adjuster']['limit'] + '_f' + str(self._fold) + '/'
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        return path
    
    def get_path(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '_l' + str(self._settings['data']['adjuster']['limit']) + '_f' + str(fold) + '/'
        return path 
    
    def save_fold(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '_l' + str(self._settings['data']['adjuster']['limit']) + '_f' + str(fold) + '/'
        os.makedirs(path, exist_ok=True)
        
        temp_path = './temp_saving_weights/'
        os.makedirs(temp_path, exist_ok=True)
        self._model.save(temp_path)
        shutil.move(temp_path, path)
        return path 