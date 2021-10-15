import os
import dill
import logging
import pickle
import numpy as np
import pandas as pd 
from typing import Tuple

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import OneHot

from ml.models.model import Model
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger

from ml.models.modellers.balanced_accuracy import BalAccScore
from sklearn.metrics import balanced_accuracy_score


class PWSkipgram(Model):
    """This class implements a pairwise skipgram model
    Args:
        Model (Model): inherits from the model class
        BaseEstimator (BaseEstimator): inherits from the BaseEstimator to use the gridsearch object from sklearn
        ClassifierMixin: inherits from the ClassifierMixin class from sklearn 
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'pairwise-skipgram'
        self._notation = 'pwsg'
        self._model_settings = settings['ML']['models']['modellers']['skipgram']
        
    def _one_cold(self, vector:list) -> int:
        return vector.index(1)
    
    def _one_hot(self, index:int) -> list:
        zeros = list(np.zeros(self._model_settings['n_states']))
        print(index)
        zeros[index] = 1
        return zeros
    
    
    def _format_data(self, x:list) -> Tuple[list, list]:
        xx = []
        yy = []
        for student in x:
            for i, onehot in enumerate(student):
                min_index = max(i - self._model_settings['window_size'], 0)
                max_index = min(i + self._model_settings['window_size'], len(student)) + 1
                for context in student[min_index:max_index]:
                    xx.append(onehot)
                    yy.append(self._one_cold(context))
        return xx, yy
    
    def _balanced_accuracy(self, y_true, y_pred):
        y_true = y_true.numpy().tolist()
        y_pred = y_pred.numpy().tolist()
        y_true = [int(yy[0]) for yy in y_true]
        y_pred = [int(np.argmax(yy)) for yy in y_pred]
        balacc = balanced_accuracy_score(y_true, y_pred)
        return balacc
    
    def  _init_model(self):
        print(self._model_settings)
        
        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Embedding(self._model_settings['n_states'], self._model_settings['embeddings'], input_length=self._model_settings['n_states']))
        self._model.add(tf.keras.layers.Flatten())
        self._model.add(tf.keras.layers.Dense(units=self._model_settings['n_states'], activation='softmax', use_bias=False))
        
        if 'balacc' in self._settings['ML']['scorers']['scoring_metrics']:
            callback_path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/balanced_accuracy'
            mc = tf.keras.callbacks.ModelCheckpoint(filepath=callback_path, monitor="val_bal_acc", verbose=1, save_best_only=True, save_freq='epoch')

        metrics = ['accuracy']
        if 'balacc' in self._settings['ML']['scorers']['scoring_metrics']:
            metrics.append(self._balanced_accuracy)
        self._model.compile(
            loss=['sparse_categorical_crossentropy'],
            optimizer=self._model_settings['optimiser'],
            metrics=metrics, 
            run_eagerly=True 
        )
        
        self._callbacks = []
        if self._model_settings['early_stopping']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, min_delta=0.001, 
                restore_best_weights=True
            )
            self._callbacks.append(early_stopping)
        if 'balacc' in self._settings['ML']['scorers']['scoring_metrics']:
            self._callbacks.append(mc)
            
        csv_path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/model_training.csv'
        csv_logger = CSVLogger(csv_path, append=True, separator=';')
        self._callbacks.append(csv_logger)
            
        self._model.summary()
        
    def fit(self, x_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format_data(x_train)
        x_val, y_val = self._format_data(x_val)
        self._xval, self._yval = x_val, y_val
        
        print(np.array(x_val).shape)
        print(np.array(x_train).shape)        
        self._init_model()
        
        
        self._history = self._model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=self._model_settings['batch_size'],
            shuffle=self._model_settings['shuffle'],
            epochs=self._model_settings['epochs'],
            verbose=self._model_settings['verbose'],
            callbacks=self._callbacks
        )
        
        self.params = self._model.layers[0].get_weights()[0]
        return self
    
    def get_window_size(self):
        return self._model_settings['window_size']
        
    def predict(self, x:list) -> list:
        x, y = self._format_data(x)
        predictions = self._model.predict(x)
        predictions = [np.argmax(x) for x in predictions]
        return predictions, y
    
    def predict_proba(self, x:list) -> list:
        x, y = self._format_data(x)
        return self._model.predict(x), y
    
    def save(self) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '/'
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        self._model = path
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/skipgram_history.pkl'
        with open(path, 'wb') as fp:
            dill.dump(self._history, fp)
        return path
    
    def get_path(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '/'
        return path
        
    def save_fold(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '_f' + str(fold) + '/'
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        return path