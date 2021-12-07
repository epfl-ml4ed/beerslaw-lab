import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

from ml.models.model import Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import get as get_loss, Loss
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.metrics import get as get_metric, Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LSTMModel(Model):
    """This class implements an LSTM
    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'long short term memory'
        self._notation = 'lstm'
        self._model_settings = settings['ML']['models']['classifiers']['lstm']
        self._maxlen = self._settings['data']['adjuster']['limit']
        self._fold = 0
        
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        #y needs to be one hot encoded
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        y_vector = to_categorical(y, num_classes=self._n_classes)
        return x_vector, y_vector
    
    def _format_features(self, x:list) -> list:
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        return x_vector
    
    def _get_rnn_layer(self, return_sequences:bool):
        if self._model_settings['cell_type'] == 'LSTM':
            layer = layers.LSTM(units=self._model_settings['n_cells'], return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'GRU':
            layer = layers.GRU(units=self._model_settings['n_cells'], return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'RNN':
            layer = layers.SimpleRNN(units=self._model_settings['n_cells'], return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'BiLSTM':
            layer = layers.LSTM(units=self._model_settings['n_cells'], return_sequences=return_sequences)
            layer = layers.Bidirectional(layer=layer)
            
        return layer

    def _get_csvlogger_path(self) -> str:
        csv_path = '../experiments/' + self._experiment_root + self._experiment_name + '/'
        csv_path += str(self._outer_fold) + '/logger/' 
        csv_path += 'ct' + self._model_settings['cell_type'] + '_nlayers' + str(self._model_settings['n_layers'])
        csv_path += '_ncells' + str(self._model_settings['n_cells']) + '_drop' + str(self._model_settings['dropout']).replace('.', '')
        csv_path += '_optim' + self._model_settings['optimiser'] + '_loss' + self._model_settings['loss']
        csv_path += '_bs' + str(self._model_settings['batch_size']) + '_ep' + str(self._model_settings['epochs'])
        csv_path += self._name 
        os.makedirs(csv_path, exist_ok=True)
        csv_path += '/f' + str(self._gs_fold) + '_model_training.csv'
        return csv_path

    def _init_model(self, x:np.array):
        # initial layers
        self._model = keras.Sequential()
        self._model.add(layers.Input((x.shape[1], x.shape[2],)))
        self._model.add(layers.Masking(mask_value=-self._model_settings['padding_value']))
        
        # Recurrent layers
        for l in range(self._model_settings['n_layers'] - 1):
            self._model.add(self._get_rnn_layer(return_sequences=True))
        self._model.add(self._get_rnn_layer(return_sequences=False))
        
        # dropout
        if self._model_settings['dropout'] != 0.0:
            self._model.add(layers.Dropout(self._model_settings['dropout']))
            
        # output layer
        self._model.add(layers.Dense(self._settings['experiment']['n_classes'], activation="softmax"))
        
        # compiling
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        
        # callbacks
        self._callbacks = []
        if self._model_settings['early_stopping']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, min_delta=0.001, 
                restore_best_weights=True
            )
            self._callbacks.append(early_stopping)
            
        # csv loggers
        csv_path = self._get_csvlogger_path()
        csv_logger = CSVLogger(csv_path, append=True, separator=';')
        self._callbacks.append(csv_logger)
        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)

        self._init_model(x_train)
        self._history = self._model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=self._model_settings['batch_size'],
            shuffle=self._model_settings['shuffle'],
            epochs=self._model_settings['epochs'],
            verbose=self._model_settings['verbose'],
            callbacks=self._callbacks
        )
        self._fold += 1
        
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
    
    def save(self) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '/'
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        self._model = path
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/lstm_history.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self._history.history, fp)
        return path
    
    def get_path(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '/'
        return path
            
    def save_fold(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._name + '_f' + str(fold) + '/'
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        return path
    
    
    
    
    
    
