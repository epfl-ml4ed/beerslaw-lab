import os
import logging
import pickle
from tabnanny import check
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree

from ml.models.model import Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model as Mod
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import History
from tensorflow.keras.losses import get as get_loss, Loss
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.metrics import get as get_metric, Metric
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from numpy.random import seed

class LSTMCNNModel(Model):
    """This class implements an LSTM-CMM as described in "Twitter Sentiment Analysis using combined LSTM-CNN Models
    by Pedro M Sosa [https://www.academia.edu/download/55829451/sosa_sentiment_analysis.pdf]

        Notion link to the details of the implementation:
            https://www.notion.so/LSTM-CNN-a9dd39cf97d544ee849aa5c4ea98b926

    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'long short term - convolutionnal neural network memory'
        self._notation = 'lstm-cnn'
        self._model_settings = settings['ML']['models']['classifiers']['cnnlstm']
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
    
    def _get_csvlogger_path(self) -> str:
        csv_path = '../experiments/{}{}/{}/logger/lstmcnn/'.format(self._experiment_root, self._experiment_name, self._outer_fold)
        csv_path += 'seed{}_lstmcells{}_cnncells{}_cnnwindow{}_poolsize{}_stride{}_padding{}'.format(
            self._model_settings['seed'], self._model_settings['lstm_cells'], self._model_settings['cnn_cells'],
            self._model_settings['cnn_window'], self._model_settings['pool_size'], self._model_settings['stride'], self._model_settings['padding']
        )
        csv_path += '_dropout{}_optim{}_loss{}_bs{}_ep{}'.format(
            self._model_settings['dropout'], self._model_settings['optimiser'], self._model_settings['loss'],
            self._model_settings['batch_size'], self._model_settings['epochs']
        )
        os.makedirs(csv_path, exist_ok=True)
        checkpoint_path = csv_path + '/f{}_model_checkpoint/'.format(self._gs_fold)
        csv_path += '/f{}_model_training.csv'.format(self._gs_fold)
        return csv_path, checkpoint_path

    def _init_model(self, x:np.array):
        self._set_seed()

        # initial layers
        input_layer = layers.Input(shape=(x.shape[1], x.shape[2]), name='input')
        full_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_prior')(input_layer)

        # LSTM cell part - output: #datapoints x #timesteps x #ncells
        whole_interaction, memory_state, carry_state = layers.RNN(
                                                                    layers.LSTMCell(self._model_settings['lstm_cells']),
                                                                    return_sequences=True,
                                                                    return_state=True
                                                                )(full_features)
        self._memory_state = memory_state
        self._carry_state = carry_state

        # CNN Part - output: #datapoints x #timesteps-convolutional_crop x #cnn_cells
        cnnd = layers.Conv1D(
            self._model_settings['cnn_cells'],
            self._model_settings['cnn_window'],
            activation='relu',
            input_shape=x[1:]
        )(whole_interaction)

        # Maxpooling 
        pooled = layers.MaxPooling1D(
            pool_size=self._model_settings['pool_size'],
            strides=self._model_settings['stride'],
            padding=self._model_settings['padding']
        )(cnnd)
        
        # Flatten
        if self._model_settings['flatten'] == 'flat':
            flatten = layers.Flatten()(pooled)
        elif self._model_settings['flatten'] == 'average':
            flatten = layers.AveragePooling1D(pool_size=self._model_settings['lstm_cells'], data_format='channels_first')(pooled)
            flatten = layers.Flatten()(flatten)

        # dropout
        if self._model_settings['dropout'] != 0.0:
            flatten = layers.Dropout(self._model_settings['dropout'])(flatten)

        # output layer
        classification_layer = layers.Dense(self._settings['experiment']['n_classes'], activation='softmax')(flatten)
        
        # Model init
        self._model = Mod(input_layer, classification_layer)

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
        csv_path, checkpoint_path = self._get_csvlogger_path()
        csv_logger = CSVLogger(csv_path, append=True, separator=';')
        self._callbacks.append(csv_logger)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_auc',
        mode='max',
        save_best_only=True)
        self._callbacks.append(model_checkpoint_callback)

        print(self._model.summary())

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
        if self._model_settings['save_best_model']:
            checkpoint_path = self._get_model_checkpoint_path()
            self.load_model_weights(x_train, checkpoint_path)
            self._best_epochs = np.argmax(self._history.history['val_auc'])
            print('best epoch: {}'.format(self._best_epochs))
        
    def predict(self, x:list) -> list:
        self.predict_tensorflow(x)
    
    def predict_proba(self, x:list) -> list:
        self.predict_proba_tensorflow(x)
    
    def save(self) -> str:
        self.save_tensorflow()
    
    def get_path(self, fold: int) -> str:
        self.get_path(fold)
            
    def save_fold(self, fold: int) -> str:
        self.save_fold_tensorflow(fold)

    def save_fold_early(self, fold: int) -> str:
        return self.save_fold_early_tensorflow(fold)
    
    
    
    
    
    
