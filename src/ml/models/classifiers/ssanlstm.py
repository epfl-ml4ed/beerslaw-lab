import os
import logging
import pickle
from tabnanny import check
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree, rmtree

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

class SSANLSTMModel(Model):
    """This class implements an SSAN as described in "Self-Attention: A Better Building Block for Sentiment Analysis Neural Network Classifiers"
    by Artaches Ambartsoumian Fred Popowich [https://arxiv.org/pdf/1812.07860.pdf]
    with an LSTM layer instead of the average pooling layer.
        Notion link to the details of the implementation:
            https://www.notion.so/SSAN-Network-02086320745d45d2b7c5b803206f0cb5

    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'self-attention sentiment analysis network'
        self._notation = 'ssan'
        self._model_settings = settings['ML']['models']['classifiers']['ssan']
        self._maxlen = self._settings['data']['adjuster']['limit']
        self._fold = 0
        
    def _set_seed(self):
        print(self._model_settings)
        # seed(self._model_settings['seed'])
        tf.random.set_seed(self._model_settings['seed'])

    def _format(self, x:list, y:list) -> Tuple[list, list]:
        #y needs to be one hot encoded
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        y_vector = to_categorical(y, num_classes=self._n_classes)
        return x_vector, y_vector
    
    def _format_features(self, x:list) -> list:
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        return x_vector
    
    def _get_csvlogger_path(self) -> str:
        csv_path = '../experiments/{}{}/{}/logger/ssan/'.format(self._experiment_root, self._experiment_name, self._outer_fold)
        csv_path += 'seed{}_key{}_value{}_query{}_poolsize{}_stride{}_padding{}'.format(
            self._model_settings['seed'], self._model_settings['key_cells'], self._model_settings['value_cells'], self._model_settings['query_cells'],
            self._model_settings['pool_size'],
            self._model_settings['stride'], self._model_settings['padding']
        )
        csv_path += '_dropout{}_optim{}_loss{}_bs{}_ep{}'.format(
            self._model_settings['dropout'], self._model_settings['optimiser'], self._model_settings['loss'],
            self._model_settings['batch_size'], self._model_settings['epochs']
        )
        os.makedirs(csv_path, exist_ok=True)
        checkpoint_path = csv_path + '/f{}_model_checkpoint/'.format(self._gs_fold)
        csv_path += '/f{}_model_training.csv'.format(self._gs_fold)
        return csv_path, checkpoint_path

    def _get_model_checkpoint_path(self) -> str:
        path = '../experiments/{}{}/{}/logger/ssan/'.format(self._experiment_root, self._experiment_name, self._outer_fold)
        path += 'seed{}_key{}_value{}_query{}_poolsize{}_stride{}_padding{}'.format(
            self._model_settings['seed'], self._model_settings['key_cells'], self._model_settings['value_cells'], self._model_settings['query_cells'],
            self._model_settings['pool_size'],
            self._model_settings['stride'], self._model_settings['padding']
        )
        path += '_dropout{}_optim{}_loss{}_bs{}_ep{}'.format(
            self._model_settings['dropout'], self._model_settings['optimiser'], self._model_settings['loss'],
            self._model_settings['batch_size'], self._model_settings['epochs']
        )
        path += '/f{}_model_checkpoint/'.format(self._gs_fold)
        return path

    def _init_model(self, x:np.array):
        self._set_seed()

        # initial layers
        input_layer = layers.Input(shape=(x.shape[1], x.shape[2]), name='input')
        full_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_prior')(input_layer)

        # Creating Key, Value, Query
        key_layer = layers.Dense(self._model_settings['key_cells'])(full_features)
        value_layer = layers.Dense(self._model_settings['value_cells'])(full_features)
        query_layer = layers.Dense(self._model_settings['query_cells'])(full_features)

        # Attention layer
        attention_layer = layers.AdditiveAttention()([query_layer, value_layer, key_layer])

        # LSTM
        gru_layer = layers.GRU(units=self._model_settings['gru_cells'], return_sequences=False)(attention_layer)

        print('at: {}'.format(attention_layer.shape))
        print('gru: {}'.format(gru_layer.shape))
        # Flatten
        flatten = layers.Flatten()(gru_layer)

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

        if self._model_settings['save_best_model']:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True)
            self._callbacks.append(model_checkpoint_callback)

        print(self._model.summary())

    def load_model_weights(self, x:np.array, checkpoint_path:str):
        """Given a data point x, this function sets the model of this object

        Args:
            x ([type]): [description]

        Raises:
            NotImplementedError: [description]
        """
        x = self._format_features(x) 
        self._init_model(x)
        cce = tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy')
        auc = tf.keras.metrics.AUC(name='auc')
        self._model.compile(
            loss=['categorical_crossentropy'], optimizer='adam', metrics=[cce, auc]
        )
        # print('pre-weight check: {}'.format(self._model.layers[2].weights[0][0]))
        checkpoint = tf.train.Checkpoint(self._model)
        temporary_path = '../experiments/temp_checkpoints/training/'
        if os.path.exists(temporary_path):
            rmtree(temporary_path)
        copytree(checkpoint_path, temporary_path, dirs_exist_ok=True)
        checkpoint.restore(temporary_path)
        # print('post-weight check: {}'.format(self._model.layers[2].weights[0][0]))

        
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
        self._best_epochs = np.argmax(self._history.history['val_auc'])
        print('best epoch: {}'.format(self._best_epochs))

        if self._model_settings['save_best_model']:
            checkpoint_path = self._get_model_checkpoint_path()
            self.load_model_weights(x_train, checkpoint_path)

        
    def predict(self, x:list) -> list:
        print('hello')
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
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._notation + '/'
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        self._model = path
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/lstm_history.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self._history.history, fp)
        return path
    
    def get_path(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._notation + '/'
        return path
            
    def save_fold(self, fold: int) -> str:
        path = '../experiments/' + self._experiment_root + '/' + self._experiment_name + '/models/' + self._notation + '_f' + str(fold) + '/'
        os.makedirs(path, exist_ok=True)
        self._model.save(path)
        return path
    
    
    
    
    
    
