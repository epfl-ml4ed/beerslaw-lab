import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree

from ml.models.model import Model
from extractors.sequencer.sequencing import Sequencing
from extractors.pipeline_maker import PipelineMaker

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

class RNNAttentionModel(Model):
    """This class implements an LSTM
    Args:
        Model (Model): inherits from the model class

    Notion link to architecture:
        https://www.notion.so/LSTM-attention-0eafaa4c7dbb428dba7f245b7f079f3e
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'long short term memory'
        self._notation = 'rnnatt'
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

    def _format_prior_features(self, x):
        priors = x[:, :, :self._prior_states]
        features = x[:, :, self._prior_states:]
        return priors, features
    
    def _get_rnn_layer(self, return_sequences:bool, l:int):
        n_cells = self._model_settings['n_cells'][l]
        if self._model_settings['cell_type'] == 'LSTM':
            layer = layers.LSTM(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'GRU':
            layer = layers.GRU(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'RNN':
            layer = layers.SimpleRNN(units=n_cells, return_sequences=return_sequences)
        elif self._model_settings['cell_type'] == 'BiLSTM':
            layer = layers.LSTM(units=n_cells, return_sequences=return_sequences)
            layer = layers.Bidirectional(layer=layer)
        return layer

    def _get_csvlogger_path(self) -> str:
        csv_path = '../experiments/{}{}/{}/logger/{}/'.format(self._experiment_root, self._experiment_name, self._outer_fold, self._notation)
        csv_path += 'ct{}_nlayers{}_ncells{}_flatten{}'.format(
            self._model_settings['cell_type'], self._model_settings['n_layers'], self._model_settings['n_cells'], self._model_settings['flatten']
        )
        csv_path += '_drop{}_optim{}_loss{}_bs{}_ep{}'.format(
            self._model_settings['dropout'], self._model_settings['optimiser'], self._model_settings['loss'], self._model_settings['batch_size'], self._model_settings['epochs']
        )
        os.makedirs(csv_path, exist_ok=True)
        checkpoint_path = csv_path + '/f{}_model_checkpoint'.format(self._gs_fold)
        csv_path += '/f' + str(self._gs_fold) + '_model_training.csv'
        return csv_path, checkpoint_path

    def _get_model_checkpoint_path(self) -> str:
        path = '../experiments/{}{}/{}/logger/{}/'.format(self._experiment_root, self._experiment_name, self._outer_fold, self._notation)
        path += 'ct{}_nlayers{}_ncells{}_flatten{}'.format(
            self._model_settings['cell_type'], self._model_settings['n_layers'], self._model_settings['n_cells'], self._model_settings['flatten']
        )
        path += '_drop{}_optim{}_loss{}_bs{}_ep{}'.format(
            self._model_settings['dropout'], self._model_settings['optimiser'], self._model_settings['loss'], self._model_settings['batch_size'], self._model_settings['epochs']
        )
        path += '/f{}_model_checkpoint'.format(self._gs_fold)
        return path

    def _retrieve_attentionlayer(self):
        return self._model.layers[4]

    def load_model_weights(self, x: np.array, checkpoint_path: str):
        return self.load_priormodel_weights(x, checkpoint_path)

    def _init_model(self, x:np.array):
        self._set_seed()
        input_layer = layers.Input(shape=(x.shape[1], x.shape[2]), name='input_prior')
        full_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_prior')(input_layer)

        for l in range(int(self._model_settings['n_layers']) -1):
            full_features = self._get_rnn_layer(return_sequences=True, l=l)(full_features)
        full_features = self._get_rnn_layer(return_sequences=True, l=self._model_settings['n_layers'] - 1)(full_features)

        if self._model_settings['dropout'] != 0.0:
            full_features = layers.Dropout(self._model_settings['dropout'])(full_features)

        selfattention_features = layers.AdditiveAttention(use_scale=True, dropout=0.05, causal=True)([full_features, full_features])
        
        # Flatten
        if self._model_settings['flatten'] == 'flat':
            flatten = layers.Flatten()(selfattention_features)
        elif self._model_settings['flatten'] == 'average':
            flatten = layers.AveragePooling1D(pool_size=self._model_settings['n_cells'][-1], data_format='channels_first')(selfattention_features)
            flatten = layers.Flatten()(flatten)

        classification_layer = layers.Dense(self._settings['experiment']['n_classes'], activation='softmax')(flatten)

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

    def load_checkpoints(self, checkpoint_path:str, x:list):
        """Sets the inner model back to the weigths present in the checkpoint folder.
        Checkpoint folder is in the format "../xxxx_model_checkpoint/ and contains an asset folder,
        a variables folder, and index and data checkpoint files.

        Args:
            checpoint_path (str): path to the checkpoint folder
            x (list): partial sample of data, to format the layers
        """
        x = self._format_features(x) 
        self._init_model(x)
        self._model.load_weights(checkpoint_path)

        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)

        self._init_model(x_train)
        self._history = self._model.fit(
            x_train,
            y_train,
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
    
    
    
    
    
    
    
