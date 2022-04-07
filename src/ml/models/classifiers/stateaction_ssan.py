import os
import logging
import pickle
from tabnanny import check
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree

from ml.models.model import Model
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

class SASSANModel(Model):
    """This class implements an SSAN as described in "Self-Attention: A Better Building Block for Sentiment Analysis Neural Network Classifiers"
    by Artaches Ambartsoumian Fred Popowich [https://arxiv.org/pdf/1812.07860.pdf]
    where the states part is equivalent to the query, and the action part to the key and value

        Notion link to the details of the implementation:
            https://www.notion.so/SSAN-Network-02086320745d45d2b7c5b803206f0cb5

    Args:
        Model (Model): inherits from the model class
    """
    
    def __init__(self, settings:dict):
        super().__init__(settings)
        self._name = 'kqv self-attention sentiment analysis network'
        self._notation = 'kqvssan'
        self._model_settings = settings['ML']['models']['classifiers']['ssan']
        self._maxlen = self._settings['data']['adjuster']['limit']
        self._fold = 0

        pipeline = PipelineMaker(settings)
        sequencer = pipeline.get_sequencer()
        self._vector_states = sequencer.get_vector_states()
        
    def _format(self, x:list, y:list) -> Tuple[list, list]:
        #y needs to be one hot encoded
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        y_vector = to_categorical(y, num_classes=self._n_classes)
        return x_vector, y_vector
    
    def _format_features(self, x:list) -> list:
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        return x_vector

    def _format_state_actions_features(self, x):
        states = x[:, :, :self._vector_states]
        actions = x[:, :, self._vector_states:]
        return states, actions
    
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
        path += '/f{}_model_training.csv'.format(self._gs_fold)
        return path

    def _init_model(self, states:np.array, actions:np.array):
        self._set_seed()

        # initial layers
        input_states_layer = layers.Input(shape=(states.shape[1], states.shape[2]), name='input_states')
        input_states_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_states')(input_states_layer)

        input_actions_layer = layers.Input(shape=(actions.shape[1], actions.shape[2]), name='input_actions')
        input_actions_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_actions')(input_actions_layer)

        # Creating Key, Value, Query
        key_layer = layers.Dense(self._model_settings['key_cells'])(input_actions_features)
        value_layer = layers.Dense(self._model_settings['value_cells'])(input_actions_features)
        query_layer = layers.Dense(self._model_settings['query_cells'])(input_states_features)

        # Attention layer
        attention_layer = layers.AdditiveAttention()([query_layer, value_layer, key_layer])

        # Average Pooling layer
        pooled = layers.AveragePooling1D(
            pool_size=self._model_settings['pool_size'],
            strides=self._model_settings['stride'],
            padding=self._model_settings['padding']
        )(attention_layer)

        # Flatten
        flatten = layers.Flatten()(pooled)

        # dropout
        if self._model_settings['dropout'] != 0.0:
            flatten = layers.Dropout(self._model_settings['dropout'])(flatten)

        # output layer
        classification_layer = layers.Dense(self._settings['experiment']['n_classes'], activation='softmax')(flatten)
        
        # Model init
        self._model = Mod([input_states_layer, input_actions_layer], classification_layer)

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
        x_states, x_actions = self._format_state_actions_features(x)
        self._init_model(x_states, x_actions)
        self._model.load_weights(checkpoint_path)

        
    def fit(self, x_train:list, y_train:list, x_val:list, y_val:list):
        x_train, y_train = self._format(x_train, y_train)
        x_val, y_val = self._format(x_val, y_val)

        train_states, train_actions = self._format_state_actions_features(x_train)
        validation_states, validation_actions = self._format_state_actions_features(x_val)

        self._init_model(train_states, train_actions)
        self._history = self._model.fit(
            [train_states, train_actions], y_train,
            validation_data=([validation_states, validation_actions], y_val),
            batch_size=self._model_settings['batch_size'],
            shuffle=self._model_settings['shuffle'],
            epochs=self._model_settings['epochs'],
            verbose=self._model_settings['verbose'],
            callbacks=self._callbacks
        )
        self._fold += 1
        
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
    
    
    
    
    
    
    
