import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from shutil import copytree, rmtree 

from ml.models.model import Model
from tensorflow.keras import Model as Mod

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


from numpy.random import seed

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
        
    def _set_seed(self):
        # print(self._model_settings)
        # seed(self._model_settings['seed'])
        tf.random.set_seed(self._model_settings['seed'])

    def _format(self, x:list, y:list) -> Tuple[list, list]:
        #y needs to be one hot encoded
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        y_vector = to_categorical(y, num_classes=self._n_classes)
        return x_vector, y_vector
    
    def _format_features(self, x:list) -> list:
        # print(np.array(x).shape)
        # print(self._model_settings['padding_value'], self._maxlen)
        x_vector = pad_sequences(x, padding="post", value=self._model_settings['padding_value'], maxlen=self._maxlen, dtype=float)
        return x_vector
    
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
        csv_path = '../experiments/' + self._experiment_root + self._experiment_name + '/'
        csv_path += str(self._outer_fold) + '/logger/' 
        csv_path += 'ct' + self._model_settings['cell_type'] + '_nlayers' + str(self._model_settings['n_layers'])
        csv_path += '_ncells' + str(self._model_settings['n_cells']) + '_drop' + str(self._model_settings['dropout']).replace('.', '')
        csv_path += '_optim' + self._model_settings['optimiser'] + '_loss' + self._model_settings['loss']
        csv_path += '_bs' + str(self._model_settings['batch_size']) + '_ep' + str(self._model_settings['epochs']) + '_seed' + str(self._model_settings['seed'])
        csv_path += self._notation 
        # with open(csv_path + '/architecture.pkl', 'wb') as fp:
        #     pickle.dump(self._model_settings, fp)

        os.makedirs(csv_path, exist_ok=True)
        checkpoint_path = csv_path + '/f' + str(self._gs_fold) + '_model_checkpoint/'
        csv_path += '/f' + str(self._gs_fold) + '_model_training.csv'
        return csv_path, checkpoint_path



    def _get_model_checkpoint_path(self) -> str:
        path = '../experiments/' + self._experiment_root + self._experiment_name + '/'
        path += str(self._outer_fold) + '/logger/'
        path += 'ct' + self._model_settings['cell_type'] + '_nlayers' + str(self._model_settings['n_layers'])
        path += '_ncells' + str(self._model_settings['n_cells']) + '_drop' + str(self._model_settings['dropout']).replace('.', '')
        path += '_optim' + self._model_settings['optimiser'] + '_loss' + self._model_settings['loss']
        path += '_bs' + str(self._model_settings['batch_size']) + '_ep' + str(self._model_settings['epochs']) + '_seed' + str(self._model_settings['seed'])
        path += self._notation
        path += '/f' + str(self._gs_fold) + '_model_checkpoint/'
        return path

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
        print('pre-weight check: {}'.format(self._model.layers[2].weights[0][0]))
        checkpoint = tf.train.Checkpoint(self._model)
        temporary_path = '../experiments/temp_checkpoints/training/'
        if os.path.exists(temporary_path):
            rmtree(temporary_path)
        copytree(checkpoint_path, temporary_path, dirs_exist_ok=True)
        checkpoint.restore(temporary_path)
        print('post-weight check: {}'.format(self._model.layers[2].weights[0][0]))

    def _init_model(self, x:np.array):
        # initial layers
        self._set_seed()

        input_feature = layers.Input(shape=(x.shape[1], x.shape[2]), name='input_features')
        full_features = layers.Masking(mask_value=self._model_settings['padding_value'], name='masking_features')(input_feature)

        for l in range(int(self._model_settings['n_layers']) -1):
            full_features = self._get_rnn_layer(return_sequences=True, l=l)(full_features)
        full_features = self._get_rnn_layer(return_sequences=False, l=self._model_settings['n_layers'] - 1)(full_features)

        if self._model_settings['dropout'] != 0.0:
            full_features = layers.Dropout(self._model_settings['dropout'])(full_features)

        classification_layer = layers.Dense(self._settings['experiment']['n_classes'], activation='softmax')(full_features)

        self._model = Mod(input_feature, classification_layer)


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
            x_train, y_train,
            validation_data=(x_val, y_val),
            batch_size=self._model_settings['batch_size'],
            shuffle=self._model_settings['shuffle'],
            epochs=self._model_settings['epochs'],
            verbose=self._model_settings['verbose'],
            callbacks=self._callbacks
        )

        if self._model_settings['save_best_model']:
            checkpoint_path = self._get_model_checkpoint_path()
            self.load_model_weights(x_train, checkpoint_path)
            self._best_epochs = np.argmax(self._history.history['val_auc'])
            print('best epoch: {}'.format(self._best_epochs))
        self._fold += 1
        
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