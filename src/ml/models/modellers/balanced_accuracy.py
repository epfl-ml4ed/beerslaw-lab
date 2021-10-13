import tensorflow as tf
from tensorflow import keras
import logging
import numpy as np

from sklearn.metrics import balanced_accuracy_score


class BalAccScore(keras.callbacks.Callback):

    def __init__(self, validation_data=None):
        super(BalAccScore, self).__init__()
        self.validation_data = validation_data
        
    def set_states(self, n_states):
        self.n_states = n_states
    
    def one_hot(self, index):
        zeros = list(np.zeros(self.n_states))
        zeros[index] = 1
        return zeros
    
    def format_y(self, n_states):
        self.set_states(n_states)
        self.y = [self.one_hot(yy) for yy in self.validation_data[1]]
        
    def on_train_begin(self, logs={}):
      self.balanced_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        y_predict = tf.argmax(self.model.predict(self.validation_data[0]), axis=1)
        y_true = tf.argmax(self.y, axis=1)
        balacc = balanced_accuracy_score(y_true, y_predict)
        self.balanced_accuracy.append(round(balacc,6))
        logs["val_bal_acc"] = balacc
        keys = list(logs.keys())

        logging.info("\n ------ validation balanced accuracy score: %f ------\n" %balacc)