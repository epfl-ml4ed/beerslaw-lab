import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

from extractors.sequencer.sequencing import Sequencing
from extractors.encoding.encoder import Encoder

class SkipgramEncoder(Encoder):
    """This class turns each event in the sequence by its corresponding row in the skipgram embedding matrix
    """
    
    def __init__(self, sequencer: Sequencing, settings: dict):
        super().__init__(sequencer, settings)
        self._name = 'skipgram encoder'
        self._notation = 'sgenc'
        self._get_embedding_matrix()
        
    def _get_embedding_matrix(self):
        print('WEIGHTS', self._settings['data']['pipeline']['skipgram_weights'])
        weights = self._settings['data']['pipeline']['skipgram_weights']
        self.w1 = tf.keras.models.load_model(weights)
        self.w1 = self.w1.layers[0].get_weights()[0]
        self.w1 = np.vstack((self.w1, np.zeros(self.w1.shape[1]))) # In case of padding
        
        with open(self._settings['data']['pipeline']['skipgram_map'] + 'index_state.pkl', 'rb') as fp:
            self._index_state = pickle.load(fp)
        with open(self._settings['data']['pipeline']['skipgram_map'] + 'state_index.pkl', 'rb') as fp:
            self._state_index = pickle.load(fp)
        
        
    def encode_sequence(self, labels: list, begins: list, end: list) -> list:
        sequence = [self._state_index[x] for x in labels]
        sequence = [list(self.w1[x]) for x in sequence]
        return sequence