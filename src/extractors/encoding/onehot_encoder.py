import numpy as np
import pandas as pd

from extractors.sequencer.sequencing import Sequencing
from extractors.encoding.encoder import Encoder

class OneHotEncoder(Encoder):
    """This class turns the sequence of events into a sequence of the one hot encoding vectors of each event
    """
    
    def __init__(self, sequencer: Sequencing, settings: dict):
        super().__init__(sequencer, settings)
        self._name = 'one-hot encoder'
        self._notation = '1hot'
        
    def _one_hot_vector(self, label):
        vector = np.zeros(self._n_states)
        vector[self._state_index[label]] = 1
        return list(vector)
    
    def encode_sequence(self, labels: list, begins: list, end: list) -> list:
        sequence = []
        for label in labels:
            sequence.append(self._one_hot_vector(label))
            
        return sequence