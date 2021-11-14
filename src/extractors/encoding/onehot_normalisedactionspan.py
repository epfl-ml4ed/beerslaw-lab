import numpy as np
import pandas as pd

from extractors.sequencer.sequencing import Sequencing
from extractors.encoding.encoder import Encoder

class OneHotActionSpan(Encoder):
    """This class turns the sequence of events into a sequence of the one hot encoding vectors of each event
    """
    
    def __init__(self, sequencer: Sequencing, settings: dict):
        super().__init__(sequencer, settings)
        self._name = 'one-hotnorm action span encoder'
        self._notation = '1hotnormas'
        
    def _one_hot_vector(self, label):
        vector = np.zeros(self._n_states)
        vector[self._state_index[label]] = 1
        return list(vector)
    
    def encode_sequence(self, labels: list, begins: list, end: list) -> list:
        normalising_factor = np.sum(np.array(end) - np.array(begins))
        sequence = []
        for i, label in enumerate(labels):
            labels = list(np.array(label) * (end[i] - begins[i])) / normalising_factor
            sequence.append(labels)
            
        return sequence