
import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator
from extractors.sequencer.sequencing import Sequencing

class TimestepNormaliser(Aggregator):
    """This class leaves the sequence as is
    """
    
    def __init__(self, sequencer:Sequencing):
        super().__init__()
        self._name = 'timestepnormalised aggregator'
        self._notation = 'tsnormagg'
        self._sequencer = sequencer
        self._state_size = self._sequencer.get_vector_states()

    def normalise_ts(self, vector):
        if np.sum(vector) == 0:
            return vector
        vec = list(np.array(vector) / np.sum(vector))
        return vec
        
    def aggregate(self, matrix: list) -> list:
        labels = []
        for row in matrix:
            state = row[0:self._state_size]
            action = self.normalise_ts(row[self._state_size:])
            labels.append(state + action)
        return labels