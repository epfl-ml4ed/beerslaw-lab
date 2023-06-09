import numpy as np
import pandas as pd

from extractors.aggregator.aggregator import Aggregator
from sklearn.preprocessing import MinMaxScaler
from extractors.sequencer.sequencing import Sequencing

class OneHotMinMaxNormaliserAggregator(Aggregator):
    """This class leaves the sequence as is
    """
    
    def __init__(self, sequencer:Sequencing):
        super().__init__()
        self._name = 'onehotnormalised aggregator'
        self._notation = '1hotnormagg'

        self._sequencer = sequencer
        self._state_size = self._sequencer.get_vector_states()
        
    def aggregate(self, matrix: list) -> list:
        states = [mm[0:self._state_size] for mm in matrix]
        actions = [mm[self._state_size:] for mm in matrix]
        
        scaler = MinMaxScaler()
        scaler.fit(actions)
        normalised = scaler.transform(actions)
        normalised = [list(n) for n in normalised]

        assert len(states) == len(normalised)
        new_matrix = [states[i] + normalised[i] for i in range(len(normalised))]
        return new_matrix
        