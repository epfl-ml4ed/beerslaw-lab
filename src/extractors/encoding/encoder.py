import numpy as np
import pandas as pd

from extractors.sequencer.sequencing import Sequencing

class Encoder:
    """This class turns the sequences of events into machine readable code
    """
    
    def __init__(self, sequencer: Sequencing, settings: dict):
        self._name = 'encoder'
        self._notation = 'enc'
        self._settings = dict(settings)
        self._states = sequencer.get_states()
        self._n_states = len(self._states)
        self._create_map()
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
    
    def get_state_index(self):
        return self._state_index
    
    def get_index_state(self):
        return self._index_state
    
    def _create_map(self):
        state_index = {}
        index_state = {}
        
        for i, state in enumerate(self._states):
            state_index[state] = i
            index_state[i] = state
            
        self._state_index = state_index 
        self._index_state = index_state
        
    def encode_sequence(self, labels: list, begins:list, end:list) -> list:
        raise NotImplementedError