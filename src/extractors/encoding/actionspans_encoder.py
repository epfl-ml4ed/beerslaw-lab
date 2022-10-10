import numpy as np
import pandas as pd

from extractors.sequencer.sequencing import Sequencing
from extractors.encoding.encoder import Encoder 

class ActionSpansEncoder(Encoder):
    """This class turns the sequence of events into a list comprising the amount of time in terms of seconds the students spent in each state
    """
    
    def __init__(self, sequencer: Sequencing, settings: dict):
        super().__init__(sequencer, settings)
        self._name = 'action span encoder'
        self._notation = 'actionspan'
        
    def encode_sequence(self, labels: list, begins: list, end: list) -> list:
        feature = []
        for index in self._index_state:
            indices = [x for x in range(len(labels)) if labels[x] == self._index_state[index]]
            
            if len(indices) == 0:
                feature.append(0)
            else:
                bs = [begins[x] for x in indices]
                es = [end[x] for x in indices]
                time_span = sum(np.array(es) - np.array(bs))
                feature.append(time_span)
                
        return feature