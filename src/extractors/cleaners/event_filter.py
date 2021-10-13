import numpy as np 
import pandas as pd
from typing import Tuple

from extractors.sequencer.sequencing import Sequencing

class EventFilter: 
    """This class filters events when not all of them need to be included, such as cutting of self transitions, ...
    """
    
    def __init__(self):
        self._name = 'event filter'
        self._notation = 'evfilt'
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
        
    def filter_events(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]:
        raise NotImplementedError