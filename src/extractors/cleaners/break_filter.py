import logging
import numpy as np 
import pandas as pd 
from typing import Tuple

from extractors.sequencer.sequencing import Sequencing

class BreakFilter:
    """This class creates the breaks and inputes them into the sequences
    """
    def __init__(self, sequencer: Sequencing):
        self._name = 'break filter'
        self._notation = 'brfilt'
        self._sequencer = sequencer
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
        
    def _get_all_breaks(self, begin: list, end: list):
        b = begin + [0]
        e = [0] + end 
        
        breaks = list(np.array(b) - np.array(e))
        breaks = breaks[1:-1]
        
        breaks = [b for b in breaks if b > 0]
        logging.info('cur: {}'.format(breaks))
        
        return breaks
        
    def inpute_all_breaks(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]:
        raise NotImplementedError
        
        
        
    
    