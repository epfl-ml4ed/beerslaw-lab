import logging
import numpy as np 
import pandas as pd 
from typing import Tuple

from extractors.sequencer.sequencing import Sequencing

class BreakFilter:
    """This class creates the breaks and inputes them into the sequences
    """
    def __init__(self, sequencer: Sequencing, break_threshold: float):
        self._name = 'break filter'
        self._notation = 'brfilt'
        self._sequencer = sequencer
        self._break_threshold = break_threshold
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
        
    def _get_all_breaks(self, begin: list, end: list):
        b = begin + [0]
        e = [0] + end 
        
        breaks = list(np.array(b) - np.array(e))
        breaks = breaks[:-1]
        
        breaks = [b for b in breaks if b > 0]
        logging.info('cur: {}'.format(breaks))
        
        return breaks

    def get_threshold(self, begins:list, ends:list, threshold:float):
        begin = [b for b in begins]
        end = [e for e in ends]
        breaks = self._get_all_breaks(begin, end)
        if len(breaks) == 0:
            return 0
        breaks.sort()
        threshold = int(np.floor(self._break_threshold * len(breaks)))
        threshold = breaks[threshold]
        return threshold
        
    def inpute_all_breaks(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]:
        raise NotImplementedError
        
        
        
    
    