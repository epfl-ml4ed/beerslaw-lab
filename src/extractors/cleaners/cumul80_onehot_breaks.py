import re
import logging
import numpy as np 
import pandas as pd
from typing import Tuple 

from extractors.cleaners.break_filter import BreakFilter
from extractors.sequencer.sequencing import Sequencing

class Cumul80OneHotBreakFilter(BreakFilter):
    """Only retains the 20% longest breaks.

    Args:
        BreakFilter (BreakFilter): inherits from the breakfilter class
    """
    def __init__(self, sequencer: Sequencing):
        super().__init__(sequencer)
        self._name = 'cumul80 one hot break filter' 
        self._notation = 'cumul801hotbr'
        
    def inpute_all_breaks(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]: 
        # Compute the threshold
        breaks = self._get_all_breaks(begin, end)
        breaks.sort()
        threshold = int(0.6 * len(breaks))
        threshold = breaks[threshold]
        
        begins = []
        ends = []
        sequence = []
        for i in range(len(labels) - 1):
            begins.append(begin[i])
            sequence.append(labels[i])
            ends.append(end[i])
            
            if begin[i+1] - end[i] > threshold:
                begins.append(end[i])
                break_vec = np.zeros(self._sequencer.get_vector_size())
                break_vec[0:self._sequencer.get_vector_states()] = labels[i][0:self._sequencer.get_vector_states()]
                break_vec[self._sequencer.get_break_state()] = 1
                labels.append(list(break_vec))
                ends.append(begin[i+1])
                
        begins.append(begin[-1])
        sequence.append(labels[-1])
        ends.append(end[-1])
        
        return sequence, begins, ends

                
            
        
        
        
