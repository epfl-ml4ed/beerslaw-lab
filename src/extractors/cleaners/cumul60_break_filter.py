import re
import logging
import numpy as np 
import pandas as pd
from typing import Tuple 

from extractors.cleaners.break_filter import BreakFilter
from extractors.sequencer.sequencing import Sequencing

class Cumul60BreakFilter(BreakFilter):
    """Only retains the 40% longest breaks.

    Args:
        BreakFilter (BreakFilter): inherits from the breakfilter class
    """
    def __init__(self, sequencer: Sequencing):
        super().__init__(sequencer)
        self._name = 'cumul60 break filter' 
        self._notation = 'cumul60br'
        
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
                sequence.append('break')
                ends.append(begin[i+1])
                
        begins.append(begin[-1])
        sequence.append(labels[-1])
        ends.append(end[-1])
        
        return sequence, begins, ends

                
            
        
        
        
