import numpy as np 
import pandas as pd
from typing import Tuple

from extractors.cleaners.break_filter import BreakFilter
from extractors.sequencer.sequencing import Sequencing

class NoBreakFilter(BreakFilter):
    """Returns the sequence as is

    Args:
        BreakFilter (BreakFilter): inherits from the breakfilter class
    """
    def __init__(self, sequencer: Sequencing, break_threshold: float):
        super().__init__(sequencer, break_threshold)
        self._name = 'no break filter' 
        self._notation = 'nobrfilt'
        
    def inpute_all_breaks(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]: 
        return labels, begin, end

                
            
        
        
        
