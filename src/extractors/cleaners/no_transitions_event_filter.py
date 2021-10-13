import numpy as np
import pandas as pd
from typing import Tuple

from extractors.cleaners.event_filter import EventFilter

class NoTransitionFilters(EventFilter):
    """Here, no two similar events should follow each other. So far, this has been used to train markov chains

    Args:
        EventFilter (EventFilter): Is used to filter the events of a filter
    """
    
    def __init__(self): 
        super().__init__()
        self._name = 'no transition filter'
        self._notation = 'notrans'
        
    def filter_events(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]:
        sequence = []
        b = []
        e = []
        
        sequence.append(labels[0])
        b.append(begin[0])
        e.append(end[0])
        
        for i in range(1, len(labels)):
            if sequence[-1] == labels[i]:
                continue
            else:
                sequence.append(labels[i])
                b.append(begin[i])
                e.append(end[i])
                
        return sequence, b, e
        
