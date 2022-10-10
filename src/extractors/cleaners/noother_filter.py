import numpy as np
import pandas as pd
from typing import Tuple

from extractors.cleaners.event_filter import EventFilter

class NoOtherFilter(EventFilter):
    """Here, no two similar events should follow each other. So far, this has been used to train markov chains

    Args:
        EventFilter (EventFilter): Is used to filter the events of a filter
    """
    
    def __init__(self): 
        super().__init__()
        self._name = 'no other filter'
        self._notation = 'noother'
        
    def filter_events(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]:
        indices = [i for i in range(len(labels)) if labels[i] != 'todelete']
        new_labels = [labels[idx] for idx in indices]
        new_begins = [begin[idx] for idx in indices]
        new_ends = [end[idx] for idx in indices]

        print('LABELS')
        print('todelete'in labels)
        print('todelete' in new_labels)
                
        return new_labels, new_begins, new_ends
        
