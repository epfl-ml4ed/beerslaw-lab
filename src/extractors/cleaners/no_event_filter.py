import numpy as np
import pandas as pd
from typing import Tuple

from extractors.cleaners.event_filter import EventFilter

class NoEventFilter(EventFilter):
    """We return the sequence as is

    Args:
        EventFilter (EventFilter]): inherits from EventFilter
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'no filter event'
        self._notation = 'nofilt'
        
    def filter_events(self, labels: list, begin: list, end: list) -> Tuple[list, list, list]:
        return labels, begin, end