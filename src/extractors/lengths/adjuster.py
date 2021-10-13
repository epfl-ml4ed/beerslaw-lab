import numpy as np
import pandas as pd
from typing import Tuple
class Adjuster:
    """This class crops and pads sequences according to certain criterion to be specified in the subclasses
    """
    
    def __init__(self):
        self._name = 'adjuster'
        self._notation = 'adj'
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
        
    def adjust_sequence(self, labels: list, begin:list, end: list, limit: float) -> Tuple[list, list, list]:
        raise NotImplementedError