import numpy as np
import pandas as pd
from typing import Tuple

from extractors.lengths.adjuster import Adjuster

class FullSequence(Adjuster):
    """This class returns the full sequence

    Args:
        Adjuster (Adjuster): Inherits from the Adjuster class
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'full sequence'
        self._notation = 'full'
        
    def adjust_sequence(self, labels: list, begin: list, end: list, timesteps: float) -> Tuple[list, list, list]:
        sequence = [x for x in labels]
        b = [x for x in begin]
        e = [x for x in end]
        
        return sequence, b, e 
    
        
        
    