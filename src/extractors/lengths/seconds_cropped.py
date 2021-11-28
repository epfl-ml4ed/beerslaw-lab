import numpy as np
import pandas as pd
from typing import Tuple

from extractors.lengths.adjuster import Adjuster 

class SecondCropper(Adjuster):
    """This class crops the sequence after n seconds, including breaks.

    Args:
        Adjuster (Adjuster): Inherits from the Adjuster class
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'seconds cropper'
        self._notation = 'scrop'
        
    def adjust_sequence(self, labels: list, begin: list, end: list, second: float) -> Tuple[list, list, list]:
        
        sequence = []
        b = []
        e = []
        
        for i in range(len(labels)):
            if begin[i] > second:
                break
            sequence.append(labels[i])
            b.append(begin[i])
            e.append(end[i])
        if e[-1] > second:
            e[-1] = second
            
        return sequence, b, e
            
        