import numpy as np
import pandas as pd
from typing import Tuple

from extractors.lengths.adjuster import Adjuster

class TimestepCropper(Adjuster):
    """This class crops the sequence after n actions, including breaks.

    Args:
        Adjuster (Adjuster): Inherits from the Adjuster class
    """
    
    def __init__(self):
        super().__init__()
        self._name = 'timestep cropper'
        self._notation = 'tscrp'
        
    def adjust_sequence(self, labels: list, begin: list, end: list, timesteps: float) -> Tuple[list, list, list]:
        if timesteps <= len(labels):
            sequence = [x for x in labels[:timesteps]]
            b = [x for x in begin[:timesteps]]
            e = [x for x in end[:timesteps]]
        else:
            sequence = [x for x in labels]
            b = [x for x in begin]
            e = [x for x in end]
        
        return sequence, b, e 
    
        
        
    