import numpy as np
import pandas as pd

from extractors.sequencer.sequencing import Sequencing
from extractors.encoding.encoder import Encoder

class RawEncoder(Encoder):
    """This class turns the sequence of events into a sequence of the one hot encoding vectors of each event
    """
    
    def __init__(self, sequencer: Sequencing, settings: dict):
        super().__init__(sequencer, settings)
        self._name = 'raw encoder'
        self._notation = 'raw'
        
    def encode_sequence(self, labels: list, begins: list, end: list) -> list:
        return [l for l in labels]