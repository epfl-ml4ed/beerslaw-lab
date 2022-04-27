import os 
import yaml
import pickle

import numpy as np 

class Concatenator:
    """
    In the case of multiple tasks during the same lab experiment, it might be useful to concatenate timelines from the 
    different activities
    """

    def __init__(self, path:str, tasks:list, limit:int):
        self._name = 'concatenator'
        self._notation = 'concat'

        self._path = path
        self._tasks = tasks
        self._limit = limit

    def concatenate(self):
        raise NotImplementedError