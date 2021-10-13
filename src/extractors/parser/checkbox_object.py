import time
import numpy as np
import pandas as pd
from typing import Tuple

from matplotlib import pyplot as plt
import seaborn as sns

class Checkbox():
    """Emulates objects that are either on or off
    """
    
    def __init__(self, name: str, simulation_id: str, toggled: int):
        self._toggled = toggled
        self._initial_state = toggled
        self._scaling = 1
        
        if self._toggled:
            self._switch_on = {'begin' : [0], 'end' : []}
            self._switch_off = {'begin' : [], 'end' : []}
        elif not self._toggled:
            self._switch_on = {'begin' : [], 'end' : []}
            self._switch_off = {'begin' : [0], 'end' : []}

        self._last_timestamp = -1
        
    def get_switch_on(self) -> dict:
        """Get beginning and end timestamps of when the checkbox is on
        """
        return {k:v for (k, v) in self._switch_on.items()}

    def get_switch_off(self) -> dict:
        """Get beginning and end timestamps of when the checkbox is off
        """
        return {k:v for (k, v) in self._switch_off.items()}

    def _checkValidity(self, on: list, off: list, time: float):
        if len(on) > 0:
            assert on[-1] <= time
        if len(off) > 0:
            assert off[-1] <= time

    def _check_switch(self, time: float):
        if self._toggled:
            self._checkValidity(self._switch_on['end'], self._switch_off['begin'], time)
            self._switch_on['end'].append(time)
            self._switch_off['begin'].append(time)
        else:
            self._checkValidity(self._switch_on['begin'], self._switch_off['end'], time)
            self._switch_on['begin'].append(time)
            self._switch_off['end'].append(time)
            
        self._toggled = int(np.abs(self._toggled - 1))
        
    def switch(self, state: int, time: float):
        """Switch the checkbox to the new state *state* if it is not already in this state
        Args:
            state (int): [description]
            time (float): [description]
        """
        if self._toggled != state:
            self._check_switch(time)
            
    def reset(self, time: float):
        """Resets to the original checkbox
        """
        if self._toggled != self._initial_state:
            self._check_switch(time)
        
    def close(self, timestamp: float):
        """Used to format the code when the student stops using the simulation
        """
        if len(self._switch_on['begin']) > len(self._switch_on['end']):
            self._switch_on['end'].append(timestamp)
        if len(self._switch_off['begin']) > len(self._switch_off['end']):
            self._switch_off['end'].append(timestamp)

        self._last_timestamp = timestamp
        
    def get_startend_sequences(self) -> Tuple[list, list]:
        return self._switch_on, self._switch_off
            
    def get_active_time_proportion(self) -> float:
        somme = (np.array(self._switch_on['end']) - np.array(self._switch_on['begin'])).sum()
        somme /= self._last_timestamp
        somme *= 100
        return somme
