import time
import numpy as np
import pandas as pd
from typing import Tuple

from matplotlib import pyplot as plt
import seaborn as sns

class SimCharacteristics():
    """Summarise the values of the characteristics of the simulation such as the dependent and independent variables of the system.
    Particularly used when tracking the values that may have an influence into an equation. 
    """
    def __init__(self, name: str, simulation_id: str, state: int):
        self._state = state
        self._initial_state = state
        self._scaling = 1
        
        self._values = [state]
        self._timesteps = [0]
        
    def get_values(self) -> list:
        return [x for x in self._values]
    
    def get_timesteps(self) -> list:
        return [x for x in self._timesteps]
    
    def get_state(self):
        """The state is the current numerical or state value of the characteristic at hand

        Returns:
            [str or float]: current state
        """
        return self._state
    
    def set_state(self, state: int, time: float):
        """To use when the state is changing

        Args:
            state (int): new state
            time (float): timestamp
        """
        self._state = state
        self._values.append(state)
        self._timesteps.append(time)
        
    def reset(self, time: float):
        """Return to its original value, especially useful when the student resets to the original configuration of the simulation

        Args:
            time (float): timestamp
        """
        self._state = self._initial_state
        self._values.append(self._initial_state)
        self._timesteps.append(time)
        
    def close(self, time: float):
        """To use when the student finishes the simulation to create the end points

        Args:
            time (float): [description]
        """
        self._values.append(self._state)
        self._timesteps.append(time)