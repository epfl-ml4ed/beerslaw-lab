from typing import Tuple
import numpy as np 
import pandas as pd
import time
import logging

from matplotlib import pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, show

class SimObjects():
    """Emulates a component from the simulation, such as a laser, a battery, ...
    """
    def __init__(self, name: str, simulation_id: str, initial_state: str, visible=True, scaling=1):
        self._scaling = scaling
        self._active = False
        self._visible = visible
        self._state = initial_state
        self._initial_value = initial_state
        # If it is in use or not
        self._on = {'begin' : [], 'end' : []}
        self._off = {'begin' : [0], 'end' : []}
        
        # If it is showing or not
        self._shown = {'begin' : [], 'end': []}
        self._hidden = {'begin' : [0], 'end' : []}
        
        # dragging interactions
        self._dragging = {
            'begin': [],
            'end': [],
            'values': []
        }
        self._values_drags = [initial_state]
        
        # fired timestamp
        self._firing = {
            'timestamps': [],
            'values': []
        }
        
        self._interaction_timesteps = [0]
        self._interaction_values = [initial_state]
        self._interactions = ['init']
        
        self._lastTime = -1
        
    def get_dragging(self) -> dict:
        return dict(self._dragging)
    
    def get_firing(self) -> dict:
        return dict(self._firing)
        
    def is_active(self) -> int:
        """returns whether the object is in use right now

        Returns:
            int: boolean
        """
        return self._active

    def _check_validity(self, on: list, off: list, time: float):
        if len(on) > 0:
            assert on[-1] <= time
        if len(off) > 0:
            assert off[-1] <= time
            
    def get_onoff_sequences(self) -> Tuple[list, list]:
        return dict(self._on), dict(self._off)
    
    def get_drags_sequences(self) -> Tuple[list, list]:
        return [x for x in self._start_drags], [x for x in self._end_drags]
    
    def get_lastvalue_drags(self) -> float:
        return self._values_drags[-1]
    
    def get_state(self) -> int:
        return self._state

    def visibility_switch(self, state: int, time: float):
        """Switch the state of visibility
           This functions always checks that the *state* is different than the current one
        Args:
            state (int): new state
            time (float): timestamp
        """
        if self._visible != state:
            if self._visible:
                self._check_validity(self._shown['end'], self._hidden['begin'], time)
                self._shown['end'].append(time)
                self._hidden['begin'].append(time)
                self._visible = 0
                self._interactions.append('hiding')
                self._interaction_values.append(0)
                self._interaction_timesteps.append(time)
            else:
                self._check_validity(self._shown['begin'], self._hidden['end'], time)
                self._shown['begin'].append(time)
                self._hidden['end'].append(time)
                self._visible = 1
                self._interactions.append('showing')
                self._interaction_values.append(1)
                self._interaction_timesteps.append(time)
        
    def switch(self, state: int, time: float):
        """Switches the state of the object (in use vs not in use)
           This functions always checks that the *state* is different than the current one
        Args:
            state (int): new state
            time (float): timestamp
        """
        if self._active != state:
            if self._active:
                self._check_validity(self._on['end'], self._off['begin'], time)
                self._on['end'].append(time)
                self._off['begin'].append(time)
                self._active = 0
                self._interactions.append('switched_off')
                self._interaction_values.append(0)
                self._interaction_timesteps.append(time)
            else:
                self._check_validity(self._on['begin'], self._off['end'], time)
                self._on['begin'].append(time)
                self._off['end'].append(time)
                self._active = 1
                self._interactions.append('switched_on')
                self._interaction_values.append(1)
                self._interaction_timesteps.append(time)
            
            
    def start_dragging(self, value: float, time: float):
        """To use when the object is started to be dragged.
        Args:
            value (float): value of the slider
            time (float): timestamp
        """
        if len(self._dragging['begin']) != len(self._dragging['end']):
            logging.info('PARSE ERROR')
            self._dragging['end'].append(self._dragging['begin'][-1] + 0.05)
            self._dragging['values'].append(self._dragging['values'][-1])
            self._interactions.append('stop_drag')
            self._interaction_values.append(self._interaction_values[-1])
            self._interaction_timesteps.append(self._interaction_timesteps[-1])
            
        value = self._values_drags[-1]
        assert len(self._dragging['begin']) == len(self._dragging['end'])
        self._dragging['begin'].append(time)
        self._dragging['values'].append(value)
        self._interactions.append('start_drag')
        self._interaction_values.append(value)
        self._interaction_timesteps.append(time)
        self._state = value
        self.switch(1, time)
        
    def is_dragging(self, value: float, time: float):
        """Records the values the slider/object goes through when being dragged

        Args:
            value (float): value of the slider/object
            time (float): timestamp
        """
        if len(self._dragging['begin']) == len(self._dragging['end']):
            self.start_dragging(value, time)
        else:
            self._dragging['values'].append(value)
            self._interactions.append('dragging')
            self._interaction_values.append(value)
            self._interaction_timesteps.append(time)
            self._state = value
            self.switch(1, time)
        
    def stop_dragging(self, value: float, time: float):
        """Records the time the sliding/dragging interaction is done, as well as the value it landed on

        Args:
            value (float): final drag value
            time (float): timestamp
        """
        if len(self._dragging['begin']) == len(self._dragging['end']) + 1:
            self._dragging['end'].append(time)
            self._dragging['values'].append(value)
            self._interactions.append('end_drag')
            self._interaction_values.append(value)
            self._interaction_timesteps.append(time)
            self._state = value
            self.switch(0, time)
            
    def fire(self, value: float, time:float):
        """To use when one click/press action is recorded
        Args:
            value (float): value of the object
            time (float): timestamp
        """
        self._firing['timestamps'].append(time)
        self._firing['values'].append(value)
        self._interactions.append('fire')
        self._interaction_timesteps.append(time)
        self._interaction_values.append(value)
        
    def check_state(self, value: float, time: float):
        self._interactions.append('check_state')
        self._interaction_values.append(value)
        self._interaction_timesteps.append(time)
        self._state = value
        
    def reset(self, time: float):
        """To use when the student resets the simulation

        Args:
            time (float)
        """
        if self._active:
            self._switch(time)
        self.close(time)
        self._interactions.append('reset')
        self._state = self._initial_value
        self._interaction_values.append(self._state)
        self._interaction_timesteps.append(time)
        
    def close(self, timestamp: float):
        """To use at the end of the simulation

        Args:
            timestamp (float)
        """
        self._lastTime = timestamp
        if len(self._on['begin']) > len(self._on['end']):
            self._on['end'].append(timestamp)
        if len(self._off['begin']) > len(self._off['end']):
            self._off['end'].append(timestamp)
            
        if len(self._shown['begin']) > len(self._shown['end']):
            self._shown['end'].append(timestamp)
        if len(self._hidden['begin']) > len(self._hidden['end']):
            self._hidden['end'].append(timestamp)
            
        if len(self._dragging['begin']) > len(self._dragging['end']):
            self._dragging['end'].append(self._dragging['begin'][-1] + 0.05)
            
        self._interaction_values.append(self._values_drags[-1])
        self._interaction_timesteps.append(timestamp)
        self._interactions.append('close')

    def get_active_time_proportion(self) -> float:
        somme = (np.array(self._on['end']) - np.array(self._on['begin'])).sum()
        somme /= self._lastTime
        somme *= 100
        return somme
        
    def get_range(self, scaling=1) -> Tuple[float, float]:
        dragging_states = ['start_drag', 'dragging', 'end_drag', 'close']
        indices = [x for x in list(range(len(self._interactions))) if self._interactions[x] in dragging_states and (isinstance(self._interaction_values[x], float) or isinstance(self._interaction_values[x], int))]
        if indices == [0]:
            ys = [self._initial_value * scaling, self._initial_value *self._scaling]
        else:
            ys = [self._interaction_values[x] for x in indices]
            ys = list(np.array(ys) * self._scaling)

        return min(ys), max(ys)

       


        
    
        
        
