import time
import numpy as np
import pandas as pd
from typing import Tuple

from extractors.sequencer.sequencing import Sequencing
from extractors.parser.simulation_parser import Simulation
from extractors.parser.checkbox_object import Checkbox
from extractors.parser.event_object import Event
from extractors.parser.simulation_object import SimObjects
from extractors.parser.value_object import SimCharacteristics
from extractors.cleaners.break_filter import BreakFilter

class ColourNobreakSecondsLSTM(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of a vector encoding:
        - time spent on actions when the laser was green and the solution was also green
        - time spent on actions when the laser was green and the solutino was red
        - time spent on actions when either the laser was not green or the solution was neither green nor red
        - time spent on the concentrationlab

        vector at time t:
            0: (s + vector(t-1)[0]) / vector(t) if state is green - green
            1: (s + vector(t-1)[1]) / vector(t) green - red
            2: (s + vector(t-1)[2]) / vector(t) no green laser or (no green solution and no red solution)
            3: (s + vector(t-1)[3]) / vector(t) concentrationlab
        => s being 0 if it's in the corresponding state, or the timing of the interaction in the current state
    """
    def __init__(self, settings):
        self._name = 'colournobreak seconds sequencer'
        self._notation = 'cnbss'
        self._settings = settings
        self._states = [
            'greengreen',
            'greenred',
            'nogreennored',
            'concentrationlab'
        ]
        self._click_interval = 0.05
        
        self._load_labelmap()
        self._break_threshold = self._settings['data']['pipeline']['break_threshold']
        self._break_filter = BreakFilter(self, self._break_threshold)

    def _load_labelmap(self):
        self._label_map = {
            'laser': 'action',
            'restarts': 'action',
            'transmittance_absorbance': 'action',

            'magnifier_position': 'action',
            'ruler': 'action',
            
            'wavelength_radiobox': 'action',
            'preset': 'action',
            'wl_variable': 'action',
            'minus_wl_slider': 'action',
            'wl_slider': 'action',
            'plus_wl_slider': 'action',
            
            'solution_menu': 'action',
            
            'minus_concentration_slider': 'action',
            'plus_concentration_slider': 'action',
            'concentration_slider': 'action',
            
            'flask': 'action',
            
            'pdf': 'action',

            'concentrationlab': 'concentrationlab',
        }
        
        self._index_vector = {
            0: 'greengreen',
            1: 'greenred',
            2: 'nogreennored',
            3: 'concentrationlab',
        }
        
        self._vector_index = {
            'greengreen': 0,
            'greenred':1,
            'nogreennored':2,
            'concentrationlab':3
        }
    
        self._vector_size = 4
        self._vector_states = 4
        self._break_state = -1
        
    def get_vector_size(self):
        return self._vector_size
    def get_vector_states(self):
        return self._vector_states
    def get_break_state(self):
        return self._break_state
        
    def _fill_vector(self, attributes: list, second:float) -> list:
        """Vector string: [m_obs, sv, wl, rm, lab]
            second: length of the interaction
        """
        vector = np.zeros(self._vector_size)

        if attributes[4] == 'concentrationlab':
            vector[3] = second
            return list(vector)

        if attributes[0] != 'absorbance':
            vector[2] = second
            return list(vector)

        if attributes[2] == 'wl' and attributes[1] == 'green':
            vector[0] = second
            return list(vector)

        if attributes[2] == 'wl' and attributes[1] == 'red':
            vector[1] = second
            return list(vector)

        else:
            vector[2] = second
            return list(vector)


    def get_sequences(self, simulation:Simulation) -> Tuple[list, list, list]:
        # simulation.close()
        self._load_sequences(simulation)
        begins = [x for x in self._begins]
        ends = [x for x in self._ends]
        labels = [x for x in self._labels]
        if len(labels) == 0:
            return [], [], []
        labels, begins, ends = self._basic_common_filtering(labels, begins, ends, simulation)
        
        # whether the measure is displayed
        measure_displayed = dict(self._measure_displayed)
        measure_begin = measure_displayed['begin']
        measure_end = measure_displayed['end']
        
        # Absorbance or Tranmisttance
        dependent_variable, dependent_var_ts = self.get_absorbance_transmittance_nothing(simulation)
        
        # whether the ruler is measuring something
        ruler_measure = dict(self._ruler_measuring)
        ruler_begin = ruler_measure['begin']
        ruler_end = ruler_measure['end']
        
        # the colour of the solution
        solution_values, solution_timestamps = self._process_solution(self._solution[0]), self._solution[1]
        
        # the wavelength of the solution
        wl_values, wl_timestamps = self._process_wl(self._wavelength[0]), self._wavelength[1]
        
        new_labels = []

        cumulative_vector = np.array([0, 0, 0, 0])

        for i, lab in enumerate(labels):
            # observable or not
            mm, measure_begin, measure_end = self._state_return(measure_begin, measure_end, begins[i])
            
            # transmittance or absorbance
            dependent_var_ts, dependent_variable, m_obs = self._get_value_timestep(dependent_var_ts, dependent_variable, begins[i])
            
            # ruler measuring
            rm, ruler_begin, ruler_end = self._state_return(ruler_begin, ruler_end, begins[i])
            
            # sol colour
            solution_timestamps, solution_values, sv = self._get_value_timestep(solution_timestamps, solution_values, begins[i])
            
            # wavelength
            wl_timestamps, wl_values, wl = self._get_value_timestep(wl_timestamps, wl_values, begins[i])
            
            instant_vector = self._fill_vector([m_obs, sv, wl, rm, lab], ends[i] - begins[i])
            cumulative_vector = np.array(cumulative_vector) + np.array(instant_vector)
            
            new_labels.append([cv for cv in cumulative_vector])
        return new_labels, begins, ends
    
    def _process_solution(self, solution_values: list):
        """Replace the values by whether the solution is green, red or from another colour
                - drink mix: red
                - cobalt (ii) nitrate: red 
                - cobalt (ii) chloride: other [orange]
                - potassium dichromate: other [orange]
                - potassium chromate: other [yellow]
                - nickel (ii) chloride: green
                - copper (ii) sulfate: other [blue]
                - potassium permanganate: other [purple]
        Args:
            solution_values (list): [replaced all solutions by red, green or other]
        """        
        colour_map = {
            'drinkMix': 'red',
            'potassiumDichromate': 'wrongcol',
            'cobaltChloride': 'red',
            'copperSulfate': 'wrongcol',
            'nickelIIChloride': 'green',
            'potassiumPermanganate': 'wrongcol',
            'potassiumChromate': 'wrongcol',
            'cobaltIINitrate': 'red'
        }
        solution_values = [s.replace('beersLawLab.beersLawScreen.solutions.', '') for s in solution_values]
        solution_values = [colour_map[s] for s in solution_values]
        return solution_values

    def _process_wl(self, wl_values: list) -> list:
        wl_values = ['wl' if (500 <= int(wl) and int(wl) <= 564) else 'no_wl' for wl in wl_values]
        return wl_values
    
    def _proces_absorbance_other(self, metric_observed: bool, absorbance: str):
        if metric_observed and 'absorbance':
            return 'absobserved'
        else:
            return 'notobserved'
            
            