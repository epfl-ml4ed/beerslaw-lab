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

class SimpleStateSecondsLSTM(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of a vector encoding:
        - 1 if the action is conducted while the absorbance is on, the laser is green, and the solution is green
        - 1 if the action is conducted while the abosrbance is on, the laser is green, and the solution is red
        - 1 if the action is conducted while the absorbance is off, or the laser is not green, or the solution is neither red nor green
        - time spent on the action if the action is other
            - wavelength
            - laser
            - restarts
            - transmittance / absorbance
            - magnifier position
            - ruler
            - solution
        - time spent on the action if the action is related to concentration
        - time spent on the action if the action is related to width
        - time spent on the action if the action is related to the concentrationlab
        - time spent not acting on the simulation

    """

    def __init__(self, settings):
        self._name = 'simple state seconds sequencer'
        self._notation = 'ssss'
        self._settings = settings
        self._states = [
            'greengreen',
            'greenred',
            'notgreennotred',
            'noobserved',
            'other',
            'concentration',
            'width',
            'concentrationlab',
            'pdf',
            'break'
        ]
        self._click_interval = 0.05
        
        self._load_labelmap()
        self._break_threshold = self._settings['data']['pipeline']['break_threshold']
        self._break_filter = BreakFilter(self, self._break_threshold)

    def _load_labelmap(self):
        self._label_map = {
            'laser': 'other',
            'restarts': 'other',
            'transmittance_absorbance': 'other',

            'magnifier_position': 'other',
            'ruler': 'other',
            
            'wavelength_radiobox': 'other',
            'preset': 'other',
            'wl_variable': 'other',
            'minus_wl_slider': 'other',
            'wl_slider': 'other',
            'plus_wl_slider': 'other',
            
            'solution_menu': 'other',
            
            'minus_concentration_slider': 'concentration',
            'plus_concentration_slider': 'concentration',
            'concentration_slider': 'concentration',
            
            'flask': 'width',
            
            'pdf': 'pdf',

            'concentrationlab': 'concentrationlab',
        }
        
        self._index_vector = {
            0: 'greengreen',
            1: 'greenred',
            2: 'notgreennotred',
            3: 'noobserved',
            4: 'other',
            5: 'concentration',
            6: 'width',
            7: 'concentrationlab',
            8: 'pdf',
            9: 'break'
        }
        
        self._vector_index = {
            'greengreen': 0,
            'greenred': 1,
            'notgreennotred': 2,
            'noobserved': 3,
            'other': 4,
            'concentration': 5,
            'width': 6,
            'concentrationlab': 7,
            'pdf': 8,
            'break': 9
        }
    
        self._vector_size = len(self._vector_index)
        self._vector_states = 4
        self._break_state = 9
        
    def get_vector_size(self):
        return self._vector_size
    def get_vector_states(self):
        return self._vector_states
    def get_break_state(self):
        return self._break_state
        
    def _fill_vector(self, attributes: list, second:float) -> list:
        """Vector string: [m_obs, sv, wl, rm, lab]
            second: length of the interaction
            break: whether it's an action or a break
        """
        vector = np.zeros(self._vector_size)

        if attributes[4] == 'concentrationlab':
            vector[7] = second
            return list(vector)

        if attributes[0] != 'absorbance':
            vector[3] = 1

        elif attributes[2] == 'wl' and attributes[1] == 'green':
            vector[0] = 1

        elif attributes[2] == 'wl' and attributes[1] == 'red':
            vector[1] = 1

        else:
            vector[2] = 1

        vector[self._vector_index[attributes[4]]] = second
        return list(vector)


    def get_sequences(self, simulation:Simulation, lid:str) -> Tuple[list, list, list]:
        # simulation.close()
        self._load_sequences(simulation)
        begins = [x for x in self._begins]
        ends = [x for x in self._ends]
        labels = [x for x in self._labels]

        for i in range(len(labels)):
            print(begins[i], ends[i], labels[i])
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
        new_begins = []
        new_ends = []

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
            
            # action
            instant_vector = self._fill_vector([m_obs, sv, wl, rm, lab], ends[i] - begins[i])
            new_begins.append(begins[i])
            new_ends.append(ends[i])
            new_labels.append([cv for cv in instant_vector])
            # print(lab, instant_vector)

        return new_labels, new_begins, new_ends
    
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
            
            