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

class KateStateSecondsLSTM(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of a vector encoding:
        - 1 if the action is conducted while the absorbance is on, the laser is green, and the solution is green
        - 1 if the action is conducted while the abosrbance is on, the laser is green, and the solution is red
        - 1 if the action is conducted while the absorbance is on and the laser is not green, or the solution is neither red nor green
        - 1 if the action is conducted while the absorbance is off
        - time spent on the action if the action is other
            - laser
            - restarts
            - transmittance / absorbance
        - time spent on the action if the action is related to concentration
        - time spent on the action if the action is related to width
        - time spent on the action if the action is related to wavelength
        - time spent on the action if the action is related to solution
        - time spent on the action if the action is related to measuring tools (ruler or magnifier)
        - time spent on the action if the action is related to the concentrationlab
        - time spent in the pdf
        - time spent not acting on the simulation

    """

    def __init__(self, settings):
        self._name = 'simple more state seconds sequencer'
        self._notation = 'ssss'
        self._settings = settings
        self._states = [
            'complementary_abs',
            'equal_abs',
            'complementary_transm',
            'equal_transm',
            'noobserved',
            'other',
            'concentration',
            'width',
            'wavelengthslider',
            'pdf',
            'break'
        ]
        self._click_interval = 0.05
        
        self._load_labelmap()
        self._break_threshold = self._settings['data']['pipeline']['break_threshold']
        self._break_filter = BreakFilter(self, self._break_threshold)

    def _load_labelmap(self):
        self._label_map = {
            'laser': 'todelete',
            'restarts': 'todelete',
            'transmittance_absorbance': 'todelete',
 
            'magnifier_position': 'todelete',
            'ruler': 'todelete',
            
            'wavelength_radiobox': 'wavelengthslider',
            'preset': 'wavelengthslider',
            'wl_variable': 'wavelengthslider',
            'minus_wl_slider': 'wavelengthslider',
            'wl_slider': 'wavelengthslider',
            'plus_wl_slider': 'wavelengthslider',
            
            'solution_menu': 'todelete',
            
            'minus_concentration_slider': 'concentration',
            'plus_concentration_slider': 'concentration',
            'concentration_slider': 'concentration',
            
            'flask': 'width',
            
            'pdf': 'pdf',
 
            'concentrationlab': 'todelete',
        }
        
        
        self._index_vector = {
            0: 'complementary_abs',
            1: 'equal_abs',
            2: 'complementary_transm',
            3: 'equal_transm',
            4: 'noobserved',
            5: 'other',
            6: 'concentration',
            7: 'width',
            8: 'wavelengthslider',
            9: 'pdf',
            10: 'break'
        }
        
        self._vector_index = {
           'complementary_abs': 0,
            'equal_abs': 1,
            'complementary_transm': 2,
            'equal_transm': 3,
            'noobserved': 4,
            'other': 5,
            'concentration': 6,
            'width': 7,
            'wavelengthslider': 8,
            'pdf': 9,
            'break': 10
        }

        self._vector_size = len(self._vector_index)
        self._vector_states = 6
        self._break_state = 10
        
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
        if attributes[4] == 'pdf':
            vector[9] = second
            return list(vector)

        if attributes[4] == 'break':
            vector[10] = second
            return list(vector)

        if attributes[4] == 'wavelengthslider':
            vector[8] = second
            return list(vector)
        
        if attributes[0] != 'absorbance' and attributes[0] != 'transmittance':
            vector[4] = 1

        elif attributes[1] == 'red' and attributes[2] == 'green' and attributes[0] == 'absorbance':
            vector[0] = 1
        
        elif attributes[1] == 'orange' and attributes[2] == 'violet' and attributes[0] == 'absorbance':
            vector[0] = 1

        elif attributes[1] == 'yellow' and attributes[2] == 'violet' and attributes[0] == 'absorbance':
            vector[0] = 1

        elif attributes[1] == 'green' and attributes[2] == 'blue' and attributes[0] == 'absorbance':
            vector[0] = 1

        elif attributes[1] == 'blue' and attributes[2] == 'brown' and attributes[0] == 'absorbance':
            vector[0] = 1
        
        elif attributes[1] == 'violet' and attributes[2] == 'green' and attributes[0] == 'absorbance':
            vector[0] = 1
        
        elif attributes[1] == 'red' and attributes[2] == 'red' and attributes[0] == 'absorbance':
            vector[1] = 1

        elif attributes[1] == 'orange' and attributes[2] == 'yellow and orange' and attributes[0] == 'absorbance':
            vector[1] = 1

        elif attributes[1] == 'yellow' and attributes[2] == 'yellow and orange' and attributes[0] == 'absorbance':
            vector[1] = 1

        elif attributes[1] == 'green' and attributes[2] == 'green' and attributes[0] == 'absorbance':
            vector[1] = 1
        
        elif attributes[1] == 'blue' and attributes[2] == 'blue' and attributes[0] == 'absorbance':
            vector[1] = 1
        
        elif attributes[1] == 'violet' and attributes[2] == 'violet' and attributes[0] == 'absorbance':
            vector[1] = 1
        
        elif attributes[1] == 'red' and attributes[2] == 'green' and attributes[0] == 'transmittance':
            vector[2] = 1

        elif attributes[1] == 'orange' and attributes[2] == 'violet' and attributes[0] == 'transmittance':
            vector[2] = 1

        elif attributes[1] == 'yellow' and attributes[2] == 'violet' and attributes[0] == 'transmittance':
            vector[2] = 1

        elif attributes[1] == 'green' and attributes[2] == 'blue' and attributes[0] == 'transmittance':
            vector[2] = 1

        elif attributes[1] == 'blue' and attributes[2] == 'brown' and attributes[0] == 'transmittance':
            vector[2] = 1

        elif attributes[1] == 'violet' and attributes[2] == 'green' and attributes[0] == 'transmittance':
            vector[2] = 1

        elif attributes[1] == 'red' and attributes[2] == 'red' and attributes[0] == 'transmittance':
            vector[3] = 1

        elif attributes[1] == 'orange' and attributes[2] == 'yellow and orange' and attributes[0] == 'transmittance':
            vector[3] = 1

        elif attributes[1] == 'yellow' and attributes[2] == 'yellow and orange' and attributes[0] == 'transmittance':
            vector[3] = 1

        elif attributes[1] == 'green' and attributes[2] == 'green' and attributes[0] == 'transmittance':
            vector[3] = 1

        elif attributes[1] == 'blue' and attributes[2] == 'blue' and attributes[0] == 'transmittance':
            vector[3] = 1
        
        elif attributes[1] == 'violet' and attributes[2] == 'violet' and attributes[0] == 'transmittance':
            vector[3] = 1

        else:
            vector[4] = 1


        vector[self._vector_index[attributes[4]]] = second
        return list(vector)


    def get_sequences(self, simulation:Simulation, lid:str) -> Tuple[list, list, list]:
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
        new_begins = []
        new_ends = []

        for i, lab in enumerate(labels):
            if lab == 'todelete':
                continue
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
            'potassiumDichromate': 'orange',
            'cobaltChloride': 'red',
            'copperSulfate': 'blue',
            'nickelIIChloride': 'green',
            'potassiumPermanganate': 'violet',
            'potassiumChromate': 'yellow',
            'cobaltIINitrate': 'red'
        }
        solution_values = [s.replace('beersLawLab.beersLawScreen.solutions.', '') for s in solution_values]
        solution_values = [colour_map[s] for s in solution_values]
        return solution_values

    def _wavelengths_to_colour(self, wl):
        if 380 <= wl and wl <= 430:
            return 'violet'
        if 430 <= wl and wl <= 470:
            return 'blue'
        if 470 <= wl and wl <= 570:
            return 'green'
        if 570 <= wl and wl <= 630:
            return 'yellow and orange'
        if 630 <= wl and wl <= 680:
            return 'red'
        if 680 <= wl and wl <= 780:
            return 'brown'

    def _wavelength_to_colour(self, wl):
        if 500 <= wl and wl<= 564:
            return 'green'

    def _process_wl(self, wl_values: list) -> list:
        wl_values = [self._wavelengths_to_colour(wl) for wl in wl_values]
        return wl_values
    
    def _basic_common_filtering(self, labels, begins, ends, simulation):
        labels, begins, ends = self._filter_doubleeveents(labels, begins, ends)
        labels, begins, ends = self._filter_overlaps(labels, begins, ends)
        
        if self._settings['data']['pipeline']['sequencer_dragasclick']:
            break_threshold = self._break_filter.get_threshold(begins, ends, self._break_threshold)
            self._break_minimum = break_threshold
            labels, begins, ends = self._filter_clickasdrag(labels, begins, ends, break_threshold)
        return labels, begins, ends