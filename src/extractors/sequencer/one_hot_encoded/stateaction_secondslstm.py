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

class StateActionSecondsLSTM(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of a vector encoding:
        - whether the student is observing the absorbance
        - whether something else than absorbance is observed
        - if the red solution is used
        - if the green solution is used
        - if neither the red nor the green solution is used
        - if the ruler is measuring
        - if the ruler is not measuring
        - if the wavelength is 520nm
        - if the wavelength is not 520nm

        - the action
            other
                laser clicks
                transmittance absorbance clicks
                restarts timestamps
            concentration  
                concentration slider's drags and clicks
            flask
                flask's drags (width changes)
            wavelength
                wavelength slider's drags and clicks
                wavelength radio box clicks
            solution
                solution choice and selection
            measuring
                magnifier movements
                ruler dragsrestarts
            concentrationlab
                any interaction in the concentrationlab
            pdf
                pdf's show and hide
                
        vector:
            0: s for observed absorbance, 0 else
            1: s if something else than absorbance is observed, else 0
            2: s for red solution, else 0
            3: s for green solution, else 0
            4: s for other solution, else 0
            5: s if ruler is measuring, else 0
            6: s if ruler is not measuring, else 0
            7: s if wavelength is 520, else 0
            8: s if wavelength is not 520
            9: s if action is on other (laser clicks, transmittance absorbance clicks, restarts timestamps)
            10: s if action is on concentration
            11: s if action is on width
            12: s if action is wavelength
            13: s if action is on solution
            14: s if action is on measuring tools (magnifier and ruler)
            15: s if action is on concentrationlab
            16: s if action is on pdf
            17: s if break
        => s being the timing of the interaction

    """
    def __init__(self, settings):
        self._name = 'stateaction encodedlstm sequencer'
        self._notation = 'saenclstmsqcr'
        self._settings = settings
        self._states = [
            'absorbance',
            'observed',
            'red',
            'green',
            'notrednotgreen_solution',
            'ruler',
            'rulernotmeasuring',
            'wl520',
            'wlnot520',
            'other',
            'concentration',
            'width', 
            'wavelength', 
            'solution',
            'tools',
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
            'restarts': 'restart',
            'transmittance_absorbance': 'other',

            'magnifier_position': 'tools',
            'ruler': 'tools',
            
            'wavelength_radiobox': 'wavelength',
            'preset': 'wavelength',
            'wl_variable': 'wavelength',
            'minus_wl_slider': 'wavelength',
            'wl_slider': 'wavelength',
            'plus_wl_slider': 'wavelength',
            
            'solution_menu': 'solution',
            
            'minus_concentration_slider': 'concentration',
            'plus_concentration_slider': 'concentration',
            'concentration_slider': 'concentration',
            
            'flask': 'width',
            
            'pdf': 'pdf',

            'concentrationlab': 'concentrationlab',
        }
        
        self._index_vector = {
            0: 'absorbance',
            1: 'observed',
            2: 'red',
            3: 'green',
            4: 'notrednotgreen_solution',
            5: 'ruler',
            6: 'rulernotmeasuring',
            7: 'wl520',
            8: 'wlnot520',
            9: 'other',
            10: 'concentration',
            11: 'width', 
            12: 'wavelength', 
            13: 'solution',
            14: 'tools',
            15: 'concentrationlab',
            16: 'pdf',
            17: 'break'
        }
        
        self._vector_index = {
            'absorbance': 0,
            'observed': 1,
            'red': 2,
            'green': 3,
            'notrednotgreen_solution': 4,
            'ruler': 5,
            'rulernotmeasuring': 6,
            'wl520': 7,
            'wlnot520': 8,
            'other': 9,
            'concentration': 10,
            'width': 11, 
            'wavelength': 12, 
            'solution': 13,
            'tools': 14,
            'concentrationlab': 15,
            'pdf': 16,
            'break': 17
        }
    
        self._vector_size = 18
        self._vector_states = 9
        self._break_state = 17
        
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
            vector[15] = second
            return list(vector)

        if attributes[0] == 'absorbance':
            vector[0] = second
        else:
            vector[1] = second
            
        if attributes[1] == 'red':
            vector[2] = second
        elif attributes[1] == 'green':
            vector[3] = second
        else:
            vector[4] = second
            
        if attributes[2] == 'wl':
            vector[7] = second
        else:
            vector[8] = second
            
        if attributes[3]:
            vector[5] = second
        else:
            vector[6] = second
            
        vector[self._vector_index[attributes[4]]] = second
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
            
            vector = self._fill_vector([m_obs, sv, wl, rm, lab], ends[i] - begins[i])
            
            new_labels.append(vector)
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
        wl_values = ['wl' if (500 <= int(wl) and int(wl) >= 564) else 'no_wl' for wl in wl_values]
        return wl_values
    
    def _proces_absorbance_other(self, metric_observed: bool, absorbance: str):
        if metric_observed and 'absorbance':
            return 'absobserved'
        else:
            return 'notobserved'
            
            