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

class Bin1hotExtendedSequencing(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of a vector encoding:
        - whether or not the student can observe the transmittance or observance
        - the colour of the solution: red, green or other
        - whether the ruler is measuring the flask or not
        - whether the wavelength is 20 or not
        - the action
            other
                laser clicks
                ruler dragsrestarts
                transmittance absorbance clicks
                magnifier movements
                restarts timestamps
                wavelength [value is taken into the state]
                    wavelength slider's drags and clicks
                    wavelength radio box clicks
                solution [value is taken into the state]
                    solution choice and selection
            concentration  
                concentration slider's drags and clicks
            flask
                flask's drags (width changes)
            concentrationlab
                any interaction in the concentrationlab
            pdf
                pdf's show and hide
                
        vector:
            0: 1 for observed absorbance, 0 for not observed absorbance
            1: 1 for red solution, else 0
            2: 1 for green solution, else 0
            3: 1 for other solution, else 0
            4: 1 if ruler is measuring, else 0
            5: 2 if wavelength is 520, else 0
            6: action is on other (laser clicks, ruler, drag restarts, transmittance/absorbance clicks, magnifier movements)
            7: action is on concentration
            8: action is on flask
            9: action is on wavelength
            10: action is on solution
            11: action is on concentrationlab
            12: action is on pdf
            13: break
    """
    def __init__(self):
        self._name = '1bin extended sequencer'
        self._notation = '1binextsqcr'
        self._states = [
            'observed_absorbance',
            'red',
            'green',
            'wrongcol',
            'rul',
            'wl',
            'other',
            'concentration',
            'flask',
            'wavelength', 
            'solution',
            'concentrationlab',
            'pdf',
            'break'
        ]
        self._click_interval = 0.05
        
        self._load_labelmap()
        
    def _load_labelmap(self):
        self._label_map = {
            'laser': 'other',
            'ruler': 'other',
            'restarts': 'other',
            'transmittance_absorbance': 'other',
            'magnifier_position': 'other',
            
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
            
            'flask': 'flask',
            
            'pdf': 'pdf',
            'concentrationlab': 'concentrationlab',
        }
        
        self._index_vector = {
            0: 'observed_absorbance',
            1: 'red',
            2: 'green',
            3: 'wrongcol',
            4: 'rul',
            5: 'wl',
            6: 'other',
            7: 'concentration',
            8: 'flask',
            9: 'wavelength', 
            10: 'solution',
            11: 'concentrationlab',
            12: 'pdf',
            13: 'break'
        }
        
        self._vector_index = {
            'observed_absorbance': 0,
            'red': 1,
            'green': 2,
            'wrongcol': 3,
            'rul': 4,
            'wl': 5,
            'other': 6,
            'concentration': 7,
            'flask': 8,
            'wavelength': 9, 
            'solution': 10,
            'concentrationlab': 11,
            'pdf': 12,
            'break': 13
        }
        
        self._vector_size = 14
        self._vector_states = 6
        self._break_state = 13
        
    def get_vector_size(self):
        return self._vector_size
    def get_vector_states(self):
        return self._vector_states
    def get_break_state(self):
        return self._break_state
        
    def _fill_vector(self, attributes: list) -> list:
        """Vector string: [m_obs, sv, wl, rm, lab]
        """
        vector = np.zeros(self._vector_size)
        if attributes[0] == 'absobserved':
            vector[0] = 1
            
        if attributes[1] == 'red':
            vector[1] = 1
        elif attributes[1] == 'green':
            vector[2] = 1
        else:
            vector[3] = 1
            
        if attributes[2] == 'wl':
            vector[5] = 1
            
        if attributes[3] == 'rul':
            vector[4] = 1
            
        vector[self._vector_index[attributes[4]]] = 1
            
        return list(vector)
    
    def get_sequences(self, simulation:Simulation) -> Tuple[list, list, list]:
        self._load_sequences(simulation)
        begins = [x for x in self._begins]
        ends = [x for x in self._ends]
        labels = [x for x in self._labels]
        
        # whether the measure is displayed
        measure_displayed = dict(self._measure_displayed)
        measure_begin = measure_displayed['begin']
        measure_end = measure_displayed['end']
        
        # Absorbance or Tranmisttance
        abs_trans_values = [x for x in self._metric[0]]
        abs_trans_timestamps = [x for x in self._metric[1]]
        
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
            abs_trans_timestamps, abs_trans_values, abstrans = self._get_value_timestep(abs_trans_timestamps, abs_trans_values, begins[i])
            
            m_obs = self._proces_absorbance_other(mm, abstrans)
            
            # ruler measuring
            rm, ruler_begin, ruler_end = self._state_return(ruler_begin, ruler_end, begins[i])
            
            # sol colour
            solution_timestamps, solution_values, sv = self._get_value_timestep(solution_timestamps, solution_values, begins[i])
            
            # wavelength
            wl_timestamps, wl_values, wl = self._get_value_timestep(wl_timestamps, wl_values, begins[i])
            
            vector = self._fill_vector([m_obs, sv, wl, rm, lab])
            new_labels.append(vector)
            
        return new_labels, begins, ends
    
    def _clean(self, label:str) -> str:
        l = self._clean_pdf(label)
        l = self._clean_concentrationlab(l)
        l = self._clean_other(l)
        return l
        
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
            'cobaltChloride': 'wrongcol',
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
        wl_values = ['wl' if '520' in str(wl) else 'no_wl' for wl in wl_values]
        return wl_values
    
    def _proces_absorbance_other(self, metric_observed: bool, absorbance: str):
        if metric_observed and 'absorbance':
            return 'absobserved'
        else:
            return 'notobserved'
            
            