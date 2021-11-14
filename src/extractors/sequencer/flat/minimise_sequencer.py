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

class MinimiseSequencing(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of 3 components
        - whether or not the student can observe the transmittance or observance
        - the colour of the solution: red, green or other
        - whether the ruler is measuring the flask or not
        - whether the wavelength is 520 or not
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
    """
    def __init__(self):
        self._name = 'minimise sequencer'
        self._notation = 'minimi sqcr'
        self._states = [
            'break',
            'notobserved_wrongcol_no_wl_rul_flask',
            'absobserved_green_no_wl_no_rul_concentration',
            'absobserved_red_no_wl_rul_concentration',
            'notobserved_red_no_wl_no_rul_flask',
            'notobserved_wrongcol_wl_rul_concentration',
            'notobserved_red_wl_no_rul_concentration',
            'notobserved_green_wl_rul_flask',
            'absobserved_wrongcol_no_wl_rul_concentration',
            'absobserved_green_wl_no_rul_concentration',
            'notobserved_red_wl_rul_concentration',
            'absobserved_wrongcol_no_wl_no_rul_flask',
            'other',
            'notobserved_red_wl_no_rul_flask',
            'notobserved_wrongcol_wl_no_rul_flask',
            'absobserved_green_no_wl_no_rul_flask',
            'absobserved_green_wl_no_rul_flask',
            'absobserved_wrongcol_wl_no_rul_concentration',
            'notobserved_wrongcol_no_wl_no_rul_concentration',
            'absobserved_red_wl_rul_concentration',
            'notobserved_red_no_wl_rul_flask',
            'notobserved_wrongcol_no_wl_rul_concentration',
            'absobserved_red_no_wl_no_rul_concentration',
            'notobserved_red_no_wl_no_rul_concentration',
            'absobserved_red_no_wl_rul_flask',
            'notobserved_wrongcol_wl_rul_flask',
            'notobserved_green_no_wl_no_rul_flask',
            'absobserved_green_no_wl_rul_concentration',
            'notobserved_red_wl_rul_flask',
            'absobserved_red_wl_no_rul_concentration',
            'absobserved_red_no_wl_no_rul_flask',
            'notobserved_green_wl_rul_concentration',
            'notobserved_green_no_wl_rul_concentration',
            'absobserved_green_no_wl_rul_flask',
            'notobserved_wrongcol_wl_no_rul_concentration',
            'absobserved_wrongcol_wl_rul_concentration',
            'pdf',
            'absobserved_wrongcol_wl_rul_flask',
            'absobserved_red_wl_rul_flask',
            'notobserved_green_wl_no_rul_flask',
            'absobserved_wrongcol_no_wl_rul_flask',
            'notobserved_green_wl_no_rul_concentration',
            'concentrationlab',
            'notobserved_green_no_wl_rul_flask',
            'absobserved_wrongcol_no_wl_no_rul_concentration',
            'notobserved_red_no_wl_rul_concentration',
            'absobserved_green_wl_rul_concentration',
            'absobserved_wrongcol_wl_no_rul_flask',
            'notobserved_green_no_wl_no_rul_concentration',
            'absobserved_red_wl_no_rul_flask',
            'notobserved_wrongcol_no_wl_no_rul_flask',
            'absobserved_green_wl_rul_flask'
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
            
            'flask': 'flask',
            
            'pdf': 'pdf',
            'concentrationlab': 'concentrationlab',
        }
        
    def get_sequences(self, simulation:Simulation) -> Tuple[list, list, list]:
        self._load_sequences(simulation)
        begins = [x for x in self._begins]
        ends = [x for x in self._ends]
        labels = [x for x in self._labels]
        
        # whether the measure is displayed
        measure_displayed = dict(self._measure_displayed)
        measure_begin = measure_displayed['begin']
        measure_end = measure_displayed['end']
        measure_map = {True: 'obs', False: 'non_obs'}
        
        # Absorbance or Tranmisttance
        abs_trans_values = [x for x in self._metric[0]]
        abs_trans_timestamps = [x for x in self._metric[1]]
        
        # whether the ruler is measuring something
        ruler_measure = dict(self._ruler_measuring)
        ruler_begin = ruler_measure['begin']
        ruler_end = ruler_measure['end']
        ruler_map = {True: 'rul', False:'no_rul'}
        
        
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
            rm = ruler_map[rm]
            
            # sol colour
            solution_timestamps, solution_values, sv = self._get_value_timestep(solution_timestamps, solution_values, begins[i])
            
            # wavelength
            wl_timestamps, wl_values, wl = self._get_value_timestep(wl_timestamps, wl_values, begins[i])
            
            new_lab = m_obs + '_' + sv + '_' + wl + '_' + rm + '_' + lab
            new_labels.append(new_lab)
            
        new_labels = [self._clean(l) for l in new_labels]
        return new_labels, begins, ends
    
    def _clean(self, label:str) -> str:
        l = self._clean_pdf(label)
        l = self._clean_concentrationlab(l)
        l = self._clean_other(l)
        return l
        
    def _clean_pdf(self, label:str) -> str:
        if 'pdf' in label:
            return 'pdf'
        else:
            return label
    
    def _clean_concentrationlab(self, label:str) -> str:
        if 'concentrationlab' in label:
            return 'concentrationlab'
        else:
            return label
        
    def _clean_other(self, label:str) -> str:
        if 'other' in label:
            return 'other'
        else:
            return label
        
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
            
            