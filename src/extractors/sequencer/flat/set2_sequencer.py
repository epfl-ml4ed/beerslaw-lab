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

class Set2Sequencing(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of 3 components
        - whether or not the student can observe the transmittance or observance
        - the colour of the solution: red, green or other
        - the wavelength: 520 or other
        - the action
            other
                wavelength radio box clicks
                laser clicks
                ruler drags
            concentrationlab
                any interaction in the concentrationlab
            wavelength
                wavelength slider's drags and clicks
            concentration  
                concentration slider's drags and clicks
            flask
                flask's drags (width changes)
            solution   
                solution choice and selection
            pdf
                pdf's show and hide
    """
    def __init__(self):
        self._name = 'set 1 sequencer'
        self._notation = 'sqcr'
        self._states = ['not initialised']
        self._click_interval = 0.05
        
        self._load_labelmap()
        
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
        
        # the colour of the solution
        solution_values, solution_timestamps = self._process_solution(self._solution[0]), self._solution[1]
        
        # the wavelength of the solution
        wl_values, wl_timestamps = self._process_wl(self._wavelength[0]), self._wavelength[1]
        
        new_labels = []
        for i, lab in enumerate(labels):
            # observable or not
            mm, measure_begin, measure_end = self._state_return(measure_begin, measure_end, begins[i])
            mm = measure_map[mm]
            
            solution_timestamps, solution_values, sv = self._get_value_timestep(solution_timestamps, solution_values, begins[i])
            
            wl_timestamps, wl_values, wl = self._get_value_timestep(wl_timestamps, wl_values, begins[i])
            
            new_lab = mm + '_' + sv + '_' + wl + '_' + lab
            new_labels.append(new_lab)
            
        new_labels = [self._clean_pdf(l) for l in new_labels]
        new_labels = [self._clean_concentrationlab(l) for l in new_labels]
        return new_labels, begins, ends
    
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
        
    def _process_solution(self, solution_values: list) -> list:
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
            'potassiumDichromate': 'other',
            'cobaltChloride': 'other',
            'copperSulfate': 'other',
            'nickelIIChloride': 'green',
            'potassiumPermanganate': 'other',
            'potassiumChromate': 'other',
            'cobaltIINitrate': 'red'
        }
        solution_values = [s.replace('beersLawLab.beersLawScreen.solutions.', '') for s in solution_values]
        solution_values = [colour_map[s] for s in solution_values]
        return solution_values
    
    def _process_wl(self, wl_values: list) -> list:
        wl_values = ['wl' if '520' in str(wl) else 'no_wl' for wl in wl_values]
        return wl_values
        
        
        
        