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

class Chem2CapSequencer(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of 3 components
        - whether or not the student can observe the transmittance or observance
        - the colour of the solution: red, green or other
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
    Except for those tree actions:
            concentrationlab
                any interaction in the concentrationlab
            pdf
                pdf's show and hide
            break
                any "inaction" on the simulation (determined by another parameter in the pipeline)
    """
    def __init__(self, settings):
        self._name = 'chem2cap sequencer'
        self._notation = 'c2csqcr'
        self._settings = settings
        self._states = [
            'break',
            'concentrationlab',
            'pdf',  
            'absobserved_red_wl_other',
            'absobserved_red_wl_wavelength',
            'absobserved_red_wl_solution',
            'absobserved_red_wl_concentration',
            'absobserved_red_wl_flask',
            'absobserved_red_no_wl_other',
            'absobserved_red_no_wl_wavelength',
            'absobserved_red_no_wl_solution',
            'absobserved_red_no_wl_concentration',
            'absobserved_red_no_wl_flask',
            'absobserved_green_wl_other',
            'absobserved_green_wl_wavelength',
            'absobserved_green_wl_solution',
            'absobserved_green_wl_concentration',
            'absobserved_green_wl_flask',
            'absobserved_green_no_wl_other',
            'absobserved_green_no_wl_wavelength',
            'absobserved_green_no_wl_solution',
            'absobserved_green_no_wl_concentration',
            'absobserved_green_no_wl_flask',
            'absobserved_wrongcol_wl_other',
            'absobserved_wrongcol_wl_wavelength',
            'absobserved_wrongcol_wl_solution',
            'absobserved_wrongcol_wl_concentration',
            'absobserved_wrongcol_wl_flask',
            'absobserved_wrongcol_no_wl_other',
            'absobserved_wrongcol_no_wl_wavelength',
            'absobserved_wrongcol_no_wl_solution',
            'absobserved_wrongcol_no_wl_concentration',
            'absobserved_wrongcol_no_wl_flask',
            'notobserved_red_wl_other',
            'notobserved_red_wl_wavelength',
            'notobserved_red_wl_solution',
            'notobserved_red_wl_concentration',
            'notobserved_red_wl_flask',
            'notobserved_red_no_wl_other',
            'notobserved_red_no_wl_wavelength',
            'notobserved_red_no_wl_solution',
            'notobserved_red_no_wl_concentration',
            'notobserved_red_no_wl_flask',
            'notobserved_green_wl_other',
            'notobserved_green_wl_wavelength',
            'notobserved_green_wl_solution',
            'notobserved_green_wl_concentration',
            'notobserved_green_wl_flask',
            'notobserved_green_no_wl_other',
            'notobserved_green_no_wl_wavelength',
            'notobserved_green_no_wl_solution',
            'notobserved_green_no_wl_concentration',
            'notobserved_green_no_wl_flask',
            'notobserved_wrongcol_wl_other',
            'notobserved_wrongcol_wl_wavelength',
            'notobserved_wrongcol_wl_solution',
            'notobserved_wrongcol_wl_concentration',
            'notobserved_wrongcol_wl_flask',
            'notobserved_wrongcol_no_wl_other',
            'notobserved_wrongcol_no_wl_wavelength',
            'notobserved_wrongcol_no_wl_solution',
            'notobserved_wrongcol_no_wl_concentration',
            'notobserved_wrongcol_no_wl_flask'  
        ]
        self._click_interval = 0.05
        
        self._load_labelmap()
        self._break_threshold = self._settings['data']['pipeline']['break_threshold']
        self._break_filter = BreakFilter(self, self._break_threshold)
        
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
        
    def get_sequences(self, simulation:Simulation) -> Tuple[list, list, list]:
        simulation.close()
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
        
        # the colour of the solution
        solution_values, solution_timestamps = self._process_solution(self._solution[0]), self._solution[1]
        
        # the wavelength of the solution
        wl_values, wl_timestamps = self._process_wl(self._wavelength[0]), self._wavelength[1]
        
        new_labels = []
        for i, lab in enumerate(labels):
            # transmittance or absorbance
            dependent_var_ts, dependent_variable, m_obs = self._get_value_timestep(dependent_var_ts, dependent_variable, begins[i])
            m_obs = self._proces_absorbance_other(m_obs)
            # sol colour
            solution_timestamps, solution_values, sv = self._get_value_timestep(solution_timestamps, solution_values, begins[i])
            
            # wavelength
            wl_timestamps, wl_values, wl = self._get_value_timestep(wl_timestamps, wl_values, begins[i])
            
            if lab == 'concentrationlab':
                action = 'concentrationlab'
            elif lab == 'pdf':
                action = 'pdf'
            else:
                action = m_obs + '_' + sv + '_' + wl + '_' + lab
            new_labels.append(action)
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
        wl_values = ['wl' if '520' in str(wl) else 'no_wl' for wl in wl_values]
        return wl_values
    
    def _proces_absorbance_other(self, metric_observed: str):
        if metric_observed == 'absorbance':
            return 'absobserved'
        else:
            return 'notobserved'
            