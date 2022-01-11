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

class BaseLSTMSampling(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Because we are sampling, the starting times and ending times will start at regular intervals.
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
            0: 1 for observed absorbance, 0 else
            1: 1 if something else than absorbance is observed, else 0
            2: 1 for red solution, else 0
            3: 1 for green solution, else 0
            4: 1 for other solution, else 0
            5: 1 if ruler is measuring, else 0
            6: 1 if ruler is not measuring, else 0
            7: 1 if wavelength is 520, else 0
            8: 1 if wavelength is not 520
            9: action is on other (laser clicks, ruler, drag restarts, transmittance/absorbance clicks, magnifier movements)
            10: action is on concentration
            11: action is on flask (width change)
            12: action is on concentrationlab
            13: action is on pdf
            14: break
    """
    def __init__(self, settings):
        self._name = 'lstm sequencer'
        self._notation = 'lstmsqcr'
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
            'concentrationlab',
            'pdf',
            'break'
        ]
        self._click_interval = 0.05
        self._sampling_frequency = 0.2
        
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
            
            'flask': 'width',
            
            'pdf': 'pdf',
            'concentrationlab': 'concentrationlab',
        }
        
        self._index_vector = {
            0:'absorbance',
            1:'observed',
            2:'red',
            3:'green',
            4:'notrednotgreen_solution',
            5:'ruler',
            6:'rulernotmeasuring',
            7:'wl520',
            8:'wlnot520',
            9:'other',
            10:'concentration',
            11:'width',
            12:'concentrationlab',
            13:'pdf',
            14:'break'
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
            'concentration':10,
            'width':11,
            'concentrationlab':12,
            'pdf':13,
            'break':14
        }
    
        self._vector_size = 15
        self._vector_states = 9
        self._break_state = 14
        
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
        if attributes[4] == 'concentrationlab':
            vector[12] = 1
            return list(vector)

        if attributes[0] == 'absorbance':
            vector[0] = 1
        else:
            vector[1] = 1
            
        if attributes[1] == 'red':
            vector[2] = 1
        elif attributes[1] == 'green':
            vector[3] = 1
        else:
            vector[4] = 1
            
        if attributes[2] == 'wl':
            vector[7] = 1
        else:
            vector[8] = 1
            
        if attributes[3]:
            vector[5] = 1
        else:
            vector[6] = 1
            
        vector[self._vector_index[attributes[4]]] = 1

        return list(vector)
        
    def get_sequences(self, simulation:Simulation, lid:str) -> Tuple[list, list, list]:
        simulation.close()
        self._load_sequences(simulation)
        begins = [x for x in self._begins]
        ends = [x for x in self._ends]
        labels = [x for x in self._labels]
        begins, ends, labels = self._change_magnifier_states(begins, ends, labels, simulation)
        
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
        for timestep in np.arange(0, ends[-1], self._sampling_frequency):
            # observable or not
            mm, measure_begin, measure_end = self._state_return(measure_begin, measure_end, timestep)
            
            # transmittance or absorbance
            dependent_var_ts, dependent_variable, m_obs = self._get_value_timestep(dependent_var_ts, dependent_variable, timestep)
            
            # ruler measuring
            rm, ruler_begin, ruler_end = self._state_return(ruler_begin, ruler_end, timestep)
            
            # sol colour
            solution_timestamps, solution_values, sv = self._get_value_timestep(solution_timestamps, solution_values, timestep)
            
            # wavelength
            wl_timestamps, wl_values, wl = self._get_value_timestep(wl_timestamps, wl_values, timestep)

            lab, begins, ends, labels = self._label_return(begins, ends, labels, timestep)
            
            vector = self._fill_vector([m_obs, sv, wl, rm, lab])
            
            new_labels.append(vector)
            new_begins.append(timestep)
            new_ends.append(timestep + self._sampling_frequency)
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

    def _change_magnifier_states(self, begins:list, ends:list, labels: list, simulation:Simulation) -> Tuple[list, list, list]:
        """While the magnifier is moving, the state of the simulation may change (the transmittance/absorbance might change) as the magnifier
        goes in front of the laser or not

        Args:
            begins (list): beginning timestamps
            ends (list): ends timestamps
            labels (list): labels

        Returns:
            Tuple[list, list, list]: updated begins, updated ends, updated labels
        """
        up_begins = []
        up_ends = []
        up_labels = []

        dependent_variable, dependent_var_ts = self.get_absorbance_transmittance_nothing(simulation)
        dependent_var_ts = np.array(dependent_var_ts)

        for i, beg in enumerate(begins):
            if labels[i] != 'other':
                up_begins.append(beg)
                up_ends.append(ends[i])
                up_labels.append(labels[i])

            else:
                states = np.where((dependent_var_ts >= beg) & (dependent_var_ts < ends[i]))
                states = [dependent_var_ts[s] for s in states]
                old_begin = beg
                if len(states[0]) > 0:
                    for s in states[0]:
                        up_begins.append(old_begin)
                        up_ends.append(s)
                        up_labels.append('other')
                        old_begin = s
                    up_begins.append(old_begin)
                    up_ends.append(ends[i])
                    up_labels.append('other')

        return up_begins, up_ends, up_labels

    def _process_wl(self, wl_values: list) -> list:
        wl_values = ['wl' if '520' in str(wl) else 'no_wl' for wl in wl_values]
        return wl_values

    def _label_return(self, begin: list, end: list, labels:list, timestep: float) -> Tuple[bool, list, list]:
        if begin == [] or end == []:
            return 'no action', begin, end, labels

        elif timestep >= begin[0] and timestep < end[0]:
            return labels[0], begin, end, labels
        
        elif timestep < begin[0]:
            return 'break', begin, end, labels

        elif timestep >= end[0]:
            begin = begin[1:]
            end = end[1:]
            labels = labels[1:]
            return self._label_return(begin, end, labels, timestep)
    
    def _proces_absorbance_other(self, metric_observed: bool, absorbance: str):
        if metric_observed and 'absorbance':
            return 'absobserved'
        else:
            return 'notobserved'
            
            