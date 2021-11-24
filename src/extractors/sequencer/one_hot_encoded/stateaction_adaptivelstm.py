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
from extractors.sequencer.one_hot_encoded.stateaction_encodedlstm import StateActionLSTMEncoding
class StateActionAdaptiveLSTM(Sequencing):
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
        self._sampling_interval = settings['data']['pipeline']['sequencer_interval']
        self._sequencer = StateActionLSTMEncoding(settings)
        self._break_threshold = self._settings['data']['pipeline']['break_threshold']
        self._break_filter = BreakFilter(self, self._break_threshold)
        self._load_labelmap()
        
    def _load_labelmap(self):
        self._label_map = {
            'laser': 'other',
            'restarts': 'other',
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
            'pdf': 16
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
        
    def get_sequences(self, simulation:Simulation) -> Tuple[list, list, list]:
        """Returns the adaptive sequences. Each click press and release produces one vector.
        If that vector is *n* times longer than the sampling rate, it's duplicated into *n*
        copies.

        self._sequencer is the normal encoded sequencer. Its use is to make it the basis for our duplication

        Args:
            simulation (Simulation): simulation to do this for.

        Returns:
            Tuple[labels, begins, ends]: [description]
        """
        labels, begins, ends = self._sequencer.get_sequences(simulation)
        if len(labels) == 0:
            return [], [], []
        break_threshold = self._break_filter.get_threshold(begins, ends, self._break_threshold)
        if self._settings['data']['pipeline']['sequencer_dragasclick']:
            labels, begins, ends = self._filter_clickasdrag(labels, begins, ends, break_threshold)

        new_labels, new_begins, new_ends = [], [], []

        for i in range(len(labels)):
            labs, bes, ees = self._duplicate_events(labels[i], begins[i], ends[i])
            new_labels = new_labels + labs
            new_begins = new_begins + bes
            new_ends = new_ends + ees

            if i < len(begins) - 1:
                # if begins[i+1] - ends[i] > self._break_threshold:
                break_vector = list(np.zeros(self._vector_size))
                break_vector[0:self._vector_states] = new_labels[-1][0:self._vector_states]
                labs, bes, ees = self._duplicate_events(break_vector, ends[i], begins[i+1])
                new_labels = new_labels + labs
                new_begins = new_begins + bes
                new_ends = new_ends + ees

        return new_labels, new_begins, new_ends

    def _filter_clickasdrag(self, labels, begins, ends, break_threshold):
        new_labels, new_begins, new_ends = [labels[0]], [begins[0]], [ends[0]]
        for i in range(1, len(labels)):
            if labels[i] != new_labels[-1]:
                new_labels.append(labels[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])
            elif begins[i] - new_ends[-1] < break_threshold:
                new_ends[-1] = ends[i]
            else:
                new_labels.append(labels[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])
        return labels, begins, ends
            
    def _duplicate_events(self, label, begin, end):
        if end - begin < self._sampling_interval:
            return [label], [begin], [end]
        else:
            duplication = int(np.round_((end - begin) / self._sampling_interval))
            labels = [label for i in range(duplication)]
            begins = [begin + (self._sampling_interval * i) for i in range(duplication)]
            ends = [begin + (self._sampling_interval * (i + 1)) for i in range(duplication)]

            return labels, begins, ends
        
