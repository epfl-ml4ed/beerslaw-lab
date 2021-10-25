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

class PlotterSequencing(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be its raw label to be used in a plotting context later
    """
    def __init__(self):
        self._name = 'plotter sequencer'
        self._notation = 'pltsqcr'

        self._click_interval = 0.05
        
        self._load_labelmap()
        
    def _load_labelmap(self):
        self._label_map = {
            'laser': 'laser',
            'ruler': 'ruler',
            'restarts': 'restarts',
            'transmittance_absorbance': 'transmittance_absorbance',
            'magnifier_position': 'magnifier',
            
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
        
    def _fill_vector(self, attributes: list) -> list:
        """Vector string: [m_obs, sv, wl, rm, lab]
        """
        vector = np.zeros(self._vector_size)
        for element in attributes:
            vector[self._vector_index[element]] = 1
        return list(vector)
        
    def get_sequences(self, simulation:Simulation) -> Tuple[list, list, list]:
        self._load_sequences(simulation)
        begins = [x for x in self._begins]
        ends = [x for x in self._ends]
        labels = [x for x in self._labels]
        
        return begins, ends, labels

    def get_ruler_timepoints(self, sim: Simulation):
        """Returns the beginning and end timestamps of when the ruler is measuring something

        Args:
            sim (Simulation): simulation for which to do it for
        Returns
            ruler_measuring: dictionaries with 'begin' and 'end' lists
        """
        ruler_pos, ruler_ts = sim.get_ruler_position()
        ruler_pos = self._process_ruler_measuring(ruler_pos, ruler_ts, sim.get_last_timestamp())
        ruler_measuring, _ = ruler_pos
        return ruler_measuring