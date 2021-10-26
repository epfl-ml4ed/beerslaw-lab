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
            'minus_wl_slider': 'wavelength_slider',
            'wl_slider': 'wavelength_slider',
            'plus_wl_slider': 'wavelength_slider',
            
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
        # print(begins, ends, labels)
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

    def get_absorbance_transmittance_nothing(self, sim: Simulation):
        """Returns the timesteps and values of when the absorbance was displayed, 
        whether the transmittance was displayed, or whether nothing was displayed

        Args:
            sim (Simulation): Simulation

        Return:
            - labels (list): list of labels [transmittance, absorbance, none]
            - timesteps (list): ts of potential changes
        """
        transmittance, absorbance = sim.get_checkbox_transmittance()
        values_displayed, timestamps_displayed = sim.get_measure_display()
        rec, not_rec = self._process_measure_observed(values_displayed, timestamps_displayed, sim.get_last_timestamp())

        abs_begin = [a for a in absorbance['begin']]
        abs_end = [a for a in absorbance['end']]
        rec_begin = [r for r in rec['begin']]
        rec_end = [r for r in rec['end']]

        timestamps = abs_begin + abs_end + rec_begin + rec_end
        timestamps.sort()
        ts = [timestamps[0]]
        for t in timestamps[1:]:
            if t == ts[-1]:
                continue
            else:
                ts.append(t)

        values = []
        for t in ts:
            abs_bool, abs_begin, abs_end = self._state_return(abs_begin, abs_end, t)
            obs_bool, rec_begin, rec_end = self._state_return(rec_begin, rec_end, t)
            if obs_bool:
                if abs_bool:
                    values.append('absorbance')
                else:
                    values.append('transmittance')
            else:
                values.append('none')

        labels = [values[0]]
        timesteps = [ts[0]]
        for i, v in enumerate(values[1:]):
            if v != labels[-1]:
                labels.append(v)
                timesteps.append(ts[i+1])

        return labels, timesteps
