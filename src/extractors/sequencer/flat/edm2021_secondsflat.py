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

class BinEDM2021SecondsFlat(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of a vector encoding:
        - 1 if the action is conducted while the stored energy is on
        - 1 if the action is conducted while the circuit is closed
        - time spent on the action if the action is other
            - checkboxes toggles
            - voltmeter moves
            - buttons
        - time spent on the voltage
        - time spent on the plate separation
        - time spent on the plate area
        - time spent on manipulating the circuit
        - time spent on something else than the lab
    """

    def __init__(self, settings):
        self._name = 'edm2021 seconds sequencer'
        self._notation = 'edm2021'
        self._settings = settings
        self._states = [
            'voltage',
            'se_voltage',
            'cc_voltage',
            'cc_se_voltage',
            'plateseparation',
            'se_plateseparation',
            'cc_plateseparation',
            'cc_se_plateseparation',
            'platearea',
            'se_platearea',
            'cc_platearea',
            'cc_se_platearea',
            'other',
            'se_other',
            'cc_other',
            'cc_se_other',
            'break',
        ]
        self._click_interval = 0.05
        
        self._load_labelmap()
        self._break_threshold = self._settings['data']['pipeline']['break_threshold']
        self._break_filter = BreakFilter(self, self._break_threshold)

    def _load_labelmap(self):
        self._label_map = {
            'checkbox_capacitance': 'other',
            'checkbox_topplate': 'other',
            'checkbox_storedenergy': 'other',
            'checkbox_platecharges': 'other',
            'checkbox_bargraphs': 'other',
            'checkbox_electricfield': 'other',
            'checkbox_currentdirection': 'other',
            'voltmeter_drags': 'other',
            'positiveprobes_drag': 'other',
            'negativeprobes_drag': 'other',
            'voltage_slider': 'voltage',
            'plate_separation_slider': 'plateseparation',
            'plate_area_slider': 'platearea',
            'topopen': 'other',
            'bottomopen': 'other',
            'topbattery': 'other',
            'bottombattery': 'other',
            'no_attention': 'other',
            'full_screen': 'other',
            'phetmenu': 'other',
            'phetabout': 'other'
        }
        
        self._index_vector = {
            0: 'voltage',
            1: 'se_voltage',
            2: 'cc_voltage',
            3: 'cc_se_voltage',
            4: 'plateseparation',
            5: 'se_plateseparation',
            6: 'cc_plateseparation',
            7: 'cc_se_plateseparation',
            8: 'platearea',
            9: 'se_platearea',
            10: 'cc_platearea',
            11: 'cc_se_platearea',
            12: 'other',
            13: 'se_other',
            14: 'cc_other',
            15: 'cc_se_other',
            16: 'break',
        }
        
        self._vector_index = {
            'voltage': 0,
            'se_voltage': 1,
            'cc_voltage': 2,
            'cc_se_voltage': 3,
            'plateseparation': 4,
            'se_plateseparation': 5,
            'cc_plateseparation': 6,
            'cc_se_plateseparation': 7,
            'platearea': 8,
            'se_platearea': 9,
            'cc_platearea': 10,
            'cc_se_platearea': 11,
            'other': 12,
            'se_other': 13,
            'cc_other': 14,
            'cc_se_other': 15,
            'break': 16,
        }
    
        self._vector_size = len(self._vector_index)
        self._vector_states = 2
        self._break_state = 8
        
    def get_vector_size(self):
        return self._vector_size
    def get_vector_states(self):
        return self._vector_states
    def get_break_state(self):
        return self._break_state
        
    def _fill_vector(self, attributes: list, second:float) -> list:
        """Vector string: [stored energy, circuit state, label]
            second: length of the interaction
        """

        feature = attributes[2]
        if attributes[0]:
            feature = 'se_' + feature

        if attributes[1] == 'closed':
            feature = 'cc_' + feature

        return feature

    def get_sequences(self, simulation:Simulation, lid:str) -> Tuple[list, list, list]:
        # simulation.close()
        self._load_sequences(simulation)
        begins = [x for x in self._begins]
        ends = [x for x in self._ends]
        labels = [x for x in self._labels]
        break_threshold = self._break_filter.get_threshold(begins, ends, self._break_threshold)

        # When there is stored energy
        stored_energy_values = [v for v in self._storedenergy_visibility[0]]
        stored_energy_timesteps = [ts for ts in self._storedenergy_visibility[1]]

        # when the circuit is opened or closed
        circuit_state_values = [v for v in self._circuit_state[0]]
        circuit_state_values = self._process_circuit(circuit_state_values)
        circuit_state_timesteps = [ts for ts in self._circuit_state[1]]


        new_labels = []
        new_begins = []
        new_ends = []

        for i, lab in enumerate(labels):
            # stored energy or not
            stored_energy_timesteps, stored_energy_values, se = self._get_value_timestep(stored_energy_timesteps, stored_energy_values, begins[i])

            # circuit state
            circuit_state_timesteps, circuit_state_values, cs = self._get_value_timestep(circuit_state_timesteps, circuit_state_values, begins[i])
            
            # action
            feature = self._fill_vector([se, cs, lab], ends[i] - begins[i])
            new_begins.append(begins[i])
            new_ends.append(ends[i])
            new_labels.append(feature)

            # breaks
            if i+1 < len(labels):
                if begins[i + 1] - ends[i] > break_threshold:
                    new_begins.append(ends[i])
                    new_ends.append(begins[i+1])
                    new_labels.append('break')

        return new_labels, new_begins, new_ends
    
    def _process_circuit(self, circuit_values: list):
        """Replace the values by whether the circuit is ultimately opened or closed
                - BATTERY_CONNECTED: closed
                - OPEN_CIRCUIT: opened
                - SWITCH_IN_TRANSIT: opened
        Args:
            solution_values (list): [replaced all solutions by red, green or other]
        """        
        circuit_state = {
            'BATTERY_CONNECTED': 'closed',
            'OPEN_CIRCUIT': 'opened',
            'SWITCH_IN_TRANSIT': 'opened'
        }
        circuit_values = [circuit_state[cs] for cs in circuit_values]
        return circuit_values
            
            