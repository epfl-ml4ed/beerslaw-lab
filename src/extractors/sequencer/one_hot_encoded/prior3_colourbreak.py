import re
import time
import numpy as np
import pandas as pd
import pickle
from typing import Tuple

from extractors.sequencer.sequencing import Sequencing
from extractors.parser.simulation_parser import Simulation
from extractors.parser.checkbox_object import Checkbox
from extractors.parser.event_object import Event
from extractors.parser.simulation_object import SimObjects
from extractors.parser.value_object import SimCharacteristics
from extractors.cleaners.break_filter import BreakFilter

class PriorColourBreakSecondsLSTM(Sequencing):
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    
    In this particular case, each feature will be made out of a vector encoding:
        - time spent on actions when the laser was green and the solution was also green
        - break when the laser was green and the solution was also green
        - time spent on actions when the laser was green and the solutino was red
        - break when the laser was green and the solution was red
        - time spent on actions when either the laser was not green or the solution was neither green nor red
        - break when the laser was not green and/or the solution was neither green nor red
        - time spent on the concentrationlab

        vector at time t:
            0: 1 if the student has no prior knowledge
            1: 1 if the student has some but not full prior knowledge
            2: 1 if the student has prior knowledge
            3: (s + vector(t-1)[0]) / vector(t) if state is green - green
            4: (s + vector(t-1)[0]) / vector(t) if state is green - green and in break
            5: (s + vector(t-1)[1]) / vector(t) green - red
            6: (s + vector(t-1)[1]) / vector(t) green - red and in break
            7: (s + vector(t-1)[2]) / vector(t) no green laser or (no green solution and no red solution)
            8: (s + vector(t-1)[2]) / vector(t) no green laser or (no green solution and no red solution) and in break
            9: (s + vector(t-1)[3]) / vector(t) concentrationlab

        => s being 0 if it's in the corresponding state, or the timing of the interaction in the current state
    """
    def __init__(self, settings):
        self._name = 'prior colourbreak seconds sequencer'
        self._notation = 'pcbss'
        self._settings = settings
        self._states = [
            'prior0',
            'prior1',
            'prior2',
            'greengreen',
            'break_greengreen',
            'greenred',
            'break_greenred',
            'nogreennored',
            'break_nogreennored',
            'noobserved',
            'break_noobserved',
            'concentrationlab'
        ]
        self._click_interval = 0.05
        
        self._load_labelmap()
        self._break_threshold = self._settings['data']['pipeline']['break_threshold']
        self._break_filter = BreakFilter(self, self._break_threshold)
        self._load_prior()

    def _load_prior(self):
        with open('../data/post_test/some_scored.pkl', 'rb') as fp:
            scored = pickle.load(fp)
            scored = scored.set_index('username')
        self._scored = scored

    def _load_labelmap(self):
        self._label_map = {
            'laser': 'action',
            'restarts': 'action',
            'transmittance_absorbance': 'action',

            'magnifier_position': 'action',
            'ruler': 'action',
            
            'wavelength_radiobox': 'action',
            'preset': 'action',
            'wl_variable': 'action',
            'minus_wl_slider': 'action',
            'wl_slider': 'action',
            'plus_wl_slider': 'action',
            
            'solution_menu': 'action',
            
            'minus_concentration_slider': 'action',
            'plus_concentration_slider': 'action',
            'concentration_slider': 'action',
            
            'flask': 'action',
            
            'pdf': 'action',

            'concentrationlab': 'concentrationlab',
        }
        
        self._index_vector = {
            0: 'prior1',
            1: 'prior2',
            2: 'prior3',
            3: 'greengreen',
            4: 'break_greengreen',
            5: 'greenred',
            6: 'break_greenred',
            7: 'nogreennored',
            8: 'break_nogreennored',
            9: 'noobserved',
            10: 'break_noobserved',
            11: 'concentrationlab',
        }
        
        self._vector_index = {
            'prior1': 0,
            'prior2': 1,
            'prior3': 2,
            'greengreen': 3,
            'break_greengreen': 4,
            'greenred': 5,
            'break_greenred': 6,
            'nogreennored': 7,
            'break_nogreennored': 8,
            'noobserved': 9,
            'break_noobserved': 10,
            'concentrationlab': 11
        }
    
        self._vector_size = len(self._vector_index)
        self._vector_states = 10
        self._break_state = -1
        
    def get_vector_size(self):
        return self._vector_size
    def get_vector_states(self):
        return self._vector_states
    def get_break_state(self):
        return self._break_state
        
    def _fill_vector(self, attributes: list, second:float, break_bool: int) -> list:
        """Vector string: [m_obs, sv, wl, rm, lab]
            second: length of the interaction
            break: whether it's an action or a break
        """
        vector = np.zeros(self._vector_size)

        if attributes[4] == 'concentrationlab':
            vector[11] = second
            return list(vector)

        if attributes[0] != 'absorbance':
            vector[9 + break_bool] = second
            return list(vector)

        if attributes[2] == 'wl' and attributes[1] == 'green':
            vector[3 + break_bool] = second
            return list(vector)

        if attributes[2] == 'wl' and attributes[1] == 'red':
            vector[5 + break_bool] = second
            return list(vector)

        else:
            vector[7 + break_bool] = second
            return list(vector)


    def get_sequences(self, simulation:Simulation, lid:str) -> Tuple[list, list, list]:
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
        new_begins = []
        new_ends = []

        cumulative_vector = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, begins[0], 0])

        # First break
        new_begins.append(0)
        new_ends.append(begins[0])
        new_labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, begins[0], 0])

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
            
            # action
            instant_vector = self._fill_vector([m_obs, sv, wl, rm, lab], ends[i] - begins[i], 0)
            cumulative_vector = np.array(cumulative_vector) + np.array(instant_vector)
            new_begins.append(begins[i])
            new_ends.append(ends[i])
            new_labels.append([cv for cv in cumulative_vector])

            # breaks
            if i+1 < len(labels):
                if begins[i + 1] - ends[i] > self._break_minimum:
                    instant_vector = self._fill_vector([m_obs, sv, wl, rm, 'break'], begins[i+1] - ends[i], 1)
                    cumulative_vector = np.array(cumulative_vector) + np.array(instant_vector)
                    new_begins.append(ends[i])
                    new_ends.append(begins[i+1])
                    new_labels.append([cv for cv in cumulative_vector])

        new_labels = self._timestep_normaliser(new_labels)
        new_labels = self._add_prior_knowledge(lid, new_labels)
        return new_labels, new_begins, new_ends
    
    def _add_prior_knowledge(self, lid:str, new_labels:list):
        """Add the prior-knowledge (in category 3) as a binary encoding at the beginning of the vector

        Args:
            lid (str): learner id of the student
            new_labels (list): final label list
        """
        nr_re = re.compile('([0-9])')
        prior = self._scored.loc[lid]['prior_3cat']
        prior = nr_re.findall(prior)
        if len(prior) > 0:
            prior = int(max(prior))
            index = prior
        else:
            year = self._scored.loc[lid]['year']
            if year == '1st':
                index = 0
            elif year == '2nd':
                index = 1
            elif year == '3rd':
                index = 2

        ls = []
        for label in new_labels:
            label[index] = 1
            ls.append(label)
        return ls

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
        wl_values = ['wl' if (500 <= int(wl) and int(wl) <= 564) else 'no_wl' for wl in wl_values]
        return wl_values
    
    def _proces_absorbance_other(self, metric_observed: bool, absorbance: str):
        if metric_observed and 'absorbance':
            return 'absobserved'
        else:
            return 'notobserved'
            
            