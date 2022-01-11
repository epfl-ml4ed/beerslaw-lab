import time
import numpy as np
import pandas as pd
from typing import Tuple

from extractors.parser.simulation_parser import Simulation
from extractors.parser.checkbox_object import Checkbox
from extractors.parser.event_object import Event
from extractors.parser.simulation_object import SimObjects
from extractors.parser.value_object import SimCharacteristics

class Sequencing:
    """This class aims at returning 3 arrays. One with the starting time of each action, one with the ending time of each action, and one with the labels of the actual action.
    Each subclass from sequencing returns those 3 arrays, but with different labels.
    """
    def __init__(self, settings):
        self._name = 'sequencer'
        self._notation = 'sqcr'
        self._states = ['not initialised']
        self._click_interval = 0.05
        self._settings = settings
        self._load_labelmap()
        
    def get_name(self):
        return self._name

    def get_notation(self):
        return self._notation
        
    def get_states(self):
        return [x for x in self._states]

    def get_settings(self):
        return dict(self._settings)

    def set_rankings(self, rankings:pd.DataFrame):
        self._rankings = rankings
    
    def _load_labelmap(self):
        """Should be customised per sequencer, according to the label we want to give to each action
        """
        self._label_map = {
            'wavelength_radiobox': 'other',
            'laser': 'other',
            'preset': 'other',
            'wl_variable': 'other',
            'minus_wl_slider': 'wavelength',
            'wl_slider': 'wavelength',
            'plus_wl_slider': 'wavelength',
            'solution_menu': 'solution',
            'minus_concentration_slider': 'concentration',
            'plus_concentration_slider': 'concentration',
            'concentration_slider': 'concentration',
            'flask': 'flask',
            'ruler': 'other',
            'pdf': 'pdf',
            'restarts': 'restart',
            'concentrationlab': 'concentrationlab',
            'transmittance_absorbance': 'other',
            'magnifier_position': 'other'
        }
        
    def _load_sequences(self, simulation: Simulation):
        """Takes all the elements and returns the sequences: lables + timestamps
        """
        
        # values of the simulation
        self._wavelength = simulation.get_wavelength()
        self._width = simulation.get_width()
        self._concentration = simulation.get_concentration()
        self._solution = simulation.get_solution()
        self._measure_recorded = simulation.get_measure_recorded()
        values_displayed, timestamps_displayed = simulation.get_measure_display()
        rec, not_rec = self._process_measure_observed(values_displayed, timestamps_displayed, simulation.get_last_timestamp())
        self._measure_displayed, self._measure_not_displayed = rec, not_rec
        self._metric = simulation.get_metric()
        self._magnifier_position = simulation.get_magnifier_position()
        self._ligh_activated = simulation.get_light()
        ruler_pos, ruler_ts = simulation.get_ruler_position()
        ruler_pos = self._process_ruler_measuring(ruler_pos, ruler_ts, simulation.get_last_timestamp())
        self._ruler_measuring, self._ruler_not_measuring = ruler_pos
        # actions
        labels = []
        begins = []
        ends = []
        
        # wavelength radiobox
        wavelength_radiobox = simulation.get_wavelength_radiobox()
        begins, ends, labels = self._process_radiobox_seq(wavelength_radiobox, begins, ends, labels, self._label_map['wavelength_radiobox'])
        # transmittance / absorbance radiobox
        transmittance_absorbance = simulation.get_checkbox_transmittance()
        #debug
        begins, ends, labels = self._process_radiobox_seq(transmittance_absorbance, begins, ends, labels, self._label_map['transmittance_absorbance'])
        # magnifier interactions
        magnif = dict(self._magnifier_position)
        begins, ends, labels = self._process_dragging(magnif, begins, ends, labels, self._label_map['magnifier_position'])
        # laser
        laser = simulation.get_laser()
        begins, ends, labels = self._process_radiobox_seq(laser, begins, ends, labels, self._label_map['laser'])
        
        # preset variable
        preset = simulation.get_wl_preset()
        if len(preset['begin']) > 1:
            begins = begins + preset['begin'][1:-1]
            ends = ends + list(np.array(preset['begin'][1:-1]) + self._click_interval)
            labels = labels + [self._label_map['preset'] for l in preset['begin'][1:-1]]
        # wavelength variable
        wlvar = simulation.get_wl_variable()
        begins = begins + wlvar['begin'][:-1]
        ends = ends + list(np.array(wlvar['begin'][:-1]) + self._click_interval)
        labels = labels + [self._label_map['wl_variable'] for l in wlvar['begin'][:-1]]
        # # minus wl slider
        # minus_wl_slider = simulation.get_wl_slider_minus()
        # begins, ends, labels = self._process_firing(minus_wl_slider, begins, ends, labels, self._label_map['minus_wl_slider'])
        # wl slider
        wl_slider = simulation.get_wl_slider()
        begins, ends, labels = self._process_dragging(wl_slider[0], begins, ends, labels, self._label_map['wl_slider'])
        # plus wl slider
        # plus_wl_slider = simulation.get_wl_slider_plus()
        # begins, ends, labels = self._process_firing(plus_wl_slider, begins, ends, labels, self._label_map['plus_wl_slider'])
        # solution menu
        solution_menu = simulation.get_solution_menu()
        begins, ends, labels = self._process_dragging(solution_menu[0], begins, ends, labels, self._label_map['solution_menu'])
        begins, ends, labels = self._process_firing(solution_menu[1], begins, ends, labels, self._label_map['solution_menu'])
        # minus concentration slider
        minus_concentration_slider = simulation.get_concentration_slider_minus()
        begins, ends, labels = self._process_firing(minus_concentration_slider, begins, ends, labels, self._label_map['minus_concentration_slider'])
        # plus concentration slider
        plus_concentration_slider = simulation.get_concentration_slider_plus()
        begins, ends, labels = self._process_firing(plus_concentration_slider, begins, ends, labels, self._label_map['plus_concentration_slider'])
        # concentration slider
        concentration_slider = simulation.get_concentration_slider()
        begins, ends, labels = self._process_dragging(concentration_slider[0], begins, ends, labels, self._label_map['concentration_slider'])
        begins, ends, labels = self._process_firing(concentration_slider[1], begins, ends, labels, self._label_map['concentration_slider'])
        # flask or width dragging
        flask = simulation.get_flask()
        begins, ends, labels = self._process_dragging(flask, begins, ends, labels, self._label_map['flask'])
        # ruler
        ruler = simulation.get_ruler()
        begins, ends, labels = self._process_dragging(ruler, begins, ends, labels, self._label_map['ruler'])
        # restarts
        restarts = simulation.get_restarts()
        restarts = {'timestamps': restarts}
        begins, ends, labels = self._process_firing(restarts, begins, ends, labels, self._label_map['restarts'])
        
        # pdf
        pdf = simulation.get_pdf()
        begins, ends, labels = self._process_dragging(pdf, begins, ends, labels, self._label_map['pdf'])
        # concentration lab
        concentrationlab = simulation.get_concentrationlab_actions()
        begins, ends, labels = self._process_dragging(concentrationlab[0], begins, ends, labels, self._label_map['concentrationlab'])
        begins, ends, labels = self._process_firing(concentrationlab[1], begins, ends, labels, self._label_map['concentrationlab'])
        
        indices = np.argsort(begins)
        bs = [begins[i] for i in indices]
        es = [ends[i] for i in indices]
        ls = [labels[i] for i in indices]

        bs, es, ls = self._clean_closing(bs, es, ls, simulation.get_last_timestamp())
        self._begins = bs
        self._ends = es
        self._labels = ls
        # for i in range(len(self._labels)):
        #     print('* {} {} {}'.format(self._begins[i], self._ends[i], self._labels[i]))
        
    # Process sequences into begins, ends and labels list
    def _process_measure_observed(self, values: list, timestamps: list, last_timestamp: float) -> Tuple[dict, dict]:
        """Returns whether the measure (transmisttance or absorbance) was displayed or not
        Args:
            values (list): values displayed
            timestamps (list): times recorded
        Returns:
            Tuple[dict, dict]: 
                - beginning [begin] and end [end] timestamps of when the measures were displayed 
                - beginning [begin] and end [end] timestamps of when the measures were not displayed
        """
        # processing the values
        values = [str(v).replace('â€ª', '') for v in values]
        values = [str(v).replace('%â€¬', '') for v in values]
        values = [self._measure_displayed_processing(v) for v in values] 
        
        # summarises into recorded and not recorded
        vs = [values[0]]
        ts = [timestamps[0]]
        for i, item in enumerate(values[1:]):
            if item == vs[-1]:
                continue
            else:
                vs.append(item)
                ts.append(timestamps[i + 1])
                
        # creates the dictionary to be returned
        not_recording = {'begin': [], 'end': []}
        recording = {'begin': [], 'end': []}
        try:
            for i, v in enumerate(vs):
                if v == 'not_recording':
                    not_recording['begin'].append(ts[i])
                    not_recording['end'].append(ts[i+1])
                else:
                    recording['begin'].append(ts[i])
                    recording['end'].append(ts[i+1])
        except IndexError:
            if v == 'not_recording':
                not_recording['end'].append(last_timestamp)
            else:
                recording['end'].append(last_timestamp)
        return recording, not_recording
        
    def _measure_displayed_processing(self, value: str) -> str:
        try:
            float(value)
            return 'recording'
        except TypeError:
            return 'not_recording'
        except ValueError:
            return 'not_recording'
    
    def _process_ruler_measuring(self, values: list, timestamps: list, last_timestamp: float) -> list:
        """Checks whether the ruler is measuring something relevant
        Args:
            values ([type]): position of the ruler
            timestamps ([list]): time when the position changed
            last_timestamp ([float]): time when the simulation was ended
        Returns:
            [type]: ruler of no ruler for each position, if they are measuring something relevant or not respectively
        """
        new_values = []
        for value in values:
            x = value[0]
            y = value[1]
            
            v = ''
            
            if 3.236 <= x and x <= 3.372:
                if 0.044 <= y and y <= 3.572:
                    v = 'ruler'
                else:
                    v = 'no_ruler'
            else:
                v = 'no_ruler'
            
            new_values.append(v)
          
        vs = [new_values[0]]
        ts = [timestamps[0]]
        for i, item in enumerate(new_values[1:]):
            if item == vs[-1]:
                continue
            else:
                vs.append(item)
                ts.append(timestamps[i+1])
                
        not_measuring = {'begin': [], 'end': []}
        measuring = {'begin': [], 'end': []}
        try:
            for i, v in enumerate(vs):
                if v == 'no_ruler':
                    not_measuring['begin'].append(ts[i])
                    not_measuring['end'].append(ts[i+1])
                else:
                    measuring['begin'].append(ts[i])
                    measuring['end'].append(ts[i+1])
        except IndexError:
            if v == 'no_ruler':
                not_measuring['end'].append(last_timestamp)
            else:
                measuring['end'].append(last_timestamp)
        return measuring, not_measuring
    
    def _process_radiobox_seq(self, ons_offs: Tuple[dict, dict], begins: list, ends: list, labels: list, label: list) -> Tuple[list, list, list]:
        ons = ons_offs[0]
        offs = ons_offs[1]
        b = [] + ons['begin'] + ons['end'] + offs['begin'] + offs['end']
        b.sort()
        
        if len(b) == 0:
            return 0
        
        begin = [b[0]]
        for i in range(1, len(b)):
            if b[i] != begin[-1]:
                begin.append(b[i])
                
        while (len(begin) > 0) and (begin[0] == 0):
            begin = begin[1:]
        end = list(np.array(begin) + self._click_interval)
        labs = [label for x in begin]
        begins = begins + begin
        ends = ends + end
        labels = labels + labs
        return begins, ends, labels
       
    def _process_firing(self, firing:dict, begins:list, ends:list, labels:list, label:str) -> Tuple[list, list, list]:
        begin = [f for f in firing['timestamps']]
        end = list(np.array(begin) + self._click_interval)
        lab = [label for l in begin]
        
        begins = begins + begin
        ends = ends + end
        labels = labels + lab
        
        return begins, ends, labels
        
    def _process_dragging(self, dragging:dict, begins:list, ends:list, labels:list, label:str) -> Tuple[list, list, list]:
        begins = begins + dragging['begin']
        ends = ends + dragging['end']
        labels = labels + [label for x in dragging['begin']]
        return begins, ends, labels
        
    def _get_value_timestep(self, timestamps: list, values: list, timestep: float) -> Tuple[list, list, str]:
        """For a given variable, returns what the value of that variable was at a particular timestep. We assume that the timesteps are processed chronologically, therefore, we send back partial lists (delete values and timestamps that already passed) for computationnal gains
        Args:
            timestamps ([list]): list of timestamps of when the variable was changed
            values ([list]): list of the values of when the variable was changed
            timestep ([float]): timestep we want the value at
        Returns:
            timestamps ([list]): timestamps cropped such that passed timesteps are taken away
            values ([list]): crossed values
            value: value of that variable at that timestep
        """
        if len(timestamps) == 1 and timestamps[0] <= timestep:
            return timestamps, values, values[0]
        if timestamps[0] <= timestep and timestamps[1] > timestep:
            return timestamps, values, values[0]
        elif timestamps[0] < timestep and timestamps[1] <= timestep:
            return self._get_value_timestep(timestamps[1:], values[1:], timestep)
        elif len(timestamps) == 2 and timestamps[0] == timestamps[1]:
            return [timestamps[0]], [values[0]], values[0]
        
    def _state_return(self, begin: list, end: list, timestep: float) -> Tuple[bool, list, list]:
        if begin == [] or end == []:
            return False, begin, end

        elif timestep >= begin[0] and timestep < end[0]:
            return True, begin, end
        
        elif timestep < begin[0]:
            return False, begin, end

        elif timestep >= end[0]:
            begin = begin[1:]
            end = end[1:]
            return self._state_return(begin, end, timestep)
    
    def _clean_other(self, event: str) -> str:
        if 'other' not in event:
            return event
        else:
            return 'other'

    def _clean_closing(self, begins:str, ends:list, labels:list, last_timestamp:float) -> Tuple[list, list, list]:
        bs, es, ls = [b for b in begins], [e for e in ends], [l for l in labels]
        # for i, b in enumerate(begins):
        #     if b < last_timestamp  and labels[i] != 'other':
        #         bs.append(b)
        #         es.append(ends[i])
        #         ls.append(labels[i])

        if len(ls) > 1:
            if ls[-1] in ['concentration', 'solution'] and ls[-1] == ls[-2]:
                if bs[-1] == bs[-2] and es[-1] == es[-2]:
                    bs = bs[:-2]
                    es = es[:-2]
                    ls = ls[:-2]
        return bs, es, ls

    def get_absorbance_transmittance_nothing(self, sim: Simulation):
        """Returns the timesteps and values of when the absorbance was displayed, 
        whether the transmittance was displayed, or whether nothing was displayed

        Args:
            sim (Simulation): Simulation

        Return:
            - labels (list): list of labels [transmittance, absorbance, none]
            - timesteps (list): ts of potential changes
        """
        values_displayed, timestamps_displayed = sim.get_measure_display()
        values_displayed = [str(v).replace('â€ª', '') for v in values_displayed]
        values_displayed = [str(v).replace('%â€¬', '') for v in values_displayed]

        values = []
        ts = []
        for i, val in enumerate(values_displayed):
            ts.append(timestamps_displayed[i])
            if '%' in val:
                values.append('transmittance')
            elif val == '-':
                values.append('none')
            else:
                values.append('absorbance')

        labels = [values[0]]
        timesteps = [ts[0]]
        for i, v in enumerate(values[1:]):
            if v != labels[-1]:
                labels.append(v)
                timesteps.append(ts[i+1])

        return labels, timesteps
    
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
            if labels[i] != 'tools':
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
                else:
                    up_begins.append(beg)
                    up_ends.append(ends[i])
                    up_labels.append(labels[i])

        return up_begins, up_ends, up_labels

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

    def _filter_concentrationlab(self, labels, begins, ends):
        """Filters the events "concentrationlab" such that recording each click in that simulation, we record being in this simulation an action.

        Args:
            labels ([type]): labels
            begins ([type]): beginning timestamps
            ends ([type]): end timestamps

        Returns:
            [type]: labels, begins, ends
        """
        new_labels, new_begins, new_ends = [labels[0]], [begins[0]], [ends[0]]
        for i in range(1, len(labels)):
            if labels[i] == 'concentrationlab' and new_labels[-1] == 'concentrationlab':
                if new_ends[-1] < ends[i]:
                    new_ends[-1] = ends[i]

            else:
                new_labels.append(labels[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])
        return new_labels, new_begins, new_ends

    def _filter_doubleeveents(self, labels, begins, ends):
        """Sometimes, events are registered as double in the event logs. We filter them here.

        Args:
            labels ([type]): [description]
            begins ([type]): [description]
            ends ([type]): [description]
        """
        new_labels, new_begins, new_ends = [labels[0]], [begins[0]], [ends[0]]
        for i in range(len(labels)):
            if labels[i] == new_labels[-1] and begins[i] == new_begins[-1] and ends[i] == new_ends[-1]:
                continue
            if begins[i] == ends[i]:
                continue
            if begins[i] == new_begins[-1] and ends[i] == new_ends[-1] and (labels[i] == 'other' or new_labels[-1] == 'other'):
                if new_labels[-1] == 'other':
                    new_labels[-1] = labels[i]
            else:
                new_labels.append(labels[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])
        return new_labels, new_begins, new_ends

    def _filter_overlaps(self, labels, begins, ends):
        """It is possible for the students to open the solution menu, then the pdf menu, 
        the close the pdf, then still have the solution menu. It is also possible that the parsing is having some issues, and some overlaps
        are observed.
        The parsing has not taken that into account, as it is true that 
        the solution menu remains opened when the pdf menu is on.
        We correct it here:
        [solution opens, pdf opens, ..., pdf closes, solution closes]
        is turned into
        [solution opens, solution closes, pdf opens, pdf closes, solution opens, solution closes]

        Args:
            labels ([type]): [description]
            begins ([type]): [description]
            ends ([type]): [description]
        """
        new_labels, new_begins, new_ends = [], [], []
        index_skip = -1
        for i in range(len(labels)):
            if i < index_skip:
                continue
            overlap_labels, overlap_begins, overlap_ends = [labels[i]], [begins[i]], [ends[i]]
            for j in range(i+1, len(labels)):
                if begins[j] < ends[i]:
                    print('** problem')
                    print(labels[i], begins[i], ends[i])
                    print(labels[j], begins[j], ends[j])
                    overlap_labels.append(labels[j])
                    overlap_begins.append(begins[j])
                    overlap_ends.append(ends[j])
                else:
                    break

            if len(overlap_labels) > 1:
                sort_indices = np.argsort(overlap_begins)
                new_ov_labels = [overlap_labels[idx] for idx in sort_indices]
                new_ov_begins = [overlap_begins[idx] for idx in sort_indices] + [ends[i]]
                new_ov_ends = [overlap_ends[idx] for idx in sort_indices]

                new_labels.append(new_ov_labels[0])
                new_begins.append(new_ov_begins[0])
                new_ends.append(new_ov_begins[1])
                for k in range(1, len(overlap_labels)):
                    new_labels.append(new_ov_labels[k])
                    new_begins.append(new_ov_begins[k])
                    new_ends.append(new_ov_ends[k])

                    if new_ov_begins[k+1] >= new_ov_ends[k]:
                        new_labels.append(new_ov_labels[0])
                        new_begins.append(new_ov_ends[k])
                        new_ends.append(new_ov_begins[k+1])

                index_skip = i + len(overlap_labels)

            else:
                new_labels.append(labels[i])
                new_begins.append(begins[i])
                new_ends.append(ends[i])
        return new_labels, new_begins, new_ends

    def _filter_restarts(self, labels, begins, ends):
        """Restarts change the simulation, so some events are "fired" without anything really changing.

        Args:
            labels ([type]): [description]
            begins ([type]): [description]
            ends ([type]): [description]
        """
        indices = [i for i in range(len(labels)) if labels[i] == 'restart']
        restart_begins = [begins[i] for i in indices]
        restart_ends = [ends[i] for i in indices]
        debug_indices = [i for i in range(len(labels)) if begins[i] == 44.791]
        debug_labels = [labels[idx] for idx in debug_indices]

        new_labels = []
        new_begins = []
        new_ends = []
        for i in range(len(labels)):
            if begins[i] in restart_begins and ends[i] in restart_ends and labels[i] != 'restart':
                continue
            elif labels[i] == 'restart':
                new_labels.append('other')
            else:
                new_labels.append(labels[i])
            new_begins.append(begins[i])
            new_ends.append(ends[i])
        return new_labels, new_begins, new_ends

    def _basic_common_filtering(self, labels, begins, ends, simulation):
        labels, begins, ends = self._filter_restarts(labels, begins, ends)
        labels, begins, ends = self._filter_doubleeveents(labels, begins, ends)
        labels, begins, ends = self._filter_concentrationlab(labels, begins, ends)
        labels, begins, ends = self._filter_overlaps(labels, begins, ends)
        begins, ends, labels = self._change_magnifier_states(begins, ends, labels, simulation)

        if self._settings['data']['pipeline']['sequencer_dragasclick']:
            break_threshold = self._break_filter.get_threshold(begins, ends, self._break_threshold)
            self._break_minimum = break_threshold
            labels, begins, ends = self._filter_clickasdrag(labels, begins, ends, break_threshold)
        return labels, begins, ends

    def _timestep_normaliser(self, labels):
        normalise = lambda x: list(np.array(x) / np.sum(x))
        return [normalise(l) for l in labels]

    
    def get_sequences(self, simulation: Simulation) -> Tuple[list, list, list]:
        raise NotImplementedError