import os
import pickle
import yaml
import numpy as np
import pandas as pd
from typing import Tuple
import logging

from extractors.concatenator.no_concat import NoConcatenator

from extractors.aggregator.aggregator import Aggregator
from extractors.aggregator.average_aggregator import AverageAggregator
from extractors.aggregator.cumulative_aggregator import CumulativeAverageAggregator
from extractors.aggregator.flatten_aggregator import FlattenAggregator
from extractors.aggregator.no_aggregator import NoAggregator
from extractors.aggregator.normalised_aggregator import NormalisedAggregator
from extractors.aggregator.onehot_minmax_normaliser import OneHotMinMaxNormaliserAggregator
from extractors.aggregator.timestep_normaliser import TimestepNormaliser

from extractors.cleaners.break_filter import BreakFilter
from extractors.cleaners.no_break_filter import NoBreakFilter
from extractors.cleaners.cumul_break_filter import CumulBreakFilter
from extractors.cleaners.cumul_statebreak_filter import CumulStateBreakFilter
from extractors.cleaners.cumul_onehot_breaks import CumulOneHotBreakFilter
from extractors.cleaners.cumul_onehot_secondsbreaks import CumulOneHotSecondsBreakFilter

from extractors.cleaners.event_filter import EventFilter
from extractors.cleaners.no_transitions_event_filter import NoTransitionFilters
from extractors.cleaners.no_event_filter import NoEventFilter

from extractors.filters.filter import Filter
from extractors.filters.nofilter import NoFilter

from extractors.encoding.encoder import Encoder
from extractors.encoding.actionspans_encoder import ActionSpansEncoder
from extractors.encoding.onehot_encoder import OneHotEncoder
from extractors.encoding.skipgrams_encoder import SkipgramEncoder
from extractors.encoding.raw_encoder import RawEncoder
from extractors.encoding.onehot_actionspan import OneHotActionSpan

from extractors.lengths.adjuster import Adjuster
from extractors.lengths.full_sequence import FullSequence
from extractors.lengths.seconds_cropped import SecondCropper
from extractors.lengths.timestep_cropped import TimestepCropper

from extractors.sequencer.sequencing import Sequencing

from extractors.sequencer.one_hot_encoded.old.onehotminimise_sequencer import OneHotMinimiseSequencing
from extractors.sequencer.one_hot_encoded.old.binaryminimise_sequencer import Bin1HotMinimiseSequencing
from extractors.sequencer.one_hot_encoded.old.binaryextended_sequencer import Bin1hotExtendedSequencing


from extractors.sequencer.capacitor.edm2021_secondslstm import BinEDM2021SecondsLSTM

class PipelineMaker:
    """This class generates the pipeline that will take a simulation in, and returns a vector 'featurised' according to what we want.
    
    We need :
    - sequencer
    - event filter
    - break filter
    - adjuster
    - encoder
    - aggregator
    """
    
    def __init__(self, settings:dict):
        self._name = 'data pipeline maker'
        self._notation = 'datamkr'
        self._settings = dict(settings) 
        self._experiment_root = self._settings['experiment']['root_name']
        self._experiment_name = self._settings['experiment']['name']
        self._paths_settings = self._settings['paths']
        self._data_settings = self._settings['data']
        
        self._pipeline_name = self._settings['experiment']['class_name'] + '_'
        self._build_pipeline()
        
        self._get_label_map()
        
    def get_id_dictionary(self):
        return dict(self._id_dictionary)
    
    def get_index_state(self):
        return self._encoder.get_index_state()
    def get_state_index(self):
        return self._encoder.get_state_index()
    
    def get_label_map(self):
        return dict(self._label_map)
        
    def _get_label_map(self):
        self._settings['experiment']['class_name']
        if self._settings['experiment']['class_name'] == 'example':
            self._settings['experiment']['class_map'] = '../data/experiment_keys/permutation_maps/colour_binary.yaml'
            self._settings['experiment']['n_classes'] = 2
            self._settings['ML']['pipeline']['scorer'] = '2clfscorer'
        
        self._label_map = self._settings['experiment']['class_map']
        with open(self._label_map) as fp:
            self._label_map = yaml.load(fp, Loader=yaml.FullLoader)
            
    def get_sequencer(self):
        return self._sequencer
        
    def _choose_sequencer(self):

        if self._data_settings['pipeline']['sequencer'] == 'example':
            self._sequencer = BinEDM2021SecondsLSTM(self._settings)
            self._sequencer_path = 'binary_edm2021'

        self._pipeline_name += self._data_settings['pipeline']['sequencer']
        self._sequenced_directory = self._paths_settings['sequenced_simulations'] + self._sequencer_path + '/'
            
    def _concatenate_sequences(self):
        if self._data_settings['pipeline']['concatenator']['type'] == 'noconcat':
            concat = NoConcatenator()
            
        concat.concatenate()
        
    def _load_sequences(self):    
        # Retrieve dictionary containing all sequence paths and details
        with open(self._sequenced_directory + 'id_dictionary.pkl', 'rb') as fp:
            self._id_dictionary = pickle.load(fp)
            self._id_indices = list(self._id_dictionary['sequences'].keys())
        self._sequenced_files = os.listdir(self._sequenced_directory)

    def _filter_data(self):
        if self._data_settings['pipeline']['demographic_filter'] == 'nofilt':
            chemlab = NoFilter(self._settings, self._id_dictionary)
            self._id_dictionary = chemlab.filter_data()

            with open('../notebooks/debug_filter_data.pkl', 'wb') as fp:
                pickle.dump(self._id_dictionary, fp)
        
    def _choose_event_filter(self):
        self._pipeline_name += self._data_settings['pipeline']['event_filter']
        if self._data_settings['pipeline']['event_filter'] == 'nofilt':
            self._event_filter = NoEventFilter()
            
        elif self._data_settings['pipeline']['event_filter'] == 'notrans':
            self._event_filter = NoTransitionFilters()
            
    def _choose_break_filter(self):
        break_threshold = self._data_settings['pipeline']['break_threshold']
        self._pipeline_name += self._data_settings['pipeline']['break_filter']
        if self._data_settings['pipeline']['break_filter'] == 'cumulbr':
            self._break_filter = CumulBreakFilter(self._sequencer, break_threshold)
            
        elif self._data_settings['pipeline']['break_filter'] == 'cumulstbr':
            self._break_filter = CumulStateBreakFilter(self._sequencer, break_threshold)
            
        elif self._data_settings['pipeline']['break_filter'] == 'cumul1hotbr':
            self._break_filter = CumulOneHotBreakFilter(self._sequencer, break_threshold)
            
        elif self._data_settings['pipeline']['break_filter'] == 'nobrfilt':
            self._break_filter = NoBreakFilter(self._sequencer, break_threshold)

        elif self._data_settings['pipeline']['break_filter'] == 'cumulseconds':
            self._break_filter = CumulOneHotSecondsBreakFilter(self._sequencer, break_threshold)
            
    def _choose_lengther(self):
        self._pipeline_name += self._data_settings['pipeline']['adjuster']
        if self._data_settings['pipeline']['adjuster'] == 'full':
            self._adjuster = FullSequence()
            
        elif self._data_settings['pipeline']['adjuster'] == 'scrop':
            self._pipeline_name += '_' + str(self._data_settings['adjuster']['limit'])
            self._adjuster = SecondCropper()
            
        elif self._data_settings['pipeline']['adjuster'] == 'tscrp':
            self._pipeline_name += '_' + str(self._data_settings['adjuster']['limit'])
            self._adjuster = TimestepCropper()
            
    def _choose_encoder(self):
        self._pipeline_name += self._data_settings['pipeline']['encoder']
        if self._data_settings['pipeline']['encoder'] == 'actionspan':
            self._encoder = ActionSpansEncoder(self._sequencer, self._settings)
        
        elif self._data_settings['pipeline']['encoder'] == '1hot':
            self._encoder = OneHotEncoder(self._sequencer, self._settings)
            
        elif self._data_settings['pipeline']['encoder'] == 'sgenc':
            self._encoder = SkipgramEncoder(self._sequencer, self._settings)
            
        elif self._data_settings['pipeline']['encoder'] == 'raw':
            self._encoder = RawEncoder(self._sequencer, self._settings)
            
        elif self._data_settings['pipeline']['encoder'] == '1hotactionspan':
            self._encoder = OneHotActionSpan(self._sequencer, self._settings)
            
    def _choose_aggregator(self):
        self._pipeline_name += self._data_settings['pipeline']['aggregator']
        if self._data_settings['pipeline']['aggregator'] == 'aveagg':
            self._aggregator = AverageAggregator()
        
        elif self._data_settings['pipeline']['aggregator'] == 'cumulaveagg':
            self._aggregator = CumulativeAverageAggregator()
            
        elif self._data_settings['pipeline']['aggregator'] == 'flatagg':
            self._aggregator = FlattenAggregator()
            
        elif self._data_settings['pipeline']['aggregator'] == 'noagg':
            self._aggregator = NoAggregator()
            
        elif self._data_settings['pipeline']['aggregator'] == 'normagg':
            self._aggregator = NormalisedAggregator()

        elif self._data_settings['pipeline']['aggregator'] == 'minmax':
            self._aggregator = OneHotMinMaxNormaliserAggregator(self._sequencer)

        elif self._data_settings['pipeline']['aggregator'] == 'tsnorm':
            self._aggregator = TimestepNormaliser(self._sequencer)
        
    def _build_pipeline(self):
        self._choose_sequencer()
        self._concatenate_sequences()
        self._load_sequences()
        self._filter_data()
        self._pipeline_name += '_'
        self._choose_event_filter()
        self._pipeline_name += '_'
        self._choose_break_filter()
        self._pipeline_name += '_'
        self._choose_lengther()
        self._pipeline_name += '_'
        self._choose_encoder()
        self._pipeline_name += '_'
        self._choose_aggregator()
        
        os.makedirs('../data/features/' + self._pipeline_name, exist_ok=True)
        self._features_path = '../data/features/' + self._pipeline_name + '/features.pkl'
        os.makedirs('../data/labels/' + self._pipeline_name, exist_ok=True)
        self._labels_path = '../data/labels/' + self._settings['experiment']['class_name'] + '_labels.pkl'
        
    def _save_index_maps(self):
        state_index_path = '../data/features/' + self._pipeline_name + '/state_index.pkl'
        with open(state_index_path, 'wb') as fp:
            pickle.dump(self._encoder.get_state_index(), fp)
        state_index_path = '../experiments/' + self._experiment_root + '/state_index.pkl'
        with open(state_index_path, 'wb') as fp:
            pickle.dump(self._encoder.get_state_index(), fp)
            
        index_state_path = '../data/features/' + self._pipeline_name + '/index_state.pkl'
        with open(index_state_path, 'wb') as fp:
            pickle.dump(self._encoder.get_index_state(), fp)
        index_state_path = '../experiments/' + self._experiment_root + '/index_state.pkl'
        with open(index_state_path, 'wb') as fp:
            pickle.dump(self._encoder.get_index_state(), fp)
    
    def _save_experiment_maps(self):
        state_index_path = '../experiments/' + self._experiment_root 
        state_index_path += self._experiment_name + '/state_index.pkl'
        with open(state_index_path, 'wb') as fp:
            pickle.dump(self._encoder.get_state_index(), fp)
            
        index_state_path = '../experiments/' + self._experiment_root 
        index_state_path += self._experiment_name + '/index_state.pkl'
        with open(index_state_path, 'wb') as fp:
            pickle.dump(self._encoder.get_index_state(), fp)
        
    def _build_feature(self, sim_seq:dict) -> list:
        logging.debug('ori: {}'.format(sim_seq))
        seq, begin, end, last = sim_seq['sequence'], sim_seq['begin'], sim_seq['end'], sim_seq['last_timestamp']
        seq, begin, end = self._event_filter.filter_events(seq, begin, end)
        logging.debug('event filtered: {}'.format(seq))
        seq, begin, end = self._break_filter.inpute_all_breaks(seq, begin, end)
        logging.debug('break filtered: {}'.format(seq))
        seq, begin, end = self._adjuster.adjust_sequence(seq, begin, end, self._data_settings['adjuster']['limit'])
        logging.debug('adjuster: {}'.format(seq))
        seq = self._encoder.encode_sequence(seq, begin, end)
        logging.debug('encoder: {}'.format(seq))
        seq = self._aggregator.aggregate(seq)
        logging.debug('aggregation: {}'.format(seq))
        logging.debug('')
        return seq
    
    def build_data(self) -> Tuple[list, list]:
        """Builds the data

        Returns:
            sequences: the features
            labels: the labels
            indices: the included indices from id_dictionary
            id_dictionary: the dictionary including all of the details
        """
        try:
            with open(self._features_path, 'rb') as fp:
                sequences = pickle.load(fp)
            with open(self._labels_path, 'rb') as fp:
                labels = pickle.load(fp)
            self._save_experiment_maps()
            return sequences, labels, self._id_dictionary
        except FileNotFoundError:
            # self._save_index_maps()
            sequences = []
            labels = []
            indices = []
            for index in self._id_dictionary['sequences']:
                seq_path = self._id_dictionary['sequences'][index]['path']
                with open(seq_path, 'rb') as fp:
                    sim_seq = pickle.load(fp)
                seq = self._build_feature(sim_seq)
                perm = str(sim_seq['permutation'])
                if perm in self._label_map['map']:
                    permu = self._label_map['map'][perm]
                    permu = self._label_map['target_index'][permu]
                    labels.append(permu)
                    sequences.append(seq)
                    indices.append(index)
                
            # self.save_sequences(sequences)
            # self.save_labels(labels)
            return sequences, labels, indices, self._id_dictionary
        
    def load_data(self) -> Tuple[list, list, list, list]:
        """Used in the early predictions scheme in order to build the data right before the cut is done
        This way, we can, in the loop, decide to pad or not

        Returns:
            Tuple[list, list, list, list]: [description]
        """
        self._choose_sequencer()
        self._choose_event_filter()
        self._choose_break_filter()
        sequences = []
        labels = []
        begins = []
        ends = []
        indices = []
        
        for index in self._id_dictionary['sequences']:
            seq_path = self._id_dictionary['sequences'][index]['path']
            with open(seq_path, 'rb') as fp:
                sim_seq = pickle.load(fp)
                seq, begin, end = sim_seq['sequence'], sim_seq['begin'], sim_seq['end']
            perm = str(sim_seq['permutation'])
            if perm in self._label_map['map']:
                permu = self._label_map['map'][perm]
                permu = self._label_map['target_index'][permu]
                labels.append(permu)
                
                seq, begin, end = self._event_filter.filter_events(seq, begin, end)
                seq, begin, end = self._break_filter.inpute_all_breaks(seq, begin, end)
                
                sequences.append(seq)
                begins.append(begin)
                ends.append(end)
                indices.append(index)
                
        return begins, sequences, ends, labels, indices
        
        
    def build_partial_sequence(self, begins:list, x:list, ends:list, y:list, indexes: list, pad: bool) -> Tuple[list, list, list, list]:
        """[summary]

        Args:
            begins (list): beginning time stamps
            x (list): features up to the break imputation
            ends (list): ends time stamps
            y (list): labels
            indices (list): indices if length too big
            pad (bool): whether too pad or not (training + testing)

        Returns:
            new indices: index from id_dictionary for sequences long enough
            new x: x data long enough
            new y: y data long enough
            too short: index from id_dictionary for sequences too short
        """
        self._choose_lengther()
        self._choose_encoder()
        self._choose_aggregator()
        
        new_x = []
        new_y = []
        new_indexes = []
        too_short = []
        for index in indexes:
            length = len(x[index])
            if not pad and length < self._data_settings['adjuster']['limit']:
                too_short.append(index)
                continue
            seq, begin, end = self._adjuster.adjust_sequence(x[index], begins[index], ends[index], self._data_settings['adjuster']['limit'])
            seq = self._encoder.encode_sequence(seq, begin, end)
            seq = self._aggregator.aggregate(seq)
            new_x.append(seq)
            new_y.append(y[index])
            new_indexes.append(index)
            
        return new_indexes, new_x, new_y, too_short
    
    def save_sequences(self, sequences:list):
        with open(self._features_path, 'wb') as fp:
            pickle.dump(sequences, fp) 
            
    def save_labels(self, labels:list):
        with open(self._labels_path, 'wb') as fp:
            pickle.dump(labels, fp)
        
        
        
        
        
            
        
            
            
        


        

