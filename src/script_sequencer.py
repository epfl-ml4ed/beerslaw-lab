import os
import re
import dill
import logging
import json
import pickle
import yaml
import argparse

import numpy as np
import pandas as pd

from extractors.parser.simulation_parser import Simulation

from extractors.sequencer.sequencing import Sequencing
from extractors.sequencer.flat.set1_sequencer import Set1Sequencing
from extractors.sequencer.flat.set2_sequencer import Set2Sequencing
from extractors.sequencer.flat.basic_sequencer import BasicSequencing
from extractors.sequencer.flat.minimise_sequencer import MinimiseSequencing
from extractors.sequencer.flat.extended_sequencer import ExtendedSequencing
from extractors.sequencer.flat.chemlab2caplab_sequencer import Chem2CapSequencer
from extractors.sequencer.flat.colournobreak_flat import ColourNobreakFlat
from extractors.sequencer.flat.colourbreak_flat import ColourbreakFlat

from extractors.sequencer.one_hot_encoded.old.binaryminimise_sequencer import Bin1HotMinimiseSequencing
from extractors.sequencer.one_hot_encoded.old.onehotminimise_sequencer import OneHotMinimiseSequencing
from extractors.sequencer.one_hot_encoded.old.binaryextended_sequencer import Bin1hotExtendedSequencing

from extractors.sequencer.one_hot_encoded.base_encodedlstm_sequencer import BaseLSTMEncoding
from extractors.sequencer.one_hot_encoded.stateaction_secondslstm import StateActionSecondsLSTM
from extractors.sequencer.one_hot_encoded.stateaction_adaptivelstm import StateActionAdaptiveLSTM
from extractors.sequencer.one_hot_encoded.stateaction_encodedlstm import StateActionLSTMEncoding
from extractors.sequencer.one_hot_encoded.colournobreak_secondslstm import ColourNobreakSecondsLSTM
from extractors.sequencer.one_hot_encoded.colourbreak_secondslstm import ColourBreakSecondsLSTM
from extractors.sequencer.one_hot_encoded.simplestate_secondslstm import SimpleStateSecondsLSTM
from extractors.sequencer.one_hot_encoded.simplemorestates_secondslstm import SimpleMoreStateSecondsLSTM
from extractors.sequencer.one_hot_encoded.year_colourbreak import YearColourBreakSecondsLSTM
from extractors.sequencer.one_hot_encoded.year_simplestates import YearSimpleStateSecondsLSTM
from extractors.sequencer.one_hot_encoded.prior3_colourbreak import PriorColourBreakSecondsLSTM
from extractors.sequencer.one_hot_encoded.prior3_simplestates import PriorSimpleStateSecondsLSTM

def process_adaptive_interval(settings):
    interval = str(settings['sequencing']['interval'])
    interval = interval.replace('.', '-')
    break_threshold = str(settings['sequencing']['break_threshold'])
    break_threshold = break_threshold.replace('.', '-')
    name = settings['sequencing']['sequencer'] + '_' + interval
    name += '_' + break_threshold
    if settings['sequencing']['dragasclick']:
        name += '_dac'
    return name, StateActionAdaptiveLSTM, settings

def sequence_simulations(settings):
    """Creates the sequenced simulation as required for the pipe-lab pipeline (see READ.me)
    Args:
        settings ([dict]): settings read from the yaml
    Flags: 
        --sequence
        -sequencer name_sequencer
    """
    log_path = './logs/sequencing_timelines.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='',
        datefmt=''
    )
    
    # Choosing the sequencer
    sequencer_map = {
        'set1': Set1Sequencing,
        'set2': Set2Sequencing,
        'basic': BasicSequencing,
        'minimise': MinimiseSequencing,
        'extended': ExtendedSequencing,
        'onehotmini': OneHotMinimiseSequencing,
        'bin1hotmini': Bin1HotMinimiseSequencing,
        'bin1hotext': Bin1hotExtendedSequencing,
        'base_lstmencoded': BaseLSTMEncoding,
        'stateaction_secondslstm': StateActionSecondsLSTM,
        'stateaction_adaptivelstm': process_adaptive_interval,
        'stateaction_encodedlstm': StateActionLSTMEncoding,
        'chem2cap': Chem2CapSequencer,
        'colourbreak_secondslstm': ColourBreakSecondsLSTM,
        'colournobreak_secondslstm': ColourNobreakSecondsLSTM,
        'simplestate_secondslstm': SimpleStateSecondsLSTM,
        'simplemorestates_secondslstm': SimpleMoreStateSecondsLSTM,
        'colournobreak_flat': ColourNobreakFlat,
        'colourbreak_flat': ColourbreakFlat,
        'year_colourbreak': YearColourBreakSecondsLSTM,
        'year_simplestate': YearSimpleStateSecondsLSTM,
        'prior_colourbreak': PriorColourBreakSecondsLSTM,
        'prior_simplestate': PriorSimpleStateSecondsLSTM
    }
    settings['data'] = {
        'pipeline': {
            'sequencer_interval': settings['sequencing']['interval'],
            'break_threshold': settings['sequencing']['break_threshold'],
            'sequencer_dragasclick': settings['sequencing']['dragasclick']
        }
    }
    if settings['sequencing']['sequencer'] == 'stateaction_adaptivelstm':
        name, sequencer, settings = process_adaptive_interval(settings)
        sequencer_map[name] = sequencer
        settings['sequencing']['sequencer'] = name
        
    sequencer = sequencer_map[settings['sequencing']['sequencer']](settings)
    
    # Setting up the data structure
    id_dictionary1 = {
        'limit': settings['sequencing']['length_limit'],
        'sequences': {},
        'index': {}
    }    
    id_dictionary2 = {
        'limit': settings['sequencing']['length_limit'],
        'sequences': {},
        'index': {}
    }
    id_dictionary3 = {
        'limit': settings['sequencing']['length_limit'],
        'sequences': {},
        'index': {}
    }
    idds = {
        '1': id_dictionary1,
        '2': id_dictionary2,
        '3': id_dictionary3
    }
    
    # list of parsed files
    ps_path = settings['paths']['parsed_simulations']
    files = os.listdir(ps_path)
    files = [f for f in files if 'simulation' in f]
    # path of sequenced files
    s_path = '../data/sequenced_simulations/' + settings['sequencing']['sequencer'] + '/'
    os.makedirs(s_path, exist_ok=True)
    
    # ranking correspondance
    with open('../data/post_test/rankings.pkl', 'rb') as fp:
        ranks = pickle.load(fp)
        ranks = ranks.set_index('username')

    sequencer.set_rankings(ranks)
    
    # regex expression to retrieve id and task number
    id_regex = re.compile('lid([^_]+)_')
    
    i = 0
    while len(files) != 0:
        file = files[0]
        print(file)
        lid = id_regex.findall(file)[0]
        print(lid)
        try:
            permutation = ranks.loc[lid]['ranking']
            print(permutation)
            gender = ranks.loc[lid]['gender']
            year = ranks.loc[lid]['year']
        except KeyError:
            print('hello')
            logging.info('No ranking for that id: {}'.format(lid))
            files_noranking = []
            files_noranking.append('permmissing_lid' + str(lid) + '_t1v_simulation.pkl')
            files_noranking.append('permmissing_lid' + str(lid) + '_t2v_simulation.pkl')
            files_noranking.append('permmissing_lid' + str(lid) + '_t3v_simulation.pkl')
            files_noranking.append('permwrong field_lid' + str(lid) + '_t1v_simulation.pkl')
            files_noranking.append('permwrong field_lid' + str(lid) + '_t2v_simulation.pkl')
            files_noranking.append('permwrong field_lid' + str(lid) + '_t3v_simulation.pkl')
            print(files_noranking)
            for f in files_noranking:
                if f in files:
                    files.remove(f)
                    print(f)
            continue
            
        
        for n_task in range(1, 4):
            # file_path = 'perm_lid' + str(lid) + '_t' + str(n_task) + 'v_simulation.pkl'
            file_path = 'perm' + str(permutation) + '_lid' + str(lid) + '_t' + str(n_task) + 'v_simulation.pkl'
            try:
                with open(ps_path + file_path, 'rb') as fp:
                    sim = dill.load(fp)
                    sim.set_permutation(permutation)
                    sim.save()
                if file_path in files:
                    files.remove(file_path)
                # # debug
                # if lid != 'xsxkdf7k':
                #     continue
                labels, begins, ends = sequencer.get_sequences(sim, lid)
                last_timestamp = sim.get_last_timestamp()
            except FileNotFoundError:
                labels, begins, ends = [], [], []
                last_timestamp = 0
            except TypeError:
                labels, begins, ends = [], [], []
                last_timestamp = 0
            
            sim_dict = {
                'sequence': labels,
                'begin': begins,
                'end': ends,
                'permutation': permutation,
                'last_timestamp': last_timestamp,
                'learner_id': lid,
                'gender': gender,
                'year': year
            }
            #debug
            # print(labels)
            path = s_path + 'p_' + permutation + '_lid' + lid + '_t' + str(n_task) + '_sequenced.pkl'
            with open(path, 'wb') as fp:
                pickle.dump(sim_dict, fp)
            
            idds[str(n_task)]['sequences'][i] = {
                'path': path,
                'length': len(labels),
                'learner_id': lid
            }
            idds[str(n_task)]['index'][lid] = i
            
        i += 1
        
    with open(s_path + 'id_dictionary1.pkl', 'wb') as fp:
        pickle.dump(id_dictionary1, fp)
    with open(s_path + 'id_dictionary2.pkl', 'wb') as fp:
        pickle.dump(id_dictionary2, fp)
    with open(s_path + 'id_dictionary3.pkl', 'wb') as fp:
        pickle.dump(id_dictionary3, fp)
        
def test_sequence(settings):
    settings['data'] = {
        'pipeline': {
            'sequencer_interval': settings['sequencing']['interval'],
            'break_threshold': settings['sequencing']['break_threshold'],
            'sequencer_dragasclick': settings['sequencing']['dragasclick']
        }
    }
    with open('../data/parsed simulations/perm0213_lidxsxkdf7k_t2v_simulation.pkl', 'rb') as fp:
        sim = pickle.load(fp)

    seq = PriorSimpleStateSecondsLSTM(settings)
    with open('../data/post_test/rankings.pkl', 'rb') as fp:
        ranks = pickle.load(fp)
        ranks = ranks.set_index('username')
    seq.set_rankings(ranks)

    labs, begins, ends = seq.get_sequences(sim, sim.get_learner_id())
    for i, lab in enumerate(labs):
        print(begins[i], ends[i], lab)
    print(len(begins))

    # for i, time in enumerate(sim._timeline):
    #     print(sim._timestamps[i], time)

    
    
def main(settings):
    if settings['sequence']:
        sequence_simulations(settings)
    if settings['test_sequence']:
        test_sequence(settings)

if __name__ == '__main__':
    with open('./configs/parsing_conf.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Logs / Simulations manipulations')
    parser.add_argument('--sequence', dest='sequence', default=False, action='store_true')
    parser.add_argument('--test', dest='test_sequence', default=False, action='store_true')
    settings.update(vars(parser.parse_args()))
        
    main(settings)