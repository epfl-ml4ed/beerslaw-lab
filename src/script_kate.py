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
from extractors.sequencer.one_hot_encoded.kate_secondslstm import KateStateSecondsLSTM

from extractors.pipeline_maker import PipelineMaker

def sequence_timelines(settings):
    # settings['data'] = {
    #     'pipeline': {
    #         'sequencer_interval': settings['sequencing']['interval'],
    #         'break_threshold': settings['sequencing']['break_threshold'],
    #         'sequencer_dragasclick': settings['sequencing']['dragasclick']
    #     }
    # }
    
    sequencer = KateStateSecondsLSTM(settings)

    id_dictionary1 = {
        'limit': 'x',
        'sequences': {},
        'index': {}
    }    
    id_dictionary2 = {
        'limit': 'x',
        'sequences': {},
        'index': {}
    }
    id_dictionary3 = {
        'limit': 'x',
        'sequences': {},
        'index': {}
    }
    idds = {
        '1': id_dictionary1,
        '2': id_dictionary2,
        '3': id_dictionary3
    }

    ps_path = settings['paths']['parsed_simulations']
    files = os.listdir(ps_path)
    files = [f for f in files if 'simulation' in f]

    with open('../data/beerslaw/post_test/rankings.pkl', 'rb') as fp:
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
            path = '{}kate/p_{}_lid{}_t{}_sequenced.pkl'.format(
                settings['paths']['sequenced_simulations'], permutation, lid, n_task
            )
            with open(path, 'wb') as fp:
                pickle.dump(sim_dict, fp)
            
            idds[str(n_task)]['sequences'][i] = {
                'path': path,
                'length': len(labels),
                'learner_id': lid
            }
            idds[str(n_task)]['index'][lid] = i
            
        i += 1
        
    with open(settings['paths']['sequenced_simulations'] + 'id_dictionary1.pkl', 'wb') as fp:
        pickle.dump(id_dictionary1, fp)

    with open(settings['paths']['sequenced_simulations'] + 'id_dictionary2.pkl', 'wb') as fp:
        pickle.dump(id_dictionary2, fp)

    with open(settings['paths']['sequenced_simulations'] + 'id_dictionary3.pkl', 'wb') as fp:
        pickle.dump(id_dictionary3, fp)

def save_data(settings):
    pipeline = PipelineMaker(settings)
    sequences, labels, indices, id_dictionary = pipeline.build_data()

    with open('{}/sequences.pkl'.format(settings['paths']['kate_data']), 'wb') as fp:
        pickle.dump(sequences, fp)
    with open('{}/labels.pkl'.format(settings['paths']['kate_data']), 'wb') as fp:
        pickle.dump(labels, fp)
    with open('{}/indices.pkl'.format(settings['paths']['kate_data']), 'wb') as fp:
        pickle.dump(indices, fp)
    with open('{}/id_dictionary.pkl'.format(settings['paths']['kate_data']), 'wb') as fp:
        pickle.dump(id_dictionary, fp)

def main(settings):
    os.makedirs('../data/sequences_kate/', exist_ok=True)
    os.makedirs('../data/beerslaw/sequenced_simulations/kate/', exist_ok=True)
    new_settings = {
        'paths': {
            'parsed_simulations': '../data/beerslaw/parsed simulations/',
            'sequenced_simulations': '../data/beerslaw/sequenced_simulations/',
            'kate_data': '../data/sequences_kate/', # change path here
            'crawl_path': '../data/beerslaw/temp'
        },
        'experiment': {
            'classname': 'binconcepts',
            'class_name': 'binconcepts',
            'root_name': 'createdata',
            'name': 'blank',
            'old_root_name': 'blank',
            'random_seed': 129,
            'n_folds': 'blank',
            'n_classes': 2,
        },
        'sequencing': {
            'length_limit': 0,
            'interval': 5,
            'break_threshold': 0.6,
            'dragasclick': True
        },
        'data': {
            'min_length': 0,
            'pipeline':{
                'sequencer': 'kate',
                'sequencer_interval': 5,
                'sequencer_dragasclick': True,
                'concatenator': {
                    'type': 'chemconcat',
                    'tasks': ['2']
                },
                'demographic_filter': 'chemlab',
                'event_filter': 'deleteother', # here is your filter to delete things you don't want,
                'break_filter': 'nobrfilt',
                'break_threshold': 0.6,
                'adjuster': 'full',
                'encoder': 'raw',
                'skipgram_weights': '',
                'skipgram_map': '',
                'aggregator': 'noagg',
                'encoders_aggregators_pairs': {
                    1: ['x', 'x']
                }
            },
            'adjuster': {
                'limit': 30,
                'limits': [30, 40, 50, 60, 70]
            },
            'filters': {
                'interactionlimit': 10
            }
        },
        'ML': {
            'pipeline': {
                'scorer': '2clfscorer'
            }
        }
    }

    settings.update(new_settings)

    if settings['create']:
        sequence_timelines(settings)

    if settings['save']:
        save_data(settings)



if __name__ == '__main__':
    settings = dict()
        
    parser = argparse.ArgumentParser(description='Train models on full sequences')
    
    parser.add_argument('--create', dest='create', default=False, action='store_true')
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    settings.update(vars(parser.parse_args()))
    
    main(settings)