import os
import re
import shutil
import pickle
import yaml
import pickle
import argparse
import logging 

import numpy as np
import pandas as pd

from extractors.pipeline_maker import PipelineMaker
from utils.experiment_creator import get_experiment_name
from utils.config_handler import ConfigHandler


from ml.xval_maker import XValMaker
from extractors.sequencer.one_hot_encoded.base_encodedlstm_sequencer import BaseLSTMEncoding
from extractors.sequencer.one_hot_encoded.base_sampledlstm_sequencer import BaseLSTMSampling
from extractors.sequencer.one_hot_encoded.stateaction_adaptivelstm import StateActionAdaptiveLSTM
from extractors.sequencer.one_hot_encoded.stateaction_secondslstm import StateActionSecondsLSTM
from extractors.sequencer.one_hot_encoded.year_simplestates import YearSimpleStateSecondsLSTM

def full_prediction_classification(settings):
    """Uses the config settings to:
    - decides what simulation to use
    - how to process the data
        - action count: 1hot + aveagg
        - action span: actionspan + normagg
    - how to conduct the nested cross validation
    
    Args:
        settings: config flag + arguments
    """
    settings['ML']['pipeline']['xvalidator'] = 'nested_xval'
    settings['experiment']['root_name'] += '/' + settings['experiment']['class_name'] + '/' + settings['ML']['pipeline']['model'] + '/' + settings['data']['pipeline']['encoder'] + '_' + settings['data']['pipeline']['adjuster'] + '/'
    cfg_handler = ConfigHandler(settings)
    settings = cfg_handler.handle_settings()
    log_path = '../experiments/' + settings['experiment']['root_name'] + settings['experiment']['name'] + '/training_logs.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG, 
        format='', 
        datefmt=''
    )
    
    logging.info('Creating the data')
    pipeline = PipelineMaker(settings)
    sequences, labels, indices, id_dictionary = pipeline.build_data()
    settings['id_dictionary'] = id_dictionary

    print('Starting with {} sequences!'.format(len(sequences)))

    xval = XValMaker(settings)
    logging.info('training! ')
    xval.train(sequences, labels, indices)

    config_path = '../experiments/' + settings['experiment']['root_name'] + settings['experiment']['name'] + '/config.yaml'
    with open(config_path, 'wb') as fp:
        pickle.dump(settings, fp)

def full_prediction_classification_comparison(settings):
    enc_adj_pairs = settings['data']['pipeline']['encoders_aggregators_pairs']
    models = settings['ML']['pipeline']['models']
    settings = dict(settings)
    settings['ML']['pipeline']['xvalidator'] = 'nested_xval'
    cfg_handler = ConfigHandler(settings)
    settings = cfg_handler.handle_experiment_name()
    settings['experiment']['base_name'] = settings['experiment']['root_name'] + '/' + settings['experiment']['class_name'] + '/' + settings['ML']['pipeline']['model'] + '/' + settings['data']['pipeline']['encoder'] + '_' + settings['data']['pipeline']['adjuster'] + '/'
    
    log_path = '../experiments/' + settings['experiment']['root_name'] + '/full_sequence_training_logs.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO, 
        format='', 
        datefmt=''
    )
    logging.info('** ' * 50)
    for model in models:
        for pair in enc_adj_pairs:
            logging.info('Training pairs {} with {}'.format(enc_adj_pairs[pair], model))
            config = dict(settings)
            config['data']['pipeline']['encoder'] = enc_adj_pairs[pair][0]
            config['data']['pipeline']['aggregator'] = enc_adj_pairs[pair][1]
            config['ML']['pipeline']['model'] = model
            
            config['experiment']['name'] = config['experiment']['class_name'] + '/' + model + '/' + enc_adj_pairs[pair][0] + '_' + enc_adj_pairs[pair][1] + '/'
            path = '../experiments/' + config['experiment']['base_name'] + '/' + config['experiment']['name']
            os.makedirs(path, exist_ok=True)
            
            logging.debug('log debug: {}'.format(config['experiment']['name']))
            cfg_handler = ConfigHandler(config)
            config = cfg_handler.handle_newpair(config['experiment']['name'])
            
            
            logging.info('Creating the data')
            pipeline = PipelineMaker(config)
            sequences, labels, indices, id_dictionary = pipeline.build_data()
            lengths = [len(s) for s in sequences]
            config['id_dictionary'] = id_dictionary

            print('Starting with {} sequences!'.format(len(sequences)))
            
            xval = XValMaker(config)
            logging.info('training! ')
            xval.train(sequences, labels, indices)
            config_path = '../experiments/' + settings['experiment']['root_name'] + settings['experiment']['name'] + '/config.yaml'
            with open(config_path, 'wb') as fp:
                pickle.dump(settings, fp)
            
def full_prediction_skipgram_comparison(settings):
    skipgram_maps = {
        'extended': [
            ['../experiments/cluster/e100_w4_ep200/2021_09_09_0/models/pairwise-skipgram/', 
             '../experiments/pw training extended/e10_w4_ep200/', 
             'e100_w4_ep200'],
            ['../experiments/pw training extended/e10_w4_ep200/2021_09_07_0/models/pairwise-skipgram/', 
             '../experiments/pw training extended/e10_w4_ep200/', 
             'e10_w4_ep200']
        ],
        'minimise':[
            ['../experiments/cluster/e50_w4_ep200/2021_09_09_0/models/pairwise-skipgram/', 
             '../experiments/pw training minimise/e10_w4_ep200/', 
             'e50_w4_ep200'],
            ['../experiments/pw training minimise/e15_w4_ep200/2021_09_08_1/models/pairwise-skipgram/', 
             '../experiments/pw training minimise/e10_w4_ep200/', 
             'e15_w4_ep200']
        ]
    }
    skipgram_map = {
        'extended': [skipgram_maps['extended'][int(settings['skipgram'])]],
        'minimise': [skipgram_maps['minimise'][int(settings['skipgram'])]]
    }
    models = settings['ML']['pipeline']['models']
    settings['data']['pipeline']['encoder'] = 'sgenc'
    settings['data']['pipeline']['aggregator'] = 'cumulaveagg'
    settings = dict(settings)
    settings['ML']['pipeline']['xvalidator'] = 'nested_xval'
    cfg_handler = ConfigHandler(settings)
    settings = cfg_handler.handle_experiment_name()
    settings['experiment']['base_name'] = settings['experiment']['root_name']
    
    log_path = '../experiments/' + settings['experiment']['root_name'] + '/full_sequence_training_logs.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO, 
        format='', 
        datefmt=''
    )
    logging.info('** ' * 50)
    for model in models:
        for couple in skipgram_map[settings['data']['pipeline']['sequencer']]:
            logging.info('Training skipgram {} with {}'.format(couple[1], model))
            config = dict(settings)
            config['data']['pipeline']['skipgram_weights'] = couple[0]
            config['data']['pipeline']['skipgram_map'] = couple[1]
            config['ML']['pipeline']['model'] = model
            
            config['experiment']['name'] = config['experiment']['class_name'] + '/' + model + '/sgenc_' + couple[2] + '/'
            path = '../experiments/' + config['experiment']['base_name'] + '/' + config['experiment']['name']
            os.makedirs(path, exist_ok=True)
            
            logging.debug('log debug: {}'.format(config['experiment']['name']))
            cfg_handler = ConfigHandler(config)
            config = cfg_handler.handle_newpair(config['experiment']['name'])
            
            logging.info('Creating the data')
            pipeline = PipelineMaker(config)
            sequences, labels, indices, id_dictionary = pipeline.build_data()
            config['id_dictionary'] = id_dictionary
            
            xval = XValMaker(config)
            logging.info('training! ')
            xval.train(sequences, labels, indices)

def early_prediction_classification(settings):
    enc_adj_pairs = settings['data']['pipeline']['encoders_aggregators_pairs']
    models = settings['ML']['pipeline']['models']
    limits = settings['data']['adjuster']['limits']
    settings = dict(settings)
    settings['ML']['pipeline']['xvalidator'] = 'early_nested_xval'
    cfg_handler = ConfigHandler(settings)
    settings = cfg_handler.handle_experiment_name()
    settings['experiment']['base_name'] = settings['experiment']['name']
    
    log_path = '../experiments/' + settings['experiment']['root_name'] + '/' + settings['experiment']['base_name'] + '/full_sequence_training_logs.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO, 
        format='', 
        datefmt=''
    )
    logging.info('** ' * 50)
    
    for pair in enc_adj_pairs:
        for model in models:
            for limit in limits:
                logging.info('Training pairs {} with {}'.format(enc_adj_pairs[pair], model))
                config = dict(settings)
                config['data']['pipeline']['encoder'] = 'raw'
                config['data']['pipeline']['aggregator'] = 'noagg'
                config['data']['adjuster']['limit'] = 9999999999999999999999
                config['ML']['pipeline']['model'] = model
                
                config['experiment']['name'] = config['experiment']['base_name'] + '/' + config['experiment']['class_name'] + '/' + model + '/' + enc_adj_pairs[pair][0] + '_' + enc_adj_pairs[pair][1] + '/'
                path = '../experiments/' + config['experiment']['root_name'] + '/' + config['experiment']['name']
                os.makedirs(path, exist_ok=True)
                
                logging.debug('log debug: {}'.format(config['experiment']['name']))
                cfg_handler = ConfigHandler(config)
                config = cfg_handler.handle_newpair()
                
                # TODO: Build data inside the loop
                # pipeline = PipelineMaker(config)
                # sequences, labels, id_dictionary = pipeline.build_data()
                # config['id_dictionary'] = id_dictionary -> return begin and end in id _dicaiontrya
                
                config['data']['pipeline']['encoder'] = enc_adj_pairs[pair][0]
                config['data']['pipeline']['aggregator'] = enc_adj_pairs[pair][1]
                config['data']['adjuster']['limit'] = limit
                xval = XValMaker(config)
                xval.train(['place'], ['holder'])

def checkpoint_prediction(settings):
    """Uses the config settings to:
    - decides what simulation to use
    - how to process the data
        - action count: 1hot + aveagg
        - action span: actionspan + normagg
    - how to conduct the nested cross validation
    
    Args:
        settings: config flag + arguments
    """
    old_name = settings['experiment']['old_root_name']
    new_name = settings['experiment']['root_name']
    settings['experiment']['root_name'] = '../experiments/' + settings['experiment']['root_name']
    settings['experiment']['root_name'] += '/' + settings['experiment']['class_name'] + '/' + settings['ML']['pipeline']['model'] + '/' + settings['data']['pipeline']['encoder'] + '_' + settings['data']['pipeline']['adjuster'] + '/'
    experiment_names = os.listdir(settings['experiment']['root_name'])
    for experiment_name in experiment_names:
        if experiment_name != '.DS_Store':
            config_path = '../experiments/' + settings['experiment']['root_name'] + experiment_name + '/config.yaml'
            with open(config_path, 'rb') as fp:
                settings = pickle.load(fp)
                settings['ML']['pipeline']['gridsearch'] = 'ckptgs'
                settings['ML']['pipeline']['xvalidator'] = 'ckpt_xval'
                settings['experiment']['root_name'] = settings['experiment']['root_name'].replace(old_name, new_name)
                gs = settings['ML']['xvalidators']['nested_xval']['param_grid']

            os.makedirs('../experiments/checkpoint-' + settings['experiment']['root_name'] + experiment_name, exist_ok=True)
            log_path = '../experiments/checkpoint-' + settings['experiment']['root_name'] + experiment_name + '/training_logs.txt'
            logging.basicConfig(
                filename=log_path,
                level=logging.DEBUG, 
                format='', 
                datefmt=''
            )
            
            logging.info('Creating the data')
            pipeline = PipelineMaker(settings)
            sequences, labels, indices, id_dictionary = pipeline.build_data()
            settings['id_dictionary'] = id_dictionary
            
            xval = XValMaker(settings)
            settings['ML']['xvalidators']['nested_xval']['param_grid'] = gs
            logging.info('training! ')
            xval.train(sequences, labels, indices)

            config_path = '../experiments/checkpoint-' + settings['experiment']['root_name'] + settings['experiment']['name'] + '/config.yaml'
            with open(config_path, 'wb') as fp:
                pickle.dump(settings, fp)


def test(settings):
    log_path = '../experiments/' + settings['experiment']['root_name'] + settings['experiment']['name'] 
    os.makedirs(log_path, exist_ok=True)
    log_path += '/training_logs.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG, 
        format='', 
        datefmt=''
    )
    # with open('../data/parsed simulations/perm0231_lidhkvk9vt9_t1v_simulation.pkl', 'rb') as fp:
    #     sim1 = pickle.load(fp)
    # with open('../data/parsed simulations/p_2013_lidsvdphyjs_t2_sequenced.pkl', 'rb') as fp:
    #     sim2 = pickle.load(fp)

    seq = YearSimpleStateSecondsLSTM(settings)
    with open('../data/post_test/rankings.pkl', 'rb') as fp:
        ranks = pickle.load(fp)
        ranks = ranks.set_index('username')
    seq.set_rankings(ranks)
    # print(seq)
    # labs, begins, ends = seq.get_sequences(sim1)
    # print(sum(np.array(ends) - np.array(begins)))
    # print(sim1.get_last_timestamp())
    # b = begins + [0]
    # e = [0] + ends

    # breaks = list(np.array(b) - np.array(e))
    # breaks = breaks[:-1]

    # breaks = [b for b in breaks if b > 0]
    # breaks = float(np.sum(breaks))
    # print(breaks)

    # print()
    # labs, begins, ends = seq.get_sequences(sim2, '2ae6q3hw')
    # print(sum(np.array(ends) - np.array(begins)))
    # print(sim2.get_last_timestamp())
    # b = begins + [0]
    # e = [0] + ends

    # breaks = list(np.array(b) - np.array(e))
    # breaks = breaks[:-1]

    # breaks = [b for b in breaks if b > 0]
    # breaks = float(np.sum(breaks))
    # print(breaks)
    # print()
    # # for i in range(1, len(labs)):
    #     if begins[i] < ends[i-1]:
    #         print('-', begins[i-1], ends[i-1], labs[i-1])
    #         print('-', begins[i], ends[i], labs[i])
    # # print('here')
    # for i, lab in enumerate(labs):
    #     print('*', begins[i], ends[i], lab)

    pipeline = PipelineMaker(settings)
    sequences, labels, indices, id_dictionary = pipeline.build_data()
    for seq in sequences[0]:
        print(seq)

    print(id_dictionary['sequences'][indices[0]]['learner_id'])
    # print(sim2.get_last_timestamp())

    # for i, time in enumerate(sim._timeline):
    #     print(sim._timestamps[i], time)




def main(settings):
    # Argument
    if settings['sequencer'] != '':
        settings['data']['pipeline']['sequencer'] = settings['sequencer']
        settings['experiment']['root_name'] += '/' + settings['sequencer']
        settings['experiment']['old_root_name'] += '/' + settings['sequencer']
        if 'old' == 'not in use': # Old parameters, here for archive
            if settings['sequencer'] == 'extended12' or settings['sequencer'] == 'minimised12':
                settings['data']['pipeline']['encoders_aggregators_pairs'] = {
                    0: ['1hot', 'aveagg'],
                    1: ['actionspan', 'normagg']
                }
                settings['data']['pipeline']['break_filter'] = 'cumul80br'
                
            if settings['sequencer'] == 'bin1hotext' or settings['sequencer'] == 'bin1hotmini':
                settings['data']['pipeline']['encoders_aggregators_pairs'] = {
                    0: ['raw', 'cumulaveagg'],
                    1: ['1hotactionspan', 'cumulaveagg']
                }
                settings['data']['pipeline']['break_filter'] = 'cumul1hot80br'

        if 'extended' in settings['sequencer']:
            settings['data']['pipeline']['encoders_aggregators_pairs'] = {
                    0: ['1hot', 'aveagg'],
                    1: ['actionspan', 'normagg']
                }
            settings['data']['pipeline']['break_filter'] = 'cumulbr'
            settings['classification'] = False
            settings['classification_comparison'] = True

        if 'colourbreak_flat' in settings['sequencer']:
            settings['data']['pipeline']['encoders_aggregators_pairs'] = {
                    0: ['1hot', 'aveagg'],
                    1: ['actionspan', 'normagg']
                }
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['classification'] = False
            settings['classification_comparison'] = True

        if 'colournobreak_flat'in settings['sequencer']:
            settings['data']['pipeline']['encoders_aggregators_pairs'] = {
                    0: ['1hot', 'aveagg'],
                    1: ['actionspan', 'normagg']
                }
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['classification'] = False
            settings['classification_comparison'] = True
        
        if 'stateaction_secondslstm' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'
        
        if 'stateaction_encodedlstm' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumul1hotbr'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'stateaction_adaptivelstm' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['sequencer_interval'] = int(settings['adaptiveseconds'])
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'colourbreak_secondslstm' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'tsnorm'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'year_colourbreak' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'prior_colourbreak' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['encoder'] = 'raw'
        
        if 'language_colourbreak' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'field_colourbreak' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'yl_colourbreak' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'ylf_colourbreak' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'noagg'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'colournobreak_secondslstm' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'nobrfilt'
            settings['data']['pipeline']['aggregator'] = 'tsnorm'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'simplestate_secondslstm' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'year_simplestate' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'
        
        if 'prior_simplestate' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'language_simplestate' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'field_simplestate' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'yl_simplestate' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'ylf_simplestate' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'simplemorestates_secondslstm' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulseconds'
            settings['data']['pipeline']['aggregator'] = 'minmax'
            settings['data']['pipeline']['encoder'] = 'raw'

        if 'chem2cap' in settings['sequencer']:
            settings['data']['pipeline']['break_filter'] = 'cumulbr'
            settings['ML']['xvalidators']['nested_xval']['inner_n_folds'] = 10
            settings['ML']['xvalidators']['nested_xval']['outer_n_folds'] = 10

    
    if settings['fulltime']:
        settings['data']['pipeline']['adjuster'] = 'full'
        settings['data']['adjuster']['limit'] = 900

    if settings['scrop']:
        settings['data']['pipeline']['adjuster'] = 'scrop'


    if settings['classname'] != '':
        settings['experiment']['class_name'] = settings['classname']
    if settings['models'] != '':
        settings['ML']['pipeline']['models'] = settings['models'].split('.')
        if 'lstm' in settings['models']:
            settings['data']['pipeline']['encoders_aggregators_pairs'] = {
                0: ['raw', 'noagg'],
            }
            
    # Task
    if settings['test']:
        test(settings)

    if settings['classification']:
        full_prediction_classification(settings)
        
    if settings['classification_comparison']:
        full_prediction_classification_comparison(settings)
        
    if settings['early_prediction']:
        early_prediction_classification(settings)
    
    if settings['skipgram_comparison']:
        full_prediction_skipgram_comparison(settings)

    if settings['checkpoint']:
        checkpoint_prediction(settings)
    
if __name__ == '__main__':
    with open('./configs/classifier_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Train models on full sequences')
    # Experiment arguments
    
    # actions
    parser.add_argument('--test', dest='test', default=False, help='testing method', action='store_true')
    parser.add_argument('--full', dest='classification', default=False, help='train on the wanted features and algorithm combinations for the classification task', action='store_true')
    parser.add_argument('--fullcombinations', dest='classification_comparison', default=False, help='train on the wanted features and algorithm combinations for the classification task', action='store_true')
    parser.add_argument('--sgcomparison', dest='skipgram_comparison', default=False, help='train on the wanted features and algorithm combinations for the classification task', action='store_true')
    parser.add_argument('--early', dest='early_prediction', default=False, help='train on the wanted features and algorithm combinations for the classification task at different time steps', action='store_true')
    parser.add_argument('--checkpoint', dest='checkpoint', default=False, help='loads the tensorflow models to make predictions on the best validation models', action='store_true')
    
    # settings
    parser.add_argument('--sequencer', dest='sequencer', default='', help='sequencer to use', action='store')
    parser.add_argument('--adaptiveseconds', dest='adaptiveseconds', default='1', help='sequencer to use', action='store')
    parser.add_argument('--classname', dest='classname', default='', help='class to use: colbin, conbin, widbin', action='store')
    parser.add_argument('--skipgram', dest='skipgram', default='', help='0 or 1', action='store')
    parser.add_argument('--models', dest='models', default='', help='rf, sknn, svc, sgd, knn, or adaboost', action='store')
    parser.add_argument('--fulltime', dest='fulltime', default=False, action='store_true')
    parser.add_argument('--scrop', dest='scrop', default=False, action='store_true')

    settings.update(vars(parser.parse_args()))
    
    main(settings)

