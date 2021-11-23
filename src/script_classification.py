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
    settings['experiment']['root_name'] += '/' + settings['experiment']['class_name'] + '/' + settings['data']['pipeline']['encoder'] + '/'
    cfg_handler = ConfigHandler(settings)
    settings = cfg_handler.handle_settings()
    log_path = '../experiments/' + settings['experiment']['root_name'] + settings['experiment']['name'] + '/training_logs.txt'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO, 
        format='', 
        datefmt=''
    )
    
    logging.info('Creating the data')
    pipeline = PipelineMaker(settings)
    sequences, labels, indices, id_dictionary = pipeline.build_data()
    settings['id_dictionary'] = id_dictionary
    
    xval = XValMaker(settings)
    logging.info('training! ')
    xval.train(sequences, labels, indices)

def full_prediction_classification_comparison(settings):
    enc_adj_pairs = settings['data']['pipeline']['encoders_aggregators_pairs']
    models = settings['ML']['pipeline']['models']
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
            
            xval = XValMaker(config)
            logging.info('training! ')
            xval.train(sequences, labels, indices)
            
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

def test(settings):
    with open('../data/parsed simulations/perm3210_lid22wyn9xy_t1v_simulation.pkl', 'rb') as fp:
        sim = pickle.load(fp)

    seq = StateActionAdaptiveLSTM(settings)
    labs, begins, ends = seq.get_sequences(sim)
    for i, lab in enumerate(labs):
        print(begins[i], ends[i], lab)

    # for i, time in enumerate(sim._timeline):
    #     print(sim._timestamps[i], time)




def main(settings):
    # Argument
    if settings['sequencer'] != '':
        settings['data']['pipeline']['sequencer'] = settings['sequencer']
        settings['experiment']['root_name'] += '/' + settings['sequencer']
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
    
    # settings
    parser.add_argument('--sequencer', dest='sequencer', default='', help='sequencer to use', action='store')
    parser.add_argument('--classname', dest='classname', default='', help='class to use: colbin, conbin, widbin', action='store')
    parser.add_argument('--skipgram', dest='skipgram', default='', help='0 or 1', action='store')
    parser.add_argument('--models', dest='models', default='', help='rf, sknn, svc, sgd, knn, or adaboost', action='store')
    
    settings.update(vars(parser.parse_args()))
    
    main(settings)

