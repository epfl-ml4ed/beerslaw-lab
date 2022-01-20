"""
This file was mainly implemented for the cluster, in order to easily change 
the parameters without going through the files.
"""


import os
import pickle
import yaml

import argparse

def process_list(parameters:str):
    return parameters.split('..')

def process_numerical_list(parameters:str):
    params = parameters.split('..')
    return [float(p) for p in params]

def process_int_list(parameters:str):
    params = parameters.split('..')
    return [int(p) for p in params]

def process_nested_list(parameters:str):
    params = parameters.split('..')
    params = [p.split('.') for p in params]
    params = [[int(pp) for pp in p] for p in params]
    return params

def main(settings):
    lstm_gridsearch = {
        'padding_value':[-1],
        'cell_type': process_list(settings['cell_type']),
        'n_layers': process_int_list(settings['n_layers']),
        'n_cells': process_nested_list(settings['n_cells']),
        'dropout': process_numerical_list(settings['dropout']),
        'optimiser': ['adam'],
        'loss': ['auc'],
        'early_stopping': [False],
        'batch_size': process_int_list(settings['batch_size']),
        'shuffle': [True],
        'epochs': process_int_list(settings['epochs']),
        'verbose': [1],
        'attention': {'dropout': process_numerical_list(settings['attentiondropout'])}
    }

    with open('./configs/gridsearch/gs_LSTM.yaml', 'w') as fp:
        yaml.dump(lstm_gridsearch, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yaml Config Files')
    # Experiment arguments
    
    # parameter changes
    parser.add_argument('--celltype', dest='cell_type', default='BiLSTM', help='BiLSTM or LSTM', action='store')
    parser.add_argument('--nlayers', dest='n_layers', default='2', help='2, 3, 4, etc.', action='store')
    parser.add_argument('--ncells', dest='n_cells', default='16.32', help='32, 64, 129, ...', action='store')
    parser.add_argument('--dropout', dest='dropout', default='0.2', help='0, 0.2, 0.5', action='store')
    parser.add_argument('--batchsize', dest='batch_size', default='64', help='64, 128, 256', action='store')
    parser.add_argument('--epochs', dest='epochs', default='50', help='50, 100, ...', action='store')
    parser.add_argument('--attentiondropout', dest='epochs', default='0.05', action='store')

    settings = {}
    settings.update(vars(parser.parse_args()))
    main(settings)
























































    