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

def change_lstm_params(settings):
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
        'attention': {'dropout': process_numerical_list(settings['attentiondropout'])},
        'seed': process_int_list(settings['seed'])
    }

    with open('./configs/gridsearch/gs_LSTM.yaml', 'w') as fp:
        yaml.dump(lstm_gridsearch, fp)

def change_cnnlstm(settings):
    cnnlstm_gridsearch = {
        'padding_value':[-1],
        'lstm_cells': process_int_list(settings['lstm_cells']),
        'cnn_cells': process_int_list(settings['cnn_cells']),
        'cnn_window': process_int_list(settings['cnn_window']),
        'pool_size': process_int_list(settings['pool_size']),
        'stride': process_int_list(settings['stride']),
        'padding': process_list(settings['padding']),
        'dropout': process_numerical_list(settings['dropout']),
        'optimiser': ['adam'],
        'loss': ['auc'],
        'early_stopping': [False],
        'batch_size': process_int_list(settings['batch_size']),
        'shuffle': [True],
        'epochs': process_int_list(settings['epochs']),
        'verbose': [1],
        'attention': {'dropout': process_numerical_list(settings['attentiondropout'])},
        'seed': process_int_list(settings['seed'])
    }
    with open('./configs/gridsearch/gs_lstmcnn.yaml', 'w') as fp:
        yaml.dump(cnnlstm_gridsearch, fp)

def change_ssan(settings):
    cnnlstm_gridsearch = {
        'padding_value':[-1],
        'kvq_cells': process_int_list(settings['kvq_cells']),
        'pool_size': process_int_list(settings['pool_size']),
        'stride': process_int_list(settings['stride']),
        'padding': process_list(settings['padding']),
        'dropout': process_numerical_list(settings['dropout']),
        'optimiser': ['adam'],
        'loss': ['auc'],
        'early_stopping': [False],
        'batch_size': process_int_list(settings['batch_size']),
        'shuffle': [True],
        'epochs': process_int_list(settings['epochs']),
        'verbose': [1],
        'attention': {'dropout': process_numerical_list(settings['attentiondropout'])},
        'seed': process_int_list(settings['seed'])
    }
    with open('./configs/gridsearch/gs_ssan.yaml', 'w') as fp:
        yaml.dump(cnnlstm_gridsearch, fp)



def main(settings):
    if settings['lstm']:
        change_lstm_params(settings)
    if settings['cnnlstm']:
        change_cnnlstm(settings)
    if settings['ssan']:
        change_ssan(settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Yaml Config Files')
    # Experiment arguments
    
    # parameter changes
    parser.add_argument('--dropout', dest='dropout', default='0.2', help='0, 0.2, 0.5', action='store')
    parser.add_argument('--batchsize', dest='batch_size', default='64', help='64, 128, 256', action='store')
    parser.add_argument('--epochs', dest='epochs', default='50', help='50, 100, ...', action='store')
    parser.add_argument('--celltype', dest='cell_type', default='BiLSTM', help='BiLSTM or LSTM', action='store')
    parser.add_argument('--seed', dest='seed', default='193', action='store')

    # lstm
    parser.add_argument('--nlayers', dest='n_layers', default='2', help='2, 3, 4, etc.', action='store')
    parser.add_argument('--ncells', dest='n_cells', default='16.32', help='32, 64, 129, ...', action='store')
    parser.add_argument('--attentiondropout', dest='attentiondropout', default='0.05', action='store')

    # cnn-lstm
    parser.add_argument('--lstmcells', dest='lstm_cells', default='8', help='2, 3, 4, etc.', action='store')
    parser.add_argument('--cnncells', dest='cnn_cells', default='32', help='2, 3, 4, etc.', action='store')
    parser.add_argument('--cnnwin', dest='cnn_window', default='4', help='2, 3, 4, etc.', action='store')
    parser.add_argument('--poolsize', dest='pool_size', default='50', help='2, 3, 4, etc.', action='store')
    parser.add_argument('--stride', dest='stride', default='25', help='2, 3, 4, etc.', action='store')
    parser.add_argument('--padding', dest='padding', default='valid', help='valid or same', action='store')

    # ssan
    parser.add_argument('--kvqcells', dest='kvq_cells', default='5', help='2, 3, 4, etc.', action='store')

    # algorithms
    parser.add_argument('--lstm', dest='lstm', default=False, action='store_true')
    parser.add_argument('--cnnlstm', dest='cnnlstm', default=False, action='store_true')
    parser.add_argument('--ssan', dest='ssan', default=False, action='store_true')

    settings = {}
    settings.update(vars(parser.parse_args()))
    main(settings)
























































    