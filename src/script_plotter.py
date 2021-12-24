import os
import yaml
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from extractors.pipeline_maker import PipelineMaker
from ml.xval_maker import XValMaker

from ml.gridsearches.gridsearch import GridSearch
from visualisers.nested_xval_plotter import NestedXValPlotter
from visualisers.early_pred_plotter import EarlyPredPlotter
from visualisers.train_validation_plotter import TrainValidationPlotter
from visualisers.checkpoint_plotter import CheckpointPlotter

def create_checkpoint_reproductions(settings):
    """Given an experiment ran as:
        python script_classification.py --full --fulltime --sequencer name_sequencer

        and with TF models recording tensorflow checkpoint, that function recreates the same files as the normal results
        with the validation models instead.
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

def plot_full_sequences(settings):
    config = dict(settings)
    plotter = NestedXValPlotter(config)
    plotter.plot_experiment()
    
def plot_reproduction(settings):
    config = dict(settings)
    plotter = NestedXValPlotter(config)
    plotter.plot_reproduction()
    
def plot_parameters_distribution(settings):
    config = dict(settings)
    plotter = NestedXValPlotter(config)
    plotter.plot_parameters()
    
def plot_parameters_distribution(settings):
    config = dict(settings)
    plotter = NestedXValPlotter(config)
    plotter.plot_separate_parameters()
    
def plot_earlypred(settings):
    config = dict(settings)
    plotter = EarlyPredPlotter(config)
    plotter.plot_experiment()
    
def plot_earlyrepro(settings):
    config = dict(settings)
    plotter = EarlyPredPlotter(config)
    plotter.plot_reproduction()

def train_validation(settings):
    config = dict(settings)
    plotter = TrainValidationPlotter(config)
    for metric in settings['train_validation']['metrics']:
        plotter.plot(metric)
    
def checkpoint_predictions(settings):
    plotter = CheckpointPlotter(settings)
    plotter.test()

def checkpoint_plot(settings):
    plotter = CheckpointPlotter(settings)
    plotter.plot()

    
def main(settings):
    if settings['full_sequences']:
        plot_full_sequences(settings)
    if settings['reproduction']:
        plot_reproduction(settings)
    if settings['early']:
        plot_earlypred(settings)
    if settings['earlyrepro']:
        plot_earlyrepro(settings)
    if settings['parameters']:
        plot_parameters_distribution(settings)
    if settings['sepparameters']:
        plot_parameters_distribution(settings)
    if settings['trainvalidation']:
        train_validation(settings)
    if settings['checkpoint']:
        checkpoint_plot(settings)
    if settings['checkpointpreds']:
        checkpoint_predictions(settings)
    if settings['checkpointrepro']:
        create_checkpoint_reproductions(settings)

        
    
if __name__ == '__main__': 
    with open('./configs/plotter_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Plot the results')
    # Tasks
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    parser.add_argument('--full', dest='full_sequences', default=False, action='store_true')
    parser.add_argument('--fullrepro', dest='reproduction', default=False, action='store_true')
    parser.add_argument('--parameters', dest='parameters', default=False, action='store_true')
    parser.add_argument('--sepparameters', dest='sepparameters', default=False, action='store_true')
    parser.add_argument('--early', dest='early', default=False, action='store_true')
    parser.add_argument('--earlyrepro', dest='earlyrepro', default=False, action='store_true')
    parser.add_argument('--trainvalidation', dest='trainvalidation', default=False, action='store_true')
    parser.add_argument('--checkpoint', dest='checkpoint', default=False, action='store_true')
    parser.add_argument('--checkpointpreds', dest='checkpointpreds', default=False, action='store_true')
    parser.add_argument('--checkpointrepro', dest='checkpointrepro', default=False, action='store_true')
    

    # Actions
    parser.add_argument('--show', dest='show', default=False, action='store_true')
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    parser.add_argument('--partial', dest='partial', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
    main(settings)
        