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
    print('python script_classification.py --checkpoint --sequencer <insert sequencer>')
    

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
    if settings['trainvalidation']:
        for metric in settings['train_validation']['metrics']:
            plotter.plot(metric)
    if settings['validation_scores']:
        plotter.print_validation_scores()
    
def checkpoint_predictions(settings):
    plotter = CheckpointPlotter(settings)
    plotter.get_predictedprobabilities_printed()

def checkpoint_plot(settings):
    plotter = CheckpointPlotter(settings)
    plotter.boxplot_validation()

def checkpoint_validation(settings):
    plotter = CheckpointPlotter(settings)
    plotter.get_validation_summaries()

def predictionprobability_density(settings):
    plotter = CheckpointPlotter(settings)
    plotter.plot_probability_distribution()

    
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
    if settings['trainvalidation'] or settings['validation_scores']:
        train_validation(settings)
    if settings['checkpoint']:
        checkpoint_plot(settings)
    if settings['checkpointpreds']:
        checkpoint_predictions(settings)
    if settings['checkpointrepro']:
        create_checkpoint_reproductions(settings)
    if settings['ckptproba']:
        predictionprobability_density(settings)
    if settings['ckptvalidationscores']:
        checkpoint_validation(settings)
    # if settings['test']:
    #     test(settings)

        
    
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
    parser.add_argument('--validationscores', dest='validation_scores', default=False, action='store_true')
    parser.add_argument('--checkpoint', dest='checkpoint', default=False, action='store_true')
    parser.add_argument('--checkpointpreds', dest='checkpointpreds', default=False, action='store_true')
    parser.add_argument('--checkpointrepro', dest='checkpointrepro', default=False, action='store_true')
    parser.add_argument('--ckptproba', dest='ckptproba', default=False, action='store_true')
    parser.add_argument('--ckptvalidation', dest='ckptvalidationscores', default=False, action='store_true')
    

    # Actions
    parser.add_argument('--show', dest='show', default=False, action='store_true')
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    parser.add_argument('--saveimg', dest='saveimg', default=False, action='store_true')
    parser.add_argument('--savepng', dest='savepng', default=False, action='store_true')
    parser.add_argument('--partial', dest='partial', default=False, action='store_true')
    parser.add_argument('--nocache', dest='nocache', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
    main(settings)
        