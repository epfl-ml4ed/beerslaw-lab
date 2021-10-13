import os
import yaml
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from ml.gridsearches.gridsearch import GridSearch
from visualisers.nested_xval_plotter import NestedXValPlotter
from visualisers.early_pred_plotter import EarlyPredPlotter

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
    
    # Actions
    parser.add_argument('--show', dest='show', default=False, action='store_true')
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
    main(settings)
        