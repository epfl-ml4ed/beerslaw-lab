import os
import pickle
import yaml
import pickle
import argparse
import logging

import numpy as np
import pandas as pd

from labels.concatenators.concatenator import Concatenator
from labels.concatenators.bi_even_concatenator import BiEvenConcatenator
from labels.concatenators.bi_nconcepts_concatenator import BiNConceptsConcatenator

from ml.scorers.fairness_scorer import FairnessScorer

from visualisers.label_plotter import LabelPlotter
from labels.xval_plotter import XvalLabelPlotter

def concatenate_labels(settings):
    concat_map = {
        'vector_labels': BiEvenConcatenator,
        'nconcepts': BiNConceptsConcatenator
    }
    concat = concat_map[settings['concatenation']['concatenator']](settings)
    new_predictions = concat.concatenate()
    settings['new_predictions'] = new_predictions
    settings['concatenator'] = concat
    
    return settings

def plot_concatenate_labels(settings):
    if settings['new_predictions'] == '':
        settings = concatenate_labels(settings)
    
    new_predictions = settings['new_predictions']
    concat = settings['concatenator']
        
    plotter = LabelPlotter(settings, concat)
    plotter.plot(new_predictions=new_predictions)
    
def measure_concatenate_label(settings):
    if settings['new_predictions'] == '':
        settings = concatenate_labels(settings)
        
    new_predictions = settings['new_predictions']
    concat = settings['concatenator']
    settings['experiment']['n_classes'] = settings['scorer']['n_classes'][settings['concatenation']['concatenator']]
    
    scorer = FairnessScorer(settings)
    results = scorer.get_scores(new_predictions)
    results = pd.DataFrame(results).transpose()
    
    plotter = LabelPlotter(settings, concat)
    plotter.plot(results=results)
    
    print(results)

def outerfold_plots(settings):
    plotter = XvalLabelPlotter(settings)
    plotter.plot()
    
    
def main(settings):
    
    if settings['concatenate_labels']:
        settings = concatenate_labels(settings)
    if settings['plot']:
        plot_concatenate_labels(settings)
    if settings['score']:
        measure_concatenate_label(settings)
    if settings['xval']:
        outerfold_plots(settings)

if __name__ == '__main__':
    with open('./configs/labelconcat_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Logs / Simulations manipulations')
    parser.add_argument('--concat', dest='concatenate_labels', default=False, action='store_true')
    parser.add_argument('--plot', dest='plot', default=False, action='store_true')
    parser.add_argument('--score', dest='score', default=False, action='store_true')
    parser.add_argument('--xval', dest='xval', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    
    parser.add_argument('--cm', dest='confusion_matrix', default=False, action='store_true')
    parser.add_argument('--label', dest='label_distribution', default=False, action='store_true')
    
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    parser.add_argument('--show', dest='show', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
        
    main(settings)