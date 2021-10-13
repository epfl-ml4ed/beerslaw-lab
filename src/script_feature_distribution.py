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
from visualisers.feature_plotter import FeaturePlotter


def test(settings):
    plotter = FeaturePlotter(settings)
    plotter.plot_distribution_label()

    
def main(settings):
    if settings['test']:
        test(settings)

if __name__ == '__main__':
    with open('./configs/featuredistribution_config.yaml', 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    parser = argparse.ArgumentParser(description='Logs / Simulations manipulations')
    parser.add_argument('--concat', dest='concatenate_labels', default=False, action='store_true')
    parser.add_argument('--plot', dest='plot', default=False, action='store_true')
    parser.add_argument('--score', dest='score', default=False, action='store_true')
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    
    parser.add_argument('--cm', dest='confusion_matrix', default=False, action='store_true')
    parser.add_argument('--label', dest='label_distribution', default=False, action='store_true')
    
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    parser.add_argument('--show', dest='show', default=False, action='store_true')
    
    settings.update(vars(parser.parse_args()))
        
    main(settings)