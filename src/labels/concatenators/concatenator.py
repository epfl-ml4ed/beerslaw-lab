import os
import json
import dill
import yaml
import pickle

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from collections import Counter

class Concatenator:
    def __init__(self, settings):
        self._name = 'label_concatenator'
        self._notation = 'lblcnct'
        
        self._settings = settings
        self._load_id_dictionary()
        self._load_demographics()
        
    def _load_id_dictionary(self):
        """Loads the summaries of sequences and information about the students
        """
        print('loading')
        path = self._settings['path']['sequenced_simulations'] + self._settings['experiment']['sequencer'] + '/'
        path += 'id_dictionary.pkl'
        with open(path, 'rb') as fp:
            self._id_dictionary = pickle.load(fp)
            
    def _load_demographics(self):
        """Returns the demographics 
        """
        index = {v:k for (k, v) in self._id_dictionary['index'].items()}
        with open('//ic1files.epfl.ch/D-VET/Projects/ChemLab/04_Processing/Processing/Data/PostTest/post_test.pkl', 'rb') as fp:
            post_test = pickle.load(fp)
            ranks = pd.DataFrame()
            ranks['lid'] = post_test[0, 'username']
            ranks['gender'] = post_test[0, 'gender']
            ranks['year'] = post_test[0, 'year']
            ranks['field'] = post_test[0, 'field']
            ranks['ranks'] = post_test[6, 'ranks']
            
            ranks = ranks[ranks['ranks'].notna()]
            ranks['permutation'] = ranks['ranks'].apply(lambda x: ''.join([str(r) for r in x]))
            ranks = ranks.set_index('lid')
        
        
        self._demographics_map = {}
        for iid in self._id_dictionary['sequences']:
            lid = index[iid]
            self._demographics_map[iid] = {
                'gender': ranks.loc[lid]['gender'],
                'year': ranks.loc[lid]['year'],
                'field': ranks.loc[lid]['field'],
                'permutation': ranks.loc[lid]['permutation']
            }
            
    def get_legend_addition(self):
        return self._legend_addition
    
    def get_order(self):
        return self._order
    
    def get_combinations(self):
        return self._combinations
    
    def _get_algo(self, path:str) -> str:
        """Returns the name of the algorithm that was used
        """
        if '1nn' in path:
            algo = '1nn'
        elif 'rf' in path:
            algo = 'rf'
        elif 'sknn' in path:
            algo = 'sknn'
        return algo
    
    def _get_feature(self, path:str) -> str:
        """Returns the name of the features
        """
        if '1hot' in path:
            feature = 'ac'
        elif 'actionspan' in path:
            feature = 'as'
        elif 'sgenc' in path:
            feature = 'sgenc'
        return feature
    
    def _get_classes(self, path:str) -> str:
        """Returns de number of classes
        """
        if 'widbin' in path:
            return 'width'
        elif 'colbin' in path:
            return 'colour'
        elif 'conbin' in path:
            return 'concentration'
    
    def _crawl_results(self) -> list:
        """Returns all files summarising the results for a particular experiment, with a specified:
        - experiment name
        - sequencer
        - algorithm
        - feature

        Returns:
            [type]: [description]
        """
        results = []
        path = '../experiments/' + self._settings['experiment']['name'] + '/'
        for (dirpath, dirnames, filenames) in os.walk(path):
            files = [os.path.join(dirpath, file) for file in filenames]
            results.extend(files)
        results = [r for r in results if 'nested_xval_' in r]
        results = [r for r in results if self._settings['experiment']['sequencer'] in r]
        results = [r for r in results if self._settings['experiment']['algorithm'] in r]
        results = [r for r in results if self._settings['experiment']['features'] in r]
        results = [r for r in results if self._settings['experiment']['classname'] in r]
        return results
    
    def _create_lists(self, results: list) -> dict:
        """Creates a dictionary summarising the probabilities from the different classes

        Args:
            results (list): list of the results files as created by the pipe lab pipeline

        Returns:
            [dict]: 
            key [index from the id_dictionary]: {
                class 1 [the first class to concatenate with the others]:
                    raw proba: the vector of probability
                    pred: the actual prediction (integer of the label)
                    proba: the probability of being pred
                ...
                class n [the nth class to concatenate with the others]:
                    raw proba: the vector of probability
                    pred: the actual prediction (integer of the label)
                    proba: the probability of being pred
            }
        """
        predictions = {}
        not_folds = ['indices', 'x', 'y', 'indices', 'optim_scoring']
        for result in results:
            with open(result, 'rb') as fp:
                r = pickle.load(fp)
                for fold in r:
                    if fold not in not_folds:
                        for i, iid in enumerate(r[fold]['test_indices']):
                            if iid not in predictions:
                                predictions[iid] = {}
                                
                            predictions[iid][self._get_classes(result)] = {
                                'raw_proba': r[fold]['y_proba'][i],
                                'pred': r[fold]['y_pred'][i],
                                'proba': r[fold]['y_proba'][i][r[fold]['y_pred'][i]]
                            }
                            
        return predictions
        
    def _concatenate_labels(self, predictions: dict):
        raise NotImplementedError
    
    def concatenate(self):
        """Crawls the given experiment and returns the concatenated labels
        """
        result_files = self._crawl_results()
        predictions = self._create_lists(result_files)
        new_predictions = self._concatenate_labels(predictions)
        return new_predictions
        