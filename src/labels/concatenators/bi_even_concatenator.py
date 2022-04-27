import os
import json
import dill
import yaml
import pickle

from typing import Tuple

import numpy as np
import pandas as pd

from labels.concatenators.concatenator import Concatenator

class BiEvenConcatenator(Concatenator):
    def __init__(self, settings):
        """Considers there are 3 labels: colour, concentration, width to be concatenated into [colour, concentration, width]. To select the correct vectors (made out of 1 if the concept is understood, and 0 if it is not), the probabilities of being understood and not being understood for each concepts are multiplied for each possibility. The one with the higher probability is then chosen. 
        

        Args:
            settings ([type]): [description]
        """
        super().__init__(settings)
        self._name = 'bi_even_label_concatenator'
        self._notation = 'bimulticoncat'
        self._legend_addition = '[col - con - width]'
        self._settings = settings
        self._combinations = {
            'labels': ['100', '010', '001', '110', '011', '101', '111', '000'],
            'liste': [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],
                      [1, 0, 1], [1, 1, 1], 0, 0, 0],
            'label_list' : {
                '100': [0, 1, 1],
                '010': [1, 0, 1],
                '001': [1, 1, 0],
                '110': [0, 0, 1],
                '011': [1, 0, 0],
                '101': [0, 1, 0],
                '111': [0, 0, 0],
                '000': [1, 1, 1]
            }
        }
        self._order = [
            '000',
            '100', '010', '001', 
            '110', '101', '011',
            '111'
        ]
        self._load_maps()
        
        
    def _load_maps(self):
        with open('../data/beerslaw/experiment_keys/permutation_maps/colour_binary.yaml', 'rb') as fp:
            self._colbin = yaml.load(fp, Loader=yaml.FullLoader)
        with open('../data/beerslaw/experiment_keys/permutation_maps/concentration_binary.yaml', 'rb') as fp:
            self._conbin = yaml.load(fp, Loader=yaml.FullLoader)
        with open('../data/beerslaw/experiment_keys/permutation_maps/width_binary.yaml', 'rb') as fp:
            self._widbin = yaml.load(fp, Loader=yaml.FullLoader)
        with open('../data/beerslaw/experiment_keys/permutation_maps/vector_binary.yaml', 'rb') as fp:
            self._binary = yaml.load(fp, Loader=yaml.FullLoader)
        
    def _get_combinations_probabilities(self, predictions:dict) -> Tuple[str, float, list]:
        """Multiplies the probabilities of each of the concepts to retrieve the label and its probability

        Args:
            predictions (dict): [description]

        Returns:
            Tuple[str, float]: [description]
        """
        probs = {}
        for label in self._combinations['labels']:
            probability = 1
            probability *= predictions['colour']['raw_proba'][self._combinations['label_list'][label][0]]    
            probability *= predictions['concentration']['raw_proba'][self._combinations['label_list'][label][1]]
            probability *= predictions['width']['raw_proba'][self._combinations['label_list'][label][2]]   
            probs[label] = probability
        label = max(probs, key=probs.get)
        prob_vector = []
        for label in self._order:
            prob_vector.append(probs[label])
        return label, probs[label], probs, prob_vector
        
    def _concatenate_labels(self, predictions: dict) -> list:
        preds = {}
        for iid in predictions:
            label, proba, prob_vector = self._get_combinations_probabilities(predictions[iid])
            permutation = self._demographics_map[iid]['permutation']
            preds[iid] = {
                'predicted_label': label,
                'proba': proba,
                'prob_vector': prob_vector,
                'ground_truth': self._binary['map'][permutation]
            }
        return preds
    
        