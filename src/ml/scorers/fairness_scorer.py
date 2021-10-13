import os
import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Tuple

from ml.scorers.scorer import Scorer
from ml.scorers.binaryclassification_scorer import BinaryClfScorer
from ml.scorers.multiclassification_scorer import MultiClfScorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score

class FairnessScorer(Scorer):
    """This class is used to create a scorer object tailored towards binary classification

    Args:
        Scorer (Scorer): Inherits from scorer
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        
        
        self._name = 'fairness scorer'
        self._notation = 'fairscorer'
        self._load_scorer()
        self._load_id_dictionary()
        self._load_demographics()
        
    def _load_scorer(self):
        scorer_map = {
            'multi': MultiClfScorer,
            'binary': BinaryClfScorer
        }
        self._settings['ML'] = {
            'scorers': {
                'scoring_metrics': self._settings['scorer']['scores']
            }
        }
        self._scorer = scorer_map[self._settings['scorer']['type']](self._settings)
        
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
        
    def _get_stratified_scores(self, preds_df: pd.DataFrame, stratifier:str) -> dict:
        results = {}
        for strat in preds_df[stratifier].unique():
            if pd.isnull(strat):
                continue
            sdf = preds_df[preds_df[stratifier] == strat]
            y_true = list(sdf['truth'])
            y_preds = list(sdf['preds'])
            y_probs = list(sdf['proba_vector'])
            results[strat] = self._scorer.get_scores(y_true, y_preds, y_probs)
        return results
        
    def get_scores(self, new_predictions: dict) -> dict:
        preds = {}
        for iid in new_predictions:
            preds[iid] = {
                'preds': new_predictions[iid]['predicted_label'],
                'truth': new_predictions[iid]['ground_truth'],
                'proba': new_predictions[iid]['proba'],
                'proba_vector': new_predictions[iid]['prob_vector'],
                'gender': self._demographics_map[iid]['gender'],
                'year': self._demographics_map[iid]['year'],
                'field': self._demographics_map[iid]['field'],
                'no_strat': 'nothing'
            }
        preds = pd.DataFrame(preds).transpose().reset_index().rename(columns={'index': 'lid'})
        y_true = list(preds['truth'])
        y_preds = list(preds['preds'])
        y_probs = list(preds['proba_vector'])
        print(y_preds)
        
        results = {}
        for strat in self._settings['scorer']['stratifiers']:
            if strat == 'none':
                results['overall'] = self._scorer.get_scores(y_true, y_preds, y_probs)
            else:
                results.update(self._get_stratified_scores(preds, strat))
        return results
                 
                
        
        
        
        
        