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
        raise NotImplementedError
            
    def _load_demographics(self):
        """Returns the demographics 
        """
        raise NotImplementedError
        
    def _get_stratified_scores(self, preds_df: pd.DataFrame, stratifier:str) -> dict:
        raise NotImplementedError
        
    def get_scores(self, new_predictions: dict) -> dict:
        raise NotImplementedError
                 
                
        
        
        
        
        